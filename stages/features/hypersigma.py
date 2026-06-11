"""PASO 3 — Carga de HyperSIGMA y extracción de embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Configuración de variantes SpatViT
_SPAT_VARIANT_CFG = {
    "spat_base": dict(
        patch_size=16, embed_dim=768,  depth=12, num_heads=12,
        out_indices=[3, 5, 7, 11], interval=3,
    ),
    "spat_large": dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        out_indices=[7, 11, 15, 23], interval=6,
    ),
    "spat_huge": dict(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        out_indices=[10, 15, 20, 31], interval=8,
    ),
}


def _ensure_hypersigma_on_path(hypersigma_root: str = "HyperSIGMA") -> None:
    """Agrega el directorio del modelo SpatViT al sys.path si no está ya."""
    model_dir = str(Path(hypersigma_root) / "ImageClassification" / "model")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)


def load_hypersigma(
    checkpoint: str,
    variant: str = "spat_base",
    in_chans: int = 242,
    device: str = "cuda",
    hypersigma_root: str = "HyperSIGMA",
    seed: int | None = 0,
) -> torch.nn.Module:
    """
    Carga el encoder SpatViT de HyperSIGMA desde un checkpoint local.

    Parameters
    ----------
    checkpoint      : ruta al .pth del checkpoint HyperSIGMA
    variant         : "spat_base" | "spat_large" | "spat_huge"
    in_chans        : número de bandas espectrales de entrada (default 242 = Hyperion)
    device          : "cuda" o "cpu"
    hypersigma_root : ruta a la carpeta raíz del repositorio HyperSIGMA clonado
    seed            : semilla para la inicialización aleatoria. El checkpoint NO
                      contiene los pesos `sampling_offsets` (SSA), que se inicializan
                      al azar — sin semilla fija cada ejecución produce embeddings
                      distintos. None = no fijar semilla (comportamiento antiguo).

    Returns
    -------
    model : SpatViT en eval mode con pesos cargados
    """
    _ensure_hypersigma_on_path(hypersigma_root)

    if seed is not None:
        torch.manual_seed(seed)

    try:
        from SpatViT_cls import SpatViT  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"No se pudo importar SpatViT desde {hypersigma_root}/ImageClassification/model/. "
            "Asegúrate de haber clonado el repositorio: "
            "git clone https://github.com/WHU-Sigma/HyperSIGMA.git"
        ) from exc

    cfg = _SPAT_VARIANT_CFG.get(variant)
    if cfg is None:
        raise ValueError(f"Variante desconocida: {variant}. "
                         f"Opciones: {list(_SPAT_VARIANT_CFG.keys())}")

    model = SpatViT(
        img_size=224,          # ignorado para inferencia flexible
        in_chans=in_chans,
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        out_indices=cfg["out_indices"],
        interval=cfg["interval"],
        qkv_bias=True,
        drop_path_rate=0.1,
        use_abs_pos_emb=False,
    )

    # init_weights maneja la carga del checkpoint y la interpolación de pos_embed
    model.init_weights(checkpoint)
    model = model.to(device).eval()
    return model


def extract_features(
    hsi: np.ndarray,
    model: torch.nn.Module,
    patch_size: int = 16,
    tile_size: int = 256,
    overlap: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extrae feature map denso con HyperSIGMA.

    Parameters
    ----------
    hsi       : (H, W, C) float32 ya alineado y normalizado
    model     : encoder HyperSIGMA en eval mode
    patch_size: tamaño del patch del ViT (16 px)
    tile_size : las imágenes grandes se procesan en tiles de este tamaño
    overlap   : solapamiento entre tiles en píxeles

    Returns
    -------
    (H', W', D) float32  donde H' = H // patch_size
    """
    H, W, C = hsi.shape

    # Padding para que H y W sean múltiplos de patch_size
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        hsi = np.pad(hsi, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    H_pad, W_pad, _ = hsi.shape

    # Decidir si procesar de una vez o por tiles
    if H_pad <= tile_size and W_pad <= tile_size:
        feat = _forward_patch(hsi, model, patch_size, device)
    else:
        feat = _tiled_forward(hsi, model, patch_size, tile_size, overlap, device)

    # Recortar al tamaño original en espacio de features
    H_f = H // patch_size
    W_f = W // patch_size
    feat = feat[:H_f, :W_f, :]  # (H', W', D)
    return feat


def _forward_patch(
    hsi: np.ndarray,
    model: torch.nn.Module,
    patch_size: int,
    device: str,
) -> np.ndarray:
    """
    Procesa una sola imagen/tile con SpatViT y devuelve el feature map denso.

    SpatViT.forward_features(x, patch_size) devuelve:
        [img_original, feat_idx0, feat_idx1, feat_idx2, feat_idx3]
    donde cada feat es (B, embed_dim, H', W') con H'=H//patch_size.
    Usamos el último feature map (representación más profunda).
    """
    H, W, C = hsi.shape
    x = torch.from_numpy(hsi.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    # x: (1, C, H, W)

    with torch.no_grad():
        out = model.forward_features(x, patch_size)

    # out = [img, feat_a, feat_b, feat_c, feat_d]  — tomamos el último
    feat_map = out[-1]          # (B, embed_dim, H', W')
    feat = feat_map[0].permute(1, 2, 0)  # (H', W', embed_dim)

    return feat.cpu().float().numpy()


def _tiled_forward(
    hsi: np.ndarray,
    model: torch.nn.Module,
    patch_size: int,
    tile_size: int,
    overlap: int,
    device: str,
) -> np.ndarray:
    """Procesa la imagen por tiles solapados y promedia los resultados."""
    H, W, C = hsi.shape
    stride = tile_size - overlap

    H_f = H // patch_size
    W_f = W // patch_size

    # Inferir D con un tile de prueba pequeño
    sample = hsi[:tile_size, :tile_size, :]
    sample_feat = _forward_patch(sample, model, patch_size, device)
    D = sample_feat.shape[-1]

    accum = np.zeros((H_f, W_f, D), dtype=np.float64)
    weight = np.zeros((H_f, W_f, 1), dtype=np.float64)

    # Pesos gaussianos para el solape
    gauss_1d = np.exp(-0.5 * ((np.arange(tile_size // patch_size)
                                - (tile_size // patch_size - 1) / 2) ** 2)
                      / ((tile_size // patch_size) / 4) ** 2)
    gauss_2d = np.outer(gauss_1d, gauss_1d)[:, :, np.newaxis]  # (Tf, Tf, 1)

    y = 0
    while y < H:
        y_end = min(y + tile_size, H)
        y_start = y_end - tile_size if y_end - y < tile_size else y

        x = 0
        while x < W:
            x_end = min(x + tile_size, W)
            x_start = x_end - tile_size if x_end - x < tile_size else x

            tile = hsi[y_start:y_end, x_start:x_end, :]
            if tile.shape[0] < patch_size or tile.shape[1] < patch_size:
                x += stride
                continue

            feat_tile = _forward_patch(tile, model, patch_size, device)
            tf_h, tf_w, _ = feat_tile.shape

            # Coordenadas en el espacio de features
            fy0 = y_start // patch_size
            fx0 = x_start // patch_size
            fy1 = fy0 + tf_h
            fx1 = fx0 + tf_w

            fy1 = min(fy1, H_f)
            fx1 = min(fx1, W_f)
            actual_h = fy1 - fy0
            actual_w = fx1 - fx0

            g = gauss_2d[:actual_h, :actual_w, :]
            accum[fy0:fy1, fx0:fx1, :] += feat_tile[:actual_h, :actual_w, :] * g
            weight[fy0:fy1, fx0:fx1, :] += g

            x += stride
        y += stride

    weight = np.where(weight < 1e-10, 1.0, weight)
    return (accum / weight).astype(np.float32)
