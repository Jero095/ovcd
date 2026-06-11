"""
Open Vocabulary Change Detection (OVCD) Pipeline
HyperSIGMA → Change Map → SAM → CLIP

Requisitos de instalación:
    pip install torch torchvision timm
    pip install spectral scipy scikit-learn
    pip install numpy opencv-python matplotlib
    pip install transformers
    pip install git+https://github.com/facebookresearch/segment-anything.git
    pip install rasterio                   # para GeoTIFF

    # HyperSIGMA (clonar y hacer pip install -e .)
    git clone https://github.com/WHU-Sigma/HyperSIGMA.git

    # Pesos SAM (vit_h)
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import scipy.io
import scipy.interpolate
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Longitudes de onda de referencia de HyperSIGMA  (EO-1 Hyperion, 242 bandas)
# 426 nm – 2395 nm, paso ≈ 10 nm
# ---------------------------------------------------------------------------
HYPERION_WAVELENGTHS: np.ndarray = np.linspace(426.0, 2395.0, 242)

# Vocabulario OVCD por defecto
DEFAULT_VOCABULARY: list[str] = [
    "deforestation or forest loss",
    "urban construction or new buildings",
    "flooding or water expansion",
    "agricultural change or crop rotation",
    "fire damage or burned area",
    "road or infrastructure construction",
    "vegetation regrowth",
    "unknown or unclassified change",
]


# ===========================================================================
# PASO 1 — Carga de imágenes bitemporales
# ===========================================================================

def load_hsi_pair(
    path_t1: str,
    path_t2: str,
    fmt: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga un par de imágenes hiperespectrales bitemporales.

    Parameters
    ----------
    path_t1, path_t2 : rutas a los archivos HSI
    fmt : "auto" | "envi" | "mat" | "tiff"
        "auto" infiere el formato por extensión.

    Returns
    -------
    t1 : (H, W, C1) float32
    t2 : (H, W, C2) float32
    wl1 : (C1,) longitudes de onda en nm
    wl2 : (C2,) longitudes de onda en nm
    """
    t1, wl1 = _load_single(path_t1, fmt)
    t2, wl2 = _load_single(path_t2, fmt)
    return t1, t2, wl1, wl2


def _infer_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".hdr", ".img", ".dat", ".bil", ".bsq", ".bip"):
        return "envi"
    if ext in (".mat",):
        return "mat"
    if ext in (".tif", ".tiff"):
        return "tiff"
    raise ValueError(f"No se puede inferir el formato para la extensión '{ext}'. "
                     "Usa el argumento fmt='envi'|'mat'|'tiff'.")


def _load_single(path: str, fmt: str) -> tuple[np.ndarray, np.ndarray]:
    """Carga una imagen HSI y sus longitudes de onda."""
    if fmt == "auto":
        fmt = _infer_format(path)

    if fmt == "envi":
        return _load_envi(path)
    elif fmt == "mat":
        return _load_mat(path)
    elif fmt == "tiff":
        return _load_tiff(path)
    else:
        raise ValueError(f"Formato desconocido: {fmt}")


def _load_envi(path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import spectral.io.envi as envi
    except ImportError as exc:
        raise ImportError("Instala 'spectral': pip install spectral") from exc

    # Acepta tanto el .hdr como el archivo de datos directamente
    hdr_path = path if path.endswith(".hdr") else path + ".hdr"
    img = envi.open(hdr_path)
    data = img.load().astype(np.float32)  # (H, W, C)

    meta = img.metadata
    if "wavelength" in meta:
        wl = np.array([float(w) for w in meta["wavelength"]], dtype=np.float32)
    else:
        warnings.warn("Header ENVI sin 'wavelength'. Usando distribución uniforme 400-2500 nm.")
        wl = np.linspace(400.0, 2500.0, data.shape[2], dtype=np.float32)
    return data, wl


def _load_mat(path: str) -> tuple[np.ndarray, np.ndarray]:
    mat = scipy.io.loadmat(path)

    # Buscar el array 3D más grande
    data = None
    for key, val in mat.items():
        if key.startswith("_"):
            continue
        if isinstance(val, np.ndarray) and val.ndim == 3:
            if data is None or val.size > data.size:
                data = val

    if data is None:
        raise ValueError(f"No se encontró ningún array 3D en {path}. "
                         f"Claves disponibles: {[k for k in mat if not k.startswith('_')]}")

    # Asegurar orden (H, W, C)
    if data.shape[0] < data.shape[2]:
        data = np.transpose(data, (1, 2, 0))  # (C,H,W) → (H,W,C)
    data = data.astype(np.float32)

    # Buscar longitudes de onda
    wl = None
    for key in ("wavelength", "wl", "Wavelength", "WL", "lambda", "Lambda"):
        if key in mat:
            wl = mat[key].ravel().astype(np.float32)
            break
    if wl is None:
        warnings.warn("Archivo .mat sin variable de longitudes de onda. "
                      "Usando distribución uniforme 400-2500 nm.")
        wl = np.linspace(400.0, 2500.0, data.shape[2], dtype=np.float32)
    return data, wl


def _load_tiff(path: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("Instala 'rasterio': pip install rasterio") from exc

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)   # (C, H, W)
        data = np.transpose(data, (1, 2, 0))   # → (H, W, C)

        wl = None
        # Buscar en tags del dataset o sidecar XML
        tags = src.tags()
        if "wavelength" in tags:
            wl = np.array([float(w) for w in tags["wavelength"].split(",")],
                          dtype=np.float32)

    if wl is None:
        warnings.warn("GeoTIFF sin metadatos de longitud de onda. "
                      "Usando distribución uniforme 400-2500 nm.")
        wl = np.linspace(400.0, 2500.0, data.shape[2], dtype=np.float32)
    return data, wl


# ===========================================================================
# PASO 2 — Alineación espectral
# ===========================================================================

def align_spectral(
    hsi: np.ndarray,
    source_wl: np.ndarray,
    target_wl: np.ndarray,
) -> np.ndarray:
    """
    Reinterpolea las bandas de hsi al espacio espectral target_wl.

    Parameters
    ----------
    hsi       : (H, W, C_source) float32
    source_wl : (C_source,) en nm
    target_wl : (C_target,) en nm

    Returns
    -------
    (H, W, C_target) float32  normalizado por banda a [0, 1]
    """
    H, W, C = hsi.shape
    flat = hsi.reshape(-1, C)  # (H*W, C_source)

    interp = scipy.interpolate.interp1d(
        source_wl, flat, axis=1,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    aligned = interp(target_wl).astype(np.float32)  # (H*W, C_target)
    aligned = aligned.reshape(H, W, -1)             # (H, W, C_target)

    # Normalización min-max por banda
    b_min = aligned.min(axis=(0, 1), keepdims=True)
    b_max = aligned.max(axis=(0, 1), keepdims=True)
    denom = np.where(b_max - b_min > 1e-8, b_max - b_min, 1.0)
    aligned = (aligned - b_min) / denom

    return aligned


# ===========================================================================
# PASO 3 — Carga de HyperSIGMA y extracción de embeddings
# ===========================================================================

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

    Returns
    -------
    model : SpatViT en eval mode con pesos cargados
    """
    _ensure_hypersigma_on_path(hypersigma_root)

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


# ===========================================================================
# PASO 4 — Mapa de cambio
# ===========================================================================

def compute_change_map(
    f1: np.ndarray,
    f2: np.ndarray,
    target_shape: tuple[int, int],
    threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Calcula el mapa de cambio a partir de los feature maps de T1 y T2.

    Parameters
    ----------
    f1, f2       : (H', W', D) embeddings de T1 y T2
    target_shape : (H, W) tamaño original de la imagen
    threshold    : None → Otsu automático; float → umbral fijo en [0,1]

    Returns
    -------
    change_map : (H, W) float32 en [0, 1]
                 Si threshold no es None, además se retorna la versión binaria
                 como segundo elemento de una tupla.

    La función siempre retorna (continuous_map, binary_map).
    """
    # Similitud coseno
    norm1 = np.linalg.norm(f1, axis=-1, keepdims=True) + 1e-8
    norm2 = np.linalg.norm(f2, axis=-1, keepdims=True) + 1e-8
    f1_n = f1 / norm1
    f2_n = f2 / norm2
    cosine_sim = (f1_n * f2_n).sum(axis=-1)  # (H', W')
    change = (1.0 - cosine_sim).astype(np.float32)

    # Upsample al tamaño original
    H, W = target_shape
    change_up = cv2.resize(change, (W, H), interpolation=cv2.INTER_LINEAR)

    # Normalizar a [0, 1]
    c_min, c_max = change_up.min(), change_up.max()
    if c_max - c_min > 1e-8:
        change_up = (change_up - c_min) / (c_max - c_min)
    else:
        change_up = np.zeros_like(change_up)

    # Umbral
    if threshold is None:
        # Otsu automático
        change_u8 = (change_up * 255).astype(np.uint8)
        _, binary = cv2.threshold(change_u8, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = (change_up >= threshold).astype(np.uint8) * 255

    return change_up, binary


# ===========================================================================
# PASO 5 — Imagen RGB desde HSI
# ===========================================================================

def hsi_to_rgb(
    hsi: np.ndarray,
    wavelengths: np.ndarray,
    method: str = "band_select",
) -> np.ndarray:
    """
    Genera una imagen RGB desde una imagen HSI.

    Parameters
    ----------
    hsi         : (H, W, C) float32 normalizado [0, 1]
    wavelengths : (C,) en nm
    method      : "band_select" (R=660, G=550, B=470) | "pca"

    Returns
    -------
    (H, W, 3) uint8
    """
    if method == "pca":
        H, W, C = hsi.shape
        flat = hsi.reshape(-1, C)
        pca = PCA(n_components=3)
        rgb_flat = pca.fit_transform(flat)  # (H*W, 3)
        rgb = rgb_flat.reshape(H, W, 3)
    else:
        # Selección de bandas
        targets = [660.0, 550.0, 470.0]  # R, G, B en nm
        bands = [int(np.argmin(np.abs(wavelengths - t))) for t in targets]
        rgb = hsi[:, :, bands].astype(np.float32)

    # Normalizar cada canal a [0, 255]
    out = np.zeros_like(rgb)
    for i in range(3):
        ch = rgb[:, :, i]
        mn, mx = ch.min(), ch.max()
        if mx - mn > 1e-8:
            out[:, :, i] = (ch - mn) / (mx - mn) * 255.0
        else:
            out[:, :, i] = 0.0

    return out.astype(np.uint8)


# ===========================================================================
# PASO 6 — Generación de prompts para SAM
# ===========================================================================

def change_map_to_prompts(
    change_map_binary: np.ndarray,
    min_area: int = 100,
    method: str = "bbox",
) -> list[dict]:
    """
    Extrae prompts para SAM a partir del mapa de cambio binarizado.

    Parameters
    ----------
    change_map_binary : (H, W) uint8 — 0=sin cambio, 255=cambio
    min_area          : descartar regiones con menos píxeles que esto
    method            : "bbox" | "points"

    Returns
    -------
    Lista de dicts con claves:
        - region_id  : int
        - bbox       : [x0, y0, x1, y1]  (si method="bbox")
        - point      : [cx, cy]          (si method="points")
        - mask_hint  : (H, W) uint8
    """
    # Limpiar ruido morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(change_map_binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    prompts = []
    for cc in range(1, num_labels):  # 0 es el fondo
        area = stats[cc, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        mask_hint = (labels == cc).astype(np.uint8) * 255

        prompt: dict = {"region_id": cc, "mask_hint": mask_hint}

        if method == "bbox":
            x0 = int(stats[cc, cv2.CC_STAT_LEFT])
            y0 = int(stats[cc, cv2.CC_STAT_TOP])
            w  = int(stats[cc, cv2.CC_STAT_WIDTH])
            h  = int(stats[cc, cv2.CC_STAT_HEIGHT])
            prompt["bbox"] = [x0, y0, x0 + w, y0 + h]
        elif method == "points":
            cx = float(centroids[cc, 0])
            cy = float(centroids[cc, 1])
            prompt["point"] = [cx, cy]
        else:
            raise ValueError(f"method debe ser 'bbox' o 'points', no '{method}'")

        prompts.append(prompt)

    return prompts


# ===========================================================================
# PASO 7 — Inferencia SAM
# ===========================================================================

def run_sam(
    rgb_t2: np.ndarray,
    prompts: list[dict],
    sam_checkpoint: str,
    model_type: str = "vit_h",
) -> list[dict]:
    """
    Ejecuta SAM sobre la imagen T2 usando los prompts del mapa de cambio.

    Parameters
    ----------
    rgb_t2         : (H, W, 3) uint8
    prompts        : salida de change_map_to_prompts
    sam_checkpoint : ruta al archivo .pth de SAM
    model_type     : "vit_h" | "vit_l" | "vit_b"

    Returns
    -------
    Lista de dicts con:
        - region_id : int
        - mask      : (H, W) bool
        - score     : float
        - bbox      : [x0, y0, x1, y1]  (tomado del prompt)
    """
    try:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Instala SAM: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb_t2)

    results = []
    for prompt in prompts:
        mask_hint = prompt["mask_hint"]

        if "bbox" in prompt:
            box = np.array(prompt["bbox"], dtype=np.float32)
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=False,
            )
        elif "point" in prompt:
            cx, cy = prompt["point"]
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
        else:
            continue

        # Confiniar SAM dentro de la región de cambio
        sam_mask = masks[0].astype(bool)        # (H, W)
        hint_bool = mask_hint.astype(bool)
        refined = sam_mask & hint_bool

        # Si la intersección queda vacía, usar la máscara SAM original
        if refined.sum() == 0:
            refined = sam_mask

        results.append({
            "region_id": prompt["region_id"],
            "mask": refined,
            "score": float(scores[0]),
            "bbox": prompt.get("bbox", None),
        })

    return results


# ===========================================================================
# PASO 8 — OVCD con CLIP
# ===========================================================================

def classify_changes_clip(
    rgb_t1: np.ndarray,
    rgb_t2: np.ndarray,
    sam_results: list[dict],
    vocabulary: list[str],
    model_name: str = "openai/clip-vit-base-patch32",
) -> list[dict]:
    """
    Clasifica cada región detectada por SAM usando CLIP.

    Parameters
    ----------
    rgb_t1, rgb_t2 : (H, W, 3) uint8
    sam_results    : salida de run_sam
    vocabulary     : lista de descripciones de tipos de cambio
    model_name     : modelo CLIP de Hugging Face

    Returns
    -------
    sam_results enriquecidos con 'change_type' y 'clip_scores'
    """
    try:
        from transformers import CLIPProcessor, CLIPModel  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Instala transformers y Pillow: pip install transformers Pillow"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()

    # Encodar el vocabulario completo una sola vez
    text_inputs = clip_processor(
        text=vocabulary, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        text_feats = clip_model.get_text_features(**text_inputs)
        if not isinstance(text_feats, torch.Tensor):
            text_feats = text_feats.pooler_output
        text_feats = F.normalize(text_feats, dim=-1)  # (V, D)

    enriched = []
    for result in sam_results:
        mask = result["mask"]  # (H, W) bool

        crop_t1 = _crop_masked_region(rgb_t1, mask, padding=10)
        crop_t2 = _crop_masked_region(rgb_t2, mask, padding=10)

        img_t1 = Image.fromarray(crop_t1)
        img_t2 = Image.fromarray(crop_t2)

        img_inputs = clip_processor(
            images=[img_t1, img_t2], return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            img_feats = clip_model.get_image_features(**img_inputs)
            if not isinstance(img_feats, torch.Tensor):
                img_feats = img_feats.pooler_output
            img_feats = F.normalize(img_feats, dim=-1)   # (2, D)

        # Similitud de T2 con cada texto
        sim_t2 = (img_feats[1:2] @ text_feats.T).squeeze(0)  # (V,)
        # Similitud de T1 para referencia
        sim_t1 = (img_feats[0:1] @ text_feats.T).squeeze(0)  # (V,)

        best_idx = int(sim_t2.argmax().item())
        scores = sim_t2.cpu().float().numpy().tolist()

        enriched.append({
            **result,
            "change_type": vocabulary[best_idx],
            "clip_scores": {vocabulary[i]: scores[i] for i in range(len(vocabulary))},
            "clip_sim_t1": sim_t1.cpu().float().numpy().tolist(),
        })

    return enriched


def _crop_masked_region(
    img: np.ndarray,
    mask: np.ndarray,
    padding: int = 10,
) -> np.ndarray:
    """Recorta la región de interés usando la máscara con padding."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        # Máscara vacía: devolver imagen completa reducida
        return cv2.resize(img, (224, 224))

    H, W = img.shape[:2]
    r0 = max(0, rows[0] - padding)
    r1 = min(H, rows[-1] + padding + 1)
    c0 = max(0, cols[0] - padding)
    c1 = min(W, cols[-1] + padding + 1)

    crop = img[r0:r1, c0:c1]
    # CLIP espera 224×224 mínimo
    if crop.shape[0] < 1 or crop.shape[1] < 1:
        return cv2.resize(img, (224, 224))
    return crop


# ===========================================================================
# PASO 9 — Visualización
# ===========================================================================

def visualize_ovcd(
    rgb_t1: np.ndarray,
    rgb_t2: np.ndarray,
    change_map: np.ndarray,
    results: list[dict],
    save_path: str = "resultado_ovcd.png",
) -> None:
    """
    Genera una imagen de visualización con:
    - T1 y T2 en RGB
    - Mapa de cambio (heatmap)
    - T2 con máscaras SAM y etiquetas CLIP superpuestas
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb_t1)
    axes[0].set_title("T1 — RGB")
    axes[0].axis("off")

    axes[1].imshow(rgb_t2)
    axes[1].set_title("T2 — RGB")
    axes[1].axis("off")

    im = axes[2].imshow(change_map, cmap="hot", norm=Normalize(0, 1))
    axes[2].set_title("Mapa de cambio")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # T2 + máscaras
    overlay = rgb_t2.copy()
    colors = plt.get_cmap("tab10")
    legend_handles = []
    for i, res in enumerate(results):
        mask = res["mask"]
        color_rgb = (np.array(colors(i % 10)[:3]) * 255).astype(np.uint8)
        overlay[mask] = (overlay[mask] * 0.5 + color_rgb * 0.5).astype(np.uint8)

        label = res.get("change_type", f"region_{res['region_id']}")
        patch = mpatches.Patch(color=colors(i % 10), label=label)
        legend_handles.append(patch)

        # Centroide
        rows, cols = np.where(mask)
        if len(rows) > 0:
            cy, cx = int(rows.mean()), int(cols.mean())
            axes[3].text(cx, cy, str(res["region_id"]),
                         color="white", fontsize=7, ha="center", va="center",
                         bbox=dict(facecolor="black", alpha=0.4, pad=1))

    axes[3].imshow(overlay)
    axes[3].set_title("SAM + CLIP")
    axes[3].axis("off")
    if legend_handles:
        axes[3].legend(handles=legend_handles, loc="lower right",
                       fontsize=6, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualización guardada en: {save_path}")


# ===========================================================================
# PASO 10 — Pipeline completo
# ===========================================================================

def ovcd_pipeline(
    path_t1: str,
    path_t2: str,
    sam_checkpoint: str,
    hypersigma_checkpoint: str,
    hsi_format: str = "auto",
    target_wavelengths: Optional[np.ndarray] = None,
    change_threshold: Optional[float] = None,
    prompt_method: str = "bbox",
    min_region_area: int = 100,
    vocabulary: Optional[list[str]] = None,
    sam_model_type: str = "vit_h",
    hypersigma_variant: str = "spat_base",
    device: str = "cuda",
    save_viz: str = "resultado_ovcd.png",
) -> dict:
    """
    Pipeline OVCD completo: carga → alineación → HyperSIGMA → cambio → SAM → CLIP.

    Parameters
    ----------
    path_t1, path_t2            : rutas a las imágenes HSI bitemporales
    sam_checkpoint               : ruta al .pth de SAM
    hypersigma_checkpoint        : ruta al .pth de HyperSIGMA
    hsi_format                   : "auto" | "envi" | "mat" | "tiff"
    target_wavelengths           : espacio espectral de referencia (None → Hyperion)
    change_threshold             : umbral para binarizar el mapa (None → Otsu)
    prompt_method                : "bbox" | "points"
    min_region_area              : área mínima en px para considerar una región
    vocabulary                   : lista de descripciones de cambio (None → default)
    sam_model_type               : "vit_h" | "vit_l" | "vit_b"
    hypersigma_variant           : "spat_base" | "spat_large" | "spec_base" | "spec_large"
    device                       : "cuda" o "cpu"
    save_viz                     : ruta de la imagen de visualización de salida

    Returns
    -------
    dict con claves:
        - change_map  : (H, W) float32 mapa continuo
        - binary_map  : (H, W) uint8 mapa binarizado
        - n_regions   : int número de regiones detectadas
        - detections  : list[dict] resultados enriquecidos con CLIP
    """
    if target_wavelengths is None:
        target_wavelengths = HYPERION_WAVELENGTHS
    if vocabulary is None:
        vocabulary = DEFAULT_VOCABULARY

    # 1. Carga
    print("[1/8] Cargando imágenes...")
    t1, t2, wl1, wl2 = load_hsi_pair(path_t1, path_t2, hsi_format)
    print(f"      T1: {t1.shape}  T2: {t2.shape}")

    # 2. Alineación espectral
    print("[2/8] Alineando espectralmente...")
    t1_al = align_spectral(t1, wl1, target_wavelengths)
    t2_al = align_spectral(t2, wl2, target_wavelengths)

    # 3. Embeddings HyperSIGMA
    print("[3/8] Cargando HyperSIGMA...")
    model = load_hypersigma(
        hypersigma_checkpoint,
        variant=hypersigma_variant,
        in_chans=int(target_wavelengths.shape[0]),
        device=device,
    )
    print("[4/8] Extrayendo embeddings...")
    f1 = extract_features(t1_al, model, patch_size=16, device=device)
    f2 = extract_features(t2_al, model, patch_size=16, device=device)
    print(f"      F1: {f1.shape}  F2: {f2.shape}")

    # 4. Mapa de cambio
    print("[5/8] Calculando mapa de cambio...")
    change_map, binary_map = compute_change_map(
        f1, f2,
        target_shape=(t1.shape[0], t1.shape[1]),
        threshold=change_threshold,
    )

    # 5. Imágenes RGB
    rgb_t1 = hsi_to_rgb(t1_al, target_wavelengths)
    rgb_t2 = hsi_to_rgb(t2_al, target_wavelengths)

    # 6. Prompts
    print("[6/8] Generando prompts...")
    prompts = change_map_to_prompts(binary_map, min_region_area, prompt_method)
    print(f"      {len(prompts)} región(es) de cambio detectada(s)")

    if len(prompts) == 0:
        print("      No se detectaron regiones de cambio significativas.")
        return {
            "change_map": change_map,
            "binary_map": binary_map,
            "n_regions": 0,
            "detections": [],
        }

    # 7. SAM
    print("[7/8] Ejecutando SAM...")
    sam_results = run_sam(rgb_t2, prompts, sam_checkpoint, sam_model_type)

    # 8. OVCD con CLIP
    print("[8/8] Clasificando cambios con CLIP...")
    detections = classify_changes_clip(rgb_t1, rgb_t2, sam_results, vocabulary)

    # Visualización
    visualize_ovcd(rgb_t1, rgb_t2, change_map, detections, save_viz)

    for det in detections:
        score = det["score"]
        ctype = det.get("change_type", "?")
        print(f"      Región {det['region_id']:>3}: score SAM={score:.3f}  tipo={ctype}")

    return {
        "change_map": change_map,
        "binary_map": binary_map,
        "n_regions": len(detections),
        "detections": detections,
    }


# ===========================================================================
# CLI mínima
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OVCD Pipeline: HyperSIGMA → SAM → CLIP"
    )
    parser.add_argument("t1",  help="Ruta imagen HSI T1")
    parser.add_argument("t2",  help="Ruta imagen HSI T2")
    parser.add_argument("--sam",   required=True, help="Checkpoint SAM (.pth)")
    parser.add_argument("--hsigma", required=True, help="Checkpoint HyperSIGMA (.pth)")
    parser.add_argument("--fmt",   default="auto",
                        choices=["auto", "envi", "mat", "tiff"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="Umbral de cambio [0,1]. None=Otsu automático.")
    parser.add_argument("--min-area", type=int, default=100,
                        help="Área mínima de región en px")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out", default="resultado_ovcd.png",
                        help="Ruta de la imagen de salida")
    parser.add_argument("--variant", default="spat_base",
                        choices=["spat_base", "spat_large", "spec_base", "spec_large"])
    args = parser.parse_args()

    results = ovcd_pipeline(
        path_t1=args.t1,
        path_t2=args.t2,
        sam_checkpoint=args.sam,
        hypersigma_checkpoint=args.hsigma,
        hsi_format=args.fmt,
        change_threshold=args.threshold,
        min_region_area=args.min_area,
        device=args.device,
        save_viz=args.out,
        hypersigma_variant=args.variant,
    )
    print(f"\nRegiones detectadas: {results['n_regions']}")
