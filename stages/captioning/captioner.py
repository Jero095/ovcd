"""PASO opcional — Captions en lenguaje natural por región (HyperCap).

BLIP-base con el encoder de visión reemplazado por 3D-ConvSST (HyperCap),
que consume patches hiperespectrales de 11×11 píxeles con sus bandas crudas.
Entrenado sobre el dataset HyperCap (ver experiments/train_captioner_colab.py).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Espacio espectral del captioner: Indian Pines (AVIRIS, 200 bandas tras quitar
# las bandas de absorción de agua). Aproximamos las longitudes de onda con una
# distribución uniforme 400-2500 nm — misma aproximación usada en entrenamiento.
# ---------------------------------------------------------------------------
CAPTIONER_WAVELENGTHS: np.ndarray = np.linspace(400.0, 2500.0, 200)

PATCH_SIZE = 11  # hardcodeado en 3D-ConvSST (pos_embed de 121 tokens)
_BLIP_NAME = "Salesforce/blip-image-captioning-base"


def _ensure_hypercap_on_path(hypercap_root: str = "HyperCap") -> None:
    """Agrega el repo HyperCap al sys.path si no está ya."""
    root = str(Path(hypercap_root))
    if root not in sys.path:
        sys.path.insert(0, root)


class _VisionModel(nn.Module):
    """3D-ConvSST adaptado como vision_model de BLIP.

    El conv stem de 3D-ConvSST produce 64 canales → embed_dim=64 (n_heads=8
    porque debe dividir a 64) + adapter Linear(64→768) hacia el decoder de BLIP.
    Devuelve un BaseModelOutputWithPooling: BLIP moderno accede a
    vision_outputs.last_hidden_state (por atributo); el ModelOutput también
    soporta indexado [0], así que sirve para versiones nuevas y viejas.
    """

    def __init__(self, bands: int):
        super().__init__()
        from model.vision._3D_ConvSST import VisionTransformer  # type: ignore
        self.base = VisionTransformer(in_chans=bands, embed_dim=64, n_heads=8)
        self.fc = nn.Linear(64, 768)

    def forward(self, pixel_values, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        x = self.fc(self.base(pixel_values)).unsqueeze(1)  # (B, 1, 768)
        return BaseModelOutputWithPooling(last_hidden_state=x)


def load_captioner(
    checkpoint: Optional[str] = None,
    bands: int = 200,
    device: str = "cpu",
    hypercap_root: str = "HyperCap",
    seed: int | None = 0,
):
    """
    Construye el captioner BLIP + 3D-ConvSST.

    Parameters
    ----------
    checkpoint    : ruta al state_dict entrenado (.pth). None → modelo SIN
                    entrenar (solo para pruebas de plomería; texto sin sentido).
    bands         : bandas de entrada del encoder (200 = Indian Pines/AVIRIS)
    device        : "cuda" o "cpu"
    hypercap_root : ruta al repo HyperCap clonado
    seed          : semilla para la init aleatoria del encoder (reproducibilidad)

    Returns
    -------
    (model, processor) : modelo en eval mode + processor de BLIP (tokenizer)
    """
    try:
        from transformers import AutoProcessor, BlipForConditionalGeneration
    except ImportError as exc:
        raise ImportError("Instala transformers: pip install transformers") from exc

    _ensure_hypercap_on_path(hypercap_root)

    if seed is not None:
        torch.manual_seed(seed)

    processor = AutoProcessor.from_pretrained(_BLIP_NAME)
    model = BlipForConditionalGeneration.from_pretrained(_BLIP_NAME)
    model.vision_model = _VisionModel(bands)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        warnings.warn(
            "Captioner sin checkpoint entrenado: los captions serán texto sin "
            "sentido. Entrena con experiments/train_captioner_colab.py y pasa "
            "el .pth resultante."
        )

    model = model.to(device).eval()
    return model, processor


def caption_patch(model, processor, patch: np.ndarray, max_length: int = 30) -> str:
    """
    Genera un caption para un patch hiperespectral.

    patch : (bands, 11, 11) float32, normalizado [0,1] por banda
            (igual que en entrenamiento).
    """
    device = next(model.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model.generate(pixel_values=x, max_length=max_length)
    return processor.decode(out[0], skip_special_tokens=True)


def _extract_patch(hsi: np.ndarray, cy: int, cx: int) -> np.ndarray:
    """Extrae un patch (C, 11, 11) centrado en (cy, cx) con padding reflect."""
    pad = PATCH_SIZE // 2
    padded = np.pad(hsi, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    patch = padded[cy:cy + PATCH_SIZE, cx:cx + PATCH_SIZE, :]
    return patch.transpose(2, 0, 1).astype(np.float32)


def caption_regions(
    hsi_t1: np.ndarray,
    hsi_t2: np.ndarray,
    detections: list[dict],
    model,
    processor,
    max_length: int = 30,
) -> list[dict]:
    """
    Añade 'caption_t1' y 'caption_t2' a cada detección.

    El patch se extrae en el centroide de la máscara SAM (si el centroide cae
    fuera de la máscara — regiones cóncavas — se usa el píxel de la máscara más
    cercano al centroide).

    Parameters
    ----------
    hsi_t1, hsi_t2 : (H, W, C) float32 alineados al espacio del captioner
                     (CAPTIONER_WAVELENGTHS) y normalizados [0,1] por banda
    detections     : salida de run_sam / classify_changes_clip
    """
    for det in detections:
        mask = det["mask"]
        ys, xs = np.where(mask)
        if len(ys) == 0:
            det["caption_t1"] = ""
            det["caption_t2"] = ""
            continue

        cy, cx = int(ys.mean()), int(xs.mean())
        if not mask[cy, cx]:
            k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
            cy, cx = int(ys[k]), int(xs[k])

        det["caption_t1"] = caption_patch(model, processor, _extract_patch(hsi_t1, cy, cx), max_length)
        det["caption_t2"] = caption_patch(model, processor, _extract_patch(hsi_t2, cy, cx), max_length)
        det["caption_pixel"] = (cy, cx)

    return detections
