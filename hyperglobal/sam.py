"""
Utilidades de SAM para el pipeline OVCD.

Funciones de conveniencia que envuelven segment_anything para
usarlo con prompts derivados de mapas de cambio HSI.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch


def build_sam_predictor(checkpoint: str, model_type: str = "vit_h"):
    """
    Carga SAM y devuelve un SamPredictor listo para usar.

    Parameters
    ----------
    checkpoint : ruta al .pth de SAM
    model_type : "vit_h" | "vit_l" | "vit_b"

    Returns
    -------
    predictor : SamPredictor
    """
    try:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Instala SAM:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam = sam.to(device)
    return SamPredictor(sam)


def predict_from_bbox(
    predictor,
    bbox: list[int | float],
    mask_hint: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """
    Predice una máscara dado un bounding box.

    Parameters
    ----------
    predictor  : SamPredictor con imagen ya seteada
    bbox       : [x0, y0, x1, y1]
    mask_hint  : (H, W) uint8 — máscara opcional para refinar la salida

    Returns
    -------
    mask  : (H, W) bool
    score : float
    """
    box = np.array(bbox, dtype=np.float32)
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    mask = masks[0].astype(bool)
    if mask_hint is not None:
        refined = mask & mask_hint.astype(bool)
        mask = refined if refined.sum() > 0 else mask
    return mask, float(scores[0])


def predict_from_point(
    predictor,
    point: list[float],
    mask_hint: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """
    Predice una máscara dado un punto centroide.

    Parameters
    ----------
    predictor  : SamPredictor con imagen ya seteada
    point      : [cx, cy]
    mask_hint  : (H, W) uint8 — máscara opcional para refinar la salida

    Returns
    -------
    mask  : (H, W) bool
    score : float
    """
    coords = np.array([point], dtype=np.float32)
    labels = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=coords, point_labels=labels, multimask_output=False
    )
    mask = masks[0].astype(bool)
    if mask_hint is not None:
        refined = mask & mask_hint.astype(bool)
        mask = refined if refined.sum() > 0 else mask
    return mask, float(scores[0])


def masks_to_color_overlay(
    image: np.ndarray,
    masks: list[np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Superpone máscaras con colores distintos sobre una imagen RGB.

    Parameters
    ----------
    image : (H, W, 3) uint8
    masks : lista de (H, W) bool
    alpha : opacidad de las máscaras

    Returns
    -------
    (H, W, 3) uint8
    """
    import matplotlib.pyplot as plt

    overlay = image.copy().astype(np.float32)
    cmap = plt.get_cmap("tab10")
    for i, mask in enumerate(masks):
        color = np.array(cmap(i % 10)[:3]) * 255
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
    return overlay.clip(0, 255).astype(np.uint8)


def expand_bbox(
    bbox: list[int],
    padding: int,
    img_h: int,
    img_w: int,
) -> list[int]:
    """Expande un bounding box [x0,y0,x1,y1] con padding, respetando los límites."""
    x0, y0, x1, y1 = bbox
    return [
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(img_w, x1 + padding),
        min(img_h, y1 + padding),
    ]
