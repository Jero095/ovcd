"""PASO 4 — Mapa de cambio."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


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
