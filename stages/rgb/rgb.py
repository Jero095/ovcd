"""PASO 5 — Imagen RGB desde HSI."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


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
