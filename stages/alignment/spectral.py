"""PASO 2 — Alineación espectral."""

from __future__ import annotations

import numpy as np
import scipy.interpolate

# ---------------------------------------------------------------------------
# Longitudes de onda de referencia de HyperSIGMA  (EO-1 Hyperion, 242 bandas)
# 426 nm – 2395 nm, paso ≈ 10 nm
# ---------------------------------------------------------------------------
HYPERION_WAVELENGTHS: np.ndarray = np.linspace(426.0, 2395.0, 242)


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
