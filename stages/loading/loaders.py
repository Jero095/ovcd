"""PASO 1 — Carga de imágenes bitemporales."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import scipy.io


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
