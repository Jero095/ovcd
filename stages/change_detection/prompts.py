"""PASO 6 — Generación de prompts para SAM."""

from __future__ import annotations

import cv2
import numpy as np


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
