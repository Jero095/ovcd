"""PASO 7 — Inferencia SAM."""

from __future__ import annotations

import numpy as np
import torch


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
