"""PASO 8 — OVCD con CLIP."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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
