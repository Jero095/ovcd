"""Etapas del pipeline OVCD.

Cada subpaquete implementa una etapa del pipeline y tiene su propio CLAUDE.md
con el contexto de esa etapa:

    loading          → carga de imágenes HSI bitemporales
    alignment        → alineación espectral al espacio de referencia
    features         → embeddings densos con HyperSIGMA
    change_detection → mapa de cambio + prompts para SAM
    rgb              → composición RGB desde HSI
    segmentation     → refinado de máscaras con SAM
    labeling         → etiquetado open-vocabulary con CLIP
    visualization    → imagen de salida de 4 paneles

`pipeline.py` (en la raíz del repo) orquesta todas las etapas.
"""
