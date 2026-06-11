# Etapa: change_detection (PASOS 4 y 6)

Dos módulos: el mapa de cambio y los prompts que se derivan de él.

## change_map.py — compute_change_map
- `cambio = 1 − cos(F1, F2)` por píxel del feature map, upsample bilineal a `(H, W)`,
  renormaliza a [0, 1].
- Binarización: Otsu automático si `threshold=None`, umbral fijo si no.
- SIEMPRE devuelve la tupla `(mapa_continuo, mapa_binario)` aunque la docstring
  histórica sugiera otra cosa.

## prompts.py — change_map_to_prompts
- Limpieza con `MORPH_OPEN` 3×3 → `connectedComponentsWithStats` (8-conectividad).
- Filtra regiones con área < `min_area` (default 100 px).
- Por región: `bbox` xyxy o `point` centroide + `mask_hint` (máscara de la
  componente, usada luego por SAM para confinar su máscara).
