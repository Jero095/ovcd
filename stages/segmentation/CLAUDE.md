# Etapa: segmentation (PASO 7) — SAM

Refina las regiones borrosas del mapa de cambio en máscaras precisas a nivel píxel.

- SAM opera sobre el RGB de T2 (estado actual de la escena), no sobre T1.
- Checkpoint `vit_h` (`sam_vit_h_4b8939.pth`); `vit_l`/`vit_b` son alternativas
  más rápidas.
- Por prompt: `predictor.predict(box=...)` o `(point_coords=...)` con
  `multimask_output=False`; la máscara se intersecta con `mask_hint` para confinarla
  a la región de cambio. Si la intersección queda vacía, se usa la máscara SAM
  completa (fallback intencional).
- Deuda: el modelo se carga desde disco en CADA llamada a `run_sam` — cachear si se
  procesan muchos pares.
- Utilidades SAM reutilizables fuera del pipeline: `hyperglobal/sam.py`.
