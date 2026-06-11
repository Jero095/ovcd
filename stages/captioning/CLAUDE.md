# Etapa: captioning (opcional) — HyperCap

Genera captions en lenguaje natural por región de cambio, desde el cubo HSI REAL
(no el RGB). Complementa a `stages/labeling` (CLIP), no lo reemplaza.

- Arquitectura: BLIP-base con el vision encoder reemplazado por 3D-ConvSST
  (del repo HyperCap). El encoder consume patches `(bands, 11, 11)` — el 11×11
  está HARDCODEADO en 3D-ConvSST. Construcción: `embed_dim=64, n_heads=8` +
  adapter `Linear(64→768)`.
- Espacio espectral: `CAPTIONER_WAVELENGTHS` = 200 bandas uniformes 400–2500 nm
  (aprox. de AVIRIS/Indian Pines). El pipeline alinea los cubos originales a este
  espacio con `stages/alignment` ANTES de llamar a `caption_regions`.
- Entrada normalizada [0,1] por banda — IGUAL que en entrenamiento (esto difiere
  del tutorial de HyperCap, que usa valores crudos; lo cambiamos para casar con
  los cubos alineados del pipeline).
- Por región: patch en el centroide de la máscara SAM (o el píxel de máscara más
  cercano si el centroide cae fuera). Añade `caption_t1`, `caption_t2`,
  `caption_pixel` a cada detección.
- **Sin checkpoint entrenado los captions son texto sin sentido** (warning al
  cargar). Entrenar con `experiments/train_captioner_colab.py` en Colab (GPU) y
  guardar el .pth en `weights/`. NO hay checkpoints públicos de HyperCap.
- FAHM (encoder original del tutorial) NO sirve aquí: requiere mamba_ssm (CUDA).
- Costo: BLIP.generate() en CPU ~1-3 s por patch → 2 captions por región. Con
  muchas regiones, subir `--min-area` o esperar al filtro `--min-score`.
