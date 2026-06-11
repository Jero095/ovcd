# Etapa: labeling (PASO 8) — CLIP, la capa "open vocabulary"

Etiqueta cada máscara SAM contra un vocabulario de texto libre.

- Modelo: `openai/clip-vit-base-patch32` vía HF `transformers` (se descarga solo).
- El vocabulario se encoda UNA vez; por región se recorta el bbox de la máscara
  (+10 px de padding) en T1 y T2 y se encoda con CLIP.
- **La etiqueta actual es `argmax(sim_t2)` — T1 se calcula y guarda
  (`clip_sim_t1`) pero NO influye.** Mejora planificada (next_steps.md Fase 2):
  scoring diferencial `sim_t2 − sim_t1` para clasificar el CAMBIO y no el estado.
- No hay umbral de confianza: siempre se asigna una etiqueta, aunque el score sea
  bajísimo. Mejora planificada: `--min-score` → etiquetar "unknown".
- Gotcha: `get_text_features`/`get_image_features` pueden devolver
  `BaseModelOutputWithPooling` según la versión de transformers → extraer
  `.pooler_output` (ya manejado).
- `DEFAULT_VOCABULARY` vive en este módulo y se re-exporta desde pipeline.py.
