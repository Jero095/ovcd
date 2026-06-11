# Plan: integración de HyperCap — captions para regiones de cambio

Actualizado: 2026-06-11 · Estado: PLANIFICADO (no iniciado)

## Qué es HyperCap

- Paper: "Hyperspectral Image Land Cover Captioning Dataset for Vision Language
  Models" (arXiv 2505.12217, mayo 2025). Repo: github.com/arya-domain/HyperCap.
- Primer dataset de captioning para HSI: 21 237 captions pixel-wise refinadas por
  expertos, 50 clases de cobertura, sobre 4 escenas clásicas: Indian Pines,
  Houston 2013, KSC, Botswana (formato `.mat` + captions en `.csv` + clases `.json`).
- Incluye código para entrenar captioners (BLIP, GIT, mPLUG — GIT y mPLUG son los
  mejores según el paper) con encoders de visión específicos para HSI (FAHM, etc.).

## Por qué encaja en el pipeline

Hoy `stages/labeling` (CLIP) elige la etiqueta de un vocabulario CERRADO definido
por el usuario, y trabaja sobre el RGB sintetizado (3 bandas de 242 → se pierde
casi toda la información espectral). Un captioner entrenado con HyperCap:

1. Genera lenguaje natural LIBRE (caption), no solo elige de una lista.
2. Trabaja sobre el cubo hiperespectral real → aprovecha la firma espectral.
3. Aplicado a T1 y T2 por región: "antes: X / después: Y" = descripción del cambio.

Es COMPLEMENTARIO a CLIP, no un reemplazo: CLIP sigue sirviendo para query/filtrado
contra vocabulario del usuario; el captioner añade la descripción rica.

## Arquitectura objetivo

```
SAM masks (PASO 7)
      │
      ├──► stages/labeling   (CLIP)      → change_type, clip_scores   (existente)
      └──► stages/captioning (HyperCap)  → caption_t1, caption_t2     (NUEVO)
                  │
                  ▼
        detections enriquecidos → visualización / salida JSON
```

Por región: recortar el cubo HSI ALINEADO (no el RGB) de T1 y T2 con el bbox de la
máscara → captioner → dos captions → componer descripción de cambio.

## Fases

### Fase 0 — Reconocimiento (sin escribir código del pipeline)
1. Clonar el repo a `HyperCap/` (vendoreado como HyperSIGMA, con CLAUDE.md de
   "no tocar" y entrada en .gitignore).
2. Responder preguntas abiertas (ver abajo): ¿publican checkpoints preentrenados?
   ¿el captioner es por píxel o por patch? ¿qué espacio espectral espera?
3. Correr su tutorial (`Tutorial_Captioning_BLIP.py`) sobre Indian Pines — ya
   tenemos `Indian_pines_corrected.mat` local en HyperSIGMA/ImageClassification/data.
   Meta: UN caption generado de un patch conocido.

### Fase 1 — Captioner standalone
- Si hay checkpoints → wrapper de inferencia. Si NO hay → entrenar con su código
  (necesita GPU; ver Riesgos).
- `stages/captioning/captioner.py`: `load_captioner(checkpoint, device)` +
  `caption_patch(hsi_patch) -> str`. Modelo cargado UNA vez, reutilizable.

### Fase 2 — Integración al pipeline
- Nueva etapa `stages/captioning/` con su CLAUDE.md.
- `caption_regions(t1_al, t2_al, detections) -> detections` añade `caption_t1`,
  `caption_t2`. Los crops salen del cubo alineado usando bbox + máscara SAM.
- Alineación: usar `align_spectral` con el `target_wl` que espere el captioner
  (cada escena HyperCap usa sensor distinto — decidir en Fase 0).
- CLI: flag `--captions` (apagado por defecto: es costoso).
- Visualización: caption corto bajo la etiqueta CLIP en el panel 4.

### Fase 3 — Caption de CAMBIO bitemporal
- Componer `caption_t1` + `caption_t2` en una frase de cambio
  ("bare agricultural soil" → "dense green vegetation" = revegetación).
- Opción simple: plantilla. Opción rica: pasar ambos captions a un LLM o usar el
  scoring diferencial CLIP como verificación cruzada.

## Preguntas abiertas (resolver en Fase 0)
- [ ] ¿El repo publica checkpoints entrenados o solo código de entrenamiento?
- [ ] ¿Captioning por píxel, por patch o por imagen? ¿Qué tamaño de patch?
- [ ] ¿Qué formato espectral espera el encoder (nº bandas, normalización)?
- [ ] ¿Licencia del dataset/código compatible con uso académico nuestro?

## Riesgos
- **GPU**: entrenar BLIP/GIT sobre HyperCap no es viable en CPU. Si no hay
  checkpoints publicados → entrenar en Colab/Kaggle y traer los pesos.
- **Inferencia en CPU**: BLIP-base sobre ~50-120 regiones × 2 tiempos es lento pero
  factible; cachear el modelo y procesar en batch.
- **Gap de dominio**: HyperCap = cobertura de suelo (agrícola/urbano); barbara es
  agrícola CA → buen fit. Escenas muy distintas (mar, nieve) darán captions pobres.
- **Bandas**: el par barbara no tiene wavelengths reales (fallback uniforme) → la
  alineación al espacio del captioner será aproximada. Conseguir metadatos reales
  o datos con wavelengths para validar.

## Orden recomendado respecto al roadmap existente
1. Fase 1 de next_steps.md (--vocab/--query/--min-score) — pequeña, independiente.
2. HyperCap Fase 0 (reconocimiento) — barata, desbloquea decisiones.
3. Evaluación GETNET (Fase 3 de next_steps.md) — ya hay determinismo (seed).
4. HyperCap Fases 1-3.
