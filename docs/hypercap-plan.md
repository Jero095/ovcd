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

### Fase 1 — Captioner standalone — CÓDIGO COMPLETO 2026-06-11 (falta entrenar)
- `stages/captioning/captioner.py`: `load_captioner()`, `caption_patch()`,
  `caption_regions()`. Encoder 3D-ConvSST + BLIP-base; espacio espectral
  `CAPTIONER_WAVELENGTHS` (200 bandas, 400-2500 nm, aprox. Indian Pines/AVIRIS).
- Script de entrenamiento listo: `experiments/train_captioner_colab.py`
  (autocontenido para Colab GPU; ~1-2 h en T4 con Indian Pines).
  **PENDIENTE: ejecutarlo en Colab y traer el .pth a `weights/`.**
- Decisión de diseño: entrenamiento e inferencia normalizan [0,1] POR BANDA
  (difiere del tutorial, que usa valores crudos) para casar con los cubos
  alineados del pipeline.

### Fase 2 — Integración al pipeline — COMPLETADA 2026-06-11
- Etapa `stages/captioning/` con su CLAUDE.md.
- `ovcd_pipeline(captions=True, captioner_checkpoint=...)` alinea los cubos
  originales a `CAPTIONER_WAVELENGTHS` y añade `caption_t1`/`caption_t2`/
  `caption_pixel` a cada detección (patch 11×11 en el centroide de la máscara).
- CLI: `--captions` + `--captioner-ckpt <pth>`. Sin checkpoint corre igual con
  warning (texto sin sentido, solo plomería).
- Verificado: test de integración con máscaras sintéticas + corrida end-to-end.
- La visualización aún NO muestra los captions (van en el dict y en stdout);
  mostrar caption corto en el panel 4 queda con Fase 3.

### Fase 3 — Caption de CAMBIO bitemporal
- Componer `caption_t1` + `caption_t2` en una frase de cambio
  ("bare agricultural soil" → "dense green vegetation" = revegetación).
- Opción simple: plantilla. Opción rica: pasar ambos captions a un LLM o usar el
  scoring diferencial CLIP como verificación cruzada.

## Fase 0 — COMPLETADA 2026-06-11. Hallazgos:

- [x] **Checkpoints: NO hay.** El repo solo trae código de entrenamiento → hay que
  entrenar nosotros (Colab/Kaggle con GPU).
- [x] **Captioning por píxel con contexto de patch 11×11** centrado en el píxel.
  El dataset asigna un caption único por píxel etiquetado desde el pool de su clase.
- [x] **Formato espectral: bandas CRUDAS del sensor de cada escena** (IP 200,
  KSC 176, Botswana 145, Houston13 48). El tutorial NO normaliza los patches.
- [x] **Licencia MIT**; paper publicado en IEEE GRSM 2026 (DOI 10.1109/MGRS.2026.3693613).

Hallazgos extra:
- **FAHM (el encoder del tutorial) requiere `mamba_ssm` → CUDA-only**, no corre en
  esta máquina ni para inferencia. Los otros 3 encoders (3D-ConvSST, DBCTNet,
  3D-RCNet) son CPU-friendly. **Decisión: usar 3D-ConvSST** — rinde ~igual que FAHM
  en los benchmarks del paper (99.4-100% OA multimodal) y solo necesita torch+einops.
- 3D-ConvSST debe construirse con `embed_dim=64, n_heads=8` (su conv produce 64
  canales) + adapter `Linear(64, 768)` hacia BLIP. Patch 11×11 HARDCODEADO
  (pos_embed de 121 tokens).
- `model/blipcaptioning.py` y `model/text/gpt.py` FALTAN en el repo (solo hay .pyc
  compilados de py3.9) → el tutorial no corre tal cual. El
  `BlipForConditionalGeneration` stock de `transformers` funciona como reemplazo
  (verificado).
- **Plomería verificada en CPU** (`experiments/hypercap_smoke.py`): patch real de
  Indian Pines → 3D-ConvSST → BLIP.generate() → texto. Sin entrenar genera texto
  basura ("the the the..."), como se espera; el camino mecánico completo funciona.
- BLIP-base ya descargado en la caché HF local (~1 GB).

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
