# OVCD — Open Vocabulary Change Detection en HSI bitemporal

## Objetivo final
Pipeline zero-shot que, dado un par de imágenes hiperespectrales (T1, T2) de la misma
escena, detecta qué cambió y lo etiqueta con vocabulario libre, sin clases predefinidas
ni entrenamiento. Cadena: HyperSIGMA (embeddings) → mapa de cambio coseno → SAM
(máscaras precisas) → CLIP (etiquetas open-vocabulary).
**Meta siguiente:** integrar HyperCap (dataset/benchmark de captioning hiperespectral,
arXiv 2505.12217, repo github.com/arya-domain/HyperCap) para generar descripciones en
lenguaje natural de las regiones de cambio. Plan en `docs/hypercap-plan.md`.

## Antes de trabajar
1. Lee `STATUS.md` — estado actual y próximos pasos. Actualízalo al cerrar la sesión.
2. Para trabajar sobre una etapa concreta, lee primero el `CLAUDE.md` del directorio
   de esa etapa en `stages/`.

## Estructura
- `pipeline.py` — orquestador + CLI. NO contiene lógica de etapas, solo las llama.
  Re-exporta todas las funciones públicas (`from pipeline import ...` es la API estable).
- `stages/<etapa>/` — una carpeta por etapa, cada una con su `CLAUDE.md`:
  `loading`, `alignment`, `features` (HyperSIGMA), `change_detection`, `rgb`,
  `segmentation` (SAM), `labeling` (CLIP), `visualization`.
- `HyperSIGMA/` — repo de investigación vendoreado. NO refactorizar (ver su CLAUDE.md).
- `hyperglobal/` — datos de prueba y utilidades (ver su CLAUDE.md).
- `weights/` y `*.pth` — checkpoints grandes, fuera de git.
- `hyper_sam.md`, `leftover.md`, `next_steps.md` — docs históricos. OJO: `leftover.md`
  describe el layout ANTERIOR al refactor (todo en un solo pipeline.py).

## Cómo ejecutar
```powershell
.venv\Scripts\python.exe pipeline.py `
    hyperglobal/imgs/barbara/barbara_2013.mat `
    hyperglobal/imgs/barbara/barbara_2014.mat `
    --sam sam_vit_h_4b8939.pth `
    --hsigma weights/spat-vit-base-ultra-checkpoint-1599.pth `
    --fmt mat --device cpu --out resultado_ovcd.png
```
En CPU tarda varios minutos (imágenes 984×740×224); con CUDA ~10× más rápido.
Ejecutar SIEMPRE desde la raíz del repo (hay rutas relativas al CWD).

## Entorno y gotchas globales
- Windows 11 + PowerShell. Python 3.14 en `.venv` (usar `.venv\Scripts\python.exe`).
- HyperSIGMA NO es paquete pip: `stages/features` añade
  `HyperSIGMA/ImageClassification/model` al `sys.path` (relativo al CWD).
- PyTorch ≥ 2.6: el checkpoint HyperSIGMA requiere `weights_only=False` — parche local
  en `HyperSIGMA/ImageClassification/model/SpatViT_cls.py:570`. NO revertir.
- CLIP en `transformers` puede devolver `BaseModelOutputWithPooling` en vez de tensor
  → usar `.pooler_output` (ya manejado en `stages/labeling`).
- No romper la API pública de `pipeline.py` ni las firmas de las funciones de etapa.
