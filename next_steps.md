Explore pipeline state and gaps

Read [](file:///c%3A/Users/User/Documentos/AAAA/Ingeniero/ovcd/pipeline.py#1-1), lines 1 to 500

Read [](file:///c%3A/Users/User/Documentos/AAAA/Ingeniero/ovcd/pipeline.py#500-500), lines 500 to 1000

Read [](file:///c%3A/Users/User/Documentos/AAAA/Ingeniero/ovcd/pipeline.py#1000-1000), lines 1000 to 1054

Created memory file [](file:///c%3A/Users/User/AppData/Roaming/Code/User/workspaceStorage/c6ac940f42eb1b5f4ab0a58929bc48d7/GitHub.copilot-chat/memory-tool/memories/MTYzOWQ2ZDgtZDQwNS00N2YwLWEwYWItYzYyZTVkM2I1ODkz/plan.md)

Aquí está el panorama completo y el plan:

---

## Plan: OVCD — Próximos pasos

**TL;DR:** El core funciona. Lo que falta principalmente es la interfaz de "prompt/query" (vocabulario personalizado + filtrado), un umbral de confianza para descartar coincidencias débiles, y mejorar cómo CLIP usa T1. Las otras tareas pendientes (evaluación cuantitativa, otros sensores) son opcionales.

---

### Fase 1 — Prompt/query por CLI *(lo que pediste)*

El mecanismo ya existe a nivel Python (`vocabulary=` en `ovcd_pipeline()`), solo falta exponerlo en el CLI y añadir filtrado. Son 3 argumentos nuevos en pipeline.py:

**Paso 1** — Añadir `--vocab` al argparse
```
--vocab "deforestation,flooding,urban construction"
```
Reemplaza el `DEFAULT_VOCABULARY` completo. Tú defines exactamente qué tipos de cambio busca CLIP.

**Paso 2** — Añadir `--query` al argparse
```
--query "flooding"
```
Filtra el output: solo muestra las regiones cuya etiqueta final coincida con ese tipo. El resto sigue procesándose pero no aparece en la imagen de salida.

**Paso 3** — Añadir `--min-score` al argparse
```
--min-score 0.22
```
Problema actual: CLIP siempre asigna una etiqueta aunque el score sea 0.08 (muy baja confianza). Con este umbral, regiones con similitud coseno por debajo del umbral se etiquetan como `"unknown"` y pueden filtrarse.

**Archivos a modificar:** solo pipeline.py — argparse (línea ~1024), `ovcd_pipeline()` (~línea 790), y `visualize_ovcd()` (~línea 892).

---

### Fase 2 — Mejorar clasificación CLIP *(calidad, opcional)*

**Problema actual:** CLIP clasifica solo con el recorte de T2. No usa información de T1. Si T2 siempre parece "vegetación", nunca clasificará como "deforestation" aunque T1 tuviera bosque.

**Mejora:** En lugar de solo `sim_t2`, usar `sim_t2 - sim_t1` como score. Una región que en T2 parece "urban construction" y en T1 NO parecía "urban construction" tendrá score diferencial alto → clasificación más precisa del *cambio*.

**Archivo:** pipeline.py líneas 775–783 en `classify_changes_clip()`.

---

### Fase 3 — Evaluación cuantitativa con Hyperspectral Change Detection Dataset

Dataset público con ground truth incluido, usado en el paper GETNET (Q. Wang, IEEE T-GRS 2019). Contiene imágenes EO-1 Hyperion (mismo sensor que HyperSIGMA) con máscaras de cambio anotadas.

#### Paso 1 — Descargar el dataset

Dos mirrors disponibles:

```
Google Drive: https://drive.google.com/open?id=1S-9Tr6plqBeIUx1p2tzO8YCLdh1BcgNV
Baidu Pan:    https://pan.baidu.com/s/14ht8k5H-8ObzHJS6msYZjQ
```

Descomprimir en `hyperglobal/imgs/getnet/`. El dataset contiene:
- `HyperCD_T1.mat` — imagen HSI en T1 (EO-1 Hyperion, ~242 bandas)
- `HyperCD_T2.mat` — imagen HSI en T2
- `gt.mat` — máscara binaria de ground truth (0 = sin cambio, 1 = cambio)

#### Paso 2 — Verificar el contenido del .mat

```python
import scipy.io as sio
import numpy as np

gt = sio.loadmat("hyperglobal/imgs/getnet/gt.mat")
print(gt.keys())               # ver el nombre de la variable
mask = list(gt.values())[-1]   # tomar el array
print(mask.shape, mask.dtype)  # debe ser (H, W) con valores 0/1
```

#### Paso 3 — Ejecutar el pipeline sobre el dataset

```powershell
.venv\Scripts\python.exe pipeline.py `
    hyperglobal/imgs/getnet/HyperCD_T1.mat `
    hyperglobal/imgs/getnet/HyperCD_T2.mat `
    --sam    sam_vit_h_4b8939.pth `
    --hsigma weights/spat-vit-base-ultra-checkpoint-1599.pth `
    --fmt    mat `
    --device cpu `
    --out    resultado_getnet.png
```

#### Paso 4 — Evaluar binary_map vs ground truth

Crear un script `evaluate.py` con:

```python
import scipy.io as sio
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
sys.path.insert(0, ".")
from pipeline import ovcd_pipeline

# Ejecutar pipeline y obtener binary_map
results = ovcd_pipeline(
    path_t1="hyperglobal/imgs/getnet/HyperCD_T1.mat",
    path_t2="hyperglobal/imgs/getnet/HyperCD_T2.mat",
    sam_checkpoint="sam_vit_h_4b8939.pth",
    hypersigma_checkpoint="weights/spat-vit-base-ultra-checkpoint-1599.pth",
    hsi_format="mat",
    device="cpu",
)

# Cargar ground truth
gt_mat = sio.loadmat("hyperglobal/imgs/getnet/gt.mat")
gt = list(gt_mat.values())[-1].astype(np.uint8).flatten()

# Aplanar la predicción al mismo tamaño
pred = results["binary_map"].flatten()

# Recortar al tamaño menor si hay diferencia de dimensiones
n = min(len(gt), len(pred))
gt, pred = gt[:n], pred[:n]

# Métricas
print(f"F1:        {f1_score(gt, pred):.4f}")
print(f"Precision: {precision_score(gt, pred):.4f}")
print(f"Recall:    {recall_score(gt, pred):.4f}")
print(f"OA (Overall Accuracy): {(gt == pred).mean():.4f}")
```

Ejecutar con:
```powershell
.venv\Scripts\python.exe evaluate.py
```

#### Métricas esperadas

El paper GETNET reporta F1 ≈ 0.70–0.80 en este dataset con un modelo supervisado entrenado. El pipeline OVCD (zero-shot, sin entrenamiento) debería obtener F1 ≈ 0.40–0.60 — menor por ser completamente no supervisado, pero comparable dado que no usa ningún dato etiquetado de entrenamiento.

#### Nota sobre variación de nombres de variables en .mat

Si `gt.mat` usa un nombre de variable distinto, inspeccionarlo con:
```python
import scipy.io as sio
d = sio.loadmat("gt.mat")
print({k: v.shape for k, v in d.items() if not k.startswith("_")})
```

---

### Verificación final (cuando esté implementado)

```powershell
# Buscar solo inundaciones con vocabulario reducido y umbral de confianza
.venv\Scripts\python.exe pipeline.py `
    hyperglobal/imgs/barbara/barbara_2013.mat `
    hyperglobal/imgs/barbara/barbara_2014.mat `
    --sam sam_vit_h_4b8939.pth `
    --hsigma weights/spat-vit-base-ultra-checkpoint-1599.pth `
    --fmt mat --device cpu `
    --vocab "flooding or water expansion,vegetation regrowth,urban construction" `
    --query "flooding or water expansion" `
    --min-score 0.20
```

---

**Archivos a modificar**

- pipeline.py — único archivo, 3 zonas: argparse, `ovcd_pipeline()`, `visualize_ovcd()` (y opcionalmente `classify_changes_clip()` para Fase 2)

**Decisiones**
- `--query` hace matching exacto contra `change_type`, por lo que el string en `--query` debe ser igual a uno de los strings del `--vocab`
- `--min-score 0.0` (default) = comportamiento actual sin filtrado

