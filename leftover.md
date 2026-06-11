# OVCD — Open Vocabulary Change Detection en HSI
## Estado del proyecto · sesión de trabajo

---

## 1. ¿Qué hace este proyecto?

Dado un par de imágenes hiperespectrales del **mismo lugar en dos momentos distintos** (T1 y T2), el sistema detecta automáticamente qué zonas cambiaron y las etiqueta con lenguaje natural sin clases predefinidas. Eso es lo que significa *Open Vocabulary Change Detection*.

La diferencia con un detector de cambios clásico es que no se limita a decir "cambió / no cambió". Por cada región modificada, el pipeline da una descripción como `"deforestation or forest loss"` o `"urban construction or new buildings"`, usando solo texto libre como vocabulario. El vocabulario se puede cambiar sin reentrenar nada.

---

## 2. Estructura de archivos del workspace

```
ovcd/
├── pipeline.py                  ← pipeline principal (todo el código)
├── hyper_sam.md                 ← spec técnico original del pipeline
├── leftover.md                  ← este archivo
│
├── hyperglobal/
│   ├── dataset.py               ← script para descargar HyperGlobal-450K
│   └── imgs/
│       └── barbara/
│           ├── barbara_2013.mat ← imagen HSI de prueba (T1, año 2013)
│           └── barbara_2014.mat ← imagen HSI de prueba (T2, año 2014)
│
├── HyperSIGMA/                  ← repositorio clonado (no es paquete pip)
│   └── ImageClassification/
│       └── model/
│           └── SpatViT_cls.py   ← clase SpatViT usada por el pipeline
│
├── weights/
│   └── spat-vit-base-ultra-checkpoint-1599.pth  ← pesos HyperSIGMA (~1.4 GB)
│
├── sam_vit_h_4b8939.pth         ← pesos SAM vit_h (~2.4 GB, en la raíz)
├── resultado_ovcd.png           ← imagen de salida del pipeline
└── .venv/                       ← entorno virtual Python 3.14.2
```

---

## 3. Los modelos que se usan

### 3.1 HyperSIGMA — extractor de features hiperespectrales

- **Qué es:** Vision Transformer (ViT) preentrenado sobre HyperGlobal-450K (450 000 imágenes hiperespectrales de sensores EO-1 Hyperion y GF-5B, de todo el mundo).
- **Para qué sirve aquí:** Tomar una imagen HSI de cualquier sensor y producir un *feature map denso* — un mapa donde cada píxel es un vector de 768 dimensiones que resume la información espectral y espacial de esa zona.
- **Variante usada:** `spat_base` (SpatViT Base) — 768 dimensiones de embedding, 12 capas transformer, patches de 16×16 px.
- **Otras variantes disponibles:** `spat_large` (1024-dim, 24 capas), `spat_huge` (1280-dim, 32 capas).
- **Dónde vive el código:** `HyperSIGMA/ImageClassification/model/SpatViT_cls.py`, clase `SpatViT`.
- **Checkpoint:** `weights/spat-vit-base-ultra-checkpoint-1599.pth` (1.37 GB).
- **Cómo se carga:**
  ```python
  # NO es un paquete pip — se importa añadiendo la carpeta al sys.path
  sys.path.insert(0, "HyperSIGMA/ImageClassification/model")
  from SpatViT_cls import SpatViT
  model = SpatViT(...)
  model.init_weights("weights/spat-vit-base-ultra-checkpoint-1599.pth")
  ```
- **API clave:** `model.forward_features(x, patch_size)` recibe un tensor `(B, C, H, W)` y devuelve una lista `[img_original, feat_0, feat_1, feat_2, feat_3]`. El pipeline usa `out[-1]` (el feature map de la última capa), de shape `(B, embed_dim, H', W')` donde `H' = H // 16`.

### 3.2 SAM — Segment Anything Model (Meta AI)

- **Qué es:** Modelo de segmentación zero-shot de Meta que, dado un prompt (bounding box o punto), produce una máscara precisa a nivel de píxel.
- **Para qué sirve aquí:** Refinar los bordes de las regiones de cambio detectadas por HyperSIGMA. El mapa de cambio es borroso (resolución H/16); SAM lo convierte en contornos exactos.
- **Variante usada:** `vit_h` (ViT-Huge), la más precisa. Tiene ~632M parámetros.
- **Otras variantes:** `vit_l` (ViT-Large, ~307M) y `vit_b` (ViT-Base, ~91M) — menores pero más rápidas.
- **Checkpoint:** `sam_vit_h_4b8939.pth` (~2.4 GB, en la raíz del workspace).
- **Instalación:** `pip install git+https://github.com/facebookresearch/segment-anything.git`
- **API usada:** `SamPredictor` con `.set_image(rgb)` y `.predict(box=...) → masks, scores, logits`.

### 3.3 CLIP — Contrastive Language-Image Pretraining (OpenAI)

- **Qué es:** Modelo multimodal que encoda imágenes y texto en el mismo espacio vectorial. Fue entrenado con ~400M pares imagen-texto.
- **Para qué sirve aquí:** Clasificar cada región de cambio en lenguaje natural. Por cada región, recorta el parche de T1 y T2, los encoda con CLIP, y calcula la similitud coseno con cada descripción del vocabulario. La descripción con mayor similitud es la etiqueta del cambio.
- **Variante usada:** `openai/clip-vit-base-patch32` (via Hugging Face `transformers`).
- **No necesita checkpoint local:** se descarga automáticamente de Hugging Face Hub la primera vez.

#### Cómo empareja el vocabulario con las máscaras de SAM

**1. Preparación del vocabulario (una sola vez, antes del bucle)**

```python
text_inputs = clip_processor(text=vocabulary, ...)
text_feats = clip_model.get_text_features(**text_inputs)
text_feats = F.normalize(text_feats, dim=-1)   # shape (V, 768)
```

El vocabulario (lista de strings) se convierte en un tensor `(V, 768)` — un vector por descripción, normalizados en la esfera unitaria.

**2. Por cada región que SAM detectó**

a) **Recorte de la región**
```python
mask = result["mask"]   # (H, W) bool — máscara de SAM
crop_t1 = _crop_masked_region(rgb_t1, mask, padding=10)
crop_t2 = _crop_masked_region(rgb_t2, mask, padding=10)
```
`_crop_masked_region` halla el bounding box de los píxeles `True`, añade 10 px de margen y recorta ese rectángulo de la imagen RGB.

b) **Encodado de las imágenes recortadas**
```python
img_feats = clip_model.get_image_features(images=[crop_t1, crop_t2])
img_feats = F.normalize(img_feats, dim=-1)   # shape (2, 768)
```
Produce un vector de 768 dims para T1 y otro para T2, en el mismo espacio vectorial que el texto.

c) **Similitud coseno imagen→texto**
```python
sim_t2 = (img_feats[1:2] @ text_feats.T).squeeze(0)   # (V,)
```
Producto matricial entre el vector de T2 `(1, 768)` y todos los vectores de texto `(768, V)`. Como ambos están normalizados, el resultado es directamente la similitud coseno por cada descripción del vocabulario.

d) **Asignación de etiqueta**
```python
best_idx = int(sim_t2.argmax())
change_type = vocabulary[best_idx]
```
La descripción con mayor similitud coseno gana. No hay umbral — siempre se asigna alguna etiqueta.

**3. Resultado por región**

| Campo | Contenido |
|---|---|
| `change_type` | String ganador del vocabulario |
| `clip_scores` | Dict `{descripción: score}` para todas las clases |
| `clip_sim_t1` | Scores de T1 (guardados como referencia, no influyen en la etiqueta) |

> **Nota:** La clasificación se hace con el recorte de **T2** únicamente. Los scores de T1 se calculan y guardan pero no determinan `change_type`. El sistema responde a *cómo quedó la zona después del cambio*, no a la diferencia entre T1 y T2.

---

## 4. Descripción detallada de los archivos de código

### `pipeline.py` — el pipeline principal

Archivo único con ~1050 líneas que implementa los 10 pasos del pipeline. Tiene CLI (argparse) para ejecutarse desde terminal.

#### Pasos implementados

| Paso | Función | Entrada | Salida |
|------|---------|---------|--------|
| 1 | `load_hsi_pair` | rutas T1/T2 | arrays `(H,W,C)` + wavelengths |
| 2 | `align_spectral` | `(H,W,C)` + wl fuente + wl destino | `(H,W,C_ref)` normalizado |
| 3 | `load_hypersigma` | ruta checkpoint | modelo SpatViT en eval mode |
| 4 | `extract_features` | `(H,W,C)` + modelo | `(H',W',D)` feature map |
| 5 | `compute_change_map` | F1, F2 + shape | mapa continuo + mapa binario |
| 6 | `hsi_to_rgb` | `(H,W,C)` + wavelengths | `(H,W,3)` uint8 |
| 6b | `change_map_to_prompts` | mapa binario | lista de prompts (bbox o punto) |
| 7 | `run_sam` | RGB T2 + prompts + checkpoint | lista de máscaras + scores |
| 8 | `classify_changes_clip` | RGB T1/T2 + máscaras + vocabulario | detecciones etiquetadas |
| 9 | `visualize_ovcd` | todos los resultados | imagen PNG 4-paneles |
| 10 | `ovcd_pipeline` | todos los parámetros | dict con resultados |

#### Funciones importantes en detalle

**`load_hsi_pair(path_t1, path_t2, fmt)`**
- Soporta ENVI (`.hdr`), MATLAB (`.mat`), GeoTIFF (`.tif`).
- Para `.mat`: busca el array 3D más grande; si las dimensiones son `(C,H,W)` lo transpone automáticamente.
- Si no hay metadatos de wavelength, asume distribución uniforme 400–2500 nm.

**`align_spectral(hsi, source_wl, target_wl)`**
- Reinterpolea las bandas de la imagen al espacio espectral de referencia (Hyperion: 426–2395 nm, 242 bandas).
- Usa `scipy.interpolate.interp1d` con relleno en 0 para wavelengths fuera de rango.
- Normaliza cada banda a `[0, 1]` con min-max.
- Esto permite comparar imágenes de sensores distintos en el mismo espacio.

**`load_hypersigma(checkpoint, variant, in_chans, device, hypersigma_root)`**
- Añade `HyperSIGMA/ImageClassification/model/` al `sys.path` en tiempo de ejecución.
- Importa `SpatViT` directamente del archivo local.
- `in_chans` debe ser igual al número de bandas después de la alineación (242 por defecto).
- `init_weights()` maneja automáticamente la interpolación de `pos_embed` y la eliminación del `patch_embed` de 3 canales (ya que el modelo fue preentrenado con RGB pero aquí se usa con HSI).

**`extract_features(hsi, model, patch_size, tile_size, overlap, device)`**
- Si la imagen es mayor que `tile_size` (256 px por defecto), la procesa en tiles solapados.
- El solapamiento entre tiles usa una ventana gaussiana 2D para evitar artefactos en los bordes.
- Devuelve `(H//16, W//16, 768)` para `spat_base`.

**`compute_change_map(f1, f2, target_shape, threshold)`**
- Calcula `change = 1 - cosine_similarity(F1, F2)` píxel a píxel.
- Hace upsample bilineal de `(H//16, W//16)` al tamaño original `(H, W)`.
- Binariza con umbral de Otsu automático si `threshold=None`, o con umbral fijo si se especifica.

**`run_sam(rgb_t2, prompts, sam_checkpoint, model_type)`**
- Carga SAM y fija la imagen T2 como contexto.
- Por cada región del mapa de cambio: usa bbox o punto como prompt.
- Intersecta la máscara de SAM con la región de cambio original (si la intersección es vacía, usa la máscara SAM completa).

**`classify_changes_clip(rgb_t1, rgb_t2, sam_results, vocabulary, model_name)`**
- Encoda todo el vocabulario una sola vez (eficiente).
- Por cada región: recorta el bounding box en T1 y T2 con padding de 10 px.
- Calcula similitud coseno entre el recorte de T2 y cada descripción textual.
- Devuelve la etiqueta con mayor similitud, más todos los scores.

#### CLI (cómo se ejecuta)

```powershell
.venv\Scripts\python.exe pipeline.py <t1> <t2> --sam <sam.pth> --hsigma <hsigma.pth> [opciones]
```

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `t1` | Ruta imagen T1 | (requerido) |
| `t2` | Ruta imagen T2 | (requerido) |
| `--sam` | Ruta checkpoint SAM | (requerido) |
| `--hsigma` | Ruta checkpoint HyperSIGMA | (requerido) |
| `--fmt` | Formato: `auto`, `envi`, `mat`, `tiff` | `auto` |
| `--threshold` | Umbral de cambio `[0,1]`. `None` = Otsu | `None` |
| `--min-area` | Área mínima de región en píxeles | `100` |
| `--device` | `cuda` o `cpu` | `cuda` |
| `--out` | Ruta imagen de salida PNG | `resultado_ovcd.png` |
| `--variant` | Variante HyperSIGMA | `spat_base` |

#### Salida

- **`resultado_ovcd.png`**: imagen de 4 paneles (T1 RGB · T2 RGB · mapa de cambio heatmap · T2 con máscaras SAM y etiquetas CLIP superpuestas).
- **Dict de retorno** con `change_map` (float32), `binary_map` (uint8), `n_regions` (int), `detections` (lista de dicts con `mask`, `score`, `change_type`, `clip_scores`).

---

### `hyperglobal/sam.py` — utilidades de SAM reutilizables

Funciones de conveniencia que envuelven `segment_anything` para otros usos fuera del pipeline:

| Función | Descripción |
|---------|-------------|
| `build_sam_predictor(checkpoint, model_type)` | Carga SAM y devuelve un `SamPredictor` listo |
| `predict_from_bbox(predictor, bbox, mask_hint)` | Segmenta dada una caja `[x0,y0,x1,y1]` |
| `predict_from_point(predictor, point, mask_hint)` | Segmenta dado un punto `[cx, cy]` |
| `masks_to_color_overlay(image, masks, alpha)` | Visualización: superpone máscaras sobre una imagen RGB |
| `expand_bbox(bbox, padding, img_h, img_w)` | Expande un bbox con padding respetando los límites de la imagen |

Todas las funciones de predicción aceptan un `mask_hint` opcional (mapa binario del área de interés) que refina la máscara SAM al intersectarla.

---

### `hyperglobal/dataset.py` — descarga de HyperGlobal-450K

Script independiente para descargar partes del dataset de preentrenamiento de HyperSIGMA desde Hugging Face Hub (`WHU-Sigma/HyperGlobal-450K`). No es parte del pipeline de inferencia — es para quienes quieran experimentar con los datos de entrenamiento o fine-tuning.

**Uso:**
```python
# Editar FILENAME = "EO1-part1.zip" (u otro)
python hyperglobal/dataset.py
```

Descarga el ZIP, lo extrae en `hyperglobal_sample/` y lista los archivos con sus metadatos.

---

### `hyper_sam.md` — especificación técnica original

Documento que describe el diseño del pipeline antes de implementarlo. Contiene:
- Justificación de usar HyperSIGMA (modelo foundation) en lugar de HyperSeg (supervisado).
- Explicación de embeddings, distancia coseno, y arquitectura ViT.
- Diagrama ASCII del pipeline completo.
- Instrucciones de implementación paso a paso con firmas de funciones.
- Detalles sobre SAM (prompts, tipos de modelo, coordenadas).
- Detalles sobre CLIP (vocabulario abierto, similitud coseno texto-imagen).

Este archivo fue la entrada principal para generar `pipeline.py`.

---

## 5. Datos de prueba

**`hyperglobal/imgs/barbara/barbara_2013.mat`** y **`barbara_2014.mat`**

- Imágenes hiperespectrales de la zona de Santa Barbara, California.
- Formato MATLAB (`.mat`), array 3D de HSI.
- T1 = año 2013, T2 = año 2014.
- No tienen variable `wavelength` en el `.mat` → el pipeline asigna automáticamente distribución uniforme 400–2500 nm.
- Son los únicos datos de prueba disponibles actualmente. Para usar datos reales de otro sensor, se pasa la ruta con `--fmt envi` o `--fmt tiff` y el header correspondiente.

---

## 6. Pesos descargados

| Archivo | Tamaño | Ubicación | Cómo obtenerlo |
|---------|--------|-----------|----------------|
| `spat-vit-base-ultra-checkpoint-1599.pth` | 1.37 GB | `weights/` | Hugging Face Hub — `WHU-Sigma/HyperSIGMA` |
| `sam_vit_h_4b8939.pth` | ~2.4 GB | raíz del workspace | `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` |

Para descargar HyperSIGMA manualmente:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="WHU-Sigma/HyperSIGMA",
    filename="spat-vit-base-ultra-checkpoint-1599.pth",
    local_dir="weights/",
)
```

Para descargar SAM manualmente:
```powershell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" `
    -OutFile "sam_vit_h_4b8939.pth"
```

---

## 7. Entorno Python

- **Versión:** Python 3.14.2
- **Entorno virtual:** `.venv/` en la raíz del workspace
- **Activar:** `.venv\Scripts\Activate.ps1` (PowerShell)
- **Ejecutar sin activar:** usar la ruta completa `.venv\Scripts\python.exe`

### Paquetes instalados clave

| Paquete | Para qué |
|---------|---------|
| `torch`, `torchvision` | Inferencia en GPU/CPU con los modelos |
| `timm` | Bloques ViT usados por HyperSIGMA |
| `mmengine` | Utilidades de entrenamiento requeridas por SpatViT_cls.py |
| `einops` | Operaciones de tensores usadas por SpatViT_cls.py |
| `spectral` | Carga de imágenes ENVI HSI |
| `scipy` | Interpolación espectral y carga de `.mat` |
| `scikit-learn` | PCA para visualización RGB |
| `numpy`, `opencv-python` | Operaciones de imagen |
| `matplotlib` | Visualización final |
| `transformers` | CLIP via Hugging Face |
| `Pillow` | Conversión de arrays a imágenes para CLIP |
| `rasterio` | Carga de GeoTIFF HSI |
| `segment-anything` | SAM de Meta AI |
| `huggingface_hub` | Descarga de checkpoints |

---

## 8. Problemas encontrados y soluciones

### 8.1 HyperSIGMA no es un paquete pip

**Problema:** `pip install -e HyperSIGMA/` falla porque el repositorio no tiene `setup.py` ni `pyproject.toml`. Es un repositorio de scripts de investigación, no una librería.

**Solución:** Añadir el directorio del modelo al `sys.path` en tiempo de ejecución:
```python
sys.path.insert(0, "HyperSIGMA/ImageClassification/model")
from SpatViT_cls import SpatViT
```

### 8.2 `torch.load` falla con PyTorch ≥ 2.6

**Problema:** PyTorch 2.6 cambió el default de `weights_only` de `False` a `True`. El checkpoint de HyperSIGMA contiene un objeto `argparse.Namespace`, que no está en la lista blanca de globales seguros.

**Error:**
```
_pickle.UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL argparse.Namespace
```

**Solución:** Editar `HyperSIGMA/ImageClassification/model/SpatViT_cls.py` línea 570:
```python
# Antes:
checkpoint = torch.load(pretrained, map_location='cpu')
# Después:
checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
```

### 8.3 CLIP devuelve objeto, no tensor

**Problema:** `clip_model.get_text_features(...)` y `get_image_features(...)` en algunas versiones de `transformers` devuelven un objeto `BaseModelOutputWithPooling` en lugar de un tensor directamente.

**Error:**
```
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'
```

**Solución:** Extraer `.pooler_output` si el resultado no es un tensor:
```python
feats = clip_model.get_text_features(**inputs)
if not isinstance(feats, torch.Tensor):
    feats = feats.pooler_output
```

---

## 9. Flujo de datos completo

```
barbara_2013.mat  ──┐
barbara_2014.mat  ──┤
                    │
                    ▼
            load_hsi_pair()
                    │
         T1:(H,W,C), T2:(H,W,C), wl1, wl2
                    │
                    ▼
          align_spectral() ×2
                    │
         T1_al, T2_al: (H,W,242) normalizado [0,1]
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
  extract_features()    extract_features()
   HyperSIGMA(T1)       HyperSIGMA(T2)
          │                    │
        F1:(H',W',768)     F2:(H',W',768)
          │                    │
          └─────────┬──────────┘
                    ▼
          compute_change_map()
          cosine_sim(F1,F2) → 1-sim
          upsample → (H,W)
          Otsu → binary_map
                    │
         change_map:(H,W)  binary_map:(H,W)
                    │
                    ▼
       change_map_to_prompts()
       connectedComponents → bboxes
                    │
              prompts: [bbox1, bbox2, ...]
                    │
                    ▼
               run_sam()
        SAM + RGB(T2) + prompts
                    │
        masks: [(H,W) bool, ...]  scores: [float,...]
                    │
                    ▼
      classify_changes_clip()
        CLIP(crop_t2) vs vocabulary
                    │
        detections: [{mask, score, change_type, clip_scores}, ...]
                    │
                    ▼
           visualize_ovcd()
                    │
          resultado_ovcd.png
```

---

## 10. Comando para ejecutar el pipeline

```powershell
# Activar entorno (opcional, o usar ruta completa)
.venv\Scripts\Activate.ps1

# Ejecutar con los datos de barbara
.venv\Scripts\python.exe pipeline.py `
    hyperglobal/imgs/barbara/barbara_2013.mat `
    hyperglobal/imgs/barbara/barbara_2014.mat `
    --sam    sam_vit_h_4b8939.pth `
    --hsigma weights/spat-vit-base-ultra-checkpoint-1599.pth `
    --variant spat_base `
    --fmt    mat `
    --device cpu
```

La salida se guarda en `resultado_ovcd.png`. Con `--device cpu` es lento (varios minutos); con GPU CUDA es ~10× más rápido.

---

## 11. Estado actual

- [x] `pipeline.py` implementado con los 10 pasos
- [x] `hyperglobal/sam.py` implementado con utilidades SAM
- [x] Todos los paquetes Python instalados
- [x] HyperSIGMA clonado y checkpoint descargado
- [x] SAM checkpoint descargado
- [x] Pipeline ejecutado sin errores (genera `resultado_ovcd.png`)
- [ ] Evaluación cuantitativa de resultados
- [ ] Prueba con imágenes de otros sensores (GF-5B, AVIRIS, etc.)
- [ ] Optimización de velocidad (batch processing, GPU)
