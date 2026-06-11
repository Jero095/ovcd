# Pipeline: Open Vocabulary Change Detection en Imágenes HSI Bitemporales

**HyperSIGMA → Mapa de cambio → SAM → OVCD**

---

## El problema que resuelve este pipeline

Tienes dos imágenes hiperespectrales de la misma escena en momentos distintos (T1 y T2). Quieres saber **qué cambió, dónde, y de qué tipo es el cambio** — sin haber definido previamente qué categorías de cambio esperar.

Eso es OVCD: Open Vocabulary Change Detection. El "open vocabulary" significa que el sistema no está restringido a clases predefinidas en tiempo de entrenamiento. Puede detectar y describir cambios que nunca vio durante el entrenamiento.

### Por qué HyperSIGMA en lugar de HyperSeg

HyperSeg es un modelo de **segmentación supervisada**: aprende a clasificar píxeles en clases fijas. Para OVCD eso es un problema porque:

- Las clases se definen en entrenamiento — si cambia el dominio, falla.
- El número de bandas de entrada es fijo — si el sensor cambia, falla.
- No hay concepto de "novedad" — no puede detectar cambios de tipos que no vio.

HyperSIGMA es un **modelo foundation** preentrenado sobre HyperGlobal-450K (450K imágenes hiperespectrales de sensores EO-1 y GF-5B, distribuidas globalmente). Su salida no son clases sino **embeddings densos** — representaciones vectoriales ricas que capturan la información espectral y espacial de cada región. Al comparar los embeddings de T1 y T2, cualquier diferencia significativa en ese espacio es un cambio, sin importar de qué tipo.

---

## Conceptos clave

### Imagen bitempoal HSI

Dos imágenes del mismo lugar, misma extensión espacial, en tiempos distintos:

```
T1: (H, W, C1)   ← imagen hiperespectral en tiempo 1
T2: (H, W, C2)   ← imagen hiperespectral en tiempo 2

C1 puede ≠ C2 si los sensores son distintos
```

Requisito mínimo: las imágenes deben estar **corregistradas** (mismo sistema de referencia espacial, mismo tamaño en píxeles). La alineación espectral la hace el pipeline.

### HyperSIGMA y sus embeddings

HyperSIGMA es un Vision Transformer con mecanismo de Sparse Sampling Attention (SSA) y un módulo de mejora espectral. Su salida es un **feature map denso**:

```
Entrada:  (H, W, C)   imagen HSI de cualquier sensor
Salida:   (H', W', D) feature map  donde D = dimensión del embedding (ej. 768)
```

`H'` y `W'` son las dimensiones espaciales después del patchifying del ViT (generalmente H/16, W/16 para patches de 16×16 px). El modelo fue preentrenado con MAE (Masked Autoencoder) sobre imágenes de todo el mundo, por lo que sus representaciones generalizan a cualquier escena y sensor.

### Mapa de cambio en espacio de embeddings

En lugar de comparar valores espectrales crudos (que dependen del sensor, la atmósfera, la hora), se comparan los embeddings:

```
F1 = HyperSIGMA(T1)        # (H', W', D)
F2 = HyperSIGMA(T2)        # (H', W', D)

change_map = distancia(F1, F2)   # (H', W')  — mayor valor = más cambio
```

La distancia coseno es más robusta que L2 porque normaliza la magnitud:

```
cosine_sim = (F1 · F2) / (||F1|| · ||F2||)
change_map = 1 - cosine_sim    # 0 = sin cambio, 2 = cambio máximo
```

### SAM en el contexto bitemoral

SAM recibe la imagen RGB de T2 (o una composición de T1 y T2) junto con prompts derivados del mapa de cambio. Refina los bordes de las regiones cambiadas con precisión de píxel.

### OVCD — la capa de vocabulario abierto

Una vez que SAM produce máscaras precisas por región cambiada, CLIP (u otro modelo multimodal) describe qué tipo de cambio ocurrió. CLIP fue entrenado con millones de pares imagen-texto y puede describir regiones sin clases predefinidas.

---

## Diagrama del pipeline completo

```
┌─────────────────────┐     ┌─────────────────────┐
│  HSI Tiempo T1      │     │  HSI Tiempo T2       │
│  (H, W, C1)         │     │  (H, W, C2)          │
└──────────┬──────────┘     └──────────┬───────────┘
           │                           │
           ▼                           ▼
┌──────────────────────────────────────────────────┐
│  ALINEACIÓN ESPECTRAL                             │
│  Interpolar C1 y C2 a espacio de longitudes      │
│  de onda común → ambas quedan en (H, W, C_ref)   │
└──────────┬───────────────────────────┬───────────┘
           │                           │
           ▼                           ▼
┌──────────────────┐       ┌──────────────────────┐
│  HyperSIGMA(T1)  │       │  HyperSIGMA(T2)       │
│  Salida: F1      │       │  Salida: F2            │
│  (H', W', D)     │       │  (H', W', D)           │
└──────────┬───────┘       └──────────┬────────────┘
           │                          │
           └──────────┬───────────────┘
                      ▼
┌──────────────────────────────────────────────────┐
│  MAPA DE CAMBIO                                   │
│  change_map = 1 - cosine_sim(F1, F2)             │
│  Upsample a (H, W) + umbral adaptativo           │
│  Salida: regiones de cambio binarias o graduales │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  GENERACIÓN DE PROMPTS PARA SAM                   │
│  Por cada región de cambio detectada:             │
│  · Bounding box  →  [x0, y0, x1, y1]            │
│  · Centroide     →  [cx, cy]                     │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  SAM — Segment Anything Model                     │
│  Entrada: imagen RGB de T2 + prompts              │
│  Salida: máscaras precisas por región de cambio  │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  OVCD — Open Vocabulary Change Detection          │
│  Por cada máscara SAM:                            │
│  · Recortar región en T1 y T2                    │
│  · CLIP encoda ambos recortes como embeddings    │
│  · Comparar con descripciones textuales           │
│  · Etiquetar: "deforestación", "inundación",     │
│    "construcción", "cambio desconocido"...        │
└──────────────────────────────────────────────────┘
```

---

## Instrucciones de implementación

### Paso 0 — Entorno

```bash
python -m venv venv
source venv/bin/activate

# Dependencias base
pip install torch torchvision
pip install timm                  # requerido por HyperSIGMA (ViT backbone)
pip install spectral scipy scikit-learn
pip install numpy opencv-python matplotlib
pip install transformers          # para CLIP en la capa OVCD

# SAM de Meta
pip install git+https://github.com/facebookresearch/segment-anything.git

# HyperSIGMA (clonar el repo)
git clone https://github.com/WHU-Sigma/HyperSIGMA.git
cd HyperSIGMA
pip install -e .

# Pesos de SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Los pesos preentrenados de HyperSIGMA se descargan desde el repositorio oficial
(ver `HyperSIGMA/README.md` para links a los checkpoints de cada variante:
`spat_base`, `spat_large`, `spec_base`, `spec_large`).

---

### Paso 1 — Carga de imágenes bitemporales

Cada imagen viene con sus metadatos de longitudes de onda. Son imprescindibles
para la alineación espectral.

**Qué implementar:**

```python
def load_hsi_pair(path_t1: str, path_t2: str,
                  fmt: str) -> tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """
    Retorna: (t1, t2, wavelengths_t1, wavelengths_t2)
    Cada imagen: (H, W, C) float32
    Cada wavelengths: (C,) array en nm
    """
```

Instrucciones por formato:

- **ENVI** (`.hdr`): usar `spectral.io.envi.open()`. Las longitudes de onda
  están en el header como `wavelength`. Leer con `img.metadata['wavelength']`
  y convertir a `np.array` de floats.
- **MATLAB** (`.mat`): el array 3D suele acompañarse de una variable
  `wavelength` o `wl` en el mismo archivo.
- **GeoTIFF** con metadatos: usar `rasterio`. Los metadatos espectrales pueden
  estar en una sidecar `.xml` o en las tags del TIFF.

Si las longitudes de onda no están disponibles, asumir distribución uniforme
entre 400 y 2500 nm como fallback (perderás precisión en la alineación).

---

### Paso 2 — Alineación espectral

Normaliza ambas imágenes al mismo espacio espectral de referencia antes de
pasarlas por HyperSIGMA.

**Qué implementar:**

```python
def align_spectral(hsi: np.ndarray,
                   source_wl: np.ndarray,
                   target_wl: np.ndarray) -> np.ndarray:
    """
    Reinterpolaa las bandas de hsi a target_wl.
    hsi:       (H, W, C_source) float32
    source_wl: (C_source,) en nm
    target_wl: (C_target,) en nm
    Retorna:   (H, W, C_target) float32
    """
```

Instrucciones:

1. Aplanar a `(H*W, C_source)`.
2. Construir interpolador: `scipy.interpolate.interp1d(source_wl, flat, axis=1,
   kind='linear', bounds_error=False, fill_value=0.0)`.
3. Evaluar en `target_wl` y reshape a `(H, W, C_target)`.
4. El `target_wl` es el espacio espectral de referencia de HyperSIGMA
   (típicamente el del sensor EO-1 Hyperion: 426–2395 nm, 242 bandas).

Aplicar también normalización min-max por banda después de la interpolación.

---

### Paso 3 — Extracción de embeddings con HyperSIGMA

HyperSIGMA tiene dos ramas: **Spatial ViT** (enfocada en features espaciales)
y **Spectral ViT** (enfocada en features espectrales). Para OVCD usar la rama
espacial (`spat`) o fusión de ambas.

**Qué implementar:**

```python
def extract_features(hsi: np.ndarray,
                     model,
                     patch_size: int = 16,
                     device: str = "cuda") -> np.ndarray:
    """
    Extrae feature map denso con HyperSIGMA.
    hsi:    (H, W, C) float32 ya alineado
    Retorna: (H', W', D) float32  donde H'=H//patch_size
    """
```

Instrucciones:

1. Convertir a tensor `(1, C, H, W)` y mover al device.
2. Hacer patchifying manual si el modelo no acepta la imagen completa:
   dividir en patches solapados de `patch_size×patch_size` y procesar por lotes.
3. Llamar al encoder del modelo en modo `forward_features` (sin cabeza de
   clasificación): esto devuelve los tokens del ViT.
4. Excluir el token `[CLS]` y hacer reshape de los tokens restantes a
   `(H', W', D)`.
5. Aplicar `torch.no_grad()` durante la inferencia.

Para imágenes grandes (>256×256) procesar por patches solapados y hacer
promedio en las zonas de solape (stride = patch_size // 2).

---

### Paso 4 — Mapa de cambio

Comparar los embeddings de T1 y T2 en el espacio de representación.

**Qué implementar:**

```python
def compute_change_map(f1: np.ndarray,
                       f2: np.ndarray,
                       target_shape: tuple,
                       threshold: float | None = None) -> np.ndarray:
    """
    f1, f2:       (H', W', D) embeddings de T1 y T2
    target_shape: (H, W) tamaño original de la imagen
    threshold:    None = retornar mapa continuo; float = binarizar

    Retorna: change_map (H, W) float32  valores en [0, 1]
    """
```

Instrucciones:

1. **Similitud coseno**: normalizar `f1` y `f2` a norma unitaria por canal D,
   luego calcular `(f1 * f2).sum(axis=-1)`. El mapa de cambio es
   `1 - similitud`.
2. **Upsample**: el feature map es H/16 × W/16. Usar
   `cv2.resize(change_map, (W, H), interpolation=cv2.INTER_LINEAR)` para
   volver al tamaño original.
3. **Normalización**: escalar a `[0, 1]`.
4. **Umbral adaptativo**: si `threshold=None`, usar el método de Otsu
   (`cv2.threshold` con `cv2.THRESH_OTSU`) para binarizar automáticamente.
   El umbral de Otsu es más robusto que un valor fijo.
5. **Opcional — clustering**: en lugar de umbral, aplicar KMeans con k=2
   (cambio / no-cambio) sobre los valores del mapa. Útil cuando los cambios
   tienen intensidades muy distintas.

---

### Paso 5 — Imagen RGB para SAM

SAM necesita una imagen RGB. Generarla desde T2 (la imagen más reciente) para
que las máscaras correspondan al estado actual.

**Qué implementar:**

```python
def hsi_to_rgb(hsi: np.ndarray,
               wavelengths: np.ndarray) -> np.ndarray:
    """
    Selecciona las bandas más cercanas a R(660nm), G(550nm), B(470nm).
    hsi:         (H, W, C) float32 normalizado
    wavelengths: (C,) en nm
    Retorna:     (H, W, 3) uint8
    """
```

Instrucciones:

1. Encontrar el índice de la banda más cercana a 660, 550 y 470 nm con
   `np.argmin(np.abs(wavelengths - target_nm))`.
2. Extraer las tres bandas, normalizar cada una a `[0, 255]`, castear a
   `uint8`.
3. Opcionalmente mejorar contraste con ecualización de histograma por canal.

Alternativa: usar PCA sobre T2 y tomar las primeras 3 componentes (más
informativo visualmente pero sin interpretación física de color).

---

### Paso 6 — Generación de prompts para SAM

El mapa de cambio binarizado define las regiones de interés. Extraer un prompt
por región conectada.

**Qué implementar:**

```python
def change_map_to_prompts(change_map_binary: np.ndarray,
                           min_area: int = 100,
                           method: str = "bbox") -> list[dict]:
    """
    change_map_binary: (H, W) uint8 — 0=sin cambio, 255=cambio
    min_area:          descartar regiones menores a este número de píxeles
    method:            "bbox" | "points"
    Retorna lista de dicts con 'region_id', 'bbox' o 'point', 'mask_hint'
    """
```

Instrucciones:

1. Encontrar componentes conectados con `cv2.connectedComponentsWithStats`.
   Cada componente es una región de cambio independiente.
2. Filtrar componentes cuya área (`stats[cc, cv2.CC_STAT_AREA]`) sea menor
   que `min_area` para descartar ruido.
3. Por cada componente válida:
   - **bbox**: extraer `(x0, y0, w, h)` de los stats y convertir a `(x0, y0,
     x0+w, y0+h)` formato xyxy.
   - **punto**: calcular centroide desde los centroids del componente.
   - Guardar la máscara de esa componente como `mask_hint`.

---

### Paso 7 — Inferencia SAM

Igual que en el pipeline anterior, pero los prompts ahora vienen del mapa de
cambio, no de un clasificador supervisado.

**Qué implementar:**

```python
def run_sam(rgb_t2: np.ndarray,
            prompts: list[dict],
            sam_checkpoint: str,
            model_type: str = "vit_h") -> list[dict]:
    """
    rgb_t2:  (H, W, 3) uint8 — imagen RGB de T2
    Retorna: lista de dicts con 'region_id', 'mask' (H,W bool), 'score'
    """
```

El flujo es idéntico al pipeline base:

1. `predictor.set_image(rgb_t2)` — una sola vez.
2. Por cada prompt, llamar `predictor.predict(...)` con `multimask_output=False`.
3. Intersectar `sam_mask & mask_hint` para confinar SAM dentro de la región
   de cambio detectada.

> Importante: SAM opera sobre T2 (o una composición T1-T2 en falso color).
> Usar T2 en solitario es más limpio — T2 representa el estado actual y SAM
> produce máscaras que describen los objetos presentes *después* del cambio.

---

### Paso 8 — OVCD con CLIP

Esta es la capa que hace el pipeline "open vocabulary". CLIP compara el
contenido visual de cada región cambiada contra descripciones textuales.

**Qué implementar:**

```python
def classify_changes_clip(rgb_t1: np.ndarray,
                           rgb_t2: np.ndarray,
                           sam_results: list[dict],
                           vocabulary: list[str]) -> list[dict]:
    """
    Por cada región enmascarada, compara T1 y T2 contra el vocabulario.
    vocabulary: lista de strings describiendo tipos de cambio posibles
                ej. ["deforestation", "urban expansion", "flooding",
                     "crop harvest", "fire damage", "unknown change"]
    Retorna: sam_results enriquecidos con 'change_type' y 'clip_scores'
    """
```

Instrucciones:

1. Cargar CLIP: `transformers.CLIPModel` y `CLIPProcessor`.
2. Por cada resultado SAM:
   - Recortar la región de T1 y T2 usando la máscara: aplicar la máscara y
     extraer el bounding box del recorte.
   - Encodar ambos recortes como embeddings de imagen con `model.get_image_features`.
   - Encodar el vocabulario completo como embeddings de texto con
     `model.get_text_features`.
   - Calcular similitud coseno entre el embedding de T2 y cada texto.
   - El tipo de cambio es el texto con mayor similitud.
   - Opcionalmente, calcular también la similitud de T1 para confirmar que el
     cambio es real (T1 no debería parecerse al label ganador si el cambio es
     genuino).
3. Agregar `change_type` y el vector de scores al dict del resultado.

**Vocabulario recomendado como punto de partida:**

```python
vocabulary = [
    "deforestation or forest loss",
    "urban construction or new buildings",
    "flooding or water expansion",
    "agricultural change or crop rotation",
    "fire damage or burned area",
    "road or infrastructure construction",
    "vegetation regrowth",
    "unknown or unclassified change",
]
```

El vocabulario es completamente editable en tiempo de inferencia — no requiere
reentrenamiento.

---

### Paso 9 — Pipeline completo

```python
def ovcd_pipeline(path_t1: str,
                  path_t2: str,
                  sam_checkpoint: str,
                  hypersigma_checkpoint: str,
                  hsi_format: str       = "envi",
                  target_wavelengths    = None,   # np.ndarray en nm
                  change_threshold      = None,   # None = Otsu automático
                  prompt_method: str    = "bbox",
                  min_region_area: int  = 100,
                  vocabulary: list[str] = None,
                  save_viz: str         = "resultado_ovcd.png") -> dict:

    # 1. Carga
    t1, t2, wl1, wl2 = load_hsi_pair(path_t1, path_t2, hsi_format)

    # 2. Alineación espectral
    t1_aligned = align_spectral(t1, wl1, target_wavelengths)
    t2_aligned = align_spectral(t2, wl2, target_wavelengths)

    # 3. Embeddings HyperSIGMA
    model = load_hypersigma(hypersigma_checkpoint)
    f1    = extract_features(t1_aligned, model)
    f2    = extract_features(t2_aligned, model)

    # 4. Mapa de cambio
    change_map = compute_change_map(f1, f2,
                                    target_shape=(t1.shape[0], t1.shape[1]),
                                    threshold=change_threshold)

    # 5. RGB para SAM y CLIP
    rgb_t1 = hsi_to_rgb(t1_aligned, target_wavelengths)
    rgb_t2 = hsi_to_rgb(t2_aligned, target_wavelengths)

    # 6. Prompts desde el mapa de cambio
    prompts = change_map_to_prompts(change_map, min_region_area, prompt_method)

    # 7. SAM
    sam_results = run_sam(rgb_t2, prompts, sam_checkpoint)

    # 8. OVCD
    results = classify_changes_clip(rgb_t1, rgb_t2, sam_results, vocabulary)

    # 9. Visualización
    visualize_ovcd(rgb_t1, rgb_t2, change_map, results, save_viz)

    return {
        "change_map"  : change_map,
        "n_regions"   : len(results),
        "detections"  : results,
    }
```

---

## Qué sale de cada etapa

| Etapa | Entrada | Salida | Tipo |
|-------|---------|--------|------|
| Carga | archivos HSI | T1, T2, wavelengths | `(H,W,C) float32` |
| Alineación | T1, T2 con wl distintos | T1, T2 en mismo espacio espectral | `(H,W,C_ref) float32` |
| HyperSIGMA | imagen HSI alineada | feature map denso F1, F2 | `(H/16,W/16,768) float32` |
| Mapa de cambio | F1, F2 | mapa de distancia espectral | `(H,W) float32 [0,1]` |
| Prompts | mapa de cambio binario | lista de bboxes / puntos | `list[dict]` |
| SAM | RGB T2 + prompts | máscaras precisas por región | `(H,W) bool` por región |
| CLIP/OVCD | recortes T1, T2 + vocabulario | tipo de cambio por región | `str` + scores |

---

## Checklist de implementación

- [ ] HyperSIGMA clonado, dependencias instaladas, checkpoint descargado
- [ ] Pesos SAM descargados
- [ ] `load_hsi_pair` retorna longitudes de onda reales (no ficticias)
- [ ] `align_spectral` verificado: ambas imágenes quedan en `C_ref` bandas
- [ ] `extract_features` retorna shape `(H/16, W/16, D)` sin errores OOM
- [ ] `compute_change_map` produce valores continuos sensatos (no todo 0 o todo 1)
- [ ] `change_map_to_prompts` genera al menos 1 prompt con imágenes que tienen cambio real
- [ ] SAM produce máscaras con score > 0.5 en regiones de cambio
- [ ] CLIP carga correctamente y el vocabulario tiene al menos 4 entradas
- [ ] Pipeline completo corre de principio a fin con un par bitempoal de prueba

---

## Datasets bitemporales para pruebas

| Dataset | Escenas | Bandas | Tipo de cambio | Fuente |
|---------|---------|--------|----------------|--------|
| HRSCD | urbano/rural Francia | 5 (MS) | uso de suelo | IEEE GRSS |
| Hermiston | agrícola Oregon | 242 (Hyperion) | cultivos | EO-1 archive |
| River HSI | fluvial China | 198 | inundaciones | paper Zheng 2022 |
| WHU-CD | urbano China | RGB+NIR | construcción | WHU-Sigma |

Para desarrollar sin datos bitemporales reales, generar pares sintéticos desde
Indian Pines aplicando transformaciones espectrales artificiales a subregiones
de la imagen.

---

## Errores frecuentes

**`Shape mismatch en HyperSIGMA`**  
El modelo espera que H y W sean múltiplos del patch size (16). Hacer padding
con `np.pad` antes de la inferencia y recortar el feature map después.

**`Mapa de cambio todo ceros o todo unos`**  
Las imágenes no están correctamente normalizadas o la alineación espectral
falló. Verificar que los rangos de wavelengths de T1 y T2 se solapen con
`target_wavelengths`.

**`CLIP scores muy similares para todos los textos`**  
Los recortes son demasiado pequeños o la máscara SAM es todo negro. Agregar
una expansión mínima del bounding box (`+10px` por lado) antes de recortar
para CLIP.

**`OOM en HyperSIGMA con imágenes grandes`**  
Procesar por tiles de 128×128 con solapamiento de 32px. Promediar embeddings
en las zonas solapadas con peso gaussiano para evitar artefactos de borde.

**`Muchas regiones pequeñas de ruido en el mapa de cambio`**  
Aplicar `cv2.morphologyEx` con `cv2.MORPH_OPEN` (erosión + dilatación) sobre
el mapa binarizado antes de extraer componentes conectados. Kernel de 3×3 es
suficiente para eliminar ruido de un píxel.