"""
Open Vocabulary Change Detection (OVCD) Pipeline
HyperSIGMA → Change Map → SAM → CLIP

Este archivo es el ORQUESTADOR: no contiene lógica de etapas, solo las llama.
La implementación de cada etapa vive en el paquete stages/ (un directorio por
etapa, cada uno con su propio CLAUDE.md de contexto):

    stages/loading           PASO 1  carga de imágenes HSI bitemporales
    stages/alignment         PASO 2  alineación espectral
    stages/features          PASO 3  embeddings HyperSIGMA
    stages/change_detection  PASOS 4 y 6  mapa de cambio + prompts SAM
    stages/rgb               PASO 5  composición RGB desde HSI
    stages/segmentation      PASO 7  máscaras SAM
    stages/labeling          PASO 8  etiquetado open-vocabulary con CLIP
    stages/visualization     PASO 9  imagen de salida

Compatibilidad: todas las funciones públicas se re-exportan desde aquí, por lo
que `from pipeline import load_hsi_pair, ovcd_pipeline, ...` sigue funcionando
igual que antes del refactor.

Requisitos de instalación:
    pip install torch torchvision timm
    pip install spectral scipy scikit-learn
    pip install numpy opencv-python matplotlib
    pip install transformers
    pip install git+https://github.com/facebookresearch/segment-anything.git
    pip install rasterio                   # para GeoTIFF

    # HyperSIGMA (clonar en la raíz del repo; NO es paquete pip)
    git clone https://github.com/WHU-Sigma/HyperSIGMA.git

    # Pesos SAM (vit_h)
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from stages.loading import load_hsi_pair
from stages.alignment import HYPERION_WAVELENGTHS, align_spectral
from stages.features import extract_features, load_hypersigma
from stages.change_detection import change_map_to_prompts, compute_change_map
from stages.rgb import hsi_to_rgb
from stages.segmentation import run_sam
from stages.labeling import DEFAULT_VOCABULARY, classify_changes_clip
from stages.captioning import CAPTIONER_WAVELENGTHS, caption_regions, load_captioner
from stages.visualization import visualize_ovcd

__all__ = [
    "HYPERION_WAVELENGTHS",
    "CAPTIONER_WAVELENGTHS",
    "DEFAULT_VOCABULARY",
    "load_hsi_pair",
    "align_spectral",
    "load_hypersigma",
    "extract_features",
    "compute_change_map",
    "hsi_to_rgb",
    "change_map_to_prompts",
    "run_sam",
    "classify_changes_clip",
    "load_captioner",
    "caption_regions",
    "visualize_ovcd",
    "ovcd_pipeline",
]


# ===========================================================================
# Pipeline completo
# ===========================================================================

def ovcd_pipeline(
    path_t1: str,
    path_t2: str,
    sam_checkpoint: str,
    hypersigma_checkpoint: str,
    hsi_format: str = "auto",
    target_wavelengths: Optional[np.ndarray] = None,
    change_threshold: Optional[float] = None,
    prompt_method: str = "bbox",
    min_region_area: int = 100,
    vocabulary: Optional[list[str]] = None,
    sam_model_type: str = "vit_h",
    hypersigma_variant: str = "spat_base",
    device: str = "cuda",
    save_viz: str = "resultado_ovcd.png",
    captions: bool = False,
    captioner_checkpoint: Optional[str] = None,
) -> dict:
    """
    Pipeline OVCD completo: carga → alineación → HyperSIGMA → cambio → SAM → CLIP.

    Parameters
    ----------
    path_t1, path_t2            : rutas a las imágenes HSI bitemporales
    sam_checkpoint               : ruta al .pth de SAM
    hypersigma_checkpoint        : ruta al .pth de HyperSIGMA
    hsi_format                   : "auto" | "envi" | "mat" | "tiff"
    target_wavelengths           : espacio espectral de referencia (None → Hyperion)
    change_threshold             : umbral para binarizar el mapa (None → Otsu)
    prompt_method                : "bbox" | "points"
    min_region_area              : área mínima en px para considerar una región
    vocabulary                   : lista de descripciones de cambio (None → default)
    sam_model_type               : "vit_h" | "vit_l" | "vit_b"
    hypersigma_variant           : "spat_base" | "spat_large" | "spec_base" | "spec_large"
    device                       : "cuda" o "cpu"
    save_viz                     : ruta de la imagen de visualización de salida
    captions                     : si True, genera captions HyperCap por región
                                   (caption_t1/caption_t2 en cada detección)
    captioner_checkpoint         : .pth del captioner entrenado (None → sin
                                   entrenar, solo para pruebas de plomería)

    Returns
    -------
    dict con claves:
        - change_map  : (H, W) float32 mapa continuo
        - binary_map  : (H, W) uint8 mapa binarizado
        - n_regions   : int número de regiones detectadas
        - detections  : list[dict] resultados enriquecidos con CLIP
    """
    if target_wavelengths is None:
        target_wavelengths = HYPERION_WAVELENGTHS
    if vocabulary is None:
        vocabulary = DEFAULT_VOCABULARY

    # 1. Carga
    print("[1/8] Cargando imágenes...")
    t1, t2, wl1, wl2 = load_hsi_pair(path_t1, path_t2, hsi_format)
    print(f"      T1: {t1.shape}  T2: {t2.shape}")

    # 2. Alineación espectral
    print("[2/8] Alineando espectralmente...")
    t1_al = align_spectral(t1, wl1, target_wavelengths)
    t2_al = align_spectral(t2, wl2, target_wavelengths)

    # 3. Embeddings HyperSIGMA
    print("[3/8] Cargando HyperSIGMA...")
    model = load_hypersigma(
        hypersigma_checkpoint,
        variant=hypersigma_variant,
        in_chans=int(target_wavelengths.shape[0]),
        device=device,
    )
    print("[4/8] Extrayendo embeddings...")
    f1 = extract_features(t1_al, model, patch_size=16, device=device)
    f2 = extract_features(t2_al, model, patch_size=16, device=device)
    print(f"      F1: {f1.shape}  F2: {f2.shape}")

    # 4. Mapa de cambio
    print("[5/8] Calculando mapa de cambio...")
    change_map, binary_map = compute_change_map(
        f1, f2,
        target_shape=(t1.shape[0], t1.shape[1]),
        threshold=change_threshold,
    )

    # 5. Imágenes RGB
    rgb_t1 = hsi_to_rgb(t1_al, target_wavelengths)
    rgb_t2 = hsi_to_rgb(t2_al, target_wavelengths)

    # 6. Prompts
    print("[6/8] Generando prompts...")
    prompts = change_map_to_prompts(binary_map, min_region_area, prompt_method)
    print(f"      {len(prompts)} región(es) de cambio detectada(s)")

    if len(prompts) == 0:
        print("      No se detectaron regiones de cambio significativas.")
        return {
            "change_map": change_map,
            "binary_map": binary_map,
            "n_regions": 0,
            "detections": [],
        }

    # 7. SAM
    print("[7/8] Ejecutando SAM...")
    sam_results = run_sam(rgb_t2, prompts, sam_checkpoint, sam_model_type)

    # 8. OVCD con CLIP
    print("[8/8] Clasificando cambios con CLIP...")
    detections = classify_changes_clip(rgb_t1, rgb_t2, sam_results, vocabulary)

    # 8b (opcional). Captions HyperCap por región
    if captions:
        print("[+]   Generando captions HyperCap por región...")
        t1_cap = align_spectral(t1, wl1, CAPTIONER_WAVELENGTHS)
        t2_cap = align_spectral(t2, wl2, CAPTIONER_WAVELENGTHS)
        cap_model, cap_processor = load_captioner(
            checkpoint=captioner_checkpoint,
            bands=int(CAPTIONER_WAVELENGTHS.shape[0]),
            device=device,
        )
        detections = caption_regions(t1_cap, t2_cap, detections,
                                     cap_model, cap_processor)

    # Visualización
    visualize_ovcd(rgb_t1, rgb_t2, change_map, detections, save_viz)

    for det in detections:
        score = det["score"]
        ctype = det.get("change_type", "?")
        print(f"      Región {det['region_id']:>3}: score SAM={score:.3f}  tipo={ctype}")
        if captions:
            print(f"            T1: {det.get('caption_t1', '')}")
            print(f"            T2: {det.get('caption_t2', '')}")

    return {
        "change_map": change_map,
        "binary_map": binary_map,
        "n_regions": len(detections),
        "detections": detections,
    }


# ===========================================================================
# CLI mínima
# ===========================================================================

if __name__ == "__main__":
    import argparse

    # Solo ASCII aquí: la consola Windows (cp1252) no imprime '→' en --help
    parser = argparse.ArgumentParser(
        description="OVCD Pipeline: HyperSIGMA -> SAM -> CLIP (+ captions HyperCap)"
    )
    parser.add_argument("t1",  help="Ruta imagen HSI T1")
    parser.add_argument("t2",  help="Ruta imagen HSI T2")
    parser.add_argument("--sam",   required=True, help="Checkpoint SAM (.pth)")
    parser.add_argument("--hsigma", required=True, help="Checkpoint HyperSIGMA (.pth)")
    parser.add_argument("--fmt",   default="auto",
                        choices=["auto", "envi", "mat", "tiff"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="Umbral de cambio [0,1]. None=Otsu automático.")
    parser.add_argument("--min-area", type=int, default=100,
                        help="Área mínima de región en px")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out", default="resultado_ovcd.png",
                        help="Ruta de la imagen de salida")
    parser.add_argument("--variant", default="spat_base",
                        choices=["spat_base", "spat_large", "spec_base", "spec_large"])
    parser.add_argument("--captions", action="store_true",
                        help="Generar captions HyperCap por región de cambio")
    parser.add_argument("--captioner-ckpt", default=None,
                        help="Checkpoint del captioner entrenado (.pth). Sin él "
                             "los captions son texto sin sentido (solo plomería).")
    args = parser.parse_args()

    results = ovcd_pipeline(
        path_t1=args.t1,
        path_t2=args.t2,
        sam_checkpoint=args.sam,
        hypersigma_checkpoint=args.hsigma,
        hsi_format=args.fmt,
        change_threshold=args.threshold,
        min_region_area=args.min_area,
        device=args.device,
        save_viz=args.out,
        hypersigma_variant=args.variant,
        captions=args.captions,
        captioner_checkpoint=args.captioner_ckpt,
    )
    print(f"\nRegiones detectadas: {results['n_regions']}")
