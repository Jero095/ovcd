# Etapa: features (PASO 3) — HyperSIGMA

Extrae feature maps densos `(H/16, W/16, embed_dim)` con el encoder SpatViT.

- HyperSIGMA NO es paquete pip: `_ensure_hypersigma_on_path` añade
  `HyperSIGMA/ImageClassification/model` al `sys.path` (ruta relativa al CWD →
  ejecutar desde la raíz del repo).
- `model.forward_features(x, patch_size)` devuelve una LISTA
  `[img, feat_0, feat_1, feat_2, feat_3]`; usamos `out[-1]`, de shape
  `(B, embed_dim, H/16, W/16)`.
- Checkpoint: `weights/spat-vit-base-ultra-checkpoint-1599.pth` (spat_base, 768-dim).
  `init_weights()` interpola `pos_embed` y descarta el `patch_embed` RGB.
- Gotcha PyTorch ≥ 2.6: el checkpoint contiene `argparse.Namespace` → necesita
  `weights_only=False` (parcheado en `SpatViT_cls.py:570`, NO revertir).
- Imágenes grandes: tiling de 256 px con solape de 32 y promedio con peso gaussiano
  (`_tiled_forward`) para evitar artefactos de borde y OOM.
- BUG latente: `_SPAT_VARIANT_CFG` solo define `spat_base|spat_large|spat_huge`,
  pero el CLI de pipeline.py ofrece `spec_base|spec_large` → ValueError si se usan.
