# Etapa: rgb (PASO 5)

Composición RGB `(H, W, 3) uint8` desde el cubo HSI alineado — entrada para SAM y CLIP.

- `band_select` (default): bandas más cercanas a R=660, G=550, B=470 nm.
- `pca`: primeras 3 componentes (más contraste, sin interpretación física de color).
- Cada canal se estira a [0, 255] por separado.
- OJO: si las wavelengths son ficticias (fallback del loader, p. ej. barbara), la
  selección de bandas no corresponde a colores reales — el RGB es solo orientativo.
