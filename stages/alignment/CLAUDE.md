# Etapa: alignment (PASO 2)

Reinterpolación espectral al espacio de referencia + normalización.

- `HYPERION_WAVELENGTHS`: 242 bandas, 426–2395 nm (EO-1 Hyperion, el sensor de
  preentrenamiento de HyperSIGMA). Es el `target_wl` por defecto del pipeline.
- `align_spectral` usa `scipy.interpolate.interp1d` lineal; wavelengths fuera del
  rango fuente se rellenan con 0 (¡bandas vacías si el sensor no cubre SWIR!).
- Después de interpolar aplica min-max POR BANDA → salida en [0, 1].
- Invariante: `in_chans` de HyperSIGMA (stages/features) debe ser igual a
  `len(target_wl)`. Si cambias el espacio de referencia, cambia ambos.
- Para la etapa de captioning HyperCap (futura): cada escena del dataset usa un
  sensor distinto (IP/KSC: AVIRIS 224b, Houston13: CASI 144b, Botswana: Hyperion).
  El captioner esperará un espacio espectral concreto → alinear con esta etapa
  usando ese espacio como `target_wl`.
