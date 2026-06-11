# Etapa: loading (PASO 1)

Carga pares HSI bitemporales → `(H, W, C) float32` + wavelengths `(C,)` en nm.

- Formatos: ENVI (`.hdr`), MATLAB (`.mat`), GeoTIFF (`.tif`); `fmt="auto"` infiere
  por extensión.
- `.mat`: toma el array 3D más grande del archivo; si `shape[0] < shape[2]` asume
  `(C,H,W)` y transpone a `(H,W,C)`. Busca wavelengths en claves
  `wavelength|wl|lambda` (con variantes de mayúsculas).
- Sin metadatos de wavelength → fallback uniforme 400–2500 nm con warning. El par
  de prueba barbara cae en este caso.
- **Punto de integración futuro de HyperCap**: la cámara exporta ENVI con `.hdr`,
  así que la entrada nueva se conecta aquí. Pendiente: co-registro T1/T2 y rango
  VNIR (~400–1000 nm) frente a la referencia Hyperion (ver stages/alignment).
