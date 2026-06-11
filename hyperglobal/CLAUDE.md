# hyperglobal/ — datos de prueba y utilidades

- `imgs/barbara/barbara_2013.mat` y `barbara_2014.mat`: par bitemporal de prueba
  (Santa Barbara, CA; 984×740×224). NO traen wavelengths → el loader usa el
  fallback uniforme 400–2500 nm. Son los únicos datos de prueba actuales.
- `dataset.py`: descarga partes de HyperGlobal-450K (dataset de preentrenamiento de
  HyperSIGMA) desde HF Hub. NO es parte del pipeline de inferencia.
- `sam.py`: utilidades SAM independientes del pipeline (predictor, bbox/punto,
  overlays). `stages/segmentation` tiene su propia implementación; estas funciones
  son para experimentos sueltos.
