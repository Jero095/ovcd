# Figuras del plan

Lista única y autoritativa de las figuras que se incluirán en el plan cuando se redacte el
LaTeX. Están agrupadas por sección. Ninguna existe todavía como archivo: todas hay que
generarlas o dibujarlas. No copiar figuras con derechos de autor de otras publicaciones.

## Marco teórico

| # | Figura | Qué es | Fuente / cómo se genera | Dónde va |
|---|--------|--------|-------------------------|----------|
| 1 | Firmas espectrales | Gráfica de reflectancia frente a banda para varios materiales de la escena | Se genera de los datos de prueba extrayendo el espectro de píxeles representativos de `hyperglobal/imgs/barbara/` | Sección 2.1, Imágenes hiperespectrales |
| 2 | Representación del hipercubo | Esquema del cubo con dos ejes espaciales y un eje espectral | Ilustración propia, dibujada; no copiada de otra fuente | Sección 2.1, Imágenes hiperespectrales |
| 3 | Par bitemporal de prueba | Composiciones RGB de las dos capturas de la misma escena | `hyperglobal/imgs/barbara/barbara_2013.mat` y `barbara_2014.mat`, renderizadas con la etapa `rgb` del pipeline | Sección 2.2, Detección de cambios en imágenes bitemporales |

## Estado del arte

| # | Figura | Qué es | Fuente / cómo se genera | Dónde va |
|---|--------|--------|-------------------------|----------|
| 4 | Comparación de benchmarks | Gráfica o tabla del desempeño de los modelos fundacionales en sus benchmarks | Cifras reportadas en los papers citados, p. ej. las 16 tareas de HyperSIGMA o el zero-shot de CLIP en ImageNet | Estado del arte, sección 3.2 o 3.5 |

## Notas

- Las figuras 1 y 3 salen de los datos de prueba y requieren ejecutar código del pipeline
  (extraer espectros y renderizar los RGB). Como estamos en fase de redacción, se generarán
  con permiso explícito cuando lleguemos a esa parte.
- La figura 2 es un diagrama que se dibuja, no sale de datos.
- La figura 4 se arma con cifras ya publicadas en las fuentes del `.bib`; hay que citar cada
  cifra a su paper.
- Candidata opcional para la Metodología, no listada aquí todavía: un diagrama del pipeline
  completo (T1, T2 → HyperSIGMA → mapa de cambio → SAM → CLIP → descripción). Si se decide
  incluir, se añade a esta lista.
