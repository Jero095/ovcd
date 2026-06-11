# Etapa: visualization (PASO 9)

Genera el PNG de salida de 4 paneles: T1 RGB · T2 RGB · heatmap del mapa de cambio ·
T2 con máscaras SAM coloreadas + leyenda con etiquetas CLIP + id de región en el
centroide.

- matplotlib se importa dentro de la función (no es dependencia dura del paquete).
- Mejora planificada relacionada (next_steps.md Fase 1): cuando exista `--query`,
  esta etapa deberá mostrar solo las regiones cuya etiqueta coincida.
