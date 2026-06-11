# Estado del proyecto — actualizar al final de cada sesión

Última actualización: 2026-06-10

## Hecho
- Pipeline end-to-end funcional sobre el par barbara (2013/2014).
- Refactor a paquete: `pipeline.py` monolítico → `stages/<etapa>/` con un `CLAUDE.md`
  por etapa. `pipeline.py` quedó como orquestador + CLI; API pública intacta.
- Repo git inicializado (el snapshot pre-refactor es el primer commit).

## Próximos pasos (en orden)
1. **CLI de vocabulario** — añadir `--vocab`, `--query`, `--min-score` al argparse de
   `pipeline.py`. Plan detallado en `next_steps.md` (Fase 1).
2. **Scoring diferencial CLIP** — usar `sim_t2 − sim_t1` en vez de solo `sim_t2` en
   `stages/labeling/clip_stage.py` (Fase 2 de `next_steps.md`).
3. **Evaluación cuantitativa** — dataset GETNET (EO-1 Hyperion con ground truth),
   script `evaluate.py` con F1/Precision/Recall (Fase 3 de `next_steps.md`).
4. **Integración HyperCap** — pendiente confirmar cuál HyperCap es (¿software de
   captura HinaLea/TruTag?). Tocará `stages/loading` (entrada ENVI desde la cámara),
   más co-registro T1/T2 y manejo del rango VNIR vs referencia Hyperion.

## Problemas conocidos / deuda
- El CLI ofrece `--variant spec_base|spec_large` pero `_SPAT_VARIANT_CFG`
  (stages/features) solo define `spat_*` → esas opciones fallarían con ValueError.
- SAM y CLIP se cargan desde disco en cada llamada (`run_sam`,
  `classify_changes_clip`) — cachear si se procesan muchos pares.
- `sam_vit_h_4b8939.pth` está duplicado (raíz y `weights/`) — se puede borrar una
  copia y liberar ~2.4 GB.
- El par barbara no trae wavelengths → fallback uniforme 400–2500 nm (la alineación
  espectral trabaja con longitudes de onda ficticias).
