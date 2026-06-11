# Estado del proyecto — actualizar al final de cada sesión

Última actualización: 2026-06-11

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
4. **Integración HyperCap** — Fases 0, 1 (código) y 2 COMPLETADAS 2026-06-11:
   etapa `stages/captioning/` integrada (`--captions` en el CLI), script de
   entrenamiento listo en `experiments/train_captioner_colab.py`.
   **ÚNICO PASO PENDIENTE: entrenar en Colab (GPU, ~1-2 h en T4) y poner el
   checkpoint en `weights/`** — sin él los captions son texto sin sentido.
   Después: Fase 3 (caption de cambio bitemporal + captions en la visualización).
   Detalles en `docs/hypercap-plan.md`.

## Problemas conocidos / deuda
- ~~Pipeline no determinista~~ **RESUELTO 2026-06-11**: el checkpoint de HyperSIGMA
  no trae los pesos `sampling_offsets` ni `patch_embed` (se inicializan al azar);
  ahora `load_hypersigma` fija `torch.manual_seed(seed)` (default `seed=0`) antes de
  construir el modelo → dos cargas producen features bit-idénticos (verificado,
  max diff 0.0). Verificado también a nivel de pipeline COMPLETO: dos corridas
  end-to-end sobre barbara dieron resultados idénticos (mismo change_map, mismas
  117 regiones, mismas máscaras y etiquetas). `seed=None` recupera el
  comportamiento antiguo.
- El CLI ofrece `--variant spec_base|spec_large` pero `_SPAT_VARIANT_CFG`
  (stages/features) solo define `spat_*` → esas opciones fallarían con ValueError.
- SAM y CLIP se cargan desde disco en cada llamada (`run_sam`,
  `classify_changes_clip`) — cachear si se procesan muchos pares.
- `sam_vit_h_4b8939.pth` está duplicado (raíz y `weights/`) — se puede borrar una
  copia y liberar ~2.4 GB.
- El par barbara no trae wavelengths → fallback uniforme 400–2500 nm (la alineación
  espectral trabaja con longitudes de onda ficticias).
