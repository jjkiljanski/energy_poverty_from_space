# Energy poverty prediction: research workflow

Develop ridge regression and random forest models using satellite and basic
administrative predictors. Each architecture/target/outer selection is independent.
The shared Python runner is the only execution layer; this README explains the
research order. There is no workflow notebook.

## Stages

| Stage | Tuning and fitting | Saved predictions |
| --- | --- | --- |
| `develop` | Exclude one outer fold; tune across the other four (fit three, validate one) | `inner_oof`, `train_in_sample` on those four folds only |
| Freeze | Checksum a completed development run | No new predictions |
| `outer_evaluate` | Reuse verified frozen four-fold pipelines; no tuning or refitting | `outer_holdout`, plus saved inner diagnostics |
| `final_evaluate` | Retune with all five training folds, then fit all training rows | `final_test`, plus training diagnostics |

Selections are `holdout_A`, `holdout_B`, `holdout_C`, `holdout_D`, `holdout_E`,
and `final_test`. Letters map to the sorted stored fold labels, recorded in each
run. The saved CSV is not relabeled or regenerated. The split-selection notebook
`3_0_spatial_test_set_selection.ipynb` remains responsible for those assignments.

1. **Develop:** start with one context and the base targets. Inspect inner residuals,
   change predictors/searches as needed, and save new runs. Inner residuals reuse
   parameters selected on their validation outcomes: they are tuning diagnostics.
2. **Freeze for review:** finish the desired contexts/architectures/targets, freeze
   the run, then evaluate the excluded outer folds. Partial runs can be frozen, but
   cannot be presented as complete five-fold outer evaluation.
3. **Revise after critique:** changing data, predictors, search, code or dependencies
   requires a new development run and freeze. Do not overwrite prior frozen results.
4. **Final freeze and assessment:** invoke final evaluation explicitly. It retunes
   the frozen procedure on all training data, rather than averaging outer parameter
   sets or picking the best outer model. RF can be selected as the final architecture.

Development never predicts/scores the outer or final test holdout. Input loading
still validates the complete table and provenance hashes it: held-out information
is not claimed to be physically inaccessible. Inspecting inner results across all
contexts involves every training fold. New freezes do not erase adaptive human
decisions or earlier outer-result inspection.

The final test was previously inspected/replaced (see `docs/model_tuning_history.md`).
It is reserved in this workflow, not historically untouched. Disclose repeated
evaluations and distinguish exploratory development from final assessment.

## Targets and inputs

| Group | Columns |
| --- | --- |
| `base` | `epg_cooling`, `epg_heating` |
| `additional` | `demand_cooling`, `consumption_cooling`, `gap_pct_cooling`, `demand_heating`, `consumption_heating`, `gap_pct_heating` |
| `all` | All eight above |

The new CSV is configured by `epvi_targets_csv` in `pipeline/config/paths.json`;
`epvi_csv` remains for historical consumers. Its raw ID scheme matches the old
source, so the established EPVI correction is applied once. Missing/duplicate IDs
fail; empty export column `Column1` is ignored. Names/spatial metadata come from
saved assignments. Missing outcomes are filtered per target without moving folds;
empty inner folds fail. Optional `predictors` selects a subset of the allowed
satellite/basic-admin columns.

## Entry points

- `3_1_model_training.ipynb`: select model, target group, and holdout; preview and run development.
- `3_2_model_performance.ipynb`: compare saved metrics and predictions by run, context, and role.
- `3_3_model_prediction_exploration.ipynb`: inspect parish plots, regional residuals,
  maps, and optional predictor relationships. It replaces the previous EPG-only
  exploration and reads saved predictions without fitting models.

The two inspection notebooks accept multiple run IDs and model IDs. They export the
selected tables, and the exploration notebook can save its figures, to a folder you
choose. Optional predictor inspection verifies that the current modeling table still
matches the run before joining predictor values to historical residuals.

From the repository root, in the activated project Conda environment:

```powershell
python -m pipeline.modeling --dry-run
python -m pipeline.modeling --models ridge --selections holdout_A --targets base
python -m pipeline.modeling --models ridge random_forest --selections all_outer_folds --targets all

python -m pipeline.modeling --freeze-run <run_directory>
python -m pipeline.modeling --stage outer_evaluate --frozen <run_directory>/frozen.json
python -m pipeline.modeling --stage final_evaluate --frozen <run_directory>/frozen.json --models random_forest
```

Use `--config` for development settings, `--paths-config` for input paths and
`--output-root` for output location. Evaluation rejects `--config` overrides;
settings come from the freeze. Evaluation also supports `--dry-run` and subsets
of the frozen architectures, targets and outer contexts.

## Artifacts

Outputs default to `<external_data_root>/outputs/epvi_prediction/runs/<run_id>/`.
A run has one index/provenance manifest, saved assignments, and readable directories
such as `random_forest-holdout_B-epg_cooling-<short_hash>/`.

Each pair holds `fit.json` (parameters/model ID), `pipeline.joblib`, search results,
predictions, metrics, fold summaries, regional holdout metrics and diagnostics.
`frozen.json` is one optional checksum snapshot per development run, not a manifest
per inner fit. Model IDs hash fit specification/data/code, excluding timestamps.
Provenance hashes actual input CSVs and the source/index definition manifests.

Performance reads saved predictions rather than silently regenerating them.
Load joblib pipelines only from trusted sources; checksums are not signatures.

```python
from pipeline.modeling import load_run

results = load_run("<exact_run_directory>")
display(results.comparison_table())  # default role follows the stage
inner_A = results.predictions.query("selection == 'holdout_A' and prediction_role == 'inner_oof'")
# Requires a complete outer-evaluation run:
outer_oof = results.outer_oof_predictions()
```

Inner residual sets overlap across contexts: never pool them as if each parish
appeared once. Keep `selection`, `model`, `target`, `ID_norm`, `prediction_role`.
Outer OOF tables are the starting point for step-4 residual explanation, subject
to the research-history limitations above.

Failed runs retain completed pairs, inspectable via
`load_run(path, allow_incomplete=True)`, but cannot be frozen. No automatic resume.
Old schema-1 tables remain readable but cannot be frozen as nested experiments.
