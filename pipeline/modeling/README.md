# Nested spatial modeling API

Research order and commands live in
[3_epvi_prediction/README.md](../3_epvi_prediction/README.md).

| Module | Responsibility |
| --- | --- |
| `data.py` | Validate input tables and allowed predictors |
| `targets.py` | Base/additional groups and target specs |
| `splits.py` | Map A-E to saved labels; outer subsets and inner CV |
| `models/` | Ridge/RF pipelines, searches and diagnostics |
| `metrics.py` | Shared metrics/objective and undefined-score reasons |
| `training.py` | Inner tuning and staged execution |
| `artifacts.py` | Persistence, model identity and result loading |
| `procedure.py` | Freeze snapshots and data/code/artifact verification |
| `inspection.py` | Select saved result tables for notebooks; optionally verify predictor inputs |
| `__main__.py` | CLI calling the same API |

## Develop

```python
from pipeline.modeling import (
    resolve_targets, load_project_data, SearchConfig,
    preview_experiment, run_experiment, freeze_run, evaluate_frozen,
)
targets = resolve_targets("base")
dataset = load_project_data(targets, exclude_overseas=True)
search = SearchConfig(seed=42, n_iter=40, n_jobs=2)
plan = preview_experiment(dataset, models=["ridge", "random_forest"],
                          targets=targets, search=search, selections=["holdout_A"])
results = run_experiment(dataset, models=["ridge", "random_forest"],
                         targets=targets, search=search, selections=["holdout_A"],
                         output_root="<output_root>")
```

The API defaults to `develop` and `all_outer_folds`. The notebook/CLI example
starts with `holdout_A` and base targets. Five saved folds produce four-fold
inner CV. Synthetic tests may use three saved folds and two-fold inner CV.
No random resplitting or target-dependent reassignment occurs.

Development saves inner predictions and a fitted outer-training pipeline without
holdout predictions. Learned preprocessing and baseline means use each fitting
partition only. Ridge exhaustively searches alpha; RF samples finite parameter
lists. `SearchConfig.spaces` partially overrides grids; `candidates` overrides
per-model random-search budgets. CV workers parallelize fits; estimator/native
threads are constrained to avoid oversubscription.

## Freeze and evaluate

```python
frozen = freeze_run(results.run_dir)
outer = evaluate_frozen(dataset, frozen, output_root="<output_root>")
# Later, after the final procedure is frozen:
final = evaluate_frozen(dataset, frozen, stage="final_evaluate",
                        models=["random_forest"], output_root="<output_root>")
```

Outer evaluation reuses verified fitted pipelines without tuning/refitting.
Final evaluation reruns the frozen search across all training folds.
Only frozen architecture/target/context subsets are accepted, not search or metric
overrides. Retain all frozen target columns and the original dataset row order/content
even for subset evaluation; the CLI does this automatically. Data, predictors, scope,
code and package versions must match. Changes require redevelopment/refreezing.

Lower-level `train_model` tunes on its supplied fitting subset; `evaluate_model`
predicts without fitting. They do not enforce the research-stage gate themselves:
use `run_experiment`/`evaluate_frozen` for experiments.

## Results

Schema 2 adds `selection` and `outer_fold` (the stored label) to tables. Roles:
`inner_oof`, `train_in_sample`, `outer_holdout`, `final_test`.
Inner sets overlap. `outer_oof_predictions()` requires complete outer coverage.
`comparison_table()` preserves selection, never silently pooling inner residuals.

Search scores use sklearn's negative-loss convention; reported errors are positive.
Mean-fold and pooled metrics differ. Undefined scores carry reasons.
Ridge coefficients and forest importances are separate diagnostics.
`inspection.py` only reads tables; predictor joins verify the recorded modeling table hash.

Run IDs identify executions; content-based model IDs identify fit specification and
training data, excluding timestamps. Provenance records input/definition/code hashes,
dependency versions and Git state. Schema-1 tables remain readable with original roles.
Histogram boosting is no longer supported; historical table reading needs no recipe.
No automatic migration, resume or hidden final evaluation.

## Verify

```powershell
python -m pytest tests/pipeline/modeling -q
```

See [the test review map](../../tests/pipeline/modeling/README.md) for the staged
lifecycle, independent nested reference, outer isolation, target extension, partial
writes and freeze checks.
