# Shared regression framework

The numbered notebooks orchestrate this importable package. Each target gets an
independent tuned estimator; there is no multi-output estimator or shared fitted state.

| Module | Responsibility |
| --- | --- |
| `data.py` | Predictor/target schema, joins, frozen split validation |
| `targets.py` | Stable target IDs, source columns, labels and units |
| `models/` | Fresh pipelines, search spaces, model-specific diagnostics |
| `metrics.py` | Shared metrics, primary scorer, undefined-value reasons |
| `training.py` | Search, full-training fit, spatial OOF, test evaluation |
| `artifacts.py` | Run manifests, incremental pair persistence, explicit run loading |
| `__main__.py` | Command-line orchestration using the same API as notebooks |

## Run

From the repository root, in the activated project's Conda environment:

```powershell
python -m pipeline.modeling --dry-run
python -m pipeline.modeling --config pipeline/modeling/experiment.example.json
```

`--dry-run` loads/validates inputs and prints resolved search spaces and fit counts
without training or creating output directories. Use `--paths-config` for alternate
local input paths and `--output-root` for an alternate output directory. The notebook
exposes the same configuration as Python objects. Preview before expensive searches.

Ridge searches 21 log-spaced alpha values exhaustively. Forest and histogram boosting
each sample up to 40 candidates per target. `SearchConfig.candidates` overrides the
random-search budget per model. `SearchConfig.spaces` partially overrides a model's
grid (for example, smaller tree counts for a smoke run). Unknown override names fail.
Finite candidate lists are required. Resolved spaces are persisted with overrides.
Default parallelism is two CV workers; inner estimator/BLAS thread counts are limited
to avoid oversubscription.

## Extend

**New target in the same CSV:** append `TargetSpec("stable_key", "column")` to the
notebook's `TARGETS`, or the corresponding object to JSON `targets`. Optional `label`
and `units` are metadata. Values are neither clipped nor transformed. Missing columns
fail. Each target drops only its own missing outcomes; missingness that empties a
saved training fold fails explicitly.

**Target from another source:** extend `load_project_data`, or construct a
`ModelingDataset` with explicit predictors and all target columns. Join new outcomes
by normalized parish ID without complete-case filtering across targets. Keep all
outcomes/metadata out of predictors. The project loader enforces the basic admin
allowlist and excludes detailed indicators. Include new file sources in
`dataset.sources` for hashing. Saved assignments supply names, NUTS3, roles and fold
IDs; duplicate IDs and NUTS3 overlap are errors.

**New model:** subclass `ModelSpec`, implement `build_pipeline(seed=...)` and
`search_space()`, optionally implement `diagnostics()`, and register an instance in
`models/registry.py`. The Python API also accepts an instance directly. Use fresh
sklearn estimators and fit learned preprocessing inside the pipeline. Existing recipes
use `regression_pipeline`, whose column filter learns usable predictors independently
within each fold. Adding a model does not change the runner.

**Metrics:** `EvaluationSpec` selects a shared subset of RMSE, MAE, R² and Spearman.
RMSE is the default tuning objective; MAE or R² can be selected explicitly. Spearman
is diagnostic only because constant predictions make it undefined. To add a metric,
extend `metrics.py` once. Losses are positive in tables/fit metadata; raw sklearn
search results retain negative loss scores, with higher `mean_test_score` better.
R² tuning rejects constant or undersized validation outcomes; diagnostic R² records
undefined values explicitly.

## Statistical contract

`train_model` receives training data only. It tunes on the saved spatial folds and
retains the search's full-training refit. `evaluate_model` only predicts/scores.
Test data never drives preprocessing, tuning, feature selection or early stopping.
Boosting uses `early_stopping=False`; spatial CV selects `max_iter`.

OOF predictions reuse parameters selected on all training folds. Each observation
is excluded from its model's fit, but not from hyperparameter selection. This is
intentional, disclosed in notebooks/manifests, and can produce optimistic OOF scores
and residuals. There is no nested CV. Treat these as diagnostics, including in later
residual-explanation analysis. The baseline uses the fitting partition's mean.

Mean fold metrics and pooled OOF metrics are separate. Selection uses an unweighted
mean over folds; global test metrics weight parishes equally. Regional test metrics
show spatial heterogeneity. These are not population-weighted metrics. Undefined
metrics carry an `undefined_reason`, not a fabricated zero. Cross-target comparisons
must account for different scales/cohorts.

## Read a run

```python
from pipeline.modeling import load_run

results = load_run("/path/to/runs/exact_run_id")
print(results.comparison_table())
oof = results.predictions.query(
    "model == 'ridge' and target == 'epg_heating' and prediction_role == 'train_spatial_cv'"
)
```

Each pair contains its fitted pipeline, selected parameters, candidate results,
usable predictor names, predictions and diagnostics. The manifest contains hashes
of input files, the modeling table and framework source, plus Git commit/dirty status
and package versions. Splits are copied into the run. All models for a target use
the same eligible observations.

`load_run(path, allow_incomplete=True)` reads completed pairs after failure.
Incomplete pair directories are ignored until published in the manifest. Automatic
resume is not implemented; subsequent execution starts a new run. Never combine
independently selected latest files or silently overwrite earlier runs.

## Verification

```powershell
python -m pytest tests/test_modeling.py -q
```

Synthetic tests cover all three estimators, target-specific cohorts, spatial
boundaries, preprocessing, fold-specific baselines, prediction serialization and
partial-run recovery. They do not depend on the private EPVI CSV.
