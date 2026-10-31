# Reviewing the modeling tests

Start with the guarantee you want to verify, then open its test file. The directory
mirrors `pipeline/modeling/api`; filenames add the usual `test_` prefix. The CLI module
`__main__.py` therefore maps to `test___main__.py`. Import-only modules need no empty
test files. Nothing here requires private data or performs production-sized searches.

| Production module / test file | Evidence |
| --- | --- |
| `data.py` / `test_data.py` | Explicit source columns; invalid joins/schema fail; target-specific missingness; positional folds cover each eligible parish once and separate regions |
| `targets.py` / `test_targets.py` | Valid source columns and safe, unique target identifiers |
| `metrics.py` / `test_metrics.py` | Hand-calculated errors/correlation; invalid arrays fail; undefined scores have reasons |
| `training.py` / `test_training.py` | Independent sklearn tuning/OOF reference for all three objectives; test-outcome isolation; adding a sparse target preserves earlier results; fold-specific mean baseline; test evaluation never refits |
| `artifacts.py` / `test_artifacts.py` | Failed training/writes preserve completed pairs and hide partial pairs; portable JSON; failed replacement preserves the previous file |
| `__main__.py` / `test___main__.py` | Dry-run validates without fitting/writing; misspelled configuration fails |
| `models/base.py` / `models/test_base.py` | Column filtering and learned imputation/scaling use fitting rows only |
| `models/ridge.py` / `models/test_ridge.py` | Predictions and labeled coefficients match a closed-form calculation |
| `models/random_forest.py` / `models/test_random_forest.py` | Predictions and labeled importances match a directly fitted sklearn forest |
| `splits.py` / `test_splits.py` | Named outer holdouts; five-fold nesting fits three regions and validates one inside each four-fold selection |
| `procedure.py` / `test_procedure.py` | Freeze integrity; changed inputs/artifacts rejected; outer evaluation does not refit; partial outer predictions cannot masquerade as complete OOF |
| `models/registry.py` / `models/test_registry.py` | Recipes create independent, cloneable pipelines; search overrides cannot mutate later recipes |

`conftest.py` contains three shared fixtures: 48 synthetic parishes with three
training regions and a separate test region, small search spaces, and a five-fold
extension for the nested reference calculation. Tests show
their own execution and assertions; there is no custom runner, test inheritance,
snapshot framework or shared assertion library.

The end-to-end test in `test_training.py` also checks serialization, leading-zero
IDs and serial/parallel execution. It stays with the runner because it crosses
module boundaries. The old mixed `tests/test_modeling.py` has been split: EPVI
parsing checks now live under `tests/pipeline/utils`, and notebook checks under
`tests/pipeline/modeling/test_notebooks.py`.

Run in the activated project Conda environment, from the repository root:

```powershell
python -m pytest tests/pipeline/modeling -q
python -m pytest tests/pipeline/modeling/test_training.py -v
python -m pytest -q
```

The nested reference test independently tunes within each outer training selection,
checks inner diagnostic predictions, then checks outer and final predictions against
explicit sklearn calculations. Inner diagnostics reuse selected parameters and may
be optimistic; only outer predictions exclude those outcomes from tuning.
Tests establish implementation behavior; they do not establish that
the chosen metrics, spatial holdout or predictor set are scientifically optimal.
