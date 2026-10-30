# EPVI Prediction

`3_1_model_training.ipynb` orchestrates separate ridge, random forest and histogram
gradient boosting models for each configured regression target. Shared code lives
in [`../modeling/`](../modeling/README.md). Five default targets produce 15 tuned
models, plus one training-mean reference baseline per target.

Predictors are satellite indices plus only the **basic** administrative indicators
in `data/adm_data_split.json`. Detailed indicators remain reserved for residual
analysis. The private target file is resolved from the local `epvi_csv` path.

`3_0_spatial_test_set_selection.ipynb` creates
`data/spatial_split_assignments.csv`. Training consumes those saved assignments;
it does not select new test regions or rebuild folds. Mainland-only scope is the
default; setting `EXCLUDE_OVERSEAS_TERRITORIES=False` retains assigned overseas
rows. Each target removes only its own missing outcomes, preserving fold labels.

## Modeling design

- Tune on training-region spatial CV only, using mean fold RMSE by default.
  This explicitly replaces the historical mean-fold R² tuning objective.
- Learn column filtering, imputation and ridge scaling within each fitting fold.
- Generate spatial OOF predictions with hyperparameters selected using all
  training folds. These are **tuning diagnostics**, not independent evaluation of
  hyperparameter selection. Scores can be optimistic. Nested CV is deliberately
  omitted to keep computation and the workflow manageable.
- Reuse the search's full-training refit for fixed-test prediction. Do not tune
  from test results or refit on test data.
- Report RMSE, MAE, R² and Spearman for every model on the original target scale.
  Keep pooled OOF metrics distinct from mean fold metrics. Compare within targets.
- The training-mean baseline learns a separate mean inside each OOF fitting fold.

The fixed holdout was previously inspected and replaced; its history is documented
in `docs/model_tuning_history.md`. It should not be described as a never-inspected
confirmatory test.

## Outputs and performance analysis

Every experiment writes one new directory outside git:

```text
<external_data_root>/outputs/epvi_prediction/runs/<run_id>/
    manifest.json
    split_assignments.csv
    <target_key>/<model_name>/
        pipeline.joblib
        fit.json
        search_results.csv
        metrics.csv
        predictions.csv
        folds.csv
        fold_metrics.csv
        regional_metrics.csv
        diagnostics.csv
```

The manifest records configuration, exact search spaces, input and code hashes,
package versions, source coverage and the OOF limitation. Each model-target pair is
saved as it completes. Failed/interrupted experiments retain completed pairs.
Pipeline serialization includes fitted preprocessing. Load joblib models only from
trusted runs; table inspection does not deserialize them.

`3_2_model_performance.ipynb` reads an explicit run ID, compares all models with the
baseline, and inspects per-fold/regional metrics, residuals and model-specific
diagnostics. Ridge coefficients and RF impurity importances have different meanings.
Boosting has no built-in exported feature importance; test permutation importance
is not automatically run.

Legacy RF outputs remain under `outputs/epvi_prediction/random_forest/`.
`3_3_epg_diagnostics.ipynb` and `linear_model_diagnostics.ipynb` are historical
analyses, not consumers of the new run schema. For new EPG residuals, use
`load_run(run_dir).predictions` and explicitly select the target, model and
`prediction_role` (`train_spatial_cv`, `train_in_sample`, or `fixed_test`).
