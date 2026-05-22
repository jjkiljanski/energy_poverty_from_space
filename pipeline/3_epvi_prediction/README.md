# EPVI Prediction

`model_training.ipynb` trains one random forest model for each EPVI-related
target using satellite-derived indicators plus only the basic administrative
indicators listed in `data/adm_data_split.json`.

Model selection is spatial. `spatial_test_set_selection.ipynb` documents why
the earlier `PT16E` + `PT16J` holdout was replaced, ranks whole-NUTS3
candidate holdouts close to 20% of the modeling rows, and freezes `PT112`,
`PT16B`, `PT16I`, and `PT16J` as the current fixed test regions. The training
notebook joins freguesias to NUTS3 regions via
`data/freguesias_to_NUTS3.csv`, removes those fixed test regions, then runs
region-held-out NUTS3 cross-validation on the remaining training regions.

`model_performance.ipynb` reads the latest saved random forest outputs and
presents the main metrics, observed-vs-predicted plots, residual summaries, and
feature-importance diagnostics.

`epg_diagnostics.ipynb` focuses on the weak spatial-transfer results for
`EPG heating` and `EPG cooling`. It compares the saved random-forest
predictions with train-mean baselines, summarizes NUTS3-level shifts and
errors, maps residuals, and relates residuals to the allowed predictors.

`utils.py` contains shared loading, ID normalization, modeling-table assembly,
and metric helpers used by the notebooks.

The privately shared EPVI file is loaded from the local path config key
`epvi_csv`; it should not be committed to the repo.

Detailed administrative indicators are intentionally excluded from this
prediction notebook. They are reserved for the next step: explaining the
out-of-fold residuals.

The notebook writes timestamped outputs outside git under:

```text
<external_data_root>/outputs/epvi_prediction/random_forest/
```

Main outputs:

- model metrics per EPVI target;
- spatial-CV training predictions and fixed-test predictions/residuals per
  freguesia;
- fitted random forest impurity feature importances;
- tuned hyperparameters per target;
- the NUTS3 regions assigned to each training-region validation fold.
