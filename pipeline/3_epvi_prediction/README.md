# EPVI Prediction

`model_training.ipynb` trains one random forest model for each EPVI-related
target using satellite-derived indicators plus only the basic administrative
indicators listed in `data/adm_data_split.json`.

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
- out-of-fold predictions and residuals per freguesia;
- fitted random forest impurity feature importances;
- tuned hyperparameters per target.
