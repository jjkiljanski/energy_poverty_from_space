# Pipeline

The pipeline separates preparation of the complete modeling dataset from modeling
and analysis. Shared configuration, definitions, and cross-stage helpers remain at
the pipeline root.

```text
pipeline/
  config/                         Internal and external path registries
  definitions/                    Data-source, index, and administrative definitions
  input_preparation/
    preprocessing/                Raw and curated source preparation
    indicator_construction/       Raster-to-freguesia indicators and Excel export
    exploration/                  Prepared-indicator maps and inspection
  modeling/
    api/                          Reusable modeling code and CLI
    0_spatial_test_set_selection.ipynb
    1_model_training.ipynb
    2_model_performance.ipynb
    3_model_prediction_exploration.ipynb
    utils/                        Modeling-specific helper functions
  results_interpretation/         Prediction and residual analysis
  utils/                          Shared helpers used across stages
```

Prepared modeling inputs live under `../data`; generated intermediate and analysis
products are written to the configured external outputs directory.
