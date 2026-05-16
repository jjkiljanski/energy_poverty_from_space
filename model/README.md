# Modeling

The modeling work is currently at the baseline-sketch stage.

`model_training.ipynb` should be read as a first draft that checks whether the assembled predictors can be joined and used for simple train/test experiments. It is not yet a finalized modeling workflow.

Immediate modeling cleanup tasks:

1. Load paths from a config file instead of hard-coded local paths.
2. Validate joins across EPVI, satellite, administrative, and geometry tables.
3. Compare clear predictor sets:
   - remote sensing only
   - basic administrative only
   - remote sensing plus basic administrative
   - optional extended model with detailed administrative or engineered data
4. Define spatial validation strategy explicitly.
5. Save model metrics and residuals reproducibly.
6. Use detailed administrative variables for residual interpretation, not just prediction.
