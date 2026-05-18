# Pipeline

This directory contains the runnable project workflow. Numbered folders follow the intended order of work.

```text
config/                    Local path configuration
utils/                     Shared importable helpers
0_preprocessing/           Raw/curated satellite data preparation
1_index_construction/      Raster-to-freguesia indicator construction
2_index_exploration/       Indicator inspection and choropleths
3_epvi_prediction/         Baseline EPVI prediction modeling
4_explore_pred_residuals/  Residual interpretation against richer variables
```

Large data stay outside git. The small repo-local snapshots used for modeling are in `../data`.
