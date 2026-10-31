# Modeling helper utilities

This package contains modeling-specific helper code used by the numbered
modeling notebooks. It supports work around the reusable API, especially source
table assembly, spatial split selection and validation, and geographic summaries.

## Modules

- `load_data.py` normalizes and combines satellite, administrative, and EPVI
tables for the split-selection notebook.
- `nuts3.py` loads parish-to-NUTS3 mappings and parish geometry, and supports
spatial split selection and maps.
- `dataset_split_selection/` scores candidate test regions, selects training
folds, builds row-level assignment tables, assigns overseas municipalities,
and writes the selected test boundary.
- `spatial_splits.py` loads and summarizes saved split assignments and plots them.
- `summaries.py` presents geographic scope and fold summaries in notebooks.

These utilities support notebook steps around split inputs and inspection. The
reusable modeling contract for training, freezing, evaluation, and saved-run
inspection lives in [`../api/`](../api/README.md). Helpers used across preparation
and modeling belong in [`../../utils/`](../../utils/README.md).
