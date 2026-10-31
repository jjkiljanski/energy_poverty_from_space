# Current State

Last reviewed: 2026-05-16. Data layout and paths updated: 2026-10-31.

## Big Picture

The project is at the transition between data preparation and modeling.

Data preparation is relatively advanced: satellite-derived indicators have been defined and computed at freguesia level, administrative predictors have been assembled, and benchmark EPVI/AIAM/EPG targets are available. Modeling has only just started and should be considered exploratory.

## What Exists

### External input and output locations

Newly organized inputs live under `E:\energy_povert_from_space_data\data`:
`CAOP_2016_shapefiles/` and `epvi_targets.csv`. Generated outputs use
`E:\energy_povert_from_space_data\outputs`, with `indices/`, `modeling/`,
and `maps/` subdirectories. Existing raw and curated raster directories and
original administrative downloads remain at their OneDrive paths configured
in `pipeline/config/external_paths.json`.

Important external files:

```text
EPVI_results_gouveia_et_al_2019.csv
EPVI results Gouveia et al. (2019).xlsx
Data downloads.txt
```

### Repo Data Snapshots

The repo currently has small modeling-ready snapshots:

```text
data/indicators/administrative.csv
data/indicators/satellite.csv
pipeline/definitions/adm_data_split.json
data/adm_maps/parishes.geojson
data/adm_maps/parishes_bounding_box.geojson
```

The satellite and EPVI tables have 3092 freguesia rows. The administrative table has 3093 rows, so joins should be checked explicitly during modeling.

The administrative predictor table was assembled manually from separate downloaded administrative datasets. The working folder for that process is:

```text
...\data\Adm_data
```

The composed CSV currently appears as:

```text
...\data\Adm_data\csv\all_used_adm_indicators.csv
data/indicators/administrative.csv
```

### Satellite Indicator Construction

The main new work is in:

```text
pipeline/input_preparation/indicator_construction/
```

`indices_manifest.json` defines 35 indicators. They cover:

- building stock and morphology
- population and density
- building age composition
- night-time-light proxies
- PM2.5/BC/OC heating-season deltas
- heating/cooling degree and extreme temperature exposure

`build_indices.py` computes these indicators by streaming over raster windows, rasterizing freguesia boundaries, and aggregating values by `ID`.

Generated outputs exist at:

```text
...\data\outputs\indices\freguesia_indices_streaming.csv
...\data\outputs\indices\freguesia_indices_streaming.parquet
```

The CSV has also been copied into the repo as `data/indicators/satellite.csv`.

### Modeling

`pipeline/modeling/1_model_training.ipynb` is an initial sketch.

Current approach in that notebook:

- load administrative predictors
- load satellite predictors
- load EPVI benchmark targets
- use the NUTS3 mapping in `data/adm_maps/freguesias_to_NUTS3.csv` for spatial splits
- keep the representative fixed spatial test set selected in
  `pipeline/modeling/0_spatial_test_set_selection.ipynb`
  (`PT112`, `PT16B`, `PT16I`, and `PT16J` at present)
- train/evaluate several candidate regressors for each target

This is not yet a final modeling design. It still needs explicit data validation, better split strategy discussion, baseline comparisons, residual analysis, and reproducible outputs.

## Known Problems

- Local Windows paths are centralized in `pipeline/config/external_paths.json`; this is intentional for the current local thesis workflow.
- No pinned Python environment exists yet.
- Generated and source data are split between repo snapshots and external OneDrive folders.
- Some scripts assume they are run from a specific working directory.
- `pipeline/input_preparation/exploration/map_index.py` has been adapted to the generated satellite indicator output and no longer maps the old administrative CSV by default.

## Recommended Next Work

1. Add a reproducible Python environment.
2. Validate all joins between EPVI, administrative, satellite, and geometry IDs.
3. Convert the modeling notebook into a cleaner baseline notebook or script.
4. Add residual analysis against detailed administrative predictors.
5. Decide which small derived data snapshots should stay in git.
