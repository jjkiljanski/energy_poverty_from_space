# Current State

Last reviewed: 2026-05-16.

## Big Picture

The project is at the transition between data preparation and modeling.

Data preparation is relatively advanced: satellite-derived indicators have been defined and computed at freguesia level, administrative predictors have been assembled, and benchmark EPVI/AIAM/EPG targets are available. Modeling has only just started and should be considered exploratory.

## What Exists

### External Data Root

The active local data root is:

```text
E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data
```

Important subfolders:

```text
Adm_data/             raw administrative downloads and manual composition work
Map/                  CAOP shapefiles for mainland and islands
outputs/indices/      generated satellite indicator outputs and choropleths
Sat_data_curated/     curated satellite rasters used by index construction
Sat_data_orig/        original downloaded satellite/NetCDF products
Sat_data_raw/         raw raster products before curation
```

Important external files:

```text
EPVI_results_gouveia_et_al_2019.csv
EPVI results Gouveia et al. (2019).xlsx
Data downloads.txt
```

### Repo Data Snapshots

The repo currently has small modeling-ready snapshots:

```text
data/all_used_adm_indicators.csv
data/all_used_sat_indicators.csv
data/adm_data_split.json
data/parishes.geojson
data/parishes_bounding_box.geojson
```

The satellite and EPVI tables have 3092 freguesia rows. The administrative table has 3093 rows, so joins should be checked explicitly during modeling.

The administrative predictor table was assembled manually from separate downloaded administrative datasets. The working folder for that process is:

```text
...\data\Adm_data
```

The composed CSV currently appears as:

```text
...\data\Adm_data\csv\all_used_adm_indicators.csv
data/all_used_adm_indicators.csv
```

### Satellite Indicator Construction

The main new work is in:

```text
pipeline/1_index_construction/
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

The CSV has also been copied into the repo as `data/all_used_sat_indicators.csv`.

### Modeling

`pipeline/3_epvi_prediction/model_training.ipynb` is an initial sketch.

Current approach in that notebook:

- load administrative predictors
- load satellite predictors
- load EPVI benchmark targets
- use a fixed spatial holdout where IDs starting with `192` or `196` form the test set
- train/evaluate several candidate regressors for each target

This is not yet a final modeling design. It still needs explicit data validation, better split strategy discussion, baseline comparisons, residual analysis, and reproducible outputs.

## Known Problems

- Local Windows paths are centralized in `pipeline/config/paths.example.json`; this is intentional for the current local thesis workflow.
- No pinned Python environment exists yet.
- Generated and source data are split between repo snapshots and external OneDrive folders.
- Some scripts assume they are run from a specific working directory.
- `pipeline/2_index_exploration/map_index.py` has been adapted to the generated satellite indicator output and no longer maps the old administrative CSV by default.

## Recommended Next Work

1. Add a reproducible Python environment.
2. Validate all joins between EPVI, administrative, satellite, and geometry IDs.
3. Convert the modeling notebook into a cleaner baseline notebook or script.
4. Add residual analysis against detailed administrative predictors.
5. Decide which small derived data snapshots should stay in git.
