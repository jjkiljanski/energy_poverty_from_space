# Energy Poverty from Space

This repository supports a master's thesis project on whether energy poverty in Portugal can be approximated from satellite-derived indicators and basic administrative data.

The current project state is early modeling: most work so far has gone into building freguesia-level predictor tables from raster and administrative sources. The first modeling notebook exists, but should be treated as a sketch rather than a final analysis pipeline.

## Research Goal

The thesis benchmarks remote-sensing and administrative predictors against the Portuguese Energy Poverty Vulnerability Index data from Gouveia et al. The main outcomes are:

- `EPG heating`
- `EPG cooling`
- `AIAM`
- `EPVI heating`
- `EPVI cooling`

The working spatial unit is the Portuguese freguesia, keyed by `ID`.

## Repository Layout

```text
data/                 Small project inputs/snapshots used by scripts and modeling
data_exploration/     Mapping and inspection scripts
docs/                 Project documentation and current-state notes
index_construction/   Manifest-driven construction of satellite-derived indicators
model/                Early modeling notebooks and notes
preprocessing/        Scripts/notebooks for curating raw satellite products
utils/                Raster reprojection/alignment utilities
config/               Example local path configuration
```

Large raw and curated data are intentionally outside git. In the current local setup, they live under:

```text
E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data
```

## Current Data Products

The most important current files are:

- `data/all_used_sat_indicators.csv`: freguesia-level satellite-derived predictors generated from `index_construction/`.
- `data/all_used_adm_indicators.csv`: freguesia-level administrative predictors.
- `data/adm_data_split.json`: current split of administrative predictors into `basic` and `detailed`.
- External `EPVI_results_gouveia_et_al_2019.csv`: benchmark targets from Gouveia et al.

The generated satellite output also exists externally at:

```text
...\data\outputs\indices\freguesia_indices_streaming.csv
...\data\outputs\indices\freguesia_indices_streaming.parquet
```

## Pipeline Summary

1. Curate raw satellite products into Mollweide GeoTIFF folders using scripts in `preprocessing/`.
2. Define freguesia-level satellite indicators in `index_construction/indices_manifest.json`.
3. Run `index_construction/build_indices.py` to aggregate raster data to freguesia-level indicators.
4. Combine satellite predictors, administrative predictors, and EPVI benchmark targets in `model/`.
5. Compare models and inspect residuals against richer administrative variables.

See [docs/pipeline.md](docs/pipeline.md) and [docs/current_state.md](docs/current_state.md) for the current practical state.

## Environment

No pinned environment exists yet. The code currently expects a geospatial Python stack, roughly:

```text
numpy
pandas
geopandas
rasterio
shapely
xarray
rioxarray
scikit-learn
matplotlib
openpyxl
netCDF4
```

Some raster mosaic functionality also benefits from GDAL Python bindings (`osgeo.gdal`).

## Git Hygiene

Do not commit the large external data directory. The repo should track code, manifests, small modeling snapshots, and documentation. Generated rasters, VRTs, parquet files, plot exports, caches, and local path configuration are ignored.

The repo currently contains some hard-coded local paths. The intended cleanup direction is to gradually replace them with a local config based on `config/paths.example.json`.
