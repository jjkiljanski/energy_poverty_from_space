# Energy Poverty from Space

This repository supports a master's thesis project on whether energy poverty in Portugal can be approximated from satellite-derived indicators and basic administrative data.

**Start here: [Data sources and indicator definitions (Excel)](data/docs/indices_manifest.xlsx).**
The workbook provides a reviewable overview; its JSON sources remain authoritative.

The repository separates preparation of the complete modeling dataset from modeling and analysis. The prepared inputs live under `data/`; modeling workflows consume them and write generated results to the configured outputs directory.

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
data/                              Prepared modeling inputs and documentation
docs/                             Project and methodology documentation
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

Large raw and curated data are intentionally outside git. Their locations are configured in `pipeline/config/external_paths.json`; the `data` and `outputs` roots can be changed there without editing pipeline code.

## Current Data Products

The most important current files are:

- `data/indicators/satellite.csv`: freguesia-level satellite-derived predictors generated from `pipeline/input_preparation/indicator_construction/`.
- `data/adm_maps/parishes.geojson`: CAOP 2016 boundaries used for raster aggregation.
- `data/indicators/administrative.csv`: freguesia-level administrative predictors.
- `pipeline/definitions/adm_data_split.json`: current split of administrative predictors into `basic` and `detailed`.
- External `EPVI_results_gouveia_et_al_2019.csv`: benchmark targets from Gouveia et al.

The generated satellite output also exists externally at:

```text
...\data\outputs\indices\freguesia_indices_streaming.csv
...\data\outputs\indices\freguesia_indices_streaming.parquet
```

## Pipeline Summary

1. Curate raw satellite products into Mollweide GeoTIFF folders using scripts in `pipeline/input_preparation/preprocessing/`.
2. Define freguesia-level satellite indicators in `pipeline/definitions/indices_manifest.json`.
3. Run `pipeline/input_preparation/indicator_construction/build_indices.py` to aggregate raster data to freguesia-level indicators.
4. Combine satellite predictors, administrative predictors, and EPVI benchmark targets in `pipeline/modeling/`.
5. Compare models and inspect residuals against richer administrative variables.

See [docs/pipeline.md](docs/pipeline.md) and [docs/current_state.md](docs/current_state.md) for the current practical state.
See [docs/data_provenance.md](docs/data_provenance.md) for a reconstruction of how the current data files were generated and which repo files are snapshots of external pipeline outputs.
See [docs/data_quality_notes.md](docs/data_quality_notes.md) for known issues in the current generated indicator CSV.

## Environment

Create the shared Conda environment from the repository root:

```powershell
conda env create --file environment.yml
conda activate energy-poverty-from-space
python -m ipykernel install --user --name energy-poverty-from-space --display-name "Python (energy-poverty-from-space)"
```

Start JupyterLab from the activated environment and select the `Python (energy-poverty-from-space)` kernel for project notebooks:

```powershell
jupyter lab
```

`environment.yml` lists the project's direct dependencies. Conda also installs the transitive packages required by those dependencies. The file currently constrains Python to 3.11 but does not lock every package build; a Conda lockfile can be added after confirming the environment solves on the platforms the project needs. GDAL Python bindings (`osgeo.gdal`) remain optional for raster mosaicking.

The ERA5-Land Earth Engine notebook under `pipeline/input_preparation/preprocessing/` is an exception to the local workflow: it is designed to run in Google Colab, authenticates with Google Earth Engine, and exports to Google Drive. See [the preprocessing notes](pipeline/input_preparation/preprocessing/README.md); creating the local Conda environment does not make that cloud workflow run locally.

## Git Hygiene

Do not commit the large external data directory. The repo should track code, manifests, small modeling snapshots, and documentation. Generated rasters, VRTs, parquet files, plot exports, caches, and local configuration overrides are ignored.

Locations are centralized in `pipeline/config/external_paths.json` and
`pipeline/config/internal_paths.json`. Use `load_paths().external(name)` or
`load_paths().internal(name)` to resolve an artifact by its logical name.
See [path configuration](pipeline/config/README.md).
