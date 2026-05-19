# Index Construction

This folder contains the current core of the satellite-derived predictor pipeline.

## Main Files

- `indices_manifest.json`: machine-readable definition of all freguesia-level satellite indicators.
- `indices_manifest.xlsx`: human-readable export of the manifest.
- `build_indices.py`: streaming raster-to-freguesia aggregation runner.
- `helpers.py`: raster mosaic, alignment, windowing, and VRT helpers.
- `distribution_compute_helpers.py`: histogram-based quantile and distribution helpers.

## Current Output

The manifest currently writes generated outputs to:

```text
E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\outputs\indices
```

Expected outputs:

```text
freguesia_indices_streaming.csv
freguesia_indices_streaming.parquet
freguesia_indices_streaming.partial.csv
index_build_state.json
```

The CSV has also been copied into `data/all_used_sat_indicators.csv` as a small modeling snapshot.

## Running

The code currently uses direct imports such as `import helpers`, so it is safest to run from this directory:

```powershell
cd pipeline\1_index_construction
python build_indices.py
```

This can be slow and requires the external curated raster folders to exist.

`rasterio` is required. GDAL Python bindings (`osgeo.gdal`) are optional: when
they are installed, the runner uses `gdal.BuildVRT`; otherwise it writes small
VRT XML mosaic files directly and still avoids loading full rasters into memory.

The runner writes `index_build_state.json` after every index. This state file is
both a cache manifest and an audit trail. It stores the latest run's command,
working directory, tile size, source and resolved manifest hashes, local path
config, Git commit/branch/dirty status, Python/package versions, final output
file stats, and per-index timings. Each index entry also stores the index
definition hash, the curated TIFF input file stats used by that index, and the
admin-units file stats. On later runs, unchanged indexes are read from
`freguesia_indices_streaming.partial.csv` or the final CSV instead of being
recomputed. The partial CSV is refreshed after every index, so a failed run can
usually resume without repeating already completed indexes.

## Implementation Notes

The runner processes rasters in windows to avoid loading full Portugal rasters into memory. It builds folder-level VRT mosaics, aligns raster inputs per index, rasterizes freguesia labels per window, and aggregates into `ID`-level outputs.
