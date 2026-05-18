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
```

The CSV has also been copied into `data/all_used_sat_indicators.csv` as a small modeling snapshot.

## Running

The code currently uses direct imports such as `import helpers`, so it is safest to run from this directory:

```powershell
cd pipeline\1_index_construction
python build_indices.py
```

This can be slow and requires the external curated raster folders to exist.

## Implementation Notes

The runner processes rasters in windows to avoid loading full Portugal rasters into memory. It builds folder-level VRT mosaics, aligns raster inputs per index, rasterizes freguesia labels per window, and aggregates into `ID`-level outputs.
