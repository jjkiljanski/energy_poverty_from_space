# Preprocessing

This folder contains scripts/notebooks that convert raw satellite products into curated raster folders under the external data root.

Current external target:

```text
E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_curated
```

## Files

- `curate_raw_tiffs.py`: general raster curation, CRS normalization to Mollweide, NTL postprocessing, and curation manifests.
- `curate_pm_2_5.py`: derives PM2.5 average and heating-season delta rasters for 2010-2012.
- `curate_oc_bc.py`: derives organic carbon and black carbon average/delta rasters for 2010-2012.
- `ERA5L_Portugal_Indices_2010_2012_nativegrid.ipynb`: Google Earth Engine workflow for ERA5-Land temperature indicators.

## Notes

These scripts are not yet a polished command-line pipeline. They contain local paths and should be run carefully after checking their constants.

Outputs are large and should stay outside git.
