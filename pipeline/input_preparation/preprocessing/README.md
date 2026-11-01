# Preprocessing

This folder contains scripts/notebooks that convert raw satellite products into curated raster folders under the external data root.

Current external target:

```text
E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_curated
```

## Files

- `build_parishes.py`: merges extracted CAOP 2016 shapefiles using the shared geometry helpers; adds the `ID` field required by raster aggregation.
- `curate_raw_tiffs.py`: general raster curation, CRS normalization to Mollweide, NTL postprocessing, and curation manifests.
- `curate_pm_2_5.py`: derives PM2.5 average and heating-season delta rasters for 2010-2012.
- `curate_oc_bc.py`: derives organic carbon and black carbon average/delta rasters for 2010-2012.
- `ERA5L_Portugal_Indices_2010_2012_nativegrid.ipynb`: Google Colab-oriented Google Earth Engine workflow for ERA5-Land temperature indicators.

## Notes

### Rebuilding parish boundaries

Download the five CAOP 2016 administrative polygon layers (mainland
PT-TM06/ETRS89, Madeira, and the three Azores groups) from
[DGT](https://www.dgterritorio.gov.pt/atividades/cartografia/cartografia-tematica/caop).
Extract them with their companion files into a dedicated directory containing
only these layers. From the repository root, in the project environment, run:

```powershell
python pipeline/input_preparation/preprocessing/build_parishes.py
```

The builder resolves `caop_shapefiles` from `pipeline/config/external_paths.json`
and `parish_geometry` from `pipeline/config/internal_paths.json`, currently
`data/adm_maps/parishes.geojson`. Use `--map-dir`, `--config`, or `--out`
for explicit overrides.

The wrapper calls `load_mainland_and_islands` and
`dump_plot_ready_geojson_if_missing` from `pipeline/utils/geometry.py`.
Those helpers repair geometries, dissolve pieces sharing a `dicofre`, align
regional CRSs, and export longitude/latitude GeoJSON. The result contains only
`dicofre`, `ID`, and geometry; administrative indicators remain in their CSV.
Codes are strings and may contain letters. Existing output paths are refused.

Keep source archives and record download URLs/dates and checksums alongside
the build record.

These scripts are not yet a polished command-line pipeline. They contain local paths and should be run carefully after checking their constants.

The ERA5-Land notebook is a cloud workflow, not a locally runnable step in the shared Conda environment. It installs the Earth Engine client and `geemap`, uses interactive Earth Engine authentication, and submits exports to Google Drive. Run it in Google Colab with access to the configured Earth Engine project and Google Drive. Local Jupyter execution may be possible after adapting authentication and export destinations, but is not currently supported or documented as a project workflow.

Outputs are large and should stay outside git.
