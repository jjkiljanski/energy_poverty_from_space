# Data Quality Notes

## Missing Combustion Per-Capita Indicators

Affected output columns in the currently generated CSV:

```text
delta_pm_25_per_capita
delta_bc_per_capita
delta_oc_per_capita
```

Observed state in `freguesia_indices_streaming.csv` generated on 2026-01-26:

```text
all 3092 rows are missing for each of these columns
```

Cause: these indicators use `reduce.method = "mean"` on coarse rasters. The index runner switches coarse rasters into fractional-overlap mode. Before the fix, fractional means were accumulated through the weighted-mean accumulator but finalized through the unweighted mean accumulator, so no values were emitted.

Fix introduced in code: fractional means now finalize from `acc.wsum / acc.w` when `acc.count` is empty.

Required data action: rerun `pipeline/1_index_construction/build_indices.py` to regenerate `freguesia_indices_streaming.csv` and update `data/all_used_sat_indicators.csv`.

## Missing ERA5 Temperature Indicators

Affected output columns:

```text
hdd_18
cdd_25
extreme_heat
extreme_cold
```

Observed state in `freguesia_indices_streaming.csv` generated on 2026-01-26:

```text
321 rows missing per column
210 missing rows are all overseas territories
111 missing rows are mainland freguesias, mostly small/coastal polygons
```

Cause: the existing curated ERA5 rasters already contain `NaN` values for the overseas territories and some coastal/small mainland areas. Spot checks show the raw ERA5 export GeoTIFFs are already missing for most island samples, so this cannot be repaired fully by the local curation step alone. Two export-side issues were identified:

- the Google Earth Engine notebook applied a Portugal land mask before export on the coarse native ERA5 grid;
- the FAO/GAUL Portugal boundary used to derive the export rectangle returned mainland-only bounds in the notebook run (`west=-9.50`, `south=36.96`, `east=-6.18`, `north=42.15`), so the rerun exports still excluded Azores and Madeira.

Fix introduced in code/notebook: `pipeline/0_preprocessing/ERA5L_Portugal_Indices_2010_2012_nativegrid.ipynb` now documents and uses unmasked export over an explicit Portugal-wide rectangle (`west=-31.35`, `south=29.95`, `east=-6.15`, `north=42.25`). Parish polygons should provide the mask later during zonal aggregation.

Required data action:

1. rerun the ERA5 notebook exports;
2. place the new GeoTIFFs under `Sat_data_raw/ERA5-Land_*`;
3. rerun `pipeline/0_preprocessing/curate_raw_tiffs.py` for ERA5 folders;
4. rerun `pipeline/1_index_construction/build_indices.py`;
5. update `data/all_used_sat_indicators.csv` from the regenerated external output.
