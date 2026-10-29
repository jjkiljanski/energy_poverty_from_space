# Data Directory

This directory is for small repo-local inputs and modeling snapshots.

Large raw and curated data live outside git in the thesis data folder. See `docs/data_management.md`.

Current files:

- `data_sources_manifest.json`: source registry, verified download lineage,
  unverified planned sources/citations, pipeline mappings, and preserved evidence.
- `parishes.geojson`: freguesia geometries, tracked via Git LFS.
- `parishes_bounding_box.geojson`: bounding box helper for raster processing.
- `all_used_adm_indicators.csv`: small administrative predictor snapshot.
- `all_used_sat_indicators.csv`: small satellite-derived predictor snapshot.
- `adm_data_split.json`: split of administrative columns into basic and detailed groups.
- `freguesias_to_NUTS3.csv`: freguesia LAU2 ID to NUTS3 correspondence used
  for spatial model tuning folds and the fixed NUTS3 test holdout.

The CSV snapshots are derived and can be regenerated from external source data.

The administrative snapshot was manually composed from separate administrative datasets under the external `Adm_data/` folder.

## Maintaining source provenance

Edit `data_sources_manifest.json` to update source information. Unknown scalar
values stay `null` (blank in Excel); empty lists mean no recorded entries.
`verified` refers only to the download lineage within `verification_scope`.
`citation_verified` is separate: references transcribed from the a priori PDF
are candidates, even when the associated download lineage is verified.
The user-verified download notes take precedence wherever they conflict with
the a priori PDF. Conflicting PDF claims are retained only as planning evidence.
`planned_metadata` contains unverified descriptions, coverage and access from
that PDF; it must not be interpreted as metadata for the actual downloads.
`rationale` remains empty unless the choice over alternative sources is documented.
The evidence section retains the original notes and PDF text, including details
that do not safely map to a current input. No external source lookup was used.

Stable source IDs link `raster_folders` to the index manifest and
`administrative_indicators` to the administrative CSV/split. `planned_only`
records have no pipeline mappings. Source-to-column mappings identify subjects;
they do not independently validate indicator derivations. Available optional
sources and benchmark targets are also recorded.

Regenerate the Excel workbook from the repository root:

```powershell
python pipeline/2_index_exploration/index_definition_to_excel.py --manifest pipeline/1_index_construction/indices_manifest.json --out pipeline/1_index_construction/indices_manifest.xlsx
```

The helper automatically reads the registry; `--sources-manifest` can override
its location. Excel includes `Data_Sources`, `Source_Coverage`, `Source_Evidence`,
and `Source_Field_Guide`, plus source IDs in the existing indicator/folder
sheets. `Source_Coverage` highlights new inputs without source mappings.
Close the workbook in Excel before regenerating it.
