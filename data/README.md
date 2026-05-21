# Data Directory

This directory is for small repo-local inputs and modeling snapshots.

Large raw and curated data live outside git in the thesis data folder. See `docs/data_management.md`.

Current files:

- `parishes.geojson`: freguesia geometries, tracked via Git LFS.
- `parishes_bounding_box.geojson`: bounding box helper for raster processing.
- `all_used_adm_indicators.csv`: small administrative predictor snapshot.
- `all_used_sat_indicators.csv`: small satellite-derived predictor snapshot.
- `adm_data_split.json`: split of administrative columns into basic and detailed groups.
- `freguesias_to_NUTS3.csv`: freguesia LAU2 ID to NUTS3 correspondence used
  for spatial model tuning folds and the fixed NUTS3 test holdout.

The CSV snapshots are derived and can be regenerated from external source data.

The administrative snapshot was manually composed from separate administrative datasets under the external `Adm_data/` folder.
