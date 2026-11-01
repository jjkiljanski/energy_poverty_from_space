# Data Management

## Policy

Large data should not be committed to git.

Track:

- code
- manifests
- documentation
- small modeling-ready snapshots when useful
- small GeoJSON boundary helpers

Do not track:

- raw satellite downloads
- curated GeoTIFFs
- VRT mosaics
- generated parquet outputs
- generated choropleth images
- large plot-ready GeoJSON exports
- ignored local configuration overrides

## Input and output locations

Repository data is organized by purpose: `data/adm_maps`, `data/indicators`,
`data/docs`, and `data/legacy`. See
[the data directory guide](../data/README.md). Path locations belong in `pipeline/config/`; analytical definitions and
preparation provenance are documented in
`pipeline/definitions/data_sources_manifest.json` rather than encoded in directory names.

External inputs are rooted at `E:\energy_povert_from_space_data\data`.
CAOP shapefiles and prepared EPVI targets are configured there. Existing
raster and administrative source directories remain in OneDrive until moved.

Generated outputs are rooted at `E:\energy_povert_from_space_data\outputs`:
`indices/` for computed tables, `modeling/runs/` for experiments, and `maps/`
for visual exports, and `splits/` for accepted split assignments and the
generated test boundary. External roots and artifacts are configured in
`pipeline/config/external_paths.json`; repository-relative locations are in
`pipeline/config/internal_paths.json`. Retained CSV snapshots under `data/indicators/`
allow modeling without rerunning raster preprocessing.

## Administrative Data

Administrative indicators were downloaded as separate files and manually composed into one predictor table. The working folder for that process is:

```text
E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Adm_data
```

The current composed administrative file is:

```text
...\data\Adm_data\csv\all_used_adm_indicators.csv
```

The repo copy at `data/indicators/administrative.csv` is a modeling snapshot of that manually composed table.

## Join Keys

The key column is `ID`.

Known row counts from current inspection:

```text
all_used_sat_indicators.csv          3092 rows
all_used_adm_indicators.csv          3093 rows
EPVI_results_gouveia_et_al_2019.csv  3092 rows
```

The extra administrative row should be understood before final modeling.

## Generated review documentation

`data/docs/indices_manifest.xlsx` and `data/docs/data_sources.xlsx` are tracked
review documents for index definitions, sources, and prepared artifacts. Their
authoritative inputs remain the JSON manifests. Regenerate and commit them when
those definitions change:

```powershell
python pipeline/input_preparation/indicator_construction/index_definition_to_excel.py
```
