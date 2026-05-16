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
- local path config

## Canonical External Data Root

Current local root:

```text
E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data
```

The repo is currently coupled to this path in several scripts. Over time, code should read paths from `config/paths.local.json`, copied from `config/paths.example.json`.

## Small Repo Snapshots

The current small CSV/JSON snapshots in `data/` are useful because they let modeling and review start without re-running raster processing:

```text
data/all_used_adm_indicators.csv
data/all_used_sat_indicators.csv
data/adm_data_split.json
```

These are derived data, not raw data. If they become too large or are regenerated often, move them out of git and document the external location instead.

## Administrative Data

Administrative indicators were downloaded as separate files and manually composed into one predictor table. The working folder for that process is:

```text
E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Adm_data
```

The current composed administrative file is:

```text
...\data\Adm_data\csv\all_used_adm_indicators.csv
```

The repo copy at `data/all_used_adm_indicators.csv` is a modeling snapshot of that manually composed table.

## Join Keys

The key column is `ID`.

Known row counts from current inspection:

```text
all_used_sat_indicators.csv          3092 rows
all_used_adm_indicators.csv          3093 rows
EPVI_results_gouveia_et_al_2019.csv  3092 rows
```

The extra administrative row should be understood before final modeling.
