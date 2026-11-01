# Analytical input definitions

These files describe what the inputs mean and how they are obtained or selected:

- `data_sources_manifest.json`: upstream sources grouped as `used` and `planned_only`. Each record keeps concise review fields; other source metadata is retained in its `notes` string.
- `indices_manifest.json`: satellite indicator construction rules.
- `adm_data_split.json`: basic and detailed administrative predictor groups.

Preparation and modeling share these definitions. Machine locations belong in
`pipeline/config/`; the path resolver exposes these files as `source_manifest`,
`index_manifest`, and `admin_groups` respectively.

After editing the definitions, regenerate the two review workbooks:

```powershell
python pipeline/input_preparation/indicator_construction/index_definition_to_excel.py
```

The workbooks are retained with the modeling dataset at
[`data/docs/indices_manifest.xlsx`](../../data/docs/indices_manifest.xlsx) and
[`data/docs/data_sources.xlsx`](../../data/docs/data_sources.xlsx).
