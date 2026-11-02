# Path configuration

The two registries give artifacts stable logical names. Update a registry when
an artifact moves; consumers keep using the same name.

- `internal_paths.json`: repository-relative locations for prepared inputs, definitions, and dataset documentation.
- `external_paths.json`: absolute input/output roots and artifact locations relative to those roots. Change one root entry to relocate its artifacts together.

```python
from pipeline.utils.paths import load_paths

paths = load_paths()
geometry = paths.internal("parish_geometry")
targets = paths.external("epvi_targets")
assignments = paths.external("split_assignments")
```

Internal paths always resolve relative to the repository root. External artifact
paths resolve under their named external root. Unknown names, missing roots,
and paths escaping their root raise errors. Resolution does not create folders
or require files to exist.

For alternative registries:

```python
paths = load_paths(
    "my_external_paths.json",
    internal_config_path="my_internal_paths.json",
)
```

The modeling CLI and workbook exporter accept `--paths-config` for the external
registry and `--internal-paths-config` for the internal registry. The parish
builder uses `--config` for the external registry and the same internal override.

Analytical definitions live in `pipeline/definitions/`. The generated workbooks
are resolved as `paths.internal("indices_manifest_workbook")` and
`paths.internal("data_sources_workbook")`, currently
`data/docs/indices_manifest.xlsx` and `data/docs/data_sources.xlsx`.
Regenerate both with:

```powershell
python pipeline/input_preparation/indicator_construction/index_definition_to_excel.py
```

Original, raw, and curated satellite raster folders use the external data root.
Administrative source files, the legacy merged map, and the EPVI source remain
under the configured legacy OneDrive root. Prepared private targets and CAOP
downloads use the external data root. New model runs, split artifacts, and map
exports use the external output root.
