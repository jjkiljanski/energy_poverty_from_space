# Repository data

This folder contains the prepared inputs retained for modeling. Preparation
method and lineage are recorded in
[`pipeline/definitions/data_sources_manifest.json`](../pipeline/definitions/data_sources_manifest.json) and
[the provenance notes](../docs/data_provenance.md).

```text
data/
    docs/
        indices_manifest.xlsx
    adm_maps/
        parishes.geojson
        parishes_bounding_box.geojson
        freguesias_to_NUTS3.csv
    indicators/
        administrative.csv
        satellite.csv
    legacy/
        parishes_with_indicators.geojson
```

- `adm_maps/`: CAOP aggregation geometry, the manually prepared processing extent, and the freguesia-to-NUTS3 correspondence.
- `indicators/`: manually assembled administrative predictors and a retained generated satellite predictor snapshot.
- `docs/`: generated documentation explaining the dataset, indicator definitions, and sources.
- `legacy/`: the historical GeoJSON containing boundaries and administrative indicators, retained for comparison. Pipeline consumers use `adm_maps/parishes.geojson`.

Predictor group definitions live in `pipeline/definitions/adm_data_split.json`;
index definitions live in `pipeline/definitions/indices_manifest.json`.
Large downloaded inputs and rasters stay external. Generated index tables,
modeling runs, and map exports go under the configured `outputs_root`, in
`indices/`, `modeling/`, and `maps/` respectively. Accepted split assignments
and their generated test boundary live under `outputs_root/splits/`; the split
selection notebook writes them there and model training reads the assignments. The satellite CSV here is an
explicitly retained snapshot, not an automatic destination for new index runs.

## Maintaining source provenance

Edit `pipeline/definitions/data_sources_manifest.json` to update source information. Unknown scalar
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

The generated review workbook is tracked at
[`docs/indices_manifest.xlsx`](docs/indices_manifest.xlsx). Regenerate and
commit it when the JSON definitions change. Run from the repository root:

```powershell
python pipeline/input_preparation/indicator_construction/index_definition_to_excel.py
```

The helper automatically reads the registry; `--sources-manifest` can override
its location. Excel includes `Data_Sources`, `Source_Coverage`, `Source_Evidence`,
and `Source_Field_Guide`, plus source IDs in the existing indicator/folder
sheets. `Source_Coverage` highlights new inputs without source mappings.
Close the workbook in Excel before regenerating it.
