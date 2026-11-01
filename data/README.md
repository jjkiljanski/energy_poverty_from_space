# Repository data

This folder contains the prepared inputs retained for modeling. Preparation
method and lineage are recorded in
[`pipeline/definitions/data_sources_manifest.json`](../pipeline/definitions/data_sources_manifest.json) and
[the provenance notes](../docs/data_provenance.md).

```text
data/
    docs/
        indices_manifest.xlsx
        data_sources.xlsx
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
`verified` refers only to the download lineage within `verification_notes`.
`citation_verified` is separate: references transcribed from the a priori PDF
are candidates, even when the associated download lineage is verified.
The user-verified download notes take precedence wherever they conflict with
the a priori PDF. Conflicting PDF claims are retained only as planning evidence.
Unverified planned descriptions, coverage, and access transcribed from the PDF
are preserved as labeled entries in `notes`; they do not describe actual downloads.
`rationale` remains empty unless the choice over alternative sources is documented.
The evidence section retains the original notes and PDF text, including details
that do not safely map to a current input. No external source lookup was used.

Sources are grouped as `used` and `planned_only`; the latter combines planned
candidates and available optional sources. Additional source metadata is kept
in each record's `notes` string. Prepared artifacts and their lineage are
listed in `pipeline/definitions/data_preparation_manifest.json`.

The generated review workbooks are tracked at
[`docs/indices_manifest.xlsx`](docs/indices_manifest.xlsx) and
[`docs/data_sources.xlsx`](docs/data_sources.xlsx). Regenerate and commit them
when the JSON definitions change. Run from the repository root:

```powershell
python pipeline/input_preparation/indicator_construction/index_definition_to_excel.py
```

The helper automatically reads both manifests; `--sources-manifest` can override
the source registry location. The data sources workbook includes used sources,
planned and optional sources, and prepared artifacts. Close the workbooks in
Excel before regenerating them.
