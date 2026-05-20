# Data Exploration

This folder contains exploratory scripts for inspecting and mapping derived data.

## Files

- `map_index.py`: joins indicator CSVs to CAOP shapefiles and exports choropleth PNGs plus plot-ready GeoJSON files. It maps both the generated satellite indicators and the private EPVI indicators.
- `index_definition_to_excel.py`: exports the JSON indicator manifest to a multi-sheet Excel workbook.

## Notes

`map_index.py` currently points at the generated satellite indicator table:

```text
E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\outputs\indices\freguesia_indices_streaming.csv
```

Generated maps and plot-ready GeoJSON exports should stay outside git.

If `data/spatial_test_set_boundary.geojson` exists, `map_index.py` overlays its
boundary in black on every map. This file should be a small precomputed
dissolved boundary for a fixed spatial test set. The current random-forest
training notebook uses cross-validation, so there is no single test-set
boundary unless a fixed hold-out is defined separately.
