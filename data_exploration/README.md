# Data Exploration

This folder contains exploratory scripts for inspecting and mapping derived data.

## Files

- `map_index.py`: joins an indicator CSV to CAOP shapefiles and exports choropleth PNGs plus a plot-ready GeoJSON.
- `index_definition_to_excel.py`: exports the JSON indicator manifest to a multi-sheet Excel workbook.

## Notes

`map_index.py` currently points at the generated satellite indicator table:

```text
E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\outputs\indices\freguesia_indices_streaming.csv
```

Generated maps and plot-ready GeoJSON exports should stay outside git.
