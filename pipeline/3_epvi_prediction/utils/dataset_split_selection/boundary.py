from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd

from pipeline.utils.geometry import (
    load_geometries_with_normalized_ids,
    union_geometries,
)

from .split_scoring import joined_codes


def write_spatial_test_boundary(
    model_df: pd.DataFrame,
    selected_test_codes: Sequence[str],
    parish_geometry_geojson: str | Path,
    output_geojson: str | Path,
) -> gpd.GeoDataFrame:
    """Dissolve selected test parishes and write their spatial boundary to GeoJSON."""
    parishes = load_geometries_with_normalized_ids(
        parish_geometry_geojson,
        source_id_col="dicofre",
    )
    selected_test_ids = model_df.loc[
        model_df["NUTS3"].isin(selected_test_codes), "ID_norm"
    ].drop_duplicates()
    boundary_parts = parishes.loc[
        parishes["ID_norm"].isin(selected_test_ids)
    ].copy()
    if boundary_parts.empty:
        raise ValueError(
            "No parish geometries matched the selected test set; boundary was not regenerated."
        )

    missing_geometry_ids = sorted(set(selected_test_ids) - set(boundary_parts["ID_norm"]))
    if missing_geometry_ids:
        raise ValueError(
            f"Missing geometries for {len(missing_geometry_ids)} selected test parishes; "
            f"first examples: {missing_geometry_ids[:10]}"
        )

    test_codes_label = joined_codes(selected_test_codes)
    boundary = gpd.GeoDataFrame(
        {
            "split_role": ["fixed_test"],
            "test_nuts3": [test_codes_label],
            "n_parishes": [len(boundary_parts)],
        },
        geometry=[union_geometries(boundary_parts)],
        crs=boundary_parts.crs,
    )
    output = Path(output_geojson)
    output.parent.mkdir(parents=True, exist_ok=True)
    boundary.to_file(output, driver="GeoJSON")

    print(f"Regenerated {output}")
    print(f"Selected test regions: {test_codes_label}")
    print(f"Parish geometries dissolved: {len(boundary_parts)}")
    return boundary
