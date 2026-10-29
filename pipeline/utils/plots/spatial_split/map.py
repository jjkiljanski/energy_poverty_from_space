from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from pipeline.utils.geometry import load_geometries_with_normalized_ids
from pipeline.utils.plots.portugal_index import plot_portugal_index

TRAINING_FOLD_COLORS = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE")
FIXED_TEST_COLOR = "#AA3377"
OVERSEAS_NUTS3 = ("PT200", "PT300")


def plot_spatial_split_assignments(
    assignments: pd.DataFrame,
    parish_geometry_geojson: str | Path,
    selected_test_nuts3: tuple[str, ...],
    *,
    include_overseas: bool = True,
    title: str | None = None,
    show: bool = True,
) -> Figure:
    """Plot row-level spatial split assignments using the shared Portugal map."""
    required = {"ID_norm", "NUTS3", "split_role", "cv_fold", "split_label"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(
            f"Spatial split map requires assignment columns: {sorted(missing)}"
        )

    parishes = load_geometries_with_normalized_ids(
        parish_geometry_geojson,
        source_id_col="dicofre",
    )
    missing_geometries = sorted(set(assignments["ID_norm"]) - set(parishes["ID_norm"]))
    if missing_geometries:
        raise ValueError(
            f"Parish geometry is missing {len(missing_geometries)} assigned IDs; "
            f"first examples: {missing_geometries[:10]}"
        )

    split_map = parishes.merge(
        assignments[["ID_norm", "NUTS3", "split_role", "cv_fold", "split_label"]],
        on="ID_norm",
        how="inner",
        validate="one_to_one",
    )

    training_labels = (
        assignments.loc[
            assignments["split_role"].eq("training"), ["cv_fold", "split_label"]
        ]
        .drop_duplicates()
        .sort_values("cv_fold")["split_label"]
        .tolist()
    )
    if len(training_labels) > len(TRAINING_FOLD_COLORS):
        raise ValueError(
            f"The split map defines {len(training_labels)} training folds, but only "
            f"{len(TRAINING_FOLD_COLORS)} fold colors are configured."
        )

    test_labels = (
        assignments.loc[assignments["split_role"].eq("fixed_test"), "split_label"]
        .drop_duplicates()
        .tolist()
    )
    category_colors = dict(zip(training_labels, TRAINING_FOLD_COLORS))
    category_colors.update({label: FIXED_TEST_COLOR for label in test_labels})
    category_labels = {label: label for label in training_labels}
    test_description = "Fixed test: " + " + ".join(selected_test_nuts3)
    category_labels.update({label: test_description for label in test_labels})

    mainland = split_map.loc[~split_map["NUTS3"].isin(OVERSEAS_NUTS3)].copy()
    islands = []
    if include_overseas:
        azores = split_map.loc[split_map["NUTS3"].eq("PT200")].copy()
        madeira = split_map.loc[split_map["NUTS3"].eq("PT300")].copy()
        islands = [
            (name, gdf)
            for name, gdf in [("Azores", azores), ("Madeira", madeira)]
            if not gdf.empty
        ]

    if title is None:
        title = (
            "Saved spatial tuning folds and fixed test regions"
            if include_overseas
            else "Mainland-only spatial split design"
        )
    return plot_portugal_index(
        mainland=mainland,
        islands=islands,
        value_col="split_label",
        mode="categorical",
        title=title,
        category_colors=category_colors,
        category_labels=category_labels,
        show_missing_count=False,
        show=show,
    )
