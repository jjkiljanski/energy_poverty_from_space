from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.utils.geometry import load_geometries_with_normalized_ids
from pipeline.utils.io import read_csv_robust
from pipeline.utils.plots import plot_portugal_index

from .load_data import normalize_id

DEFAULT_TEST_NUTS3 = ("PT112", "PT16B", "PT16I", "PT16J")
OVERSEAS_NUTS3 = ("PT200", "PT300")


def load_nuts3_mapping(mapping_csv: str | Path) -> pd.DataFrame:
    """Load and validate the parish-to-NUTS3 correspondence."""
    mapping = read_csv_robust(mapping_csv)
    required = {"LAU_CODE", "NUTS_3"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"NUTS3 mapping is missing required columns: {sorted(missing)}")

    mapping = mapping.copy()
    mapping["ID_norm"] = normalize_id(mapping["LAU_CODE"])
    mapping["NUTS3"] = mapping["NUTS_3"].astype(str).str.strip().str.upper()
    mapping = mapping[mapping["ID_norm"].notna() & mapping["NUTS3"].ne("")].copy()

    duplicate_ids = mapping[mapping["ID_norm"].duplicated(keep=False)]
    conflicting = duplicate_ids.groupby("ID_norm")["NUTS3"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        raise ValueError(
            "NUTS3 mapping assigns multiple regions to parish IDs: "
            f"{conflicting.index[:10].tolist()}"
        )

    return mapping.drop_duplicates("ID_norm")[["ID_norm", "NUTS3"]]


def attach_nuts3(model_df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Attach one NUTS3 code to each modeling row and fail on missing mappings."""
    out = model_df.merge(mapping, on="ID_norm", how="left", validate="one_to_one")
    missing_ids = out.loc[out["NUTS3"].isna(), "ID_norm"].dropna().tolist()
    if missing_ids:
        raise ValueError(
            f"NUTS3 mapping is missing {len(missing_ids)} modeling IDs; "
            f"first examples: {missing_ids[:10]}"
        )
    return out


def plot_nuts3_scope(
    mapping: pd.DataFrame,
    parish_geometry_geojson: str | Path,
    included_nuts3: set[str],
) -> None:
    """Plot included and excluded Portuguese territories using island insets."""
    parishes = load_geometries_with_normalized_ids(
        parish_geometry_geojson,
        source_id_col="dicofre",
    )
    territories = parishes.merge(mapping, on="ID_norm", how="left", validate="one_to_one")

    missing_nuts3 = territories["NUTS3"].isna()
    if missing_nuts3.any():
        examples = territories.loc[missing_nuts3, "ID_norm"].dropna().head(10).tolist()
        raise ValueError(
            f"NUTS3 mapping is missing {int(missing_nuts3.sum())} parish geometries; "
            f"first examples: {examples}"
        )

    territories["included"] = territories["NUTS3"].isin(included_nuts3)
    mainland = territories.loc[~territories["NUTS3"].isin(OVERSEAS_NUTS3)].copy()
    azores = territories.loc[territories["NUTS3"].eq("PT200")].copy()
    madeira = territories.loc[territories["NUTS3"].eq("PT300")].copy()

    plot_portugal_index(
        mainland=mainland,
        islands=[("Azores", azores), ("Madeira", madeira)],
        value_col="included",
        mode="categorical",
        title="Territories included in the modeling scope",
        category_colors={True: "#2E8B57", False: "#D9D9D9"},
        category_labels={True: "Included", False: "Excluded"},
        show_missing_count=False,
        show=True,
    )


def apply_nuts3_scope(
    model_df: pd.DataFrame,
    mapping_csv: str | Path,
    parish_geometry_geojson: str | Path,
    exclude_overseas: bool = True,
) -> pd.DataFrame:
    """Attach NUTS3, visualize the selected scope, and return included rows."""
    mapping = load_nuts3_mapping(mapping_csv)
    model_with_nuts3 = attach_nuts3(model_df, mapping)
    included_nuts3 = set(mapping["NUTS3"].unique())
    if exclude_overseas:
        included_nuts3.difference_update(OVERSEAS_NUTS3)

    plot_nuts3_scope(mapping, parish_geometry_geojson, included_nuts3)

    scoped_model = model_with_nuts3.loc[
        model_with_nuts3["NUTS3"].isin(included_nuts3)
    ].copy()
    excluded_rows = len(model_with_nuts3) - len(scoped_model)
    if excluded_rows:
        print(f"Excluded {excluded_rows} rows outside the selected territory scope.")
    return scoped_model
