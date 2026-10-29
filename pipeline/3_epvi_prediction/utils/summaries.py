from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - used only outside notebook environments
    display = print


def display_geographic_scope_summary(
    model_df: pd.DataFrame,
    mainland_df: pd.DataFrame,
    overseas_df: pd.DataFrame,
    *,
    region_col: str = "NUTS3",
    municipality_col: str = "municipality_id",
) -> pd.DataFrame:
    """Print scope statistics, display region sizes, and return the summary table."""
    required_columns = {
        "model_df": (model_df, {region_col}),
        "mainland_df": (mainland_df, {region_col, municipality_col}),
        "overseas_df": (overseas_df, {region_col, municipality_col}),
    }
    for name, (df, required) in required_columns.items():
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    if mainland_df.empty:
        raise ValueError("mainland_df is empty; mainland row shares cannot be calculated.")
    if len(mainland_df) + len(overseas_df) != len(model_df):
        raise ValueError("mainland_df and overseas_df do not partition model_df by row count.")

    region_sizes = (
        model_df[region_col]
        .value_counts()
        .sort_index()
        .rename("n_rows")
        .to_frame()
    )
    region_sizes["row_share_all"] = region_sizes["n_rows"] / len(model_df)
    region_sizes["row_share_mainland"] = region_sizes["n_rows"] / len(mainland_df)
    region_sizes = region_sizes.sort_values("n_rows", ascending=False)

    print("Mainland:")
    print(f"- {mainland_df[region_col].nunique()} regions")
    print(f"- {mainland_df[municipality_col].nunique()} municipalities")
    print(f"- {len(mainland_df)} freguesias")
    print()
    print("Overseas:")
    print(f"- {overseas_df[region_col].nunique()} regions")
    print(f"- {overseas_df[municipality_col].nunique()} municipalities")
    print(f"- {len(overseas_df)} freguesias")
    display(region_sizes)
    return region_sizes


def display_test_split_candidate_summary(
    test_candidates: pd.DataFrame,
    *,
    top_n: int = 20,
) -> None:
    """Display candidate counts and the leading rows from a ranked test-split table."""
    if test_candidates.empty:
        raise ValueError("The test-split candidate table is empty.")

    required_columns = {"n_test_regions"}
    missing = required_columns - set(test_candidates.columns)
    if missing:
        raise ValueError(
            f"Test-split candidate table is missing columns: {sorted(missing)}"
        )

    candidate_counts = (
        test_candidates["n_test_regions"]
        .value_counts()
        .sort_index()
        .rename("n_candidates")
        .to_frame()
    )
    print("Eligible mainland test candidates:", len(test_candidates))
    print("Candidate counts by number of regions:")
    display(candidate_counts)
    print("Best test-set-only candidates:")
    display(test_candidates.head(top_n))


def display_spatial_split_selection_summary(
    split_designs: pd.DataFrame,
    selected_design: pd.Series,
    selected_test_codes: Sequence[str],
    selected_fold_summary: pd.DataFrame,
    *,
    top_n: int = 20,
) -> None:
    """Display ranked split designs and details of the selected spatial split."""
    if split_designs.empty:
        raise ValueError("The spatial split-design table is empty.")
    if selected_fold_summary.empty:
        raise ValueError("The selected fold summary is empty.")

    required_design_values = {"size_valid_fold_assignments"}
    missing_design_values = required_design_values - set(selected_design.index)
    if missing_design_values:
        raise ValueError(
            "Selected design is missing values: "
            f"{sorted(missing_design_values)}"
        )

    required_fold_columns = {
        "min_acceptable_rows",
        "max_acceptable_rows",
    }
    missing_fold_columns = required_fold_columns - set(selected_fold_summary.columns)
    if missing_fold_columns:
        raise ValueError(
            "Selected fold summary is missing columns: "
            f"{sorted(missing_fold_columns)}"
        )

    print("Best full mainland split designs:")
    display(split_designs.head(top_n))
    print("Selected mainland test NUTS3:", selected_test_codes)
    print(
        "Fold size bounds:",
        int(selected_fold_summary["min_acceptable_rows"].iloc[0]),
        "to",
        int(selected_fold_summary["max_acceptable_rows"].iloc[0]),
        "rows",
    )
    print(
        "Collected size-valid fold assignments evaluated:",
        int(selected_design["size_valid_fold_assignments"]),
    )
    display(selected_fold_summary)
