from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .split_scoring import joined_codes

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - used only outside notebook environments
    display = print


def display_spatial_split_assignment_summary(
    split_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Display and return row counts by saved split role and training fold."""
    required = {"split_role", "cv_fold"}
    missing = required - set(split_assignments.columns)
    if missing:
        raise ValueError(
            f"Spatial split assignments are missing columns: {sorted(missing)}"
        )
    summary = (
        split_assignments.groupby(["split_role", "cv_fold"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    display(summary)
    return summary


def display_selected_split_summary(
    mainland_df: pd.DataFrame,
    selected_test_df: pd.DataFrame,
    selected_train_mainland_df: pd.DataFrame,
    selected_train_all_territories_df: pd.DataFrame,
    selected_test_codes: Sequence[str],
    *,
    default_test_codes: Sequence[str],
    targets: Sequence[str],
    region_col: str = "NUTS3",
) -> None:
    """Display geographic scope and target statistics for a selected split."""
    scope_summary = pd.DataFrame(
        [
            _scope_summary_row(
                "mainland_only",
                "fixed_test",
                selected_test_df,
                region_col=region_col,
            ),
            _scope_summary_row(
                "mainland_only",
                "training_pool",
                selected_train_mainland_df,
                region_col=region_col,
            ),
            _scope_summary_row(
                "all_territories",
                "fixed_test",
                selected_test_df,
                region_col=region_col,
            ),
            _scope_summary_row(
                "all_territories",
                "training_pool",
                selected_train_all_territories_df,
                region_col=region_col,
            ),
        ]
    )

    print("Selected fixed test regions:", tuple(selected_test_codes))
    print("Current utility default:", tuple(default_test_codes))
    if tuple(selected_test_codes) != tuple(default_test_codes):
        print(
            "NOTE: selected test set differs from DEFAULT_TEST_NUTS3. "
            "Update the default only after accepting this design."
        )
    display(scope_summary)

    samples = [
        ("mainland reference", mainland_df),
        ("selected fixed test", selected_test_df),
        ("mainland training pool", selected_train_mainland_df),
        ("all-territory training pool", selected_train_all_territories_df),
    ]
    target_comparison = pd.DataFrame(
        [_target_summary_row(label, df, targets) for label, df in samples]
    )
    display(target_comparison)


def _scope_summary_row(
    scope: str,
    role: str,
    df: pd.DataFrame,
    *,
    region_col: str,
) -> dict[str, object]:
    return {
        "scope": scope,
        "role": role,
        "n_rows": len(df),
        "n_nuts3": df[region_col].nunique(),
        "nuts3": joined_codes(df[region_col].unique()),
    }


def _target_summary_row(
    label: str,
    df: pd.DataFrame,
    targets: Sequence[str],
) -> dict[str, object]:
    row = {"sample": label, "n_rows": len(df)}
    for target in targets:
        row[f"{target}_mean"] = df[target].mean()
        row[f"{target}_sd"] = df[target].std()
    return row
