from __future__ import annotations

from collections.abc import Sequence
from tokenize import group

import pandas as pd

from .split_scoring import (
    code_list,
    compute_normalized_wasserstein_per_target,
    joined_codes,
    sum_squared_normalized_wasserstein_distances,
)
from .training_fold_selection import fold_size_penalty

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - used only outside notebook environments
    display = print


def build_training_fold_dataframe(
    mainland_region_codes: Sequence[str],
    overseas_ids: set[object],
    *,
    mainland_df: pd.DataFrame,
    overseas_df: pd.DataFrame,
    region_col: str = "NUTS3",
    id_col: str = "ID_norm",
) -> pd.DataFrame:
    """Combine one fold's mainland regions with its assigned overseas rows."""
    mainland_part = mainland_df.loc[
        mainland_df[region_col].isin(mainland_region_codes)
    ]
    if not overseas_ids:
        return mainland_part

    overseas_part = overseas_df.loc[overseas_df[id_col].isin(overseas_ids)]
    return pd.concat([mainland_part, overseas_part], ignore_index=True)


def evaluate_single_assignment(
    folds: Sequence[Sequence[str]],
    overseas_ids_by_fold: Sequence[set[object]],
    *,
    mainland_df: pd.DataFrame,
    overseas_df: pd.DataFrame,
    selected_test_codes: Sequence[str],
    reference_df: pd.DataFrame,
    targets: Sequence[str],
    reference_std: pd.Series,
    fold_size_penalty_weight: float,
    region_col: str = "NUTS3",
    id_col: str = "ID_norm",
) -> tuple[float, list[dict[str, object]]]:
    """Score one complete assignment of overseas rows across all training folds."""
    if not folds:
        raise ValueError("At least one training fold is required.")
    if len(folds) != len(overseas_ids_by_fold):
        raise ValueError("Each training fold must have one set of overseas IDs.")

    fold_rows = []
    distribution_total = 0.0
    fold_sizes = []
    test_rows = len(
        mainland_df.loc[mainland_df[region_col].isin(selected_test_codes)]
    )
    target_fold_size = (
        len(mainland_df) - test_rows + len(overseas_df)
    ) / len(folds)

    for fold_idx, mainland_region_codes in enumerate(folds):
        fold_df = build_training_fold_dataframe(
            mainland_region_codes,
            overseas_ids_by_fold[fold_idx],
            mainland_df=mainland_df,
            overseas_df=overseas_df,
            region_col=region_col,
            id_col=id_col,
        )
        fold_sizes.append(len(fold_df))
        normalized_wasserstein_per_target = (
            compute_normalized_wasserstein_per_target(
                fold_df,
                reference_df=reference_df,
                targets=targets,
                reference_std=reference_std,
            )
        )
        distribution_score = sum_squared_normalized_wasserstein_distances(
            normalized_wasserstein_per_target
        )
        distribution_total += distribution_score
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_rows": len(fold_df),
                "mainland_validation_nuts3": joined_codes(
                    mainland_region_codes
                ),
                "overseas_municipalities": "",
                "combined_distribution_sse": distribution_score,
            }
        )

    objective = distribution_total + fold_size_penalty_weight * (
        fold_size_penalty(fold_sizes, target_fold_size)
    )
    return objective, fold_rows


def assign_overseas_municipalities_to_folds(
    mainland_fold_summary: pd.DataFrame,
    *,
    mainland_df: pd.DataFrame,
    overseas_df: pd.DataFrame,
    selected_test_codes: Sequence[str],
    n_train_splits: int,
    reference_df: pd.DataFrame,
    targets: Sequence[str],
    reference_std: pd.Series,
    fold_size_penalty_weight: float,
    region_col: str = "NUTS3",
    municipality_col: str = "municipality_id",
    id_col: str = "ID_norm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily assign each overseas municipality to one existing training fold."""
    if overseas_df.empty:
        raise ValueError("overseas_df must contain at least one row.")
    if n_train_splits <= 0:
        raise ValueError("n_train_splits must be positive.")
    if len(mainland_fold_summary) != n_train_splits:
        raise ValueError(
            "mainland_fold_summary must contain one row per training split."
        )

    # Recover the mainland NUTS3 membership of each previously selected fold.
    folds = [
        list(code_list(row.validation_nuts3.split(", ")))
        for row in mainland_fold_summary.itertuples(index=False)
    ]

    # Keep municipalities intact and place larger groups first because they are
    # the hardest to accommodate without unbalancing fold sizes.
    overseas_municip_groups = [
        {
            "municipality_id": municipality_id,
            "nuts3": joined_codes(municip_group[region_col].unique()),
            "n_rows": len(municip_group),
            "ids": set(municip_group[id_col]),
        }
        for municipality_id, municip_group in overseas_df.groupby(municipality_col)
    ]
    overseas_municip_groups.sort(key=lambda group: group["n_rows"], reverse=True)

    fold_extra_ids = [set() for _ in range(n_train_splits)]
    assignment_rows = []

    # Greedily place each municipality in the fold that produces the lowest
    # combined distribution-and-size objective after that one addition.
    for municip_group in overseas_municip_groups:
        best_candidate = None
        for fold_idx in range(n_train_splits):
            trial_extra_ids = [set(ids) for ids in fold_extra_ids]
            trial_extra_ids[fold_idx].update(municip_group["ids"])
            objective, _ = evaluate_single_assignment(
                folds,
                trial_extra_ids,
                mainland_df=mainland_df,
                overseas_df=overseas_df,
                selected_test_codes=selected_test_codes,
                reference_df=reference_df,
                targets=targets,
                reference_std=reference_std,
                fold_size_penalty_weight=fold_size_penalty_weight,
                region_col=region_col,
                id_col=id_col,
            )
            candidate = (objective, fold_idx)
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        assigned_fold = best_candidate[1]
        fold_extra_ids[assigned_fold].update(municip_group["ids"])
        assignment_rows.append(
            {
                "municipality_id": municip_group["municipality_id"],
                "nuts3": municip_group["nuts3"],
                "n_rows": municip_group["n_rows"],
                "assigned_fold": assigned_fold,
            }
        )

    # Re-evaluate the completed assignment to build its final fold summary.
    _, summaries_of_combined_folds = evaluate_single_assignment(
        folds,
        fold_extra_ids,
        mainland_df=mainland_df,
        overseas_df=overseas_df,
        selected_test_codes=selected_test_codes,
        reference_df=reference_df,
        targets=targets,
        reference_std=reference_std,
        fold_size_penalty_weight=fold_size_penalty_weight,
        region_col=region_col,
        id_col=id_col,
    )
    summaries_of_combined_folds_df = pd.DataFrame(summaries_of_combined_folds)
    overseas_assignments = pd.DataFrame(assignment_rows)
    if overseas_assignments.empty:
        summaries_of_combined_folds_df["overseas_rows"] = 0
        return summaries_of_combined_folds_df, overseas_assignments

    # Add compact municipality and row-count diagnostics to each fold row.
    municipalities_by_fold = (
        overseas_assignments.groupby("assigned_fold")["municipality_id"]
        .apply(lambda values: ", ".join(sorted(values)))
        .to_dict()
    )
    rows_by_fold = (
        overseas_assignments.groupby("assigned_fold")["n_rows"].sum().to_dict()
    )
    summaries_of_combined_folds_df["overseas_municipalities"] = (
        summaries_of_combined_folds_df["fold"]
        .map(municipalities_by_fold)
        .fillna("")
    )
    summaries_of_combined_folds_df["overseas_rows"] = (
        summaries_of_combined_folds_df["fold"]
        .map(rows_by_fold)
        .fillna(0)
        .astype(int)
    )
    return summaries_of_combined_folds_df, overseas_assignments


def display_overseas_fold_assignment_summary(
    summaries_of_combined_folds_df: pd.DataFrame,
    overseas_assignments: pd.DataFrame,
) -> None:
    """Display overseas municipality assignments and combined training folds."""
    print("Overseas municipality assignments:")
    display(overseas_assignments)
    print("Combined all-territory training folds:")
    display(summaries_of_combined_folds_df)
