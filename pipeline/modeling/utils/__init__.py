from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

from .load_data import (
    EPVI_ID_CORRECTIONS,
    build_modeling_table,
    coerce_numeric_frame,
    id_coverage_summary,
    load_modeling_inputs,
    load_prediction_inputs,
    normalize_epvi_ids,
    normalize_id,
)
from .nuts3 import (
    DEFAULT_TEST_NUTS3,
    OVERSEAS_NUTS3,
    apply_nuts3_scope,
    attach_nuts3,
    load_nuts3_mapping,
    plot_nuts3_scope,
)
from .spatial_splits import (
    apply_spatial_split_assignments,
    load_spatial_split_assignments,
    prepare_spatial_splits,
)
from pipeline.utils.plots.spatial_split import (
    plot_fold_distribution_diagnostics,
    plot_sample_distribution_diagnostics,
    plot_spatial_split_assignments,
)
from .dataset_split_selection import (
    assign_overseas_municipalities_to_folds,
    best_fold_assignment,
    build_training_fold_dataframe,
    build_spatial_split_assignments,
    build_test_split_candidate_table,
    canonical_folds,
    code_list,
    collect_size_valid_fold_assignments,
    compute_fold_metrics,
    compute_normalized_wasserstein_per_target,
    display_overseas_fold_assignment_summary,
    evaluate_test_split_candidate,
    evaluate_single_assignment,
    fold_size_bounds,
    fold_size_penalty,
    joined_codes,
    display_selected_split_summary,
    display_spatial_split_assignment_summary,
    score_size_valid_fold_assignments,
    select_best_spatial_split_design,
    single_random_fold_assignment,
    summarize_fold_sizes,
    sum_squared_normalized_wasserstein_distances,
    write_spatial_test_boundary,
)
from .summaries import (
    display_geographic_scope_summary,
    display_spatial_split_selection_summary,
    display_test_split_candidate_summary,
)


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from a notebook or script working directory."""
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "pipeline" / "config").exists():
            return path
    raise FileNotFoundError("Could not find repository root containing pipeline/config.")


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def spearman_corr(y_true, y_pred) -> float:
    return pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")

def drop_constant_predictors(
    model_df: pd.DataFrame,
    candidate_predictor_cols: list[str],
) -> tuple[list[str], list[str]]:
    """Drop constant predictors from the modeling table and return remaining columns."""
    constant_cols = [c for c in candidate_predictor_cols if model_df[c].nunique(dropna=True) <= 1]
    predictor_cols = [c for c in candidate_predictor_cols if c not in constant_cols]

    print(f"Constant columns excluded from predictor_cols ({len(constant_cols)}): {constant_cols}.")
    print(f"Remaining predictors: {len(predictor_cols)}.")

    return predictor_cols, constant_cols
