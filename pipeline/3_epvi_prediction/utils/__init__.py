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
