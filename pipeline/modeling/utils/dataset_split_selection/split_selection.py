from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import pandas as pd

from .split_scoring import (
    code_list,
    compute_normalized_wasserstein_per_target,
    joined_codes,
    sum_squared_normalized_wasserstein_distances,
)


# Evaluate one proposed NUTS3 test split for target representativeness and size balance.
def evaluate_test_split_candidate(
    test_codes: Iterable[object],
    *,
    reference_df: pd.DataFrame,
    targets: Sequence[str],
    reference_std: pd.Series,
    target_share: float,
    max_test_share: float,
    size_penalty_weight: float,
    region_col: str = "NUTS3",
) -> dict[str, object]:
    test_candidate_codes = code_list(test_codes)
    test_df = reference_df.loc[
        reference_df[region_col].isin(test_candidate_codes)
    ]
    test_share = len(test_df) / len(reference_df)
    normalized_wasserstein_per_target = compute_normalized_wasserstein_per_target(
        test_df,
        reference_df=reference_df,
        targets=targets,
        reference_std=reference_std,
    )
    normalized_wasserstein_squared_sum = (
        sum_squared_normalized_wasserstein_distances(
            normalized_wasserstein_per_target
        )
    )

    size_tolerance = max_test_share - target_share
    if size_tolerance <= 0:
        raise ValueError("max_test_share must be greater than target_share.")
    size_penalty = ((test_share - target_share) / size_tolerance) ** 2

    result = {
        "test_nuts3": joined_codes(test_candidate_codes),
        "n_test_regions": len(test_candidate_codes),
        "n_test_rows": len(test_df),
        "test_share": test_share,
        "test_normalized_wasserstein_squared_sum": (
            normalized_wasserstein_squared_sum
        ),
        "test_size_penalty": size_penalty,
        "test_score": (
            normalized_wasserstein_squared_sum
            + size_penalty_weight * size_penalty
        ),
    }
    result.update(normalized_wasserstein_per_target)
    return result


# Evaluate eligible region combinations and return their results ordered by score.
def build_test_split_candidate_table(
    *,
    region_counts: Sequence[int],
    reference_df: pd.DataFrame,
    targets: Sequence[str],
    reference_std: pd.Series,
    target_share: float,
    min_test_share: float,
    max_test_share: float,
    size_penalty_weight: float,
    region_col: str = "NUTS3",
) -> pd.DataFrame:
    all_candidate_codes = sorted(reference_df[region_col].unique())
    candidate_results = []
    for region_count in region_counts:
        for test_codes in combinations(
            code_list(all_candidate_codes), region_count
        ):
            candidate_result = evaluate_test_split_candidate(
                test_codes,
                reference_df=reference_df,
                targets=targets,
                reference_std=reference_std,
                target_share=target_share,
                max_test_share=max_test_share,
                size_penalty_weight=size_penalty_weight,
                region_col=region_col,
            )
            if min_test_share <= candidate_result["test_share"] <= max_test_share:
                candidate_results.append(candidate_result)

    return (
        pd.DataFrame(candidate_results)
        .sort_values(["test_score", "test_nuts3"])
        .reset_index(drop=True)
    )
