from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
from scipy.stats import wasserstein_distance


# Calculate each target's Wasserstein distance normalized by its reference standard deviation.
def compute_normalized_wasserstein_per_target(
    sample_df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame,
    targets: Sequence[str],
    reference_std: pd.Series,
) -> dict[str, float]:
    normalized_wasserstein_per_target = {}
    for target in targets:
        sample_values = sample_df[target].dropna().astype(float).to_numpy()
        reference_values = reference_df[target].dropna().astype(float).to_numpy()
        target_wasserstein_distance = float(
            wasserstein_distance(sample_values, reference_values)
        )
        normalized_wasserstein_per_target[f"{target}_w1_norm"] = (
            target_wasserstein_distance / reference_std[target]
        )
    return normalized_wasserstein_per_target


# Sum the squares of normalized per-target Wasserstein distances, ignoring missing values.
def sum_squared_normalized_wasserstein_distances(
    normalized_wasserstein_per_target: Mapping[str, float],
) -> float:
    available_distances = (
        distance
        for distance in normalized_wasserstein_per_target.values()
        if pd.notna(distance)
    )
    return float(sum(distance**2 for distance in available_distances))


# Normalize region codes to a deterministic sorted tuple of strings.
def code_list(codes: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted(str(code) for code in codes))


# Format normalized region codes as a comma-separated string.
def joined_codes(codes: Iterable[object]) -> str:
    return ", ".join(code_list(codes))
