from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import random
import time

import numpy as np
import pandas as pd

from .split_scoring import (
    code_list,
    compute_normalized_wasserstein_per_target,
    joined_codes,
    sum_squared_normalized_wasserstein_distances,
)


# Measure the mean squared relative deviation of fold sizes from their target size.
def fold_size_penalty(sizes: Iterable[float], target_size: float) -> float:
    size_values = np.asarray(sizes, dtype=float)
    return float(np.mean(((size_values - target_size) / target_size) ** 2))


# Convert minimum and maximum fold shares into inclusive row-count bounds.
def fold_size_bounds(
    total_rows: int,
    *,
    min_fold_share: float,
    max_fold_share: float,
) -> tuple[int, int]:
    return (
        int(np.floor(total_rows * min_fold_share)),
        int(np.ceil(total_rows * max_fold_share)),
    )


# Summarize fold sizes and mark folds outside the acceptable row-count bounds.
def summarize_fold_sizes(
    folds: Sequence[Sequence[object]],
    *,
    region_counts: Mapping[str, int],
    n_splits: int,
    min_fold_share: float,
    max_fold_share: float,
) -> tuple[float, pd.DataFrame]:
    fold_rows = []
    for fold_idx, codes in enumerate(folds):
        fold_codes = code_list(codes)
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_rows": sum(region_counts[code] for code in fold_codes),
                "n_regions": len(fold_codes),
                "validation_nuts3": joined_codes(fold_codes),
            }
        )

    fold_summary = pd.DataFrame(fold_rows)
    total_rows = int(fold_summary["n_rows"].sum())
    target_fold_size = total_rows / n_splits
    size_penalty = fold_size_penalty(
        fold_summary["n_rows"], target_fold_size
    )
    min_rows, max_rows = fold_size_bounds(
        total_rows,
        min_fold_share=min_fold_share,
        max_fold_share=max_fold_share,
    )
    fold_summary["min_acceptable_rows"] = min_rows
    fold_summary["max_acceptable_rows"] = max_rows
    fold_summary["size_valid"] = fold_summary["n_rows"].between(
        min_rows, max_rows
    )
    return size_penalty, fold_summary


# Evaluate target distributions only when every fold has an acceptable size.
def compute_fold_metrics(
    folds: Sequence[Sequence[object]],
    *,
    region_counts: Mapping[str, int],
    reference_df: pd.DataFrame,
    targets: Sequence[str],
    reference_std: pd.Series,
    n_splits: int,
    min_fold_share: float,
    max_fold_share: float,
    region_col: str = "NUTS3",
) -> tuple[float, float, pd.DataFrame] | None:
    """Evaluate one fold assignment's target distributions and row-count balance.

    ``folds`` normally groups NUTS3 codes by validation fold, for example::

        folds = [("PT111", "PT112"), ("PT119", "PT11A"), ("PT11B", "PT11C")]
    """
    size_penalty, fold_summary = summarize_fold_sizes(
        folds,
        region_counts=region_counts,
        n_splits=n_splits,
        min_fold_share=min_fold_share,
        max_fold_share=max_fold_share,
    )
    if not fold_summary["size_valid"].all():
        return None

    distribution_rows = []
    total_normalized_wasserstein_squared_sum = 0.0
    for fold_idx, codes in enumerate(folds):
        fold_codes = code_list(codes)
        fold_df = reference_df.loc[reference_df[region_col].isin(fold_codes)]
        normalized_wasserstein_per_target = (
            compute_normalized_wasserstein_per_target(
                fold_df,
                reference_df=reference_df,
                targets=targets,
                reference_std=reference_std,
            )
        )
        normalized_wasserstein_squared_sum = (
            sum_squared_normalized_wasserstein_distances(
                normalized_wasserstein_per_target
            )
        )
        total_normalized_wasserstein_squared_sum += (
            normalized_wasserstein_squared_sum
        )
        distribution_result = {
            "fold": fold_idx,
            "fold_normalized_wasserstein_squared_sum": (
                normalized_wasserstein_squared_sum
            ),
        }
        distribution_result.update(normalized_wasserstein_per_target)
        distribution_rows.append(distribution_result)

    fold_summary = fold_summary.merge(
        pd.DataFrame(distribution_rows),
        on="fold",
        how="left",
        validate="one_to_one",
    )
    return total_normalized_wasserstein_squared_sum, size_penalty, fold_summary


# Normalize a fold assignment so equivalent fold orderings share one signature.
def canonical_folds(
    folds: Sequence[Sequence[object]],
) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(code_list(fold) for fold in folds))


# Randomly assign region codes to folds while respecting maximum fold size.
def single_random_fold_assignment(
    available_codes: Sequence[str],
    *,
    region_counts: Mapping[str, int],
    n_splits: int,
    rng: random.Random,
    min_rows: int,
    max_rows: int,
) -> list[tuple[str, ...]] | None:
    """Attempt to place every region into one size-constrained fold.

    Returns one complete assignment, or ``None`` when this random attempt
    cannot satisfy the configured fold-size bounds.
    """
    total_rows = sum(region_counts[code] for code in available_codes)
    target_size = total_rows / n_splits

    # Place large regions first because they are the hardest to fit later.
    # Random tie-breaking lets repeated attempts explore different assignments.
    ordered_codes = list(available_codes)
    rng.shuffle(ordered_codes)
    ordered_codes = sorted(
        ordered_codes,
        key=lambda code: (region_counts[code], rng.random()),
        reverse=True,
    )

    folds = [[] for _ in range(n_splits)]
    sizes = [0] * n_splits
    for code in ordered_codes:
        code_n = region_counts[code]

        # Discard a destination that would immediately exceed the upper bound.
        feasible = [
            idx for idx in range(n_splits) if sizes[idx] + code_n <= max_rows
        ]
        if not feasible:
            return None
        rng.shuffle(feasible)

        # Usually choose the fold brought closest to the ideal size; occasionally
        # choose any feasible fold so repeated attempts do not converge identically.
        if rng.random() < 0.85:
            fold_idx = min(
                feasible,
                key=lambda idx: (
                    abs((sizes[idx] + code_n) - target_size),
                    rng.random(),
                ),
            )
        else:
            fold_idx = rng.choice(feasible)
        folds[fold_idx].append(code)
        sizes[fold_idx] += code_n

    # The upper bound was enforced incrementally; now check lower bounds and
    # reject any incomplete assignment before it reaches Wasserstein scoring.
    if any(size < min_rows or size > max_rows for size in sizes):
        return None
    if any(not fold for fold in folds):
        return None
    return [code_list(fold) for fold in folds]


# Collect unique random fold assignments that satisfy all size constraints.
def collect_size_valid_fold_assignments(
    available_codes: Sequence[str],
    *,
    region_counts: Mapping[str, int],
    n_splits: int,
    min_fold_share: float,
    max_fold_share: float,
    max_partitions: int,
    max_attempts: int,
    seed: int,
    progress_every: int,
) -> list[tuple[tuple[str, ...], ...]]:
    rng = random.Random(seed)
    total_rows = sum(region_counts[code] for code in available_codes)
    min_rows, max_rows = fold_size_bounds(
        total_rows,
        min_fold_share=min_fold_share,
        max_fold_share=max_fold_share,
    )
    valid = []
    seen = set()
    size_rejected = 0
    duplicate_rejected = 0
    attempts_run = 0
    started_at = time.time()

    print(
        f"Collecting up to {max_partitions:,} size-valid fold assignments "
        f"from at most {max_attempts:,} random attempts with fold bounds "
        f"{min_rows}-{max_rows} rows and seed {seed}...",
        flush=True,
    )

    for attempt in range(1, max_attempts + 1):
        attempts_run = attempt
        folds = single_random_fold_assignment(
            available_codes,
            region_counts=region_counts,
            n_splits=n_splits,
            rng=rng,
            min_rows=min_rows,
            max_rows=max_rows,
        )
        if folds is None:
            size_rejected += 1
        else:
            signature = canonical_folds(folds)
            if signature in seen:
                duplicate_rejected += 1
            else:
                seen.add(signature)
                valid.append(signature)
                if len(valid) == 1 or len(valid) % 100 == 0:
                    elapsed = time.time() - started_at
                    print(
                        f"  collected {len(valid):,}/{max_partitions:,} valid "
                        f"assignments after {attempt:,} attempts | "
                        f"size-rejected={size_rejected:,} | "
                        f"duplicates={duplicate_rejected:,} | "
                        f"elapsed={elapsed/60:.1f} min",
                        flush=True,
                    )
                if len(valid) >= max_partitions:
                    break

        if progress_every and attempt % progress_every == 0:
            elapsed = time.time() - started_at
            print(
                f"  attempts {attempt:,}/{max_attempts:,} | "
                f"collected={len(valid):,}/{max_partitions:,} | "
                f"size-rejected={size_rejected:,} | "
                f"duplicates={duplicate_rejected:,} | "
                f"elapsed={elapsed/60:.1f} min",
                flush=True,
            )

    elapsed = time.time() - started_at
    print(
        f"Size-valid fold collection finished in {elapsed/60:.1f} min: "
        f"{len(valid):,} valid assignments collected from "
        f"{attempts_run:,} attempts; size-rejected={size_rejected:,}, "
        f"duplicates={duplicate_rejected:,}.",
        flush=True,
    )
    return valid


# Score size-valid fold assignments and return the lowest-objective result.
def score_size_valid_fold_assignments(
    valid_partitions: Sequence[Sequence[Sequence[object]]],
    *,
    region_counts: Mapping[str, int],
    reference_df: pd.DataFrame,
    targets: Sequence[str],
    reference_std: pd.Series,
    n_splits: int,
    min_fold_share: float,
    max_fold_share: float,
    size_penalty_weight: float,
    progress_every: int,
    region_col: str = "NUTS3",
) -> tuple[float, float, float, list[tuple[str, ...]], pd.DataFrame]:
    if not valid_partitions:
        raise RuntimeError(
            "No size-valid fold assignment found. Loosen the fold-share bounds "
            "or increase the maximum number of assignment attempts."
        )

    best = None
    started_at = time.time()
    print(
        f"Scoring {len(valid_partitions):,} collected size-valid assignments "
        "with EPG Wasserstein distances...",
        flush=True,
    )

    for idx, signature in enumerate(valid_partitions, start=1):
        folds = [code_list(fold) for fold in signature]
        fold_result = compute_fold_metrics(
            folds,
            region_counts=region_counts,
            reference_df=reference_df,
            targets=targets,
            reference_std=reference_std,
            n_splits=n_splits,
            min_fold_share=min_fold_share,
            max_fold_share=max_fold_share,
            region_col=region_col,
        )
        if fold_result is None:
            continue
        fold_wasserstein_sum, size_penalty, fold_summary = fold_result
        objective = fold_wasserstein_sum + size_penalty_weight * size_penalty
        candidate = (
            objective,
            fold_wasserstein_sum,
            size_penalty,
            folds,
            fold_summary,
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
            print(
                f"  new best at {idx:,}/{len(valid_partitions):,}: "
                f"objective={objective:.4f}, "
                f"fold_wasserstein_sum={fold_wasserstein_sum:.4f}, "
                f"size_penalty={size_penalty:.4f}, "
                f"fold_sizes={fold_summary['n_rows'].tolist()}",
                flush=True,
            )
        if progress_every and idx % progress_every == 0:
            elapsed = time.time() - started_at
            print(
                f"  scored {idx:,}/{len(valid_partitions):,} | "
                f"current_best={best[0]:.4f} | "
                f"elapsed={elapsed/60:.1f} min",
                flush=True,
            )

    if best is None:
        raise RuntimeError("No size-valid fold assignment remained for scoring.")
    elapsed = time.time() - started_at
    print(
        f"Fold scoring finished in {elapsed/60:.1f} min. "
        f"Best objective={best[0]:.4f}",
        flush=True,
    )
    return best


# Collect and score candidate assignments for one proposed test set.
def best_fold_assignment(
    available_codes: Sequence[str],
    *,
    region_counts: Mapping[str, int],
    reference_df: pd.DataFrame,
    targets: Sequence[str],
    reference_std: pd.Series,
    n_splits: int,
    min_fold_share: float,
    max_fold_share: float,
    max_partitions: int,
    max_attempts: int,
    seed: int,
    collection_progress_every: int,
    scoring_progress_every: int,
    size_penalty_weight: float,
    region_col: str = "NUTS3",
) -> tuple[float, float, float, list[tuple[str, ...]], pd.DataFrame, int]:
    valid_partitions = collect_size_valid_fold_assignments(
        available_codes,
        region_counts=region_counts,
        n_splits=n_splits,
        min_fold_share=min_fold_share,
        max_fold_share=max_fold_share,
        max_partitions=max_partitions,
        max_attempts=max_attempts,
        seed=seed,
        progress_every=collection_progress_every,
    )
    objective, fold_score, size_penalty, folds, fold_summary = (
        score_size_valid_fold_assignments(
            valid_partitions,
            region_counts=region_counts,
            reference_df=reference_df,
            targets=targets,
            reference_std=reference_std,
            n_splits=n_splits,
            min_fold_share=min_fold_share,
            max_fold_share=max_fold_share,
            size_penalty_weight=size_penalty_weight,
            progress_every=scoring_progress_every,
            region_col=region_col,
        )
    )
    return (
        objective,
        fold_score,
        size_penalty,
        folds,
        fold_summary,
        len(valid_partitions),
    )


# Select the best joint test-set and validation-fold spatial design.
def select_best_spatial_split_design(
    test_split_candidates: pd.DataFrame,
    *,
    reference_df: pd.DataFrame,
    targets: Sequence[str],
    reference_std: pd.Series,
    n_splits: int,
    min_fold_share: float,
    max_fold_share: float,
    max_partitions: int,
    max_attempts: int,
    fold_random_seed: int,
    collection_progress_every: int,
    scoring_progress_every: int,
    top_test_candidates: int,
    test_size_penalty_weight: float,
    fold_size_penalty_weight: float,
    region_col: str = "NUTS3",
) -> tuple[pd.DataFrame, pd.Series, tuple[str, ...], pd.DataFrame]:
    region_counts = {
        str(code): int(count)
        for code, count in reference_df[region_col].value_counts().items()
    }
    all_candidate_codes = code_list(region_counts)
    design_rows = []
    fold_designs = {}

    for _, test_candidate in test_split_candidates.head(
        top_test_candidates
    ).iterrows():
        test_codes = code_list(test_candidate["test_nuts3"].split(", "))
        available_codes = [
            code for code in all_candidate_codes if code not in test_codes
        ]
        (
            _,
            fold_wasserstein_sum,
            fold_size_penalty_value,
            _,
            fold_summary,
            valid_partition_count,
        ) = best_fold_assignment(
            available_codes,
            region_counts=region_counts,
            reference_df=reference_df,
            targets=targets,
            reference_std=reference_std,
            n_splits=n_splits,
            min_fold_share=min_fold_share,
            max_fold_share=max_fold_share,
            max_partitions=max_partitions,
            max_attempts=max_attempts,
            seed=fold_random_seed,
            collection_progress_every=collection_progress_every,
            scoring_progress_every=scoring_progress_every,
            size_penalty_weight=fold_size_penalty_weight,
            region_col=region_col,
        )
        full_objective = (
            test_candidate["test_normalized_wasserstein_squared_sum"]
            + fold_wasserstein_sum
            + test_size_penalty_weight * test_candidate["test_size_penalty"]
            + fold_size_penalty_weight * fold_size_penalty_value
        )
        test_nuts3 = test_candidate["test_nuts3"]
        fold_designs[test_nuts3] = fold_summary
        design_rows.append(
            {
                "test_nuts3": test_nuts3,
                "n_test_rows": test_candidate["n_test_rows"],
                "test_share": test_candidate["test_share"],
                "test_normalized_wasserstein_squared_sum": test_candidate[
                    "test_normalized_wasserstein_squared_sum"
                ],
                "fold_normalized_wasserstein_squared_sum": (
                    fold_wasserstein_sum
                ),
                "test_size_penalty": test_candidate["test_size_penalty"],
                "fold_size_penalty": fold_size_penalty_value,
                "fold_size_valid": bool(fold_summary["size_valid"].all()),
                "size_valid_fold_assignments": valid_partition_count,
                "min_fold_rows": int(fold_summary["n_rows"].min()),
                "max_fold_rows": int(fold_summary["n_rows"].max()),
                "full_split_objective": full_objective,
            }
        )

    split_designs = (
        pd.DataFrame(design_rows)
        .sort_values(["full_split_objective", "test_nuts3"])
        .reset_index(drop=True)
    )
    selected_design = split_designs.iloc[0]
    selected_test_codes = code_list(selected_design["test_nuts3"].split(", "))
    selected_fold_summary = fold_designs[selected_design["test_nuts3"]].copy()
    return (
        split_designs,
        selected_design,
        selected_test_codes,
        selected_fold_summary,
    )
