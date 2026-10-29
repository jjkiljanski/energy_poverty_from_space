from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .map import FIXED_TEST_COLOR, TRAINING_FOLD_COLORS

DistributionSample = tuple[str, pd.DataFrame, str]


def _finite_values(df: pd.DataFrame, column: str) -> np.ndarray:
    """Return the finite numeric values from one DataFrame column."""
    values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _target_bins(
    target: str,
    samples: Sequence[pd.DataFrame],
    n_bins: int = 24,
) -> np.ndarray:
    """Build shared equal-width bins spanning a target across all samples."""
    sample_values = [_finite_values(sample, target) for sample in samples]
    nonempty_values = [values for values in sample_values if len(values)]
    if not nonempty_values:
        return np.linspace(0, 1, n_bins + 1)

    values = np.concatenate(nonempty_values)
    value_min, value_max = np.min(values), np.max(values)
    if np.isclose(value_min, value_max):
        padding = max(abs(value_min) * 0.05, 1.0)
        value_min -= padding
        value_max += padding
    return np.linspace(value_min, value_max, n_bins + 1)


def _plot_grouped_histogram(
    ax: Axes,
    samples: Sequence[DistributionSample],
    target: str,
    bins: np.ndarray,
) -> None:
    """Plot sample densities as side-by-side bars within shared bins."""
    bin_widths = np.diff(bins)
    group_width = bin_widths * 0.86
    bar_widths = group_width / len(samples)
    left_padding = (bin_widths - group_width) / 2

    for sample_index, (label, sample, color) in enumerate(samples):
        values = _finite_values(sample, target)
        if not len(values):
            continue
        counts, _ = np.histogram(values, bins=bins)
        density = (
            counts / (counts.sum() * bin_widths)
            if counts.sum()
            else counts.astype(float)
        )
        left = bins[:-1] + left_padding + sample_index * bar_widths
        ax.bar(
            left,
            density,
            width=bar_widths,
            align="edge",
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            label=f"{label} (n={len(sample)})",
        )

    ax.set_xlim(bins[0], bins[-1])
    ax.set_title(target)
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.2)


def _plot_distribution_diagnostics(
    samples: Sequence[DistributionSample],
    targets: Sequence[str],
    title: str,
    *,
    show: bool,
) -> Figure:
    """Plot comparable grouped density histograms for several target columns."""
    if len(targets) == 0:
        raise ValueError("At least one diagnostic target is required.")
    if len(samples) == 0:
        raise ValueError("At least one diagnostic sample is required.")

    fig, axes = plt.subplots(
        len(targets),
        1,
        figsize=(10, max(2.7 * len(targets), 4)),
        squeeze=False,
    )
    sample_frames = [sample for _, sample, _ in samples]
    for ax, target in zip(axes[:, 0], targets):
        bins = _target_bins(target, sample_frames)
        _plot_grouped_histogram(ax, samples, target, bins)

    axes[-1, 0].set_xlabel("Target value")
    axes[0, 0].legend(loc="upper right", fontsize=8, ncols=2)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_fold_distribution_diagnostics(
    reference_df: pd.DataFrame,
    fold_summary_df: pd.DataFrame,
    targets: Sequence[str],
    *,
    region_col: str = "NUTS3",
    reference_label: str = "Total mainland",
    title: str = "Mainland CV fold target distributions vs total mainland",
    show: bool = True,
) -> Figure:
    """Compare each spatial training fold's targets with the reference population."""
    fold_samples: list[DistributionSample] = []
    fold_rows = list(fold_summary_df.sort_values("fold").itertuples(index=False))
    if len(fold_rows) > len(TRAINING_FOLD_COLORS):
        raise ValueError(
            f"Received {len(fold_rows)} folds but only "
            f"{len(TRAINING_FOLD_COLORS)} fold colors are configured."
        )

    for color, row in zip(TRAINING_FOLD_COLORS, fold_rows):
        fold_codes = [
            code.strip()
            for code in str(row.validation_nuts3).split(",")
            if code.strip()
        ]
        fold_df = reference_df.loc[reference_df[region_col].isin(fold_codes)].copy()
        fold_samples.append((f"Fold {int(row.fold) + 1}", fold_df, color))

    samples = [(reference_label, reference_df, "#999999"), *fold_samples]
    return _plot_distribution_diagnostics(
        samples,
        list(targets),
        title,
        show=show,
    )


def plot_sample_distribution_diagnostics(
    training_pool_df: pd.DataFrame,
    fixed_test_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    overseas_df: pd.DataFrame,
    targets: Sequence[str],
    *,
    show: bool = True,
) -> Figure:
    """Compare target distributions across the selected spatial split samples."""
    samples = [
        ("Mainland training pool", training_pool_df, TRAINING_FOLD_COLORS[0]),
        ("Fixed mainland test", fixed_test_df, FIXED_TEST_COLOR),
        ("Total mainland", reference_df, "#999999"),
        ("Overseas territories", overseas_df, TRAINING_FOLD_COLORS[1]),
    ]
    return _plot_distribution_diagnostics(
        samples,
        list(targets),
        "Selected split target distributions",
        show=show,
    )
