"""Named outer holdouts and their inner spatial CV partitions.

Names use letters A, B, ... for the sorted saved fold labels. No rows are
reassigned: five saved folds yield five four-fold inner training selections.
"""

from dataclasses import dataclass
from string import ascii_uppercase
import pandas as pd

from .data import spatial_cv

STAGES = ("develop", "outer_evaluate", "final_evaluate")


def outer_selections(dataset):
    training = dataset.frame.loc[dataset.frame.spatial_split_role.eq("training")]
    labels = sorted(training.spatial_cv_fold.dropna().unique())
    if len(labels) < 3:
        raise ValueError("Nested CV needs at least three saved training folds.")
    if len(labels) > len(ascii_uppercase):
        raise ValueError("At most 26 named outer folds are supported.")
    return {f"holdout_{ascii_uppercase[i]}": int(label) for i, label in enumerate(labels)}


def resolve_selections(dataset, selections="all_outer_folds", *, stage="develop"):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose from {STAGES}.")
    available = outer_selections(dataset)
    if selections == "all_outer_folds":
        selected = list(available)
    else:
        selected = [selections] if isinstance(selections, str) else list(selections)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Selections must be nonempty and unique.")
    allowed = {"final_test"} if stage == "final_evaluate" else set(available)
    if not set(selected) <= allowed:
        raise ValueError(f"Stage {stage!r} only permits selections {sorted(allowed)}.")
    return selected


@dataclass
class NestedSplit:
    selection: str
    outer_fold: int | None
    train: pd.DataFrame
    holdout: pd.DataFrame
    inner_cv: list
    inner_summary: pd.DataFrame


def select_split(dataset, target, selection, *, stage="develop", primary="rmse"):
    """Apply target missingness separately; never put fixed-test rows into CV.

    Development returns an empty holdout, so its caller cannot accidentally score
    outer observations through this object. Eligibility of that holdout is checked
    only when evaluation is explicitly requested.
    """
    resolve_selections(dataset, [selection], stage=stage)
    frame = dataset.frame
    train_all = frame.loc[frame.spatial_split_role.eq("training")]
    outer_fold = None if selection == "final_test" else outer_selections(dataset)[selection]
    if outer_fold is None:
        fitting = train_all
        held = frame.loc[frame.spatial_split_role.eq("fixed_test")]
    else:
        fitting = train_all.loc[train_all.spatial_cv_fold.ne(outer_fold)]
        held = train_all.loc[train_all.spatial_cv_fold.eq(outer_fold)]
    train = fitting.dropna(subset=[target.column]).reset_index(drop=True)
    if set(train.spatial_cv_fold) != set(fitting.spatial_cv_fold):
        raise ValueError(f"{target.key}/{selection}: target missingness empties an inner fold.")
    inner_cv, summary = spatial_cv(
        train, require_variable_target=target.column if primary == "r2" else None)
    holdout = held.iloc[:0].copy() if stage == "develop" else held.dropna(subset=[target.column]).reset_index(drop=True)
    if stage != "develop" and holdout.empty:
        raise ValueError(f"{target.key}/{selection}: no eligible holdout outcomes.")
    return NestedSplit(selection, outer_fold, train, holdout, inner_cv, summary)
