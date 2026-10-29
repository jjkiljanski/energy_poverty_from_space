from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def _mainland_fold_by_nuts3(fold_summary_df: pd.DataFrame) -> dict[str, int]:
    """Expand each fold's comma-separated NUTS3 list into a region-to-fold mapping."""
    required = {"fold", "validation_nuts3"}
    missing = required - set(fold_summary_df.columns)
    if missing:
        raise ValueError(f"Mainland fold summary is missing columns: {sorted(missing)}")

    fold_by_nuts3: dict[str, int] = {}
    for row in fold_summary_df.itertuples(index=False):
        codes = [code.strip() for code in str(row.validation_nuts3).split(",")]
        for code in filter(None, codes):
            fold = int(row.fold)
            if code in fold_by_nuts3 and fold_by_nuts3[code] != fold:
                raise ValueError(f"NUTS3 region {code!r} is assigned to multiple folds.")
            fold_by_nuts3[code] = fold
    return fold_by_nuts3


def _overseas_fold_by_municipality(
    overseas_assignments_df: pd.DataFrame,
) -> dict[object, int]:
    """Build a municipality-to-fold mapping from overseas assignments."""
    required = {"municipality_id", "assigned_fold"}
    missing = required - set(overseas_assignments_df.columns)
    if missing:
        raise ValueError(
            f"Overseas assignments are missing columns: {sorted(missing)}"
        )

    mapping: dict[object, int] = {}
    for row in overseas_assignments_df.itertuples(index=False):
        fold = int(row.assigned_fold)
        if row.municipality_id in mapping and mapping[row.municipality_id] != fold:
            raise ValueError(
                f"Overseas municipality {row.municipality_id!r} is assigned to "
                "multiple folds."
            )
        mapping[row.municipality_id] = fold
    return mapping


def _training_fold_label(fold: object) -> object:
    """Format a zero-based fold number for the saved split table."""
    return f"Training CV fold {int(fold) + 1}" if pd.notna(fold) else pd.NA


def build_spatial_split_assignments(
    model_df: pd.DataFrame,
    mainland_fold_summary_df: pd.DataFrame,
    overseas_assignments_df: pd.DataFrame,
    selected_test_codes: Sequence[str],
    overseas_nuts3: Sequence[str],
    *,
    split_scope: str = "all_territories_with_mainland_fixed_test",
) -> pd.DataFrame:
    """Build and validate one fixed-test or training-fold assignment per model row."""
    assignment_columns = [
        "ID_norm",
        "ID",
        "NUTS3",
        "municipality_id",
        "parish_name",
    ]
    missing = set(assignment_columns) - set(model_df.columns)
    if missing:
        raise ValueError(f"Modeling table is missing columns: {sorted(missing)}")
    duplicate_ids = model_df["ID_norm"].duplicated(keep=False)
    if duplicate_ids.any():
        examples = model_df.loc[duplicate_ids, "ID_norm"].head(10).tolist()
        raise ValueError(f"Modeling table contains duplicate IDs; first examples: {examples}")

    test_codes = set(selected_test_codes)
    overseas_codes = set(overseas_nuts3)
    overlapping_codes = sorted(test_codes & overseas_codes)
    if overlapping_codes:
        raise ValueError(
            "Fixed-test regions must be mainland regions; overseas overlap: "
            f"{overlapping_codes}"
        )

    fold_by_mainland_nuts3 = _mainland_fold_by_nuts3(mainland_fold_summary_df)
    fold_by_overseas_municipality = _overseas_fold_by_municipality(
        overseas_assignments_df
    )

    assignments = model_df[assignment_columns].copy()
    assignments["split_scope"] = split_scope
    assignments["split_role"] = "training"
    assignments["cv_fold"] = pd.NA
    assignments["split_label"] = pd.NA

    is_test = assignments["NUTS3"].isin(test_codes)
    is_overseas = assignments["NUTS3"].isin(overseas_codes)
    is_mainland_training = ~is_test & ~is_overseas

    assignments.loc[is_test, "split_role"] = "fixed_test"
    assignments.loc[is_test, "split_label"] = "Fixed test"

    assignments.loc[is_mainland_training, "cv_fold"] = assignments.loc[
        is_mainland_training, "NUTS3"
    ].map(fold_by_mainland_nuts3)
    assignments.loc[is_overseas, "cv_fold"] = assignments.loc[
        is_overseas, "municipality_id"
    ].map(fold_by_overseas_municipality)

    is_training = assignments["split_role"].eq("training")
    assignments.loc[is_training, "split_label"] = assignments.loc[
        is_training, "cv_fold"
    ].map(_training_fold_label)

    missing_labels = assignments["split_label"].isna()
    if missing_labels.any():
        examples = assignments.loc[missing_labels, "ID_norm"].head(10).tolist()
        raise ValueError(
            f"Split assignment is missing labels for {int(missing_labels.sum())} rows; "
            f"first IDs: {examples}"
        )
    missing_training_folds = is_training & assignments["cv_fold"].isna()
    if missing_training_folds.any():
        examples = assignments.loc[
            missing_training_folds, "ID_norm"
        ].head(10).tolist()
        raise ValueError(
            "Training split assignment is missing folds for "
            f"{int(missing_training_folds.sum())} rows; first IDs: {examples}"
        )

    assignments["cv_fold"] = assignments["cv_fold"].astype("Int64")
    return assignments
