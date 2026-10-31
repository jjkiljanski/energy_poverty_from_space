"""Explicit predictor/target schemas and frozen spatial assignments."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.utils.io import normalize_id, read_csv_robust
from pipeline.utils.epvi import normalize_epvi_ids, coerce_numeric_frame
from pipeline.utils.paths import load_paths, path_value, repo_data_path
from .targets import TargetSpec

METADATA = {"ID", "ID_norm", "NUTS3", "parish_name", "municipality_id",
            "spatial_split_role", "spatial_cv_fold"}


def require_unique_ids(frame, source):
    if frame["ID_norm"].isna().any() or frame["ID_norm"].duplicated().any():
        raise ValueError(f"{source}: missing or duplicate normalized parish IDs.")


@dataclass
class ModelingDataset:
    frame: pd.DataFrame
    predictors: tuple[str, ...]
    scope: str = "custom"
    sources: dict[str, str] = field(default_factory=dict)
    coverage: list[dict] = field(default_factory=list)

    def validate(self, targets):
        if not targets or len({t.key for t in targets}) != len(targets):
            raise ValueError("Targets must have nonempty, unique keys.")
        if len({t.column for t in targets}) != len(targets):
            raise ValueError("Target columns must be unique.")
        if not self.predictors or len(set(self.predictors)) != len(self.predictors):
            raise ValueError("Predictors must be nonempty and unique.")
        forbidden = set(self.predictors) & (METADATA | {t.column for t in targets})
        if forbidden:
            raise ValueError(f"Metadata/targets cannot be predictors: {sorted(forbidden)}")
        required = {"ID_norm", "NUTS3", "spatial_split_role", "spatial_cv_fold",
                    *self.predictors, *(t.column for t in targets)}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Missing required modeling columns: {sorted(missing)}")
        require_unique_ids(self.frame, "modeling table")
        if self.frame["NUTS3"].isna().any():
            raise ValueError("Missing NUTS3 region assignments.")
        roles = self.frame["spatial_split_role"]
        if set(roles) != {"training", "fixed_test"}:
            raise ValueError("Both training and fixed_test roles are required, with no other roles.")
        train = self.frame.loc[roles.eq("training")]
        test = self.frame.loc[roles.eq("fixed_test")]
        if set(train.NUTS3) & set(test.NUTS3):
            raise ValueError("NUTS3 regions overlap between training and fixed test.")
        folds = pd.to_numeric(train.spatial_cv_fold, errors="raise")
        if folds.isna().any() or not np.isfinite(folds).all() or (folds % 1 != 0).any():
            raise ValueError("Training CV folds must be finite integers.")
        if test.spatial_cv_fold.notna().any():
            raise ValueError("Fixed test rows cannot belong to training CV folds.")
        if train.groupby("NUTS3").spatial_cv_fold.nunique().gt(1).any():
            raise ValueError("A training NUTS3 region spans multiple validation folds.")
        for col in [*self.predictors, *(t.column for t in targets)]:
            values = pd.to_numeric(self.frame[col], errors="raise")
            if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"Infinite values in {col!r}; clean them explicitly before training.")

    def for_target(self, target: TargetSpec):
        eligible = self.frame.loc[self.frame[target.column].notna()]
        train = eligible.loc[eligible.spatial_split_role.eq("training")].reset_index(drop=True)
        test = eligible.loc[eligible.spatial_split_role.eq("fixed_test")].reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError(f"{target.key}: no eligible training or test rows.")
        expected = set(self.frame.loc[self.frame.spatial_split_role.eq("training"), "spatial_cv_fold"])
        if set(train.spatial_cv_fold) != expected:
            raise ValueError(f"{target.key}: target missingness empties a saved validation fold.")
        return train, test


def spatial_cv(train, *, require_variable_target=None):
    folds = train.spatial_cv_fold.to_numpy()
    splits, summary = [], []
    for fold in sorted(set(folds)):
        fit_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        if not len(fit_idx) or not len(val_idx):
            raise ValueError("Spatial CV needs nonempty fitting and validation partitions.")
        if require_variable_target is not None:
            y = train.iloc[val_idx][require_variable_target]
            if len(y) < 2 or y.nunique() < 2:
                raise ValueError("R2 tuning requires at least two, nonconstant outcomes in every fold.")
        splits.append((fit_idx, val_idx))
        summary.append({"fold": int(fold), "n_fit_rows": len(fit_idx),
                        "n_validation_rows": len(val_idx),
                        "validation_nuts3": ", ".join(sorted(train.iloc[val_idx].NUTS3.unique()))})
    if len(splits) < 2:
        raise ValueError("Spatial CV needs at least two folds.")
    return splits, pd.DataFrame(summary)


def load_project_data(targets, *, exclude_overseas=True, config_path=None, predictors=None):
    """Join basic admin + satellite predictors, target columns and saved splits.

    Target joins are left joins: a new target never changes another target's cohort.
    Parish names and spatial metadata come explicitly from the saved assignments.
    Existing private-table ID corrections and numeric parsing are reused.
    """
    paths = load_paths(config_path)
    sources = {
        "satellite": path_value(paths, "sat_data_predictors"),
        "administrative": repo_data_path("all_used_adm_indicators.csv"),
        "targets": path_value(paths, "epvi_targets_csv" if "epvi_targets_csv" in paths else "epvi_csv"),
        "admin_selection": repo_data_path("adm_data_split.json"),
        "split_assignments": repo_data_path("spatial_split_assignments.csv"),
    }
    tables = {k: read_csv_robust(sources[k]) for k in
              ("satellite", "administrative", "targets", "split_assignments")}
    raw_counts = {name: len(table) for name, table in tables.items()}
    # CSV exports can contain fully empty trailing records; count and discard only
    # these, never a populated record whose parish identifier is missing.
    tables = {name: table.dropna(how="all").copy() for name, table in tables.items()}
    target_cols = [t.column for t in targets]
    missing = set(target_cols) - set(tables["targets"].columns)
    if missing:
        raise ValueError(f"Requested target columns not found: {sorted(missing)}")
    for name, table in tables.items():
        table["ID_norm"] = (
            normalize_epvi_ids(table["ID"]) if name == "targets" else
            normalize_id(table["ID_norm"] if name == "split_assignments" else table["ID"]))
        require_unique_ids(table, name)
    admin_selection = json.loads(sources["admin_selection"].read_text(encoding="utf-8"))
    basic = admin_selection["basic"]
    missing = set(basic) - set(tables["administrative"].columns)
    if missing:
        raise ValueError(f"Requested basic administrative columns not found: {sorted(missing)}")
    satellite = [c for c in tables["satellite"] if c not in {"ID", "ID_norm"}]
    candidate_predictors = tuple([*satellite, *basic])
    prohibited = set(candidate_predictors) & (set(target_cols) | METADATA | set(admin_selection["detailed"]))
    if prohibited or len(set(candidate_predictors)) != len(candidate_predictors):
        raise ValueError(f"Overlapping or prohibited predictor names: {sorted(prohibited)}")
    predictors = candidate_predictors if predictors is None else tuple(predictors)
    if not set(predictors) <= set(candidate_predictors):
        raise ValueError("Selected predictors must belong to the satellite/basic-admin allowlist.")
    prepared = {}
    for name, cols in [("satellite", satellite), ("administrative", basic), ("targets", target_cols)]:
        raw = tables[name][["ID_norm", *cols]]
        numeric = coerce_numeric_frame(raw, skip={"ID_norm"})
        for col in cols:
            if (raw[col].notna() & numeric[col].isna()).any():
                raise ValueError(f"{name}.{col}: nonmissing values failed numeric conversion.")
        prepared[name] = numeric
    frame = prepared["satellite"].merge(prepared["administrative"], on="ID_norm", validate="one_to_one")
    frame = frame.merge(prepared["targets"], on="ID_norm", how="left", validate="one_to_one")
    assignments = tables["split_assignments"]
    assignment_cols = ["ID_norm", "NUTS3", "split_role", "cv_fold"]
    if "parish_name" in assignments:
        assignment_cols.append("parish_name")
    frame = frame.merge(assignments[assignment_cols], on="ID_norm", how="left", validate="one_to_one")
    if frame.NUTS3.isna().any():
        raise ValueError("Saved split assignments are missing modeling parish IDs.")
    frame = frame.rename(columns={"split_role": "spatial_split_role", "cv_fold": "spatial_cv_fold"})
    if exclude_overseas:
        frame = frame.loc[~frame.NUTS3.isin(["PT200", "PT300"])].copy()
    frame["ID"] = frame.ID_norm
    coverage = [{"source": k, "rows": len(v), "raw_rows": raw_counts[k],
                 "empty_records_removed": raw_counts[k] - len(v),
                 "rows_present_in_scope": int(v.ID_norm.isin(frame.ID_norm).sum())}
                for k, v in tables.items()]
    # Include source definitions as well as actual input files in run provenance.
    for key, path in {
        "source_definitions": repo_data_path("data_sources_manifest.json"),
        "index_definitions": Path(__file__).resolve().parents[1] / "1_index_construction/indices_manifest.json",
    }.items():
        if path.exists():
            sources[key] = path
    dataset = ModelingDataset(frame.reset_index(drop=True), predictors,
                              "mainland_only" if exclude_overseas else "all_territories",
                              {k: str(v) for k, v in sources.items()}, coverage)
    dataset.validate(targets)
    return dataset
