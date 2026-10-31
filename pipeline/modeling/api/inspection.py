"""Select saved artifacts for notebooks without loading or running estimators."""

import json
from pathlib import Path

import pandas as pd

from .artifacts import TABLES, load_run, provenance
from .data import load_project_data
from .targets import TargetSpec


def select_artifacts(run_dirs, *, role="inner_oof", models=None, targets=None,
                     selections=None, model_ids=None):
    """Return tables for explicit runs and filters; None means all available values.

    Every selected model must have the requested role. Contexts and runs remain
    separate, including when the same model ID occurs in multiple executions.
    """
    directories = [Path(p).resolve() for p in run_dirs]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("Choose nonempty, unique run directories.")
    runs = [load_run(p) for p in directories]
    if any(run.manifest["schema_version"] != 2 for run in runs):
        raise ValueError("These notebooks require schema-2 nested runs.")
    pairs = pd.DataFrame([
        {**pair, "run_id": run.manifest["run_id"], "run_dir": str(run.run_dir),
         "scope": run.manifest["config"]["scope"], "stage": run.manifest["config"]["stage"]}
        for run in runs for pair in run.manifest["pairs"]
    ])
    for column, values in [("model", models), ("target", targets),
                           ("selection", selections), ("model_id", model_ids)]:
        if values is not None:
            if isinstance(values, str):
                values = [values]
            if not values or not set(values) <= set(pairs[column]):
                raise ValueError(f"Unavailable or empty {column} selection: {values}")
            pairs = pairs.loc[pairs[column].isin(values)]
    if pairs.empty:
        raise ValueError("No models match the requested filters.")
    keys = ["model", "target", "selection"]
    tables = {name: [] for name in TABLES}
    for run in runs:
        selected = pairs.loc[pairs.run_dir.eq(str(run.run_dir)), [*keys, "model_id", "run_id"]]
        if selected.empty:
            continue
        for name in TABLES:
            table = getattr(run, name).merge(selected, on=keys, how="inner", validate="many_to_one")
            if "prediction_role" in table:
                table = table.loc[table.prediction_role.eq(role)]
            tables[name].append(table)
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    expected = set(zip(pairs.run_id, pairs.model_id))
    actual = set(zip(tables["predictions"].run_id, tables["predictions"].model_id))
    if expected != actual:
        raise ValueError(f"Selected models do not all have saved {role!r} predictions. Choose another role or run.")
    tables["models"] = pairs.reset_index(drop=True)
    return tables


def load_inspection_predictors(run_dir, *, config_path=None):
    """Load original predictor values only if they match the recorded modeling table.

    Inspection may use newer reporting code. It must not silently join revised
    input values onto historical residuals. No estimator is loaded or fitted.
    """
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    dataset = load_project_data(tuple(TargetSpec(**t) for t in config["targets"]),
        exclude_overseas=config["scope"] == "mainland_only", predictors=config["predictors"],
        config_path=config_path)
    if provenance(dataset)["modeling_table_sha256"] != manifest["provenance"]["modeling_table_sha256"]:
        raise ValueError("Current inputs differ from this run; restore the recorded inputs before predictor inspection.")
    return dataset.frame[["ID_norm", *dataset.predictors]].copy()
