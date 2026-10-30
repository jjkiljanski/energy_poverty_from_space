"""One immutable run directory; completed pairs survive interrupted experiments."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import uuid

import joblib
import numpy as np
import pandas as pd

OOF_METHOD = "spatial_oof_with_globally_selected_hyperparameters"
OOF_NOTE = (
    "Spatial OOF predictions exclude each observation's fold from model fitting, but use "
    "hyperparameters previously selected using all training folds. They are not independent "
    "of hyperparameter selection and may yield optimistic performance estimates. These "
    "predictions and residuals are diagnostics; fixed-test results provide held-out assessment."
)
TABLES = ("metrics", "predictions", "folds", "fold_metrics", "regional_metrics", "diagnostics")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(dataset):
    root = Path(__file__).resolve().parents[2]
    git = {}
    for key, args in [("commit", ["rev-parse", "HEAD"]),
                      ("status", ["status", "--porcelain"])]:
        try:
            git[key] = subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root,
                stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            git[key] = "unavailable"
    return {
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
        "git": git,
        "sources": {k: {"path": v, "sha256": fingerprint(v)} for k, v in dataset.sources.items()},
        "modeling_table_sha256": hashlib.sha256(
            dataset.frame.to_csv(index=False).encode("utf-8")).hexdigest(),
        "framework_sha256": {str(p.relative_to(root)): fingerprint(p)
                             for p in sorted([*Path(__file__).parent.rglob("*.py"),
                                              root / "pipeline/utils/epvi.py",
                                              root / "pipeline/utils/io.py",
                                              root / "pipeline/utils/paths.py"])},
    }


def create_run(output_root, config, dataset):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
    directory = Path(output_root) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "run_id": run_id, "status": "running",
                "oof_method": OOF_METHOD, "oof_note": OOF_NOTE,
                "config": config, "provenance": provenance(dataset), "pairs": []}
    dataset.frame[["ID_norm", "NUTS3", "spatial_split_role", "spatial_cv_fold"]].to_csv(
        directory / "split_assignments.csv", index=False)
    write_json(directory / "manifest.json", manifest)
    return directory, manifest


def save_pair(directory, manifest, result):
    relative = f"{result.target.key}/{result.model_name}"
    pair_dir = directory / relative
    pair_dir.mkdir(parents=True, exist_ok=False)
    for name in TABLES:
        getattr(result, name).to_csv(pair_dir / f"{name}.csv", index=False)
    result.search_results.to_csv(pair_dir / "search_results.csv", index=False)
    joblib.dump(result.pipeline, pair_dir / "pipeline.joblib")
    write_json(pair_dir / "fit.json", result.details)
    # Publish a pair only once all its artifacts have been written.
    manifest["pairs"].append({"target": result.target.key, "model": result.model_name,
                              "directory": relative, "status": "complete"})
    write_json(directory / "manifest.json", manifest)


@dataclass
class RunResults:
    run_dir: Path
    manifest: dict
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    folds: pd.DataFrame
    fold_metrics: pd.DataFrame
    regional_metrics: pd.DataFrame
    diagnostics: pd.DataFrame

    def comparison_table(self, role="fixed_test"):
        rows = self.metrics.loc[self.metrics.prediction_role.eq(role)]
        return rows.pivot(index=["target", "model"], columns="metric", values="value")


def load_run(run_dir, *, allow_incomplete=False):
    """Load an explicit run, never independently select 'latest' artifact files.

    Reading tables does not deserialize fitted estimators. Load pipeline.joblib
    separately only for artifacts you trust.
    """
    directory = Path(run_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported run schema.")
    if manifest["status"] != "complete" and not allow_incomplete:
        raise ValueError("Run is incomplete; use allow_incomplete=True to inspect completed pairs.")
    tables = {name: [] for name in TABLES}
    for pair in manifest["pairs"]:
        pair_dir = (directory / pair["directory"]).resolve()
        if not pair_dir.is_relative_to(directory.resolve()):
            raise ValueError("Pair path escapes the run directory.")
        for name in TABLES:
            tables[name].append(pd.read_csv(pair_dir / f"{name}.csv", dtype={"ID_norm": str}))
    if not manifest["pairs"]:
        raise ValueError("Run has no completed model-target pairs.")
    combined = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    # Empty diagnostic tables (baseline/boosting) can promote numeric columns to
    # object during concat, notably on pandas 3. Restore the declared value type.
    for table in combined.values():
        if "value" in table:
            table["value"] = pd.to_numeric(table["value"], errors="raise")
    return RunResults(directory, manifest, **combined)
