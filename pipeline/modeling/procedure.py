"""Freeze one completed development run, including its selected outer-context models.

One checksum snapshot lives next to the run manifest. Evaluation never updates
the source run. Hashes detect accidental edits; they are not digital signatures.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

from .artifacts import TABLES, content_hash, fingerprint, provenance, write_json


def freeze_run(run_dir):
    directory = Path(run_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["schema_version"] != 2 or manifest["status"] != "complete"
            or manifest["config"].get("stage") != "develop"):
        raise ValueError("Only a complete nested development run can be frozen.")
    files = [directory / "manifest.json", directory / "split_assignments.csv"]
    for pair in manifest["pairs"]:
        pair_dir = (directory / pair["directory"]).resolve()
        if not pair_dir.is_relative_to(directory):
            raise ValueError("Pair path escapes run directory.")
        files.extend(pair_dir / name for name in [
            *(f"{table}.csv" for table in TABLES), "search_results.csv", "pipeline.joblib", "fit.json"])
    if any(not path.is_file() for path in files):
        raise ValueError("Cannot freeze a run with missing artifacts.")
    checksums = {p.relative_to(directory).as_posix(): fingerprint(p) for p in files}
    snapshot = {"schema_version": 1, "checksums": checksums}
    snapshot["freeze_id"] = content_hash(snapshot)
    snapshot["frozen_at"] = datetime.now(timezone.utc).isoformat()
    destination = directory / "frozen.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous["freeze_id"] != snapshot["freeze_id"]:
            raise ValueError("A frozen run was changed; create a new development run instead.")
        return destination
    write_json(destination, snapshot)
    return destination


def read_frozen(path, dataset=None):
    """Verify the snapshot and every artifact before any fitted model is loaded."""
    path = Path(path).resolve()
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    identity = {"schema_version": snapshot["schema_version"], "checksums": snapshot["checksums"]}
    if snapshot["schema_version"] != 1 or content_hash(identity) != snapshot["freeze_id"]:
        raise ValueError("Invalid freeze snapshot.")
    if "manifest.json" not in snapshot["checksums"]:
        raise ValueError("Freeze snapshot is missing the run manifest.")
    for relative, expected in snapshot["checksums"].items():
        artifact = (path.parent / relative).resolve()
        if not artifact.is_relative_to(path.parent):
            raise ValueError("Frozen artifact path escapes run directory.")
        if not artifact.is_file() or fingerprint(artifact) != expected:
            raise ValueError(f"Frozen artifact changed or missing: {relative}")
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["config"].get("stage") != "develop":
        raise ValueError("Freeze must reference a completed development run.")
    if dataset is not None:
        current, recorded = provenance(dataset), manifest["provenance"]
        for key in ("modeling_table_sha256", "framework_sha256", "packages"):
            if current[key] != recorded[key]:
                raise ValueError(f"Frozen procedure mismatch: {key}. Develop and freeze a new version.")
        current_sources = {k: v["sha256"] for k, v in current["sources"].items()}
        recorded_sources = {k: v["sha256"] for k, v in recorded["sources"].items()}
        if current_sources != recorded_sources:
            raise ValueError("Frozen procedure mismatch: input files or source definitions changed.")
        if list(dataset.predictors) != manifest["config"]["predictors"] or dataset.scope != manifest["config"]["scope"]:
            raise ValueError("Frozen predictor selection or scope changed.")
    return path.parent, manifest, snapshot
