from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[2]


def load_paths(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load local path configuration.

    Preference order:
    1. explicit config_path argument
    2. pipeline/config/paths.local.json, ignored by git
    3. pipeline/config/paths.example.json, tracked with the current local defaults
    """
    root = repo_root()
    if config_path is not None:
        path = Path(config_path)
    else:
        local = root / "pipeline" / "config" / "paths.local.json"
        path = local if local.exists() else root / "pipeline" / "config" / "paths.example.json"

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(root)
    return cfg


def path_value(cfg: dict[str, Any], key: str) -> Path:
    """Return a config value as a Path, resolving repo-relative paths."""
    value = Path(cfg[key])
    if value.is_absolute():
        return value
    return repo_root() / value


def repo_data_path(cfg: dict[str, Any], *parts: str) -> Path:
    return path_value(cfg, "repo_data_dir").joinpath(*parts)
