from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[2]


def load_paths(config_path: str | Path | None = None) -> dict[str, Path]:
    """
    Load local path configuration.

    Defaults to pipeline/config/paths.json.
    """
    root = repo_root()
    path = Path(config_path) if config_path is not None else root / "pipeline" / "config" / "paths.json"

    with path.open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    cfg = {key: Path(value) for key, value in raw_cfg.items()}
    cfg["_config_path"] = path
    cfg["_repo_root"] = root
    return cfg


def path_value(cfg: Mapping[str, str | Path], key: str) -> Path:
    """Return a config value as a Path, resolving repo-relative paths."""
    value = Path(cfg[key])
    if value.is_absolute():
        return value
    return repo_root() / value


def repo_data_path(*parts: str) -> Path:
    repo_data_dir = repo_root() / Path("data")
    return repo_data_dir.joinpath(*parts)
