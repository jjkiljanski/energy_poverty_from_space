"""Targets are configuration, independent of estimator implementations."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TargetSpec:
    key: str
    column: str
    label: str | None = None
    units: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.key):
            raise ValueError("Target key must contain lowercase letters, digits, '_' or '-'.")
        if not self.column:
            raise ValueError("Target column cannot be empty.")


BASE_TARGETS = tuple(TargetSpec(key, key) for key in ("epg_cooling", "epg_heating"))
ADDITIONAL_TARGETS = tuple(TargetSpec(key, key) for key in (
    "demand_cooling", "consumption_cooling", "gap_pct_cooling",
    "demand_heating", "consumption_heating", "gap_pct_heating",
))
DEFAULT_TARGETS = BASE_TARGETS + ADDITIONAL_TARGETS


def resolve_targets(selection="base"):
    """Accept a group, one registered key, or a list of keys/explicit target specs."""
    groups = {"base": BASE_TARGETS, "additional": ADDITIONAL_TARGETS, "all": DEFAULT_TARGETS}
    if isinstance(selection, str) and selection in groups:
        return groups[selection]
    items = [selection] if isinstance(selection, str) else list(selection)
    registered = {target.key: target for target in DEFAULT_TARGETS}
    targets = []
    for item in items:
        if isinstance(item, TargetSpec):
            targets.append(item)
        elif isinstance(item, dict):
            targets.append(TargetSpec(**item))
        elif item in registered:
            targets.append(registered[item])
        else:
            raise ValueError(f"Unknown target {item!r}.")
    if not targets or len({target.key for target in targets}) != len(targets):
        raise ValueError("Targets must be nonempty and unique.")
    return tuple(targets)
