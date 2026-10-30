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


DEFAULT_TARGETS = (
    TargetSpec("epg_heating", "EPG heating"),
    TargetSpec("epg_cooling", "EPG cooling"),
    TargetSpec("aiam", "AIAM"),
    TargetSpec("epvi_heating", "EPVI heating"),
    TargetSpec("epvi_cooling", "EPVI cooling"),
)
