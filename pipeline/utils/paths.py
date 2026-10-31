"""Resolve named repository inputs and external artifacts from two path registries."""
from __future__ import annotations

import json
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Paths:
    """Resolve paths without creating directories or requiring artifacts to exist."""

    def __init__(self, external_config=None, internal_config=None, *, root=None):
        self.root = Path(root if root is not None else repo_root()).resolve()
        config_dir = self.root / 'pipeline/config'
        self.external_config = Path(external_config if external_config is not None else config_dir / 'external_paths.json').resolve()
        self.internal_config = Path(internal_config if internal_config is not None else config_dir / 'internal_paths.json').resolve()
        self._external = json.loads(self.external_config.read_text(encoding='utf-8'))
        self._internal = json.loads(self.internal_config.read_text(encoding='utf-8'))
        for name, value in self._external['roots'].items():
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f'External root {name!r} must be an absolute path')
        # Validate definitions eagerly; file existence is deliberately not checked.
        for name in self._internal:
            self.internal(name)
        for name in self._external['artifacts']:
            self.external(name)

    @staticmethod
    def _within(base: Path, relative: str, name: str) -> Path:
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f'Path {name!r} must be relative to its configured root')
        result = (base / relative).resolve()
        if not result.is_relative_to(base.resolve()):
            raise ValueError(f'Path {name!r} escapes its configured root')
        return result

    def internal(self, name: str) -> Path:
        if name not in self._internal:
            raise KeyError(f'Unknown internal path {name!r} in {self.internal_config}')
        return self._within(self.root, self._internal[name], name)

    def external(self, name: str) -> Path:
        if name not in self._external['artifacts']:
            raise KeyError(f'Unknown external path {name!r} in {self.external_config}')
        entry = self._external['artifacts'][name]
        root_name = entry['root']
        if root_name not in self._external['roots']:
            raise ValueError(f'Unknown external root {root_name!r} for {name!r}')
        return self._within(Path(self._external['roots'][root_name]), entry['path'], name)

    def records(self) -> list[dict[str, str]]:
        """Resolved locations for the workbook's path reference sheet."""
        return [
            {'scope': scope, 'key': name, 'path': str(resolve(name))}
            for scope, names, resolve in (
                ('internal', self._internal, self.internal),
                ('external', self._external['artifacts'], self.external),
            )
            for name in sorted(names)
        ]


def load_paths(config_path=None, *, internal_config_path=None, root=None) -> Paths:
    """Load the external registry (optional override) and internal registry."""
    return Paths(config_path, internal_config_path, root=root)
