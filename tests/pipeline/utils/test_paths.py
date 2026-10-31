"""Path registries separate portable repository locations from external roots."""
import json
from pathlib import Path

import pytest
from pipeline.utils.paths import Paths


def registry(tmp_path, *, internal=None, external=None):
    internal_config = tmp_path / 'internal_paths.json'
    external_config = tmp_path / 'external_paths.json'
    internal_config.write_text(json.dumps(internal if internal is not None else {
        'input': 'data/example.csv'}))
    external_config.write_text(json.dumps(external if external is not None else {
        'roots': {'outputs': str(tmp_path / 'external')},
        'artifacts': {'result': {'root': 'outputs', 'path': 'runs/result.csv'}}}))
    return Paths(external_config, internal_config, root=tmp_path)


def test_paths_resolve_without_existing_artifacts_or_side_effects(tmp_path, monkeypatch):
    paths = registry(tmp_path)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert paths.internal('input') == tmp_path / 'data/example.csv'
    assert paths.external('result') == tmp_path / 'external/runs/result.csv'
    assert not (tmp_path / 'data').exists()
    assert not (tmp_path / 'external').exists()
    assert {row['scope'] for row in paths.records()} == {'internal', 'external'}


def test_locations_change_only_in_registries(tmp_path):
    paths = registry(tmp_path)
    paths.internal_config.write_text(json.dumps({'input': 'prepared/renamed.csv'}))
    paths.external_config.write_text(json.dumps({
        'roots': {'outputs': str(tmp_path / 'new_external')},
        'artifacts': {'result': {'root': 'outputs', 'path': 'runs/result.csv'}}}))
    moved = Paths(paths.external_config, paths.internal_config, root=tmp_path)
    assert moved.internal('input') == tmp_path / 'prepared/renamed.csv'
    assert moved.external('result') == tmp_path / 'new_external/runs/result.csv'


@pytest.mark.parametrize('scope', ['internal', 'external'])
def test_unknown_names_identify_the_registry(tmp_path, scope):
    paths = registry(tmp_path)
    with pytest.raises(KeyError, match=f'Unknown {scope} path'):
        getattr(paths, scope)('missing')


@pytest.mark.parametrize('relative', ['../outside.csv', '/absolute.csv'])
def test_internal_paths_stay_inside_the_repo(tmp_path, relative):
    with pytest.raises(ValueError):
        registry(tmp_path, internal={'input': relative})


def test_external_paths_validate_roots_and_containment(tmp_path):
    for external in [
        {'roots': {'outputs': 'relative'}, 'artifacts': {}},
        {'roots': {}, 'artifacts': {'result': {'root': 'missing', 'path': 'result.csv'}}},
        {'roots': {'outputs': str(tmp_path)}, 'artifacts': {'result': {'root': 'outputs', 'path': '../outside.csv'}}},
    ]:
        with pytest.raises(ValueError):
            registry(tmp_path, external=external)
