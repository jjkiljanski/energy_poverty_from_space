import pytest

from pipeline.modeling.targets import DEFAULT_TARGETS, TargetSpec, resolve_targets


@pytest.mark.parametrize("key", ["", "../target", "a/b", "a\\b", "EPVI heating"])
def test_keys_cannot_be_unsafe_artifact_paths(key):
    with pytest.raises(ValueError, match="Target key"):
        TargetSpec(key, "Source column")


def test_target_requires_a_source_column():
    with pytest.raises(ValueError, match="column"):
        TargetSpec("new_target", "")


def test_default_targets_have_unique_keys_and_columns():
    assert len({t.key for t in DEFAULT_TARGETS}) == len(DEFAULT_TARGETS)
    assert len({t.column for t in DEFAULT_TARGETS}) == len(DEFAULT_TARGETS)


def test_target_groups_resolve_to_eight_explicit_source_columns():
    assert len(resolve_targets("base")) == 2
    assert len(resolve_targets("additional")) == 6
    assert resolve_targets("all") == DEFAULT_TARGETS
    assert resolve_targets("epg_cooling") == (TargetSpec("epg_cooling", "epg_cooling"),)
    assert all(t.key == t.column for t in DEFAULT_TARGETS)
