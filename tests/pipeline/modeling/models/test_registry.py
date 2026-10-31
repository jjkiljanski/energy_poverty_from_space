import pytest
from sklearn.base import clone

from pipeline.modeling.models.registry import MODEL_REGISTRY, get_model


@pytest.mark.parametrize("name", ["ridge", "random_forest"])
def test_recipes_build_independent_cloneable_pipelines(name):
    spec = get_model(name)
    first, second = spec.build_pipeline(seed=17), spec.build_pipeline(seed=17)
    assert spec.name == name
    for step in first.named_steps:
        assert first.named_steps[step] is not second.named_steps[step]
    assert type(clone(first).named_steps["model"]) is type(first.named_steps["model"])
    space = spec.search_space()
    assert space and all(len(values) > 0 for values in space.values())
    first.set_params(**{key: values[0] for key, values in space.items()})
    # Callers may override a search without mutating the registry's future recipes.
    space[next(iter(space))].clear()
    assert all(len(values) > 0 for values in spec.search_space().values())


def test_unknown_model_lists_available_choices():
    with pytest.raises(ValueError) as error:
        get_model("typo")
    assert all(name in str(error.value) for name in MODEL_REGISTRY)
