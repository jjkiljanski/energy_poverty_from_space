from dataclasses import replace
import json
import pytest

from pipeline.modeling.api import TargetSpec, run_experiment, freeze_run, evaluate_frozen


def test_freeze_is_repeatable_and_outer_evaluation_never_retunes(dataset, search, tmp_path, monkeypatch):
    development = run_experiment(dataset, models=["ridge"], targets=[TargetSpec("y", "y")],
        selections=["holdout_A"], search=replace(search, spaces={"ridge": search.spaces["ridge"]}), output_root=tmp_path)
    frozen = freeze_run(development.run_dir)
    original = frozen.read_bytes()
    assert freeze_run(development.run_dir).read_bytes() == original

    def forbidden_training(*args, **kwargs):
        pytest.fail("Outer evaluation must reuse frozen fitted pipelines")

    monkeypatch.setattr("pipeline.modeling.api.training.train_model", forbidden_training)
    evaluated = evaluate_frozen(dataset, frozen, output_root=tmp_path)
    assert set(evaluated.predictions.prediction_role) == {"inner_oof", "train_in_sample", "outer_holdout"}
    assert not evaluated.predictions.NUTS3.eq("TEST").any()
    assert development.manifest["pairs"][1]["model_id"] == evaluated.manifest["pairs"][1]["model_id"]
    assert frozen.read_bytes() == original
    with pytest.raises(ValueError, match="every outer fold"):
        evaluated.outer_oof_predictions()


@pytest.mark.parametrize("change", ["artifact", "data", "predictors"])
def test_evaluation_rejects_changed_frozen_inputs(dataset, search, tmp_path, change):
    development = run_experiment(dataset, models=["ridge"], targets=[TargetSpec("y", "y")],
        selections=["holdout_A"], search=replace(search, spaces={"ridge": search.spaces["ridge"]}), output_root=tmp_path)
    frozen = freeze_run(development.run_dir)
    if change == "artifact":
        pair = development.manifest["pairs"][1]
        (development.run_dir / pair["directory"] / "fit.json").write_text("{}")
    elif change == "data":
        dataset.frame.loc[1, "y"] += 1
    else:
        dataset.predictors = ("x",)
    with pytest.raises(ValueError, match="changed|mismatch"):
        evaluate_frozen(dataset, frozen, output_root=tmp_path)


def test_evaluation_requires_freeze_and_rejects_procedure_overrides(dataset, search, tmp_path):
    with pytest.raises(ValueError, match="requires frozen"):
        run_experiment(dataset, models=["ridge"], targets=[TargetSpec("y", "y")],
                       stage="outer_evaluate", output_root=tmp_path)
    with pytest.raises(ValueError, match="forbids"):
        run_experiment(dataset, models=["ridge"], targets=[TargetSpec("y", "y")],
                       stage="final_evaluate", frozen="unused.json", search=search, output_root=tmp_path)


def test_incomplete_run_cannot_be_frozen(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 2, "status": "failed", "config": {"stage": "develop"}}))
    with pytest.raises(ValueError, match="complete nested development"):
        freeze_run(tmp_path)
