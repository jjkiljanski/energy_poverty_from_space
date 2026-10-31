import json
from dataclasses import replace
import pytest

from pipeline.modeling import __main__ as cli
from pipeline.modeling import TargetSpec, run_experiment, freeze_run


def test_dry_run_validates_configuration_without_training_or_output(dataset, tmp_path, monkeypatch, capsys):
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps({"models": ["ridge"], "targets": [{"key": "y", "column": "y"}]}))
    output = tmp_path / "runs"
    monkeypatch.setattr("sys.argv", ["modeling", "--config", str(config), "--dry-run", "--output-root", str(output)])
    monkeypatch.setattr(cli, "load_project_data", lambda *args, **kwargs: dataset)

    def forbidden_training(*args, **kwargs):
        pytest.fail("--dry-run must not train")

    monkeypatch.setattr(cli, "run_experiment", forbidden_training)
    cli.main()
    assert "y/ridge" in capsys.readouterr().out
    assert not output.exists()


def test_unknown_configuration_field_fails_before_loading_data(tmp_path, monkeypatch, capsys):
    config = tmp_path / "experiment.json"
    config.write_text('{"model_typo": ["ridge"]}')
    monkeypatch.setattr("sys.argv", ["modeling", "--config", str(config)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Unknown experiment configuration fields" in capsys.readouterr().err


@pytest.mark.parametrize("stage,selections", [
    ("outer_evaluate", ["all_outer_folds"]),
    ("final_evaluate", ["final_test"]),
])
def test_frozen_evaluation_dry_run(dataset, search, tmp_path, monkeypatch, capsys, stage, selections):
    developed = run_experiment(dataset, models=["ridge"], targets=[TargetSpec("y", "y")],
        search=replace(search, spaces={"ridge": search.spaces["ridge"]}), output_root=tmp_path)
    frozen = freeze_run(developed.run_dir)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(cli, "load_project_data", lambda *args, **kwargs: dataset)
    monkeypatch.setattr("sys.argv", ["modeling", "--stage", stage, "--frozen", str(frozen),
        "--dry-run", "--output-root", str(output), "--selections", *selections])

    def forbidden_evaluation(*args, **kwargs):
        pytest.fail("Evaluation dry-run must not create predictions")

    monkeypatch.setattr(cli, "evaluate_frozen", forbidden_evaluation)
    cli.main()
    assert stage in capsys.readouterr().out
    assert not output.exists()
