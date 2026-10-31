from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from pipeline.modeling import TargetSpec, run_experiment, load_run
from pipeline.modeling.artifacts import write_json


def test_failed_run_retains_completed_pairs(dataset, search, tmp_path, monkeypatch):
    import pipeline.modeling.training as training
    original = training.train_model

    def fail_model(spec, *args, **kwargs):
        if spec is not None:
            raise RuntimeError("simulated failure")
        return original(spec, *args, **kwargs)

    monkeypatch.setattr(training, "train_model", fail_model)
    with pytest.raises(RuntimeError, match="simulated failure"):
        run_experiment(dataset, models=["ridge"], targets=[TargetSpec("y", "y")],
                       search=replace(search, spaces={"ridge": search.spaces["ridge"]}), output_root=tmp_path)
    run_dir = next(tmp_path.iterdir())
    with pytest.raises(ValueError, match="incomplete"):
        load_run(run_dir)
    recovered = load_run(run_dir, allow_incomplete=True)
    assert recovered.manifest["status"] == "failed"
    assert list(recovered.predictions.model.unique()) == ["training_mean"]


def test_failed_model_write_never_publishes_partial_pair(dataset, search, tmp_path, monkeypatch):
    import pipeline.modeling.artifacts as artifacts
    original_dump = artifacts.joblib.dump

    def fail_ridge_write(model, filename):
        if Path(filename).parent.name.startswith("ridge-"):
            raise OSError("simulated disk failure")
        return original_dump(model, filename)

    monkeypatch.setattr(artifacts.joblib, "dump", fail_ridge_write)
    with pytest.raises(OSError, match="simulated disk failure"):
        run_experiment(dataset, models=["ridge"], targets=[TargetSpec("y", "y")],
                       search=replace(search, spaces={"ridge": search.spaces["ridge"]}), output_root=tmp_path)
    run_dir = next(tmp_path.iterdir())
    assert list(run_dir.glob("ridge-*/predictions.csv"))  # Failure really happened mid-save.
    recovered = load_run(run_dir, allow_incomplete=True)
    assert recovered.manifest["status"] == "failed"
    assert [pair["model"] for pair in recovered.manifest["pairs"]] == ["training_mean"]
    assert set(recovered.predictions.model) == {"training_mean"}


def test_json_artifacts_use_portable_values_and_preserve_previous_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "fit.json"
    write_json(path, {"score": np.float64(np.nan), "params": np.array([1, 2]), "path": Path("relative")})
    assert json.loads(path.read_text()) == {"score": None, "params": [1, 2], "path": "relative"}
    previous = path.read_bytes()

    def fail_replace(*args, **kwargs):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        write_json(path, {"new": "value"})
    assert path.read_bytes() == previous

