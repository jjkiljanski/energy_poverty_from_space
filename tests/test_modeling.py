"""Synthetic integration checks for the shared spatial modeling contract."""

from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from pipeline.modeling import (
    EvaluationSpec, ModelingDataset, SearchConfig, TargetSpec,
    load_run, preview_experiment, run_experiment,
)
from pipeline.modeling.metrics import evaluate
from pipeline.modeling.models import get_model
from pipeline.modeling.models.base import UsableColumns
from pipeline.modeling.training import train_model


@pytest.fixture
def dataset():
    rng = np.random.default_rng(42)
    n = 48
    x = rng.normal(size=n)
    frame = pd.DataFrame({
        "ID_norm": [f"{i:06d}" for i in range(n)],
        "NUTS3": np.repeat(["A", "B", "C", "TEST"], 12),
        "spatial_split_role": ["training"] * 36 + ["fixed_test"] * 12,
        "spatial_cv_fold": np.repeat([0, 1, 2, np.nan], 12),
        "x": x, "z": rng.normal(size=n), "constant": 1.0, "empty": np.nan,
        "test_only": [0.0] * 36 + list(range(12)),
        "y": 3 * x + rng.normal(scale=0.1, size=n),
        "other": x ** 2 + rng.normal(scale=0.1, size=n),
    })
    frame.loc[[0, 14, 28, 40], "z"] = np.nan
    frame.loc[[2, 16, 30, 42], "other"] = np.nan
    return ModelingDataset(frame, ("x", "z", "constant", "empty", "test_only"))


@pytest.fixture
def search():
    return SearchConfig(n_iter=1, n_jobs=1, verbose=0, spaces={
        "ridge": {"model__alpha": [0.1, 1.0]},
        "random_forest": {"model__n_estimators": [5], "model__max_features": [1.0],
                          "model__min_samples_leaf": [1], "model__min_samples_split": [2],
                          "model__max_depth": [3]},
        "histogram_boosting": {"model__max_iter": [5], "model__max_leaf_nodes": [3],
                               "model__min_samples_leaf": [3], "model__learning_rate": [0.1],
                               "model__l2_regularization": [0.0]},
    })


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_end_to_end_all_models_and_target_missingness(dataset, search, tmp_path, n_jobs):
    search = replace(search, n_jobs=n_jobs)
    targets = [TargetSpec("y", "y"), TargetSpec("other", "other")]
    result = run_experiment(dataset, models=list(search.spaces), targets=targets,
                            search=search, output_root=tmp_path)
    assert result.manifest["status"] == "complete"
    assert len(result.manifest["pairs"]) == 8
    assert result.comparison_table().shape == (8, 4)
    assert pd.api.types.is_numeric_dtype(result.diagnostics.value)
    assert len(result.diagnostics.assign(magnitude=result.diagnostics.value.abs()).nlargest(2, "magnitude")) == 2
    predictions = result.predictions
    counts = predictions.groupby(["target", "model", "prediction_role"]).size()
    for name in ["training_mean", *search.spaces]:
        assert counts["y", name, "train_spatial_cv"] == 36
        assert counts["other", name, "train_spatial_cv"] == 33
        assert counts["y", name, "fixed_test"] == 12
        assert counts["other", name, "fixed_test"] == 11
    assert predictions.ID_norm.str.len().eq(6).all()
    assert np.allclose(predictions.residual, predictions.observed - predictions.predicted)
    for name in search.spaces:
        pipeline = joblib.load(result.run_dir / "y" / name / "pipeline.joblib")
        assert pipeline.named_steps["columns"].columns_ == ["x", "z"]
        test = dataset.frame.iloc[36:]
        saved = predictions.query("target == 'y' and model == @name and prediction_role == 'fixed_test'")
        np.testing.assert_allclose(pipeline.predict(test[list(dataset.predictors)]), saved.predicted)
    baseline = predictions.query("target == 'y' and model == 'training_mean' and prediction_role == 'train_spatial_cv'")
    train = dataset.frame.iloc[:36]
    for fold in range(3):
        expected = train.loc[train.spatial_cv_fold.ne(fold), "y"].mean()
        np.testing.assert_allclose(baseline.loc[baseline.spatial_cv_fold.eq(fold), "predicted"], expected)
    undefined = result.metrics.query("model == 'training_mean' and metric == 'spearman' and prediction_role == 'fixed_test'")
    assert undefined.value.isna().all()
    assert undefined.undefined_reason.eq("constant_predictions").all()


def test_test_outcomes_cannot_affect_training(dataset, search):
    target = TargetSpec("y", "y")
    train, _ = dataset.for_target(target)
    first = train_model(get_model("ridge"), train, predictors=dataset.predictors,
                        target=target, scope="custom", search=search)
    dataset.frame.loc[dataset.frame.spatial_split_role.eq("fixed_test"), "y"] = 100000
    train, _ = dataset.for_target(target)
    second = train_model(get_model("ridge"), train, predictors=dataset.predictors,
                         target=target, scope="custom", search=search)
    assert first.details["best_params"] == second.details["best_params"]
    np.testing.assert_array_equal(first.predictions.predicted, second.predictions.predicted)


def test_column_filter_learns_each_training_partition_only(dataset):
    fit = dataset.frame.iloc[:24].copy()
    validation = dataset.frame.iloc[24:].copy()
    fit["z"] = 2
    selector = UsableColumns().fit(fit[list(dataset.predictors)])
    assert selector.columns_ == ["x"]
    assert list(selector.transform(validation).columns) == ["x"]


@pytest.mark.parametrize("issue", ["missing_target", "duplicate_id", "region_overlap", "region_across_folds", "target_predictor", "empty_fold", "infinity"])
def test_invalid_experiments_fail_before_training(dataset, issue):
    targets = [TargetSpec("y", "y")]
    if issue == "missing_target":
        targets = [TargetSpec("missing", "missing")]
    elif issue == "duplicate_id":
        dataset.frame.loc[1, "ID_norm"] = dataset.frame.loc[0, "ID_norm"]
    elif issue == "region_overlap":
        dataset.frame.loc[36, "NUTS3"] = "A"
    elif issue == "region_across_folds":
        dataset.frame.loc[0, "spatial_cv_fold"] = 1
    elif issue == "target_predictor":
        dataset.predictors = (*dataset.predictors, "y")
    elif issue == "empty_fold":
        dataset.frame.loc[:11, "y"] = np.nan
    elif issue == "infinity":
        dataset.frame.loc[0, "x"] = np.inf
    with pytest.raises(ValueError):
        preview_experiment(dataset, models=["ridge"], targets=targets)


def test_metrics_handle_degenerate_targets():
    rows = pd.DataFrame(evaluate([1, 1], [1, 1], EvaluationSpec())).set_index("metric")
    assert rows.loc["rmse", "value"] == 0
    assert rows.loc["r2", "undefined_reason"] == "constant_observations"
    assert np.isnan(rows.loc["r2", "value"])


@pytest.mark.parametrize("dtype", [object, "string"])
def test_decimal_comma_parser_supports_object_and_string_columns(dtype):
    from pipeline.utils.epvi import coerce_numeric_frame, normalize_epvi_ids
    raw = pd.DataFrame({"value": pd.Series(["0,25", "1 234,5", None], dtype=dtype)})
    parsed = coerce_numeric_frame(raw, skip=set())
    np.testing.assert_allclose(parsed.value.iloc[:2], [0.25, 1234.5])
    assert pd.isna(parsed.value.iloc[2])
    assert normalize_epvi_ids(pd.Series(["031401", "031402"])).tolist() == ["031404", "031401"]


def test_project_loader_explicit_schema_and_empty_records(tmp_path, monkeypatch):
    import pipeline.modeling.data as data
    target_path, satellite_path = tmp_path / "targets.csv", tmp_path / "satellite.csv"
    pd.DataFrame({"ID": [1, 2, 3], "y": [2, 4, 6], "new": [1, np.nan, 3],
                  "unrelated_metadata": ["a", "b", "c"]}).to_csv(target_path, index=False)
    pd.DataFrame({"ID": [1, 2, 3], "sat": [0.1, 0.2, 0.3]}).to_csv(satellite_path, index=False)
    pd.DataFrame({"ID": [1, 2, 3, np.nan], "basic": [3, 4, 5, np.nan],
                  "detailed": [9, 8, 7, np.nan]}).to_csv(tmp_path / "all_used_adm_indicators.csv", index=False)
    (tmp_path / "adm_data_split.json").write_text(json.dumps({"basic": ["basic"], "detailed": ["detailed"]}))
    pd.DataFrame({"ID_norm": [1, 2, 3], "NUTS3": ["A", "B", "TEST"],
                  "split_role": ["training", "training", "fixed_test"],
                  "cv_fold": [0, 1, np.nan], "parish_name": ["Alpha", "Beta", "Gamma"]}).to_csv(
                      tmp_path / "spatial_split_assignments.csv", index=False)
    config = tmp_path / "paths.json"
    config.write_text(json.dumps({"sat_data_predictors": str(satellite_path), "epvi_csv": str(target_path)}))
    monkeypatch.setattr(data, "repo_data_path", lambda name: tmp_path / name)
    target = TargetSpec("y", "y")
    loaded = data.load_project_data([target, TargetSpec("new", "new")], config_path=config)
    assert loaded.predictors == ("sat", "basic")
    assert loaded.frame.parish_name.tolist() == ["Alpha", "Beta", "Gamma"]
    assert loaded.for_target(target)[0].shape[0] == 2
    assert next(r for r in loaded.coverage if r["source"] == "administrative")["empty_records_removed"] == 1
    with pytest.raises(ValueError, match="Requested target columns"):
        data.load_project_data([TargetSpec("absent", "absent")], config_path=config)
    target_frame = pd.read_csv(target_path)
    target_frame["y"] = target_frame.y.astype(object)
    target_frame.loc[0, "y"] = "bad number"
    target_frame.to_csv(target_path, index=False)
    with pytest.raises(ValueError, match="numeric conversion"):
        data.load_project_data([target], config_path=config)


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


def test_notebook_code_compiles_without_stale_outputs():
    root = Path(__file__).resolve().parents[1]
    for name in ["3_1_model_training.ipynb", "3_2_model_performance.ipynb"]:
        notebook = json.loads((root / "pipeline/3_epvi_prediction" / name).read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{name}:{index}", "exec")
                assert cell["outputs"] == []
                assert cell["execution_count"] is None
