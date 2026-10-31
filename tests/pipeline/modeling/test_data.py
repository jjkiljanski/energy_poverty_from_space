import json
import numpy as np
import pandas as pd
import pytest
from pipeline.modeling.api import TargetSpec, preview_experiment
from pipeline.modeling.api.data import spatial_cv


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



def test_project_loader_explicit_schema_and_empty_records(tmp_path, monkeypatch):
    import pipeline.modeling.api.data as data
    target_path, satellite_path = tmp_path / "targets.csv", tmp_path / "satellite.csv"
    pd.DataFrame({"ID": [1, 2, 3], "y": [2, 4, 6], "new": [1, np.nan, 3],
                  "unrelated_metadata": ["a", "b", "c"]}).to_csv(target_path, index=False)
    pd.DataFrame({"ID": [1, 2, 3], "sat": [0.1, 0.2, 0.3]}).to_csv(satellite_path, index=False)
    pd.DataFrame({"ID": [1, 2, 3, np.nan], "basic": [3, 4, 5, np.nan],
                  "detailed": [9, 8, 7, np.nan]}).to_csv(tmp_path / "administrative.csv", index=False)
    (tmp_path / "adm_data_split.json").write_text(json.dumps({"basic": ["basic"], "detailed": ["detailed"]}))
    pd.DataFrame({"ID_norm": [1, 2, 3], "NUTS3": ["A", "B", "TEST"],
                  "split_role": ["training", "training", "fixed_test"],
                  "cv_fold": [0, 1, np.nan], "parish_name": ["Alpha", "Beta", "Gamma"]}).to_csv(
                      tmp_path / "spatial_split_assignments.csv", index=False)
    config = tmp_path / "external_paths.json"
    (tmp_path / "splits").mkdir()
    (tmp_path / "spatial_split_assignments.csv").rename(tmp_path / "splits/spatial_split_assignments.csv")
    config.write_text(json.dumps({"roots": {"fixture": str(tmp_path)}, "artifacts": {
        "epvi_targets": {"root": "fixture", "path": "targets.csv"},
        "split_assignments": {"root": "fixture", "path": "splits/spatial_split_assignments.csv"}}}))
    internal_config = tmp_path / "internal_paths.json"
    internal_config.write_text(json.dumps({
        "satellite_indicators": "satellite.csv", "administrative_indicators": "administrative.csv",
        "admin_groups": "adm_data_split.json", "source_manifest": "sources.json", "index_manifest": "indices.json"}))
    monkeypatch.setattr("pipeline.utils.paths.repo_root", lambda: tmp_path)
    target = TargetSpec("y", "y")
    loaded = data.load_project_data([target, TargetSpec("new", "new")], config_path=config, internal_config_path=internal_config)
    assert loaded.predictors == ("sat", "basic")
    assert loaded.frame.parish_name.tolist() == ["Alpha", "Beta", "Gamma"]
    assert loaded.for_target(target)[0].shape[0] == 2
    assert next(r for r in loaded.coverage if r["source"] == "administrative")["empty_records_removed"] == 1
    with pytest.raises(ValueError, match="Requested target columns"):
        data.load_project_data([TargetSpec("absent", "absent")], config_path=config, internal_config_path=internal_config)
    target_frame = pd.read_csv(target_path)
    target_frame["y"] = target_frame.y.astype(object)
    target_frame.loc[0, "y"] = "bad number"
    target_frame.to_csv(target_path, index=False)
    with pytest.raises(ValueError, match="numeric conversion"):
        data.load_project_data([target], config_path=config, internal_config_path=internal_config)


def test_spatial_folds_cover_eligible_rows_once_without_region_overlap(dataset):
    train, test = dataset.for_target(TargetSpec("other", "other"))
    # Use a non-contiguous index to ensure splits refer to row positions, not labels.
    train.index = np.arange(len(train)) * 7
    splits, summary = spatial_cv(train)
    validation_positions = []
    for fit, validation in splits:
        assert not set(fit) & set(validation)
        assert set(fit) | set(validation) == set(range(len(train)))
        assert not set(train.iloc[fit].NUTS3) & set(train.iloc[validation].NUTS3)
        validation_positions.extend(validation)
    assert sorted(validation_positions) == list(range(len(train)))
    assert summary.n_validation_rows.tolist() == [11, 11, 11]
    assert not set(train.NUTS3) & set(test.NUTS3)


def test_r2_tuning_rejects_constant_validation_outcomes(dataset):
    train, _ = dataset.for_target(TargetSpec("y", "y"))
    train.loc[train.spatial_cv_fold.eq(0), "y"] = 1
    with pytest.raises(ValueError, match="R2 tuning requires"):
        spatial_cv(train, require_variable_target="y")

