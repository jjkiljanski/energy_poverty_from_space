from dataclasses import replace
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from pipeline.modeling.api import TargetSpec, run_experiment, freeze_run, evaluate_frozen
from pipeline.modeling.api import EvaluationSpec, SearchConfig, preview_experiment
from pipeline.modeling.api.models import get_model
from pipeline.modeling.api.training import train_model, evaluate_model
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



@pytest.mark.parametrize("n_jobs", [1, 2])
def test_nested_lifecycle_preserves_cohorts_and_serialized_predictions(dataset, search, tmp_path, n_jobs):
    search = replace(search, n_jobs=n_jobs)
    targets = [TargetSpec("y", "y"), TargetSpec("other", "other")]
    developed = run_experiment(dataset, models=list(search.spaces), targets=targets,
                               search=search, output_root=tmp_path)
    assert len(developed.manifest["pairs"]) == 18  # 3 contexts x 2 targets x (2 models + baseline).
    assert set(developed.predictions.prediction_role) == {"inner_oof", "train_in_sample"}
    assert not developed.predictions.NUTS3.eq("TEST").any()
    assert developed.predictions.ID_norm.str.len().eq(6).all()
    for pair in developed.manifest["pairs"]:
        rows = developed.predictions
        rows = rows.loc[rows.target.eq(pair["target"]) & rows.model.eq(pair["model"])
                        & rows.selection.eq(pair["selection"]) & rows.prediction_role.eq("inner_oof")]
        assert len(rows) == (24 if pair["target"] == "y" else 22)
        assert np.allclose(rows.residual, rows.observed - rows.predicted)
        if pair["model"] == "training_mean":
            outer_fold = ord(pair["selection"][-1]) - ord("A")
            eligible = dataset.frame.loc[dataset.frame.spatial_split_role.eq("training")
                & dataset.frame.spatial_cv_fold.ne(outer_fold)].dropna(subset=[pair["target"]])
            for fold, group in rows.groupby("spatial_cv_fold"):
                expected_mean = eligible.loc[eligible.spatial_cv_fold.ne(fold), pair["target"]].mean()
                np.testing.assert_allclose(group.predicted, expected_mean)
    frozen = freeze_run(developed.run_dir)
    outer = evaluate_frozen(dataset, frozen, output_root=tmp_path)
    oof = outer.outer_oof_predictions()
    assert oof.groupby(["target", "model"]).size().to_dict() == {
        (t, m): count for t, count in [("y", 36), ("other", 33)]
        for m in ["training_mean", *search.spaces]}
    assert not oof.NUTS3.eq("TEST").any()
    for pair in outer.manifest["pairs"]:
        rows = oof.loc[oof.target.eq(pair["target"]) & oof.model.eq(pair["model"])
                       & oof.selection.eq(pair["selection"])]
        fitted = joblib.load(outer.run_dir / pair["directory"] / "pipeline.joblib")
        X = dataset.frame.set_index("ID_norm").loc[rows.ID_norm, list(dataset.predictors)]
        np.testing.assert_allclose(fitted.predict(X), rows.predicted)
    final = evaluate_frozen(dataset, frozen, stage="final_evaluate", output_root=tmp_path)
    test = final.predictions.query("prediction_role == 'final_test'")
    assert test.NUTS3.eq("TEST").all()
    assert set(test.selection) == {"final_test"}
    assert final.comparison_table().shape == (6, 4)
    # The final search uses all saved training folds, not an outer winner's parameters.
    for pair in final.manifest["pairs"]:
        details = json.loads((final.run_dir / pair["directory"] / "fit.json").read_text())
        assert details["n_folds"] == 3
        assert details["n_train"] == (36 if pair["target"] == "y" else 33)


def test_outer_and_final_outcomes_cannot_affect_inner_tuning(dataset, search, tmp_path):
    target = TargetSpec("y", "y")
    first = run_experiment(dataset, models=list(search.spaces), targets=[target],
        search=search, selections=["holdout_A"], output_root=tmp_path)
    changed = replace(dataset, frame=dataset.frame.copy())
    held = changed.frame.spatial_cv_fold.eq(0) | changed.frame.spatial_split_role.eq("fixed_test")
    changed.frame.loc[held, "y"] += 1000
    changed.frame.loc[held, "x"] += 1000000
    second = run_experiment(changed, models=list(search.spaces), targets=[target],
        search=search, selections=["holdout_A"], output_root=tmp_path)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    for before, after in zip(first.manifest["pairs"], second.manifest["pairs"]):
        assert before["model_id"] == after["model_id"]
        a = joblib.load(first.run_dir / before["directory"] / "pipeline.joblib")
        b = joblib.load(second.run_dir / after["directory"] / "pipeline.joblib")
        if before["model"] != "training_mean":
            np.testing.assert_array_equal(a.named_steps["impute"].statistics_, b.named_steps["impute"].statistics_)


def test_sparse_target_extension_preserves_existing_nested_results(dataset, search, tmp_path):
    target = TargetSpec("y", "y")
    first = run_experiment(dataset, models=list(search.spaces), targets=[target],
        search=search, selections=["holdout_A"], output_root=tmp_path)
    extended = run_experiment(dataset, models=list(search.spaces), targets=[TargetSpec("other", "other"), target],
        search=search, selections=["holdout_A"], output_root=tmp_path)
    pd.testing.assert_frame_equal(first.predictions, extended.predictions.query("target == 'y'").reset_index(drop=True))
    assert [p["model_id"] for p in first.manifest["pairs"]] == [p["model_id"] for p in extended.manifest["pairs"] if p["target"] == "y"]


def test_five_fold_inner_residuals_match_explicit_reference(nested_dataset, search, tmp_path):
    developed = run_experiment(nested_dataset, models=["ridge"], targets=[TargetSpec("y", "y")],
        search=replace(search, spaces={"ridge": search.spaces["ridge"]}),
        selections=["holdout_A"], output_root=tmp_path)
    train = nested_dataset.frame.loc[nested_dataset.frame.spatial_split_role.eq("training")
                                     & nested_dataset.frame.spatial_cv_fold.ne(0)].reset_index(drop=True)
    X, y = train[["x", "z"]], train.y
    cv = [(np.flatnonzero(train.spatial_cv_fold.ne(f)), np.flatnonzero(train.spatial_cv_fold.eq(f))) for f in [1, 2, 3, 4]]
    reference = GridSearchCV(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge()),
        {"ridge__alpha": [0.1, 1.]}, cv=cv, scoring="neg_root_mean_squared_error").fit(X, y)
    expected = np.empty(len(y))
    for fit, validation in cv:
        assert len(fit) == 36 and len(validation) == 12
        expected[validation] = clone(reference.best_estimator_).fit(X.iloc[fit], y.iloc[fit]).predict(X.iloc[validation])
    actual = developed.predictions.query("model == 'ridge' and prediction_role == 'inner_oof'")
    assert actual.ID_norm.tolist() == train.ID_norm.tolist()
    np.testing.assert_allclose(actual.predicted, expected)
    frozen = freeze_run(developed.run_dir)
    evaluated = evaluate_frozen(nested_dataset, frozen, output_root=tmp_path)
    actual_outer = evaluated.predictions.query("model == 'ridge' and prediction_role == 'outer_holdout'")
    held = nested_dataset.frame.loc[nested_dataset.frame.spatial_cv_fold.eq(0)]
    np.testing.assert_allclose(actual_outer.predicted, reference.predict(held[["x", "z"]]))
    # Final tuning is a new five-fold search on all training data.
    final = evaluate_frozen(nested_dataset, frozen, stage="final_evaluate", output_root=tmp_path)
    full = nested_dataset.frame.query("spatial_split_role == 'training'").reset_index(drop=True)
    full_cv = [(np.flatnonzero(full.spatial_cv_fold.ne(f)), np.flatnonzero(full.spatial_cv_fold.eq(f))) for f in range(5)]
    final_reference = GridSearchCV(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge()),
        {"ridge__alpha": [0.1, 1.]}, cv=full_cv, scoring="neg_root_mean_squared_error").fit(full[["x", "z"]], full.y)
    fixed_test = nested_dataset.frame.query("spatial_split_role == 'fixed_test'")
    actual_final = final.predictions.query("model == 'ridge' and prediction_role == 'final_test'")
    np.testing.assert_allclose(actual_final.predicted, final_reference.predict(fixed_test[["x", "z"]]))


def test_preview_reports_inner_fit_budget(dataset, search):
    plan = preview_experiment(dataset, models=list(search.spaces), targets=[TargetSpec("y", "y")], search=search)
    assert len(plan) == 6
    assert plan.n_folds.eq(2).all()
    assert plan.n_holdout.eq(0).all()
    assert plan.total_fits_with_oof_and_refit.eq(plan.candidates * 2 + 3).all()
    with pytest.raises(ValueError, match="unselected models"):
        preview_experiment(dataset, models=["ridge"], targets=[TargetSpec("y", "y")], search=search)


@pytest.mark.parametrize("metric,scoring,sign", [
    ("rmse", "neg_root_mean_squared_error", -1),
    ("mae", "neg_mean_absolute_error", -1),
    ("r2", "r2", 1),
])
def test_tuning_and_oof_match_explicit_sklearn_reference(dataset, metric, scoring, sign):
    # No framework split, preprocessing, scoring or prediction helpers in the reference.
    train = dataset.frame.query("spatial_split_role == 'training'").reset_index(drop=True)
    X, y = train[["x", "z"]], train.y
    folds = train.spatial_cv_fold.to_numpy()
    cv = [(np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)) for fold in [0, 1, 2]]
    reference = GridSearchCV(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge()),
                             {"ridge__alpha": [0.1, 10000.]}, scoring=scoring, cv=cv).fit(X, y)
    expected_oof = np.empty(len(y))
    for fit, validation in cv:
        fold_model = clone(reference.best_estimator_).fit(X.iloc[fit], y.iloc[fit])
        expected_oof[validation] = fold_model.predict(X.iloc[validation])

    result = train_model(get_model("ridge"), train, predictors=dataset.predictors,
                         target=TargetSpec("y", "y"), scope="custom",
                         evaluation=EvaluationSpec(primary=metric),
                         search=SearchConfig(n_jobs=1, verbose=0, spaces={
                             "ridge": {"model__alpha": [0.1, 10000.]}}))

    # Require a meaningful score difference so reversing selection cannot pass by a tie.
    assert np.ptp(reference.cv_results_["mean_test_score"]) > 0.01
    assert result.details["best_params"]["model__alpha"] == reference.best_params_["ridge__alpha"]
    assert result.details["best_mean_cv_score"] == pytest.approx(sign * reference.best_score_)
    actual_oof = result.predictions.query("prediction_role == 'inner_oof'").predicted
    np.testing.assert_allclose(actual_oof, expected_oof)
    test = dataset.frame.query("spatial_split_role == 'fixed_test'")
    np.testing.assert_allclose(result.pipeline.predict(test[list(dataset.predictors)]),
                               reference.predict(test[["x", "z"]]))



def test_test_evaluation_never_refits(dataset, search, monkeypatch):
    target = TargetSpec("y", "y")
    train, test = dataset.for_target(target)
    result = train_model(get_model("ridge"), train, predictors=dataset.predictors,
                         target=target, scope="custom", search=search)

    def forbidden_fit(*args, **kwargs):
        pytest.fail("Test evaluation must not fit a pipeline")

    monkeypatch.setattr(result.pipeline, "fit", forbidden_fit)
    evaluate_model(result, test, predictors=dataset.predictors)
    assert len(result.predictions.query("prediction_role == 'outer_holdout'")) == len(test)

