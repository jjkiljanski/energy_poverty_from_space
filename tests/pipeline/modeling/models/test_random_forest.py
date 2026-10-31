import numpy as np
from sklearn.ensemble import RandomForestRegressor

from pipeline.modeling.models.random_forest import RandomForestSpec


def test_predictions_and_importances_match_direct_forest(dataset):
    X = dataset.frame.loc[:35, ["x", "z", "constant", "empty"]]
    y = dataset.frame.loc[:35, "y"]
    # Independent preprocessing reference: retain x/z, impute training medians, no scaling.
    reference_X = X[["x", "z"]].fillna(X[["x", "z"]].median())
    reference = RandomForestRegressor(n_estimators=8, max_depth=3, random_state=19,
                                      n_jobs=1, bootstrap=True).fit(reference_X, y)
    spec = RandomForestSpec()
    pipeline = spec.build_pipeline(seed=19).set_params(
        model__n_estimators=8, model__max_depth=3).fit(X, y)

    np.testing.assert_allclose(pipeline.predict(X), reference.predict(reference_X))
    assert "scale" not in pipeline.named_steps
    assert pipeline.named_steps["model"].n_jobs == 1
    diagnostic = spec.diagnostics(pipeline)
    assert diagnostic.feature.tolist() == ["x", "z"]
    assert diagnostic.diagnostic.eq("impurity_importance").all()
    np.testing.assert_allclose(diagnostic.value, reference.feature_importances_)
