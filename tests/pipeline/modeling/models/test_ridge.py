import numpy as np
import pandas as pd

from pipeline.modeling.api.models.ridge import RidgeSpec


def test_predictions_and_coefficients_match_closed_form_ridge():
    # Missing x is imputed to 2; each feature is standardized with population SD.
    X = pd.DataFrame({"x": [0., 2., np.nan, 4.], "z": [1., 0., 3., 2.], "constant": 7.})
    y = np.array([0., 2., 1., 5.])
    imputed = np.array([[0., 1.], [2., 0.], [2., 3.], [4., 2.]])
    standardized = (imputed - imputed.mean(axis=0)) / imputed.std(axis=0)
    coefficients = np.linalg.solve(standardized.T @ standardized + 2 * np.eye(2),
                                   standardized.T @ (y - y.mean()))

    spec = RidgeSpec()
    pipeline = spec.build_pipeline(seed=42).set_params(model__alpha=2).fit(X, y)

    np.testing.assert_allclose(pipeline.predict(X), standardized @ coefficients + y.mean())
    diagnostic = spec.diagnostics(pipeline)
    assert diagnostic.feature.tolist() == ["x", "z"]
    assert diagnostic.diagnostic.eq("coefficient_per_predictor_standard_deviation").all()
    np.testing.assert_allclose(diagnostic.value, coefficients)
