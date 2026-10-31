import numpy as np
import pandas as pd
import pytest
from pipeline.modeling.models.base import UsableColumns
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from pipeline.modeling.models.base import regression_pipeline


def test_column_filter_learns_each_training_partition_only(dataset):
    fit = dataset.frame.iloc[:24].copy()
    validation = dataset.frame.iloc[24:].copy()
    fit["z"] = 2
    selector = UsableColumns().fit(fit[list(dataset.predictors)])
    assert selector.columns_ == ["x"]
    assert list(selector.transform(validation).columns) == ["x"]


def test_transform_preserves_learned_column_order_and_does_not_refit():
    selector = UsableColumns().fit(pd.DataFrame({"a": [1, 2], "b": [4, 5], "constant": [0, 0]}))
    validation = pd.DataFrame({"b": [99], "constant": [7], "a": [np.nan]}, index=[12])
    transformed = selector.transform(validation)
    assert list(transformed) == ["a", "b"]
    assert transformed.index.tolist() == [12]
    assert pd.isna(transformed.loc[12, "a"])
    assert selector.get_feature_names_out().tolist() == ["a", "b"]


def test_filter_rejects_unfitted_use_and_no_usable_features():
    with pytest.raises(NotFittedError):
        UsableColumns().transform(pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(ValueError, match="No varying"):
        UsableColumns().fit(pd.DataFrame({"constant": [1, 1], "empty": [np.nan, np.nan]}))


def test_validation_extremes_cannot_change_imputation_or_scaling():
    X = pd.DataFrame({"x": [0., 2., np.nan, 10.], "test_only": [0., 0., 0., 0.]})
    pipeline = regression_pipeline(Ridge(), scale=True).fit(X, [0., 1., 2., 3.])
    validation = pd.DataFrame({"x": [np.nan, 1e9], "test_only": [100., 200.]})
    pipeline.predict(validation)
    assert pipeline.named_steps["columns"].columns_ == ["x"]
    np.testing.assert_allclose(pipeline.named_steps["impute"].statistics_, [2.])
    np.testing.assert_allclose(pipeline.named_steps["scale"].mean_, [3.5])
    np.testing.assert_allclose(pipeline.named_steps["scale"].scale_, [np.sqrt(14.75)])

