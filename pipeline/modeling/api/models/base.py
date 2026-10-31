"""Small model recipes; tuning and evaluation live in the shared runner."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class UsableColumns(TransformerMixin, BaseEstimator):
    """Learn nonempty, nonconstant columns on each fitting partition only."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_ = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
        if not self.columns_:
            raise ValueError("No varying, nonempty predictors in this training partition.")
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        return X.loc[:, self.columns_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")
        return np.asarray(self.columns_, dtype=object)


def regression_pipeline(estimator, *, scale=False):
    steps = [("columns", UsableColumns()), ("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline([*steps, ("model", estimator)])


class ModelSpec(ABC):
    name: str
    search_kind = "random"

    @abstractmethod
    def build_pipeline(self, *, seed):
        """Return a fresh sklearn pipeline for one target."""

    @abstractmethod
    def search_space(self):
        """Return a finite sklearn parameter grid (pipeline parameter names)."""

    def diagnostics(self, fitted_pipeline):
        return pd.DataFrame(columns=["feature", "diagnostic", "value"])
