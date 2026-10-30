import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .base import ModelSpec, regression_pipeline


class RidgeSpec(ModelSpec):
    name = "ridge"
    search_kind = "grid"

    def build_pipeline(self, *, seed):
        return regression_pipeline(Ridge(), scale=True)

    def search_space(self):
        return {"model__alpha": np.logspace(-4, 6, 21).tolist()}

    def diagnostics(self, fitted_pipeline):
        return pd.DataFrame({
            "feature": fitted_pipeline.named_steps["columns"].get_feature_names_out(),
            "diagnostic": "coefficient_per_predictor_standard_deviation",
            "value": fitted_pipeline.named_steps["model"].coef_,
        })
