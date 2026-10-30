import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .base import ModelSpec, regression_pipeline


class RandomForestSpec(ModelSpec):
    name = "random_forest"

    def build_pipeline(self, *, seed):
        # Parallelize CV fits, not both CV and individual forests.
        return regression_pipeline(RandomForestRegressor(
            random_state=seed, n_jobs=1, bootstrap=True))

    def search_space(self):
        return {
            "model__n_estimators": [300, 500, 800],
            "model__max_features": ["sqrt", 0.35, 0.5, 0.75, 1.0],
            "model__min_samples_leaf": [1, 2, 4, 8, 12],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__max_depth": [None, 8, 12, 18, 24, 32],
        }

    def diagnostics(self, fitted_pipeline):
        return pd.DataFrame({
            "feature": fitted_pipeline.named_steps["columns"].get_feature_names_out(),
            "diagnostic": "impurity_importance",
            "value": fitted_pipeline.named_steps["model"].feature_importances_,
        })
