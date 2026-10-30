from sklearn.ensemble import HistGradientBoostingRegressor
from .base import ModelSpec, regression_pipeline


class HistogramBoostingSpec(ModelSpec):
    name = "histogram_boosting"

    def build_pipeline(self, *, seed):
        # Spatial CV selects max_iter; do not create a random early-stopping split.
        return regression_pipeline(HistGradientBoostingRegressor(
            random_state=seed, early_stopping=False))

    def search_space(self):
        return {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_iter": [100, 200, 400],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__min_samples_leaf": [10, 20, 40],
            "model__l2_regularization": [0.0, 1.0, 10.0],
        }
