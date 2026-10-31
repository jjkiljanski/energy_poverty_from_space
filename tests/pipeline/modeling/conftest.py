import numpy as np
import pandas as pd
import pytest
from pipeline.modeling.api import ModelingDataset, SearchConfig


@pytest.fixture
def dataset():
    # Dataset layout (48 rows; numeric values are reproducible with seed 42):
    # ID_norm range   NUTS3   spatial_split_role   spatial_cv_fold   test_only
    # 000000-000011   A       training             0                 0
    # 000012-000023   B       training             1                 0
    # 000024-000035   C       training             2                 0
    # 000036-000047   TEST    fixed_test           NaN               0, 1, ..., 11
    # Other columns:
    #   Predictors: x and z ~ N(0, 1); constant = 1; empty = NaN throughout.
    #   Targets: y = 3*x + noise; other = x**2 + noise (noise SD = 0.1).
    # Missing values: z at rows 0/14/28/40; other at rows 2/16/30/42.
    # Thus y has 36 train / 12 test rows; other has 33 train / 11 test rows.
    # constant/empty/test_only exercise removal of unusable training predictors.
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
    })


@pytest.fixture
def nested_dataset(dataset):
    # Five training folds plus the original fixed test: each outer fit has four
    # folds, each inner fit has three. Extra regions reuse synthetic values only.
    extra = dataset.frame.iloc[:24].copy()
    extra["ID_norm"] = [f"{i:06d}" for i in range(100, 124)]
    extra["NUTS3"] = np.repeat(["D", "E"], 12)
    extra["spatial_cv_fold"] += 3
    return ModelingDataset(pd.concat([dataset.frame, extra], ignore_index=True), dataset.predictors)

