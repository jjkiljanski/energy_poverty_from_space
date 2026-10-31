import numpy as np
import pandas as pd
import pytest
from pipeline.modeling.api.metrics import EvaluationSpec, evaluate


def test_metrics_handle_degenerate_targets():
    rows = pd.DataFrame(evaluate([1, 1], [1, 1], EvaluationSpec())).set_index("metric")
    assert rows.loc["rmse", "value"] == 0
    assert rows.loc["r2", "undefined_reason"] == "constant_observations"
    assert np.isnan(rows.loc["r2", "value"])


def test_metrics_match_hand_calculated_values_in_target_units():
    # Residuals: [0, -1, 2]; squared residual sum: 5; target squared deviation sum: 8.
    rows = pd.DataFrame(evaluate([1, 3, 5], [1, 4, 3], EvaluationSpec())).set_index("metric")
    assert rows.loc["rmse", "value"] == pytest.approx(np.sqrt(5 / 3))
    assert rows.loc["mae", "value"] == pytest.approx(1)
    assert rows.loc["r2", "value"] == pytest.approx(1 - 5 / 8)
    assert rows.loc["spearman", "value"] == pytest.approx(0.5)
    assert rows.n_rows.eq(3).all()
    assert rows.undefined_reason.isna().all()


@pytest.mark.parametrize("observed,predicted", [([], []), ([1, 2], [1]),
    ([1, np.inf], [1, 2]), ([1, 2], [1, np.nan]), ([[1], [2]], [[1], [2]])])
def test_metrics_reject_invalid_arrays(observed, predicted):
    with pytest.raises(ValueError):
        evaluate(observed, predicted, EvaluationSpec())


def test_single_observation_and_constant_predictions_have_explicit_reasons():
    singleton = pd.DataFrame(evaluate([2], [1], EvaluationSpec())).set_index("metric")
    assert singleton.loc["rmse", "value"] == 1
    assert singleton.loc["r2", "undefined_reason"] == "fewer_than_two_observations"
    constant = pd.DataFrame(evaluate([1, 2, 3], [2, 2, 2], EvaluationSpec())).set_index("metric")
    assert constant.loc["r2", "value"] == 0
    assert constant.loc["spearman", "undefined_reason"] == "constant_predictions"


@pytest.mark.parametrize("metrics,primary", [((), "rmse"), (("rmse", "rmse"), "rmse"),
    (("unknown",), "unknown"), (("mae",), "rmse"), (("spearman",), "spearman")])
def test_invalid_metric_contracts_fail_explicitly(metrics, primary):
    with pytest.raises(ValueError):
        EvaluationSpec(metrics=metrics, primary=primary)

