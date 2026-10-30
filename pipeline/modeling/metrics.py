"""One evaluation contract for all regression models; losses are reported positive."""

from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

METRICS = ("rmse", "mae", "r2", "spearman")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def spearman(y_true, y_pred):
    if len(y_true) < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


@dataclass(frozen=True)
class EvaluationSpec:
    metrics: tuple[str, ...] = METRICS
    primary: str = "rmse"

    def __post_init__(self):
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("Specify a nonempty, unique metric list.")
        if set(self.metrics) - set(METRICS) or self.primary not in self.metrics:
            raise ValueError(f"Metrics must be from {METRICS}; primary must be reported.")
        if self.primary == "spearman":
            raise ValueError("Spearman is diagnostic only: constant predictions make it undefined.")

    @property
    def scorer(self):
        return {"rmse": make_scorer(rmse, greater_is_better=False),
                "mae": "neg_mean_absolute_error", "r2": "r2"}[self.primary]

    def reported_score(self, sklearn_score):
        return float(sklearn_score * (-1 if self.primary in {"rmse", "mae"} else 1))


def evaluate(y_true, y_pred, spec: EvaluationSpec):
    """Return long-form metric rows, explicitly explaining undefined values."""
    y, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y.shape != pred.shape or y.ndim != 1 or not len(y):
        raise ValueError("Metrics require equally sized nonempty 1-D arrays.")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require finite observations and predictions.")
    rows = []
    for metric in spec.metrics:
        reason = None
        if metric in {"r2", "spearman"} and len(y) < 2:
            reason = "fewer_than_two_observations"
        elif metric in {"r2", "spearman"} and np.ptp(y) == 0:
            reason = "constant_observations"
        elif metric == "spearman" and np.ptp(pred) == 0:
            reason = "constant_predictions"
        value = float("nan") if reason else float({
            "rmse": rmse, "mae": mean_absolute_error,
            "r2": r2_score, "spearman": spearman,
        }[metric](y, pred))
        rows.append({"metric": metric, "value": value, "n_rows": len(y),
                     "undefined_reason": reason})
    return rows
