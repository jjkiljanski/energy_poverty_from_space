import numpy as np


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility wrapper for NumPy 1.x and 2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _hist_edges_from_minmax(vmin: float, vmax: float, n_bins: int) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # fallback: avoid zero-width bins
        vmin = 0.0 if not np.isfinite(vmin) else float(vmin)
        vmax = vmin + 1.0
    return np.linspace(vmin, vmax, n_bins + 1, dtype=np.float64)


def _quantile_from_hist(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    """
    counts: shape (n_bins,)
    edges: shape (n_bins+1,)
    Returns approximate quantile value.
    """
    total = counts.sum()
    if total <= 0:
        return np.nan
    target = q * total
    cum = np.cumsum(counts)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = max(0, min(idx, len(counts) - 1))
    # return left edge as threshold (conservative); could interpolate inside bin
    return float(edges[idx])


def _weighted_quantile_from_hist(wcounts: np.ndarray, edges: np.ndarray, q: float) -> float:
    return _quantile_from_hist(wcounts, edges, q)


def _gini_from_weighted_hist(wcounts: np.ndarray, edges: np.ndarray) -> float:
    """
    Approximate weighted Gini from histogram using bin midpoints.
    """
    W = wcounts.sum()
    if W <= 0:
        return np.nan

    mids = 0.5 * (edges[:-1] + edges[1:])
    # sort by mids already increasing
    w = wcounts.astype(np.float64)
    x = mids.astype(np.float64)

    mu = (w * x).sum() / W
    if mu == 0:
        return 0.0

    # Lorenz curve approximation
    cw = np.cumsum(w) / W
    cwx = np.cumsum(w * x) / (W * mu)

    # Area under Lorenz curve via trapezoid
    # prepend (0,0)
    cw0 = np.concatenate(([0.0], cw))
    cwx0 = np.concatenate(([0.0], cwx))
    area = _trapezoid(cwx0, cw0)
    g = 1.0 - 2.0 * area
    return float(g)
