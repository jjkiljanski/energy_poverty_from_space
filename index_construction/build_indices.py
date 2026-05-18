from __future__ import annotations

# ============================================================
# Purpose
# ------------------------------------------------------------
# Streaming construction of all parish (freguesia)-level indices
# defined in indices_manifest.json.
#
# Key constraints addressed:
# - Raster inputs can be huge (near 1GB). We avoid loading full
#   Portugal rasters into memory by processing in tiles/windows.
# - Freguesias can be split across tiles. We use per-zone
#   streaming accumulators and (for distributions) per-zone
#   histograms to obtain correct results regardless of tiling.
#
# Output:
# - A DataFrame indexed by freguesia identifier (string), with
#   one column per index output.
# ============================================================

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
import time
import logging

import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import transform as window_transform
from shapely.geometry import box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

import helpers  # your helpers.py (mosaic+alignment+iter_windows+read_window_arrays)
from distribution_compute_helpers import _hist_edges_from_minmax, _quantile_from_hist, _gini_from_weighted_hist, _weighted_quantile_from_hist # for computing distribution indices where a freguesia is split between multiple tiles.

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.paths import load_paths, path_value, repo_data_path  # noqa: E402


# distribution_compute_helpers contains histogram utilities used to:
# - compute zone/global quantile thresholds in a streaming context
# - compute per-zone distribution outputs (p10/p50/p90/gini) from histograms


# ============================================================
# Manifest helper functions (Age recoding)
# ------------------------------------------------------------
# These are helpers that can be referenced inside manifest
# expressions (via _safe_eval).
# ============================================================

def recode_age_midyear(age_class: np.ndarray, age_bins: Dict[str, Any]) -> np.ndarray:
    """
    Convert categorical age codes (e.g., from GHSL age) into a single
    representative mid-year (float).
    Cells in excluded bins remain NaN.

    Parameters
    ----------
    age_class : np.ndarray
        Integer-coded raster indicating age class per cell.
    age_bins : dict
        Mapping bin_name -> {codes: [...], mid_year: ..., exclude: bool}

    Returns
    -------
    np.ndarray (float32)
        Mid-year per cell or NaN.
    """
    mid = np.full(age_class.shape, np.nan, dtype=np.float32)
    for _, cfg in age_bins.items():
        codes = cfg.get("codes", [])
        mid_year = cfg.get("mid_year", None)
        exclude = cfg.get("exclude", False)
        if not codes:
            continue
        m = np.isin(age_class, np.array(codes))
        if exclude or mid_year is None:
            continue
        mid[m] = float(mid_year)
    return mid


def is_age_bin(age_class: np.ndarray, bin_name: str, age_bins: Dict[str, Any]) -> np.ndarray:
    """
    Convenience helper: return a 0/1 array indicating membership in a named age bin.
    Useful for "share of pre-1975" style indicators.

    Returns uint8 (0/1) to keep memory small on tiles.
    """
    cfg = age_bins.get(bin_name)
    if cfg is None:
        raise KeyError(f"Age bin '{bin_name}' not found in manifest helpers.age_bins")
    codes = cfg.get("codes", [])
    if not codes:
        return np.zeros(age_class.shape, dtype=np.uint8)
    return np.isin(age_class, np.array(codes)).astype(np.uint8)


# ============================================================
# Expression evaluation for manifest strings
# ------------------------------------------------------------
# The manifest expresses masks/values/vars as strings, e.g.:
#   "(res_volume > 0) && (pop > 0)"
# We evaluate these on each tile using eval() with a restricted
# global namespace.
#
# Two critical conversions:
# - JSON literals: null/true/false -> None/True/False
# - JSON boolean ops: &&, ||, ! -> np.logical_and/or/not
#   (done via a small top-level-splitting transformer that
#    respects parentheses nesting)
# ============================================================

def _split_top_level(expr: str, op: str) -> List[str]:
    """
    Split expr by operator op ('&&' or '||') only at top level
    (i.e., not inside parentheses).

    Example:
      "(a>0) && ((b+c)>0) && (d>0)"
    becomes:
      ["(a>0)", "((b+c)>0)", "(d>0)"]

    This avoids precedence bugs like:
      a > 0 & b > 0  (wrong in numpy without parentheses).
    """
    parts = []
    depth = 0
    i = 0
    last = 0
    n = len(expr)
    oplen = len(op)

    while i < n:
        ch = expr[i]
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue

        if depth == 0 and expr.startswith(op, i):
            parts.append(expr[last:i].strip())
            i += oplen
            last = i
            continue

        i += 1

    parts.append(expr[last:].strip())
    return parts

def _rewrite_bool_ops(expr: str) -> str:
    """
    Rewrite JSON-style boolean ops into NumPy-safe logical ops.
    Handles nested parentheses and chained expressions.

    - "A && B" -> np.logical_and(A, B)
    - "A || B" -> np.logical_or(A, B)
    - "!A"     -> np.logical_not(A)   (but not "!=")

    Notes
    -----
    We rewrite OR first (conceptually lower precedence), then AND.
    This preserves typical boolean expression semantics.
    """
    if expr is None:
        return expr

    # Rewrite standalone '!' to np.logical_not (avoid !=)
    expr = re.sub(r'(?<![=!<>])!(?!=)', 'np.logical_not', expr)

    # First rewrite OR (lower precedence than AND conceptually)
    or_parts = _split_top_level(expr, "||")
    if len(or_parts) > 1:
        rewritten = _rewrite_bool_ops(or_parts[0])
        for p in or_parts[1:]:
            rewritten = f"np.logical_or(({rewritten}),({_rewrite_bool_ops(p)}))"
        return rewritten

    # Then rewrite AND
    and_parts = _split_top_level(expr, "&&")
    if len(and_parts) > 1:
        rewritten = _rewrite_bool_ops(and_parts[0])
        for p in and_parts[1:]:
            rewritten = f"np.logical_and(({rewritten}),({_rewrite_bool_ops(p)}))"
        return rewritten

    return expr


def _normalize_literals(expr: str) -> str:
    """
    Convert JSON-ish literals to Python:
      null  -> None
      true  -> True
      false -> False
    Only replaces whole-word tokens.
    """
    expr = re.sub(r"\bnull\b", "None", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\btrue\b", "True", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bfalse\b", "False", expr, flags=re.IGNORECASE)
    return expr

def _safe_eval(expr: str, env: Dict[str, Any]) -> Any:
    """
    Evaluate a manifest expression string in a restricted namespace.

    Important:
    - We silence divide-by-zero warnings because value expressions may
      compute ratios before a mask is applied (the invalid cells are
      discarded later via mask and finite checks).
    - Only a controlled set of names is available in globals.
    """
    if expr is None:
        return None

    # 1) JSON → Python literals
    expr2 = _normalize_literals(expr)

    # 2) Boolean logic rewrite (&&, ||, !)
    expr2 = _rewrite_bool_ops(expr2)

    safe_globals = {
        "__builtins__": {},
        "np": np,
        "math": math,
        "recode_age_midyear": recode_age_midyear,
        "is_age_bin": is_age_bin
    }

    # Silence divide/invalid warnings that are expected before masking
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return eval(expr2, safe_globals, env)


# ============================================================
# String ID -> integer zone_code mapping for rasterization
# ------------------------------------------------------------
# Your freguesia IDs can be strings (e.g. "0302FA"), but raster
# label grids must be integers. We create a stable mapping:
#
#   freguesia_id (string) -> zone_code (int, 1..N)
#
# Then later map zone_code results back to string IDs.
# ============================================================

def build_zone_mapping(freg: gpd.GeoDataFrame, id_field: str) -> pd.DataFrame:
    """
    Create stable mapping table for rasterization.
    zone_code starts at 1 (0 reserved for "outside any polygon").
    """
    ids = freg[id_field].astype(str).values
    unique_ids = pd.Index(pd.unique(ids)).sort_values()
    return pd.DataFrame({
        id_field: unique_ids.astype(str),
        "zone_code": np.arange(1, len(unique_ids) + 1, dtype=np.int32)
    })


# ============================================================
# Rasterize labels per tile/window (fast via spatial index)
# ------------------------------------------------------------
# We cannot rasterize the entire Portugal label grid into memory
# (can be >1GB). Instead we rasterize labels only for the current
# window and only for polygons intersecting that window.
# ============================================================

def choose_supersampling_factor(
    pixel_size_m: float,
    target_subpixel_m: float = 250.0,
    min_ss: int = 2,
    max_ss: int = 32
) -> int:
    """
    Choose supersampling factor so that subpixels are about target_subpixel_m in size.
    Caps ss to keep runtime reasonable.
    """
    if pixel_size_m <= 0 or not np.isfinite(pixel_size_m):
        return min_ss
    ss = int(np.ceil(pixel_size_m / target_subpixel_m))
    ss = max(min_ss, ss)
    ss = min(max_ss, ss)
    return ss

def rasterize_labels_for_window(
    freg_proj: gpd.GeoDataFrame,
    sindex,
    code_field: str,
    aligned,
    window: rasterio.windows.Window
) -> np.ndarray:
    """
    Rasterize zone_code values only for polygons intersecting this window.

    Returns
    -------
    labels : np.ndarray (int32)
        Same shape as the window. 0 outside polygons.
    """
    # window bounds in raster CRS coordinates
    wb = rasterio.windows.bounds(window, aligned.transform)
    win_poly = box(*wb)

    # spatial filter
    cand_idx = list(sindex.intersection(win_poly.bounds))
    if not cand_idx:
        return np.zeros((int(window.height), int(window.width)), dtype=np.int32)

    subset = freg_proj.iloc[cand_idx]
    subset = subset[subset.intersects(win_poly)]
    if subset.empty:
        return np.zeros((int(window.height), int(window.width)), dtype=np.int32)

    shapes = [
        (geom, int(val))
        for geom, val in zip(subset.geometry, subset[code_field])
        if geom is not None and not geom.is_empty and pd.notnull(val)
    ]
    if not shapes:
        return np.zeros((int(window.height), int(window.width)), dtype=np.int32)

    t = window_transform(window, aligned.transform)
    out_shape = (int(window.height), int(window.width))

    labels = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=t,
        fill=0,
        dtype=np.int32,
        all_touched=False
    )
    return labels

def _pixel_size_m(transform: rasterio.Affine) -> float:
    """
    Returns approximate pixel size in meters (Mollweide / projected CRS).
    """
    return float(max(abs(transform.a), abs(transform.e)))


def rasterize_labels_for_window_supersampled(
    freg_proj: gpd.GeoDataFrame,
    sindex,
    code_field: str,
    aligned,
    window: rasterio.windows.Window,
    ss: int
) -> np.ndarray:
    """
    Rasterize zone codes on a supersampled grid:
      out_shape = (H*ss, W*ss)

    Each original pixel is subdivided into ss×ss subpixels.
    This allows fractional overlap computation.
    """
    if ss <= 1:
        raise ValueError("ss must be >= 2 for supersampling")

    # window bounds in raster CRS coordinates
    wb = rasterio.windows.bounds(window, aligned.transform)
    win_poly = box(*wb)

    # spatial filter candidates
    cand_idx = list(sindex.intersection(win_poly.bounds))
    if not cand_idx:
        return np.zeros((int(window.height) * ss, int(window.width) * ss), dtype=np.int32)

    subset = freg_proj.iloc[cand_idx]
    subset = subset[subset.intersects(win_poly)]
    if subset.empty:
        return np.zeros((int(window.height) * ss, int(window.width) * ss), dtype=np.int32)

    shapes = [
        (geom, int(val))
        for geom, val in zip(subset.geometry, subset[code_field])
        if geom is not None and not geom.is_empty and pd.notnull(val)
    ]
    if not shapes:
        return np.zeros((int(window.height) * ss, int(window.width) * ss), dtype=np.int32)

    # Window transform at native resolution
    t = window_transform(window, aligned.transform)

    # Create a supersampled transform (pixel size divided by ss)
    # Affine(a, b, c, d, e, f) where a is pixel width, e is pixel height (negative)
    t_hi = rasterio.Affine(t.a / ss, t.b, t.c, t.d, t.e / ss, t.f)

    out_shape_hi = (int(window.height) * ss, int(window.width) * ss)

    labels_hi = rasterize(
        shapes=shapes,
        out_shape=out_shape_hi,
        transform=t_hi,
        fill=0,
        dtype=np.int32,
        all_touched=False
    )
    return labels_hi


def iter_zone_fractions_from_supersampled(labels_hi: np.ndarray, H: int, W: int, ss: int):
    """
    Yield (zone_code, frac_grid) for each zone present in supersampled labels.

    frac_grid has shape (H, W), values in [0,1], representing area overlap fraction.
    """
    # reshape into blocks: (H, ss, W, ss)
    blocks = labels_hi.reshape(H, ss, W, ss)

    zones = np.unique(labels_hi)
    zones = zones[zones > 0]  # ignore background
    denom = float(ss * ss)

    for z in zones:
        # count subpixels that belong to zone z in each parent pixel
        cnt = (blocks == z).sum(axis=(1, 3)).astype(np.float32)
        frac = cnt / denom
        yield int(z), frac


# ============================================================
# Streaming accumulators for reducers (exact)
# ------------------------------------------------------------
# We update per-zone summary statistics tile-by-tile.
# This avoids holding full rasters in memory.
#
# For distribution-type indices we use histograms (below).
# ============================================================

class ZoneAccumulator:
    """
    Stores partial sums needed to compute common reducers.
    Uses dictionaries keyed by zone_code to avoid large arrays.
    """
    def __init__(self):
        self.sum = defaultdict(float)
        self.count = defaultdict(int)
        self.wsum = defaultdict(float)
        self.w = defaultdict(float)
        self.num = defaultdict(float)
        self.den = defaultdict(float)
        self.wcond = defaultdict(float)

    def update_mean(self, labels, value, mask):
        """Mean: accumulate sum and count per zone."""
        valid = mask & (labels > 0) & np.isfinite(value)
        if not np.any(valid):
            return
        z = labels[valid].ravel().astype(np.int32)
        v = value[valid].ravel().astype(np.float64)

        # pandas groupby is fast for aggregating by zone on each tile
        df = pd.DataFrame({"z": z, "v": v})
        g = df.groupby("z")["v"]
        s = g.sum()
        c = g.size()
        for k, vv in s.items():
            self.sum[int(k)] += float(vv)
        for k, cc in c.items():
            self.count[int(k)] += int(cc)

    def update_sum(self, labels, value, mask):
        """Sum: accumulate sum(value) per zone."""
        valid = mask & (labels > 0) & np.isfinite(value)
        if not np.any(valid):
            return
        z = labels[valid].ravel().astype(np.int32)
        v = value[valid].ravel().astype(np.float64)
        df = pd.DataFrame({"z": z, "v": v})
        s = df.groupby("z")["v"].sum()
        for k, vv in s.items():
            self.sum[int(k)] += float(vv)

    def update_weighted_mean(self, labels, value, weight, mask):
        """Weighted mean: accumulate sum(value*weight) and sum(weight)."""
        valid = mask & (labels > 0) & np.isfinite(value) & np.isfinite(weight) & (weight > 0)
        if not np.any(valid):
            return
        z = labels[valid].ravel().astype(np.int32)
        v = value[valid].ravel().astype(np.float64)
        w = weight[valid].ravel().astype(np.float64)
        df = pd.DataFrame({"z": z, "vw": v * w, "w": w})
        g = df.groupby("z")
        s_vw = g["vw"].sum()
        s_w = g["w"].sum()
        for k, vv in s_vw.items():
            self.wsum[int(k)] += float(vv)
        for k, ww in s_w.items():
            self.w[int(k)] += float(ww)

    def update_ratio(self, labels, numerator, denominator, mask):
        """
        Ratio-of-sums:
          sum(numerator)/sum(denominator)
        """
        valid = mask & (labels > 0) & np.isfinite(numerator) & np.isfinite(denominator)
        if not np.any(valid):
            return
        z = labels[valid].ravel().astype(np.int32)
        n = numerator[valid].ravel().astype(np.float64)
        d = denominator[valid].ravel().astype(np.float64)
        df = pd.DataFrame({"z": z, "n": n, "d": d})
        g = df.groupby("z")
        sn = g["n"].sum()
        sd = g["d"].sum()
        for k, vv in sn.items():
            self.num[int(k)] += float(vv)
        for k, vv in sd.items():
            self.den[int(k)] += float(vv)

    def update_weighted_share(self, labels, condition, weight, mask):
        """
        Weighted share:
          sum(weight where condition)/sum(weight)
        """
        valid = mask & (labels > 0) & np.isfinite(weight) & (weight > 0)
        if not np.any(valid):
            return
        z = labels[valid].ravel().astype(np.int32)
        w = weight[valid].ravel().astype(np.float64)
        c = condition[valid].ravel().astype(bool)
        df = pd.DataFrame({"z": z, "w": w, "wc": w * c.astype(np.float64)})
        g = df.groupby("z")
        sw = g["w"].sum()
        swc = g["wc"].sum()
        for k, vv in sw.items():
            self.w[int(k)] += float(vv)
        for k, vv in swc.items():
            self.wcond[int(k)] += float(vv)


# ============================================================
# Streaming (tile-based) index computation with 2-pass quantiles
# ------------------------------------------------------------
# Supports:
# - exact reducers: weighted_mean, mean, sum, ratio_of_sums, weighted_share
# - zone_quantile filter: 2-pass (build per-zone hist -> threshold -> apply)
# - global_quantile filter: 2-pass (build global hist -> threshold -> apply)
# - weighted_distribution: 2-pass (min/max -> per-zone weighted hist -> stats)
# ============================================================

def compute_index_streaming(
    index_def: Dict[str, Any],
    aligned,
    freg_proj: gpd.GeoDataFrame,
    sindex,
    constants: Dict[str, Any],
    age_bins: Dict[str, Any],
    tile_size: int = 2048,
    hist_bins: int = 256,
    fractional_threshold_m: float = 250.0,
    ss: int = 4
) -> pd.DataFrame:
    """
    Compute one index definition via streaming windows.

    Parameters
    ----------
    index_def : dict
        One index definition from indices_manifest.json.
    aligned : helpers.AlignedRasterSet
        Aligned VRTs for all bound inputs (already coarsened to a common grid).
    freg_proj : GeoDataFrame
        Freguesia polygons projected into aligned raster CRS, includes zone_code.
    sindex : spatial index
        Built from freg_proj for quick per-window polygon lookup.
    constants : dict
        Global constants (epsilon, thresholds, etc.).
    age_bins : dict
        Age bin configuration used by recode_age_midyear/is_age_bin.
    tile_size : int
        Window size in pixels; tune based on RAM (1024–4096).
    hist_bins : int
        Number of histogram bins for quantile thresholds and distributions.

    Returns
    -------
    DataFrame indexed by zone_code with output columns specified by manifest.
    """
    cell = index_def.get("cell", {}) or {}
    reduce_def = index_def.get("reduce", {}) or {}
    outputs = index_def.get("outputs", []) or []
    method = reduce_def["method"]

    filters = cell.get("filters", []) or []
    zone_q_filters = [f for f in filters if f["type"] == "zone_quantile"]
    global_q_filters = [f for f in filters if f["type"] == "global_quantile"]
    needs_distribution = (method == "weighted_distribution")

    # Decide whether to use fractional overlap allocation (supersampling)
    pix_m = _pixel_size_m(aligned.transform)

    # Use fractional overlap only above threshold
    use_fractional = (pix_m >= fractional_threshold_m)

    # Choose adaptive supersampling factor (only used if use_fractional=True)
    ss = choose_supersampling_factor(
        pixel_size_m=pix_m,
        target_subpixel_m=250.0,   # << you choose this
        min_ss=2,
        max_ss=32
    )

    # For debugging/logging (optional, but useful)
    logger.info(f"Pixel size ~{pix_m:.1f}m -> fractional overlap = {use_fractional}")

    # We need 2-pass if we have zone_quantile, or if we want global_quantile accurately, or distributions
    needs_pass1 = (len(zone_q_filters) > 0) or (len(global_q_filters) > 0) or needs_distribution

    # ---- PASS 0: prepare env-independent items
    max_zone = int(freg_proj["zone_code"].max())
    # We'll compute per-zone thresholds for zone_quantile filters
    zone_thresholds: Dict[Tuple[str, float], np.ndarray] = {}  # key=(var, q) -> thresholds array [zone_code]

    # Also for global_quantile filters: thresholds scalar by var,q
    global_thresholds: Dict[Tuple[str, float], float] = {}

    # For distribution: we build weighted hist per zone in pass2, but we need value min/max first
    value_min = np.inf
    value_max = -np.inf

    # For zone_quantile filter vars: min/max for consistent bin edges (global)
    filter_minmax: Dict[str, Tuple[float, float]] = {}

    # Helper to build env and compute base mask+vars for a window
    def build_env_and_base_mask(arrays: Dict[str, np.ndarray], labels: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Create the eval environment and compute the base mask (without quantile filters).
        This is called in both pass1 and pass2.
        """
        env: Dict[str, Any] = {}
        env.update(constants)
        env["age_bins"] = age_bins
        env.update(arrays)

        # cell.vars
        for name, expr in (cell.get("vars", {}) or {}).items():
            env[name] = _safe_eval(expr, env)

        # base mask (WITHOUT applying quantile filters)
        mask_expr = cell.get("mask", "true")
        if isinstance(mask_expr, str) and mask_expr.strip().lower() == "true":
            base_mask = np.ones(labels.shape, dtype=bool)
        else:
            base_mask = np.asarray(_safe_eval(mask_expr, env), dtype=bool)

        # In the previously implemented winner-takes-all mode we exclude outside pixels via labels>0.
        # In fractional mode, pixels can overlap zones even if no single "winner" exists,
        # so we do NOT apply labels>0 here. The overlap fractions handle zone membership.
        if not use_fractional:
            base_mask &= (labels > 0)
        return env, base_mask

    # ==========================================================
    # PASS 1: compute thresholds/hist minmax (if needed)
    # ==========================================================
    if needs_pass1:
        # We need to build min/max for:
        # - each zone_quantile filter var (for bins)
        # - each global_quantile filter var (for bins)
        # - distribution value expr (for bins)
        # And for zone_quantile thresholds: per-zone hist counts.

        # First subpass: compute global min/max for all needed vars (cheap, streaming)
        needed_filter_vars = list({f["var"] for f in (zone_q_filters + global_q_filters)})
        # For distribution: value expression itself
        value_expr = cell.get("value")

        # Track min/max for filter vars and value_expr
        for window in helpers.iter_windows(aligned, tile_size=tile_size):
            arrays = helpers.read_window_arrays(aligned, window, masked=True)
            labels = rasterize_labels_for_window(
                freg_proj=freg_proj, sindex=sindex, code_field="zone_code", aligned=aligned, window=window
            )
            if labels.max() == 0:
                continue

            env, base_mask = build_env_and_base_mask(arrays, labels)

            # filter vars min/max
            for vname in needed_filter_vars:
                var_arr = env.get(vname)
                if var_arr is None:
                    var_arr = np.asarray(_safe_eval(vname, env))
                else:
                    var_arr = np.asarray(var_arr)

                m = base_mask & np.isfinite(var_arr)
                if np.any(m):
                    vmin = float(np.nanmin(var_arr[m]))
                    vmax = float(np.nanmax(var_arr[m]))
                    cur = filter_minmax.get(vname, (np.inf, -np.inf))
                    filter_minmax[vname] = (min(cur[0], vmin), max(cur[1], vmax))

            # distribution value min/max
            if needs_distribution:
                val_arr = np.asarray(_safe_eval(value_expr, env), dtype=np.float32)
                m = base_mask & np.isfinite(val_arr)
                if np.any(m):
                    value_min = min(value_min, float(np.nanmin(val_arr[m])))
                    value_max = max(value_max, float(np.nanmax(val_arr[m])))

        # Build bin edges for each filter var + for distribution value
        filter_edges: Dict[str, np.ndarray] = {}
        for vname, (vmin, vmax) in filter_minmax.items():
            filter_edges[vname] = _hist_edges_from_minmax(vmin, vmax, hist_bins)

        value_edges = None
        if needs_distribution:
            value_edges = _hist_edges_from_minmax(value_min, value_max, hist_bins)

        # Second subpass: compute histograms
        # - zone_quantile: per-zone unweighted hist for that var
        # - global_quantile: global unweighted hist for that var
        zone_hist: Dict[str, np.ndarray] = {}
        for f in zone_q_filters:
            vname = f["var"]
            zone_hist[vname] = np.zeros((max_zone + 1, hist_bins), dtype=np.float64)

        global_hist: Dict[str, np.ndarray] = {}
        for f in global_q_filters:
            vname = f["var"]
            global_hist[vname] = np.zeros(hist_bins, dtype=np.float64)

        for window in helpers.iter_windows(aligned, tile_size=tile_size):
            arrays = helpers.read_window_arrays(aligned, window, masked=True)
            labels = rasterize_labels_for_window(
                freg_proj=freg_proj, sindex=sindex, code_field="zone_code", aligned=aligned, window=window
            )
            if labels.max() == 0:
                continue

            env, base_mask = build_env_and_base_mask(arrays, labels)

            # zone hist
            for f in zone_q_filters:
                vname = f["var"]
                edges = filter_edges[vname]
                var_arr = env.get(vname)
                if var_arr is None:
                    var_arr = np.asarray(_safe_eval(vname, env))
                else:
                    var_arr = np.asarray(var_arr)

                m = base_mask & np.isfinite(var_arr)
                if not np.any(m):
                    continue

                zz = labels[m].ravel().astype(np.int32)
                vv = var_arr[m].ravel().astype(np.float64)
                # bin index
                b = np.searchsorted(edges, vv, side="right") - 1
                b = np.clip(b, 0, hist_bins - 1)
                # accumulate counts per (zone, bin)
                # zone hist
                for f in zone_q_filters:
                    vname = f["var"]
                    edges = filter_edges[vname]

                    var_arr = env.get(vname)
                    if var_arr is None:
                        var_arr = np.asarray(_safe_eval(vname, env))
                    else:
                        var_arr = np.asarray(var_arr)

                    # ----- NEW: fractional zone histogram accumulation -----
                    if use_fractional:
                        labels_hi = rasterize_labels_for_window_supersampled(
                            freg_proj=freg_proj,
                            sindex=sindex,
                            code_field="zone_code",
                            aligned=aligned,
                            window=window,
                            ss=ss
                        )
                        H, W = int(window.height), int(window.width)

                        for z, frac in iter_zone_fractions_from_supersampled(labels_hi, H, W, ss):
                            m = base_mask & (frac > 0) & np.isfinite(var_arr)
                            if not np.any(m):
                                continue

                            vv = var_arr[m].ravel().astype(np.float64)
                            ww = frac[m].ravel().astype(np.float64)  # fractional area weights

                            b = np.searchsorted(edges, vv, side="right") - 1
                            b = np.clip(b, 0, hist_bins - 1)

                            # add fractional weights into histogram bins
                            np.add.at(zone_hist[vname][z], b, ww)

                    # ----- old behavior: winner-takes-all label grid -----
                    else:
                        m = base_mask & np.isfinite(var_arr)
                        if not np.any(m):
                            continue

                        zz = labels[m].ravel().astype(np.int32)
                        vv = var_arr[m].ravel().astype(np.float64)

                        b = np.searchsorted(edges, vv, side="right") - 1
                        b = np.clip(b, 0, hist_bins - 1)

                        np.add.at(zone_hist[vname], (zz, b), 1.0)

            # global hist
            for f in global_q_filters:
                vname = f["var"]
                edges = filter_edges[vname]
                var_arr = env.get(vname)
                if var_arr is None:
                    var_arr = np.asarray(_safe_eval(vname, env))
                else:
                    var_arr = np.asarray(var_arr)

                m = base_mask & np.isfinite(var_arr)
                if not np.any(m):
                    continue
                vv = var_arr[m].ravel().astype(np.float64)
                b = np.searchsorted(edges, vv, side="right") - 1
                b = np.clip(b, 0, hist_bins - 1)
                np.add.at(global_hist[vname], b, 1.0)

        # Compute thresholds
        # zone thresholds: for each (var,q)
        for f in zone_q_filters:
            vname = f["var"]
            q = float(f["q"])
            edges = filter_edges[vname]
            counts = zone_hist[vname]  # (zones, bins)
            thr = np.full(max_zone + 1, np.nan, dtype=np.float64)
            for z in range(1, max_zone + 1):
                thr[z] = _quantile_from_hist(counts[z], edges, q)
            zone_thresholds[(vname, q)] = thr

        # global thresholds
        for f in global_q_filters:
            vname = f["var"]
            q = float(f["q"])
            edges = filter_edges[vname]
            thr = _quantile_from_hist(global_hist[vname], edges, q)
            global_thresholds[(vname, q)] = thr

    # ==========================================================
    # PASS 2: apply filters and accumulate reducer statistics
    # ==========================================================
    acc = ZoneAccumulator()

    # For weighted_distribution: weighted histogram per zone
    wdist_hist = None
    value_edges = None
    if needs_distribution:
        # use same edges computed in pass1
        # (If no pass1, we’d need to compute min/max here; we force pass1 when distribution)
        # Rebuild edges from min/max stored above
        value_edges = _hist_edges_from_minmax(value_min, value_max, hist_bins)
        wdist_hist = np.zeros((max_zone + 1, hist_bins), dtype=np.float64)

    for window in helpers.iter_windows(aligned, tile_size=tile_size):
        arrays = helpers.read_window_arrays(aligned, window, masked=True)
        labels = rasterize_labels_for_window(
            freg_proj=freg_proj, sindex=sindex, code_field="zone_code", aligned=aligned, window=window
        )
        if labels.max() == 0:
            continue

        env, mask = build_env_and_base_mask(arrays, labels)

        # Apply quantile filters (now thresholded)
        for f in filters:
            ftype = f["type"]
            vname = f["var"]
            q = float(f["q"])
            op = f["op"]

            var_arr = env.get(vname)
            if var_arr is None:
                var_arr = np.asarray(_safe_eval(vname, env))
            else:
                var_arr = np.asarray(var_arr)

            if ftype == "global_quantile":
                thr = global_thresholds.get((vname, q), np.nan)
                if not np.isfinite(thr):
                    continue
                if op == ">":
                    mask &= (var_arr > thr)
                elif op == "<":
                    mask &= (var_arr < thr)
                else:
                    raise ValueError(f"Unknown op: {op}")

            elif ftype == "zone_quantile":
                thr_arr = zone_thresholds.get((vname, q))
                if thr_arr is None:
                    raise RuntimeError("zone_quantile thresholds missing (pass1 failed?)")
                # per-pixel threshold based on its zone
                thr_pix = thr_arr[labels]
                if op == ">":
                    mask &= (var_arr > thr_pix)
                elif op == "<":
                    mask &= (var_arr < thr_pix)
                else:
                    raise ValueError(f"Unknown op: {op}")

            else:
                raise ValueError(f"Unknown filter type: {ftype}")

        # value expression
        value = np.asarray(_safe_eval(cell.get("value"), env), dtype=np.float32)

        # ----------------------------------------------------------
        # Reduction updates (winner-takes-all OR fractional overlap)
        # ----------------------------------------------------------

        if use_fractional:
            # Supersampled zone membership for this tile
            labels_hi = rasterize_labels_for_window_supersampled(
                freg_proj=freg_proj,
                sindex=sindex,
                code_field="zone_code",
                aligned=aligned,
                window=window,
                ss=ss
            )
            H, W = int(window.height), int(window.width)

            # Evaluate auxiliary arrays once at pixel resolution
            # (we apply fractional weights per zone)
            if method in ("weighted_mean", "weighted_share", "weighted_distribution"):
                w_expr = reduce_def["weight"]
                w = env.get(w_expr)
                if w is None:
                    w = _safe_eval(w_expr, env)
                w = np.asarray(w, dtype=np.float32)
            else:
                w = None

            if method == "ratio_of_sums":
                num = np.asarray(_safe_eval(reduce_def["numerator"], env), dtype=np.float32)
                den = np.asarray(_safe_eval(reduce_def["denominator"], env), dtype=np.float32)

            if method == "weighted_share":
                cond = np.asarray(_safe_eval(reduce_def["condition"], env))

            # Loop over zones present in this tile with their per-pixel overlap fractions
            for z, frac in iter_zone_fractions_from_supersampled(labels_hi, H, W, ss):
                # Construct a per-zone label grid (int32)
                labels_z = np.where(frac > 0, z, 0).astype(np.int32)

                # Mask for pixels contributing to this zone
                mask_z = mask & (frac > 0)

                if method == "weighted_mean":
                    # Effective weight is reduced by overlap fraction
                    w_eff = w * frac
                    acc.update_weighted_mean(labels_z, value, w_eff, mask_z)

                elif method == "mean":
                    # Mean is area-weighted implicitly by splitting pixels; here we treat each pixel
                    # with weight=frac by converting mean into weighted_mean with weight=frac.
                    # Easiest: use weighted_mean where weight=frac.
                    acc.update_weighted_mean(labels_z, value, frac, mask_z)

                elif method == "sum":
                    # Sum should scale by overlap fraction
                    acc.update_sum(labels_z, value * frac, mask_z)

                elif method == "ratio_of_sums":
                    # Both numerator and denominator scale by overlap fraction
                    acc.update_ratio(labels_z, num * frac, den * frac, mask_z)

                elif method == "weighted_share":
                    # Total weights must also be fractionally allocated
                    w_eff = w * frac
                    acc.update_weighted_share(labels_z, cond, w_eff, mask_z)

                elif method == "weighted_distribution":
                    # Distribution weights incorporate overlap fraction
                    valid = mask_z & (labels_z > 0) & np.isfinite(value) & np.isfinite(w) & (w > 0)
                    if not np.any(valid):
                        continue

                    vv = value[valid].ravel().astype(np.float64)
                    ww = (w[valid] * frac[valid]).ravel().astype(np.float64)

                    b = np.searchsorted(value_edges, vv, side="right") - 1
                    b = np.clip(b, 0, hist_bins - 1)
                    np.add.at(wdist_hist[z], b, ww)

                else:
                    raise ValueError(f"Unknown reduce.method: {method}")

        else:
            # ------------------------
            # Winner-takes-all version
            # ------------------------
            if method == "weighted_mean":
                w_expr = reduce_def["weight"]
                w = env.get(w_expr)
                if w is None:
                    w = _safe_eval(w_expr, env)
                w = np.asarray(w, dtype=np.float32)
                acc.update_weighted_mean(labels, value, w, mask)

            elif method == "mean":
                acc.update_mean(labels, value, mask)

            elif method == "sum":
                acc.update_sum(labels, value, mask)

            elif method == "ratio_of_sums":
                num = np.asarray(_safe_eval(reduce_def["numerator"], env), dtype=np.float32)
                den = np.asarray(_safe_eval(reduce_def["denominator"], env), dtype=np.float32)
                acc.update_ratio(labels, num, den, mask)

            elif method == "weighted_share":
                w_expr = reduce_def["weight"]
                cond_expr = reduce_def["condition"]
                w = env.get(w_expr)
                if w is None:
                    w = _safe_eval(w_expr, env)
                w = np.asarray(w, dtype=np.float32)
                cond = np.asarray(_safe_eval(cond_expr, env))
                acc.update_weighted_share(labels, cond, w, mask)

            elif method == "weighted_distribution":
                w_expr = reduce_def["weight"]
                w = env.get(w_expr)
                if w is None:
                    w = _safe_eval(w_expr, env)
                w = np.asarray(w, dtype=np.float32)

                valid = mask & (labels > 0) & np.isfinite(value) & np.isfinite(w) & (w > 0)
                if not np.any(valid):
                    continue
                zz = labels[valid].ravel().astype(np.int32)
                vv = value[valid].ravel().astype(np.float64)
                ww = w[valid].ravel().astype(np.float64)

                b = np.searchsorted(value_edges, vv, side="right") - 1
                b = np.clip(b, 0, hist_bins - 1)
                np.add.at(wdist_hist, (zz, b), ww)

            else:
                raise ValueError(f"Unknown reduce.method: {method}")

    # ==========================================================
    # Finalize reducers into a zone_code-indexed DataFrame
    # ==========================================================
    zones = np.arange(1, max_zone + 1, dtype=np.int32)

    if method == "weighted_mean":
        col = outputs[0]["name"]
        out = np.full(max_zone + 1, np.nan, dtype=np.float64)
        for z in zones:
            if acc.w[z] > 0:
                out[z] = acc.wsum[z] / acc.w[z]
        return pd.DataFrame({col: out[1:]}, index=pd.Index(zones, name="zone_code"))

    if method == "mean":
        col = outputs[0]["name"]
        out = np.full(max_zone + 1, np.nan, dtype=np.float64)
        for z in zones:
            if acc.count[z] > 0:
                out[z] = acc.sum[z] / acc.count[z]
        return pd.DataFrame({col: out[1:]}, index=pd.Index(zones, name="zone_code"))

    if method == "sum":
        col = outputs[0]["name"]
        out = np.full(max_zone + 1, np.nan, dtype=np.float64)
        for z in zones:
            out[z] = acc.sum[z] if z in acc.sum else 0.0
        return pd.DataFrame({col: out[1:]}, index=pd.Index(zones, name="zone_code"))

    if method == "ratio_of_sums":
        col = outputs[0]["name"]
        out = np.full(max_zone + 1, np.nan, dtype=np.float64)
        for z in zones:
            den = acc.den[z]
            if den != 0:
                out[z] = acc.num[z] / den
        return pd.DataFrame({col: out[1:]}, index=pd.Index(zones, name="zone_code"))

    if method == "weighted_share":
        col = outputs[0]["name"]
        out = np.full(max_zone + 1, np.nan, dtype=np.float64)
        for z in zones:
            if acc.w[z] > 0:
                out[z] = acc.wcond[z] / acc.w[z]
        return pd.DataFrame({col: out[1:]}, index=pd.Index(zones, name="zone_code"))

    if method == "weighted_distribution":
        # outputs correspond to p10/p50/p90/gini in your manifest
        df = pd.DataFrame(index=pd.Index(zones, name="zone_code"))
        # map desired stats by suffix
        for out_spec in outputs:
            name = out_spec["name"]
            suffix = name.split("_")[-1].lower()
            if suffix.startswith("p") and suffix[1:].isdigit():
                q = float(suffix[1:]) / 100.0
                df[name] = [ _weighted_quantile_from_hist(wdist_hist[z], value_edges, q) for z in zones ]
            elif suffix == "gini":
                df[name] = [ _gini_from_weighted_hist(wdist_hist[z], value_edges) for z in zones ]
            else:
                # if you add other stats later
                df[name] = np.nan
        return df

    raise ValueError("Unexpected method at finalize.")


# ============================================================
# Main: manifest -> compute all indices -> join results
# ============================================================

def load_manifest(path: str | Path) -> Dict[str, Any]:
    """Load indices_manifest.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_local_path_config(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override machine-local paths in the manifest from config/paths.*.json.

    The manifest remains the conceptual definition of the indicators. Local file
    locations come from config so the same definitions can move between machines.
    """
    cfg = load_paths()
    manifest = dict(manifest)
    paths = dict(manifest.get("paths", {}))
    paths["data_root"] = str(path_value(cfg, "sat_data_curated"))
    paths["admin_units"] = str(repo_data_path(cfg, "parishes.geojson"))
    paths["admin_id_field"] = paths.get("admin_id_field", "ID")
    paths["output_dir"] = str(path_value(cfg, "outputs_indices_dir"))
    manifest["paths"] = paths
    return manifest


def run_all_indices_streaming(manifest_path: str | Path, tile_size: int = 2048) -> pd.DataFrame:
    """
    Compute all indices listed in the manifest using streaming raster processing.
    Returns a DataFrame indexed by freguesia string ID.
    """
    start_time = time.time()
    print(f"[START] Running all indices (streaming) from manifest: {manifest_path}")
    manifest = apply_local_path_config(load_manifest(manifest_path))
    paths = manifest["paths"]
    data_root = paths["data_root"]
    admin_units = paths["admin_units"]
    admin_id_field = paths["admin_id_field"]
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bounding box path (as in your earlier setup)
    portugal_bbox = str(repo_data_path(load_paths(), "parishes_bounding_box.geojson"))

    constants = manifest.get("global_defaults", {})
    age_bins = (manifest.get("helpers", {}) or {}).get("age_bins", {})

    indices: List[Dict[str, Any]] = manifest["indices"]

    freg = gpd.read_file(admin_units)
    if admin_id_field not in freg.columns:
        raise KeyError(f"admin_id_field '{admin_id_field}' not found in {admin_units}")

    # Handle missing CRS for GeoJSON (assume WGS84)
    if freg.crs is None:
        freg = freg.set_crs("EPSG:4326")

    zone_map = build_zone_mapping(freg, admin_id_field)
    freg = freg.copy()
    freg[admin_id_field] = freg[admin_id_field].astype(str)
    freg = freg.merge(zone_map, on=admin_id_field, how="left")

    # Output keyed by true string ID
    out_df = pd.DataFrame(index=pd.Index(zone_map[admin_id_field].values, name=admin_id_field)).sort_index()

    for i, index_def in enumerate(indices, start=1):
        index_id = index_def["id"]
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"[{i}/{len(indices)}] "
            f"[{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} elapsed] "
            f"Computing: {index_id}")

        aligned = helpers.build_aligned_raster_set_for_index(
            index_def=index_def,
            data_root=data_root,
            portugal_bbox_geojson=portugal_bbox
        )

        try:
            # Reproject freg to raster CRS and build spatial index once per index
            freg_proj = freg.to_crs(aligned.crs.to_string())
            sindex = freg_proj.sindex

            df_zone = compute_index_streaming(
                index_def=index_def,
                aligned=aligned,
                freg_proj=freg_proj,
                sindex=sindex,
                constants=constants,
                age_bins=age_bins,
                tile_size=tile_size,
                hist_bins=256
            )

            # Map zone_code -> string id, then join
            df_zone = df_zone.join(zone_map.set_index("zone_code"), how="left")
            df_zone = df_zone.set_index(admin_id_field)

            out_df = out_df.join(df_zone, how="left")

        finally:
            helpers.close_aligned(aligned, cleanup_tmp=False)

    return out_df


def save_outputs(df: pd.DataFrame, manifest_path: str | Path) -> None:
    """
    Save outputs to CSV + Parquet.
    Note: Parquet requires pyarrow or fastparquet installed.
    """
    manifest = apply_local_path_config(load_manifest(manifest_path))
    out_dir = Path(manifest["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "freguesia_indices_streaming.parquet"
    csv_path = out_dir / "freguesia_indices_streaming.csv"

    df.to_csv(csv_path, index=True)
    df.to_parquet(parquet_path)

    print(f"[DONE] Wrote: {parquet_path}")
    print(f"[DONE] Wrote: {csv_path}")


if __name__ == "__main__":
    manifest_path = "indices_manifest.json"
    df = run_all_indices_streaming(manifest_path, tile_size=2048)
    save_outputs(df, manifest_path)
