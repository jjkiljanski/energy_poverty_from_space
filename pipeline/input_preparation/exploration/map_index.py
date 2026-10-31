import os
import re
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger("csv_clean")

sys.path.append(str(Path(__file__).resolve().parents[3]))
from pipeline.utils.paths import load_paths  # noqa: E402
from pipeline.utils.plots import plot_portugal_index  # noqa: E402
from pipeline.utils.geometry import (  # noqa: E402
    dump_plot_ready_geojson_if_missing,
    load_mainland_and_islands,
    load_test_set_boundary,
)


# -------- Paths --------
PATHS = load_paths()
MAP_DIR = str(PATHS.external("caop_shapefiles"))
CSV_FOLDER = str(PATHS.external("indices_output"))
CSV_FILE = r"freguesia_indices_streaming"
CSV_PATH = os.path.join(CSV_FOLDER, CSV_FILE + ".csv")
OUT_DIR = str(PATHS.external("satellite_maps"))

EPVI_CSV_PATH = str(PATHS.external("epvi_source"))
EPVI_CSV_FILE = "epvi_gouveia_et_al_2019"
EPVI_OUT_DIR = str(PATHS.external("epvi_maps"))

# NEW: where to dump the plot-ready merged GeoDataFrame
OUT_GEOJSON_PATH = os.path.join(OUT_DIR, f"{CSV_FILE}_plot_ready.geojson")
EPVI_OUT_GEOJSON_PATH = os.path.join(EPVI_OUT_DIR, f"{EPVI_CSV_FILE}_plot_ready.geojson")

# -------- Keys --------
SHP_KEY_COL = "dicofre"   # normalized to lowercase inside the script
CSV_KEY_COL = "ID"

# -------- Mainland shapefile name --------
MAINLAND_SHP_NAME = "Cont_AAD_CAOP2016.shp"

# Optional test-set boundary overlay. This should be a small, precomputed
# GeoJSON in repo data if a fixed spatial test set is defined.
TEST_SET_BOUNDARY_PATH = str(PATHS.external("test_boundary"))


def safe_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\w\-\. ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:150]


def pad_keys_to_same_width(gdf: gpd.GeoDataFrame, df: pd.DataFrame, gdf_key: str, df_key: str):
    gdf = gdf.copy()
    df = df.copy()

    g = gdf[gdf_key].astype(str).str.strip()
    d = df[df_key].astype(str).str.strip()

    g_max = int(g.str.len().max()) if g.notna().any() else 0
    d_max = int(d.str.len().max()) if d.notna().any() else 0
    pad_width = max(g_max, d_max)

    if pad_width > 0:
        gdf[gdf_key] = g.str.zfill(pad_width)
        df[df_key] = d.str.zfill(pad_width)

    return gdf, df


def _coerce_decimal_comma_series_to_float(
    s: pd.Series,
    colname: str = "",
    log_bad: bool = True,
    max_examples: int = 20,
) -> pd.Series:
    """
    Coerce a column to float, handling decimal commas and common junk tokens.
    Logs examples of values that fail conversion.

    Returns a NumPy float64 Series (NOT pandas nullable Float64) to keep GeoPandas/Fiona happy.
    """
    # Fast path: already numeric -> cast to plain float64
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("float64")

    # Work in pandas StringDtype (safe .str operations)
    s_str = s.astype("string")

    # Keep original for debugging
    s_orig = s_str.copy()

    # String normalization
    s_str = s_str.str.strip()
    s_str = s_str.str.replace("\u00a0", "", regex=False)  # NBSP
    s_str = s_str.str.replace(" ", "", regex=False)
    s_str = s_str.str.replace(",", ".", regex=False)

    # Normalize missing tokens (case-insensitive)
    s_lower = s_str.str.lower()
    missing_mask = s_lower.isin(["", "nan", "none", "<na>"])
    s_str = s_str.mask(missing_mask, pd.NA)

    # Convert
    out = pd.to_numeric(s_str, errors="coerce")

    # Logging of "bad" values: values that were not missing but became NaN
    if log_bad:
        # not missing originally AND conversion failed
        bad_mask = s_str.notna() & out.isna()
        n_bad = int(bad_mask.sum())
        if n_bad > 0:
            examples = (
                pd.DataFrame(
                    {
                        "row": s.index[bad_mask].to_list(),
                        "original": s_orig[bad_mask].astype("string").to_list(),
                        "normalized": s_str[bad_mask].astype("string").to_list(),
                    }
                )
                .head(max_examples)
            )
            log.warning(
                "Column '%s': %d values could not be parsed as float. Examples:\n%s",
                colname, n_bad, examples.to_string(index=False)
            )

    # IMPORTANT: return plain numpy float64 (works with GeoPandas/Fiona)
    return out.astype("float64")


def read_csv_data(csv_path: str, csv_key_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)

    if csv_key_col not in df.columns:
        raise ValueError(f"CSV does not contain required key column '{csv_key_col}'. Columns: {list(df.columns)}")

    df = df.copy()
    df[csv_key_col] = df[csv_key_col].astype(str).str.strip()

    df = df[df[csv_key_col].notna()].copy()
    df = df[df[csv_key_col].str.lower() != "nan"].copy()
    df = df[df[csv_key_col] != ""].copy()

    for c in df.columns:
        if c == csv_key_col:
            continue
        df[c] = _coerce_decimal_comma_series_to_float(df[c])

    return df


def join_values(gdf: gpd.GeoDataFrame, df: pd.DataFrame, gdf_key: str, df_key: str) -> gpd.GeoDataFrame:
    return gdf.merge(df, left_on=gdf_key, right_on=df_key, how="left")


def plot_all_columns(mainland_merged: gpd.GeoDataFrame,
                     islands_merged: list[tuple[str, gpd.GeoDataFrame]],
                     key_col: str,
                     csv_key_col: str,
                     out_dir: str,
                     test_boundary: gpd.GeoDataFrame | None = None):
    os.makedirs(out_dir, exist_ok=True)

    cols = [c for c in mainland_merged.columns if c not in (key_col, "geometry", csv_key_col)]
    if not cols:
        raise ValueError("No data columns found to plot after merge.")

    for col in cols:
        numeric_main = pd.to_numeric(mainland_merged[col], errors="coerce")
        numeric_islands = [pd.to_numeric(gdf[col], errors="coerce") for _, gdf in islands_merged]

        total_non_na = int(numeric_main.notna().sum() + sum(v.notna().sum() for v in numeric_islands))
        if total_non_na == 0:
            warnings.warn(f"Skipping '{col}' (no numeric values found in mainland+islands).")
            continue

        out_path = os.path.join(out_dir, f"{safe_filename(col)}.png")
        plot_portugal_index(
            mainland=mainland_merged,
            islands=islands_merged,
            value_col=col,
            mode="continuous",
            title=col,
            legend_title=col,
            test_boundary=test_boundary,
            out_path=out_path,
        )
        print(f"Saved: {out_path}")



def plot_dataset(
    label: str,
    csv_path: str,
    out_dir: str,
    out_geojson_path: str,
    mainland: gpd.GeoDataFrame,
    islands: list[tuple[str, gpd.GeoDataFrame]],
    test_boundary: gpd.GeoDataFrame,
):
    print(f"\n=== {label} ===")
    print("Reading CSV (handling decimal commas)...")
    df = read_csv_data(csv_path, CSV_KEY_COL)
    print(f"CSV rows: {len(df):,} | unique IDs: {df[CSV_KEY_COL].nunique():,}")

    print("Normalizing key widths (zero-padding)...")
    mainland_padded, df = pad_keys_to_same_width(mainland, df, SHP_KEY_COL, CSV_KEY_COL)
    islands_padded = [(name, pad_keys_to_same_width(gdf, df, SHP_KEY_COL, CSV_KEY_COL)[0]) for name, gdf in islands]

    print("Merging values...")
    mainland_m = join_values(mainland_padded, df, SHP_KEY_COL, CSV_KEY_COL)
    islands_m = [(name, join_values(gdf, df, SHP_KEY_COL, CSV_KEY_COL)) for name, gdf in islands_padded]

    dump_plot_ready_geojson_if_missing(mainland_m, islands_m, out_geojson_path)

    print("Plotting (mainland-centered with island insets)...")
    plot_all_columns(mainland_m, islands_m, SHP_KEY_COL, CSV_KEY_COL, out_dir, test_boundary=test_boundary)


def main():
    print("Loading mainland + island shapefiles...")
    mainland, islands = load_mainland_and_islands(MAP_DIR, SHP_KEY_COL, MAINLAND_SHP_NAME)

    islands_rows = sum(len(gdf) for _, gdf in islands)
    print(f"Mainland parishes: {len(mainland):,} | CRS: {mainland.crs}")
    print(f"Island layers: {len(islands):,} | total island rows: {islands_rows:,}")
    print(f"Total parishes (mainland + islands): {len(mainland) + islands_rows:,}")

    print(f"Loading stored test-set boundary overlay: {TEST_SET_BOUNDARY_PATH}")
    test_boundary = load_test_set_boundary(TEST_SET_BOUNDARY_PATH, mainland.crs)

    plot_dataset("Satellite-derived indices", CSV_PATH, OUT_DIR, OUT_GEOJSON_PATH, mainland, islands, test_boundary)
    plot_dataset("EPVI indices", EPVI_CSV_PATH, EPVI_OUT_DIR, EPVI_OUT_GEOJSON_PATH, mainland, islands, test_boundary)

    print("Done.")


if __name__ == "__main__":
    main()
