import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.validation import make_valid  # shapely >= 2.0
from shapely.ops import unary_union


# -------- Paths --------
MAP_DIR = r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Map"
CSV_FOLDER = r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Adm_data\csv"
CSV_FILE = r"all_used_adm_indicators"
CSV_PATH = os.path.join(CSV_FOLDER, CSV_FILE + ".csv")
OUT_DIR = os.path.join(os.path.dirname(CSV_PATH), f"choropleths_{CSV_FILE}")

# NEW: where to dump the plot-ready merged GeoDataFrame
OUT_GEOJSON_PATH = os.path.join(OUT_DIR, f"{CSV_FILE}_plot_ready.geojson")

# -------- Keys --------
SHP_KEY_COL = "dicofre"   # normalized to lowercase inside the script
CSV_KEY_COL = "ID"

# -------- Mainland shapefile name --------
MAINLAND_SHP_NAME = "Cont_AAD_CAOP2016.shp"


def safe_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\w\-\. ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:150]


def normalize_columns_lower(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def force_polygonal(geom):
    """
    Ensure geometries are polygonal so GeoPandas can plot them reliably.
    - Keeps Polygon / MultiPolygon as-is
    - For GeometryCollection: keeps only polygonal parts and unions them
    - Drops non-polygonal geometries (returns None)
    """
    if geom is None or geom.is_empty:
        return None

    gt = geom.geom_type

    if gt in ("Polygon", "MultiPolygon"):
        return geom

    if gt == "GeometryCollection":
        polys = []
        for g in geom.geoms:
            if g is None or g.is_empty:
                continue
            if g.geom_type == "Polygon":
                polys.append(g)
            elif g.geom_type == "MultiPolygon":
                polys.extend(list(g.geoms))

        if not polys:
            return None

        u = unary_union(polys)
        if u is None or u.is_empty:
            return None
        if u.geom_type in ("Polygon", "MultiPolygon"):
            return u

        return force_polygonal(u)

    return None


def repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Make geometries valid and ensure they are polygonal (Polygon/MultiPolygon).
    """
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    try:
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf["geometry"] = gdf.geometry.apply(force_polygonal)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def dissolve_by_id(gdf: gpd.GeoDataFrame, key_col: str) -> gpd.GeoDataFrame:
    """
    Union/dissolve multiple rows sharing the same ID into a single geometry.
    """
    gdf = gdf[[key_col, "geometry"]].copy()
    dissolved = gdf.dissolve(by=key_col, as_index=False)
    return dissolved


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


def find_shapefiles(map_dir: str):
    shp_paths = sorted(glob.glob(os.path.join(map_dir, "**", "*.shp"), recursive=True))
    if not shp_paths:
        raise FileNotFoundError(f"No .shp files found under: {map_dir}")
    return shp_paths


def load_shapefile(path: str, key_col_lower: str) -> gpd.GeoDataFrame | None:
    gdf = gpd.read_file(path)
    gdf = normalize_columns_lower(gdf)

    if key_col_lower not in gdf.columns:
        return None

    gdf = gdf[[key_col_lower, "geometry"]].copy()
    gdf[key_col_lower] = gdf[key_col_lower].astype(str).str.strip()

    gdf = repair_geometries(gdf)

    if gdf[key_col_lower].duplicated().any():
        before = len(gdf)
        gdf = dissolve_by_id(gdf, key_col_lower)
        gdf = repair_geometries(gdf)
        after = len(gdf)
        print(f"[{os.path.basename(path)}] dissolved duplicates: {before:,} rows -> {after:,} unique IDs")

    return gdf


def load_mainland_and_islands(map_dir: str, key_col_lower: str, mainland_name: str):
    shp_paths = find_shapefiles(map_dir)

    mainland_path = None
    for p in shp_paths:
        if os.path.basename(p).lower() == mainland_name.lower():
            mainland_path = p
            break

    if mainland_path is None:
        raise FileNotFoundError(f"Could not find mainland shapefile named '{mainland_name}' under: {map_dir}")

    mainland = load_shapefile(mainland_path, key_col_lower)
    if mainland is None:
        raise ValueError(f"Mainland shapefile '{mainland_name}' does not contain key column '{key_col_lower}'.")

    base_crs = mainland.crs

    islands = []
    skipped = []

    for p in shp_paths:
        if p == mainland_path:
            continue

        gdf = load_shapefile(p, key_col_lower)
        if gdf is None:
            skipped.append(p)
            continue

        if base_crs is not None and gdf.crs is not None and gdf.crs != base_crs:
            gdf = gdf.to_crs(base_crs)
        elif base_crs is not None and gdf.crs is None:
            warnings.warn(f"Shapefile has no CRS; assuming it matches mainland CRS: {p}")

        islands.append((os.path.basename(p), gdf))

    if skipped:
        warnings.warn(
            "Skipped shapefiles without the key column "
            f"'{key_col_lower}' (case-insensitive handled):\n"
            + "\n".join(skipped[:20])
            + (f"\n... (+{len(skipped)-20} more)" if len(skipped) > 20 else "")
        )

    return mainland, islands


def _coerce_decimal_comma_series_to_float(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    s2 = (
        s.astype(str)
         .str.strip()
         .replace({"": np.nan, "nan": np.nan, "None": np.nan})
         .str.replace("\u00a0", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s2, errors="coerce")


def read_csv_data(csv_path: str, csv_key_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8", dtype=str)

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


def dump_plot_ready_geojson_if_missing(
    mainland_merged: gpd.GeoDataFrame,
    islands_merged: list[tuple[str, gpd.GeoDataFrame]],
    out_geojson_path: str,
) -> str:
    """
    Build the exact GeoDataFrame used for plotting (mainland + islands, merged with CSV),
    and dump it to GeoJSON if it doesn't already exist.
    """
    if os.path.exists(out_geojson_path):
        print(f"GeoJSON already exists, skipping export: {out_geojson_path}")
        return out_geojson_path

    os.makedirs(os.path.dirname(out_geojson_path), exist_ok=True)

    parts = [mainland_merged]
    parts.extend([gdf for _, gdf in islands_merged])

    gdf_plot = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        crs=mainland_merged.crs
    )

    # Ensure geometries are valid/polygonal for portability (optional but helps)
    gdf_plot = repair_geometries(gdf_plot)

    # GeoJSON: safer to write as WGS84 (EPSG:4326) so most tools open it easily
    try:
        if gdf_plot.crs is not None and gdf_plot.crs.to_epsg() != 4326:
            gdf_plot = gdf_plot.to_crs(epsg=4326)
    except Exception:
        # If CRS is weird/unset, just write as-is
        pass

    gdf_plot.to_file(out_geojson_path, driver="GeoJSON")
    print(f"Exported plot-ready GeoJSON: {out_geojson_path}")
    return out_geojson_path


def plot_mainland_with_insets(mainland: gpd.GeoDataFrame,
                              islands: list[tuple[str, gpd.GeoDataFrame]],
                              value_col: str,
                              key_col: str,
                              out_path: str,
                              title: str):

    main_vals = pd.to_numeric(mainland[value_col], errors="coerce")

    fig = plt.figure(figsize=(10, 10))
    ax_main = fig.add_axes([0.33, 0.05, 0.64, 0.90])
    ax_main.set_axis_off()

    minx, miny, maxx, maxy = mainland.total_bounds
    ax_main.set_xlim(minx, maxx)
    ax_main.set_ylim(miny, maxy)

    mainland.plot(ax=ax_main, color="#eeeeee", linewidth=0)
    mainland.assign(_val=main_vals).plot(
        ax=ax_main,
        column="_val",
        legend=True,
        linewidth=0,
        missing_kwds={"color": "#eeeeee", "label": "Missing"},
    )

    missing = int(main_vals.isna().sum())
    ax_main.set_title(f"{title}  |  missing: {missing}", fontsize=14)

    n = len(islands)
    if n > 0:
        top = 0.92
        bottom = 0.10
        left = 0.05
        width = 0.25
        gap = 0.02
        height = (top - bottom - gap * (min(n, 4) - 1)) / min(n, 4)

        islands_sorted = sorted(
            islands,
            key=lambda t: float(t[1].geometry.area.sum()),
            reverse=True
        )

        slots = min(n, 4)
        for i, (name, gdf_island) in enumerate(islands_sorted):
            slot = i if i < slots else (i % slots)
            y0 = top - (slot + 1) * height - slot * gap

            ax_in = fig.add_axes([left, y0, width, height], frameon=True)
            ax_in.set_xticks([])
            ax_in.set_yticks([])
            ax_in.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            ax_in.set_facecolor("white")
            for spine in ax_in.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(1.2)

            vals = pd.to_numeric(gdf_island[value_col], errors="coerce")
            gdf_island.plot(ax=ax_in, color="#eeeeee", linewidth=0)
            gdf_island.assign(_val=vals).plot(
                ax=ax_in,
                column="_val",
                legend=False,
                linewidth=0,
                missing_kwds={"color": "#eeeeee"},
            )

            bx0, by0, bx1, by1 = gdf_island.total_bounds
            padx = (bx1 - bx0) * 0.05 if bx1 > bx0 else 1
            pady = (by1 - by0) * 0.05 if by1 > by0 else 1
            ax_in.set_xlim(bx0 - padx, bx1 + padx)
            ax_in.set_ylim(by0 - pady, by1 + pady)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_columns(mainland_merged: gpd.GeoDataFrame,
                     islands_merged: list[tuple[str, gpd.GeoDataFrame]],
                     key_col: str,
                     csv_key_col: str,
                     out_dir: str):
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
        plot_mainland_with_insets(
            mainland=mainland_merged,
            islands=islands_merged,
            value_col=col,
            key_col=key_col,
            out_path=out_path,
            title=col,
        )
        print(f"Saved: {out_path}")



def main():
    print("Loading mainland + island shapefiles...")
    mainland, islands = load_mainland_and_islands(MAP_DIR, SHP_KEY_COL, MAINLAND_SHP_NAME)

    islands_rows = sum(len(gdf) for _, gdf in islands)
    print(f"Mainland parishes: {len(mainland):,} | CRS: {mainland.crs}")
    print(f"Island layers: {len(islands):,} | total island rows: {islands_rows:,}")
    print(f"Total parishes (mainland + islands): {len(mainland) + islands_rows:,}")

    print("Reading CSV (handling decimal commas)...")
    df = read_csv_data(CSV_PATH, CSV_KEY_COL)
    print(f"CSV rows: {len(df):,} | unique IDs: {df[CSV_KEY_COL].nunique():,}")

    print("Normalizing key widths (zero-padding)...")
    mainland, df = pad_keys_to_same_width(mainland, df, SHP_KEY_COL, CSV_KEY_COL)
    islands = [(name, pad_keys_to_same_width(gdf, df, SHP_KEY_COL, CSV_KEY_COL)[0]) for name, gdf in islands]

    print("Merging values...")
    mainland_m = join_values(mainland, df, SHP_KEY_COL, CSV_KEY_COL)
    islands_m = [(name, join_values(gdf, df, SHP_KEY_COL, CSV_KEY_COL)) for name, gdf in islands]

    # NEW: dump the merged, cleaned GeoDataFrame used for plotting (if missing)
    dump_plot_ready_geojson_if_missing(mainland_m, islands_m, OUT_GEOJSON_PATH)

    print("Plotting (mainland-centered with island insets)...")
    plot_all_columns(mainland_m, islands_m, SHP_KEY_COL, CSV_KEY_COL, OUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()