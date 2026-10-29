from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from pipeline.utils.io import normalize_id


def normalize_columns_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped, lowercase column names."""
    out = df.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    return out


def force_polygonal(geometry: BaseGeometry | None) -> BaseGeometry | None:
    """Keep only polygonal content from a geometry."""
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if geometry.geom_type != "GeometryCollection":
        return None

    polygons = []
    for part in geometry.geoms:
        if part is None or part.is_empty:
            continue
        if part.geom_type == "Polygon":
            polygons.append(part)
        elif part.geom_type == "MultiPolygon":
            polygons.extend(part.geoms)
    if not polygons:
        return None

    union = unary_union(polygons)
    if union is None or union.is_empty:
        return None
    if union.geom_type in {"Polygon", "MultiPolygon"}:
        return union
    return force_polygonal(union)


def repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make geometries valid and discard empty or non-polygonal content."""
    out = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    try:
        out["geometry"] = out.geometry.apply(make_valid)
    except Exception:
        out["geometry"] = out.geometry.buffer(0)
    out["geometry"] = out.geometry.apply(force_polygonal)
    return out.loc[out.geometry.notna() & ~out.geometry.is_empty].copy()


def dissolve_by_id(gdf: gpd.GeoDataFrame, key_col: str) -> gpd.GeoDataFrame:
    """Union rows sharing an identifier into one geometry per identifier."""
    return gdf[[key_col, "geometry"]].copy().dissolve(by=key_col, as_index=False)


def union_geometries(gdf: gpd.GeoDataFrame) -> BaseGeometry:
    """Union all geometries in a non-empty GeoDataFrame."""
    if gdf.empty:
        raise ValueError("Cannot union an empty GeoDataFrame.")
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def load_geometries_with_normalized_ids(
    path: str | Path,
    source_id_col: str,
    *,
    normalized_id_col: str = "ID_norm",
) -> gpd.GeoDataFrame:
    """Load geometries and add consistently normalized feature identifiers."""
    gdf = gpd.read_file(path)[[source_id_col, "geometry"]].copy()
    gdf[normalized_id_col] = normalize_id(gdf[source_id_col])
    return gdf


def find_shapefiles(map_dir: str | Path) -> list[str]:
    """Find all shapefiles recursively below a map directory."""
    paths = sorted(glob.glob(os.path.join(str(map_dir), "**", "*.shp"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .shp files found under: {map_dir}")
    return paths


def load_shapefile(path: str | Path, key_col_lower: str) -> gpd.GeoDataFrame | None:
    """Load, normalize, repair, and deduplicate an identified shapefile layer."""
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        warnings.warn(
            f"Default shapefile read failed for {path}; retrying with pyogrio "
            f"on_invalid='fix'. Original error: {exc}"
        )
        gdf = gpd.read_file(path, engine="pyogrio", on_invalid="fix")
    gdf = normalize_columns_lower(gdf)
    if key_col_lower not in gdf.columns:
        return None

    gdf = gdf[[key_col_lower, "geometry"]].copy()
    gdf[key_col_lower] = gdf[key_col_lower].astype(str).str.strip()
    gdf = repair_geometries(gdf)
    if gdf[key_col_lower].duplicated().any():
        before = len(gdf)
        gdf = repair_geometries(dissolve_by_id(gdf, key_col_lower))
        print(
            f"[{Path(path).name}] dissolved duplicates: "
            f"{before:,} rows -> {len(gdf):,} unique IDs"
        )
    return gdf


def island_complex_name(path: str | Path) -> str:
    """Map CAOP island-layer filenames to presentation-level complexes."""
    filename = Path(path).name.lower()
    if "acores" in filename:
        return "Azores"
    if "madeira" in filename:
        return "Madeira"
    return Path(path).stem


def load_mainland_and_islands(
    map_dir: str | Path,
    key_col_lower: str,
    mainland_name: str,
) -> tuple[gpd.GeoDataFrame, list[tuple[str, gpd.GeoDataFrame]]]:
    """Load a mainland CAOP layer and combine island files by island complex."""
    shapefile_paths = find_shapefiles(map_dir)
    mainland_path = next(
        (
            path
            for path in shapefile_paths
            if Path(path).name.lower() == mainland_name.lower()
        ),
        None,
    )
    if mainland_path is None:
        raise FileNotFoundError(
            f"Could not find mainland shapefile named '{mainland_name}' under: {map_dir}"
        )

    mainland = load_shapefile(mainland_path, key_col_lower)
    if mainland is None:
        raise ValueError(
            f"Mainland shapefile '{mainland_name}' does not contain key column "
            f"'{key_col_lower}'."
        )

    base_crs = mainland.crs
    island_parts: dict[str, list[gpd.GeoDataFrame]] = {}
    skipped = []
    for path in shapefile_paths:
        if path == mainland_path:
            continue
        gdf = load_shapefile(path, key_col_lower)
        if gdf is None:
            skipped.append(path)
            continue
        if base_crs is not None and gdf.crs is not None and gdf.crs != base_crs:
            gdf = gdf.to_crs(base_crs)
        elif base_crs is not None and gdf.crs is None:
            warnings.warn(f"Shapefile has no CRS; assuming it matches mainland CRS: {path}")
        island_parts.setdefault(island_complex_name(path), []).append(gdf)

    if skipped:
        warnings.warn(
            "Skipped shapefiles without the key column "
            f"'{key_col_lower}' (case-insensitive handled):\n"
            + "\n".join(skipped[:20])
            + (f"\n... (+{len(skipped) - 20} more)" if len(skipped) > 20 else "")
        )

    islands = []
    for complex_name, parts in island_parts.items():
        combined = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True),
            geometry="geometry",
            crs=base_crs,
        )
        if combined[key_col_lower].duplicated().any():
            combined = repair_geometries(dissolve_by_id(combined, key_col_lower))
        islands.append((complex_name, combined))
    return mainland, islands


def load_test_set_boundary(path: str | Path, target_crs) -> gpd.GeoDataFrame:
    """Load an optional test boundary and align it to the target CRS."""
    if not Path(path).exists():
        warnings.warn(
            "Test-set boundary file not found, so no black boundary overlay will be "
            f"drawn: {path}"
        )
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    gdf = gpd.read_file(path)
    if gdf.crs is not None and target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def dump_plot_ready_geojson_if_missing(
    mainland_merged: gpd.GeoDataFrame,
    islands_merged: list[tuple[str, gpd.GeoDataFrame]],
    out_geojson_path: str | Path,
) -> Path:
    """Export the merged mainland and island plotting layers if not already present."""
    output = Path(out_geojson_path)
    if output.exists():
        print(f"GeoJSON already exists, skipping export: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    gdf_plot = gpd.GeoDataFrame(
        pd.concat([mainland_merged, *(gdf for _, gdf in islands_merged)], ignore_index=True),
        crs=mainland_merged.crs,
    )
    gdf_plot = repair_geometries(gdf_plot)
    try:
        if gdf_plot.crs is not None and gdf_plot.crs.to_epsg() != 4326:
            gdf_plot = gdf_plot.to_crs(epsg=4326)
    except Exception:
        pass

    gdf_plot.to_file(output, driver="GeoJSON")
    print(f"Exported plot-ready GeoJSON: {output}")
    return output
