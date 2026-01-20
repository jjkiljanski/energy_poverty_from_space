from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass, asdict
from typing import Optional, Union, Dict, Any, Tuple, Literal

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import Affine

import math

@dataclass(frozen=True)
class GridSpec:
    """A lightweight, transparent description of a raster grid."""
    crs_wkt: str
    transform: Tuple[float, float, float, float, float, float]  # Affine as 6-tuple
    width: int
    height: int
    bounds: Tuple[float, float, float, float]  # left, bottom, right, top
    res_x_m: float
    res_y_m: float

    @property
    def cell_size_m(self) -> float:
        """Convenience: returns a single cell size if square, else min(|x|,|y|)."""
        if abs(self.res_x_m) == abs(self.res_y_m):
            return abs(self.res_x_m)
        return min(abs(self.res_x_m), abs(self.res_y_m))

    def affine(self) -> Affine:
        return Affine(*self.transform)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cell_size_m"] = self.cell_size_m
        return d


def grid_from_tiff(
    tif_path: str,
    *,
    return_geojson: bool = False,
    max_cells_for_geojson: int = 50_000,
) -> Union[GridSpec, Tuple[GridSpec, Optional[Dict[str, Any]]]]:
    """
    Read a GeoTIFF and return its grid definition + (optionally) a GeoJSON grid.

    Notes:
    - For real-world rasters, a full cell-by-cell GeoJSON is usually enormous.
      This function only creates a grid GeoJSON when the cell count is small
      (<= max_cells_for_geojson). Otherwise it returns None for the GeoJSON.
    - Assumes CRS units are meters for the "meters" interpretation (true for EPSG:3763).
    """
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS. Can't interpret cell size in meters.")

        # Resolution in CRS units (for EPSG:3763 these are meters)
        res_x, res_y = src.res  # (pixel width, pixel height) in CRS units

        spec = GridSpec(
            crs_wkt=src.crs.to_wkt(),
            transform=tuple(src.transform)[:6],
            width=src.width,
            height=src.height,
            bounds=tuple(src.bounds),
            res_x_m=float(res_x),
            res_y_m=float(res_y),
        )

        if not return_geojson:
            return spec

        n_cells = src.width * src.height
        if n_cells > max_cells_for_geojson:
            return spec, None  # avoid generating massive GeoJSON

        # Build a cell polygon grid GeoJSON (only safe for small rasters)
        try:
            from shapely.geometry import box, mapping
        except ImportError as e:
            raise ImportError(
                "To return GeoJSON grids, install shapely: pip install shapely"
            ) from e

        transform = src.transform
        features = []
        for row in range(src.height):
            for col in range(src.width):
                # pixel bounds
                x_left, y_top = transform * (col, row)
                x_right, y_bottom = transform * (col + 1, row + 1)
                geom = box(x_left, y_bottom, x_right, y_top)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": {"row": row, "col": col},
                    }
                )

        geojson = {"type": "FeatureCollection", "features": features}
        return spec, geojson


def _parse_resampling(method: Union[str, Resampling]) -> Resampling:
    if isinstance(method, Resampling):
        return method
    m = method.lower().strip()
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
    }
    if m not in mapping:
        raise ValueError(f"Unknown resampling='{method}'. Choose from: {list(mapping)}")
    return mapping[m]


def regrid_tiff_to_grid(
    src_tif: str,
    dst_grid: Union[GridSpec, Dict[str, Any]],
    out_tif: str,
    *,
    resampling: Union[str, Resampling] = "nearest",
    dst_nodata: Optional[float] = None,
    compress: str = "deflate",
    tiled: bool = True,
    bigtiff: str = "if_safer",
) -> str:
    """
    Resample/warp a GeoTIFF onto a target grid (same CRS assumed, e.g., EPSG:3763).

    This does NOT "reproject" CRS if different; it matches the destination transform+shape.
    If CRS differs, it will still work (rasterio can reproject), but the assumption is the same EPSG.
    """
    if isinstance(dst_grid, dict):
        # Accept dicts produced by GridSpec.to_dict() or manual configs
        crs_wkt = dst_grid["crs_wkt"]
        transform = Affine(*dst_grid["transform"])
        width = int(dst_grid["width"])
        height = int(dst_grid["height"])
    else:
        crs_wkt = dst_grid.crs_wkt
        transform = dst_grid.affine()
        width = dst_grid.width
        height = dst_grid.height

    rs = _parse_resampling(resampling)

    with rasterio.open(src_tif) as src:
        src_crs = src.crs
        if src_crs is None:
            raise ValueError("Source raster has no CRS.")
        dst_crs = rasterio.crs.CRS.from_wkt(crs_wkt)

        # Prepare output profile
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            compress=compress,
            tiled=tiled,
            bigtiff=bigtiff,
        )

        # Decide nodata
        if dst_nodata is None:
            dst_nodata = src.nodata
        if dst_nodata is not None:
            profile.update(nodata=dst_nodata)

        # Allocate destination array
        count = src.count
        dst_dtype = src.dtypes[0]
        dst = np.zeros((count, height, width), dtype=dst_dtype)

        for b in range(1, count + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=dst[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=rs,
                src_nodata=src.nodata,
                dst_nodata=dst_nodata,
            )

        with rasterio.open(out_tif, "w", **profile) as out:
            out.write(dst)

    return out_tif

VarType = Literal["count", "share", "continuous", "categorical", "binary"]


def _resampling_for_var(
    var_type: VarType,
    *,
    downsample: bool,
) -> str:
    """
    Choose a resampling method given variable type and direction.
    """
    if var_type in ("categorical", "binary"):
        return "mode" if downsample else "nearest"
    if var_type == "continuous":
        return "average" if downsample else "bilinear"
    if var_type == "share":
        return "average" if downsample else "bilinear"
    if var_type == "count":
        # counts handled separately (sum), but if forced to resample we use average and then scale
        return "average" if downsample else "bilinear"
    raise ValueError(f"Unknown var_type='{var_type}'")


def _base_name_no_ext(p: str) -> str:
    p = Path(p)
    stem = p.name
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return p.stem


def _sum_regrid_counts_to_target(
    src_tif: str,
    target_grid,  # GridSpec
    out_tif: str,
    *,
    dst_nodata: Optional[float] = None,
    compress: str = "deflate",
    tiled: bool = True,
    bigtiff: str = "if_safer",
) -> str:
    """
    Conserve totals when going from finer grid -> coarser grid for COUNT variables.
    Strategy:
      1) Reproject with Resampling.average to get mean value per output pixel
      2) Multiply by area_ratio = (dst_pixel_area / src_pixel_area) to convert mean->sum
    This is fast and works well when both rasters are in a metric CRS (e.g., EPSG:3763).
    """
    with rasterio.open(src_tif) as src:
        if src.crs is None:
            raise ValueError("Source raster has no CRS.")
        dst_crs = rasterio.crs.CRS.from_wkt(target_grid.crs_wkt)

        src_px_x, src_px_y = map(abs, src.res)
        dst_px_x, dst_px_y = abs(target_grid.res_x_m), abs(target_grid.res_y_m)
        src_area = src_px_x * src_px_y
        dst_area = dst_px_x * dst_px_y
        if src_area <= 0 or dst_area <= 0:
            raise ValueError("Invalid pixel areas (check transforms/resolution).")

        area_ratio = dst_area / src_area

        # Output profile
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=target_grid.affine(),
            width=target_grid.width,
            height=target_grid.height,
            compress=compress,
            tiled=tiled,
            bigtiff=bigtiff,
        )

        if dst_nodata is None:
            dst_nodata = src.nodata
        if dst_nodata is not None:
            profile.update(nodata=dst_nodata)

        count = src.count
        # Use float for intermediate to preserve sums accurately, then cast back if you want
        dst = np.zeros((count, target_grid.height, target_grid.width), dtype=np.float32)

        for b in range(1, count + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=dst[b - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_grid.affine(),
                dst_crs=dst_crs,
                resampling=Resampling.average,
                src_nodata=src.nodata,
                dst_nodata=dst_nodata,
            )

        # Convert mean -> sum
        dst *= area_ratio

        # Decide output dtype:
        # - If you want integer counts, round and cast; otherwise keep float.
        # Here: keep float unless original was integer.
        if "int" in src.dtypes[0].lower():
            dst_out = np.rint(dst).astype(np.int64)
            profile.update(dtype="int64")
        else:
            dst_out = dst.astype(np.float32)
            profile.update(dtype="float32")

        with rasterio.open(out_tif, "w", **profile) as out:
            out.write(dst_out)

    return out_tif

VarType = Literal["count", "share", "continuous", "categorical", "binary"]


def align_two_tiffs_auto(
    tif_a_path: str,
    var_type_a: VarType,
    tif_b_path: str,
    var_type_b: VarType,
    save_folder_path: str,
    *,
    dst_nodata: Optional[float] = None,
    overwrite: bool = False,
    equal_grid_prefer: Literal["A_to_B", "B_to_A"] = "A_to_B",
    px_tol: float = 1e-6,
) -> Tuple[str, str]:
    """
    Align two rasters by resampling ONE of them onto the other's grid.

    You pass var_type for BOTH rasters. The function decides which raster to resample:
      - If pixel sizes differ: resample the finer (smaller pixel) onto the coarser grid.
      - If pixel sizes are equal (within px_tol): resample according to equal_grid_prefer.

    Variable type handling (applies to the raster being resampled):
      - count: conserve totals when downsampling (finer -> coarser) using sum-conserving method
      - share: average (down), bilinear (up or equal)
      - continuous: average (down), bilinear (up or equal)
      - categorical/binary: mode (down), nearest (up or equal)
      - count at equal pixel size: nearest (preserve values; only grid alignment changes)

    Output filename:
      {src_name}_reprojected_to_{tgt_name}_grid__{src_var_type}.tif

    Returns:
      (out_tif_path, which_was_resampled) where which_was_resampled is "A" or "B".

    Example usage:
        out_path, who = align_two_tiffs_auto(
            "PM25_1km.tif", "continuous",
            "NO2_1km.tif", "continuous",
            "data_proc/1km_aligned",
            equal_grid_prefer="A_to_B",   # resample PM25 onto NO2 grid
        )
        print(out_path, who)
    """
    out_dir = Path(save_folder_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_a = grid_from_tiff(tif_a_path)
    grid_b = grid_from_tiff(tif_b_path)

    a_px = float(abs(grid_a.res_x_m))
    b_px = float(abs(grid_b.res_x_m))

    def _is_equal(x: float, y: float) -> bool:
        return abs(x - y) <= px_tol

    # Decide direction: which becomes source (resampled) and which becomes target grid
    if _is_equal(a_px, b_px):
        if equal_grid_prefer == "A_to_B":
            src_path, src_type, target_grid = tif_a_path, var_type_a, grid_b
            src_label = "A"
            src_name, tgt_name = _base_name_no_ext(tif_a_path), _base_name_no_ext(tif_b_path)
        else:
            src_path, src_type, target_grid = tif_b_path, var_type_b, grid_a
            src_label = "B"
            src_name, tgt_name = _base_name_no_ext(tif_b_path), _base_name_no_ext(tif_a_path)

        out_name = f"{src_name}_reprojected_to_{tgt_name}_grid__{src_type}.tif"
        out_path = out_dir / out_name
        if out_path.exists() and not overwrite:
            return str(out_path), src_label

        # Equal pixel size: just align transform/extent.
        # Choose safest defaults by type.
        if src_type in ("categorical", "binary", "count"):
            method = "nearest"
        else:
            method = "bilinear"

        regrid_tiff_to_grid(
            src_tif=src_path,
            dst_grid=target_grid,
            out_tif=str(out_path),
            resampling=method,
            dst_nodata=dst_nodata,
        )
        return str(out_path), src_label

    # Pixel sizes differ: resample the finer -> coarser
    if a_px < b_px:
        src_path, src_type, target_grid = tif_a_path, var_type_a, grid_b
        src_label = "A"
        src_name, tgt_name = _base_name_no_ext(tif_a_path), _base_name_no_ext(tif_b_path)
        src_px, dst_px = a_px, b_px
    else:
        src_path, src_type, target_grid = tif_b_path, var_type_b, grid_a
        src_label = "B"
        src_name, tgt_name = _base_name_no_ext(tif_b_path), _base_name_no_ext(tif_a_path)
        src_px, dst_px = b_px, a_px

    out_name = f"{src_name}_reprojected_to_{tgt_name}_grid__{src_type}.tif"
    out_path = out_dir / out_name
    if out_path.exists() and not overwrite:
        return str(out_path), src_label

    downsample = src_px < dst_px  # should be True here

    # Counts: conserve totals when downsampling (finer -> coarser)
    if src_type == "count" and downsample:
        _sum_regrid_counts_to_target(
            src_tif=src_path,
            target_grid=target_grid,
            out_tif=str(out_path),
            dst_nodata=dst_nodata,
        )
        return str(out_path), src_label

    # Otherwise: choose resampling method by type
    method = _resampling_for_var(src_type, downsample=downsample)
    regrid_tiff_to_grid(
        src_tif=src_path,
        dst_grid=target_grid,
        out_tif=str(out_path),
        resampling=method,
        dst_nodata=dst_nodata,
    )
    return str(out_path), src_label