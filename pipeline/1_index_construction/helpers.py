from __future__ import annotations

import json
import os
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import math

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from rasterio.windows import transform as win_transform
from rasterio.warp import Resampling
from rasterio.features import rasterize
import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipeline.utils.paths import load_paths, path_value, repo_data_path  # noqa: E402

# Optional GDAL: best way to build VRT mosaics without loading arrays
try:
    from osgeo import gdal  # type: ignore
    GDAL_AVAILABLE = True
except Exception:
    GDAL_AVAILABLE = False


# ----------------------------
# Dataclasses
# ----------------------------

@dataclass(frozen=True)
class FolderMosaic:
    """A folder-level mosaic: many tiles combined into a VRT dataset."""
    folder_name: str
    tile_paths: List[Path]               # all selected tiles (mainland + islands etc.)
    vrt_path: Path                       # path to VRT file (tiny)
    crs: CRS
    res: Tuple[float, float]             # representative/coarsest res among tiles
    bounds: Tuple[float, float, float, float]  # bounds of mosaic dataset


@dataclass
class AlignedRasterSet:
    """
    Holds aligned VRT readers (WarpedVRT) for each bound variable.
    All aligned to reference grid.
    """
    ref_var: str
    ref_source: Path
    crs: CRS
    transform: rasterio.Affine
    width: int
    height: int
    bounds: Tuple[float, float, float, float]  # Portugal bbox in CRS coords
    vrt_map: Dict[str, WarpedVRT]              # var_name -> aligned reader
    tmp_vrts: List[Path]                       # to delete later if desired


# ----------------------------
# CRS helpers
# ----------------------------

def is_mollweide(crs: CRS) -> bool:
    if crs is None:
        return False
    try:
        proj4 = crs.to_proj4()
    except Exception:
        proj4 = str(crs)
    p = proj4.lower()
    return ("+proj=moll" in p) or ("mollweide" in p)


def assert_mollweide_or_raise(crs: CRS, source_name: str) -> None:
    if not is_mollweide(crs):
        raise ValueError(f"[CRS ERROR] Not Mollweide: {source_name}\nDetected CRS: {crs}")


# ----------------------------
# Index definition helpers
# ----------------------------

def load_index_definition(index_json: str | Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(index_json) if isinstance(index_json, str) else index_json


def required_folders_from_index(index_def: Dict[str, Any]) -> Dict[str, str]:
    bind = index_def.get("bind", {})
    if not isinstance(bind, dict) or not bind:
        raise ValueError("Index definition has no valid 'bind' dict.")
    return bind


# ----------------------------
# Bounding box helpers
# ----------------------------

def read_bbox_geometry(bbox_geojson_path: str | Path) -> gpd.GeoDataFrame:
    bbox_geojson_path = Path(bbox_geojson_path)
    if not bbox_geojson_path.exists():
        raise FileNotFoundError(f"Bounding box file not found: {bbox_geojson_path}")
    gdf = gpd.read_file(bbox_geojson_path)
    if gdf.empty or gdf.geometry.isnull().all():
        raise ValueError("Bounding box GeoJSON contains no valid geometry.")
    return gdf


def reproject_bounds_to_crs(bbox_gdf: gpd.GeoDataFrame, target_crs: CRS) -> Tuple[float, float, float, float]:
    if bbox_gdf.crs is None:
        bbox_gdf = bbox_gdf.set_crs("EPSG:4326")
    bbox_proj = bbox_gdf.to_crs(target_crs.to_string())
    return tuple(bbox_proj.total_bounds)  # (minx, miny, maxx, maxy)

# ----------------------------
# TIFF windowing helpers
# ----------------------------

def iter_windows(aligned, tile_size: int = 2048):
    """
    Yield rasterio windows that cover the aligned grid.
    """
    n_rows = aligned.height
    n_cols = aligned.width
    for row_off in range(0, n_rows, tile_size):
        h = min(tile_size, n_rows - row_off)
        for col_off in range(0, n_cols, tile_size):
            w = min(tile_size, n_cols - col_off)
            yield rasterio.windows.Window(col_off=col_off, row_off=row_off, width=w, height=h)


def read_window_arrays(aligned, window, masked: bool = True, band: int = 1, dtype=np.float32):
    """
    Read a window for all vars; always returns float arrays with NaN for nodata.
    """
    arrays = {}
    for var_name, vrt in aligned.vrt_map.items():
        arr = vrt.read(band, window=window, masked=masked)
        if np.ma.isMaskedArray(arr):
            arrays[var_name] = arr.astype(dtype).filled(np.nan)
        else:
            arrays[var_name] = np.asarray(arr, dtype=dtype)
    return arrays

# --------------------------------------
# Rasterization of freguesia labels
# --------------------------------------

def rasterize_labels_for_window(freg_proj, code_field, aligned, window):
    """
    Rasterize only the window extent.
    """
    t = rasterio.windows.transform(window, aligned.transform)
    out_shape = (int(window.height), int(window.width))

    shapes = [
        (geom, int(val))
        for geom, val in zip(freg_proj.geometry, freg_proj[code_field])
        if geom is not None and not geom.is_empty and pd.notnull(val)
    ]

    labels = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=t,
        fill=0,
        dtype=np.int32,
        all_touched=False
    )
    return labels

# ----------------------------
# Folder scanning (ALL tiles)
# ----------------------------

def list_tifs_in_folder(folder_path: Path) -> List[Path]:
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    candidates: List[Path] = []
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        candidates.extend(folder_path.glob(ext))
    return sorted(candidates)


def overlap_area(bounds_a: rasterio.coords.BoundingBox, bounds_b: Tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = bounds_b
    ixmin = max(bounds_a.left, minx)
    iymin = max(bounds_a.bottom, miny)
    ixmax = min(bounds_a.right, maxx)
    iymax = min(bounds_a.top, maxy)
    if ixmax <= ixmin or iymax <= iymin:
        return 0.0
    return (ixmax - ixmin) * (iymax - iymin)


def select_all_tiles_overlapping_bounds(folder_path: Path, bounds: Tuple[float, float, float, float]) -> List[Path]:
    """
    Returns ALL tiffs in the folder that overlap the Portugal bbox (mainland + islands, etc.).
    This fixes the 'multiple tiles' issue and ensures we don't silently drop islands.
    """
    tifs = list_tifs_in_folder(folder_path)
    if not tifs:
        raise FileNotFoundError(f"No TIFFs found in folder: {folder_path}")

    selected: List[Path] = []
    for tif in tifs:
        with rasterio.open(tif) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no CRS: {tif}")
            assert_mollweide_or_raise(src.crs, f"{folder_path.name}/{tif.name}")
            if overlap_area(src.bounds, bounds) > 0:
                selected.append(tif)

    if not selected:
        raise ValueError(
            f"No TIFF tiles in folder '{folder_path.name}' overlap Portugal bbox.\n"
            f"Bounds: {bounds}"
        )
    return selected


def coarsest_resolution_among_tiles(tile_paths: List[Path]) -> Tuple[float, float]:
    """
    Returns the coarsest (largest pixel size by area) resolution across tiles.
    """
    best = None
    best_area = -1.0
    for p in tile_paths:
        with rasterio.open(p) as src:
            rx, ry = src.res
            area = abs(rx) * abs(ry)
            if area > best_area:
                best_area = area
                best = (rx, ry)
    assert best is not None
    return best


# ----------------------------
# Build folder-level mosaic VRT
# ----------------------------

def build_vrt_mosaic(tile_paths: List[Path], vrt_path: Path) -> None:
    """
    Builds a VRT mosaic referencing tile_paths.
    Uses GDAL if available (best). Otherwise raises with guidance.
    """
    if GDAL_AVAILABLE:
        # GDAL wants strings
        srcs = [str(p) for p in tile_paths]
        vrt = gdal.BuildVRT(str(vrt_path), srcs)
        if vrt is None:
            raise RuntimeError(f"GDAL BuildVRT failed for: {vrt_path}")
        vrt.FlushCache()
        vrt = None
        return

    # No GDAL available -> We can’t build a real mosaic VRT safely.
    # (rasterio.merge would load arrays, defeating the point for big rasters.)
    raise RuntimeError(
        "GDAL (osgeo.gdal) not available, cannot build VRT mosaics without loading data.\n"
        "Install GDAL / osgeo, or we can fall back to rasterio.merge for smaller windows."
    )


def open_folder_mosaic(
    data_root: Path,
    folder_name: str,
    portugal_bounds_moll: Tuple[float, float, float, float],
    tmp_dir: Path
) -> FolderMosaic:
    folder_path = data_root / folder_name
    tile_paths = select_all_tiles_overlapping_bounds(folder_path, portugal_bounds_moll)

    # Build a tiny VRT that mosaics these tiles
    vrt_path = tmp_dir / f"{folder_name}.vrt"
    build_vrt_mosaic(tile_paths, vrt_path)

    with rasterio.open(vrt_path) as src:
        if src.crs is None:
            raise ValueError(f"Mosaic VRT has no CRS: {vrt_path}")
        assert_mollweide_or_raise(src.crs, f"VRT({folder_name})")
        crs = src.crs
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)

    res = coarsest_resolution_among_tiles(tile_paths)

    return FolderMosaic(
        folder_name=folder_name,
        tile_paths=tile_paths,
        vrt_path=vrt_path,
        crs=crs,
        res=res,
        bounds=bounds
    )


# ----------------------------
# Alignment helpers
# ----------------------------

def default_resampling_for_folder(folder_name: str) -> Resampling:
    categorical = {"Age", "SMOD"}
    return Resampling.nearest if folder_name in categorical else Resampling.bilinear


def choose_reference_coarsest(mosaics: List[FolderMosaic], bind: Dict[str, str]) -> Tuple[str, FolderMosaic]:
    """
    Choose reference variable based on coarsest resolution among *folder mosaics*.
    Returns (ref_var_name, ref_mosaic).
    """
    if not mosaics:
        raise ValueError("No mosaics provided to choose_reference_coarsest().")

    # Map folder_name -> coarsest pixel area
    folder_to_area = {m.folder_name: abs(m.res[0]) * abs(m.res[1]) for m in mosaics}

    # Choose folder with max area, then pick the first variable bound to that folder
    ref_folder = max(folder_to_area, key=lambda k: folder_to_area[k])
    ref_var = next(v for v, f in bind.items() if f == ref_folder)
    ref_mosaic = next(m for m in mosaics if m.folder_name == ref_folder)
    return ref_var, ref_mosaic


def make_aligned_vrt(
    src_path: Path,
    ref_crs: CRS,
    ref_transform: rasterio.Affine,
    ref_width: int,
    ref_height: int,
    resampling: Resampling
) -> WarpedVRT:
    src = rasterio.open(src_path)
    return WarpedVRT(
        src,
        crs=ref_crs,
        transform=ref_transform,
        width=ref_width,
        height=ref_height,
        resampling=resampling
    )


def reference_grid_from_bbox(
    ref_dataset_path: Path,
    portugal_bounds: Tuple[float, float, float, float]
) -> Tuple[rasterio.Affine, int, int]:
    """
    Defines the reference grid as the window of the reference dataset covering Portugal bounds.
    The aligned rasters will match this grid exactly (for the bbox region).
    """
    with rasterio.open(ref_dataset_path) as src:
        win = from_bounds(*portugal_bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        transform = rasterio.windows.transform(win, src.transform)
        width = int(win.width)
        height = int(win.height)
    return transform, width, height


def build_aligned_raster_set_for_index(
    index_def: Dict[str, Any],
    data_root: str | Path,
    portugal_bbox_geojson: str | Path
) -> AlignedRasterSet:
    """
    Full pipeline for ONE index definition:
    - Load all needed folders as VRT mosaics (ALL tiles overlapping bbox)
    - Verify Mollweide
    - Choose coarsest mosaic as reference
    - Align all mosaics to ref grid (bbox-only grid)
    """
    data_root = Path(data_root)
    bind = required_folders_from_index(index_def)

    # Step 1: determine Mollweide CRS by reading any raster from the first folder
    first_folder = next(iter(bind.values()))
    any_tif = list_tifs_in_folder(data_root / first_folder)[0]
    with rasterio.open(any_tif) as src0:
        if src0.crs is None:
            raise ValueError(f"Raster has no CRS: {any_tif}")
        assert_mollweide_or_raise(src0.crs, f"{first_folder}/{any_tif.name}")
        moll_crs = src0.crs

    # Step 2: compute Portugal bounds in Mollweide coords
    bbox_gdf = read_bbox_geometry(portugal_bbox_geojson)
    portugal_bounds_moll = reproject_bounds_to_crs(bbox_gdf, moll_crs)

    # Step 3: build mosaics for each required folder (unique folders only)
    tmp_dir = Path(tempfile.mkdtemp(prefix="folder_vrts_"))
    unique_folders = sorted(set(bind.values()))
    mosaics: Dict[str, FolderMosaic] = {}
    for folder_name in unique_folders:
        mosaics[folder_name] = open_folder_mosaic(
            data_root=data_root,
            folder_name=folder_name,
            portugal_bounds_moll=portugal_bounds_moll,
            tmp_dir=tmp_dir
        )

    # Step 4: choose reference based on coarsest folder mosaic
    ref_var, ref_mosaic = choose_reference_coarsest(list(mosaics.values()), bind)

    # Step 5: define reference grid as the bbox-window grid on the reference mosaic
    ref_transform, ref_width, ref_height = reference_grid_from_bbox(ref_mosaic.vrt_path, portugal_bounds_moll)

    # Step 6: build aligned WarpedVRTs for each bound variable, using its folder mosaic VRT as the source
    vrt_map: Dict[str, WarpedVRT] = {}
    for var_name, folder_name in bind.items():
        src_vrt_path = mosaics[folder_name].vrt_path
        rs = default_resampling_for_folder(folder_name)
        vrt_map[var_name] = make_aligned_vrt(
            src_path=src_vrt_path,
            ref_crs=moll_crs,
            ref_transform=ref_transform,
            ref_width=ref_width,
            ref_height=ref_height,
            resampling=rs
        )

    return AlignedRasterSet(
        ref_var=ref_var,
        ref_source=ref_mosaic.vrt_path,
        crs=moll_crs,
        transform=ref_transform,
        width=ref_width,
        height=ref_height,
        bounds=portugal_bounds_moll,
        vrt_map=vrt_map,
        tmp_vrts=[tmp_dir]  # keep dir for later cleanup
    )


# ----------------------------
# Reading helpers
# ----------------------------

def read_all_arrays(
    aligned: AlignedRasterSet,
    masked: bool = True,
    band: int = 1,
    dtype: np.dtype = np.float32
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}

    for var_name, vrt in aligned.vrt_map.items():
        arr = vrt.read(band, masked=masked)

        if np.ma.isMaskedArray(arr):
            # Convert first (so NaN is representable even if original is uint16/int32)
            arrays[var_name] = arr.astype(dtype).filled(np.nan)
        else:
            arrays[var_name] = np.asarray(arr, dtype=dtype)

    return arrays


def close_aligned(aligned: AlignedRasterSet, cleanup_tmp: bool = False) -> None:
    """
    Close VRTs and optionally delete temporary VRT directory.
    """
    for vrt in aligned.vrt_map.values():
        try:
            vrt.close()
        except Exception:
            pass

    if cleanup_tmp:
        for p in aligned.tmp_vrts:
            try:
                # tmp_vrts stores the directory Path
                import shutil
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    paths = load_paths()
    example_index = {
        "id": "example_pm25",
        "bind": {
            "delta": "PM_2_5_delta",
            "pop": "Population"
        }
    }

    data_root = path_value(paths, "sat_data_curated")
    bbox_path = repo_data_path(paths, "parishes_bounding_box.geojson")

    aligned = build_aligned_raster_set_for_index(example_index, data_root, bbox_path)
    arrays = read_all_arrays(aligned)

    print("Vars:", list(arrays.keys()))
    for k, v in arrays.items():
        print(k, v.shape, np.nanmin(v), np.nanmax(v))

    close_aligned(aligned, cleanup_tmp=False)
