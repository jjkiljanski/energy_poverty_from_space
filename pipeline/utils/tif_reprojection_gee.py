# ---------------------------------------------------------------------
# Minimal, transparent reprojection/alignment utils for Google Earth Engine (GEE)
# Designed to mirror the local rasterio helpers, but for ee.Image assets.
#
# Requires:
#   pip install earthengine-api
# and in your notebook/script:
#   import ee
#   ee.Initialize()
#
# Key idea:
# - In GEE you align ee.Images, not TIFF files.
# - "Downsampling with weighted overlap" is handled by reduceResolution(reducer=...)
#   which is GEE's explicit equivalent of overlap-weighted aggregation.
# - Final "snapping" to the target grid is enforced via .reproject(target.projection()).
# - Export to an Asset to persist the aligned image.
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import ee

VarType = Literal["count", "share", "continuous", "categorical", "binary"]
EqualPrefer = Literal["A_to_B", "B_to_A"]


# ---------------------------
# Grid spec / inspection
# ---------------------------

@dataclass(frozen=True)
class EEGridSpec:
    """
    Lightweight description of an ee.Image grid.

    Fields are server-side objects except crs (string-ish).
    - crs: ee.String (or python str in some contexts)
    - nominal_scale_m: ee.Number
    - transform: ee.List (may be None for some images)
    """
    crs: Any
    nominal_scale_m: ee.Number
    transform: Optional[ee.List] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"crs": self.crs, "nominal_scale_m": self.nominal_scale_m, "transform": self.transform}


def gee_grid_from_image(img: ee.Image) -> EEGridSpec:
    """
    Equivalent of grid_from_tiff(), but for ee.Image.
    Returns CRS + nominal scale (+ transform if available).

    Note: nominalScale() is in meters for projected CRS.
    For EPSG:4326 it returns meters at the equator-ish; still OK for
    deciding 'finer vs coarser' most of the time.
    """
    proj = img.projection()
    crs = proj.crs()
    scale = proj.nominalScale()
    try:
        transform = proj.transform()
    except Exception:
        transform = None
    return EEGridSpec(crs=crs, nominal_scale_m=scale, transform=transform)


# ---------------------------
# Variable-type logic
# ---------------------------

def gee_reducer_for_var(var_type: VarType) -> ee.Reducer:
    """
    Reducer used for explicit downsampling via reduceResolution().
    """
    if var_type == "count":
        return ee.Reducer.sum()
    if var_type in ("share", "continuous"):
        return ee.Reducer.mean()
    if var_type in ("categorical", "binary"):
        return ee.Reducer.mode()
    raise ValueError(f"Unknown var_type='{var_type}'")


def gee_resample_method_for_var(var_type: VarType) -> str:
    """
    GEE supports 'nearest', 'bilinear', 'bicubic'.

    We use this mainly for upsampling / smoothing after reduction.
    For categorical/count, nearest is safest.
    """
    if var_type in ("categorical", "binary", "count"):
        return "nearest"
    return "bilinear"


def gee_align_image_to_target_grid(
    src_img: ee.Image,
    src_var_type: VarType,
    target_img: ee.Image,
    *,
    max_pixels: int = 4096,
    do_downsample_reduce: bool = True,
) -> ee.Image:
    """
    Align src_img onto target_img's grid.

    Steps (transparent):
      1) If downsampling is needed and do_downsample_reduce=True:
         apply reduceResolution with reducer based on src_var_type
         (sum for counts, mean for continuous/share, mode for categorical).
      2) Set resampling method (nearest/bilinear).
      3) Snap to target grid using reproject(target_img.projection()).

    This function DOES NOT clip; do that separately if needed.

    Returns:
      ee.Image aligned to target_img's projection.
    """
    src_grid = gee_grid_from_image(src_img)
    tgt_grid = gee_grid_from_image(target_img)

    src_scale = ee.Number(src_grid.nominal_scale_m)
    tgt_scale = ee.Number(tgt_grid.nominal_scale_m)

    needs_downsample = src_scale.lt(tgt_scale)

    out = src_img

    if do_downsample_reduce:
        out = ee.Image(
            ee.Algorithms.If(
                needs_downsample,
                out.reduceResolution(reducer=gee_reducer_for_var(src_var_type), maxPixels=max_pixels),
                out,
            )
        )

    # Resample (important mostly for upsampling / visualization)
    out = out.resample(gee_resample_method_for_var(src_var_type))

    # Snap to target grid
    out = out.reproject(target_img.projection())

    return out


# ---------------------------
# Main helper: gee_align_two_tiffs_auto
# ---------------------------

def gee_align_two_tiffs_auto(
    asset_a: str,
    var_type_a: VarType,
    asset_b: str,
    var_type_b: VarType,
    *,
    equal_grid_prefer: EqualPrefer = "A_to_B",
    scale_tol_m: float = 1e-6,
    max_pixels: int = 4096,
    do_downsample_reduce: bool = True,
) -> Tuple[ee.Image, ee.String]:
    """
    GEE equivalent of your local align_two_tiffs_auto().

    It loads two raster *assets* (GeoTIFFs uploaded as GEE assets or any asset images),
    automatically detects which one is finer/coarser using nominal pixel size,
    and returns the aligned version of the finer (or the preferred one if equal scale),
    snapped to the other's grid.

    Parameters:
      asset_a, asset_b: GEE asset IDs, e.g. "projects/your-project/assets/VIIRS_2019"
      var_type_a, var_type_b: variable types for BOTH rasters. Only the type of the
        raster being aligned is used, but you pass both so you don't need to remember which.
      equal_grid_prefer:
        - "A_to_B": if scales equal, align A to B grid
        - "B_to_A": if scales equal, align B to A grid
      scale_tol_m: tolerance for treating scales as equal
      max_pixels: passed to reduceResolution (controls downsampling behavior)
      do_downsample_reduce: if True, uses reduceResolution for downsampling (recommended)

    Returns:
      (aligned_image, which_was_aligned) where which_was_aligned is ee.String("A") or ee.String("B").

    Example usage:
        aligned_img, who = gee_align_two_tiffs_auto(
            "projects/myproj/assets/GHS_POP_100m", "count",
            "projects/myproj/assets/VIIRS_2019_500m", "continuous",
            equal_grid_prefer="A_to_B",
            max_pixels=4096
        )
        # aligned_img is POP aligned to VIIRS grid (sum-conserving downsample)
        # who is "A"

    Notes:
      - This function returns only the aligned *one* (the one chosen to be resampled),
        matching your local design.
      - For exact identical origin/transform snapping, reproject(target.projection())
        is the simplest robust approach when both assets are already in the same CRS.
    """
    img_a = ee.Image(asset_a)
    img_b = ee.Image(asset_b)

    grid_a = gee_grid_from_image(img_a)
    grid_b = gee_grid_from_image(img_b)

    a_scale = ee.Number(grid_a.nominal_scale_m)
    b_scale = ee.Number(grid_b.nominal_scale_m)

    is_equal = a_scale.subtract(b_scale).abs().lte(ee.Number(scale_tol_m))
    a_finer = a_scale.lt(b_scale)

    # Precompute both possible alignments
    a_to_b = gee_align_image_to_target_grid(
        src_img=img_a,
        src_var_type=var_type_a,
        target_img=img_b,
        max_pixels=max_pixels,
        do_downsample_reduce=do_downsample_reduce,
    )
    b_to_a = gee_align_image_to_target_grid(
        src_img=img_b,
        src_var_type=var_type_b,
        target_img=img_a,
        max_pixels=max_pixels,
        do_downsample_reduce=do_downsample_reduce,
    )

    # Equal: choose based on preference; Not equal: choose finer -> coarser
    aligned_img = ee.Image(
        ee.Algorithms.If(
            is_equal,
            ee.Algorithms.If(equal_grid_prefer == "A_to_B", a_to_b, b_to_a),
            ee.Algorithms.If(a_finer, a_to_b, b_to_a),
        )
    )

    aligned_label = ee.String(
        ee.Algorithms.If(
            is_equal,
            ee.Algorithms.If(equal_grid_prefer == "A_to_B", "A", "B"),
            ee.Algorithms.If(a_finer, "A", "B"),
        )
    )

    return aligned_img, aligned_label


# ---------------------------
# Export helper: write aligned image to an asset
# ---------------------------

def gee_export_image_to_asset(
    img: ee.Image,
    out_asset_id: str,
    region: ee.Geometry,
    *,
    description: str = "aligned_export",
    max_pixels: int = 1_000_000_000_000,
    pyramiding_policy: Optional[Dict[str, str]] = None,
) -> ee.batch.Task:
    """
    Export an ee.Image to an Earth Engine Asset.

    - region: geometry to export (use Portugal boundary/buffer)
    - pyramiding_policy: optionally set per-band policy:
        {"band1": "mean"} for continuous
        {"band1": "mode"} for categorical
        {"band1": "sample"} sometimes for classes
      If None, EE picks defaults.

    Returns:
      The started ee.batch.Task. (You still need to monitor it in the EE Tasks tab.)
    """
    kwargs: Dict[str, Any] = dict(
        image=img,
        description=description,
        assetId=out_asset_id,
        region=region,
        maxPixels=max_pixels,
    )
    if pyramiding_policy is not None:
        kwargs["pyramidingPolicy"] = pyramiding_policy

    task = ee.batch.Export.image.toAsset(**kwargs)
    task.start()
    return task


# ---------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------