#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

# Adds .rio accessor
import rioxarray  # noqa: F401


# ----------------------------
# USER SETTINGS
# ----------------------------
IN_DIR_BC = Path(
    r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_orig\CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_bc_v6.2_monthly"
)
IN_DIR_OC = Path(
    r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_orig\CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_oc_v6.2_monthly"
)

OUT_BASE = Path(
    r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_curated"
)

OUT_DIR_OC_DELTA = OUT_BASE / "OC_delta"
OUT_DIR_OC_AVG = OUT_BASE / "OC_average"
OUT_DIR_BC_DELTA = OUT_BASE / "BC_delta"
OUT_DIR_BC_AVG = OUT_BASE / "BC_average"

YEARS = (2010, 2011, 2012)

# Heating season definition: Jan-Feb
HEATING_MONTHS = (1, 2)

# Output CRS: Mollweide
TARGET_CRS = "ESRI:54009"

# Target resolution (meters). 0.1° source is ~10km; 10,000 m keeps roughly native scale.
TARGET_RESOLUTION_M: Optional[float] = 10000.0

# Typical NetCDF fill value; we will also treat any absurdly large values as NoData
FILL_DEFAULT = 9.96921e36
FILL_THRESHOLD = 1e20

# Chunk by time to keep memory stable when using skipna reductions
CHUNKS = {"time": 1}

# Regex to find the year in filename
YEAR_RE = re.compile(r"(19|20)\d{2}")


# ----------------------------
# Helpers
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wkt_mollweide_esri54009() -> str:
    return (
        'PROJCS["World_Mollweide",'
        'GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],'
        'PROJECTION["Mollweide"],PARAMETER["Central_Meridian",0],'
        'PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],UNIT["Meter",1]]'
    )


def find_yearly_files(in_dir: Path) -> Dict[int, Path]:
    """
    Returns {year: file_path} for years found in directory.
    Accepts any file extension (NetCDF typically .nc/.nc4, sometimes none).
    """
    candidates = sorted([p for p in in_dir.iterdir() if p.is_file()])
    out: Dict[int, Path] = {}
    for p in candidates:
        m = YEAR_RE.search(p.name)
        if not m:
            continue
        y = int(m.group(0))
        if y in YEARS:
            out[y] = p
    return out


def choose_emissions_var(ds: xr.Dataset) -> str:
    """
    Your CAMS-GLOB-ANT files (as you printed) have only one numeric data_var: 'res'.
    We still keep a robust selector in case that's not always true.
    """
    if not ds.data_vars:
        raise ValueError("Dataset has no data variables.")

    # Prefer exact 'res' if present
    if "res" in ds.data_vars:
        return "res"

    # Otherwise pick the first numeric var
    for v in ds.data_vars:
        da = ds[v]
        if np.issubdtype(da.dtype, np.number):
            return v

    return list(ds.data_vars)[0]


def standardize_lon_lat_to_rio(da: xr.DataArray) -> xr.DataArray:
    """
    Rename lon/lat dims to x/y for rioxarray and attach EPSG:4326.
    """
    dims = list(da.dims)
    lat_names = ["lat", "latitude", "y"]
    lon_names = ["lon", "longitude", "x"]

    lat_dim = next((d for d in dims if d.lower() in lat_names), None)
    lon_dim = next((d for d in dims if d.lower() in lon_names), None)

    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Could not identify lat/lon dims in {dims}.")

    da2 = da.rename({lat_dim: "y", lon_dim: "x"})
    da2 = da2.rio.write_crs("EPSG:4326", inplace=False)
    da2 = da2.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    return da2


def write_geotiff_wgs84(da: xr.DataArray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    da2 = da
    # Squeeze singleton dims
    for d in list(da2.dims):
        if d not in ("y", "x") and da2.sizes.get(d, 0) == 1:
            da2 = da2.squeeze(d, drop=True)

    if tuple(da2.dims) != ("y", "x"):
        da2 = da2.transpose("y", "x")

    da2 = da2.astype("float32", copy=False)
    da2.rio.to_raster(out_path.as_posix(), compress="DEFLATE")


def reproject_tif_to_mollweide(src_tif: Path, dst_tif: Path) -> None:
    """
    Reproject a GeoTIFF to Mollweide (ESRI:54009) using rioxarray.
    """
    dst_tif.parent.mkdir(parents=True, exist_ok=True)

    da = xr.open_dataarray(src_tif.as_posix(), engine="rasterio")
    try:
        if TARGET_RESOLUTION_M is not None:
            da_reproj = da.rio.reproject(TARGET_CRS, resolution=TARGET_RESOLUTION_M)
        else:
            da_reproj = da.rio.reproject(TARGET_CRS)

        da_reproj = da_reproj.astype("float32", copy=False)
        da_reproj.rio.to_raster(dst_tif.as_posix(), compress="DEFLATE")
    finally:
        try:
            da.close()
        except Exception:
            pass
        try:
            da_reproj.close()
        except Exception:
            pass


def manifest_for_file(
    folder: Path,
    file_name: str,
    src_path: str,
    dst_path: str,
    src_crs: str,
    dst_crs: str,
    reprojected: bool,
    output_resolution: str,
    postprocess: List[str],
) -> Dict:
    return {
        "folder": str(folder),
        "generated_utc": utc_now_iso(),
        "files": {
            file_name: {
                "file": {
                    "name": file_name,
                    "src_path": src_path,
                    "dst_path": dst_path,
                },
                "general": {
                    "action": "reprojected" if reprojected else "derived",
                    "src_crs": src_crs,
                    "dst_crs": dst_crs,
                    "reprojected": reprojected,
                    "copied_as_is": False,
                    "resampling": None,
                    "target_resolution_m": TARGET_RESOLUTION_M,
                    "output_resolution": output_resolution,
                    "target_crs": wkt_mollweide_esri54009()
                    if dst_crs.upper().startswith("ESRI:54009")
                    else dst_crs,
                },
                "specific": {"postprocess": postprocess},
            }
        },
    }


def save_manifest(folder: Path, manifest: Dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / "curation_manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ----------------------------
# Core computation
# ----------------------------
def compute_delta_and_average(
    in_dir: Path,
    src_label: str,
) -> Tuple[xr.DataArray, xr.DataArray, str, str]:
    """
    Returns:
      (avg_2010_2012, delta_2010_2012, var_used, out_units)

    Input units: kg m-2 s-1 (as in your debug output)
    We convert to: kg m-2 month-1 (multiply by seconds in each month)
    delta per year: mean(Jan-Feb) - mean(all other months)
    Final delta: mean across years 2010-2012
    Final avg: mean across ALL months 2010-2012 (weighted by month length, since we convert per-month)
    """
    year_files = find_yearly_files(in_dir)
    missing = [y for y in YEARS if y not in year_files]
    if missing:
        raise SystemExit(f"Missing files for years {missing} in: {in_dir}")

    deltas_per_year: List[xr.DataArray] = []
    template_shape: Optional[Tuple[int, int]] = None
    var_used: Optional[str] = None

    # For overall average across all months across all years:
    # We'll compute sum of per-month emissions and count of months (not seconds).
    total_sum: Optional[xr.DataArray] = None
    total_count: Optional[xr.DataArray] = None

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    out_units = "kg m-2 month-1"

    for y in YEARS:
        fp = year_files[y]
        ds = xr.open_dataset(
            fp.as_posix(),
            engine="netcdf4",
            decode_times=time_coder,
            mask_and_scale=True,
            chunks=CHUNKS,
        )
        try:
            if var_used is None:
                var_used = choose_emissions_var(ds)

            da = ds[var_used]

            if "time" not in da.dims:
                raise ValueError(f"[{src_label}] No 'time' dimension in {fp.name}. dims={da.dims}")

            # --- NoData handling (common CAMS fill is ~9.96921e36) ---
            # Prefer file-provided fill if present; otherwise use default.
            fill = da.attrs.get("_FillValue", None)
            if fill is None:
                fill = da.attrs.get("missing_value", None)
            if fill is None:
                fill = FILL_DEFAULT

            # Robust: treat any absurdly large value as nodata (covers slight float rounding too)
            da = da.where(da < FILL_THRESHOLD)

            # Convert to standard x/y for rioxarray
            da = standardize_lon_lat_to_rio(da)

            # Convert from kg m-2 s-1 to kg m-2 month-1 (broadcast explicitly)
            sec = da["time"].dt.days_in_month.astype("float64") * 86400.0
            sec3d = sec.broadcast_like(da)
            da = da * sec3d
            da.attrs["units"] = out_units

            # Grid consistency check
            shape = (da.sizes["y"], da.sizes["x"])
            if template_shape is None:
                template_shape = shape
            elif shape != template_shape:
                raise ValueError(
                    f"[{src_label}] Grid shape mismatch in {fp.name}: got {shape}, expected {template_shape}"
                )

            months = da["time"].dt.month

            # Heating vs rest means (ignore NaNs from fill)
            heat = da.where(months.isin(HEATING_MONTHS), drop=True).mean("time", skipna=True)
            rest = da.where(~months.isin(HEATING_MONTHS), drop=True).mean("time", skipna=True)

            deltas_per_year.append(heat - rest)

            # Streaming overall mean across all months:
            sum_y = da.sum("time", skipna=True, dtype="float64")
            cnt_y = da.count("time")  # counts non-NaN months per pixel

            if total_sum is None:
                total_sum = sum_y
                total_count = cnt_y
            else:
                total_sum = total_sum + sum_y
                total_count = total_count + cnt_y

        finally:
            ds.close()

    if total_sum is None or total_count is None:
        raise RuntimeError(f"[{src_label}] Failed to compute streaming sums (no data?).")
    if not deltas_per_year:
        raise RuntimeError(f"[{src_label}] No deltas computed (empty deltas_per_year).")

    avg = (total_sum / total_count).astype("float32")
    delta = xr.concat(deltas_per_year, dim="year").mean("year", skipna=True).astype("float32")

    return avg, delta, (var_used or "UNKNOWN"), out_units


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    # ---- Compute OC ----
    oc_avg, oc_delta, oc_var_used, oc_units = compute_delta_and_average(
        in_dir=IN_DIR_OC,
        src_label="OC",
    )

    # ---- Compute BC ----
    bc_avg, bc_delta, bc_var_used, bc_units = compute_delta_and_average(
        in_dir=IN_DIR_BC,
        src_label="BC",
    )

    # If dask is in play, make sure results are computed before writing rasters
    try:
        oc_avg = oc_avg.compute()
        oc_delta = oc_delta.compute()
        bc_avg = bc_avg.compute()
        bc_delta = bc_delta.compute()
    except Exception:
        # compute() may not exist if dask isn't used; safe to ignore
        pass

    # ---- Write temporary WGS84 GeoTIFFs first ----
    OUT_DIR_OC_AVG.mkdir(parents=True, exist_ok=True)
    OUT_DIR_OC_DELTA.mkdir(parents=True, exist_ok=True)
    OUT_DIR_BC_AVG.mkdir(parents=True, exist_ok=True)
    OUT_DIR_BC_DELTA.mkdir(parents=True, exist_ok=True)

    oc_avg_tmp = OUT_DIR_OC_AVG / "oc_average_2010_2012_wgs84.tif"
    oc_delta_tmp = OUT_DIR_OC_DELTA / "oc_delta_JanFeb_minus_rest_2010_2012_wgs84.tif"

    bc_avg_tmp = OUT_DIR_BC_AVG / "bc_average_2010_2012_wgs84.tif"
    bc_delta_tmp = OUT_DIR_BC_DELTA / "bc_delta_JanFeb_minus_rest_2010_2012_wgs84.tif"

    write_geotiff_wgs84(oc_avg, oc_avg_tmp)
    write_geotiff_wgs84(oc_delta, oc_delta_tmp)
    write_geotiff_wgs84(bc_avg, bc_avg_tmp)
    write_geotiff_wgs84(bc_delta, bc_delta_tmp)

    # ---- Reproject to Mollweide ----
    oc_avg_name = "oc_average_2010_2012_mollweide.tif"
    oc_delta_name = "oc_delta_JanFeb_minus_rest_2010_2012_mollweide.tif"
    bc_avg_name = "bc_average_2010_2012_mollweide.tif"
    bc_delta_name = "bc_delta_JanFeb_minus_rest_2010_2012_mollweide.tif"

    oc_avg_out = OUT_DIR_OC_AVG / oc_avg_name
    oc_delta_out = OUT_DIR_OC_DELTA / oc_delta_name
    bc_avg_out = OUT_DIR_BC_AVG / bc_avg_name
    bc_delta_out = OUT_DIR_BC_DELTA / bc_delta_name

    reproject_tif_to_mollweide(oc_avg_tmp, oc_avg_out)
    reproject_tif_to_mollweide(oc_delta_tmp, oc_delta_out)
    reproject_tif_to_mollweide(bc_avg_tmp, bc_avg_out)
    reproject_tif_to_mollweide(bc_delta_tmp, bc_delta_out)

    # Remove temporary WGS84 files
    for tmp in [oc_avg_tmp, oc_delta_tmp, bc_avg_tmp, bc_delta_tmp]:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    # Determine output resolution in meters from one reprojected raster
    da_check = xr.open_dataarray(oc_avg_out.as_posix(), engine="rasterio")
    try:
        res_x, res_y = da_check.rio.resolution()
        output_resolution = f"{abs(res_x)} x {abs(res_y)}"
    finally:
        da_check.close()

    # ---- Manifests ----
    oc_avg_manifest = manifest_for_file(
        folder=OUT_DIR_OC_AVG,
        file_name=oc_avg_name,
        src_path=str(IN_DIR_OC),
        dst_path=str(oc_avg_out),
        src_crs="EPSG:4326",
        dst_crs=TARGET_CRS,
        reprojected=True,
        output_resolution=output_resolution,
        postprocess=[
            f"NoData handling: values >= {FILL_THRESHOLD:.1e} treated as NaN (typical fill ~{FILL_DEFAULT:.5e})",
            f"Converted units from kg m-2 s-1 to {oc_units} using seconds per month",
            f"Computed mean OC emissions over all available months for years {YEARS[0]}-{YEARS[-1]} (var: {oc_var_used})",
            f"Reprojected to {TARGET_CRS} (Mollweide)",
        ],
    )

    oc_delta_manifest = manifest_for_file(
        folder=OUT_DIR_OC_DELTA,
        file_name=oc_delta_name,
        src_path=str(IN_DIR_OC),
        dst_path=str(oc_delta_out),
        src_crs="EPSG:4326",
        dst_crs=TARGET_CRS,
        reprojected=True,
        output_resolution=output_resolution,
        postprocess=[
            f"NoData handling: values >= {FILL_THRESHOLD:.1e} treated as NaN (typical fill ~{FILL_DEFAULT:.5e})",
            f"Converted units from kg m-2 s-1 to {oc_units} using seconds per month",
            f"Heating season months: {', '.join(str(m).zfill(2) for m in HEATING_MONTHS)} (Jan-Feb)",
            "For each year: delta = mean(heating months) - mean(all other months)",
            f"Final delta = mean(delta_year) across years {YEARS[0]}-{YEARS[-1]} (var: {oc_var_used})",
            f"Reprojected to {TARGET_CRS} (Mollweide)",
        ],
    )

    bc_avg_manifest = manifest_for_file(
        folder=OUT_DIR_BC_AVG,
        file_name=bc_avg_name,
        src_path=str(IN_DIR_BC),
        dst_path=str(bc_avg_out),
        src_crs="EPSG:4326",
        dst_crs=TARGET_CRS,
        reprojected=True,
        output_resolution=output_resolution,
        postprocess=[
            f"NoData handling: values >= {FILL_THRESHOLD:.1e} treated as NaN (typical fill ~{FILL_DEFAULT:.5e})",
            f"Converted units from kg m-2 s-1 to {bc_units} using seconds per month",
            f"Computed mean BC emissions over all available months for years {YEARS[0]}-{YEARS[-1]} (var: {bc_var_used})",
            f"Reprojected to {TARGET_CRS} (Mollweide)",
        ],
    )

    bc_delta_manifest = manifest_for_file(
        folder=OUT_DIR_BC_DELTA,
        file_name=bc_delta_name,
        src_path=str(IN_DIR_BC),
        dst_path=str(bc_delta_out),
        src_crs="EPSG:4326",
        dst_crs=TARGET_CRS,
        reprojected=True,
        output_resolution=output_resolution,
        postprocess=[
            f"NoData handling: values >= {FILL_THRESHOLD:.1e} treated as NaN (typical fill ~{FILL_DEFAULT:.5e})",
            f"Converted units from kg m-2 s-1 to {bc_units} using seconds per month",
            f"Heating season months: {', '.join(str(m).zfill(2) for m in HEATING_MONTHS)} (Jan-Feb)",
            "For each year: delta = mean(heating months) - mean(all other months)",
            f"Final delta = mean(delta_year) across years {YEARS[0]}-{YEARS[-1]} (var: {bc_var_used})",
            f"Reprojected to {TARGET_CRS} (Mollweide)",
        ],
    )

    save_manifest(OUT_DIR_OC_AVG, oc_avg_manifest)
    save_manifest(OUT_DIR_OC_DELTA, oc_delta_manifest)
    save_manifest(OUT_DIR_BC_AVG, bc_avg_manifest)
    save_manifest(OUT_DIR_BC_DELTA, bc_delta_manifest)

    print("✅ Done.")
    print(f"OC average (Mollweide): {oc_avg_out}")
    print(f"OC delta   (Mollweide): {oc_delta_out}")
    print(f"BC average (Mollweide): {bc_avg_out}")
    print(f"BC delta   (Mollweide): {bc_delta_out}")
    print("✅ Manifests created in each output folder.")


if __name__ == "__main__":
    main()
