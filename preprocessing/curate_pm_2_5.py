#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr

# Adds .rio accessor
import rioxarray  # noqa: F401


# ----------------------------
# USER SETTINGS
# ----------------------------
IN_DIR = Path(r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_orig\SatPM2.5")

OUT_DIR_DELTA = Path(r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_curated\PM_2_5_delta")
OUT_DIR_AVG = Path(r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_curated\PM_2_5_average")

YEARS = (2010, 2011, 2012)

# Heating season definition
HEATING_MONTHS = (1, 2)

# Invalid value rule
INVALID_LT = -100.0

# Output CRS: Mollweide
TARGET_CRS = "ESRI:54009"

# Target resolution (meters). If None, rioxarray chooses an automatic resolution.
# If you want alignment with other curated layers, set e.g. 1000.0
TARGET_RESOLUTION_M: Optional[float] = 1000.0

# Filename pattern: ...YYYYMM-YYYYMM_cropped.nc
FNAME_RE = re.compile(r".*\.(\d{6})-\d{6}_cropped\.nc$", re.IGNORECASE)

# If you know the variable name, set it here
PM25_VAR_NAME: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wkt_mollweide_esri54009() -> str:
    # ESRI:54009 WKT commonly used (approx; sufficient for manifest readability)
    return (
        'PROJCS["World_Mollweide",'
        'GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],'
        'PROJECTION["Mollweide"],PARAMETER["Central_Meridian",0],'
        'PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],UNIT["Meter",1]]'
    )


def find_nc_files(in_dir: Path) -> List[Path]:
    return sorted(in_dir.glob("*_cropped.nc"))


def parse_year_month(p: Path) -> Tuple[int, int]:
    m = FNAME_RE.match(p.name)
    if not m:
        raise ValueError(f"Filename does not match expected '*YYYYMM-YYYYMM_cropped.nc': {p.name}")
    yyyymm = m.group(1)
    return int(yyyymm[:4]), int(yyyymm[4:6])


def choose_pm25_var(ds: xr.Dataset) -> str:
    if PM25_VAR_NAME is not None:
        if PM25_VAR_NAME not in ds.data_vars:
            raise ValueError(f"PM25_VAR_NAME='{PM25_VAR_NAME}' not in data_vars={list(ds.data_vars)}")
        return PM25_VAR_NAME

    candidates = list(ds.data_vars)
    if not candidates:
        raise ValueError("Dataset has no data variables.")

    scored = []
    for v in candidates:
        da = ds[v]
        if not np.issubdtype(da.dtype, np.number):
            continue
        name = v.lower()
        score = 0
        if "pm25" in name:
            score += 10
        if "pm_25" in name or "pm2_5" in name:
            score += 9
        if "pm" in name:
            score += 3
        scored.append((score, v))

    if not scored:
        return candidates[0]

    scored.sort(reverse=True)
    return scored[0][1]


def standardize_lon_lat_to_rio(da: xr.DataArray) -> xr.DataArray:
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

    # Squeeze singleton non-spatial dims
    da2 = da
    for d in list(da2.dims):
        if d not in ("y", "x") and da2.sizes.get(d, 0) == 1:
            da2 = da2.squeeze(d, drop=True)

    if tuple(da2.dims) != ("y", "x"):
        # Could be ("x","y") or other ordering
        da2 = da2.transpose("y", "x")

    da2.rio.to_raster(out_path.as_posix(), compress="DEFLATE")


def reproject_tif_to_mollweide(src_tif: Path, dst_tif: Path) -> None:
    """
    Reproject a GeoTIFF to Mollweide (ESRI:54009) using rioxarray.
    """
    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    da = xr.open_dataarray(src_tif.as_posix(), engine="rasterio")

    # rioxarray will carry CRS from file
    if TARGET_RESOLUTION_M is not None:
        da_reproj = da.rio.reproject(TARGET_CRS, resolution=TARGET_RESOLUTION_M)
    else:
        da_reproj = da.rio.reproject(TARGET_CRS)

    da_reproj.rio.to_raster(dst_tif.as_posix(), compress="DEFLATE")

    da.close()
    da_reproj.close()


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
                    "target_crs": wkt_mollweide_esri54009() if dst_crs.upper().startswith("ESRI:54009") else dst_crs,
                },
                "specific": {
                    "postprocess": postprocess
                },
            }
        }
    }


def save_manifest(folder: Path, manifest: Dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / "curation_manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ----------------------------
# Main computation
# ----------------------------
def main() -> None:
    nc_files = find_nc_files(IN_DIR)
    if not nc_files:
        raise SystemExit(f"No '*_cropped.nc' files found in {IN_DIR}")

    # Group by year -> month -> filepath
    by_year: Dict[int, Dict[int, Path]] = {}
    for p in nc_files:
        y, m = parse_year_month(p)
        if y in YEARS:
            by_year.setdefault(y, {})[m] = p

    for y in YEARS:
        if y not in by_year:
            raise SystemExit(f"Missing any files for year {y}. Found years={sorted(by_year.keys())}")
        for hm in HEATING_MONTHS:
            if hm not in by_year[y]:
                raise SystemExit(f"Year {y} missing heating month {hm:02d} file.")

    all_month_arrays: List[xr.DataArray] = []
    deltas_per_year: List[xr.DataArray] = []
    template_da: Optional[xr.DataArray] = None
    var_used: Optional[str] = None

    for y in YEARS:
        months_map = by_year[y]
        year_month_das: Dict[int, xr.DataArray] = {}

        for m, fp in sorted(months_map.items()):
            ds = xr.open_dataset(fp, engine="netcdf4")
            try:
                if var_used is None:
                    var_used = choose_pm25_var(ds)

                da = ds[var_used]

                # Apply invalid-value rule: values < -100 are NaN
                da = da.where(da >= INVALID_LT)

                # Drop singleton time dims if present
                for d in list(da.dims):
                    if d.lower() in ("time", "t") and da.sizes.get(d, 0) == 1:
                        da = da.squeeze(d, drop=True)

                da = standardize_lon_lat_to_rio(da)

                if template_da is None:
                    template_da = da
                else:
                    if da.sizes["y"] != template_da.sizes["y"] or da.sizes["x"] != template_da.sizes["x"]:
                        raise ValueError(f"Grid shape mismatch in {fp.name}")

                year_month_das[m] = da
                all_month_arrays.append(da)
            finally:
                ds.close()

        heat = xr.concat([year_month_das[m] for m in HEATING_MONTHS], dim="month").mean("month", skipna=True)
        rest_months = [m for m in sorted(year_month_das.keys()) if m not in HEATING_MONTHS]
        rest = xr.concat([year_month_das[m] for m in rest_months], dim="month").mean("month", skipna=True)

        delta_y = heat - rest
        deltas_per_year.append(delta_y)

    pm25_avg = xr.concat(all_month_arrays, dim="month_all").mean("month_all", skipna=True)
    pm25_delta = xr.concat(deltas_per_year, dim="year").mean("year", skipna=True)

    # Write temporary WGS84 GeoTIFFs first (then reproject)
    OUT_DIR_AVG.mkdir(parents=True, exist_ok=True)
    OUT_DIR_DELTA.mkdir(parents=True, exist_ok=True)

    avg_tmp = OUT_DIR_AVG / "pm25_average_2010_2012_wgs84.tif"
    delta_tmp = OUT_DIR_DELTA / "pm25_delta_JanFeb_minus_rest_2010_2012_wgs84.tif"

    write_geotiff_wgs84(pm25_avg, avg_tmp)
    write_geotiff_wgs84(pm25_delta, delta_tmp)

    # Reproject to Mollweide
    avg_tif_name = "pm25_average_2010_2012_mollweide.tif"
    delta_tif_name = "pm25_delta_JanFeb_minus_rest_2010_2012_mollweide.tif"

    avg_out_path = OUT_DIR_AVG / avg_tif_name
    delta_out_path = OUT_DIR_DELTA / delta_tif_name

    reproject_tif_to_mollweide(avg_tmp, avg_out_path)
    reproject_tif_to_mollweide(delta_tmp, delta_out_path)

    # Optionally remove temporary files
    try:
        avg_tmp.unlink(missing_ok=True)
        delta_tmp.unlink(missing_ok=True)
    except Exception:
        pass

    # Determine output resolution in meters from the reprojected raster metadata
    da_check = xr.open_dataarray(avg_out_path.as_posix(), engine="rasterio")
    try:
        res_x, res_y = da_check.rio.resolution()
        output_resolution = f"{abs(res_x)} x {abs(res_y)}"
    finally:
        da_check.close()

    # Manifests
    src_folder_str = str(IN_DIR)

    avg_manifest = manifest_for_file(
        folder=OUT_DIR_AVG,
        file_name=avg_tif_name,
        src_path=src_folder_str,
        dst_path=str(avg_out_path),
        src_crs="EPSG:4326",
        dst_crs=TARGET_CRS,
        reprojected=True,
        output_resolution=output_resolution,
        postprocess=[
            f"All PM2.5 values < {INVALID_LT} interpreted as NaN",
            f"Computed mean PM2.5 over all available months for years {YEARS[0]}-{YEARS[-1]} (var: {var_used})",
            f"Reprojected to {TARGET_CRS} (Mollweide)",
        ],
    )

    delta_manifest = manifest_for_file(
        folder=OUT_DIR_DELTA,
        file_name=delta_tif_name,
        src_path=src_folder_str,
        dst_path=str(delta_out_path),
        src_crs="EPSG:4326",
        dst_crs=TARGET_CRS,
        reprojected=True,
        output_resolution=output_resolution,
        postprocess=[
            f"All PM2.5 values < {INVALID_LT} interpreted as NaN",
            f"Heating season months: {', '.join(str(m).zfill(2) for m in HEATING_MONTHS)} (Jan-Feb)",
            "For each year: delta = mean(heating months) - mean(all other months)",
            f"Final delta = mean(delta_year) across years {YEARS[0]}-{YEARS[-1]} (var: {var_used})",
            f"Reprojected to {TARGET_CRS} (Mollweide)",
        ],
    )

    save_manifest(OUT_DIR_AVG, avg_manifest)
    save_manifest(OUT_DIR_DELTA, delta_manifest)

    print("✅ Done.")
    print(f"Average (Mollweide): {avg_out_path}")
    print(f"Delta   (Mollweide): {delta_out_path}")
    print(f"Manifests: {OUT_DIR_AVG / 'curation_manifest.json'} and {OUT_DIR_DELTA / 'curation_manifest.json'}")


if __name__ == "__main__":
    main()
