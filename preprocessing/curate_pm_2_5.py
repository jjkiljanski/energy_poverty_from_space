#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr

# Required for GeoTIFF writing
import rioxarray  # noqa: F401  (adds .rio accessor)


# ----------------------------
# USER SETTINGS
# ----------------------------
IN_DIR = Path(r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_orig\SatPM2.5")

OUT_DIR_DELTA = Path(r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_curated\PM_2_5_delta")
OUT_DIR_AVG = Path(r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_curated\PM_2_5_average")

# years to use
YEARS = (2010, 2011, 2012)

# Heating season definition (Portugal): January + February
HEATING_MONTHS = (1, 2)

# Filename pattern: ...YYYYMM-YYYYMM_cropped.nc
FNAME_RE = re.compile(r".*\.(\d{6})-\d{6}_cropped\.nc$", re.IGNORECASE)

# If you know the variable name, set it here (e.g. "PM25" / "CNNPM25" / etc.)
PM25_VAR_NAME: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wkt_epsg4326() -> str:
    # Enough for your manifest style; matches many tools
    return 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'


def find_nc_files(in_dir: Path) -> List[Path]:
    files = sorted(in_dir.glob("*_cropped.nc"))
    return files


def parse_year_month(p: Path) -> Tuple[int, int]:
    m = FNAME_RE.match(p.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern '*YYYYMM-YYYYMM_cropped.nc': {p.name}")
    yyyymm = m.group(1)
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])
    return year, month


def choose_pm25_var(ds: xr.Dataset) -> str:
    if PM25_VAR_NAME is not None:
        if PM25_VAR_NAME not in ds.data_vars:
            raise ValueError(f"PM25_VAR_NAME='{PM25_VAR_NAME}' not found in data_vars={list(ds.data_vars)}")
        return PM25_VAR_NAME

    # Heuristic: pick first numeric data_var (excluding coords), prefer ones containing 'pm' or 'pm25'
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
        # fall back to first data_var
        return candidates[0]

    scored.sort(reverse=True)
    return scored[0][1]


def standardize_lon_lat(da: xr.DataArray) -> xr.DataArray:
    """
    Try to ensure the DataArray has identifiable spatial dims and CRS for GeoTIFF writing.
    Common dims: ('lat','lon') or ('latitude','longitude').

    Returns a DataArray with dims named ('y','x') for rioxarray if possible.
    """
    dims = list(da.dims)

    # Identify lat/lon dim names
    lat_names = ["lat", "latitude", "y"]
    lon_names = ["lon", "longitude", "x"]

    lat_dim = next((d for d in dims if d.lower() in lat_names), None)
    lon_dim = next((d for d in dims if d.lower() in lon_names), None)

    if lat_dim is None or lon_dim is None:
        # Some NetCDFs might already be x/y; try coords
        for d in dims:
            if d.lower() == "lat":
                lat_dim = d
            if d.lower() == "lon":
                lon_dim = d

    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Could not identify lat/lon dims in {dims}. Please set PM25_VAR_NAME and share ds.dims/ds.coords.")

    # Rename to y/x for raster writing
    da2 = da.rename({lat_dim: "y", lon_dim: "x"})

    # Ensure CRS
    da2 = da2.rio.write_crs("EPSG:4326", inplace=False)

    # Ensure spatial dims
    da2 = da2.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    return da2


def write_geotiff(da: xr.DataArray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # rioxarray expects 2D (y,x). If there's a singleton time dim, squeeze it.
    da2 = da
    for d in list(da2.dims):
        if d not in ("y", "x") and da2.sizes.get(d, 0) == 1:
            da2 = da2.squeeze(d, drop=True)

    if set(da2.dims) != {"y", "x"}:
        raise ValueError(f"GeoTIFF writer expects 2D (y,x). Got dims={da2.dims}")

    da2.rio.to_raster(out_path.as_posix(), compress="DEFLATE")


def manifest_for_file(
    folder: Path,
    file_name: str,
    src_path: str,
    dst_path: str,
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
                    "action": "derived",
                    "src_crs": "EPSG:4326",
                    "dst_crs": "EPSG:4326",
                    "reprojected": False,
                    "copied_as_is": False,
                    "resampling": None,
                    "target_resolution_m": None,
                    "output_resolution": output_resolution,
                    "target_crs": wkt_epsg4326(),
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

    missing_years = [y for y in YEARS if y not in by_year]
    if missing_years:
        raise SystemExit(f"Missing data for year(s): {missing_years}. Found years={sorted(by_year.keys())}")

    # Load data as monthly rasters and compute:
    # - avg PM25 across all months 2010-2012
    # - delta per year: mean(Jan-Feb) - mean(Mar-Dec), then avg across years

    all_month_arrays: List[xr.DataArray] = []
    deltas_per_year: List[xr.DataArray] = []
    template_da: Optional[xr.DataArray] = None
    var_used: Optional[str] = None

    for y in YEARS:
        months_map = by_year[y]

        # Ensure we have at least Jan and Feb for that year
        for hm in HEATING_MONTHS:
            if hm not in months_map:
                raise SystemExit(f"Year {y} missing heating month {hm:02d} file.")

        # Load all months that exist for this year (we expect 1..12)
        year_month_das: Dict[int, xr.DataArray] = {}
        for m, fp in sorted(months_map.items()):
            ds = xr.open_dataset(fp, engine="netcdf4")
            try:
                if var_used is None:
                    var_used = choose_pm25_var(ds)
                da = ds[var_used]

                # Treat invalid values (< -100) as NaN
                da = da.where(da >= -100)

                # Drop any non-spatial singleton dims (e.g., time=1)
                for d in list(da.dims):
                    if d.lower() in ("time", "t") and da.sizes.get(d, 0) == 1:
                        da = da.squeeze(d, drop=True)

                # Standardize to y/x and CRS
                da = standardize_lon_lat(da)

                # Keep a template for resolution/shape checks
                if template_da is None:
                    template_da = da
                else:
                    # basic shape consistency check
                    if da.sizes["y"] != template_da.sizes["y"] or da.sizes["x"] != template_da.sizes["x"]:
                        raise ValueError(f"Grid shape mismatch in {fp.name}")

                year_month_das[m] = da
                all_month_arrays.append(da)
            finally:
                ds.close()

        # Compute heating mean
        heat = xr.concat([year_month_das[m] for m in HEATING_MONTHS], dim="month").mean("month", skipna=True)

        # Compute rest-of-year mean: all other months present except heating months
        rest_months = [m for m in sorted(year_month_das.keys()) if m not in HEATING_MONTHS]
        if not rest_months:
            raise SystemExit(f"Year {y}: no non-heating months found, cannot compute delta.")

        rest = xr.concat([year_month_das[m] for m in rest_months], dim="month").mean("month", skipna=True)

        delta_y = heat - rest
        delta_y = delta_y.assign_attrs({"long_name": f"PM2.5 delta (Jan-Feb minus rest) for {y}", "units": year_month_das[1].attrs.get("units", "")})
        deltas_per_year.append(delta_y)

    # Average across all months (2010-2012)
    pm25_avg = xr.concat(all_month_arrays, dim="month_all").mean("month_all", skipna=True)
    pm25_avg = pm25_avg.assign_attrs({"long_name": "PM2.5 average (2010-2012)", "units": template_da.attrs.get("units", "") if template_da is not None else ""})

    # Average delta across years
    pm25_delta = xr.concat(deltas_per_year, dim="year").mean("year", skipna=True)
    pm25_delta = pm25_delta.assign_attrs({"long_name": "PM2.5 delta (Jan-Feb minus rest), averaged over 2010-2012", "units": template_da.attrs.get("units", "") if template_da is not None else ""})

    # Output filenames
    avg_tif_name = "pm25_average_2010_2012.tif"
    delta_tif_name = "pm25_delta_JanFeb_minus_rest_2010_2012.tif"

    avg_out_path = OUT_DIR_AVG / avg_tif_name
    delta_out_path = OUT_DIR_DELTA / delta_tif_name

    # Write GeoTIFFs
    write_geotiff(pm25_avg, avg_out_path)
    write_geotiff(pm25_delta, delta_out_path)

    # Determine output resolution in degrees (approx from coords)
    # (Assumes regular grid)
    if template_da is None:
        raise SystemExit("No template DataArray available (unexpected).")

    x = template_da["x"].values
    y = template_da["y"].values
    # Handle ascending/descending coords
    dx = float(np.abs(x[1] - x[0])) if x.size > 1 else float("nan")
    dy = float(np.abs(y[1] - y[0])) if y.size > 1 else float("nan")
    output_resolution = f"{dx} x {dy}"

    # Build manifests (one per output folder, like your examples)
    src_folder_str = str(IN_DIR)

    avg_manifest = manifest_for_file(
        folder=OUT_DIR_AVG,
        file_name=avg_tif_name,
        src_path=src_folder_str,
        dst_path=str(avg_out_path),
        output_resolution=output_resolution,
        postprocess=[
            f"Computed mean PM2.5 over all available months for years {YEARS[0]}-{YEARS[-1]} (source: {IN_DIR.name}, var: {var_used})"
        ],
    )

    delta_manifest = manifest_for_file(
        folder=OUT_DIR_DELTA,
        file_name=delta_tif_name,
        src_path=src_folder_str,
        dst_path=str(delta_out_path),
        output_resolution=output_resolution,
        postprocess=[
            f"Heating season months: {', '.join(str(m).zfill(2) for m in HEATING_MONTHS)} (Jan-Feb)",
            "For each year: delta = mean(heating months) - mean(all other months)",
            f"Final delta = mean(delta_year) across years {YEARS[0]}-{YEARS[-1]} (source: {IN_DIR.name}, var: {var_used})",
        ],
    )

    save_manifest(OUT_DIR_AVG, avg_manifest)
    save_manifest(OUT_DIR_DELTA, delta_manifest)

    print("✅ Done.")
    print(f"Average GeoTIFF: {avg_out_path}")
    print(f"Delta GeoTIFF:   {delta_out_path}")
    print(f"Manifests written to: {OUT_DIR_AVG / 'curation_manifest.json'} and {OUT_DIR_DELTA / 'curation_manifest.json'}")


if __name__ == "__main__":
    main()
