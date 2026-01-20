# curate_raw_tiffs.py
#
# "Good enough" TIFF curation pipeline:
# - Input layout:  Sat_data_raw/<DATA_TYPE>/*.tif(f)
# - Output layout: Sat_data_curated/<DATA_TYPE>/*.tif(f)  (same filenames)
#
# What it does per file:
# 1) GENERAL: ensure Mollweide CRS
#    - If not Mollweide: reproject to Mollweide (window-by-window) using WarpedVRT (correct + memory-safe)
# 2) SPECIFIC: optional per-dataset postprocessing
#    - Currently for NTL: log(x + 2) on band 1
#
# Extras:
# - Console logging with progress for long files (per N windows)
# - Per-output-folder manifest: curation_manifest.json
#   Includes: action, whether reprojected/copied, CRS info, output resolution, and postprocess steps

from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


# Mollweide CRS (meters). Using a PROJ string avoids EPSG/ESRI quirks.
MOLLWEIDE_CRS = "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


# ----------------------------
# Config + small utilities
# ----------------------------

@dataclass(frozen=True)
class PipelineConfig:
    raw_root: Path
    curated_root: Path
    mollweide_crs: str = MOLLWEIDE_CRS
    default_resampling: Resampling = Resampling.bilinear


def setup_logging() -> logging.Logger:
    """Create a single console logger (avoid duplicate handlers in IDEs/notebooks)."""
    logger = logging.getLogger("tiff_curator")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def crs_label(crs) -> str:
    """Best-effort CRS string for logs/manifests."""
    if crs is None:
        return "None"
    try:
        return crs.to_string()
    except Exception:
        return str(crs)


def resolution_label(transform) -> str:
    """Readable pixel size from affine transform (xres, yres)."""
    try:
        xres = abs(float(transform.a))
        yres = abs(float(transform.e))
        return f"{xres:g} x {yres:g}"
    except Exception:
        return "unknown"


def is_mollweide(crs, mollweide_crs_str: str) -> bool:
    """
    CRS equivalence check that works across ESRI:54009, WKT, PROJ strings, etc.
    """
    if crs is None:
        return False

    # Semantic comparison (preferred)
    try:
        moll = rasterio.crs.CRS.from_string(mollweide_crs_str)
        if crs.equals(moll):
            return True
    except Exception:
        pass

    # Common ESRI alias for Mollweide
    try:
        s = (crs.to_string() or "").lower()
        if "54009" in s:
            return True
    except Exception:
        pass

    # WKT may contain the name
    try:
        wkt = (crs.to_wkt() or "").lower()
        if "mollweide" in wkt:
            return True
    except Exception:
        pass

    return False


# ----------------------------
# Dataset-specific steps
# ----------------------------

def ntl_log_plus2_block(arr: np.ndarray, nodata) -> np.ndarray:
    """
    NTL postprocess (band 1): out = log(x + 2), safely.
    - invalid values (nodata, non-finite, x+2 <= 0) become nodata (numeric)
    """
    x = arr.astype(np.float32, copy=False)

    # Use numeric nodata for good GIS compatibility
    nd = nodata
    if nd is None or (isinstance(nd, float) and math.isnan(nd)):
        nd = -9999.0

    out = np.full_like(x, float(nd), dtype=np.float32)

    valid = np.isfinite(x)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        valid &= (x != nodata)

    valid &= (x + 2.0) > 0.0

    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        out[valid] = np.log(x[valid] + 2.0)

    return out


# ----------------------------
# Manifest IO
# ----------------------------

def _write_folder_manifest(folder: Path, file_entry: dict) -> None:
    """
    Write/update a per-folder manifest (curation_manifest.json).
    We keep one entry per filename and overwrite it on re-run.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "curation_manifest.json"

    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}

    manifest.setdefault("folder", str(folder))
    manifest.setdefault("generated_utc", "")
    manifest.setdefault("files", {})

    manifest["files"][file_entry["file"]["name"]] = file_entry
    manifest["generated_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------
# Core processing (one file)
# ----------------------------

def _copy_and_summarize(
    src_path: Path,
    dst_path: Path,
    src_crs_str: str,
    target_crs_str: str,
) -> dict:
    """
    Copy the file as-is (fast path), then open ONCE to get output resolution and CRS.
    IMPORTANT: This does NOT walk windows/tiles; it’s O(1) file opens.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)

    with rasterio.open(dst_path) as out_ds:
        out_crs_str = crs_label(out_ds.crs)
        out_res = resolution_label(out_ds.transform)

    return {
        "file": {"name": src_path.name, "src_path": str(src_path), "dst_path": str(dst_path)},
        "general": {
            "action": "copied",
            "src_crs": src_crs_str,
            "dst_crs": out_crs_str,  # should equal target CRS (Mollweide) in this fast path
            "reprojected": False,
            "copied_as_is": True,
            "resampling": None,
            "target_resolution_m": None,
            "output_resolution": out_res,
            "target_crs": target_crs_str,
        },
        "specific": {"postprocess": []},
    }


def reproject_and_write(
    src_path: Path,
    dst_path: Path,
    cfg: PipelineConfig,
    logger: logging.Logger,
    resampling: Optional[Resampling] = None,
    target_resolution_m: Optional[float] = None,
    postprocess_band1: Optional[Callable[[np.ndarray, object], np.ndarray]] = None,
    postprocess_name: Optional[str] = None,
    log_every_windows: int = 250,
) -> dict:
    """
    Process a single TIFF, returning a manifest entry describing what happened.
    """

    resampling = resampling or cfg.default_resampling
    target_crs = rasterio.crs.CRS.from_string(cfg.mollweide_crs)
    target_crs_str = crs_label(target_crs)

    with rasterio.open(src_path) as src:
        src_crs_str = crs_label(src.crs)
        src_is_moll = is_mollweide(src.crs, cfg.mollweide_crs)

        logger.info(f"CRS check: {src_path.name} | CRS={src_crs_str} | Mollweide={src_is_moll}")

        # FAST PATH:
        # If already Mollweide and no postprocess => copy file without reading tiles/windows.
        if src_is_moll and postprocess_band1 is None:
            info = _copy_and_summarize(src_path, dst_path, src_crs_str, target_crs_str)
            logger.info(
                f"Finished: {dst_path.name} | action=copied | out_crs={info['general']['dst_crs']} "
                f"| out_res={info['general']['output_resolution']}"
            )
            return info

        # Otherwise we need to write a new file:
        needs_reproject = not src_is_moll
        action = "reprojected" if needs_reproject else "postprocessed_only"

        # Reader:
        # - If reprojection needed: WarpedVRT gives a correct virtual Mollweide view (windowed + correct).
        # - If already Mollweide: read directly from src.
        if needs_reproject:
            vrt_kwargs = {"crs": target_crs, "resampling": resampling}
            if target_resolution_m is not None:
                vrt_kwargs["resolution"] = (target_resolution_m, target_resolution_m)
            reader = WarpedVRT(src, **vrt_kwargs)
        else:
            reader = src

        post_steps: list[str] = []
        if postprocess_band1 is not None:
            post_steps.append(postprocess_name or "band1_postprocess")

        try:
            # Output metadata comes from reader (src or VRT)
            meta = reader.profile.copy()
            meta.update(driver="GTiff")

            # Reasonable GeoTIFF defaults
            meta.setdefault("compress", "deflate")
            meta.setdefault("tiled", True)
            meta.setdefault("blockxsize", 256)
            meta.setdefault("blockysize", 256)

            # If postprocessing, write float outputs with numeric nodata
            if postprocess_band1 is not None:
                meta["dtype"] = "float32"
                meta["nodata"] = -9999.0

            # Log what we’re about to do + output grid resolution (from output transform)
            out_res_pre = resolution_label(meta.get("transform"))
            logger.info(
                f"Start: {src_path.name} | action={action} | out_res={out_res_pre}"
                + (f" | target_res={target_resolution_m}m" if target_resolution_m else "")
                + (f" | post={post_steps}" if post_steps else "")
            )

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Write window-by-window (memory-safe + gives progress for big files)
            with rasterio.open(dst_path, "w", **meta) as dst:
                windows = list(dst.block_windows(1))
                total = len(windows)

                for b in range(1, reader.count + 1):
                    for idx, (_, window) in enumerate(windows, start=1):
                        # Read warped pixels for this window
                        out_dtype = np.float32 if (postprocess_band1 is not None and b == 1) else dst.dtypes[b - 1]
                        arr = reader.read(b, window=window, out_dtype=out_dtype)

                        # Apply optional postprocess on band 1
                        if postprocess_band1 is not None and b == 1:
                            arr = postprocess_band1(arr, dst.nodata)

                        dst.write(arr, b, window=window)

                        # Progress logging only for band 1 (keeps logs readable)
                        if b == 1 and (idx == 1 or idx % log_every_windows == 0 or idx == total):
                            pct = 100.0 * idx / total
                            logger.info(f"  {src_path.name}: {idx}/{total} windows ({pct:.1f}%)")

            # Open ONCE to record final CRS/resolution in the manifest (cheap)
            with rasterio.open(dst_path) as out_ds:
                out_crs_str = crs_label(out_ds.crs)
                out_res = resolution_label(out_ds.transform)

            logger.info(
                f"Finished: {dst_path.name} | action={action} | out_crs={out_crs_str} | out_res={out_res}"
            )

            return {
                "file": {"name": src_path.name, "src_path": str(src_path), "dst_path": str(dst_path)},
                "general": {
                    "action": action,
                    "src_crs": src_crs_str,
                    "dst_crs": out_crs_str,
                    "reprojected": bool(needs_reproject),
                    "copied_as_is": False,
                    "resampling": str(resampling),
                    "target_resolution_m": target_resolution_m,
                    "output_resolution": out_res,
                    "target_crs": target_crs_str,
                },
                "specific": {"postprocess": post_steps},
            }

        finally:
            if needs_reproject:
                reader.close()


# ----------------------------
# Driver (all folders/files)
# ----------------------------

def process_all(cfg: PipelineConfig) -> None:
    """
    Walk through Sat_data_raw subfolders and process all TIFFs.
    Writes curated outputs + per-folder manifests.
    """
    logger = setup_logging()

    if not cfg.raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {cfg.raw_root}")

    # Minimal rules table; add more datasets as you go.
    dataset_rules = {
        "NTL": {
            # Prevent runaway output size for global rasters:
            "target_resolution_m": 1000.0,
            "postprocess_band1": ntl_log_plus2_block,
            "postprocess_name": "log(x + 2) on band 1",
            "resampling": Resampling.bilinear,
        }
    }

    type_dirs = sorted([p for p in cfg.raw_root.iterdir() if p.is_dir()])
    logger.info(f"Found {len(type_dirs)} data-type folder(s) in: {cfg.raw_root}")

    for data_type_dir in type_dirs:
        data_type = data_type_dir.name.upper()
        rule = dataset_rules.get(data_type, {})

        tifs = sorted([*data_type_dir.glob("*.tif"), *data_type_dir.glob("*.tiff")])
        if not tifs:
            logger.info(f"[{data_type}] No TIFFs found, skipping.")
            continue

        logger.info(f"[{data_type}] Processing {len(tifs)} file(s)...")

        for i, src_path in enumerate(tifs, start=1):
            rel = src_path.relative_to(cfg.raw_root)
            dst_path = cfg.curated_root / rel

            logger.info(f"[{data_type}] ({i}/{len(tifs)}) {src_path.name}")

            info = reproject_and_write(
                src_path=src_path,
                dst_path=dst_path,
                cfg=cfg,
                logger=logger,
                resampling=rule.get("resampling"),
                target_resolution_m=rule.get("target_resolution_m"),
                postprocess_band1=rule.get("postprocess_band1"),
                postprocess_name=rule.get("postprocess_name"),
                log_every_windows=250,
            )

            # Update manifest in the output folder for this dataset/file
            _write_folder_manifest(dst_path.parent, info)

        logger.info(f"[{data_type}] Done. Manifest: {(cfg.curated_root / data_type_dir.name / 'curation_manifest.json')}")

    logger.info("All done.")


if __name__ == "__main__":
    cfg = PipelineConfig(
        raw_root=Path(
            r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_raw"
        ),
        curated_root=Path(
            r"E:\OneDrive\Studia\Studia magisterskie\Masterarbeit 2 - Sozialwissenschaften\data\Sat_data_curated"
        ),
    )
    process_all(cfg)
