#!/usr/bin/env python3
"""
manifest_to_excel.py

Convert indices and data-source JSON manifests into a multi-sheet Excel workbook.

Usage:
  python pipeline/input_preparation/indicator_construction/index_definition_to_excel.py

Requires:
  pip install openpyxl pandas
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import rasterio
except Exception:  # pragma: no cover - optional dependency for metadata enrichment
    rasterio = None

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from pipeline.utils.paths import Paths, load_paths  # noqa: E402
from pipeline.utils.io import read_csv_robust


def _as_str(x: Any) -> str:
    """Safe stringifier for Excel cells."""
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        # Keep readable but compact
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _list_tifs(folder_path: Path) -> List[Path]:
    tifs: List[Path] = []
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        tifs.extend(folder_path.glob(pattern))
    return sorted(tifs)


def _format_resolution(rx: Any, ry: Any) -> str:
    if rx in ("", None) or ry in ("", None):
        return ""
    try:
        return f"{float(abs(rx)):.6g} x {float(abs(ry)):.6g}"
    except Exception:
        return ""


def _inspect_raster_folder(folder_name: str, folder_path: Optional[Path]) -> Dict[str, Any]:
    """
    Inspect a curated raster folder without changing the JSON manifest.

    The index runner uses the coarsest input grid as the reference grid per
    index. Therefore the most useful folder-level value here is the coarsest
    native pixel size among TIFFs in the folder.
    """
    row: Dict[str, Any] = {
        "folder_path": _as_str(folder_path) if folder_path is not None else "",
        "tif_count": "",
        "crs": "",
        "resolution_x": "",
        "resolution_y": "",
        "resolution": "",
        "resolution_unit": "",
        "pixel_area": "",
        "unique_resolutions": "",
        "sample_tif": "",
        "resolution_notes": "",
    }

    if folder_path is None:
        row["resolution_notes"] = "not inspected: data_root path could not be resolved"
        return row
    if rasterio is None:
        row["resolution_notes"] = "not inspected: rasterio is not available"
        return row
    if not folder_path.exists():
        row["resolution_notes"] = "not inspected: folder does not exist"
        return row

    tifs = _list_tifs(folder_path)
    row["tif_count"] = len(tifs)
    if not tifs:
        row["resolution_notes"] = "not inspected: no TIFF files found"
        return row

    best: Optional[Dict[str, Any]] = None
    unique_resolutions = set()
    crs_values = set()
    errors = []
    for tif in tifs:
        try:
            with rasterio.open(tif) as src:
                rx, ry = src.res
                rx_abs = float(abs(rx))
                ry_abs = float(abs(ry))
                area = rx_abs * ry_abs
                unique_resolutions.add((round(rx_abs, 8), round(ry_abs, 8)))
                crs_text = src.crs.to_string() if src.crs is not None else ""
                if crs_text:
                    crs_values.add(crs_text)
                candidate = {
                    "resolution_x": rx_abs,
                    "resolution_y": ry_abs,
                    "pixel_area": area,
                    "crs": crs_text,
                    "sample_tif": tif.name,
                }
                if best is None or area > best["pixel_area"]:
                    best = candidate
        except Exception as exc:
            errors.append(f"{tif.name}: {exc}")

    if best is None:
        row["resolution_notes"] = "not inspected: could not open TIFF files"
        return row

    row.update(best)
    row["resolution"] = _format_resolution(best["resolution_x"], best["resolution_y"])
    row["resolution_unit"] = "CRS units; meters for Mollweide"
    row["unique_resolutions"] = "; ".join(
        _format_resolution(rx, ry) for rx, ry in sorted(unique_resolutions)
    )
    notes = []
    if len(unique_resolutions) > 1:
        notes.append("mixed native resolutions; coarsest resolution reported")
    if len(crs_values) > 1:
        notes.append("mixed CRS values")
    if errors:
        notes.append(f"{len(errors)} TIFF(s) could not be inspected")
    row["resolution_notes"] = "; ".join(notes)
    return row


def _build_folder_resolution_catalog(
    manifest: Dict[str, Any],
    paths_cfg: Optional[Paths],
) -> Dict[str, Dict[str, Any]]:
    data_root = paths_cfg.external("satellite_curated") if paths_cfg else None
    folder_catalog = manifest.get("folder_catalog", {})
    if not isinstance(folder_catalog, dict):
        return {}

    return {
        folder_name: _inspect_raster_folder(
            folder_name,
            data_root / folder_name if data_root is not None else None,
        )
        for folder_name in folder_catalog
    }


def _index_resolution_summary(
    inputs: List[Any],
    folder_resolution_catalog: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    rows = []
    for folder in inputs:
        meta = folder_resolution_catalog.get(str(folder), {})
        rx = meta.get("resolution_x", "")
        ry = meta.get("resolution_y", "")
        area = meta.get("pixel_area", "")
        rows.append({
            "folder": str(folder),
            "resolution": _format_resolution(rx, ry),
            "pixel_area": area,
        })

    available = [
        row for row in rows
        if row.get("pixel_area") not in ("", None) and pd.notna(row.get("pixel_area"))
    ]
    ref = max(available, key=lambda row: float(row["pixel_area"])) if available else {}
    return {
        "input_grid_resolutions": "; ".join(
            f"{row['folder']}={row['resolution']}" for row in rows if row["resolution"]
        ),
        "reference_grid_folder": ref.get("folder", ""),
        "reference_grid_resolution": ref.get("resolution", ""),
        "reference_grid_pixel_area": ref.get("pixel_area", ""),
    }


def _build_admin_indicator_table(paths_cfg: Optional[Paths] = None) -> pd.DataFrame:
    paths = paths_cfg or load_paths()
    admin_csv = paths.internal("administrative_indicators")
    split_json = paths.internal("admin_groups")

    split: Dict[str, List[str]] = {}
    if split_json is not None and split_json.exists():
        with split_json.open("r", encoding="utf-8") as f:
            raw_split = json.load(f)
        if isinstance(raw_split, dict):
            split = {
                str(group): [str(col) for col in cols]
                for group, cols in raw_split.items()
                if isinstance(cols, list)
            }

    csv_columns: List[str] = []
    row_count = ""
    if admin_csv is not None and admin_csv.exists():
        admin_df = read_csv_robust(admin_csv)
        csv_columns = [str(col) for col in admin_df.columns]
        row_count = str(len(admin_df)) if "ID" in csv_columns else ""

    category_by_indicator: Dict[str, str] = {}
    for category, indicators in split.items():
        label = "basic_prediction" if category == "basic" else "complex_detailed"
        for indicator in indicators:
            category_by_indicator[indicator] = label

    indicators = sorted(set(category_by_indicator) | {col for col in csv_columns if col != "ID"})
    rows = []
    for indicator in indicators:
        category = category_by_indicator.get(indicator, "not_classified_in_split")
        rows.append({
            "indicator": indicator,
            "category": category,
            "used_for_prediction": category == "basic_prediction",
            "reserved_for_residual_analysis": category == "complex_detailed",
            "available_in_admin_csv": indicator in csv_columns,
            "admin_csv": _as_str(admin_csv),
            "admin_split_json": _as_str(split_json),
            "admin_csv_rows": row_count,
            "notes": "" if indicator in csv_columns else "listed in split file but not present in admin CSV",
        })
    return pd.DataFrame(rows)


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


SOURCE_COLUMNS = [
    "id", "name", "category", "verified", "verification_notes", "link",
    "download_notes", "apa_citation", "citation_verified", "rationale",
    "dataset_version", "temporal_coverage", "spatial_coverage", "download_date",
    "local_paths", "notes",
]
NOT_USED_SOURCE_COLUMNS = SOURCE_COLUMNS + ["why_not_used"]


def build_source_tables(registry: Dict[str, Any]):
    """Validate and shape the used and planned source tables."""
    groups = registry.get("sources")
    if not isinstance(groups, dict) or set(groups) != {"used", "planned_only"}:
        raise ValueError("sources must contain exactly 'used' and 'planned_only' groups")
    seen = set()
    tables = {}
    for group_name, table_name, expected_columns in (
        ("used", "Data_Sources", SOURCE_COLUMNS),
        ("planned_only", "Not_Used_Data_Sources", NOT_USED_SOURCE_COLUMNS),
    ):
        rows = []
        for source in groups[group_name]:
            source_id = source.get("id")
            if not source_id or source_id in seen:
                raise ValueError(f"Missing or duplicate data source ID: {source_id!r}")
            seen.add(source_id)
            if not isinstance(source.get("verified"), bool):
                raise ValueError(f"Source {source_id} must have a boolean verified field")
            missing = set(expected_columns) - set(source)
            if missing:
                raise ValueError(f"Source {source_id} is missing fields: {sorted(missing)}")
            extra = set(source) - set(expected_columns)
            if extra:
                raise ValueError(f"Source {source_id} has unexpected fields: {sorted(extra)}")
            row = dict(source)
            for field in ("local_paths",):
                if isinstance(row[field], list):
                    row[field] = "\n".join(str(value) for value in row[field])
            rows.append(row)
        tables[table_name] = pd.DataFrame(rows, columns=expected_columns)
    return tables


def build_artifact_table(
    preparation_manifest: Dict[str, Any], sources_manifest: Dict[str, Any]
) -> pd.DataFrame:
    """Export prepared artifacts and check their source-ID references."""
    groups = sources_manifest.get("sources", {})
    sources = {
        source["id"]: source
        for group in groups.values()
        for source in group
    }
    artifacts = preparation_manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("data_preparation_manifest['artifacts'] must be a list")
    seen = set()
    rows = []
    for artifact in artifacts:
        artifact_id = artifact.get("id")
        if not artifact_id or artifact_id in seen:
            raise ValueError(f"Missing or duplicate artifact ID: {artifact_id!r}")
        seen.add(artifact_id)
        source_ids = artifact.get("source_ids", [])
        unknown = set(source_ids) - set(sources)
        if unknown:
            raise ValueError(f"Artifact {artifact_id} references unknown source IDs: {sorted(unknown)}")
        rows.append({
            "id": artifact_id,
            "name": artifact.get("name", ""),
            "source_description": artifact.get("source_description", ""),
            "path": _as_str(artifact.get("path")),
            "preparation_method": artifact.get("preparation_method", ""),
            "source_ids": "; ".join(source_ids),
            "definition_files": "\n".join(artifact.get("definition_files", [])),
            "process": artifact.get("process", ""),
            "details_status": artifact.get("details_status", ""),
            "notes": "\n".join(artifact.get("notes", [])) if isinstance(artifact.get("notes"), list) else artifact.get("notes", ""),
        })
    return pd.DataFrame(rows, columns=[
        "id", "name", "source_description", "path", "preparation_method",
        "source_ids", "definition_files", "process", "details_status", "notes",
    ])


def build_tables(
    manifest: Dict[str, Any],
    paths_cfg: Optional[Paths] = None,
) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    folder_resolution_catalog = _build_folder_resolution_catalog(manifest, paths_cfg)

    # --- Folder catalog ---
    folder_catalog = manifest.get("folder_catalog", {})
    folder_rows = []
    if isinstance(folder_catalog, dict):
        for folder_name, meta in folder_catalog.items():
            metadata = folder_resolution_catalog.get(folder_name, {})
            folder_rows.append({
                "folder": folder_name,
                "notes": _as_str(meta.get("notes", "")) if isinstance(meta, dict) else _as_str(meta),
                "folder_path": metadata.get("folder_path", ""),
                "tiff_crs": metadata.get("crs", ""),
                "tiff_resolution": metadata.get("resolution", ""),
                "tiff_resolution_unit": metadata.get("resolution_unit", ""),
                "tiff_pixel_area": metadata.get("pixel_area", ""),
                "sample_tiff": metadata.get("sample_tif", ""),
            })
    tables["Folder_Catalog"] = pd.DataFrame(folder_rows, columns=[
        "folder", "notes", "folder_path", "tiff_crs", "tiff_resolution",
        "tiff_resolution_unit", "tiff_pixel_area", "sample_tiff",
    ])

    # --- Indices: summary + nested tables ---
    indices = manifest.get("indices", [])
    if not isinstance(indices, list):
        raise ValueError("manifest['indices'] must be a list")

    summary_rows: List[Dict[str, Any]] = []
    cell_vars_rows: List[Dict[str, Any]] = []
    cell_filters_rows: List[Dict[str, Any]] = []
    reduce_rows: List[Dict[str, Any]] = []
    reduce_stats_rows: List[Dict[str, Any]] = []
    outputs_rows: List[Dict[str, Any]] = []

    for idx in indices:
        if not isinstance(idx, dict):
            continue

        idx_id = idx.get("id", "")
        group = idx.get("group", "")
        desc = idx.get("description", "")
        inputs = idx.get("inputs", [])
        bind = idx.get("bind", {})

        cell = idx.get("cell", {}) or {}
        reduce = idx.get("reduce", {}) or {}
        outputs = idx.get("outputs", []) or []

        resolution_summary = _index_resolution_summary(
            inputs if isinstance(inputs, list) else [],
            folder_resolution_catalog,
        )
        # Summary row
        summary_rows.append(
            {
                "id": idx_id,
                "group": group,
                "description": desc,
                "inputs": _as_str(inputs),
                "input_grid_resolutions": _as_str(resolution_summary["input_grid_resolutions"]),
                "reference_grid_folder": _as_str(resolution_summary["reference_grid_folder"]),
                "reference_grid_resolution": _as_str(resolution_summary["reference_grid_resolution"]),
                "reference_grid_pixel_area": _as_str(resolution_summary["reference_grid_pixel_area"]),
                "bind": _as_str(bind),
                "cell_value": _as_str(cell.get("value")),
                "cell_mask": _as_str(cell.get("mask")),
                "reduce_method": _as_str(reduce.get("method")),
                "reduce_weight": _as_str(reduce.get("weight")),
                "reduce_numerator": _as_str(reduce.get("numerator")),
                "reduce_denominator": _as_str(reduce.get("denominator")),
                "reduce_condition": _as_str(reduce.get("condition")),
                "notes": _as_str(idx.get("notes")),
            }
        )

        # Cell vars
        vars_dict = cell.get("vars", {}) or {}
        if isinstance(vars_dict, dict):
            for var_name, expr in vars_dict.items():
                cell_vars_rows.append(
                    {
                        "index_id": idx_id,
                        "var_name": var_name,
                        "expression": _as_str(expr),
                    }
                )

        # Cell filters
        filters = cell.get("filters", []) or []
        if isinstance(filters, list):
            for f in filters:
                if isinstance(f, dict):
                    row = {"index_id": idx_id}
                    # keep stable columns even if some absent
                    for k in ["type", "var", "q", "op", "value", "notes"]:
                        if k in f:
                            row[k] = _as_str(f.get(k))
                    # include any other keys too
                    for k, v in f.items():
                        if k not in row:
                            row[k] = _as_str(v)
                    cell_filters_rows.append(row)
                else:
                    cell_filters_rows.append({"index_id": idx_id, "filter": _as_str(f)})

        # Reduce table
        reduce_row = {"index_id": idx_id}
        if isinstance(reduce, dict):
            for k, v in reduce.items():
                if k == "stats":
                    continue
                reduce_row[k] = _as_str(v)
        reduce_rows.append(reduce_row)

        # Reduce stats
        stats = reduce.get("stats", []) if isinstance(reduce, dict) else []
        if isinstance(stats, list):
            for s in stats:
                if isinstance(s, dict):
                    row = {"index_id": idx_id}
                    for k, v in s.items():
                        row[k] = _as_str(v)
                    reduce_stats_rows.append(row)
                else:
                    reduce_stats_rows.append({"index_id": idx_id, "stat": _as_str(s)})

        # Outputs
        if isinstance(outputs, list):
            for out in outputs:
                if isinstance(out, dict):
                    row = {"index_id": idx_id}
                    for k, v in out.items():
                        row[k] = _as_str(v)
                    outputs_rows.append(row)
                else:
                    outputs_rows.append({"index_id": idx_id, "output": _as_str(out)})

    tables["Indices_Summary"] = pd.DataFrame(summary_rows)
    tables["Cell_Vars"] = pd.DataFrame(cell_vars_rows)
    tables["Cell_Filters"] = pd.DataFrame(cell_filters_rows)
    tables["Reduce"] = pd.DataFrame(reduce_rows)
    tables["Reduce_Stats"] = pd.DataFrame(reduce_stats_rows)
    tables["Outputs"] = pd.DataFrame(outputs_rows)
    tables["Administrative_Indicators"] = _build_admin_indicator_table(paths_cfg)
    # --- README sheet content as a small DF (easy write) ---
    readme_lines = [
        "This workbook is generated from the index definitions. Edit the JSON source, then regenerate this workbook when definitions change.",
        "Index definitions: pipeline/definitions/indices_manifest.json",
        "Data source provenance is documented separately in data/docs/data_sources.xlsx.",
        "Supporting inputs: pipeline/config/external_paths.json, pipeline/config/internal_paths.json, pipeline/definitions/adm_data_split.json, and data/indicators/administrative.csv. Local raster metadata is optional enrichment.",
        "Regenerate from the repository root: python pipeline/input_preparation/indicator_construction/index_definition_to_excel.py",
        "",
        "Sheets:",
        "- Folder_Catalog: raster folders, descriptions, paths, and TIFF-derived spatial metadata",
        "- Indices_Summary: one row per index, including input and reference grid resolutions when raster folders can be inspected",
        "- Cell_Vars: one row per computed cell variable",
        "- Cell_Filters: one row per filter condition",
        "- Reduce: reduction/aggregation configuration per index",
        "- Reduce_Stats: additional distribution stats per index",
        "- Outputs: final output columns produced per index",
        "- Administrative_Indicators: administrative indicators split into basic prediction variables and complex/detailed residual-analysis variables",
        "",
        "Tip: You can filter Indices_Summary by group to quickly review definitions.",
    ]
    tables["README"] = pd.DataFrame({"README": readme_lines})

    return tables


def autosize_columns(writer, sheet_name: str, df: pd.DataFrame, max_width: int = 60):
    """Autosize columns in openpyxl worksheet (bounded)."""
    ws = writer.sheets[sheet_name]
    for col_idx, col_name in enumerate(df.columns, start=1):
        # compute best width based on header + a sample of rows
        series = df[col_name].astype(str)
        # avoid huge cost: look at first N rows only
        N = min(len(series), 500)
        max_len = max([len(str(col_name))] + [len(str(series.iloc[i])) for i in range(N)]) if N > 0 else len(str(col_name))
        width = min(max_len + 2, max_width)
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = width


def write_excel(tables: Dict[str, pd.DataFrame], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use openpyxl engine for formatting
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Order matters (README first)
        sheet_order = [
            "README",
            "Folder_Catalog",
            "Indices_Summary",
            "Cell_Vars",
            "Cell_Filters",
            "Reduce",
            "Reduce_Stats",
            "Outputs",
            "Administrative_Indicators",
        ]

        # Write all tables, even if empty
        for name in sheet_order:
            df = tables.get(name, pd.DataFrame())
            df.to_excel(writer, sheet_name=name, index=False)
            autosize_columns(writer, name, df)

        # Write any additional tables not in sheet_order
        for name, df in tables.items():
            if name in sheet_order:
                continue
            df.to_excel(writer, sheet_name=name[:31], index=False)
            autosize_columns(writer, name[:31], df)

        from openpyxl.styles import Alignment, Font, PatternFill

        for name, ws in writer.sheets.items():
            ws.freeze_panes = "A2"
            ws.sheet_view.showGridLines = False
            if ws.max_column and name != "README" and name != "Data_Sources":
                ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="244763")
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[1].height = 30
            headers = {cell.value: cell.column for cell in ws[1]}
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                    # Export source text literally, including text beginning with '='.
                    if cell.data_type == "f":
                        cell.data_type = "s"
                    if isinstance(cell.value, str) and cell.value.startswith(("https://", "http://")) and not any(c.isspace() for c in cell.value):
                        cell.hyperlink = cell.value
                        cell.font = Font(color="0563C1", underline="single")
            if name == "README":
                ws.column_dimensions["A"].width = 115
                for row in range(2, ws.max_row + 1):
                    ws.row_dimensions[row].height = 32
    print(f"Wrote Excel: {out_path}")


def write_data_sources_excel(
    tables: Dict[str, pd.DataFrame], artifact_table: pd.DataFrame, out_path: Path
) -> None:
    """Write used sources, not-used sources, and prepared artifacts separately."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.table import Table, TableStyleInfo
    from openpyxl.utils import get_column_letter

    out_path.parent.mkdir(parents=True, exist_ok=True)
    used = tables["Data_Sources"]
    not_used = tables["Not_Used_Data_Sources"]
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        readme = pd.DataFrame({"README": [
            "Generated source registry from pipeline/definitions/data_sources_manifest.json.",
            "Data_Sources lists sources used by the current pipeline.",
            "Not_Used_Data_Sources lists candidates and optional sources not selected for current pipeline inputs.",
            "The why_not_used field is blank until a reason is documented.",
            "Prepared artifacts and their source lineage are listed on the Artifacts sheet.",
            "verified covers only the lineage scope in verification_notes; citation_verified is assessed separately.",
            "Additional source metadata is retained in notes.",
            "Regenerate both workbooks from the repository root with:",
            "python pipeline/input_preparation/indicator_construction/index_definition_to_excel.py",
        ]})
        readme.to_excel(writer, sheet_name="README", index=False)
        autosize_columns(writer, "README", readme, max_width=115)

        used.to_excel(writer, sheet_name="Data_Sources", index=False, startrow=1)
        used_ws = writer.sheets["Data_Sources"]
        used_ws.cell(1, 1, "Sources used by the current pipeline")

        not_used.to_excel(writer, sheet_name="Not_Used_Data_Sources", index=False, startrow=2)
        not_used_ws = writer.sheets["Not_Used_Data_Sources"]
        not_used_ws.cell(1, 1, "Sources not used by the current pipeline")
        not_used_ws.cell(2, 1, "These candidates and optional sources are not selected as current pipeline inputs. Reasons are not recorded yet.")
        artifact_table.to_excel(writer, sheet_name="Artifacts", index=False)
        artifact_ws = writer.sheets["Artifacts"]

        # Add independent Excel tables and apply consistent, readable formatting.
        used_last_col = get_column_letter(len(SOURCE_COLUMNS))
        not_used_last_col = get_column_letter(len(NOT_USED_SOURCE_COLUMNS))
        for name, reference, ws in (
            ("UsedSources", f"A2:{used_last_col}{len(used) + 2}", used_ws),
            ("NotUsedSources", f"A3:{not_used_last_col}{len(not_used) + 3}", not_used_ws),
        ):
            table = Table(displayName=name, ref=reference)
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2", showFirstColumn=False,
                showLastColumn=False, showRowStripes=True, showColumnStripes=False,
            )
            ws.add_table(table)

        for sheet in (writer.sheets["README"], used_ws, not_used_ws, artifact_ws):
            sheet.freeze_panes = "A3" if sheet in (used_ws, not_used_ws) else "A2"
            sheet.sheet_view.showGridLines = False
            for row in sheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                    if isinstance(cell.value, str) and cell.value.startswith(("https://", "http://")) and not any(c.isspace() for c in cell.value):
                        cell.hyperlink = cell.value
                        cell.font = Font(color="0563C1", underline="single")
                    if cell.data_type == "f":
                        cell.data_type = "s"
            header_row = 2 if sheet is used_ws else 3 if sheet is not_used_ws else 1
            for cell in sheet[header_row]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="244763")
            sheet.row_dimensions[header_row].height = 26

        for sheet, title in (
            (used_ws, "Sources used by the current pipeline"),
            (not_used_ws, "Sources not used by the current pipeline"),
        ):
            sheet.cell(1, 1).font = Font(bold=True, color="FFFFFF")
            sheet.cell(1, 1).fill = PatternFill("solid", fgColor="244763")
        not_used_ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(NOT_USED_SOURCE_COLUMNS))
        not_used_ws.cell(2, 1).alignment = Alignment(wrap_text=True, vertical="top")
        not_used_ws.row_dimensions[2].height = 32
        artifact_ws.freeze_panes = "A2"
        artifact_ws.auto_filter.ref = artifact_ws.dimensions
        for col_idx, column in enumerate(artifact_table.columns, start=1):
            values = [str(column), *(str(v) for v in artifact_table.iloc[:250, col_idx - 1])]
            artifact_ws.column_dimensions[get_column_letter(col_idx)].width = min(max(map(len, values)) + 2, 60)
        for col_idx, column in enumerate(SOURCE_COLUMNS, start=1):
            sample_values = [column]
            sample_values.extend(str(used.iloc[row, col_idx - 1]) for row in range(min(len(used), 250)))
            used_ws.column_dimensions[get_column_letter(col_idx)].width = min(max(map(len, sample_values)) + 2, 60)
        used_ws.column_dimensions[get_column_letter(SOURCE_COLUMNS.index("notes") + 1)].width = 60
        for col_idx, column in enumerate(NOT_USED_SOURCE_COLUMNS, start=1):
            sample_values = [column]
            sample_values.extend(str(not_used.iloc[row, col_idx - 1]) for row in range(min(len(not_used), 250)))
            not_used_ws.column_dimensions[get_column_letter(col_idx)].width = min(max(map(len, sample_values)) + 2, 60)
        not_used_ws.column_dimensions[get_column_letter(NOT_USED_SOURCE_COLUMNS.index("notes") + 1)].width = 60
        for row in range(3, len(used) + 3):
            used_ws.row_dimensions[row].height = 60
        for row in range(4, len(not_used) + 4):
            not_used_ws.row_dimensions[row].height = 60

    print(f"Wrote Excel: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None, help="Index definitions (defaults to pipeline/definitions/indices_manifest.json)")
    ap.add_argument("--out", type=Path, help="Index workbook (defaults to data/docs/indices_manifest.xlsx)")
    ap.add_argument("--sources-out", type=Path, help="Data-source workbook (defaults to data/docs/data_sources.xlsx)")
    ap.add_argument("--preparation-manifest", type=Path, help="Prepared-artifacts manifest (defaults to pipeline/definitions/data_preparation_manifest.json)")
    ap.add_argument(
        "--sources-manifest", type=Path,
        default=None,
        help="Data-source registry JSON (defaults to the repo pipeline/definitions/data_sources_manifest.json)",
    )
    ap.add_argument(
        "--paths-config",
        default=None,
        help=(
            "Optional path to pipeline path config. Defaults to pipeline/config/external_paths.json."
        ),
    )
    ap.add_argument("--internal-paths-config", type=Path, help="Override internal_paths.json")
    args = ap.parse_args()

    PATHS = load_paths(args.paths_config, internal_config_path=args.internal_paths_config)
    manifest_path = args.manifest or PATHS.internal("index_manifest")
    source_path = args.sources_manifest or PATHS.internal("source_manifest")
    preparation_path = args.preparation_manifest or PATHS.internal("data_preparation_manifest")
    manifest = load_manifest(manifest_path)
    index_out_path = args.out if args.out is not None else PATHS.internal("indices_manifest_workbook")
    sources_out_path = args.sources_out if args.sources_out is not None else PATHS.internal("data_sources_workbook")
    index_tables = build_tables(manifest, paths_cfg=PATHS)
    source_manifest = load_manifest(source_path)
    source_tables = build_source_tables(source_manifest)
    artifact_table = build_artifact_table(load_manifest(preparation_path), source_manifest)
    write_excel(index_tables, index_out_path)
    write_data_sources_excel(source_tables, artifact_table, sources_out_path)


if __name__ == "__main__":
    main()
