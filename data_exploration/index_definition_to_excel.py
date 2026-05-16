#!/usr/bin/env python3
"""
manifest_to_excel.py

Convert an indices JSON manifest into a multi-sheet Excel workbook.

Usage:
  python manifest_to_excel.py --manifest path/to/indices_manifest.json --out out.xlsx

Requires:
  pip install openpyxl pandas
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _as_str(x: Any) -> str:
    """Safe stringifier for Excel cells."""
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        # Keep readable but compact
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_tables(manifest: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    # --- Project / top-level info ---
    project = manifest.get("project", {})
    paths = manifest.get("paths", {})
    defaults = manifest.get("global_defaults", {})
    helpers = manifest.get("helpers", {})

    project_rows = []
    for section_name, section in [
        ("project", project),
        ("paths", paths),
        ("global_defaults", defaults),
        ("helpers", helpers),
    ]:
        if isinstance(section, dict):
            for k, v in section.items():
                project_rows.append(
                    {"section": section_name, "key": k, "value": _as_str(v)}
                )
        else:
            project_rows.append({"section": section_name, "key": "", "value": _as_str(section)})

    tables["Project"] = pd.DataFrame(project_rows)

    # --- Folder catalog ---
    folder_catalog = manifest.get("folder_catalog", {})
    folder_rows = []
    if isinstance(folder_catalog, dict):
        for folder_name, meta in folder_catalog.items():
            row = {"folder": folder_name}
            if isinstance(meta, dict):
                for k, v in meta.items():
                    row[k] = _as_str(v)
            else:
                row["meta"] = _as_str(meta)
            folder_rows.append(row)
    tables["FolderCatalog"] = pd.DataFrame(folder_rows)

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

        # Summary row
        summary_rows.append(
            {
                "id": idx_id,
                "group": group,
                "description": desc,
                "inputs": _as_str(inputs),
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

    # --- README sheet content as a small DF (easy write) ---
    readme_lines = [
        "This workbook is generated from an indices JSON manifest.",
        "",
        "Sheets:",
        "- Project: top-level metadata, paths, defaults, helpers",
        "- FolderCatalog: available raster folders (logical sources)",
        "- Indices_Summary: one row per index (high-level overview)",
        "- Cell_Vars: one row per computed cell variable",
        "- Cell_Filters: one row per filter condition",
        "- Reduce: reduction/aggregation configuration per index",
        "- Reduce_Stats: additional distribution stats per index",
        "- Outputs: final output columns produced per index",
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
        max_len = max([len(str(col_name))] + [len(series.iloc[i]) for i in range(N)]) if N > 0 else len(str(col_name))
        width = min(max_len + 2, max_width)
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = width


def write_excel(tables: Dict[str, pd.DataFrame], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use openpyxl engine for formatting
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Order matters (README first)
        sheet_order = [
            "README",
            "Project",
            "FolderCatalog",
            "Indices_Summary",
            "Cell_Vars",
            "Cell_Filters",
            "Reduce",
            "Reduce_Stats",
            "Outputs",
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

        # Freeze header row for key sheets
        for key_sheet in ["FolderCatalog", "Indices_Summary", "Cell_Vars", "Cell_Filters", "Reduce", "Reduce_Stats", "Outputs"]:
            ws = writer.sheets.get(key_sheet)
            if ws:
                ws.freeze_panes = "A2"

    print(f"Wrote Excel: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to indices_manifest.json")
    ap.add_argument("--out", required=True, help="Output .xlsx path")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)

    manifest = load_manifest(manifest_path)
    tables = build_tables(manifest)
    write_excel(tables, out_path)


if __name__ == "__main__":
    main()
