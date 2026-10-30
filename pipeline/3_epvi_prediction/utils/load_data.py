from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pandas as pd

from pipeline.utils.io import normalize_id, read_csv_robust
from pipeline.utils.epvi import EPVI_ID_CORRECTIONS, coerce_numeric_frame, normalize_epvi_ids


def load_modeling_inputs(
    sat_csv: str | Path,
    adm_csv: str | Path,
    epvi_csv: str | Path,
    adm_split_json: str | Path,
    prediction_targets: List[str],
) -> dict[str, Any]:
    """Load, normalize, merge, and prepare the EPVI modeling inputs."""
    sat = read_csv_robust(sat_csv)
    print("Loaded satellite predictors:", sat_csv, "rows:", len(sat), "columns:", len(sat.columns))
    adm = read_csv_robust(adm_csv)
    print("Loaded administrative data:", adm_csv, "rows:", len(adm), "columns:", len(adm.columns))
    epvi = read_csv_robust(epvi_csv)
    print("Loaded EPVI data:", epvi_csv, "rows:", len(epvi), "columns:", len(epvi.columns))

    for df in [sat, adm]:
        df["ID_norm"] = normalize_id(df["ID"])
    epvi["ID_norm"] = normalize_epvi_ids(epvi["ID"])

    targets = [c for c in prediction_targets if c in epvi.columns]
    name_col = next((c for c in epvi.columns if c not in {"ID", "ID_norm", *targets}), None)

    sat = coerce_numeric_frame(sat, skip={"ID", "ID_norm"})
    adm = coerce_numeric_frame(adm, skip={"ID", "ID_norm"})

    epvi_skip = {"ID", "ID_norm"}
    if name_col is not None:
        epvi_skip.add(name_col)
    epvi = coerce_numeric_frame(epvi, skip=epvi_skip)

    with Path(adm_split_json).open("r", encoding="utf-8") as f:
        adm_split = json.load(f)

    basic_admin_cols = [c for c in adm_split["basic"] if c in adm.columns]
    print("Using basic admin columns:", basic_admin_cols)
    basic_admin_cols_missing = [c for c in adm_split["basic"] if c not in adm.columns]
    if basic_admin_cols_missing:
        print("Warning: basic admin columns missing from loaded adm. data:", basic_admin_cols_missing)

    detailed_admin_cols = [c for c in adm_split["detailed"] if c in adm.columns]
    sat_cols = [c for c in sat.columns if c not in {"ID", "ID_norm"}]
    print("Using satellite predictor columns:", sat_cols)
    candidate_predictor_cols = sat_cols + basic_admin_cols

    epvi_keep_cols = ["ID", "ID_norm", *targets]
    if name_col is not None:
        epvi_keep_cols.append(name_col)

    model_df = (
        epvi.drop_duplicates("ID_norm")[epvi_keep_cols]
        .merge(adm.drop_duplicates("ID_norm")[["ID_norm", *basic_admin_cols]], on="ID_norm", how="inner")
        .merge(sat.drop_duplicates("ID_norm")[["ID_norm", *sat_cols]], on="ID_norm", how="inner")
    )

    if name_col is None:
        model_df["parish_name"] = pd.NA
    else:
        model_df = model_df.rename(columns={name_col: "parish_name"})

    predictor_cols = [
        c for c in candidate_predictor_cols
        if c in model_df.columns and model_df[c].notna().any()
    ]

    return {
        "sat": sat,
        "adm": adm,
        "epvi": epvi,
        "targets": targets,
        "name_col": name_col,
        "sat_cols": sat_cols,
        "basic_admin_cols": basic_admin_cols,
        "detailed_admin_cols": detailed_admin_cols,
        "candidate_predictor_cols": candidate_predictor_cols,
        "model_df": model_df,
        "predictor_cols": predictor_cols,
    }


def id_coverage_summary(inputs: dict[str, Any]) -> pd.DataFrame:
    """Summarize raw rows and normalized ID coverage across input tables."""
    rows = []
    id_sets: dict[str, set[str]] = {}
    for name in ["sat", "adm", "epvi"]:
        df = inputs[name]
        ids = df["ID_norm"].dropna()
        id_sets[name] = set(ids)
        rows.append({
            "table": name,
            "rows": len(df),
            "unique_ids": ids.nunique(),
            "duplicate_ids": int(ids.duplicated().sum()),
            "missing_ids": int(df["ID_norm"].isna().sum()),
        })

    common_ids = set.intersection(*id_sets.values())
    rows.append({
        "table": "common_sat_adm_epvi",
        "rows": None,
        "unique_ids": len(common_ids),
        "duplicate_ids": None,
        "missing_ids": None,
    })
    return pd.DataFrame(rows)


def load_prediction_inputs(
    sat_csv: str | Path,
    adm_csv: str | Path,
    epvi_csv: str | Path,
    adm_split_json: str | Path,
    prediction_targets: List[str],
) -> dict[str, Any]:
    """Backward-compatible alias for load_modeling_inputs."""
    return load_modeling_inputs(
        sat_csv=sat_csv,
        adm_csv=adm_csv,
        epvi_csv=epvi_csv,
        adm_split_json=adm_split_json,
        prediction_targets=prediction_targets,
    )


def build_modeling_table(inputs: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Backward-compatible accessor for modeling outputs."""
    return inputs["model_df"], inputs["predictor_cols"], inputs["constant_cols"]

def latest_file(directory: str | Path, pattern: str) -> Path | None:
    """Return the most recently modified file matching a pattern."""
    directory = Path(directory)
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None
