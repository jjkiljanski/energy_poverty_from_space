from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import mean_squared_error


EPVI_TARGETS = ["EPG heating", "EPG cooling", "AIAM", "EPVI heating", "EPVI cooling"]
DEFAULT_TEST_NUTS3 = ("PT16E", "PT16J")
OVERSEAS_NUTS3 = ("PT200", "PT300")

# The private EPVI table uses an alternate LAU2 code variant for a handful of
# 2013-era freguesias. Some codes collide with different current CAOP/Eurostat
# parish codes, so normalize these EPVI IDs before any table join.
EPVI_ID_CORRECTIONS = {
    "031401": "031404",  # Infias
    "031402": "031401",  # Santa Eulalia
    "031403": "031406",  # Vizela (Santo Adriao)
    "110813": "110812",  # Uniao das Freguesias de Lourinha e Atalaia
    "110814": "110813",  # Uniao das Freguesias de Miragaia e Marteleira
    "110815": "110814",  # Uniao das Freguesias de Sao Bartolomeu dos Galegos e Moledo
    "111010": "111009",  # Porto Salvo
    "141003": "141001",  # Sao Joao Baptista
}


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root from a notebook or script working directory."""
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "pipeline" / "config").exists():
            return path
    raise FileNotFoundError("Could not find repository root containing pipeline/config.")


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    """Read comma/semicolon CSVs with common European encodings."""
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}") from last_error


def normalize_id(series: pd.Series) -> pd.Series:
    """Normalize parish IDs across files that may store them as ints/floats/strings."""
    missing = series.isna()
    ids = series.astype(str).str.strip()
    missing |= ids.str.lower().isin({"", "nan", "none", "null"})
    ids = ids.str.replace(r"\.0$", "", regex=True)
    ids = ids.str.upper().str.replace(r"[^0-9A-Z]", "", regex=True)

    # Numeric mainland IDs are usually stored without leading zeroes in CSVs.
    # Island IDs can contain letters (e.g. 0302FA); keep those letters intact.
    numeric = ids.str.fullmatch(r"\d+").fillna(False)
    ids.loc[numeric] = ids.loc[numeric].str.zfill(6)
    ids.loc[missing] = pd.NA
    return ids


def normalize_epvi_ids(series: pd.Series) -> pd.Series:
    """Normalize private EPVI parish IDs to the current pipeline ID variant."""
    ids = normalize_id(series)
    return ids.replace(EPVI_ID_CORRECTIONS)


def coerce_numeric_frame(df: pd.DataFrame, skip: set[str]) -> pd.DataFrame:
    """Convert decimal-comma/object columns to numeric where possible."""
    out = df.copy()
    for col in out.columns:
        if col in skip:
            continue
        if out[col].dtype == object:
            cleaned = (
                out[col]
                .astype(str)
                .str.strip()
                .str.replace("\u00a0", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            out[col] = pd.to_numeric(cleaned, errors="coerce")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def choose_satellite_csv(paths: dict[str, Any], repo_data_path, path_value) -> tuple[Path, Path, Path]:
    """
    Prefer regenerated index outputs when present; fall back to the committed
    modeling snapshot.
    """
    regenerated = path_value(paths, "outputs_indices_dir") / "freguesia_indices_streaming.csv"
    snapshot = repo_data_path(paths, "all_used_sat_indicators.csv")
    selected = regenerated if regenerated.exists() else snapshot
    return selected, regenerated, snapshot


def load_prediction_inputs(
    sat_csv: str | Path,
    adm_csv: str | Path,
    epvi_csv: str | Path,
    adm_split_json: str | Path,
) -> dict[str, Any]:
    """Load and normalize the EPVI prediction inputs."""
    sat = read_csv_robust(sat_csv)
    adm = read_csv_robust(adm_csv)
    epvi = read_csv_robust(epvi_csv)

    for df in [sat, adm]:
        df["ID_norm"] = normalize_id(df["ID"])
    epvi["ID_norm"] = normalize_epvi_ids(epvi["ID"])

    targets = [c for c in EPVI_TARGETS if c in epvi.columns]
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
    detailed_admin_cols = [c for c in adm_split["detailed"] if c in adm.columns]
    sat_cols = [c for c in sat.columns if c not in {"ID", "ID_norm"}]
    predictor_cols = sat_cols + basic_admin_cols

    return {
        "sat": sat,
        "adm": adm,
        "epvi": epvi,
        "targets": targets,
        "name_col": name_col,
        "sat_cols": sat_cols,
        "basic_admin_cols": basic_admin_cols,
        "detailed_admin_cols": detailed_admin_cols,
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


def build_modeling_table(inputs: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge EPVI, basic admin predictors, and satellite predictors."""
    epvi = inputs["epvi"]
    adm = inputs["adm"]
    sat = inputs["sat"]
    targets = inputs["targets"]
    name_col = inputs["name_col"]
    basic_admin_cols = inputs["basic_admin_cols"]
    sat_cols = inputs["sat_cols"]
    predictor_cols = inputs["predictor_cols"]

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

    predictor_cols = [c for c in predictor_cols if c in model_df.columns and model_df[c].notna().any()]
    constant_cols = [c for c in predictor_cols if model_df[c].nunique(dropna=True) <= 1]
    predictor_cols = [c for c in predictor_cols if c not in constant_cols]
    return model_df, predictor_cols, constant_cols


def load_nuts3_mapping(mapping_csv: str | Path) -> pd.DataFrame:
    """Load the freguesia-to-NUTS3 correspondence used for spatial splits."""
    mapping = read_csv_robust(mapping_csv)
    required = {"LAU2_NAT_CODE", "NUTS_3"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"NUTS3 mapping is missing required columns: {sorted(missing)}")

    mapping = mapping.copy()
    mapping["ID_norm"] = normalize_id(mapping["LAU2_NAT_CODE"])
    mapping["NUTS3"] = mapping["NUTS_3"].astype(str).str.strip().str.upper()
    mapping = mapping[mapping["ID_norm"].notna() & mapping["NUTS3"].ne("")].copy()

    duplicate_ids = mapping[mapping["ID_norm"].duplicated(keep=False)]
    conflicting = duplicate_ids.groupby("ID_norm")["NUTS3"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        raise ValueError(
            "NUTS3 mapping assigns multiple regions to parish IDs: "
            f"{conflicting.index[:10].tolist()}"
        )

    return mapping.drop_duplicates("ID_norm")[["ID_norm", "NUTS3"]]


def attach_nuts3(model_df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Attach one NUTS3 code to each modeling row and fail on missing mappings."""
    out = model_df.merge(mapping, on="ID_norm", how="left", validate="one_to_one")
    missing_ids = out.loc[out["NUTS3"].isna(), "ID_norm"].dropna().tolist()
    if missing_ids:
        raise ValueError(
            f"NUTS3 mapping is missing {len(missing_ids)} modeling IDs; "
            f"first examples: {missing_ids[:10]}"
        )
    return out


def split_fixed_nuts3_holdout(
    model_df: pd.DataFrame,
    test_nuts3: tuple[str, ...] = DEFAULT_TEST_NUTS3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the fixed spatial test regions away from the training regions."""
    if "NUTS3" not in model_df.columns:
        raise ValueError("Modeling table has no NUTS3 column.")
    test_codes = {str(code).upper() for code in test_nuts3}
    is_test = model_df["NUTS3"].astype(str).str.upper().isin(test_codes)
    train_df = model_df.loc[~is_test].copy()
    test_df = model_df.loc[is_test].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Fixed NUTS3 split failed: train rows={len(train_df)}, test rows={len(test_df)}, "
            f"test codes={sorted(test_codes)}."
        )
    return train_df, test_df


def make_nuts3_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "NUTS3",
    overseas_nuts3: tuple[str, ...] = OVERSEAS_NUTS3,
) -> tuple[list[tuple[list[int], list[int]]], pd.DataFrame]:
    """
    Build deterministic region-held-out folds balanced by row count.

    Every NUTS3 region is assigned to exactly one validation fold. Portuguese
    overseas NUTS3 regions are seeded into different folds before the remaining
    regions are assigned greedily by size.
    """
    if group_col not in df.columns:
        raise ValueError(f"Data frame has no spatial group column '{group_col}'.")
    if n_splits < 2:
        raise ValueError("Spatial CV needs at least two folds.")

    groups = df[group_col].astype(str).str.strip().str.upper()
    if groups.isna().any() or groups.eq("").any():
        raise ValueError(f"Spatial group column '{group_col}' contains missing values.")

    group_sizes = groups.value_counts()
    if len(group_sizes) < n_splits:
        raise ValueError(
            f"Spatial CV needs at least {n_splits} NUTS3 groups; found {len(group_sizes)}."
        )

    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    assigned: set[str] = set()

    present_overseas = [code for code in overseas_nuts3 if code in group_sizes.index]
    for fold_idx, code in enumerate(present_overseas):
        fold_groups[fold_idx].append(code)
        fold_sizes[fold_idx] += int(group_sizes[code])
        assigned.add(code)

    remaining = group_sizes.drop(index=list(assigned), errors="ignore")
    for code, size in remaining.sort_values(ascending=False).items():
        fold_idx = min(range(n_splits), key=lambda idx: (fold_sizes[idx], idx))
        fold_groups[fold_idx].append(str(code))
        fold_sizes[fold_idx] += int(size)

    splits: list[tuple[list[int], list[int]]] = []
    rows = []
    positions = pd.Series(range(len(df)), index=df.index)
    for fold_idx, validation_groups in enumerate(fold_groups):
        is_validation = groups.isin(validation_groups)
        train_idx = positions.loc[~is_validation].tolist()
        validation_idx = positions.loc[is_validation].tolist()
        splits.append((train_idx, validation_idx))
        rows.append({
            "fold": fold_idx,
            "n_rows": len(validation_idx),
            "n_nuts3": len(validation_groups),
            "validation_nuts3": ", ".join(sorted(validation_groups)),
        })

    return splits, pd.DataFrame(rows)


def latest_file(directory: str | Path, pattern: str) -> Path | None:
    """Return the most recently modified file matching a pattern."""
    directory = Path(directory)
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def spearman_corr(y_true, y_pred) -> float:
    return pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
