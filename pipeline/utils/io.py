from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_id(series: pd.Series) -> pd.Series:
    """Normalize parish IDs across files that may store them in different forms."""
    missing = series.isna()
    ids = series.astype(str).str.strip()
    missing |= ids.str.lower().isin({"", "nan", "none", "null"})
    ids = ids.str.replace(r"\.0$", "", regex=True)
    ids = ids.str.upper().str.replace(r"[^0-9A-Z]", "", regex=True)

    numeric = ids.str.fullmatch(r"\d+").fillna(False)
    ids.loc[numeric] = ids.loc[numeric].str.zfill(6)
    ids.loc[missing] = pd.NA
    return ids


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


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """Write a DataFrame to CSV, creating its parent directory and logging its size."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=index, encoding=encoding)
    print(f"Wrote {output}")
    print(f"Rows: {len(df)}")
    return output
