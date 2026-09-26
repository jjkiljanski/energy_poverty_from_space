from pathlib import Path
import pandas as pd

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