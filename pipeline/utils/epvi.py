"""Private EPVI ID alignment and numeric parsing shared by modeling loaders."""

import pandas as pd
from .io import normalize_id

# Alternate 2013-era LAU2 codes; some collide with current CAOP/Eurostat codes.
# Apply once to private EPVI IDs, before joining any current-code tables.
EPVI_ID_CORRECTIONS = {
    "031401": "031404",  # Infias
    "031402": "031401",  # Santa Eulalia
    "031403": "031406",  # Vizela (Santo Adriao)
    "110813": "110812",  # Uniao das Freguesias de Lourinha e Atalaia
    "110814": "110813",  # Uniao das Freguesias de Miragaia e Marteleira
    "110815": "110814",  # Uniao das Freguesias de Sao Bartolomeu dos Galegos
    "111010": "111009",  # Porto Salvo
    "141003": "141001",  # Sao Joao Baptista
}


def normalize_epvi_ids(series):
    return normalize_id(series).replace(EPVI_ID_CORRECTIONS)


def coerce_numeric_frame(df, skip):
    """Convert decimal-comma/object columns to numeric where possible."""
    out = df.copy()
    for col in out.columns:
        if col in skip:
            continue
        if not pd.api.types.is_numeric_dtype(out[col].dtype):
            cleaned = (out[col].astype(str).str.strip().str.replace("\u00a0", "", regex=False)
                       .str.replace(" ", "", regex=False).str.replace(",", ".", regex=False))
            out[col] = pd.to_numeric(cleaned, errors="coerce")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
