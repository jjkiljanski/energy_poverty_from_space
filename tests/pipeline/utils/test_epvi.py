import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("dtype", [object, "string"])
def test_decimal_comma_parser_supports_object_and_string_columns(dtype):
    from pipeline.utils.epvi import coerce_numeric_frame, normalize_epvi_ids
    raw = pd.DataFrame({"value": pd.Series(["0,25", "1 234,5", None], dtype=dtype)})
    parsed = coerce_numeric_frame(raw, skip=set())
    np.testing.assert_allclose(parsed.value.iloc[:2], [0.25, 1234.5])
    assert pd.isna(parsed.value.iloc[2])
    assert normalize_epvi_ids(pd.Series(["031401", "031402"])).tolist() == ["031404", "031401"]

