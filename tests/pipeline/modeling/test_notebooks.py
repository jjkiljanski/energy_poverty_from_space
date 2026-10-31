import json
from pathlib import Path


def test_notebook_code_compiles_without_stale_outputs():
    root = Path(__file__).resolve().parents[3]
    for name in ["1_model_training.ipynb", "2_model_performance.ipynb"]:
        notebook = json.loads((root / "pipeline/modeling" / name).read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{name}:{index}", "exec")
                assert cell["outputs"] == []
                assert cell["execution_count"] is None

