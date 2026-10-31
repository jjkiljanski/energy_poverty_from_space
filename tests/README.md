# Tests

Mirror the production path: `pipeline/modeling/data.py` is tested in
`tests/pipeline/modeling/test_data.py`. Use `test_<module>.py`; do not create empty
test files for import-only `__init__.py` modules. Cross-module tests live beside the
module that orchestrates the behavior (here, `test_training.py`).

Keep tests readable without learning another framework:

- State the guarantee in the test name; show setup, action and assertions directly.
- Share small data/configuration fixtures only when several files need them.
- Use explicit numbers or independent calculations for expected results.
- Mock external boundaries or deliberate failures, not the calculation being checked.
- Use synthetic data and pytest temporary directories; never require private inputs.
- Prefer a few meaningful cases over exhaustive grids, score snapshots or helper layers.

Run from the repository root in the activated project Conda environment:

```powershell
python -m pytest tests/pipeline/modeling -q
python -m pytest -q
```

See [the modeling review map](pipeline/modeling/README.md) for the guarantees tested.
The pre-existing `test_source_manifest.py` remains in its original location; it is
outside the modeling refactor.
