# Configuration

`paths.example.json` documents the local path assumptions used by the current scripts.

For local use, copy it to:

```text
config/paths.local.json
```

and adjust paths if your external data folder moves. `paths.local.json` is ignored by git.

For now, `paths.example.json` intentionally contains the current local Windows paths. This is less elegant than placeholders, but it makes the thesis workflow easier to debug. If the project is later prepared for publication, the example can be rewritten with placeholder paths.
