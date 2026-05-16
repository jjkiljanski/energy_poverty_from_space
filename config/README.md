# Configuration

`paths.example.json` documents the local path assumptions used by the current scripts.

For local use, copy it to:

```text
config/paths.local.json
```

and adjust paths if your external data folder moves. `paths.local.json` is ignored by git.

The existing scripts still contain hard-coded paths in several places. Treat this config file as the target convention for the next cleanup pass.
