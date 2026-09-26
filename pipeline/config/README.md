# Configuration

`paths.json` contains machine-local paths for data outside the repository.

Repository-owned data is assumed to live under the top-level `data` directory.
Keep those paths out of `paths.json` and access them with `repo_data_path(...)`.
