"""Run from the repository root: python -m pipeline.modeling --dry-run."""

import argparse
import json
from pathlib import Path

from . import DEFAULT_TARGETS, EvaluationSpec, SearchConfig, TargetSpec
from . import load_project_data, preview_experiment, run_experiment
from pipeline.utils.paths import load_paths, path_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Experiment JSON; defaults to experiment.example.json")
    parser.add_argument("--paths-config", type=Path, help="Local input path configuration")
    parser.add_argument("--output-root", type=Path, help="Override external run output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and show fit counts without fitting or writing")
    args = parser.parse_args()
    config_path = args.config or Path(__file__).with_name("experiment.example.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unknown = set(config) - {"models", "targets", "exclude_overseas", "evaluation", "search"}
    if unknown:
        parser.error(f"Unknown experiment configuration fields: {sorted(unknown)}")
    targets = tuple(TargetSpec(**t) for t in config["targets"]) if "targets" in config else DEFAULT_TARGETS
    dataset = load_project_data(targets, exclude_overseas=config.get("exclude_overseas", True),
                                config_path=args.paths_config)
    evaluation = EvaluationSpec(**config.get("evaluation", {}))
    search = SearchConfig(**config.get("search", {}))
    models = config.get("models", ["ridge", "random_forest", "histogram_boosting"])
    plan = preview_experiment(dataset, models=models, targets=targets, evaluation=evaluation, search=search)
    print(plan.drop(columns="search_space").to_string(index=False))
    for row in plan.itertuples():
        print(f"{row.target}/{row.model}: {row.search_space}")
    print("Model fits:", plan.total_fits_with_oof_and_refit.sum(), "plus training-mean baseline fits")
    if not args.dry_run:
        output_root = args.output_root or (
            path_value(load_paths(args.paths_config), "external_data_root") / "outputs/epvi_prediction/runs")
        results = run_experiment(dataset, models=models, targets=targets, evaluation=evaluation,
                                 search=search, output_root=output_root)
        print(results.comparison_table().to_string())


if __name__ == "__main__":
    main()
