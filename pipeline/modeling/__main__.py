"""Develop, freeze and evaluate nested spatial experiments from the repository root."""

import argparse
import json
from pathlib import Path

from . import (EvaluationSpec, SearchConfig, TargetSpec, resolve_targets,
              load_project_data, preview_experiment, run_experiment,
              freeze_run, evaluate_frozen)
from .procedure import read_frozen
from pipeline.utils.paths import load_paths, path_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Development experiment JSON")
    parser.add_argument("--paths-config", type=Path, help="Local input path configuration")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stage", choices=["develop", "outer_evaluate", "final_evaluate"], default="develop")
    parser.add_argument("--freeze-run", type=Path, help="Freeze a completed development run and exit")
    parser.add_argument("--frozen", type=Path, help="Verified frozen.json required for evaluation")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--targets", nargs="+", help="base/additional/all, or target keys")
    parser.add_argument("--selections", nargs="+", help="holdout_A, etc.; final_test for final evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without fitting or writing")
    args = parser.parse_args()
    if args.freeze_run:
        if args.dry_run or args.frozen or args.stage != "develop":
            parser.error("--freeze-run is a separate action; do not combine with stage/frozen/dry-run.")
        print(freeze_run(args.freeze_run))
        return
    if args.stage == "develop":
        if args.frozen:
            parser.error("Development does not consume --frozen.")
        config_path = args.config or Path(__file__).with_name("experiment.example.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        unknown = set(config) - {"models", "targets", "exclude_overseas", "evaluation", "search", "selections", "predictors"}
        if unknown:
            parser.error(f"Unknown experiment configuration fields: {sorted(unknown)}")
        requested = args.targets or config.get("targets", "base")
        if isinstance(requested, list) and len(requested) == 1 and isinstance(requested[0], str):
            requested = requested[0]
        targets = resolve_targets(requested)
        models = args.models or config.get("models", ["ridge", "random_forest"])
        selections = args.selections or config.get("selections", ["holdout_A"])
        search = SearchConfig(**config.get("search", {}))
        evaluation = EvaluationSpec(**config.get("evaluation", {}))
        dataset = load_project_data(targets, exclude_overseas=config.get("exclude_overseas", True),
                                    config_path=args.paths_config, predictors=config.get("predictors"))
    else:
        if args.frozen is None or args.config is not None:
            parser.error("Evaluation requires --frozen and forbids --config overrides.")
        _, manifest, _ = read_frozen(args.frozen)
        config = manifest["config"]
        all_targets = tuple(TargetSpec(**t) for t in config["targets"])
        target_keys = args.targets
        if target_keys and len(target_keys) == 1 and target_keys[0] in {"base", "additional", "all"}:
            target_keys = [t.key for t in resolve_targets(target_keys[0])]
        if target_keys and not set(target_keys) <= {t.key for t in all_targets}:
            parser.error("Requested targets are not in the frozen procedure.")
        targets = tuple(t for t in all_targets if target_keys is None or t.key in target_keys)
        models = args.models or config["models"]
        if not set(models) <= set(config["models"]):
            parser.error("Requested models are not in the frozen procedure.")
        selections = args.selections or (config["selections"] if args.stage == "outer_evaluate" else ["final_test"])
        if selections == ["all_outer_folds"]:
            selections = list(config["available_selections"])
        if args.stage == "outer_evaluate" and not set(selections) <= set(config["selections"]):
            parser.error("Requested outer selections are not in the frozen run.")
        dataset = load_project_data(all_targets, exclude_overseas=config["scope"] == "mainland_only",
                                    config_path=args.paths_config, predictors=config["predictors"])
        read_frozen(args.frozen, dataset)
        settings = config["search"]
        search = SearchConfig(**{**settings,
            "spaces": {k: v for k, v in settings["spaces"].items() if k in models},
            "candidates": {k: v for k, v in settings["candidates"].items() if k in models}})
        evaluation = EvaluationSpec(**config["evaluation"])
    if selections == ["all_outer_folds"]:
        selections = "all_outer_folds"
    plan = preview_experiment(dataset, models=models, targets=targets, evaluation=evaluation,
                              search=search, selections=selections, stage=args.stage)
    print(plan.drop(columns="search_space").to_string(index=False))
    for row in plan.itertuples():
        print(f"{row.selection}/{row.target}/{row.model}: {row.search_space}")
    print("Model fits:", plan.total_fits_with_oof_and_refit.sum(), "(baseline fits additional when training)")
    if args.dry_run:
        return
    output_root = args.output_root or (
        path_value(load_paths(args.paths_config), "external_data_root") / "outputs/epvi_prediction/runs")
    if args.stage == "develop":
        results = run_experiment(dataset, models=models, targets=targets, evaluation=evaluation,
                                 search=search, selections=selections, output_root=output_root)
    else:
        results = evaluate_frozen(dataset, args.frozen, models=models, targets=targets,
                                  selections=selections, stage=args.stage, output_root=output_root)
    print("Run directory:", results.run_dir)
    print(results.comparison_table().to_string())


if __name__ == "__main__":
    main()
