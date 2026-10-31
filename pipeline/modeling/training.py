"""Nested tuning with separate development, outer and final evaluation stages."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, cross_val_predict
from threadpoolctl import threadpool_limits

from .artifacts import OOF_METHOD, create_run, load_run, save_pair, write_json
from .data import spatial_cv
from .metrics import EvaluationSpec, evaluate
from .models import get_model
from .splits import outer_selections, resolve_selections, select_split


@dataclass(frozen=True)
class SearchConfig:
    seed: int = 42
    n_iter: int = 40
    n_jobs: int = 2
    verbose: int = 1
    # Optional partial overrides, indexed by model name. Useful for smoke tests.
    spaces: dict = field(default_factory=dict)
    candidates: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.n_iter < 1 or self.n_jobs == 0 or any(v < 1 for v in self.candidates.values()):
            raise ValueError("Search candidate counts must be positive; n_jobs cannot be zero.")

    def resolved_space(self, model):
        return {**model.search_space(), **self.spaces.get(model.name, {})}

    def candidate_count(self, model):
        size = len(ParameterGrid(self.resolved_space(model)))
        return size if model.search_kind == "grid" else min(
            self.candidates.get(model.name, self.n_iter), size)


@dataclass
class TrainingResult:
    model_name: str
    target: object
    pipeline: object
    details: dict
    search_results: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    folds: pd.DataFrame
    fold_metrics: pd.DataFrame
    regional_metrics: pd.DataFrame
    diagnostics: pd.DataFrame


def _models(models):
    resolved = [get_model(m) if isinstance(m, str) else m for m in models]
    if not resolved or len({m.name for m in resolved}) != len(resolved):
        raise ValueError("Select a nonempty list of unique models.")
    from .targets import TargetSpec
    for model in resolved:
        TargetSpec(model.name, model.name)  # Safe artifact directory component.
        if model.name == "training_mean" or model.search_kind not in {"grid", "random"}:
            raise ValueError("Reserved model name or unsupported search kind.")
    return resolved


def preview_experiment(dataset, *, models, targets, evaluation=None, search=None,
                       selections="all_outer_folds", stage="develop"):
    evaluation, search = evaluation or EvaluationSpec(), search or SearchConfig()
    dataset.validate(targets)
    models = _models(models)
    selections = resolve_selections(dataset, selections, stage=stage)
    unknown = (set(search.spaces) | set(search.candidates)) - {m.name for m in models}
    if unknown:
        raise ValueError(f"Search overrides reference unselected models: {sorted(unknown)}")
    rows = []
    for target in targets:
        for selection in selections:
            split = select_split(dataset, target, selection, stage=stage, primary=evaluation.primary)
            for model in models:
                space = search.resolved_space(model)
                pipeline = model.build_pipeline(seed=search.seed)
                if set(space) - set(pipeline.get_params()):
                    raise ValueError(f"{model.name}: unknown pipeline parameters in search space.")
                candidates = search.candidate_count(model)
                folds = len(split.inner_cv)
                rows.append({"target": target.key, "model": model.name, "selection": selection,
                             "outer_fold": split.outer_fold, "stage": stage,
                             "n_train": len(split.train), "n_holdout": len(split.holdout),
                             "n_predictors": len(dataset.predictors), "n_folds": folds,
                             "primary_metric": evaluation.primary, "search_kind": model.search_kind,
                             "candidates": candidates, "search_fits": 0 if stage == "outer_evaluate" else candidates * folds,
                             "total_fits_with_oof_and_refit": 0 if stage == "outer_evaluate" else candidates * folds + folds + 1,
                             "search_space": space})
    return pd.DataFrame(rows)


def _identity(model_name, target, scope):
    return {"model": model_name, "target": target.key,
            "target_column": target.column, "model_scope": scope}


def _predictions(data, predicted, target, identity, role):
    result = data[["ID_norm", "NUTS3", "spatial_cv_fold"]].copy()
    result["name"] = data["parish_name"] if "parish_name" in data else pd.NA
    for key, value in identity.items():
        result[key] = value
    result["prediction_role"] = role
    result["observed"] = data[target.column].to_numpy(dtype=float)
    result["predicted"] = predicted
    result["residual"] = result.observed - result.predicted
    return result


def _metric_rows(data, predicted, target, identity, role, evaluation, **extra):
    return [{**identity, "prediction_role": role, **extra, **row}
            for row in evaluate(data[target.column], predicted, evaluation)]


def train_model(model_spec, train, *, predictors, target, scope,
                evaluation=None, search=None, selection="all_training", outer_fold=None):
    """Tune and fit on training rows only. No test data enters this function.

    Passing model_spec=None constructs a training-mean reference baseline.
    """
    evaluation, search = evaluation or EvaluationSpec(), search or SearchConfig()
    started = perf_counter()
    cv, folds = spatial_cv(train, require_variable_target=target.column if evaluation.primary == "r2" else None)
    X, y = train[list(predictors)], train[target.column].astype(float)
    name = "training_mean" if model_spec is None else model_spec.name
    identity = _identity(name, target, scope)
    identity.update(selection=selection, outer_fold=outer_fold)
    search_results = pd.DataFrame(columns=["params", "mean_test_score", "rank_test_score"])
    best_params, best_score = {}, None
    # Prevent nested CPU oversubscription in forests and BLAS.
    with parallel_backend("loky", inner_max_num_threads=1), threadpool_limits(limits=1):
        if model_spec is None:
            best = DummyRegressor(strategy="mean").fit(X, y)
        else:
            kwargs = dict(estimator=model_spec.build_pipeline(seed=search.seed),
                          scoring=evaluation.scorer, cv=cv, n_jobs=search.n_jobs,
                          refit=True, error_score="raise", verbose=search.verbose,
                          return_train_score=False)
            if model_spec.search_kind == "grid":
                tuner = GridSearchCV(param_grid=search.resolved_space(model_spec), **kwargs)
            else:
                tuner = RandomizedSearchCV(param_distributions=search.resolved_space(model_spec),
                                           n_iter=search.candidate_count(model_spec),
                                           random_state=search.seed, **kwargs)
            tuner.fit(X, y)
            if not np.isfinite(tuner.cv_results_["mean_test_score"]).all():
                raise ValueError("Nonfinite tuning scores; inspect target and fold eligibility.")
            best, best_params = tuner.best_estimator_, tuner.best_params_
            best_score = evaluation.reported_score(tuner.best_score_)
            search_results = pd.DataFrame(tuner.cv_results_)
        # Inner diagnostics reuse parameters selected in this outer-training subset.
        # cross_val_predict clones best; its already fitted full-training instance remains intact.
        oof = cross_val_predict(best, X, y, cv=cv, n_jobs=search.n_jobs)
        in_sample = best.predict(X)
    predictions = pd.concat([
        _predictions(train, oof, target, identity, "inner_oof"),
        _predictions(train, in_sample, target, identity, "train_in_sample"),
    ], ignore_index=True)
    metrics = []
    for role, predicted in [("inner_oof", oof), ("train_in_sample", in_sample)]:
        metrics.extend(_metric_rows(train, predicted, target, identity, role, evaluation))
    fold_rows = []
    for fold, (_, idx) in zip(folds.fold, cv):
        fold_rows.extend(_metric_rows(train.iloc[idx], oof[idx], target, identity,
                                     "inner_oof", evaluation, fold=int(fold)))
    for key, value in identity.items():
        folds[key] = value
    diagnostics = (model_spec.diagnostics(best) if model_spec is not None else
                   pd.DataFrame(columns=["feature", "diagnostic", "value"]))
    for key, value in identity.items():
        diagnostics[key] = value
    selected = (best.named_steps["columns"].columns_ if model_spec is not None else [])
    details = {**identity, "best_params": best_params, "primary_metric": evaluation.primary,
               "selection_aggregation": "unweighted_mean_of_spatial_fold_scores",
               "best_mean_cv_score": best_score, "oof_method": OOF_METHOD,
               "n_train": len(train), "n_folds": len(cv),
               "requested_predictors": list(predictors), "fitted_predictors": selected,
               "dropped_predictors": [c for c in predictors if c not in selected],
               "training_seconds": perf_counter() - started,
               "training_data_sha256": hashlib.sha256(train[[
                   "ID_norm", "NUTS3", "spatial_cv_fold", *predictors, target.column
               ]].to_csv(index=False).encode("utf-8")).hexdigest()}
    return TrainingResult(name, target, best, details, search_results, pd.DataFrame(metrics),
                          predictions, folds, pd.DataFrame(fold_rows),
                          pd.DataFrame(columns=[*identity, "prediction_role", "NUTS3", "metric", "value", "n_rows", "undefined_reason"]),
                          diagnostics)


def evaluate_model(result, test, *, predictors, evaluation=None, role="outer_holdout"):
    """Evaluate the already fitted pipeline without refitting or selecting parameters."""
    evaluation = evaluation or EvaluationSpec()
    if role not in {"outer_holdout", "final_test"}:
        raise ValueError("Evaluation role must be outer_holdout or final_test.")
    identity = _identity(result.model_name, result.target, result.details["model_scope"])
    identity.update(selection=result.details["selection"], outer_fold=result.details["outer_fold"])
    with threadpool_limits(limits=1):
        predicted = result.pipeline.predict(test[list(predictors)])
    result.predictions = pd.concat([result.predictions,
        _predictions(test, predicted, result.target, identity, role)], ignore_index=True)
    result.metrics = pd.concat([result.metrics, pd.DataFrame(_metric_rows(
        test, predicted, result.target, identity, role, evaluation))], ignore_index=True)
    regional = []
    for region, indices in test.groupby("NUTS3").indices.items():
        regional.extend(_metric_rows(test.iloc[indices], predicted[indices], result.target,
                                     identity, role, evaluation, NUTS3=region))
    result.regional_metrics = pd.DataFrame(regional)
    result.details["n_holdout"] = len(test)
    return result


def run_experiment(dataset, *, models, targets, output_root, evaluation=None, search=None,
                   selections=None, stage="develop", frozen=None):
    """Develop nested models, or explicitly evaluate a previously frozen procedure.

    Evaluation takes its search/evaluation configuration from the frozen run.
    Pass neither search nor evaluation overrides for an evaluation stage.
    """
    if stage != "develop":
        if frozen is None or search is not None or evaluation is not None:
            raise ValueError("Evaluation requires frozen=... and forbids search/evaluation overrides.")
        return evaluate_frozen(dataset, frozen, output_root=output_root, stage=stage,
                               models=models, targets=targets, selections=selections)
    if frozen is not None:
        raise ValueError("Development does not consume a freeze; create a new run.")
    evaluation, search = evaluation or EvaluationSpec(), search or SearchConfig()
    models = _models(models)
    selections = "all_outer_folds" if selections is None else selections
    selections = resolve_selections(dataset, selections, stage=stage)
    plan = preview_experiment(dataset, models=models, targets=targets, evaluation=evaluation,
                              search=search, selections=selections, stage=stage)
    config = {"scope": dataset.scope, "targets": [asdict(t) for t in targets],
              "models": [m.name for m in models], "predictors": list(dataset.predictors),
              "evaluation": asdict(evaluation), "search": asdict(search),
              "coverage": dataset.coverage, "plan": plan.to_dict("records"),
              "baseline": "training_mean", "residual_definition": "observed - predicted",
              "stage": stage, "selections": selections,
              "available_selections": outer_selections(dataset)}
    directory, manifest = create_run(output_root, config, dataset)
    print(f"Run directory: {directory}", flush=True)
    try:
        for target in targets:
            for selection in selections:
                split = select_split(dataset, target, selection, primary=evaluation.primary)
                for spec in [None, *models]:
                    name = "training_mean" if spec is None else spec.name
                    print(f"{selection} / {target.key} / {name}: train={len(split.train)}", flush=True)
                    result = train_model(spec, split.train, predictors=dataset.predictors, target=target,
                                         scope=dataset.scope, evaluation=evaluation, search=search,
                                         selection=selection, outer_fold=split.outer_fold)
                    save_pair(directory, manifest, result)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(directory / "manifest.json", manifest)
    return load_run(directory)


def evaluate_frozen(dataset, frozen, *, output_root, stage="outer_evaluate",
                    models=None, targets=None, selections=None):
    """Outer evaluation reuses verified fitted pipelines; final evaluation retunes.

    Only load frozen model artifacts you trust: joblib deserializes Python objects.
    Frozen runs must retain their exact data, code and dependency versions.
    """
    from .artifacts import TABLES
    from .procedure import read_frozen
    from .targets import TargetSpec

    if stage not in {"outer_evaluate", "final_evaluate"}:
        raise ValueError("Frozen evaluation stage must be outer_evaluate or final_evaluate.")
    source, original, snapshot = read_frozen(frozen, dataset)
    config = original["config"]
    models = _models(config["models"] if models is None else models)
    targets = tuple(TargetSpec(**t) for t in config["targets"]) if targets is None else tuple(targets)
    if not {m.name for m in models} <= set(config["models"]):
        raise ValueError("Requested architecture is not in the frozen procedure.")
    if any(asdict(t) not in config["targets"] for t in targets) or not targets:
        raise ValueError("Requested target is not in the frozen procedure.")
    search = SearchConfig(**config["search"])
    # Restrict overrides when evaluating only a subset of frozen architectures.
    search = SearchConfig(**{**asdict(search),
        "spaces": {m.name: search.resolved_space(m) for m in models},
        "candidates": {m.name: search.candidate_count(m) for m in models}})
    evaluation = EvaluationSpec(**config["evaluation"])
    if selections is None:
        selections = config["selections"] if stage == "outer_evaluate" else ["final_test"]
    selections = resolve_selections(dataset, selections, stage=stage)
    if stage == "outer_evaluate" and not set(selections) <= set(config["selections"]):
        raise ValueError("Requested outer context was not developed in this frozen run.")
    plan = preview_experiment(dataset, models=models, targets=targets, search=search,
                              evaluation=evaluation, selections=selections, stage=stage)
    new_config = {**config, "stage": stage, "selections": selections,
                  "models": [m.name for m in models], "targets": [asdict(t) for t in targets],
                  "plan": plan.to_dict("records"), "freeze_id": snapshot["freeze_id"],
                  "frozen_source": str(source)}
    directory, manifest = create_run(output_root, new_config, dataset)
    try:
        for target in targets:
            for selection in selections:
                split = select_split(dataset, target, selection, stage=stage, primary=evaluation.primary)
                for spec in [None, *models]:
                    name = "training_mean" if spec is None else spec.name
                    if stage == "outer_evaluate":
                        pair = next(p for p in original["pairs"] if p["target"] == target.key
                                    and p["model"] == name and p["selection"] == selection)
                        folder = source / pair["directory"]
                        tables = {table: pd.read_csv(folder / f"{table}.csv", dtype={"ID_norm": str})
                                  for table in TABLES}
                        result = TrainingResult(name, target, joblib.load(folder / "pipeline.joblib"),
                            json.loads((folder / "fit.json").read_text(encoding="utf-8")),
                            pd.read_csv(folder / "search_results.csv"), **tables)
                    else:
                        result = train_model(spec, split.train, predictors=dataset.predictors,
                            target=target, scope=dataset.scope, evaluation=evaluation, search=search,
                            selection=selection, outer_fold=None)
                    evaluate_model(result, split.holdout, predictors=dataset.predictors,
                        evaluation=evaluation, role="outer_holdout" if stage == "outer_evaluate" else "final_test")
                    save_pair(directory, manifest, result)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(directory / "manifest.json", manifest)
    return load_run(directory)
