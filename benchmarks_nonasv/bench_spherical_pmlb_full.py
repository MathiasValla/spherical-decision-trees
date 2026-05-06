"""Resumable PMLB benchmark for axis-aligned, oblique, and spherical trees.

The benchmark writes one raw CSV row per successful model/fold and one error
CSV row per skipped dataset. By default, a dataset is skipped if any requested
model fails on it, so the summary compares models only on complete datasets.
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from pmlb import classification_dataset_names, fetch_data, regression_dataset_names
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.datasets import make_gaussian_quantiles, make_moons
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from treeple.ensemble import (
    ObliqueRandomForestClassifier,
    ObliqueRandomForestRegressor,
    SphericalRandomForestClassifier,
    SphericalRandomForestRegressor,
)
from treeple.tree import (
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
    SphericalDecisionTreeClassifier,
    SphericalDecisionTreeRegressor,
)


def _patch_treeple_sklearn_compatibility():
    """Adapt treeple's vendored sklearn fork to scikit-learn 1.7."""
    from sklearn.utils.validation import _check_sample_weight
    from treeple._lib.sklearn.ensemble import _forest as forest_module
    from treeple._lib.sklearn.tree import _classes as tree_classes

    def _check_sample_weight_compat(sample_weight, X, dtype=None, **kwargs):
        if dtype is None:
            return _check_sample_weight(sample_weight, X, **kwargs)
        return _check_sample_weight(sample_weight, X, dtype=dtype, **kwargs)

    def _forest_regressor_tags_compat(self):
        tags = super(forest_module.ForestRegressor, self).__sklearn_tags__()
        if hasattr(tags.regressor_tags, "multi_label"):
            tags.regressor_tags.multi_label = True
        return tags

    tree_classes._check_sample_weight = _check_sample_weight_compat
    forest_module.ForestRegressor.__sklearn_tags__ = _forest_regressor_tags_compat


_patch_treeple_sklearn_compatibility()


RAW_COLUMNS = [
    "task",
    "dataset",
    "n_samples",
    "n_features",
    "n_used_samples",
    "fold",
    "model",
    "score",
    "secondary_score",
    "fit_time_s",
    "predict_time_s",
    "total_time_s",
    "random_state",
]

CENTER_STRATEGIES = ("default", "random", "target", "hybrid", "radial", "target_radial")
TOY_CLASSIFICATION_DATASETS = ("toy_moons", "toy_gaussian_quantiles", "toy_xor")


def _parse_radius_candidates(value):
    if isinstance(value, str) and value.lower() in {"all", "none"}:
        return None
    return int(value)


def _center_strategy_kwargs(center_strategy):
    if center_strategy == "default":
        return {}
    return {"center_strategy": center_strategy}


class _CostComplexityPrunedTreeBase(BaseEstimator):
    """Validation-selected cost-complexity pruning wrapper for single trees."""

    _task = None

    def __init__(
        self,
        base_estimator,
        *,
        validation_fraction=0.25,
        max_alphas=5,
        random_state=42,
    ):
        self.base_estimator = base_estimator
        self.validation_fraction = validation_fraction
        self.max_alphas = max_alphas
        self.random_state = random_state

    def _score_predictions(self, y_true, pred):
        if self._task == "classification":
            return balanced_accuracy_score(y_true, pred)
        return r2_score(y_true, pred)

    def _split_train_validation(self, X, y):
        if X.shape[0] < 8 or self.validation_fraction <= 0.0:
            return X, X, y, y, False

        stratify = None
        if self._task == "classification":
            _, counts = np.unique(y, return_counts=True)
            if counts.min() >= 2:
                stratify = y

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            return X, X, y, y, False
        return X_train, X_val, y_train, y_val, True

    def _candidate_alphas(self, X, y):
        path_estimator = clone(self.base_estimator).set_params(ccp_alpha=0.0)
        path = path_estimator.cost_complexity_pruning_path(X, y)
        alphas = np.asarray(path.ccp_alphas, dtype=float)
        alphas = np.unique(alphas[np.isfinite(alphas)])
        if alphas.size == 0:
            return np.array([0.0])
        if alphas[0] != 0.0:
            alphas = np.r_[0.0, alphas]
        if self.max_alphas is not None and alphas.size > self.max_alphas:
            chosen = np.linspace(0, alphas.size - 1, int(self.max_alphas))
            alphas = alphas[np.unique(np.round(chosen).astype(int))]
        return alphas

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y)
        X_train, X_val, y_train, y_val, has_validation = self._split_train_validation(X, y)
        alphas = self._candidate_alphas(X_train, y_train)

        if not has_validation:
            best_alpha = float(alphas[min(1, alphas.size - 1)])
        else:
            best_alpha = 0.0
            best_score = -np.inf
            for alpha in alphas:
                candidate = clone(self.base_estimator).set_params(ccp_alpha=float(alpha))
                try:
                    candidate.fit(X_train, y_train)
                    score = self._score_predictions(y_val, candidate.predict(X_val))
                except Exception:
                    continue
                if score > best_score + 1e-12 or (
                    abs(score - best_score) <= 1e-12 and float(alpha) > best_alpha
                ):
                    best_score = score
                    best_alpha = float(alpha)

        self.best_ccp_alpha_ = best_alpha
        self.estimator_ = clone(self.base_estimator).set_params(ccp_alpha=best_alpha)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self.estimator_.fit(X, y, **fit_kwargs)

        self.n_features_in_ = getattr(self.estimator_, "n_features_in_", X.shape[1])
        if self._task == "classification":
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


class CostComplexityPrunedTreeClassifier(
    ClassifierMixin,
    _CostComplexityPrunedTreeBase,
):
    _task = "classification"


class CostComplexityPrunedTreeRegressor(
    RegressorMixin,
    _CostComplexityPrunedTreeBase,
):
    _task = "regression"


ERROR_COLUMNS = [
    "task",
    "dataset",
    "stage",
    "model",
    "fold",
    "error_type",
    "error",
    "traceback",
]


class TimeoutError(RuntimeError):
    pass


@contextmanager
def time_limit(seconds):
    if seconds is None or seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _classification_models(
    random_state,
    n_estimators,
    n_center_candidates,
    radius_candidates,
    center_strategy="default",
    prune_single_trees=False,
    pruning_validation_fraction=0.25,
    max_pruning_alphas=5,
):
    center_strategy_kwargs = _center_strategy_kwargs(center_strategy)
    single_trees = {
        "CART": DecisionTreeClassifier(random_state=random_state),
        "SphericalTree": SphericalDecisionTreeClassifier(
            random_state=random_state,
            max_features=None,
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
            **center_strategy_kwargs,
        ),
        "ObliqueTree": ObliqueDecisionTreeClassifier(random_state=random_state),
    }
    if prune_single_trees:
        single_trees = {
            name: CostComplexityPrunedTreeClassifier(
                estimator,
                validation_fraction=pruning_validation_fraction,
                max_alphas=max_pruning_alphas,
                random_state=random_state,
            )
            for name, estimator in single_trees.items()
        }

    return {
        "CART": single_trees["CART"],
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
        "SphericalTree": single_trees["SphericalTree"],
        "SphericalRandomForest": SphericalRandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
            max_features="sqrt",
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
            **center_strategy_kwargs,
        ),
        "ObliqueTree": single_trees["ObliqueTree"],
        "ObliqueRandomForest": ObliqueRandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
    }


def _regression_models(
    random_state,
    n_estimators,
    n_center_candidates,
    radius_candidates,
    center_strategy="default",
    prune_single_trees=False,
    pruning_validation_fraction=0.25,
    max_pruning_alphas=5,
):
    center_strategy_kwargs = _center_strategy_kwargs(center_strategy)
    single_trees = {
        "CART": DecisionTreeRegressor(random_state=random_state),
        "SphericalTree": SphericalDecisionTreeRegressor(
            random_state=random_state,
            max_features=None,
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
            **center_strategy_kwargs,
        ),
        "ObliqueTree": ObliqueDecisionTreeRegressor(random_state=random_state),
    }
    if prune_single_trees:
        single_trees = {
            name: CostComplexityPrunedTreeRegressor(
                estimator,
                validation_fraction=pruning_validation_fraction,
                max_alphas=max_pruning_alphas,
                random_state=random_state,
            )
            for name, estimator in single_trees.items()
        }

    return {
        "CART": single_trees["CART"],
        "RandomForest": RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
        "SphericalTree": single_trees["SphericalTree"],
        "SphericalRandomForest": SphericalRandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
            max_features="sqrt",
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
            **center_strategy_kwargs,
        ),
        "ObliqueTree": single_trees["ObliqueTree"],
        "ObliqueRandomForest": ObliqueRandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
    }


def _pipeline(estimator):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def _append_rows(path, rows, columns):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _load_seen_datasets(output_path, error_path):
    seen = set()
    if not output_path.exists():
        results = pd.DataFrame(columns=["task", "dataset"])
    else:
        results = pd.read_csv(output_path, usecols=["task", "dataset"])
        seen.update(map(tuple, results.drop_duplicates().to_numpy()))
    if error_path.exists():
        errors = pd.read_csv(error_path, usecols=["task", "dataset"])
        seen.update(map(tuple, errors.drop_duplicates().to_numpy()))
    return seen


def _subsample_dataset(X, y, task, args):
    if args.max_samples_per_dataset is None or X.shape[0] <= args.max_samples_per_dataset:
        return X, y

    stratify = None
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if counts.min() >= 2:
            stratify = y

    _, X_sub, _, y_sub = train_test_split(
        X,
        y,
        test_size=args.max_samples_per_dataset,
        random_state=args.random_state,
        stratify=stratify,
    )
    return X_sub, y_sub


def _fetch_toy_classification_dataset(dataset):
    feature_names = ["Feature #0", "Feature #1"]
    if dataset == "toy_moons":
        X, y = make_moons(n_samples=100, noise=0.13, random_state=42)
    elif dataset == "toy_gaussian_quantiles":
        X, y = make_gaussian_quantiles(
            n_samples=100,
            n_features=2,
            n_classes=2,
            random_state=42,
        )
    elif dataset == "toy_xor":
        X = np.random.RandomState(0).uniform(low=-1.0, high=1.0, size=(200, 2))
        y = np.logical_xor(X[:, 0] > 0.0, X[:, 1] > 0.0).astype(np.int32)
    else:
        raise KeyError(dataset)

    frame = pd.DataFrame(X, columns=feature_names)
    frame["class"] = y
    return frame[feature_names].to_numpy(dtype=np.float64), frame["class"].to_numpy()


def _fetch_dataset(dataset, task, args):
    if task == "classification" and dataset in TOY_CLASSIFICATION_DATASETS:
        X, y = _fetch_toy_classification_dataset(dataset)
    else:
        X, y = fetch_data(dataset, return_X_y=True)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if task == "regression":
        y = y.astype(np.float64)
    X_sub, y_sub = _subsample_dataset(X, y, task, args)
    return X, y, X_sub, y_sub


def _classification_cv(X, y, args):
    return StratifiedKFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=args.random_state,
    ).split(X, y)


def _regression_cv(X, args):
    return KFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=args.random_state,
    ).split(X)


def _score(task, y_true, pred):
    if task == "classification":
        return balanced_accuracy_score(y_true, pred), accuracy_score(y_true, pred)
    return r2_score(y_true, pred), math.sqrt(mean_squared_error(y_true, pred))


def run_dataset(dataset, task, args):
    stage = "fetch"
    try:
        with time_limit(args.dataset_timeout):
            X_full, y_full, X, y = _fetch_dataset(dataset, task, args)
            n_samples, n_features = X_full.shape
            rows = []
            models = (
                _classification_models(
                    args.random_state,
                    args.n_estimators,
                    args.n_center_candidates,
                    args.radius_candidates,
                    args.center_strategy,
                    args.prune_single_trees,
                    args.pruning_validation_fraction,
                    args.max_pruning_alphas,
                )
                if task == "classification"
                else _regression_models(
                    args.random_state,
                    args.n_estimators,
                    args.n_center_candidates,
                    args.radius_candidates,
                    args.center_strategy,
                    args.prune_single_trees,
                    args.pruning_validation_fraction,
                    args.max_pruning_alphas,
                )
            )
            splits = (
                list(_classification_cv(X, y, args))
                if task == "classification"
                else list(_regression_cv(X, args))
            )

            for fold, (train, test) in enumerate(splits):
                for model_name, estimator in models.items():
                    stage = "fit_predict"
                    model = _pipeline(estimator)
                    start_total = time.perf_counter()
                    with time_limit(args.model_timeout):
                        start = time.perf_counter()
                        model.fit(X[train], y[train])
                        fit_time = time.perf_counter() - start
                        start = time.perf_counter()
                        pred = model.predict(X[test])
                        predict_time = time.perf_counter() - start
                    total_time = time.perf_counter() - start_total
                    score, secondary_score = _score(task, y[test], pred)
                    row = {
                        "task": task,
                        "dataset": dataset,
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_used_samples": X.shape[0],
                        "fold": fold,
                        "model": model_name,
                        "score": score,
                        "secondary_score": secondary_score,
                        "fit_time_s": fit_time,
                        "predict_time_s": predict_time,
                        "total_time_s": total_time,
                        "random_state": args.random_state,
                    }
                    rows.append(row)
                    if args.verbose:
                        metric = "bal_acc" if task == "classification" else "r2"
                        print(
                            f"{task[:3]} {dataset:>42s} fold={fold} {model_name:>23s} "
                            f"{metric}={score:.3f} fit={fit_time:.3f}s",
                            flush=True,
                        )
            print(
                f"OK {task} {dataset}: {len(rows)} rows, "
                f"{X.shape[0]}/{n_samples} samples used",
                flush=True,
            )
            return rows, None
    except Exception as exc:
        err = {
            "task": task,
            "dataset": dataset,
            "stage": stage,
            "model": model_name if "model_name" in locals() else "",
            "fold": fold if "fold" in locals() else "",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(
            f"SKIP {task} {dataset}: {err['stage']} {err['model']} "
            f"{err['error_type']}: {err['error']}",
            flush=True,
        )
        return [], err


def summarize(output_path, error_path, args):
    if not output_path.exists():
        return None, None
    results = pd.read_csv(output_path)
    if error_path.exists():
        errors = pd.read_csv(error_path)
        errors = errors.drop_duplicates(subset=["task", "dataset"], keep="first")
        errors.to_csv(error_path, index=False)

    expected_rows = args.cv * len(
        _classification_models(
            0,
            args.n_estimators,
            args.n_center_candidates,
            args.radius_candidates,
            args.center_strategy,
            args.prune_single_trees,
            args.pruning_validation_fraction,
            args.max_pruning_alphas,
        )
    )
    complete = (
        results.groupby(["task", "dataset"])["model"]
        .count()
        .reset_index(name="n_rows")
        .query("n_rows == @expected_rows")
    )
    complete_keys = set(map(tuple, complete[["task", "dataset"]].to_numpy()))
    mask = [(task, dataset) in complete_keys for task, dataset in results[["task", "dataset"]].to_numpy()]
    complete_results = results.loc[mask].copy()

    summary = (
        complete_results.groupby(["task", "dataset", "model"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            secondary_mean=("secondary_score", "mean"),
            fit_time_mean_s=("fit_time_s", "mean"),
            predict_time_mean_s=("predict_time_s", "mean"),
            total_time_mean_s=("total_time_s", "mean"),
        )
        .sort_values(["task", "dataset", "score_mean"], ascending=[True, True, False])
    )

    ranked = summary.copy()
    ranked["score_rank"] = ranked.groupby(["task", "dataset"])["score_mean"].rank(
        ascending=False,
        method="average",
    )
    ranked["fit_time_rank"] = ranked.groupby(["task", "dataset"])["fit_time_mean_s"].rank(
        ascending=True,
        method="average",
    )
    model_summary = (
        ranked.groupby(["task", "model"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            score_rank_mean=("score_rank", "mean"),
            fit_time_rank_mean=("fit_time_rank", "mean"),
            score_mean=("score_mean", "mean"),
            fit_time_mean_s=("fit_time_mean_s", "mean"),
            predict_time_mean_s=("predict_time_mean_s", "mean"),
        )
        .sort_values(["task", "score_rank_mean", "fit_time_rank_mean"])
    )

    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    model_summary_path = output_path.with_name(output_path.stem + "_model_summary.csv")
    markdown_path = output_path.with_suffix(".md")
    summary.to_csv(summary_path, index=False)
    model_summary.to_csv(model_summary_path, index=False)
    write_markdown_summary(markdown_path, results, summary, model_summary, error_path, args)
    return summary_path, model_summary_path


def write_markdown_summary(markdown_path, results, summary, model_summary, error_path, args):
    errors = pd.read_csv(error_path) if error_path.exists() else pd.DataFrame(columns=ERROR_COLUMNS)
    usage = results[["task", "dataset", "n_samples", "n_used_samples"]].drop_duplicates()
    usage = usage.assign(capped=usage["n_used_samples"] < usage["n_samples"])
    usage_summary = (
        usage.groupby(["task", "capped"])["dataset"]
        .nunique()
        .rename("n_datasets")
        .reset_index()
        .replace({"capped": {False: "full rows", True: "sample capped"}})
    )
    lines = [
        "# Full PMLB Spherical/Oblique Tree Benchmark",
        "",
        "Primary score is balanced accuracy for classification and R2 for regression.",
        "Secondary score is accuracy for classification and RMSE for regression.",
        "Dataset-level summary includes only datasets with all requested model/fold rows.",
        "",
        f"- CV folds: {args.cv}",
        f"- Forest estimators: {args.n_estimators}",
        f"- Spherical center candidates per node: {args.n_center_candidates}",
        f"- Spherical radius candidates per center: {args.radius_candidates}",
        f"- Spherical center strategy: {args.center_strategy}",
        f"- Single-tree CCP pruning: {args.prune_single_trees}",
        f"- Pruning validation fraction: {args.pruning_validation_fraction}",
        f"- Max pruning alphas: {args.max_pruning_alphas}",
        f"- Last resume max samples per dataset: {args.max_samples_per_dataset}",
        f"- Dataset timeout: {args.dataset_timeout}",
        f"- Model timeout: {args.model_timeout}",
        f"- Complete datasets summarized: {summary[['task', 'dataset']].drop_duplicates().shape[0]}",
        f"- Skipped/error datasets: {errors[['task', 'dataset']].drop_duplicates().shape[0]}",
        "",
        "## Sample Usage",
        "",
        usage_summary.to_markdown(index=False),
        "",
        "## Model Summary",
        "",
        model_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    if not errors.empty:
        lines.extend(
            [
                "## First Errors",
                "",
                errors[
                    ["task", "dataset", "stage", "model", "fold", "error_type", "error"]
                ]
                .head(30)
                .to_markdown(index=False),
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-datasets", nargs="+")
    parser.add_argument("--regression-datasets", nargs="+")
    parser.add_argument("--max-datasets-per-task", type=int)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--n-center-candidates", type=int, default=8)
    parser.add_argument("--radius-candidates", type=_parse_radius_candidates, default=64)
    parser.add_argument(
        "--center-strategy",
        choices=CENTER_STRATEGIES,
        default="default",
    )
    parser.add_argument("--prune-single-trees", action="store_true")
    parser.add_argument("--pruning-validation-fraction", type=float, default=0.25)
    parser.add_argument("--max-pruning-alphas", type=int, default=5)
    parser.add_argument("--max-samples-per-dataset", type=int)
    parser.add_argument("--dataset-timeout", type=int, default=0)
    parser.add_argument("--model-timeout", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-toy-datasets", action="store_true")
    parser.add_argument("--dataset-spec-file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks_nonasv/results/spherical_pmlb_full.csv"),
    )
    return parser.parse_args()


def _dataset_list(user_datasets, pmlb_datasets, max_datasets):
    datasets = list(user_datasets) if user_datasets is not None else list(pmlb_datasets)
    if max_datasets is not None:
        datasets = datasets[:max_datasets]
    return datasets


def _tasks_from_spec_file(path):
    spec = pd.read_csv(path)
    if not {"task", "dataset"}.issubset(spec.columns):
        raise ValueError("dataset spec file must have 'task' and 'dataset' columns.")
    grouped = []
    for task in ("classification", "regression"):
        datasets = spec.loc[spec["task"] == task, "dataset"].astype(str).tolist()
        grouped.append((task, datasets))
    return grouped


def main():
    args = parse_args()
    error_path = args.output.with_name(args.output.stem + "_errors.csv")
    if args.summarize_only:
        summarize(args.output, error_path, args)
        return

    completed = set() if args.rerun_existing else _load_seen_datasets(args.output, error_path)
    if args.dataset_spec_file is not None:
        tasks = _tasks_from_spec_file(args.dataset_spec_file)
    else:
        classification_defaults = list(classification_dataset_names)
        if not args.no_toy_datasets:
            classification_defaults = list(TOY_CLASSIFICATION_DATASETS) + classification_defaults
        tasks = [
            (
                "classification",
                _dataset_list(
                    args.classification_datasets,
                    classification_defaults,
                    args.max_datasets_per_task,
                ),
            ),
            (
                "regression",
                _dataset_list(
                    args.regression_datasets,
                    regression_dataset_names,
                    args.max_datasets_per_task,
                ),
            ),
        ]

    for task, datasets in tasks:
        for dataset in datasets:
            if (task, dataset) in completed:
                print(f"SKIP existing {task} {dataset}", flush=True)
                continue
            rows, err = run_dataset(dataset, task, args)
            if rows:
                _append_rows(args.output, rows, RAW_COLUMNS)
                completed.add((task, dataset))
            if err is not None:
                _append_rows(error_path, [err], ERROR_COLUMNS)

    summary_path, model_summary_path = summarize(args.output, error_path, args)
    print(f"Wrote raw results to {args.output}")
    print(f"Wrote errors to {error_path}")
    print(f"Wrote dataset summary to {summary_path}")
    print(f"Wrote model summary to {model_summary_path}")
    print(f"Wrote markdown summary to {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
