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


def _classification_models(random_state, n_estimators, n_center_candidates, radius_candidates):
    return {
        "CART": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
        "SphericalTree": SphericalDecisionTreeClassifier(
            random_state=random_state,
            max_features=None,
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
        ),
        "SphericalRandomForest": SphericalRandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
            max_features="sqrt",
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
        ),
        "ObliqueTree": ObliqueDecisionTreeClassifier(random_state=random_state),
        "ObliqueRandomForest": ObliqueRandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
    }


def _regression_models(random_state, n_estimators, n_center_candidates, radius_candidates):
    return {
        "CART": DecisionTreeRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
        "SphericalTree": SphericalDecisionTreeRegressor(
            random_state=random_state,
            max_features=None,
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
        ),
        "SphericalRandomForest": SphericalRandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
            max_features="sqrt",
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
        ),
        "ObliqueTree": ObliqueDecisionTreeRegressor(random_state=random_state),
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


def _fetch_dataset(dataset, task, args):
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
                )
                if task == "classification"
                else _regression_models(
                    args.random_state,
                    args.n_estimators,
                    args.n_center_candidates,
                    args.radius_candidates,
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
        _classification_models(0, args.n_estimators, args.n_center_candidates, args.radius_candidates)
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
    parser.add_argument("--radius-candidates", type=int, default=64)
    parser.add_argument("--max-samples-per-dataset", type=int)
    parser.add_argument("--dataset-timeout", type=int, default=0)
    parser.add_argument("--model-timeout", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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


def main():
    args = parse_args()
    error_path = args.output.with_name(args.output.stem + "_errors.csv")
    if args.summarize_only:
        summarize(args.output, error_path, args)
        return

    completed = set() if args.rerun_existing else _load_seen_datasets(args.output, error_path)
    tasks = [
        (
            "classification",
            _dataset_list(
                args.classification_datasets,
                classification_dataset_names,
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
