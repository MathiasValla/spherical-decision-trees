"""Small PMLB benchmark for spherical tree prototypes.

This script intentionally keeps the benchmark modest: a few small PMLB
datasets, conventional tree/forest defaults, and a fixed 3-fold CV split.
It is meant as an early research signal, not a final benchmark suite.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from treeple.ensemble import (
    ObliqueRandomForestClassifier,
    ObliqueRandomForestRegressor,
    SphericalRandomForestClassifier,
    SphericalRandomForestRegressor,
)
from treeple.tree import SphericalDecisionTreeClassifier, SphericalDecisionTreeRegressor


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


DEFAULT_CLASSIFICATION_DATASETS = ("iris", "breast_cancer", "prnn_crabs")
DEFAULT_REGRESSION_DATASETS = ("1027_ESL", "192_vineyard")
CENTER_STRATEGIES = ("default", "random", "target", "hybrid")


def _spherical_model_name(base_name, center_strategy, center_strategies):
    if len(center_strategies) == 1 or center_strategy == "default":
        return base_name
    return f"{base_name}[{center_strategy}]"


def _center_strategy_kwargs(center_strategy):
    if center_strategy == "default":
        return {}
    return {"center_strategy": center_strategy}


def _classification_models(
    random_state,
    n_estimators,
    n_center_candidates,
    radius_candidates,
    center_strategies,
):
    models = {
        "CART": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
        "ObliqueRandomForest": ObliqueRandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
    }
    for center_strategy in center_strategies:
        center_strategy_kwargs = _center_strategy_kwargs(center_strategy)
        models[_spherical_model_name("SphericalTree", center_strategy, center_strategies)] = (
            SphericalDecisionTreeClassifier(
                random_state=random_state,
                max_features=None,
                n_center_candidates=n_center_candidates,
                radius_candidates=radius_candidates,
                **center_strategy_kwargs,
            )
        )
        models[
            _spherical_model_name("SphericalRandomForest", center_strategy, center_strategies)
        ] = SphericalRandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
            max_features="sqrt",
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
            **center_strategy_kwargs,
        )
    return models


def _regression_models(
    random_state,
    n_estimators,
    n_center_candidates,
    radius_candidates,
    center_strategies,
):
    models = {
        "CART": DecisionTreeRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
        "ObliqueRandomForest": ObliqueRandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
    }
    for center_strategy in center_strategies:
        center_strategy_kwargs = _center_strategy_kwargs(center_strategy)
        models[_spherical_model_name("SphericalTree", center_strategy, center_strategies)] = (
            SphericalDecisionTreeRegressor(
                random_state=random_state,
                max_features=None,
                n_center_candidates=n_center_candidates,
                radius_candidates=radius_candidates,
                **center_strategy_kwargs,
            )
        )
        models[
            _spherical_model_name("SphericalRandomForest", center_strategy, center_strategies)
        ] = SphericalRandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
            max_features="sqrt",
            n_center_candidates=n_center_candidates,
            radius_candidates=radius_candidates,
            **center_strategy_kwargs,
        )
    return models


def _pipeline(estimator):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def _dataset_size(X):
    return X.shape[0], X.shape[1]


def run_classification_dataset(dataset, args):
    X, y = fetch_data(dataset, return_X_y=True)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_samples, n_features = _dataset_size(X)
    cv = StratifiedKFold(
        n_splits=args.cv,
        shuffle=True,
        random_state=args.random_state,
    )
    models = _classification_models(
        args.random_state,
        args.n_estimators,
        args.n_center_candidates,
        args.radius_candidates,
        args.center_strategies,
    )
    rows = []

    for fold, (train, test) in enumerate(cv.split(X, y)):
        for model_name, estimator in models.items():
            model = _pipeline(estimator)
            start = time.perf_counter()
            model.fit(X[train], y[train])
            fit_time = time.perf_counter() - start
            start = time.perf_counter()
            pred = model.predict(X[test])
            predict_time = time.perf_counter() - start
            rows.append(
                {
                    "task": "classification",
                    "dataset": dataset,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "fold": fold,
                    "model": model_name,
                    "score": balanced_accuracy_score(y[test], pred),
                    "secondary_score": accuracy_score(y[test], pred),
                    "fit_time_s": fit_time,
                    "predict_time_s": predict_time,
                }
            )
            print(
                f"{dataset:>20s} fold={fold} {model_name:>23s} "
                f"bal_acc={rows[-1]['score']:.3f} fit={fit_time:.3f}s",
                flush=True,
            )
    return rows


def run_regression_dataset(dataset, args):
    X, y = fetch_data(dataset, return_X_y=True)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_samples, n_features = _dataset_size(X)
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    models = _regression_models(
        args.random_state,
        args.n_estimators,
        args.n_center_candidates,
        args.radius_candidates,
        args.center_strategies,
    )
    rows = []

    for fold, (train, test) in enumerate(cv.split(X, y)):
        for model_name, estimator in models.items():
            model = _pipeline(estimator)
            start = time.perf_counter()
            model.fit(X[train], y[train])
            fit_time = time.perf_counter() - start
            start = time.perf_counter()
            pred = model.predict(X[test])
            predict_time = time.perf_counter() - start
            rmse = math.sqrt(mean_squared_error(y[test], pred))
            rows.append(
                {
                    "task": "regression",
                    "dataset": dataset,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "fold": fold,
                    "model": model_name,
                    "score": r2_score(y[test], pred),
                    "secondary_score": rmse,
                    "fit_time_s": fit_time,
                    "predict_time_s": predict_time,
                }
            )
            print(
                f"{dataset:>20s} fold={fold} {model_name:>23s} "
                f"r2={rows[-1]['score']:.3f} fit={fit_time:.3f}s",
                flush=True,
            )
    return rows


def summarize(results):
    summary = (
        results.groupby(["task", "dataset", "model"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            secondary_mean=("secondary_score", "mean"),
            fit_time_mean_s=("fit_time_s", "mean"),
            predict_time_mean_s=("predict_time_s", "mean"),
        )
        .sort_values(["task", "dataset", "score_mean"], ascending=[True, True, False])
    )
    return summary


def write_markdown_summary(summary, output_path, args):
    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# Small PMLB Spherical Tree Benchmark",
        "",
        "Primary score is balanced accuracy for classification and R2 for regression.",
        "Secondary score is accuracy for classification and RMSE for regression.",
        "",
        f"- CV folds: {args.cv}",
        f"- Forest estimators: {args.n_estimators}",
        f"- Spherical center candidates per node: {args.n_center_candidates}",
        f"- Spherical radius candidates per center: {args.radius_candidates}",
        f"- Spherical center strategies: {', '.join(args.center_strategies)}",
        "",
    ]
    for (task, dataset), frame in summary.groupby(["task", "dataset"], sort=False):
        lines.append(f"## {dataset} ({task})")
        lines.append("")
        lines.append(
            frame[
                [
                    "model",
                    "score_mean",
                    "score_std",
                    "secondary_mean",
                    "fit_time_mean_s",
                    "predict_time_mean_s",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classification-datasets",
        nargs="+",
        default=DEFAULT_CLASSIFICATION_DATASETS,
    )
    parser.add_argument("--regression-datasets", nargs="+", default=DEFAULT_REGRESSION_DATASETS)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--n-center-candidates", type=int, default=8)
    parser.add_argument("--radius-candidates", type=int, default=64)
    parser.add_argument(
        "--center-strategies",
        nargs="+",
        choices=CENTER_STRATEGIES,
        default=("default",),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks_nonasv/results/spherical_pmlb_small.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for dataset in args.classification_datasets:
        rows.extend(run_classification_dataset(dataset, args))
    for dataset in args.regression_datasets:
        rows.extend(run_regression_dataset(dataset, args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.output, index=False)
    summary = summarize(results)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    markdown_path = write_markdown_summary(summary, args.output, args)

    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nWrote raw results to {args.output}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote markdown summary to {markdown_path}")


if __name__ == "__main__":
    main()
