"""Mine practical heuristics for when spherical tree models win.

The analysis joins benchmark margins with simple dataset descriptors:

* sample/predictor ratios,
* classification target balance and number of classes,
* regression target scale/shape descriptors,
* predictor distribution imbalance, sparsity, discreteness, and redundancy.

The rules are descriptive rather than confirmatory. They are meant to suggest
research hypotheses and next benchmarks, not to certify a production model
selection policy.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "results" / ".matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RAW_CSV = RESULTS_DIR / "spherical_pmlb_full.csv"
REGIME_CSV = RESULTS_DIR / "spherical_pmlb_regime_table.csv"

RANDOM_STATE = 42
EPS = 1e-12

PRACTICAL_MARGIN = {
    "classification": 0.03,
    "regression": 0.05,
}
LARGE_MARGIN = {
    "classification": 0.05,
    "regression": 0.10,
}


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < EPS:
        return np.nan
    return float(num / den)


def _subsample_like_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    n_used_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_used_samples >= X.shape[0]:
        return X, y

    stratify = None
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if counts.size > 1 and counts.min() >= 2:
            stratify = y

    _, X_sub, _, y_sub = train_test_split(
        X,
        y,
        test_size=int(n_used_samples),
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return X_sub, y_sub


def _finite_imputed_matrix(X: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X)
    missing_fraction = 1.0 - float(finite.mean())
    col_missing = 1.0 - finite.mean(axis=0)

    X_nan = np.where(finite, X, np.nan)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(X_nan, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    X_imp = np.where(finite, X, medians)

    return X_imp, {
        "missing_fraction": missing_fraction,
        "missing_predictor_fraction": float(np.mean(col_missing > 0)),
        "max_predictor_missing_fraction": float(np.max(col_missing)),
    }


def _feature_descriptors(X: np.ndarray) -> dict[str, object]:
    X_imp, missing = _finite_imputed_matrix(X)
    n, p = X_imp.shape

    means = X_imp.mean(axis=0)
    stds = X_imp.std(axis=0)
    positive = stds > EPS
    positive_stds = stds[positive]
    constant_fraction = 1.0 - float(np.mean(positive)) if p else np.nan

    if positive_stds.size:
        p05 = np.percentile(positive_stds, 5)
        p95 = np.percentile(positive_stds, 95)
        scale_ratio = _safe_ratio(p95, max(p05, EPS))
    else:
        scale_ratio = np.nan

    unique_counts = []
    if n * p <= 8_000_000:
        for j in range(p):
            unique_counts.append(np.unique(X_imp[:, j]).size)
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        row_idx = rng.choice(n, size=min(n, 5000), replace=False)
        for j in range(p):
            unique_counts.append(np.unique(X_imp[row_idx, j]).size)
    unique_counts = np.asarray(unique_counts)

    binary_fraction = float(np.mean(unique_counts <= 2))
    discrete_fraction = float(np.mean(unique_counts <= 10))
    zero_fraction = float(np.mean(X_imp == 0))

    if positive.any():
        X_valid = X_imp[:, positive]
        Z = (X_valid - means[positive]) / np.maximum(stds[positive], EPS)
        skew = np.mean(Z**3, axis=0)
        median_abs_skew = float(np.nanmedian(np.abs(skew)))
        p90_abs_skew = float(np.nanpercentile(np.abs(skew), 90))
    else:
        median_abs_skew = np.nan
        p90_abs_skew = np.nan

    valid_indices = np.flatnonzero(positive)
    if valid_indices.size > 1:
        if valid_indices.size > 200:
            rng = np.random.default_rng(RANDOM_STATE)
            valid_indices = np.sort(rng.choice(valid_indices, size=200, replace=False))
        X_corr = X_imp[:, valid_indices]
        X_corr = (X_corr - X_corr.mean(axis=0)) / np.maximum(X_corr.std(axis=0), EPS)
        corr = np.corrcoef(X_corr, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
        mean_abs_corr = float(np.mean(upper)) if upper.size else np.nan
        high_corr_pair_fraction = float(np.mean(upper >= 0.80)) if upper.size else 0.0
    else:
        mean_abs_corr = np.nan
        high_corr_pair_fraction = np.nan

    mostly_binary = binary_fraction >= 0.80
    mostly_discrete = discrete_fraction >= 0.80
    sparse_predictors = zero_fraction >= 0.50
    scale_unbalanced = bool(np.isfinite(scale_ratio) and scale_ratio >= 100)
    skewed_predictors = bool(np.isfinite(median_abs_skew) and median_abs_skew >= 2)
    redundant_predictors = bool(
        np.isfinite(high_corr_pair_fraction)
        and high_corr_pair_fraction >= 0.10
        or np.isfinite(mean_abs_corr)
        and mean_abs_corr >= 0.30
    )
    balanced_predictors = bool(
        missing["missing_fraction"] < 0.05
        and np.isfinite(scale_ratio)
        and scale_ratio <= 10
        and np.isfinite(median_abs_skew)
        and median_abs_skew <= 1
        and (not np.isfinite(high_corr_pair_fraction) or high_corr_pair_fraction < 0.02)
        and zero_fraction < 0.25
    )

    profile = []
    if mostly_binary:
        profile.append("mostly_binary")
    elif mostly_discrete:
        profile.append("mostly_discrete")
    else:
        profile.append("mostly_continuous")
    if sparse_predictors:
        profile.append("sparse")
    if scale_unbalanced:
        profile.append("scale_unbalanced")
    if skewed_predictors:
        profile.append("skewed")
    if redundant_predictors:
        profile.append("redundant")
    if balanced_predictors:
        profile.append("balanced")

    return {
        **missing,
        "constant_predictor_fraction": constant_fraction,
        "feature_scale_p95_p05_ratio": scale_ratio,
        "binary_predictor_fraction": binary_fraction,
        "discrete_predictor_fraction": discrete_fraction,
        "zero_fraction": zero_fraction,
        "median_abs_feature_skew": median_abs_skew,
        "p90_abs_feature_skew": p90_abs_skew,
        "mean_abs_corr": mean_abs_corr,
        "high_corr_pair_fraction": high_corr_pair_fraction,
        "mostly_binary_predictors": mostly_binary,
        "mostly_discrete_predictors": mostly_discrete,
        "sparse_predictors": sparse_predictors,
        "scale_unbalanced_predictors": scale_unbalanced,
        "skewed_predictors": skewed_predictors,
        "redundant_predictors": redundant_predictors,
        "balanced_predictors": balanced_predictors,
        "predictor_profile": ";".join(profile),
    }


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    return float(-(probs * np.log(probs)).sum() / np.log(probs.size))


def _classification_descriptors(y: np.ndarray) -> dict[str, object]:
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    n_classes = int(classes.size)
    min_share = float(probs.min())
    max_share = float(probs.max())
    entropy = _entropy(probs)
    imbalance_ratio = _safe_ratio(max_share, min_share)
    is_binary = n_classes == 2

    if is_binary:
        if min_share >= 0.35:
            target_balance = "balanced_binary"
        elif min_share < 0.20:
            target_balance = "unbalanced_binary"
        else:
            target_balance = "moderate_binary"
    else:
        if entropy >= 0.85 and min_share >= 0.05:
            target_balance = "balanced_multiclass"
        elif entropy < 0.75 or min_share < 0.03:
            target_balance = "unbalanced_multiclass"
        else:
            target_balance = "moderate_multiclass"

    return {
        "n_classes": n_classes,
        "is_binary_classification": is_binary,
        "is_multiclass_classification": not is_binary,
        "min_class_share": min_share,
        "max_class_share": max_share,
        "target_entropy_norm": entropy,
        "target_imbalance_ratio": imbalance_ratio,
        "target_balance": target_balance,
        "target_balanced": target_balance.startswith("balanced"),
        "target_unbalanced": target_balance.startswith("unbalanced"),
    }


def _regression_descriptors(y: np.ndarray) -> dict[str, object]:
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return {
            "target_abs_median": np.nan,
            "target_abs_mean": np.nan,
            "target_mean": np.nan,
            "target_std": np.nan,
            "target_range": np.nan,
            "target_cv_about_mean": np.nan,
            "target_skew": np.nan,
            "target_zero_fraction": np.nan,
        }

    target_mean = float(np.mean(y))
    target_std = float(np.std(y))
    centered = y - target_mean
    target_skew = (
        float(np.mean((centered / max(target_std, EPS)) ** 3))
        if target_std > EPS
        else 0.0
    )
    return {
        "target_abs_median": float(np.median(np.abs(y))),
        "target_abs_mean": float(np.mean(np.abs(y))),
        "target_mean": target_mean,
        "target_std": target_std,
        "target_range": float(np.max(y) - np.min(y)),
        "target_cv_about_mean": _safe_ratio(target_std, abs(target_mean) + EPS),
        "target_skew": target_skew,
        "target_zero_fraction": float(np.mean(y == 0)),
    }


def compute_dataset_metadata(regime: pd.DataFrame) -> pd.DataFrame:
    keys = (
        regime[["task", "dataset", "n_samples", "n_used_samples", "n_features"]]
        .drop_duplicates()
        .sort_values(["task", "dataset"])
    )

    rows = []
    for i, row in enumerate(keys.itertuples(index=False), start=1):
        task = row.task
        dataset = row.dataset
        print(f"[{i}/{len(keys)}] metadata {task} {dataset}", flush=True)
        try:
            X, y = fetch_data(dataset, return_X_y=True)
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y)
            if task == "regression":
                y = y.astype(np.float64)
            X, y = _subsample_like_benchmark(X, y, task, int(row.n_used_samples))
            descriptors = _feature_descriptors(X)
            if task == "classification":
                descriptors.update(_classification_descriptors(y))
            else:
                descriptors.update(_regression_descriptors(y))
            descriptors.update(
                {
                    "task": task,
                    "dataset": dataset,
                    "descriptor_status": "ok",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive for remote datasets
            descriptors = {
                "task": task,
                "dataset": dataset,
                "descriptor_status": f"{type(exc).__name__}: {exc}",
            }
        rows.append(descriptors)

    metadata = pd.DataFrame(rows)
    regression = metadata["task"] == "regression"
    if regression.any() and "target_abs_median" in metadata:
        scale = np.log10(1.0 + metadata.loc[regression, "target_abs_median"].astype(float))
        q25, q75 = scale.quantile([0.25, 0.75])
        values = np.full(metadata.shape[0], "not_regression", dtype=object)
        reg_index = metadata.index[regression]
        values[reg_index] = np.where(
            scale <= q25,
            "low_target_values",
            np.where(scale >= q75, "high_target_values", "medium_target_values"),
        )
        metadata["target_value_scale"] = values
    else:
        metadata["target_value_scale"] = "not_regression"

    if "target_skew" in metadata:
        metadata["target_skew_level"] = np.where(
            metadata["task"] != "regression",
            "not_regression",
            np.where(
                metadata["target_skew"].abs() >= 2,
                "strongly_skewed_target",
                np.where(
                    metadata["target_skew"].abs() >= 1,
                    "moderately_skewed_target",
                    "weakly_skewed_target",
                ),
            ),
        )
    return metadata


def fold_margin_table(raw: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in regime.itertuples(index=False):
        sub = raw[(raw["task"] == row.task) & (raw["dataset"] == row.dataset)]
        pivot = sub.pivot_table(index="fold", columns="model", values="score", aggfunc="mean")
        if row.best_spherical_model not in pivot or row.best_non_spherical_model not in pivot:
            continue
        margins = pivot[row.best_spherical_model] - pivot[row.best_non_spherical_model]
        margins = margins.dropna()
        if margins.empty:
            continue
        mean = float(margins.mean())
        std = float(margins.std(ddof=1)) if margins.size > 1 else 0.0
        se = std / math.sqrt(margins.size) if margins.size > 1 else 0.0
        all_positive = bool((margins > 0).all())
        all_negative = bool((margins < 0).all())
        two_se_consistent = bool(mean > 0 and (se <= EPS or mean > 2 * se))
        two_se_consistent_loss = bool(mean < 0 and (se <= EPS or abs(mean) > 2 * se))
        rows.append(
            {
                "comparison": row.comparison,
                "task": row.task,
                "dataset": row.dataset,
                "best_spherical_model": row.best_spherical_model,
                "best_non_spherical_model": row.best_non_spherical_model,
                "fold_margin_mean": mean,
                "fold_margin_std": std,
                "fold_margin_se": se,
                "fold_margin_min": float(margins.min()),
                "fold_margin_max": float(margins.max()),
                "positive_fold_count": int((margins > 0).sum()),
                "n_folds": int(margins.size),
                "all_folds_positive": all_positive,
                "all_folds_negative": all_negative,
                "two_se_consistent": two_se_consistent,
                "two_se_consistent_loss": two_se_consistent_loss,
                "practical_margin_win": bool(
                    all_positive and mean >= PRACTICAL_MARGIN[row.task]
                ),
                "large_margin_win": bool(all_positive and mean >= LARGE_MARGIN[row.task]),
                "consistent_practical_win": bool(
                    all_positive
                    and two_se_consistent
                    and mean >= PRACTICAL_MARGIN[row.task]
                ),
                "practical_margin_loss": bool(
                    all_negative and mean <= -PRACTICAL_MARGIN[row.task]
                ),
                "large_margin_loss": bool(
                    all_negative and mean <= -LARGE_MARGIN[row.task]
                ),
                "consistent_practical_loss": bool(
                    all_negative
                    and two_se_consistent_loss
                    and mean <= -PRACTICAL_MARGIN[row.task]
                ),
            }
        )
    return pd.DataFrame(rows)


def enrich_regime(
    regime: pd.DataFrame,
    metadata: pd.DataFrame,
    fold_margins: pd.DataFrame,
) -> pd.DataFrame:
    enriched = regime.merge(metadata, on=["task", "dataset"], how="left")
    enriched = enriched.merge(
        fold_margins,
        on=[
            "comparison",
            "task",
            "dataset",
            "best_spherical_model",
            "best_non_spherical_model",
        ],
        how="left",
    )
    enriched["n_over_p"] = enriched["n_used_samples"] / enriched["n_features"]
    enriched["p_over_n"] = enriched["n_features"] / enriched["n_used_samples"]
    enriched["log10_n"] = np.log10(enriched["n_used_samples"])
    enriched["log10_p"] = np.log10(enriched["n_features"])
    enriched["log10_n_over_p"] = np.log10(enriched["n_over_p"])
    enriched["log10_p_over_n"] = np.log10(enriched["p_over_n"])
    enriched["p_ge_n"] = enriched["p_over_n"] >= 1
    enriched["p_at_least_10pct_n"] = enriched["p_over_n"] >= 0.10
    enriched["n_at_least_50p"] = enriched["n_over_p"] >= 50
    enriched["n_at_least_100p"] = enriched["n_over_p"] >= 100
    enriched["small_n"] = enriched["n_used_samples"] <= 250
    enriched["medium_or_small_n"] = enriched["n_used_samples"] <= 1000
    enriched["large_n"] = enriched["n_used_samples"] > 5000
    enriched["low_p"] = enriched["n_features"] <= 10
    enriched["moderate_p"] = (enriched["n_features"] > 10) & (enriched["n_features"] <= 50)
    enriched["high_p"] = enriched["n_features"] > 50
    enriched["very_high_p"] = enriched["n_features"] > 200
    return enriched


Condition = tuple[str, Callable[[pd.DataFrame], pd.Series]]


def condition_list(task: str) -> list[Condition]:
    generic: list[Condition] = [
        ("n <= 250", lambda df: df["small_n"]),
        ("n <= 1000", lambda df: df["medium_or_small_n"]),
        ("n > 1000", lambda df: ~df["medium_or_small_n"]),
        ("p <= 10", lambda df: df["low_p"]),
        ("10 < p <= 50", lambda df: df["moderate_p"]),
        ("p > 50", lambda df: df["high_p"]),
        ("p > 200", lambda df: df["very_high_p"]),
        ("p >= n", lambda df: df["p_ge_n"]),
        ("p/n >= 0.10", lambda df: df["p_at_least_10pct_n"]),
        ("n/p >= 50", lambda df: df["n_at_least_50p"]),
        ("n/p >= 100", lambda df: df["n_at_least_100p"]),
        ("mostly binary predictors", lambda df: df["mostly_binary_predictors"]),
        ("mostly discrete predictors", lambda df: df["mostly_discrete_predictors"]),
        ("sparse predictors", lambda df: df["sparse_predictors"]),
        ("balanced predictors", lambda df: df["balanced_predictors"]),
        ("scale-unbalanced predictors", lambda df: df["scale_unbalanced_predictors"]),
        ("skewed predictors", lambda df: df["skewed_predictors"]),
        ("redundant predictors", lambda df: df["redundant_predictors"]),
        (
            "high predictor correlation",
            lambda df: df["high_corr_pair_fraction"].fillna(0) >= 0.10,
        ),
    ]
    if task == "classification":
        generic.extend(
            [
                ("binary classification", lambda df: df["is_binary_classification"]),
                ("multiclass classification", lambda df: df["is_multiclass_classification"]),
                ("balanced target", lambda df: df["target_balanced"]),
                ("unbalanced target", lambda df: df["target_unbalanced"]),
                ("min class share >= 0.35", lambda df: df["min_class_share"] >= 0.35),
                ("target entropy >= 0.9", lambda df: df["target_entropy_norm"] >= 0.90),
                ("target entropy < 0.75", lambda df: df["target_entropy_norm"] < 0.75),
            ]
        )
    else:
        generic.extend(
            [
                (
                    "low target values",
                    lambda df: df["target_value_scale"] == "low_target_values",
                ),
                (
                    "high target values",
                    lambda df: df["target_value_scale"] == "high_target_values",
                ),
                (
                    "weakly skewed target",
                    lambda df: df["target_skew_level"] == "weakly_skewed_target",
                ),
                (
                    "strongly skewed target",
                    lambda df: df["target_skew_level"] == "strongly_skewed_target",
                ),
                ("target CV >= 1", lambda df: df["target_cv_about_mean"] >= 1),
            ]
        )
    return generic


def mine_rules(
    enriched: pd.DataFrame,
    max_terms: int = 3,
) -> pd.DataFrame:
    rows = []
    for (comparison, task), frame in enriched.groupby(["comparison", "task"]):
        frame = frame[frame["descriptor_status"] == "ok"].copy()
        if frame.empty:
            continue
        baseline = float(frame["spherical_wins"].mean())
        min_support = 8 if frame.shape[0] >= 100 else 5
        conditions = condition_list(task)
        masks: list[tuple[str, pd.Series]] = []
        for name, func in conditions:
            mask = func(frame).fillna(False).astype(bool)
            if min_support <= mask.sum() < frame.shape[0]:
                masks.append((name, mask))

        for n_terms in range(1, max_terms + 1):
            for combo in combinations(masks, n_terms):
                names = [item[0] for item in combo]
                mask = combo[0][1].copy()
                for _, next_mask in combo[1:]:
                    mask &= next_mask
                support = int(mask.sum())
                if support < min_support or support == frame.shape[0]:
                    continue
                subset = frame[mask]
                win_rate = float(subset["spherical_wins"].mean())
                rows.append(
                    {
                        "comparison": comparison,
                        "task": task,
                        "rule": " & ".join(names),
                        "n_terms": n_terms,
                        "support": support,
                        "baseline_win_rate": baseline,
                        "spherical_win_rate": win_rate,
                        "lift_vs_baseline": _safe_ratio(win_rate, baseline),
                        "mean_spherical_margin": float(subset["spherical_margin"].mean()),
                        "median_spherical_margin": float(subset["spherical_margin"].median()),
                        "consistent_practical_win_rate": float(
                            subset["consistent_practical_win"].fillna(False).mean()
                        ),
                        "large_margin_win_rate": float(
                            subset["large_margin_win"].fillna(False).mean()
                        ),
                        "consistent_practical_loss_rate": float(
                            subset["consistent_practical_loss"].fillna(False).mean()
                        ),
                        "large_margin_loss_rate": float(
                            subset["large_margin_loss"].fillna(False).mean()
                        ),
                    }
                )
    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules
    return rules.sort_values(
        [
            "comparison",
            "task",
            "spherical_win_rate",
            "support",
            "mean_spherical_margin",
        ],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)


def summarize_contexts(enriched: pd.DataFrame) -> pd.DataFrame:
    context_cols = [
        "row_band",
        "feature_band",
        "predictor_profile",
        "target_balance",
        "target_value_scale",
        "target_skew_level",
    ]
    rows = []
    for col in context_cols:
        if col not in enriched:
            continue
        grouped = (
            enriched.groupby(["comparison", "task", col], dropna=False)
            .agg(
                support=("dataset", "nunique"),
                spherical_win_rate=("spherical_wins", "mean"),
                mean_spherical_margin=("spherical_margin", "mean"),
                median_spherical_margin=("spherical_margin", "median"),
                consistent_practical_win_rate=(
                    "consistent_practical_win",
                    lambda s: s.fillna(False).mean(),
                ),
                consistent_practical_loss_rate=(
                    "consistent_practical_loss",
                    lambda s: s.fillna(False).mean(),
                ),
            )
            .reset_index()
            .rename(columns={col: "context_value"})
        )
        grouped.insert(2, "context_variable", col)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def plot_top_rules(rules: pd.DataFrame) -> Path | None:
    selected_rows = []
    for (comparison, task), frame in rules.groupby(["comparison", "task"]):
        candidates = frame[
            (frame["support"] >= 8)
            & (frame["spherical_win_rate"] > frame["baseline_win_rate"])
            & (frame["mean_spherical_margin"] > 0)
        ].copy()
        selected_rows.append(candidates.head(5))
    if not selected_rows:
        return None

    selected = pd.concat(selected_rows, ignore_index=True)
    if selected.empty:
        return None
    selected["label"] = (
        selected["comparison"]
        + " / "
        + selected["task"]
        .str.replace("classification", "class.", regex=False)
        .str.replace("regression", "reg.", regex=False)
        + " | "
        + selected["rule"]
    )
    selected["label"] = selected["label"].str.wrap(62)
    selected = selected.sort_values("spherical_win_rate", ascending=True)

    height = max(8.0, 0.82 * selected.shape[0])
    fig, ax = plt.subplots(figsize=(14.0, height))
    colors = np.where(selected["mean_spherical_margin"] > 0, "#e45756", "#9aa1aa")
    ax.barh(selected["label"], selected["spherical_win_rate"], color=colors, alpha=0.85)
    ax.scatter(
        selected["baseline_win_rate"],
        np.arange(selected.shape[0]),
        color="#111111",
        s=24,
        label="baseline",
        zorder=3,
    )
    for y, (_, row) in enumerate(selected.iterrows()):
        ax.text(
            row["spherical_win_rate"] + 0.015,
            y,
            f"n={int(row['support'])}, lift={row['lift_vs_baseline']:.1f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, min(1.05, max(1.0, selected["spherical_win_rate"].max() + 0.20)))
    ax.set_xlabel("Spherical win rate inside rule")
    ax.set_title("Top descriptive rules for spherical wins")
    ax.grid(axis="x", color="#d9dde3", alpha=0.8)
    ax.legend(loc="lower right", frameon=False)
    ax.tick_params(axis="y", labelsize=8)
    fig.subplots_adjust(left=0.36, right=0.87, bottom=0.08, top=0.95)

    path = FIGURES_DIR / "spherical_heuristic_top_rules.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_poor_rules(rules: pd.DataFrame) -> Path | None:
    selected_rows = []
    for (comparison, task), frame in rules.groupby(["comparison", "task"]):
        candidates = frame[
            (frame["support"] >= 8)
            & (frame["spherical_win_rate"] < frame["baseline_win_rate"])
            & (frame["mean_spherical_margin"] < 0)
        ].copy()
        candidates = candidates.sort_values(
            ["spherical_win_rate", "mean_spherical_margin", "support"],
            ascending=[True, True, False],
        )
        selected_rows.append(candidates.head(5))
    if not selected_rows:
        return None

    selected = pd.concat(selected_rows, ignore_index=True)
    if selected.empty:
        return None
    selected["label"] = (
        selected["comparison"]
        + " / "
        + selected["task"]
        .str.replace("classification", "class.", regex=False)
        .str.replace("regression", "reg.", regex=False)
        + " | "
        + selected["rule"]
    )
    selected["label"] = selected["label"].str.wrap(62)
    selected = selected.sort_values("mean_spherical_margin", ascending=False)

    height = max(8.0, 0.82 * selected.shape[0])
    fig, ax = plt.subplots(figsize=(14.0, height))
    ax.barh(selected["label"], selected["mean_spherical_margin"], color="#4c78a8", alpha=0.82)
    ax.axvline(0, color="#111111", linewidth=0.9)
    for y, (_, row) in enumerate(selected.iterrows()):
        ax.text(
            min(row["mean_spherical_margin"] - 0.015, -0.01),
            y,
            f"n={int(row['support'])}, win={row['spherical_win_rate']:.2f}",
            va="center",
            ha="right",
            fontsize=8,
        )
    x_min = min(float(selected["mean_spherical_margin"].min()) * 1.18, -0.05)
    ax.set_xlim(x_min, 0.05)
    ax.set_xlabel("Mean spherical margin inside rule")
    ax.set_title("Top descriptive rules for poor spherical performance")
    ax.grid(axis="x", color="#d9dde3", alpha=0.8)
    ax.tick_params(axis="y", labelsize=8)
    fig.subplots_adjust(left=0.36, right=0.87, bottom=0.08, top=0.95)

    path = FIGURES_DIR / "spherical_heuristic_poor_rules.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows._"
    return df.to_markdown(index=False, floatfmt=floatfmt)


def write_report(
    enriched: pd.DataFrame,
    rules: pd.DataFrame,
    context_summary: pd.DataFrame,
    fold_margins: pd.DataFrame,
    rule_plot: Path | None,
    poor_rule_plot: Path | None,
) -> Path:
    path = RESULTS_DIR / "spherical_pmlb_heuristic_analysis.md"

    def top_rules(comparison: str, task: str, n: int = 8) -> pd.DataFrame:
        frame = rules[
            (rules["comparison"] == comparison)
            & (rules["task"] == task)
            & (rules["support"] >= 8)
            & (rules["spherical_win_rate"] > rules["baseline_win_rate"])
            & (rules["mean_spherical_margin"] > 0)
        ].copy()
        cols = [
            "rule",
            "support",
            "baseline_win_rate",
            "spherical_win_rate",
            "lift_vs_baseline",
            "mean_spherical_margin",
            "consistent_practical_win_rate",
        ]
        return frame[cols].head(n)

    def poor_rules(comparison: str, task: str, n: int = 8) -> pd.DataFrame:
        frame = rules[
            (rules["comparison"] == comparison)
            & (rules["task"] == task)
            & (rules["support"] >= 8)
            & (rules["spherical_win_rate"] < rules["baseline_win_rate"])
            & (rules["mean_spherical_margin"] < 0)
        ].copy()
        frame = frame.sort_values(
            ["spherical_win_rate", "mean_spherical_margin", "support"],
            ascending=[True, True, False],
        )
        cols = [
            "rule",
            "support",
            "baseline_win_rate",
            "spherical_win_rate",
            "lift_vs_baseline",
            "mean_spherical_margin",
            "consistent_practical_loss_rate",
        ]
        return frame[cols].head(n)

    strong = fold_margins[fold_margins["consistent_practical_win"]].copy()
    strong = strong.merge(
        enriched[
            [
                "comparison",
                "task",
                "dataset",
                "n_used_samples",
                "n_features",
                "n_over_p",
                "target_balance",
                "target_value_scale",
                "predictor_profile",
            ]
        ],
        on=["comparison", "task", "dataset"],
        how="left",
    )
    strong = strong.sort_values("fold_margin_mean", ascending=False)

    losses = fold_margins[fold_margins["consistent_practical_loss"]].copy()
    losses = losses.merge(
        enriched[
            [
                "comparison",
                "task",
                "dataset",
                "n_used_samples",
                "n_features",
                "n_over_p",
                "target_balance",
                "target_value_scale",
                "predictor_profile",
            ]
        ],
        on=["comparison", "task", "dataset"],
        how="left",
    )
    losses = losses.sort_values("fold_margin_mean", ascending=True)

    strong_counts = (
        fold_margins.groupby(["comparison", "task"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            consistent_practical_wins=("consistent_practical_win", "sum"),
            large_margin_wins=("large_margin_win", "sum"),
            consistent_practical_losses=("consistent_practical_loss", "sum"),
            large_margin_losses=("large_margin_loss", "sum"),
        )
        .sort_values(["comparison", "task"])
    )

    plot_line = (
        f"- `figures/{rule_plot.name}`\n" if rule_plot is not None else ""
    )
    poor_plot_line = (
        f"- `figures/{poor_rule_plot.name}`\n" if poor_rule_plot is not None else ""
    )
    strong_cols = [
        "comparison",
        "task",
        "dataset",
        "best_spherical_model",
        "best_non_spherical_model",
        "fold_margin_mean",
        "fold_margin_min",
        "n_used_samples",
        "n_features",
        "n_over_p",
        "target_balance",
        "target_value_scale",
        "predictor_profile",
    ]
    strong_display = strong[strong_cols].head(25).copy()
    for col in ["fold_margin_mean", "fold_margin_min", "n_over_p"]:
        strong_display[col] = strong_display[col].astype(float).round(3)

    loss_display = losses[strong_cols].head(25).copy()
    for col in ["fold_margin_mean", "fold_margin_min", "n_over_p"]:
        loss_display[col] = loss_display[col].astype(float).round(3)

    forest_class_rules = top_rules("forests", "classification")
    tree_class_rules = top_rules("trees", "classification")
    forest_reg_rules = top_rules("forests", "regression")
    tree_reg_rules = top_rules("trees", "regression")
    forest_class_poor_rules = poor_rules("forests", "classification")
    tree_class_poor_rules = poor_rules("trees", "classification")
    forest_reg_poor_rules = poor_rules("forests", "regression")
    tree_reg_poor_rules = poor_rules("trees", "regression")

    text = f"""# Spherical Heuristic Analysis

This report extends the PMLB benchmark regime analysis with target and predictor
descriptors. The response variable `spherical_wins` means that the best
spherical model beats the best non-spherical model in the same comparison set.

Target scale for regression is included because it was requested, but it should
not be interpreted causally: regression target values can be arbitrarily rescaled
without changing the prediction problem, and the benchmark scores use R2.

## Outputs

- `spherical_pmlb_dataset_descriptors.csv`
- `spherical_pmlb_heuristic_enriched.csv`
- `spherical_pmlb_heuristic_rules.csv`
- `spherical_pmlb_heuristic_context_summary.csv`
- `spherical_pmlb_heuristic_fold_margins.csv`
{plot_line}
{poor_plot_line}

## Strong Practical Wins

`consistent_practical_win` requires the spherical model to beat the selected
non-spherical comparator on every fold, to exceed a practical margin threshold
({PRACTICAL_MARGIN['classification']:.2f} balanced-accuracy/R2 points for
classification, {PRACTICAL_MARGIN['regression']:.2f} for regression), and to be
larger than two fold-level standard errors. With only three folds this is not a
formal significance test; it is a conservative practical screen.

{_md_table(strong_counts, floatfmt='.0f')}

## Top Strong Cases

{_md_table(strong_display)}

## Strong Practical Losses

The same fold screen can be read in the opposite direction:
`consistent_practical_loss` requires the best spherical model to lose to the
best non-spherical comparator on every fold, by at least the practical margin,
and by more than two fold-level standard errors.

{_md_table(strong_counts, floatfmt='.0f')}

## Top Strong Loss Cases

{_md_table(loss_display)}

## Heuristic Rules: Spherical Forests, Classification

{_md_table(forest_class_rules)}

## Heuristic Rules: Spherical Trees, Classification

{_md_table(tree_class_rules)}

## Heuristic Rules: Spherical Forests, Regression

{_md_table(forest_reg_rules)}

## Heuristic Rules: Spherical Trees, Regression

{_md_table(tree_reg_rules)}

## Poor-Performance Rules: Spherical Forests, Classification

{_md_table(forest_class_poor_rules)}

## Poor-Performance Rules: Spherical Trees, Classification

{_md_table(tree_class_poor_rules)}

## Poor-Performance Rules: Spherical Forests, Regression

{_md_table(forest_reg_poor_rules)}

## Poor-Performance Rules: Spherical Trees, Regression

{_md_table(tree_reg_poor_rules)}

## Practical Reading

- The strongest repeatable signal is classification with discrete/binary
  predictors and moderate dimensionality. This fits the geometric motivation:
  spherical splits can isolate compact interaction regions that axis-aligned
  splits need several levels to approximate.
- Spherical forests are the more promising default than single spherical trees.
  They keep many of the classification gains while reducing the brittleness of a
  single center/radius choice.
- Regression target magnitude is not a useful heuristic by itself. Any apparent
  high/low target-value rule should be treated as a dataset-family artifact,
  because target scaling is arbitrary under R2.
- Rows and predictors alone do not produce a clean monotone rule. The useful
  heuristic is conditional: low-to-moderate p, enough observations to estimate
  centers, discrete/binary or radially clustered predictors, and a classification
  target that is not extremely imbalanced.

## Candidate Heuristic

Use the spherical forest as a serious candidate when all or most of the following
hold:

1. The task is classification.
2. `p <= 50`, preferably `p <= 10` for the strongest practical gains.
3. `n/p >= 50`, so centers and radii are estimated from a reasonably dense cloud.
4. Predictors are mostly discrete, mostly binary, or sparse, or prior knowledge
   suggests compact/radial class regions.
5. The target is balanced or only moderately imbalanced. Unbalanced multiclass
   can still work in forests, but unbalanced binary tasks did not show a strong
   signal here.

For single spherical trees, be more conservative: the favorable cases are
classification datasets with high `n/p`, low-to-moderate `p`, and either balanced
predictors/targets or known radial geometry. The tree wins are less stable than
the forest wins.

Avoid treating regression target magnitude as a model-selection criterion. In
this benchmark, high and low target-value groups both had weak spherical win
rates and no consistent practical regression wins.

## Candidate Anti-Heuristic

Expect spherical methods to perform poorly when one or more of the following
hold:

1. The task is regression, especially with `10 < p <= 50`; the observed margins
   are strongly negative and target magnitude does not rescue the method.
2. The task is classification with many predictors (`p > 50`, and especially
   `p > 200`) relative to the number of observations. Spherical splits then
   search centers/radii in a space where distances are noisy and local spheres
   become hard to estimate.
3. Predictors are mostly continuous with scale imbalance or redundancy but no
   clear compact/radial class structure. In those cases, oblique or axis-aligned
   splits often approximate the boundary more directly.
4. Binary classification is strongly target-imbalanced. This benchmark had a
   weak spherical signal for unbalanced binary tasks.
5. Single spherical trees should be avoided more often than spherical forests:
   the failure rates are higher because one bad center/radius choice propagates
   down the whole tree.

## Context Summary

The full context summary is in `spherical_pmlb_heuristic_context_summary.csv`.
The mined rules are in `spherical_pmlb_heuristic_rules.csv`.
"""
    path.write_text(text)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_CSV)
    regime = pd.read_csv(REGIME_CSV)

    metadata_path = RESULTS_DIR / "spherical_pmlb_dataset_descriptors.csv"
    expected_keys = set(
        map(tuple, regime[["task", "dataset"]].drop_duplicates().to_numpy())
    )
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        observed_keys = set(
            map(tuple, metadata[["task", "dataset"]].drop_duplicates().to_numpy())
        )
    else:
        metadata = pd.DataFrame()
        observed_keys = set()

    if expected_keys.issubset(observed_keys):
        print(f"Using cached descriptors from {metadata_path}")
    else:
        metadata = compute_dataset_metadata(regime)
        metadata.to_csv(metadata_path, index=False)

    fold_margins = fold_margin_table(raw, regime)
    fold_margins.to_csv(
        RESULTS_DIR / "spherical_pmlb_heuristic_fold_margins.csv", index=False
    )

    enriched = enrich_regime(regime, metadata, fold_margins)
    enriched.to_csv(RESULTS_DIR / "spherical_pmlb_heuristic_enriched.csv", index=False)

    rules = mine_rules(enriched)
    rules.to_csv(RESULTS_DIR / "spherical_pmlb_heuristic_rules.csv", index=False)

    context_summary = summarize_contexts(enriched)
    context_summary.to_csv(
        RESULTS_DIR / "spherical_pmlb_heuristic_context_summary.csv", index=False
    )

    rule_plot = plot_top_rules(rules)
    poor_rule_plot = plot_poor_rules(rules)
    report = write_report(
        enriched,
        rules,
        context_summary,
        fold_margins,
        rule_plot,
        poor_rule_plot,
    )

    print(f"Wrote {report}")
    print(f"Wrote {RESULTS_DIR / 'spherical_pmlb_heuristic_rules.csv'}")
    if rule_plot is not None:
        print(f"Wrote {rule_plot}")
    if poor_rule_plot is not None:
        print(f"Wrote {poor_rule_plot}")


if __name__ == "__main__":
    main()
