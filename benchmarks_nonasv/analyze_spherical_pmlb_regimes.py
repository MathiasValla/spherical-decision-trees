"""Analyze regimes where spherical tree models win on the PMLB benchmark.

This script consumes the outputs of ``bench_spherical_pmlb_full.py`` and writes:

* dataset-level winner/margin tables,
* compact summary tables,
* regime maps over number of rows used and number of predictors.

The plotted x-axis uses ``n_used_samples`` because some large PMLB datasets were
sample-capped during the benchmark. The original ``n_samples`` is retained in
the CSV outputs.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "results" / ".matplotlib"),
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUMMARY_CSV = RESULTS_DIR / "spherical_pmlb_full_summary.csv"
RAW_CSV = RESULTS_DIR / "spherical_pmlb_full.csv"

COMPARISONS = {
    "overall": [
        "CART",
        "RandomForest",
        "SphericalTree",
        "SphericalRandomForest",
        "ObliqueTree",
        "ObliqueRandomForest",
    ],
    "trees": ["CART", "SphericalTree", "ObliqueTree"],
    "forests": ["RandomForest", "SphericalRandomForest", "ObliqueRandomForest"],
}

MODEL_LABELS = {
    "CART": "CART",
    "RandomForest": "Random forest",
    "SphericalTree": "Spherical tree",
    "SphericalRandomForest": "Spherical forest",
    "ObliqueTree": "Oblique tree",
    "ObliqueRandomForest": "Oblique forest",
}

MODEL_COLORS = {
    "CART": "#4c78a8",
    "RandomForest": "#54a24b",
    "SphericalTree": "#f58518",
    "SphericalRandomForest": "#e45756",
    "ObliqueTree": "#72b7b2",
    "ObliqueRandomForest": "#b279a2",
}

TASK_LABELS = {
    "classification": "Classification",
    "regression": "Regression",
}

ROW_BINS = [0, 250, 1000, 5000, np.inf]
ROW_LABELS = ["<=250", "251-1k", "1k-5k", ">5k"]
FEATURE_BINS = [0, 10, 50, 200, np.inf]
FEATURE_LABELS = ["<=10", "11-50", "51-200", ">200"]


def _is_spherical(model: str) -> bool:
    return model.startswith("Spherical")


def _first_or_none(values: pd.Series) -> object:
    non_missing = values.dropna()
    if non_missing.empty:
        return np.nan
    return non_missing.iloc[0]


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_CSV)
    raw = pd.read_csv(RAW_CSV)

    meta = (
        raw.groupby(["task", "dataset"], as_index=False)
        .agg(
            n_samples=("n_samples", "first"),
            n_used_samples=("n_used_samples", "first"),
            n_features=("n_features", "first"),
        )
        .assign(
            capped=lambda df: df["n_used_samples"].astype(float)
            < df["n_samples"].astype(float)
        )
    )
    return summary, meta


def make_regime_table(summary: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison, models in COMPARISONS.items():
        sub = summary[summary["model"].isin(models)].copy()
        counts = sub.groupby(["task", "dataset"])["model"].nunique()
        complete_index = counts[counts == len(models)].index
        sub = (
            sub.set_index(["task", "dataset"])
            .loc[complete_index]
            .reset_index()
            .sort_values(
                ["task", "dataset", "score_mean", "total_time_mean_s"],
                ascending=[True, True, False, True],
            )
        )

        pivot_score = sub.pivot(
            index=["task", "dataset"], columns="model", values="score_mean"
        )
        pivot_fit = sub.pivot(
            index=["task", "dataset"], columns="model", values="fit_time_mean_s"
        )
        pivot_total = sub.pivot(
            index=["task", "dataset"], columns="model", values="total_time_mean_s"
        )

        winners = sub.groupby(["task", "dataset"], as_index=False).first()
        spherical_models = [model for model in models if _is_spherical(model)]
        non_spherical_models = [model for model in models if not _is_spherical(model)]

        best_spherical_score = pivot_score[spherical_models].max(axis=1)
        best_non_spherical_score = pivot_score[non_spherical_models].max(axis=1)
        best_spherical_model = pivot_score[spherical_models].idxmax(axis=1)
        best_non_spherical_model = pivot_score[non_spherical_models].idxmax(axis=1)

        comparison_table = winners[
            [
                "task",
                "dataset",
                "model",
                "score_mean",
                "fit_time_mean_s",
                "predict_time_mean_s",
                "total_time_mean_s",
            ]
        ].rename(
            columns={
                "model": "winner",
                "score_mean": "winner_score_mean",
                "fit_time_mean_s": "winner_fit_time_mean_s",
                "predict_time_mean_s": "winner_predict_time_mean_s",
                "total_time_mean_s": "winner_total_time_mean_s",
            }
        )
        comparison_table["comparison"] = comparison
        comparison_table = comparison_table.merge(
            best_spherical_score.rename("best_spherical_score_mean"),
            on=["task", "dataset"],
        )
        comparison_table = comparison_table.merge(
            best_non_spherical_score.rename("best_non_spherical_score_mean"),
            on=["task", "dataset"],
        )
        comparison_table = comparison_table.merge(
            best_spherical_model.rename("best_spherical_model"),
            on=["task", "dataset"],
        )
        comparison_table = comparison_table.merge(
            best_non_spherical_model.rename("best_non_spherical_model"),
            on=["task", "dataset"],
        )
        comparison_table["spherical_margin"] = (
            comparison_table["best_spherical_score_mean"]
            - comparison_table["best_non_spherical_score_mean"]
        )
        comparison_table["spherical_wins"] = comparison_table["spherical_margin"] > 0
        comparison_table["winner_is_spherical"] = comparison_table["winner"].map(
            _is_spherical
        )

        for model in models:
            comparison_table[f"{model}_score_mean"] = comparison_table.set_index(
                ["task", "dataset"]
            ).index.map(pivot_score[model])
            comparison_table[f"{model}_fit_time_mean_s"] = comparison_table.set_index(
                ["task", "dataset"]
            ).index.map(pivot_fit[model])
            comparison_table[f"{model}_total_time_mean_s"] = comparison_table.set_index(
                ["task", "dataset"]
            ).index.map(pivot_total[model])

        rows.append(comparison_table)

    regime = pd.concat(rows, ignore_index=True)
    regime = regime.merge(meta, on=["task", "dataset"], how="left")
    regime["row_band"] = pd.cut(
        regime["n_used_samples"], bins=ROW_BINS, labels=ROW_LABELS
    )
    regime["feature_band"] = pd.cut(
        regime["n_features"], bins=FEATURE_BINS, labels=FEATURE_LABELS
    )
    return regime.sort_values(["comparison", "task", "dataset"]).reset_index(drop=True)


def make_summary_tables(regime: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    winner_counts = (
        regime.groupby(["comparison", "task", "winner"], observed=True)
        .size()
        .rename("n_wins")
        .reset_index()
        .sort_values(["comparison", "task", "n_wins"], ascending=[True, True, False])
    )

    spherical_summary = (
        regime.groupby(["comparison", "task"], observed=True)
        .agg(
            n_datasets=("dataset", "nunique"),
            n_spherical_wins=("spherical_wins", "sum"),
            spherical_win_rate=("spherical_wins", "mean"),
            median_spherical_margin=("spherical_margin", "median"),
            mean_spherical_margin=("spherical_margin", "mean"),
            median_rows_all=("n_used_samples", "median"),
            median_features_all=("n_features", "median"),
        )
        .reset_index()
    )

    positive = (
        regime[regime["spherical_wins"]]
        .groupby(["comparison", "task"], observed=True)
        .agg(
            median_rows_when_spherical_wins=("n_used_samples", "median"),
            median_features_when_spherical_wins=("n_features", "median"),
            median_margin_when_spherical_wins=("spherical_margin", "median"),
        )
        .reset_index()
    )
    spherical_summary = spherical_summary.merge(
        positive, on=["comparison", "task"], how="left"
    )

    return winner_counts, spherical_summary


def make_grid_counts(regime: pd.DataFrame) -> pd.DataFrame:
    return (
        regime.groupby(
            ["comparison", "task", "row_band", "feature_band", "winner"],
            observed=True,
        )
        .size()
        .rename("n_wins")
        .reset_index()
        .sort_values(
            ["comparison", "task", "row_band", "feature_band", "n_wins"],
            ascending=[True, True, True, True, False],
        )
    )


def _log_edges(values: pd.Series, n_bins: int = 5) -> np.ndarray:
    positive = values[values > 0].astype(float)
    low = np.floor(np.log10(positive.min()))
    high = np.ceil(np.log10(positive.max()))
    if high <= low:
        high = low + 1
    return np.geomspace(10**low, 10**high, n_bins + 1)


def _draw_modal_background(ax: plt.Axes, frame: pd.DataFrame) -> None:
    x_edges = _log_edges(frame["n_used_samples"], n_bins=5)
    y_edges = _log_edges(frame["n_features"], n_bins=5)
    binned = frame.copy()
    binned["x_bin"] = pd.cut(binned["n_used_samples"], x_edges, include_lowest=True)
    binned["y_bin"] = pd.cut(binned["n_features"], y_edges, include_lowest=True)

    for (x_bin, y_bin), cell in binned.groupby(["x_bin", "y_bin"], observed=True):
        if cell.empty:
            continue
        winner = cell["winner"].mode().iloc[0]
        rect = Rectangle(
            (x_bin.left, y_bin.left),
            x_bin.right - x_bin.left,
            y_bin.right - y_bin.left,
            facecolor=MODEL_COLORS[winner],
            edgecolor="none",
            alpha=0.11,
            zorder=0,
        )
        ax.add_patch(rect)


def plot_regime_map(regime: pd.DataFrame, comparison: str) -> Path:
    frame = regime[regime["comparison"] == comparison].copy()
    models = COMPARISONS[comparison]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharex=True, sharey=True)

    for ax, task in zip(axes, ["classification", "regression"]):
        task_frame = frame[frame["task"] == task]
        _draw_modal_background(ax, task_frame)

        for model in models:
            subset = task_frame[task_frame["winner"] == model]
            if subset.empty:
                continue
            ax.scatter(
                subset["n_used_samples"],
                subset["n_features"],
                s=42,
                c=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                edgecolors=np.where(subset["capped"], "#111111", "#ffffff"),
                linewidths=np.where(subset["capped"], 0.9, 0.4),
                alpha=0.92,
                zorder=2,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", color="#d9dde3", linewidth=0.6, alpha=0.75)
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Rows used in benchmark")
        ax.set_ylabel("Predictors")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="#ffffff",
            markersize=8,
            label=MODEL_LABELS[model],
        )
        for model in models
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ffffff",
            markeredgecolor="#111111",
            markersize=8,
            label="Sample-capped dataset",
        )
    )
    fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.985, 0.5),
        ncol=1,
        frameon=False,
    )
    fig.suptitle(
        f"PMLB regime map: winning model ({comparison})",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.44,
        0.055,
        "Pale rectangles show the modal winner in log-spaced bins; points show individual datasets.",
        ha="center",
        fontsize=10,
        color="#4a4f58",
    )
    fig.subplots_adjust(left=0.07, right=0.82, bottom=0.16, top=0.84, wspace=0.08)

    path = FIGURES_DIR / f"spherical_regime_{comparison}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_margin_map(regime: pd.DataFrame, comparison: str) -> Path:
    frame = regime[regime["comparison"] == comparison].copy()
    limit = np.nanpercentile(np.abs(frame["spherical_margin"]), 95)
    if not np.isfinite(limit) or limit <= 0:
        limit = max(np.nanmax(np.abs(frame["spherical_margin"])), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), sharex=True, sharey=True)
    scatter = None
    for ax, task in zip(axes, ["classification", "regression"]):
        task_frame = frame[frame["task"] == task]
        scatter = ax.scatter(
            task_frame["n_used_samples"],
            task_frame["n_features"],
            c=task_frame["spherical_margin"],
            s=np.where(task_frame["spherical_wins"], 58, 36),
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            edgecolors=np.where(task_frame["spherical_wins"], "#111111", "#ffffff"),
            linewidths=np.where(task_frame["spherical_wins"], 0.85, 0.35),
            alpha=0.9,
        )
        ax.axhline(10, color="#848991", linewidth=0.7, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", color="#d9dde3", linewidth=0.6, alpha=0.75)
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Rows used in benchmark")
        ax.set_ylabel("Predictors")

    assert scatter is not None
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.16, top=0.84, wspace=0.10)
    cax = fig.add_axes([0.885, 0.22, 0.018, 0.58])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("Best spherical score minus best non-spherical score")
    fig.suptitle(
        f"PMLB spherical advantage map ({comparison})",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.045,
        "Positive values mean a spherical model beats every non-spherical model in the comparison set.",
        ha="center",
        fontsize=10,
        color="#4a4f58",
    )

    path = FIGURES_DIR / f"spherical_margin_{comparison}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_report(
    regime: pd.DataFrame,
    winner_counts: pd.DataFrame,
    spherical_summary: pd.DataFrame,
    figures: list[Path],
) -> Path:
    report_path = RESULTS_DIR / "spherical_pmlb_regime_analysis.md"

    count_table = winner_counts.copy()
    count_table["winner"] = count_table["winner"].map(MODEL_LABELS)

    compact_summary = spherical_summary.copy()
    compact_summary["spherical_win_rate"] = (
        100 * compact_summary["spherical_win_rate"]
    ).round(1)
    numeric_cols = [
        "median_spherical_margin",
        "mean_spherical_margin",
        "median_margin_when_spherical_wins",
    ]
    for col in numeric_cols:
        compact_summary[col] = compact_summary[col].round(4)
    for col in [
        "median_rows_all",
        "median_features_all",
        "median_rows_when_spherical_wins",
        "median_features_when_spherical_wins",
    ]:
        compact_summary[col] = compact_summary[col].round(1)

    spherical_wins = regime[regime["spherical_wins"]].copy()
    top_wins = (
        spherical_wins[spherical_wins["comparison"].isin(["trees", "forests"])]
        .sort_values("spherical_margin", ascending=False)
        .head(15)
    )
    top_wins = top_wins[
        [
            "comparison",
            "task",
            "dataset",
            "best_spherical_model",
            "best_non_spherical_model",
            "spherical_margin",
            "n_used_samples",
            "n_features",
        ]
    ].copy()
    top_wins["best_spherical_model"] = top_wins["best_spherical_model"].map(
        MODEL_LABELS
    )
    top_wins["best_non_spherical_model"] = top_wins["best_non_spherical_model"].map(
        MODEL_LABELS
    )
    top_wins["spherical_margin"] = top_wins["spherical_margin"].round(4)

    def summary_value(comparison: str, task: str, column: str) -> float:
        row = spherical_summary[
            (spherical_summary["comparison"] == comparison)
            & (spherical_summary["task"] == task)
        ].iloc[0]
        return float(row[column])

    def model_wins(comparison: str, task: str, winner: str) -> int:
        row = winner_counts[
            (winner_counts["comparison"] == comparison)
            & (winner_counts["task"] == task)
            & (winner_counts["winner"] == winner)
        ]
        if row.empty:
            return 0
        return int(row["n_wins"].iloc[0])

    forest_class_spherical_wins = model_wins(
        "forests", "classification", "SphericalRandomForest"
    )
    forest_class_rf_wins = model_wins("forests", "classification", "RandomForest")
    forest_class_oblique_wins = model_wins(
        "forests", "classification", "ObliqueRandomForest"
    )
    tree_class_spherical_wins = model_wins("trees", "classification", "SphericalTree")
    overall_class_rate = 100 * summary_value(
        "overall", "classification", "spherical_win_rate"
    )
    overall_reg_rate = 100 * summary_value(
        "overall", "regression", "spherical_win_rate"
    )
    tree_class_rate = 100 * summary_value(
        "trees", "classification", "spherical_win_rate"
    )
    forest_class_rate = 100 * summary_value(
        "forests", "classification", "spherical_win_rate"
    )
    tree_class_rows = summary_value(
        "trees", "classification", "median_rows_when_spherical_wins"
    )
    tree_class_features = summary_value(
        "trees", "classification", "median_features_when_spherical_wins"
    )
    forest_class_rows = summary_value(
        "forests", "classification", "median_rows_when_spherical_wins"
    )
    forest_class_features = summary_value(
        "forests", "classification", "median_features_when_spherical_wins"
    )

    figure_lines = "\n".join(
        f"- `{path.relative_to(RESULTS_DIR)}`" for path in figures
    )

    text = f"""# Spherical PMLB Regime Analysis

This analysis uses `spherical_pmlb_full.csv` and `spherical_pmlb_full_summary.csv`.
Classification winners are selected by mean balanced accuracy; regression winners
are selected by mean R2. The regime maps use `n_used_samples`, not always the
original PMLB row count, because several large datasets were capped during the
benchmark.

## Outputs

{figure_lines}

Additional tables:

- `spherical_pmlb_regime_table.csv`: dataset-level winners, margins, metadata, and model scores.
- `spherical_pmlb_regime_winner_counts.csv`: winner counts by comparison and task.
- `spherical_pmlb_regime_spherical_summary.csv`: spherical win rates and margin summaries.
- `spherical_pmlb_regime_grid_counts.csv`: winner counts in coarse row/predictor bands.
- `spherical_pmlb_spherical_wins.csv`: all positive spherical-margin datasets.

## Empirical Takeaways

- The current spherical signal is mainly a classification signal: spherical
  models are positive-margin overall winners on {overall_class_rate:.1f}% of
  classification datasets versus {overall_reg_rate:.1f}% of regression datasets.
- Ensembling helps substantially. Among forests on classification tasks,
  spherical random forests win {forest_class_spherical_wins} datasets, compared
  with {forest_class_rf_wins} for classical random forests and
  {forest_class_oblique_wins} for oblique random forests. Positive-margin
  spherical forest wins have median size {forest_class_rows:.0f} rows and
  {forest_class_features:.0f} predictors.
- Single spherical trees win {tree_class_spherical_wins} classification
  tree-only comparisons ({tree_class_rate:.1f}%). These wins are mostly in
  small-to-medium, low-dimensional problems: the median positive-margin spherical
  tree win has {tree_class_rows:.0f} rows and {tree_class_features:.0f}
  predictors.
- There is no clean monotone frontier in rows and predictors alone. The regime
  maps show patchy wins, which suggests that the geometry of the response
  boundary is the missing explanatory variable. The row/predictor map is useful
  as a first diagnostic, but the next analysis should add shape diagnostics such
  as class overlap, radial separability, interaction strength, and manifold or
  cluster structure.

## Winner Counts

{_markdown_table(count_table)}

## Spherical Advantage Summary

Here, a spherical win means the best spherical model in the comparison set beats
the best non-spherical model in the same set.

{_markdown_table(compact_summary)}

## Largest Positive Spherical Margins

{_markdown_table(top_wins)}

## Interpretation Notes

- `overall` asks whether any spherical model beats CART, random forest, oblique
  tree, and oblique forest on the same dataset.
- `trees` isolates single-tree behavior: spherical tree versus CART and oblique
  tree.
- `forests` isolates ensemble behavior: spherical random forest versus random
  forest and oblique random forest.
- The pale background bins in the regime maps show the modal winner in each
  log-spaced row/predictor cell; the points are the actual dataset winners.
"""
    report_path.write_text(text)
    return report_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    summary, meta = load_results()
    regime = make_regime_table(summary, meta)
    winner_counts, spherical_summary = make_summary_tables(regime)
    grid_counts = make_grid_counts(regime)
    spherical_wins = regime[regime["spherical_wins"]].copy()

    regime.to_csv(RESULTS_DIR / "spherical_pmlb_regime_table.csv", index=False)
    winner_counts.to_csv(
        RESULTS_DIR / "spherical_pmlb_regime_winner_counts.csv", index=False
    )
    spherical_summary.to_csv(
        RESULTS_DIR / "spherical_pmlb_regime_spherical_summary.csv", index=False
    )
    grid_counts.to_csv(
        RESULTS_DIR / "spherical_pmlb_regime_grid_counts.csv", index=False
    )
    spherical_wins.to_csv(RESULTS_DIR / "spherical_pmlb_spherical_wins.csv", index=False)

    figures = []
    for comparison in COMPARISONS:
        figures.append(plot_regime_map(regime, comparison))
    for comparison in ["trees", "forests"]:
        figures.append(plot_margin_map(regime, comparison))

    report_path = write_report(regime, winner_counts, spherical_summary, figures)

    print(f"Wrote {RESULTS_DIR / 'spherical_pmlb_regime_table.csv'}")
    print(f"Wrote {report_path}")
    for path in figures:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
