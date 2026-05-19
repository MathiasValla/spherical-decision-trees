"""Build the PRL score/time trade-off figure from benchmark summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = (
    ROOT.parent
    / "benchmarks_nonasv"
    / "results"
    / "spherical_pmlb_target_radial_pruned_sample1000_full_model_summary.csv"
)
FIGURE_PATH = ROOT / "Figure_2_score_time_tradeoff.png"

MODEL_LABELS = {
    "CART": "CART",
    "RandomForest": "Random forest",
    "ObliqueTree": "Oblique tree",
    "ObliqueRandomForest": "Oblique forest",
    "SphericalTree": "Spherical tree",
    "SphericalRandomForest": "Spherical forest",
}
MODEL_COLORS = {
    "CART": "#6f6f6f",
    "RandomForest": "#2f2f2f",
    "ObliqueTree": "#5f7fb4",
    "ObliqueRandomForest": "#224c8d",
    "SphericalTree": "#b65757",
    "SphericalRandomForest": "#8f1d1d",
}
MODEL_MARKERS = {
    "CART": "s",
    "RandomForest": "o",
    "ObliqueTree": "^",
    "ObliqueRandomForest": "o",
    "SphericalTree": "^",
    "SphericalRandomForest": "o",
}
LABEL_OFFSETS = {
    ("classification", "CART"): (6, -15),
    ("classification", "RandomForest"): (7, 10),
    ("classification", "ObliqueRandomForest"): (8, -12),
    ("classification", "ObliqueTree"): (7, -15),
    ("classification", "SphericalTree"): (7, -15),
    ("classification", "SphericalRandomForest"): (-92, 8),
    ("regression", "CART"): (7, 8),
    ("regression", "RandomForest"): (8, 10),
    ("regression", "ObliqueRandomForest"): (8, -14),
    ("regression", "ObliqueTree"): (7, -14),
    ("regression", "SphericalTree"): (7, -14),
    ("regression", "SphericalRandomForest"): (-95, -14),
}


def _annotate(ax, row, dx=6, dy=5):
    ax.annotate(
        MODEL_LABELS[row.model],
        xy=(row.fit_time_mean_s, row.score_mean),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.5,
        ha="left",
        va="bottom",
    )


def main() -> None:
    summary = pd.read_csv(SUMMARY_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharex=False)
    task_titles = {
        "classification": "Classification: balanced accuracy",
        "regression": "Regression: $R^2$",
    }

    for ax, task in zip(axes, ["classification", "regression"]):
        task_df = summary.loc[summary["task"] == task].copy()
        task_df = task_df.sort_values("fit_time_mean_s")
        for row in task_df.itertuples(index=False):
            ax.scatter(
                row.fit_time_mean_s,
                row.score_mean,
                s=115 if "RandomForest" in row.model else 95,
                marker=MODEL_MARKERS[row.model],
                color=MODEL_COLORS[row.model],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            dx, dy = LABEL_OFFSETS[(task, row.model)]
            _annotate(ax, row, dx=dx, dy=dy)

        ax.set_xscale("log")
        ax.grid(True, which="both", axis="both", color="#d9d9d9", linewidth=0.7)
        ax.set_xlabel("Mean fit time per fold (seconds, log scale)")
        ax.set_ylabel("Mean score")
        ax.set_title(task_titles[task], fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylim(0.66, 0.78)
    axes[1].set_ylim(0.20, 0.76)
    fig.suptitle(
        "Benchmark score/time trade-off across complete PMLB datasets",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.01,
        "Spherical forests lead the classification mean, but their current split search is substantially slower.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), w_pad=2.4)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
