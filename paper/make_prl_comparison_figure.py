"""Build a PRL-friendly 2D partition comparison figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from make_axis_vs_spherical_figure import (
    CIRCLE_COLOR,
    FRONTIER_COLOR,
    draw_panel,
    fit_models,
    load_examples,
    make_grid,
    prepare_xy,
    ROOT,
)


FIGURE_PATH = ROOT / "Figure_1_axis_oblique_spherical.png"


def main() -> None:
    examples = load_examples()
    fig, axes = plt.subplots(
        3,
        len(examples),
        figsize=(13.2, 7.4),
        constrained_layout=False,
    )

    row_labels = ["Axis-aligned", "Oblique", "Spherical"]

    for col, (name, X, y) in enumerate(examples):
        X_scaled = prepare_xy(X)
        labels = np.unique(y)
        models = fit_models(X_scaled, y)
        exact_bounds = name == "Toy XOR"
        xlim, ylim, xx, yy, grid = make_grid(X_scaled, exact_bounds=exact_bounds)

        for row, (model, row_label) in enumerate(zip(models, row_labels)):
            draw_panel(
                axes[row, col],
                model,
                X_scaled,
                y,
                xx,
                yy,
                grid,
                labels,
                name if row == 0 else "",
                spherical=row_label == "Spherical",
            )
            axes[row, col].set_xlim(*xlim)
            axes[row, col].set_ylim(*ylim)
            if col == 0:
                axes[row, col].set_ylabel(row_label, fontsize=12, fontweight="bold")

    handles = [
        Patch(facecolor="#999999", alpha=0.20, edgecolor="none", label="terminal regions"),
        Line2D([0], [0], color=FRONTIER_COLOR, linewidth=1.0, label="terminal frontier"),
        Line2D([0], [0], color=CIRCLE_COLOR, linewidth=1.25, label="spherical split circle"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        top=0.92,
        bottom=0.03,
        hspace=0.09,
        wspace=0.05,
    )
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
