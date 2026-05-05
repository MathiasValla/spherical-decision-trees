"""Draw a simple 2D spherical decision tree partition on Iris.

The plot uses Iris petal length and petal width, standardized before fitting.
Spherical trees are distance-based, so the displayed covariate space is the
standardized covariate space used by the model.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "results" / ".matplotlib"),
)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from treeple.tree import SphericalDecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 7


def node_depths(children_left: np.ndarray, children_right: np.ndarray) -> np.ndarray:
    depths = np.zeros(children_left.shape[0], dtype=int)
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        left = children_left[node_id]
        right = children_right[node_id]
        if left != -1:
            stack.append((int(left), depth + 1))
            stack.append((int(right), depth + 1))
    return depths


def extract_splits(model: SphericalDecisionTreeClassifier, scaler: StandardScaler) -> pd.DataFrame:
    tree = model.tree_
    centers = tree.get_projection_matrix()
    depths = node_depths(tree.children_left, tree.children_right)
    rows = []
    for node_id in range(tree.node_count):
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        if left == -1:
            continue

        center_std = centers[node_id, :2].astype(float)
        center_original = scaler.inverse_transform(center_std.reshape(1, -1))[0]
        rows.append(
            {
                "node": node_id,
                "depth": int(depths[node_id]),
                "left_child_inside": left,
                "right_child_outside": right,
                "n_node_samples": int(tree.n_node_samples[node_id]),
                "impurity": float(tree.impurity[node_id]),
                "center_petal_length_standardized": center_std[0],
                "center_petal_width_standardized": center_std[1],
                "center_petal_length_original_cm": center_original[0],
                "center_petal_width_original_cm": center_original[1],
                "radius_standardized": float(np.sqrt(tree.threshold[node_id])),
                "radius_squared_standardized": float(tree.threshold[node_id]),
                "inside_rule": (
                    "(x_petal_length - center_petal_length)^2 + "
                    "(x_petal_width - center_petal_width)^2 <= radius^2"
                ),
            }
        )
    return pd.DataFrame(rows)


def make_grid(X: np.ndarray, padding: float = 0.55, resolution: int = 450):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def draw_circle(ax, center, radius, *, color, linestyle, label):
    circle = Circle(
        center,
        radius,
        fill=False,
        edgecolor=color,
        linestyle=linestyle,
        linewidth=2.2,
        alpha=0.95,
    )
    ax.add_patch(circle)
    ax.scatter([center[0]], [center[1]], marker="x", color=color, s=80, linewidths=2)
    ax.text(
        center[0],
        center[1] + radius + 0.08,
        label,
        color=color,
        ha="center",
        va="bottom",
        fontsize=9,
        weight="bold",
    )


def plot_partition() -> tuple[Path, Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    feature_indices = [2, 3]
    X_raw = iris.data[:, feature_indices]
    y = iris.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = SphericalDecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=5,
        max_features=None,
        n_center_candidates=64,
        radius_candidates=128,
        center_strategy="target",
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)

    splits = extract_splits(model, scaler)
    splits_path = RESULTS_DIR / "spherical_iris_splits.csv"
    splits.to_csv(splits_path, index=False)

    xx, yy, grid = make_grid(X)
    tree = model.tree_
    centers = tree.get_projection_matrix()
    root_center = centers[0, :2].astype(float)
    root_radius = float(np.sqrt(tree.threshold[0]))
    root_inside = (
        (grid[:, 0] - root_center[0]) ** 2
        + (grid[:, 1] - root_center[1]) ** 2
        <= root_radius**2
    ).reshape(xx.shape)
    full_pred = model.predict(grid).reshape(xx.shape)

    point_colors = ["#4c78a8", "#f58518", "#54a24b"]
    class_cmap = ListedColormap(["#dbe8f6", "#fde7c9", "#ddedda"])
    root_cmap = ListedColormap(["#eef2f6", "#f6ddc8"])
    circle_colors = ["#111111", "#9d3c5f", "#5f4690", "#1b9e77"]
    circle_styles = ["-", "--", ":", "-."]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=True, sharey=True)

    axes[0].contourf(xx, yy, root_inside.astype(int), levels=[-0.5, 0.5, 1.5], cmap=root_cmap, alpha=0.88)
    axes[0].contour(xx, yy, root_inside.astype(int), levels=[0.5], colors=["#111111"], linewidths=1.0)
    draw_circle(
        axes[0],
        root_center,
        root_radius,
        color="#111111",
        linestyle="-",
        label=f"root sphere, r={root_radius:.2f}",
    )
    axes[0].set_title("Root binary split: inside vs outside sphere")

    axes[1].contourf(
        xx,
        yy,
        full_pred,
        levels=np.arange(len(iris.target_names) + 1) - 0.5,
        cmap=class_cmap,
        alpha=0.88,
    )
    axes[1].contour(xx, yy, full_pred, levels=[0.5, 1.5], colors=["#5f6368"], linewidths=0.9)
    depths = node_depths(tree.children_left, tree.children_right)
    for node_id in range(tree.node_count):
        if tree.children_left[node_id] == -1:
            continue
        depth = int(depths[node_id])
        center = centers[node_id, :2].astype(float)
        radius = float(np.sqrt(tree.threshold[node_id]))
        draw_circle(
            axes[1],
            center,
            radius,
            color=circle_colors[depth % len(circle_colors)],
            linestyle=circle_styles[depth % len(circle_styles)],
            label=f"node {node_id}, r={radius:.2f}",
        )
    axes[1].set_title("Depth-2 spherical tree decision regions")

    for ax in axes:
        for class_id, color in enumerate(point_colors):
            mask = y == class_id
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                c=color,
                s=34,
                edgecolor="white",
                linewidth=0.6,
                label=iris.target_names[class_id],
                zorder=4,
            )
        ax.set_xlabel("standardized petal length")
        ax.set_ylabel("standardized petal width")
        ax.grid(color="#d9dde3", linewidth=0.55, alpha=0.75)
        ax.set_aspect("equal", adjustable="box")

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=name,
        )
        for color, name in zip(point_colors, iris.target_names)
    ]
    circle_handles = [
        Line2D(
            [0],
            [0],
            color="#111111",
            linewidth=2,
            label="sphere boundary",
        )
    ]
    fig.legend(
        handles=class_handles + circle_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Spherical tree splits on Iris petal covariates", fontsize=15, y=0.98)
    fig.text(
        0.5,
        0.06,
        "At each internal node, samples inside the circle go to the left child; samples outside go to the right child.",
        ha="center",
        fontsize=10,
        color="#4a4f58",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.88, wspace=0.12)

    figure_path = FIGURES_DIR / "spherical_iris_partition.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    report_path = RESULTS_DIR / "spherical_iris_partition.md"
    split_table = splits[
        [
            "node",
            "depth",
            "left_child_inside",
            "right_child_outside",
            "n_node_samples",
            "center_petal_length_standardized",
            "center_petal_width_standardized",
            "radius_standardized",
        ]
    ].copy()
    for col in [
        "center_petal_length_standardized",
        "center_petal_width_standardized",
        "radius_standardized",
    ]:
        split_table[col] = split_table[col].round(3)

    report_path.write_text(
        "\n".join(
            [
                "# Iris Spherical Partition",
                "",
                "This example fits a depth-2 spherical decision tree on Iris petal",
                "length and petal width after standardization. The displayed",
                "circles are the actual split rules in the standardized two-dimensional",
                "covariate space.",
                "",
                "At each internal node, the left child receives observations satisfying",
                "`(x_1 - c_1)^2 + (x_2 - c_2)^2 <= r^2`; the right child receives the",
                "outside of that sphere.",
                "",
                "## Split Table",
                "",
                split_table.to_markdown(index=False),
                "",
                "## Outputs",
                "",
                "- `figures/spherical_iris_partition.png`",
                "- `spherical_iris_splits.csv`",
                "",
            ]
        )
    )
    return figure_path, splits_path, report_path


def main() -> None:
    figure_path, splits_path, report_path = plot_partition()
    print(f"Wrote {figure_path}")
    print(f"Wrote {splits_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
