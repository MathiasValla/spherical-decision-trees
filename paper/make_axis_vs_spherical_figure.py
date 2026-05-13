"""Build the paper's 2D axis-aligned, oblique, and spherical partition figure."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from pmlb import fetch_data
from sklearn.datasets import load_iris, make_gaussian_quantiles, make_moons
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from treeple.tree import ObliqueDecisionTreeClassifier, SphericalDecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
FIGURE_PATH = ROOT / "figures" / "axis_vs_spherical_2d.png"
RANDOM_STATE = 42
GRID_RESOLUTION = 320
MAX_SAMPLES = 700

CLASS_COLORS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#b279a2",
    "#e45756",
]
FRONTIER_COLOR = "#20242a"
CIRCLE_COLOR = "#7a1e3a"


def _subsample(X: np.ndarray, y: np.ndarray, max_samples: int = MAX_SAMPLES):
    if X.shape[0] <= max_samples:
        return X, y
    stratify = y
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        stratify = None
    _, X_small, _, y_small = train_test_split(
        X,
        y,
        test_size=max_samples,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return X_small, y_small


def load_examples():
    iris = load_iris()
    X_iris = iris.data[:, [2, 3]]
    y_iris = iris.target

    banana = fetch_data("banana")
    X_banana = banana.iloc[:, :2].to_numpy(dtype=float)
    y_banana = banana.iloc[:, -1].to_numpy()
    X_banana, y_banana = _subsample(X_banana, y_banana)

    X_moons, y_moons = make_moons(
        n_samples=160,
        noise=0.13,
        random_state=RANDOM_STATE,
    )
    X_gauss, y_gauss = make_gaussian_quantiles(
        n_samples=180,
        n_features=2,
        n_classes=2,
        random_state=RANDOM_STATE,
    )
    X_xor = np.random.RandomState(0).uniform(low=-1.0, high=1.0, size=(220, 2))
    y_xor = np.logical_xor(X_xor[:, 0] > 0.0, X_xor[:, 1] > 0.0).astype(int)

    return [
        ("Iris petal projection", X_iris, y_iris),
        ("PMLB banana", X_banana, y_banana),
        ("Toy moons", X_moons, y_moons),
        ("Toy Gaussian quantiles", X_gauss, y_gauss),
        ("Toy XOR", X_xor, y_xor),
    ]


def prepare_xy(X: np.ndarray):
    X = SimpleImputer(strategy="median").fit_transform(X)
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def make_grid(X_scaled: np.ndarray, *, exact_bounds: bool = False):
    x_span = X_scaled[:, 0].max() - X_scaled[:, 0].min()
    y_span = X_scaled[:, 1].max() - X_scaled[:, 1].min()
    padding = 0.0 if exact_bounds else max(0.45, 0.08 * max(x_span, y_span))
    xlim = (float(X_scaled[:, 0].min() - padding), float(X_scaled[:, 0].max() + padding))
    ylim = (float(X_scaled[:, 1].min() - padding), float(X_scaled[:, 1].max() + padding))
    xx, yy = np.meshgrid(
        np.linspace(*xlim, GRID_RESOLUTION),
        np.linspace(*ylim, GRID_RESOLUTION),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xlim, ylim, xx, yy, grid


def encode(values: np.ndarray, labels: np.ndarray):
    lookup = {label: i for i, label in enumerate(labels)}
    return np.fromiter((lookup[value] for value in values), dtype=float, count=len(values))


def fit_models(X_scaled: np.ndarray, y: np.ndarray):
    common = {
        "max_depth": 3,
        "min_samples_leaf": 3,
        "random_state": RANDOM_STATE,
    }
    axis_model = DecisionTreeClassifier(**common)
    oblique_model = ObliqueDecisionTreeClassifier(**common)
    sphere_model = SphericalDecisionTreeClassifier(
        **common,
        max_features=None,
        n_center_candidates=500,
        radius_candidates=None,
        center_strategy="target_radial",
    )
    axis_model.fit(X_scaled, y)
    oblique_model.fit(X_scaled, y)
    sphere_model.fit(X_scaled, y)
    return axis_model, oblique_model, sphere_model


def draw_panel(ax, model, X_scaled, y, xx, yy, grid, labels, title, *, spherical=False):
    pred = model.predict(grid)
    regions = encode(pred, labels).reshape(xx.shape)
    point_colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(len(labels))]
    cmap = ListedColormap(point_colors)
    ax.imshow(
        regions,
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        origin="lower",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(labels) - 0.5,
        alpha=0.20,
        interpolation="nearest",
        zorder=0,
    )
    if len(labels) > 1:
        ax.contour(
            xx,
            yy,
            regions,
            levels=np.arange(0.5, len(labels), 1.0),
            colors=FRONTIER_COLOR,
            linewidths=1.0,
            alpha=0.70,
            zorder=2,
        )
    for i, label in enumerate(labels):
        mask = y == label
        ax.scatter(
            X_scaled[mask, 0],
            X_scaled[mask, 1],
            s=14,
            c=point_colors[i],
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )
    if spherical:
        tree = model.tree_
        centers = tree.get_projection_matrix()
        for node_id in range(tree.node_count):
            if tree.children_left[node_id] == -1:
                continue
            circle = Circle(
                centers[node_id, :2],
                float(np.sqrt(tree.threshold[node_id])),
                fill=False,
                edgecolor=CIRCLE_COLOR,
                linewidth=1.25,
                alpha=0.82,
                zorder=3,
            )
            ax.add_patch(circle)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def main() -> None:
    examples = load_examples()
    fig, axes = plt.subplots(
        len(examples),
        3,
        figsize=(10.2, 11.4),
        constrained_layout=False,
    )

    for row, (name, X, y) in enumerate(examples):
        X_scaled = prepare_xy(X)
        labels = np.unique(y)
        axis_model, oblique_model, sphere_model = fit_models(X_scaled, y)
        exact_bounds = name == "Toy XOR"
        xlim, ylim, xx, yy, grid = make_grid(X_scaled, exact_bounds=exact_bounds)
        draw_panel(
            axes[row, 0],
            axis_model,
            X_scaled,
            y,
            xx,
            yy,
            grid,
            labels,
            f"{name}: axis-aligned",
        )
        draw_panel(
            axes[row, 1],
            oblique_model,
            X_scaled,
            y,
            xx,
            yy,
            grid,
            labels,
            f"{name}: oblique",
        )
        draw_panel(
            axes[row, 2],
            sphere_model,
            X_scaled,
            y,
            xx,
            yy,
            grid,
            labels,
            f"{name}: spherical",
            spherical=True,
        )
        for ax in axes[row]:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        axes[row, 0].set_ylabel(name, fontsize=9)

    handles = [
        Patch(facecolor="#999999", alpha=0.20, edgecolor="none", label="terminal regions"),
        Line2D([0], [0], color=FRONTIER_COLOR, linewidth=1.0, label="terminal frontier"),
        Line2D([0], [0], color=CIRCLE_COLOR, linewidth=1.25, label="spherical split circle"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        top=0.955,
        bottom=0.025,
        hspace=0.18,
        wspace=0.08,
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
