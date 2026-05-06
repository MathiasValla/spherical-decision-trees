"""Generate 2D spherical split frontier plots for eligible PMLB datasets.

Eligibility is intentionally strict: the benchmark dataset must have exactly two
predictors and both must look continuous under the descriptor heuristic used in
``analyze_spherical_heuristics.py``. The fitted model uses the same spherical
tree hyperparameters as the full PMLB benchmark.

The figures keep the spherical split frontiers explicit as circles. The
all-frontiers panel also shades the terminal regions induced by the fitted tree.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "results" / ".matplotlib"),
)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from treeple.tree import SphericalDecisionTreeClassifier, SphericalDecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "spherical_2d_partitions"
SPLITS_DIR = RESULTS_DIR / "spherical_2d_partition_splits"

REGIME_CSV = RESULTS_DIR / "spherical_pmlb_regime_table.csv"
DESCRIPTORS_CSV = RESULTS_DIR / "spherical_pmlb_dataset_descriptors.csv"

RANDOM_STATE = 42
N_CENTER_CANDIDATES = 500
RADIUS_CANDIDATES = None
CENTER_STRATEGY = "target_radial"
MAX_CIRCLES = 1000
MAX_LABELED_CIRCLES = 8
GRID_RESOLUTION = 420

CLASS_PALETTE = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#b279a2",
    "#e45756",
    "#72b7b2",
    "#ff9da6",
    "#9d755d",
]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def benchmark_candidates() -> pd.DataFrame:
    regime = pd.read_csv(REGIME_CSV)
    descriptors = pd.read_csv(DESCRIPTORS_CSV)
    keys = regime[["task", "dataset", "n_used_samples", "n_features"]].drop_duplicates()
    candidates = keys.merge(
        descriptors[
            [
                "task",
                "dataset",
                "discrete_predictor_fraction",
                "predictor_profile",
            ]
        ],
        on=["task", "dataset"],
        how="left",
    )
    return candidates[
        (candidates["n_features"] == 2)
        & (candidates["discrete_predictor_fraction"] == 0.0)
    ].sort_values(["task", "dataset"])


def fetch_xy(dataset: str, task: str, n_used_samples: int):
    frame = fetch_data(dataset)
    feature_names = [str(col) for col in frame.columns[:-1]]
    X = frame.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = frame.iloc[:, -1].to_numpy()
    if task == "regression":
        y = y.astype(np.float64)

    if n_used_samples < X.shape[0]:
        stratify = None
        if task == "classification":
            _, counts = np.unique(y, return_counts=True)
            if counts.min() >= 2:
                stratify = y
        _, X, _, y = train_test_split(
            X,
            y,
            test_size=int(n_used_samples),
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    return X, y, feature_names


def fit_spherical_tree(X: np.ndarray, y: np.ndarray, task: str):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if task == "classification":
        model = SphericalDecisionTreeClassifier(
            max_features=None,
            n_center_candidates=N_CENTER_CANDIDATES,
            radius_candidates=RADIUS_CANDIDATES,
            center_strategy=CENTER_STRATEGY,
            random_state=RANDOM_STATE,
        )
    else:
        model = SphericalDecisionTreeRegressor(
            max_features=None,
            n_center_candidates=N_CENTER_CANDIDATES,
            radius_candidates=RADIUS_CANDIDATES,
            center_strategy=CENTER_STRATEGY,
            random_state=RANDOM_STATE,
        )
    model.fit(X_scaled, y)
    return model, scaler, X_scaled


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


def extract_splits(model, scaler: StandardScaler, feature_names: list[str]) -> pd.DataFrame:
    tree = model.tree_
    centers = tree.get_projection_matrix()
    depths = node_depths(tree.children_left, tree.children_right)
    rows = []
    for node_id in range(tree.node_count):
        if tree.children_left[node_id] == -1:
            continue
        center_scaled = centers[node_id, :2].astype(float)
        center_original = scaler.inverse_transform(center_scaled.reshape(1, -1))[0]
        rows.append(
            {
                "node": int(node_id),
                "depth": int(depths[node_id]),
                "left_child_inside": int(tree.children_left[node_id]),
                "right_child_outside": int(tree.children_right[node_id]),
                "n_node_samples": int(tree.n_node_samples[node_id]),
                "impurity": float(tree.impurity[node_id]),
                f"center_{feature_names[0]}_standardized": center_scaled[0],
                f"center_{feature_names[1]}_standardized": center_scaled[1],
                f"center_{feature_names[0]}_original": center_original[0],
                f"center_{feature_names[1]}_original": center_original[1],
                "radius_standardized": float(np.sqrt(tree.threshold[node_id])),
                "radius_squared_standardized": float(tree.threshold[node_id]),
            }
        )
    return pd.DataFrame(rows)


def axis_limits(X_scaled: np.ndarray):
    xs = [X_scaled[:, 0]]
    ys = [X_scaled[:, 1]]
    x_values = np.concatenate(xs)
    y_values = np.concatenate(ys)
    x_span = x_values.max() - x_values.min()
    y_span = y_values.max() - y_values.min()
    padding = max(0.35, 0.06 * max(x_span, y_span))
    return (
        (float(x_values.min() - padding), float(x_values.max() + padding)),
        (float(y_values.min() - padding), float(y_values.max() + padding)),
    )


def make_grid(xlim: tuple[float, float], ylim: tuple[float, float]):
    xs = np.linspace(xlim[0], xlim[1], GRID_RESOLUTION)
    ys = np.linspace(ylim[0], ylim[1], GRID_RESOLUTION)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, points


def encode_labels(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    lookup = {label: i for i, label in enumerate(labels)}
    return np.fromiter((lookup[value] for value in values), dtype=float, count=values.size)


def draw_root_region_fill(ax, xx: np.ndarray, yy: np.ndarray, centers: np.ndarray, tree, xlim, ylim):
    center = centers[0, :2]
    radius_sq = float(tree.threshold[0])
    inside = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius_sq
    root_regions = np.where(inside, 0.0, 1.0)
    cmap = ListedColormap(["#9ecae1", "#fdd0a2"])
    ax.imshow(
        root_regions,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        cmap=cmap,
        alpha=0.18,
        interpolation="nearest",
        zorder=0,
    )


def draw_terminal_region_fill(
    ax,
    model,
    grid_points: np.ndarray,
    grid_shape: tuple[int, int],
    task: str,
    y: np.ndarray,
    xlim,
    ylim,
    *,
    class_labels: np.ndarray | None = None,
    regression_norm: Normalize | None = None,
):
    predictions = model.predict(grid_points)
    if task == "classification":
        if class_labels is None:
            class_labels = np.unique(np.concatenate([np.asarray(y), predictions]))
        encoded = encode_labels(predictions, class_labels).reshape(grid_shape)
        colors = [CLASS_PALETTE[i % len(CLASS_PALETTE)] for i in range(len(class_labels))]
        cmap = ListedColormap(colors)
        ax.imshow(
            encoded,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            cmap=cmap,
            vmin=-0.5,
            vmax=len(class_labels) - 0.5,
            alpha=0.17,
            interpolation="nearest",
            zorder=0,
        )
        return None

    prediction_grid = predictions.reshape(grid_shape)
    return ax.imshow(
        prediction_grid,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        cmap="viridis",
        norm=regression_norm,
        alpha=0.22,
        interpolation="nearest",
        zorder=0,
    )


def draw_split_circles(
    ax,
    splits: pd.DataFrame,
    centers: np.ndarray,
    *,
    label_prefix: str = "node",
    all_frontiers: bool = False,
):
    if splits.empty:
        return 0
    draw = splits.sort_values(["depth", "node"]).head(MAX_CIRCLES)
    colors = ["#111111", "#9d3c5f", "#5f4690", "#1b9e77", "#c44e52", "#4c78a8"]
    styles = ["-", "--", ":", "-."]
    thin_alpha = 0.18 if draw.shape[0] <= 200 else 0.08
    thin_width = 0.75 if draw.shape[0] <= 200 else 0.45
    for rank, row in enumerate(draw.itertuples(index=False)):
        color = colors[int(row.depth) % len(colors)]
        emphasize = rank < MAX_LABELED_CIRCLES or not all_frontiers
        circle = Circle(
            centers[int(row.node), :2],
            float(row.radius_standardized),
            fill=False,
            edgecolor=color,
            linestyle=styles[int(row.depth) % len(styles)],
            linewidth=2.2 if emphasize else thin_width,
            alpha=0.94 if emphasize else thin_alpha,
            zorder=5 if emphasize else 3,
        )
        ax.add_patch(circle)
        if emphasize:
            center = centers[int(row.node), :2]
            ax.scatter(
                [center[0]],
                [center[1]],
                marker="x",
                color=color,
                s=60,
                linewidths=1.6,
                zorder=6,
            )
            ax.text(
                center[0],
                center[1] + float(row.radius_standardized) + 0.06,
                f"{label_prefix} {int(row.node)}, r={float(row.radius_standardized):.2f}",
                color=color,
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
                clip_on=True,
            )
    return draw.shape[0]


def plot_dataset(row: pd.Series) -> dict[str, object]:
    task = str(row.task)
    dataset = str(row.dataset)
    X, y, feature_names = fetch_xy(dataset, task, int(row.n_used_samples))
    model, scaler, X_scaled = fit_spherical_tree(X, y, task)
    tree = model.tree_
    centers = tree.get_projection_matrix()
    splits = extract_splits(model, scaler, feature_names)

    slug = f"{task}_{slugify(dataset)}"
    split_path = SPLITS_DIR / f"{slug}_splits.csv"
    splits.to_csv(split_path, index=False)

    n_internal = int(splits.shape[0])
    xlim, ylim = axis_limits(X_scaled)
    xx, yy, grid_points = make_grid(xlim, ylim)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.2), sharex=True, sharey=True)
    draw_root_region_fill(axes[0], xx, yy, centers, tree, xlim, ylim)
    regression_norm = None
    if task == "regression":
        regression_norm = Normalize(
            vmin=float(np.nanmin(y)),
            vmax=float(np.nanmax(y)),
        )
    draw_terminal_region_fill(
        axes[1],
        model,
        grid_points,
        xx.shape,
        task,
        y,
        xlim,
        ylim,
        class_labels=np.unique(y) if task == "classification" else None,
        regression_norm=regression_norm,
    )
    draw_split_circles(axes[0], splits.head(1), centers, label_prefix="root")
    axes[0].set_title("Root split frontier")
    n_drawn = draw_split_circles(axes[1], splits, centers, all_frontiers=True)
    axes[1].set_title("All frontiers and terminal regions")

    regression_mappable = None
    if task == "classification":
        classes = np.unique(y)
        for i, label in enumerate(classes):
            mask = y == label
            for ax in axes:
                ax.scatter(
                    X_scaled[mask, 0],
                    X_scaled[mask, 1],
                    c=CLASS_PALETTE[i % len(CLASS_PALETTE)],
                    s=28,
                    edgecolor="white",
                    linewidth=0.55,
                    label=str(label),
                    zorder=4,
                )
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=CLASS_PALETTE[i % len(CLASS_PALETTE)],
                markeredgecolor="white",
                markersize=7,
                label=str(label),
            )
            for i, label in enumerate(classes[:8])
        ]
    else:
        scatter0 = None
        for ax in axes:
            scatter0 = ax.scatter(
                X_scaled[:, 0],
                X_scaled[:, 1],
                c=y,
                cmap="viridis",
                norm=regression_norm,
                s=30,
                edgecolor="white",
                linewidth=0.55,
                zorder=4,
            )
        regression_mappable = scatter0
        legend_handles = []

    for ax in axes:
        ax.set_xlabel(f"standardized {feature_names[0]}")
        ax.set_ylabel(f"standardized {feature_names[1]}")
        ax.grid(color="#d9dde3", linewidth=0.55, alpha=0.75)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    circle_handle = Line2D([0], [0], color="#111111", linewidth=2, label="sphere boundary")
    fill_handle = Patch(facecolor="#999999", alpha=0.17, edgecolor="none", label="terminal regions")
    if legend_handles:
        fig.legend(
            handles=legend_handles + [circle_handle, fill_handle],
            loc="lower center",
            ncol=min(len(legend_handles) + 2, 5),
            frameon=False,
        )
    else:
        fig.legend(handles=[circle_handle, fill_handle], loc="lower center", ncol=2, frameon=False)

    if n_internal > n_drawn:
        note = (
            f"Full tree has {n_internal} circular frontiers; "
            f"the first {n_drawn} are drawn by depth for readability."
        )
    else:
        note = f"All {n_internal} circular frontiers are drawn."
    fig.suptitle(f"{dataset}: spherical split frontiers in two continuous predictors", fontsize=14, y=0.98)
    fig.text(
        0.5,
        0.06,
        "Each circle is a learned split frontier: inside goes left, outside goes right. "
        f"The right panel shades terminal leaf regions. {note}",
        ha="center",
        fontsize=9,
        color="#4a4f58",
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98 if regression_mappable is None else 0.91,
        bottom=0.16,
        top=0.88,
        wspace=0.12,
    )
    if regression_mappable is not None:
        fig.colorbar(
            regression_mappable,
            ax=axes.ravel().tolist(),
            shrink=0.80,
            pad=0.02,
            label="target / terminal leaf prediction",
        )

    figure_path = FIGURES_DIR / f"{slug}.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    return {
        "task": task,
        "dataset": dataset,
        "n_used_samples": int(row.n_used_samples),
        "n_features": int(row.n_features),
        "feature_1": feature_names[0],
        "feature_2": feature_names[1],
        "node_count": int(tree.node_count),
        "internal_node_count": n_internal,
        "drawn_circle_count": n_drawn,
        "figure": str(figure_path.relative_to(RESULTS_DIR)),
        "splits": str(split_path.relative_to(RESULTS_DIR)),
    }


def write_index(index: pd.DataFrame) -> Path:
    path = RESULTS_DIR / "spherical_2d_partitions.md"
    display = index.copy()
    text = [
        "# Spherical 2D Partition Figures",
        "",
        "These figures are generated for benchmark datasets with exactly two",
        "continuous-looking numerical predictors. Each model is fitted with the",
        "same spherical tree hyperparameters used in the full PMLB benchmark:",
        f"`n_center_candidates={N_CENTER_CANDIDATES}`,",
        f"`radius_candidates={RADIUS_CANDIDATES}`,",
        f"`center_strategy={CENTER_STRATEGY}`.",
        "",
        "The displayed coordinates are standardized because spherical splits are",
        "distance-based and the benchmark pipeline standardizes predictors before",
        "fitting.",
        "",
        "Circles are the explicit split frontiers. The right-hand panel adds a",
        "light fill for the terminal leaf regions induced by those frontiers;",
        "the left-hand panel only shades the two root children for orientation.",
        "The axes are centered on the observed data cloud, so far-field circles",
        "may appear as low-curvature arcs clipped by the plotting window.",
        "",
        display.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(text))
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    candidates = benchmark_candidates()
    rows = []
    for i, row in enumerate(candidates.itertuples(index=False), start=1):
        print(
            f"[{i}/{len(candidates)}] plotting {row.task} {row.dataset}",
            flush=True,
        )
        rows.append(plot_dataset(pd.Series(row._asdict())))
    index = pd.DataFrame(rows)
    index_path = RESULTS_DIR / "spherical_2d_partitions_index.csv"
    index.to_csv(index_path, index=False)
    markdown_path = write_index(index)
    print(f"Wrote {index_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
