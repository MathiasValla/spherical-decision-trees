"""Create graphical abstract assets for the spherical trees manuscript."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


PNG_PATH = "paper/Graphical_Abstract.png"
PDF_PATH = "paper/graphical_abstract.pdf"
TIFF_PATH = "paper/graphical_abstract.tiff"


def _toy_points(seed: int = 3):
    rng = np.random.default_rng(seed)
    outer = rng.normal(size=(70, 2))
    outer /= np.maximum(np.linalg.norm(outer, axis=1, keepdims=True), 1e-12)
    outer *= rng.uniform(0.9, 1.35, size=(70, 1))
    inner = rng.normal(scale=0.28, size=(45, 2))
    x = np.vstack([outer, inner])
    y = np.r_[np.zeros(len(outer)), np.ones(len(inner))]
    return x, y


def _scatter(ax, x, y):
    colors = np.where(y > 0, "#2878b5", "#d04f3b")
    ax.scatter(x[:, 0], x[:, 1], c=colors, s=24, edgecolor="white", linewidth=0.6)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def main() -> None:
    x, y = _toy_points()
    fig, axs = plt.subplots(1, 4, figsize=(13.28, 5.31), dpi=100)
    fig.patch.set_facecolor("white")

    _scatter(axs[0], x, y)
    axs[0].add_patch(Rectangle((-0.48, -0.48), 0.96, 0.96, fill=False, lw=2.5, ec="#222222"))
    axs[0].axvline(-0.48, color="#222222", lw=1.3)
    axs[0].axvline(0.48, color="#222222", lw=1.3)
    axs[0].axhline(-0.48, color="#222222", lw=1.3)
    axs[0].axhline(0.48, color="#222222", lw=1.3)
    axs[0].set_title("Axis-aligned\nrectangles", fontsize=13, fontweight="bold")

    _scatter(axs[1], x, y)
    grid = np.linspace(-1.6, 1.6, 200)
    axs[1].plot(grid, -0.55 * grid + 0.15, color="#222222", lw=2.5)
    axs[1].plot(grid, 0.35 * grid - 0.55, color="#222222", lw=1.7)
    axs[1].set_title("Oblique\nhalfspaces", fontsize=13, fontweight="bold")

    _scatter(axs[2], x, y)
    axs[2].add_patch(Circle((0.02, 0.02), 0.62, fill=False, lw=2.8, ec="#7f1d1d"))
    axs[2].add_patch(Circle((0.35, -0.2), 1.15, fill=False, lw=1.6, ec="#7f1d1d", alpha=0.65))
    axs[2].set_title("Spherical\ninside/outside", fontsize=13, fontweight="bold")

    ax = axs[3]
    names = ["RF", "Oblique\nRF", "Spherical\nRF"]
    values = [0.738, 0.737, 0.765]
    colors = ["#8a8a8a", "#6f8db8", "#7f1d1d"]
    bars = ax.bar(names, values, color=colors, width=0.62)
    ax.set_ylim(0.68, 0.78)
    ax.set_ylabel("Mean balanced accuracy", fontsize=10)
    ax.set_title("Classification\nbenchmark", fontsize=13, fontweight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle(
        "Spherical decision trees add learned hypersphere splits to recursive partitioning",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        "Distant centers recover linear frontiers locally; CART pruning applies after growth.",
        ha="center",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.92), w_pad=2.0)
    fig.savefig(PNG_PATH, dpi=100)
    fig.savefig(PDF_PATH)
    fig.savefig(TIFF_PATH, dpi=300, pil_kwargs={"compression": "tiff_lzw"})


if __name__ == "__main__":
    main()
