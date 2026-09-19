"""Shared plotting helpers.

These scripts emit the pre-assembly plot, not the published panel. Every figure in the paper
passed through Illustrator for composition, fonts, arrows, callouts, and (on the heatmaps) deletion
of the non-highlighted gene labels. What you get here matches the published panel in **data and
visual encoding** — values, colour map, normalisation, axes — not pixel for pixel.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "figures" / "output"

#: The paper's heatmaps and dot plots use `batlow` (Crameri's perceptually uniform scientific
#: colour map). If the `cmcrameri` package is unavailable, fall back to viridis and say so —
#: silently substituting a colour map would change the figure's visual encoding.
def batlow():
    try:
        from cmcrameri import cm
        return cm.batlow
    except ImportError:
        print("  note: cmcrameri not installed — falling back to viridis. "
              "Install cmcrameri to match the published colour map.")
        return plt.get_cmap("viridis")


def save(fig, name: str, subdir: str = "") -> Path:
    out = OUTPUT / subdir if subdir else OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")
    return path


def trace_panel(groups, xlabel: str, ylabel: str, title: str, shade_sd: bool = True):
    """Mean +/- spread against a distance or angle axis, one line per group."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for label, x, mean, spread in groups:
        order = np.argsort(x)
        x, mean = np.asarray(x)[order], np.asarray(mean)[order]
        line, = ax.plot(x, mean, linewidth=1.8, label=label)
        if shade_sd and spread is not None:
            spread = np.asarray(spread)[order]
            ax.fill_between(x, mean - spread, mean + spread, alpha=0.18, color=line.get_color(),
                            linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def dot_panel(matrix, sizes, row_labels, col_labels, cbar_label: str, title: str,
              vmin=None, vmax=None):
    """Dot plot: colour encodes expression, dot area encodes the fraction expressing."""
    fig, ax = plt.subplots(figsize=(0.30 * len(col_labels) + 3.0, 0.22 * len(row_labels) + 1.8))
    ys, xs = np.mgrid[0:len(row_labels), 0:len(col_labels)]
    scatter = ax.scatter(xs.ravel(), ys.ravel(), s=(np.asarray(sizes).ravel() * 90) + 1,
                         c=np.asarray(matrix).ravel(), cmap=batlow(), vmin=vmin, vmax=vmax,
                         edgecolors="none")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6, style="italic")
    ax.set_ylim(len(row_labels) - 0.5, -0.5)
    ax.set_title(title, fontsize=9)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    fig.colorbar(scatter, ax=ax, shrink=0.45, label=cbar_label).ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig
