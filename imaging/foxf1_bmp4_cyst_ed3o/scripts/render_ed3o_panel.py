#!/usr/bin/env python
"""Redraw ED Fig 3o from the shipped table and print its n and R².

Reads (in ../derived/, or --derived):
    ed3o_per_pixel.tsv  one row per analysed pixel: FOXF1 and BMP4, each rescaled to 0-1
    ed3o_summary.tsv    the constants the notebook used, and the values it printed

No images are needed. The correlation is unchanged by the rescaling, so the R² below is the one in
the figure legend even though this table holds the rescaled values.

Writes figures/output/ed3o_foxf1_bmp4_density.{pdf,png}.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.ndimage

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(LANE))

BINS = 256
CONTOUR_LEVELS = np.linspace(0.25, 1, 4)
CONTOUR_SIGMA = (2, 1)
# The notebook's white-to-green scale.
CMAP = colors.LinearSegmentedColormap.from_list(
    "WhiteToGreen", ["#FFFFFF", "#B5E48C", "#76C893", "#1A7431"])


def load(derived):
    xy = np.loadtxt(os.path.join(derived, "ed3o_per_pixel.tsv"), delimiter="\t", skiprows=1)
    summary = {}
    with open(os.path.join(derived, "ed3o_summary.tsv")) as fh:
        next(fh)
        for line in fh:
            key, value, _ = line.rstrip("\n").split("\t", 2)
            summary[key] = value
    return xy[:, 0], xy[:, 1], summary


def r_squared(x, y):
    """Squared Pearson coefficient, the least-squares fit's R²."""
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) ** 2 / ((x @ x) * (y @ y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default=os.path.join(LANE, "derived"))
    ap.add_argument("--outdir", default=os.path.join(REPO, "figures", "output"))
    args = ap.parse_args()

    x, y, summary = load(args.derived)
    r2 = r_squared(x, y)

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=BINS)
    density = np.log1p(counts)
    density /= density.max()
    smoothed = scipy.ndimage.gaussian_filter(density, sigma=CONTOUR_SIGMA)

    fig, ax = plt.subplots(facecolor="white")
    image = ax.imshow(density.T, origin="lower", cmap=CMAP, vmin=0, vmax=1, aspect="auto",
                      extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    ax.set_aspect("equal")
    ax.contour(0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:]),
               smoothed.T, levels=CONTOUR_LEVELS, colors="black", linewidths=0.5)
    ax.set_xlabel("FOXF1 intensity (A.U.)")
    ax.set_ylabel("BMP4 intensity (A.U.)")
    fig.colorbar(image, ax=ax, label="Pixel density")
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42
    stem = os.path.join(args.outdir, "ed3o_foxf1_bmp4_density")
    fig.savefig(stem + ".pdf", format="pdf", dpi=300)
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)

    recorded = float(summary["r_squared_notebook"])
    print(f"pixels           {x.size}")
    print(f"R2 from table    {r2:.16f}")
    print(f"R2 the notebook printed from the unrounded values  {recorded:.16f}")
    print(f"legend value     {r2:.3f}")
    print(f"wrote            {stem}.pdf and .png")


if __name__ == "__main__":
    main()
