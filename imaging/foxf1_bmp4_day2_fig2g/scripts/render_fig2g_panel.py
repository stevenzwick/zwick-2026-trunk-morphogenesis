#!/usr/bin/env python
"""Fig 2g: FOXF1 vs BMP4 reporter pixel density in day 2 cysts, redrawn from derived/ alone.

Reads the three tables the last cell of notebooks/03_global_q995_density_review.ipynb writes and
draws the panel that notebook's export cell draws, with the same normalisation, colour-scale
clipping, contours, cutoff lines, axis limits, labels and annotations. No raw images are needed.

    fig2g_histogram_counts.tsv  pixels in each 96 x 96 bin, before normalisation
    fig2g_bin_edges.tsv         bin edges in display units
    fig2g_summary.tsv           n, pixel count, cutoffs, display settings, quadrant pixel counts and
                                the sums over the pixels that give the Pearson R2

Prints n, the Pearson R2 computed from the stored sums, and the quadrant percentages.

Env: pandas, numpy, scipy, matplotlib.
Run: python imaging/foxf1_bmp4_day2_fig2g/scripts/render_fig2g_panel.py
Writes figures/output/fig2g_foxf1_bmp4_pixel_density.{png,pdf}. `--derived DIR` and `--out DIR`
read and write elsewhere.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANE = Path(__file__).resolve().parents[1]
REPO = LANE.parents[1]
NAME = "fig2g_foxf1_bmp4_pixel_density"

# As in notebook 03's QUADRANT_LABELS and export cell.
QUADRANTS = ["RFP-/YFP-", "RFP+/YFP-", "RFP-/YFP+", "RFP+/YFP+"]
QUADRANT_TEXT_POSITIONS = {
    "RFP-/YFP+": (0.03, 0.96, "left", "top"),
    "RFP+/YFP+": (0.97, 0.96, "right", "top"),
    "RFP-/YFP-": (0.03, 0.04, "left", "bottom"),
    "RFP+/YFP-": (0.97, 0.04, "right", "bottom"),
}
CONTOUR_LEVELS = [0.15, 0.35, 0.55, 0.75]


def load(derived: Path):
    summary = pd.read_csv(derived / "fig2g_summary.tsv", sep="\t", dtype=str).set_index("quantity")["value"]
    value = lambda key: float(summary[key])                                  # noqa: E731
    count = lambda key: int(summary[key])                                    # noqa: E731
    edges = pd.read_csv(derived / "fig2g_bin_edges.tsv", sep="\t")
    x_edges = edges.loc[edges["axis"] == "foxf1"].sort_values("edge_index")["edge"].to_numpy(float)
    y_edges = edges.loc[edges["axis"] == "bmp4"].sort_values("edge_index")["edge"].to_numpy(float)
    table = pd.read_csv(derived / "fig2g_histogram_counts.tsv", sep="\t")
    counts = np.zeros((x_edges.size - 1, y_edges.size - 1), dtype=np.float64)
    counts[table["foxf1_bin"].to_numpy(), table["bmp4_bin"].to_numpy()] = table["pixel_count"].to_numpy()
    return summary, value, count, x_edges, y_edges, counts


def pearson_r_squared(n: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
    r = (n * sxy - sx * sy) / math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return r * r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--derived", type=Path, default=LANE / "derived")
    parser.add_argument("--out", type=Path, default=REPO / "figures" / "output")
    args = parser.parse_args()

    summary, value, count, x_edges, y_edges, counts = load(args.derived)
    n_cysts, pixel_count = count("n_cysts"), count("pixel_count")
    # The bins span the 0.1-99.9th percentile range of each axis, so they hold slightly fewer pixels
    # than the pooled total that R2 and the quadrant percentages use.
    if int(counts.sum()) > pixel_count:
        raise SystemExit(f"histogram holds {int(counts.sum())} pixels, more than the {pixel_count} pooled")
    r2 = pearson_r_squared(pixel_count, value("sum_x"), value("sum_y"), value("sum_xx"),
                           value("sum_yy"), value("sum_xy"))
    fractions = {q: count(f"pixels_{q}") / pixel_count for q in QUADRANTS}
    x_cut, y_cut = value("foxf1_cutoff_display"), value("bmp4_cutoff_display")

    # plot_density_panel_from_arrays in notebook 03, from the stored counts.
    display_grid = counts.copy()
    if np.nanmax(display_grid) > 0:
        display_grid = display_grid / np.nanmax(display_grid)
    if np.any(display_grid > 0):
        clip_value = float(np.quantile(display_grid[display_grid > 0], value("display_clip_quantile")))
        if np.isfinite(clip_value) and clip_value > 0:
            display_grid = np.clip(display_grid, 0.0, clip_value) / clip_value
    smoothed = ndi.gaussian_filter(display_grid, sigma=value("contour_sigma"))

    fig, ax = plt.subplots(figsize=(6.7, 5.8), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, y_edges, display_grid.T, shading="auto", cmap="YlGn", vmin=0.0, vmax=1.0)
    if float(np.nanmax(smoothed)) > 0:
        ax.contour(0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:]), smoothed.T,
                   levels=CONTOUR_LEVELS, colors="#2f2f2f", linewidths=0.9)
    ax.axvline(x_cut, color="#111111", linewidth=1.2, linestyle="--")
    ax.axhline(y_cut, color="#111111", linewidth=1.2, linestyle="--")
    ax.set_xlim(float(x_edges[0]), min(float(x_edges[-1]), value("foxf1_axis_max_display")))
    ax.set_ylim(float(y_edges[0]), min(float(y_edges[-1]), value("bmp4_axis_max_display")))
    ax.set_title("FOXF1 vs BMP4 Pixel Density")
    for label, (x_pos, y_pos, ha, va) in QUADRANT_TEXT_POSITIONS.items():
        ax.text(x_pos, y_pos, f"{label}\n{100.0 * fractions[label]:.1f}%", transform=ax.transAxes,
                ha=ha, va=va, fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cfd8dc", "boxstyle": "round,pad=0.28"})
    ax.set_xlabel("FOXF1 Intensity (A.U.)")
    ax.set_ylabel("BMP4 Intensity (A.U.)")
    ax.text(0.50, 0.97, f"n = {n_cysts} cysts\nPearson R$^2$ = {r2:.3f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cfd8dc", "boxstyle": "round,pad=0.24"})
    colorbar = fig.colorbar(mesh, ax=ax, location="right", pad=0.02, shrink=0.95)
    colorbar.set_label("Normalized Pixel density")

    args.out.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(args.out / f"{NAME}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)

    print(f"Fig 2g  n = {n_cysts} cysts, {pixel_count:,} pixels ({int(counts.sum()):,} inside the binned range)")
    print(f"  Pearson R2 = {r2:.6f} (panel: {r2:.3f})")
    print(f"  cutoffs: FOXF1 {value('foxf1_cutoff_raw'):g} raw = {x_cut:.6f} display; "
          f"BMP4 {value('bmp4_cutoff_raw'):g} raw = {y_cut:.6f} display")
    for q in QUADRANTS:
        print(f"  {q}  {count(f'pixels_{q}'):>10,} pixels  {fractions[q]:.6f}  (panel: {100.0 * fractions[q]:.1f}%)")
    shown = args.out.resolve()
    shown = shown.relative_to(REPO) if shown.is_relative_to(REPO) else shown
    print(f"  wrote {shown / NAME}.png and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
