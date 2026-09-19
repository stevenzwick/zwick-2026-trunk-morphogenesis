#!/usr/bin/env python
"""Fig 5e: CFP versus FOXF1 pixel density in 10% BMPR1A knockdown morphs, redrawn from derived/ alone.

Reads the three tables the last cell of notebooks/cfp_foxf1_pixel_density.ipynb writes and draws the
panel that notebook's figure cell draws, with the same log transform, normalisation, colour map,
colour-scale clipping, contours, gates, labels and quadrant percentages. No images are needed.

    fig5e_histogram_counts.tsv  pixels in each non-empty bin of the 256 x 256 histogram
    fig5e_bin_edges.tsv         bin edges in image intensity units
    fig5e_summary.tsv           n, pixel count, DAPI cutoff, gates and quadrant pixel counts

Prints n, the pixel count, the gates and the quadrant percentages.

Env: pandas, numpy, scipy, matplotlib.
Run: python imaging/cfp_foxf1_density_fig5e/scripts/render_fig5e_panel.py
Writes figures/output/fig5e_cfp_foxf1_pixel_density.{png,pdf}. `--derived DIR` and `--out DIR`
read and write elsewhere.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

LANE = Path(__file__).resolve().parents[1]
REPO = LANE.parents[1]
NAME = "fig5e_cfp_foxf1_pixel_density"
QUADRANTS = {"Qll": "CFP-/FOXF1-", "Qlr": "CFP+/FOXF1-", "Qul": "CFP-/FOXF1+", "Qur": "CFP+/FOXF1+"}


def load(derived: Path):
    summary = pd.read_csv(derived / "fig5e_summary.tsv", sep="\t", dtype=str).set_index("quantity")["value"]
    edges = pd.read_csv(derived / "fig5e_bin_edges.tsv", sep="\t")
    xedges = edges.loc[edges["axis"] == "cfp"].sort_values("edge_index")["edge"].to_numpy(float)
    yedges = edges.loc[edges["axis"] == "foxf1"].sort_values("edge_index")["edge"].to_numpy(float)
    bins = int(summary["bins"])
    if xedges.size != bins + 1 or yedges.size != bins + 1:
        raise SystemExit(f"expected {bins + 1} bin edges per axis")
    table = pd.read_csv(derived / "fig5e_histogram_counts.tsv", sep="\t")
    H = np.zeros((bins, bins), dtype=np.float64)
    H[table["cfp_bin"].to_numpy(), table["foxf1_bin"].to_numpy()] = table["pixel_count"].to_numpy()
    return summary, xedges, yedges, H


# As in the notebook's annotate_quadrants.
def annotate_quadrants(ax, x_thresh, y_thresh, props, text_kwargs=None, line_kwargs=None):
    if text_kwargs is None:
        text_kwargs = dict(fontsize=10, color='black', weight='bold')
    if line_kwargs is None:
        line_kwargs = dict(color='gray', lw=1.0, ls='--')
    ax.axvline(x_thresh, **line_kwargs)
    ax.axhline(y_thresh, **line_kwargs)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = 0.05 * (xmax - xmin)
    dy = 0.05 * (ymax - ymin)
    ax.text(x_thresh - dx, y_thresh - dy, f"{props['Qll']*100:.1f}%", ha='right', va='top', **text_kwargs)
    ax.text(x_thresh + dx, y_thresh - dy, f"{props['Qlr']*100:.1f}%", ha='left', va='top', **text_kwargs)
    ax.text(x_thresh - dx, y_thresh + dy, f"{props['Qul']*100:.1f}%", ha='right', va='bottom', **text_kwargs)
    ax.text(x_thresh + dx, y_thresh + dy, f"{props['Qur']*100:.1f}%", ha='left', va='bottom', **text_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--derived", type=Path, default=LANE / "derived")
    parser.add_argument("--out", type=Path, default=REPO / "figures" / "output")
    args = parser.parse_args()

    summary, xedges, yedges, H = load(args.derived)
    n_stacks, pixel_count = int(summary["n_stacks"]), int(summary["pixel_count"])
    x_gate, y_gate = float(summary["cfp_gate"]), float(summary["foxf1_gate"])
    counts = {key: int(summary[f"pixels_{key}"]) for key in QUADRANTS}
    # The histogram spans the full range of both channels, so it holds every pooled pixel.
    if int(H.sum()) != pixel_count or sum(counts.values()) != pixel_count:
        raise SystemExit(f"histogram ({int(H.sum())}) and quadrants ({sum(counts.values())}) "
                         f"must both hold the {pixel_count} pooled pixels")
    total = float(sum(counts.values()))
    props = {key: value / total for key, value in counts.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    # The notebook's figure cell, from the stored histogram.
    with plt.rc_context({"figure.dpi": (300)}):
        plt.style.use('default')
        fig, ax = plt.subplots(facecolor='white')

        teal_cmap = colors.LinearSegmentedColormap.from_list(
            "WhiteToTeal",
            ["#005a66",  # deep teal
             "#00bfae",  # mid teal
             "#d9fff9"]  # pale green-white
        )
        colors_mod = teal_cmap(np.linspace(0.3, 1, 256))  # skip darkest 30%
        colors_mod[:1] = [1, 1, 1, 1]  # ensure bottom is white
        white_to_inferno = colors.ListedColormap(colors_mod)

        H_log = np.log1p(H)
        H_norm = H_log / H_log.max()
        gamma = 1
        H_stretched = H_norm ** gamma
        H_smooth = scipy.ndimage.gaussian_filter(H_stretched.astype(float), sigma=(2, 1))

        vmax = np.percentile(H_stretched, 99)
        im = ax.imshow(
            H_stretched.T,
            origin='lower',
            cmap=white_to_inferno,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect='auto',
            vmin=0,
            vmax=vmax,
        )
        levels = np.linspace(0.1, 1, 8)
        ax.contour(
            0.5 * (xedges[:-1] + xedges[1:]),
            0.5 * (yedges[:-1] + yedges[1:]),
            H_smooth.T,
            levels=levels,
            colors='black',
            linewidths=0.5,
        )
        ax.set_xlabel("CFP intensity")
        ax.set_ylabel("FOXF1 intensity")
        fig.colorbar(im, ax=ax, label="Pixel density (log-scaled)")
        ax.set_title("FOXF1 vs CFP anticorrelation (density heatmap)")
        annotate_quadrants(ax, x_gate, y_gate, props)

        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(args.out / f"{NAME}.{ext}", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"Fig 5e  n = {n_stacks} stacks, {pixel_count:,} pixels above DAPI {summary['dapi_cutoff']}")
    print(f"  gates: CFP >= {x_gate:g}, FOXF1 >= {y_gate:g}")
    for key, name in QUADRANTS.items():
        print(f"  {name}  {counts[key]:>10,} pixels  {props[key]:.6f}  (panel: {100.0 * props[key]:.1f}%)")
    print(f"  colour scale: 0 to {vmax:.6f} (99th percentile of the normalised log counts)")
    print(f"  wrote {args.out / NAME}.png and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
