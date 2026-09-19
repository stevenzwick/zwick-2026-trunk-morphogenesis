#!/usr/bin/env python
"""Redraw ED Fig 3g from the shipped table and print pixel counts and quadrant percentages.

Reads (in ../derived/, or --derived):
    ed3g_per_pixel.tsv  one row per analysed pixel: FOXA2, FOXF1 (FOXA2 panel), SOX2, FOXF1
                        (SOX2 panel), all normalized 0-1, undithered
    ed3g_summary.tsv    the constants the notebook used, and the values it printed

No images are needed. The two panels analyse the same 504,789 pooled pixels from seven day-2
cysts (three independent differentiations): FOXA2 versus FOXF1 on top, SOX2 versus FOXF1 on the
bottom. Quadrant statistics are computed from the undithered columns, exactly as the notebook
does; the dither (SEED, JITTER) is applied only to the histogram used for the density image.

One precision note carried over from the notebook: the SOX2 gate, 15/44 raw units, does not
terminate at 6 decimal digits, while the shipped table is rounded to 6 decimals. Comparing the
rounded table against the unrounded gate value undercounts the SOX2-positive pixels by 9,416 (a
whole quantization level sitting exactly on the rounded gate). This script rounds each gate to
the table's own 6-decimal precision before comparing, which is the precision the table can
support; ed3g_summary.tsv records both figures under sox2_table_Qlr and
sox2_table_unrounded_gate_Qlr.

Writes figures/output/ed3g_foxa2_sox2_foxf1_density.{pdf,png}.
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

SEED = 0
JITTER = 0.5
HIST_BINS = 256
CONTOUR_LEVELS = np.linspace(0.05, 1, 12)
SMOOTH_SIGMA = (2, 1)
CLIP_PCT = 99  # colour-scale reference percentile, see density_image
GATE_X_RAW = 15.0
GATE_Y_RAW = 45.0
# The notebook's own colormaps for the FOXA2 and SOX2 panels.
CMAPS = {
    "FOXA2": ["#FFFFFF", "#F4C4F3", "#E052D0", "#9D0191"],
    "SOX2":  ["#FFFFFF", "#FFF7B2", "#FFE600", "#FFC300"],
}


def load(derived):
    xy = np.loadtxt(os.path.join(derived, "ed3g_per_pixel.tsv"), delimiter="\t", skiprows=1)
    summary = {}
    with open(os.path.join(derived, "ed3g_summary.tsv")) as fh:
        next(fh)
        for line in fh:
            key, value, _ = line.rstrip("\n").split("\t", 2)
            summary[key] = value
    return xy, summary


def stats(x, y, xg, yg):
    """Quadrant proportions and derived statistics, matching the notebook's stats()."""
    xp, yp = x >= xg, y >= yg
    Qll = int(np.sum(~xp & ~yp)); Qlr = int(np.sum(xp & ~yp))
    Qul = int(np.sum(~xp & yp));  Qur = int(np.sum(xp & yp))
    N = Qll + Qlr + Qul + Qur
    either = Qlr + Qul + Qur
    nan = float("nan")
    return {
        "Qll": Qll / N, "Qlr": Qlr / N, "Qul": Qul / N, "Qur": Qur / N,
        "n_total": N, "n_marker_only": Qlr, "n_foxf1_only": Qul, "n_double": Qur,
        "n_either": either,
        "jaccard": Qur / either if either else nan,
        "p_foxf1_given_marker": Qur / (Qur + Qlr) if (Qur + Qlr) else nan,
        "p_marker_given_foxf1": Qur / (Qur + Qul) if (Qur + Qul) else nan,
    }


def density_image(x, y, xd, yd, cmap_colors):
    H, xe, ye = np.histogram2d(xd, yd, bins=HIST_BINS)
    H_norm = np.log1p(H) / np.log1p(H).max()
    H_smooth = scipy.ndimage.gaussian_filter(H_norm, sigma=SMOOTH_SIGMA)
    # Colour scale. Dividing by the maximum leaves almost the whole panel pale, because the
    # single bin at the origin is far denser than everything else. The fill is therefore scaled
    # to the CLIP_PCT-th percentile of bin density and clipped, so that 1 on the colour bar means
    # "at or above that percentile". The contours are drawn from H_norm and are unaffected.
    H_disp = np.clip(H_norm / np.percentile(H_norm, CLIP_PCT), 0, 1)
    cmap = colors.LinearSegmentedColormap.from_list("c", cmap_colors)
    return H_disp, H_smooth, xe, ye, cmap


def panel(ax, marker, x, y, xscale, yscale, rng, s):
    """One axes: density image, contours, gate lines, corner annotations."""
    xd = x + rng.uniform(-JITTER, JITTER, size=x.shape) / xscale
    yd = y + rng.uniform(-JITTER, JITTER, size=y.shape) / yscale
    xd, yd = np.clip(xd, 0, 1), np.clip(yd, 0, 1)

    H_disp, H_smooth, xe, ye, cmap = density_image(x, y, xd, yd, CMAPS[marker])
    im = ax.imshow(H_disp.T, origin="lower", cmap=cmap,
                   extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto", vmin=0, vmax=1)
    ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), H_smooth.T,
               levels=CONTOUR_LEVELS, colors="black", linewidths=0.5)

    xg = round(GATE_X_RAW / xscale, 6)
    yg = round(GATE_Y_RAW / yscale, 6)
    ax.axvline(xg, color="0.35", ls="--", lw=0.8)
    ax.axhline(yg, color="0.35", ls="--", lw=0.8)

    xlo, xhi = float(x.min()), float(x.max())
    ylo, yhi = float(y.min()), float(y.max())
    for tx, ty, key in [(xlo, ylo, "Qll"), (xhi, ylo, "Qlr"), (xlo, yhi, "Qul"), (xhi, yhi, "Qur")]:
        ax.annotate(f"{s[key] * 100:.1f}%", xy=((tx + xg) / 2, (ty + yg) / 2),
                    ha="center", va="center", fontsize=9, weight="bold")

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(marker + " normalized intensity")
    ax.set_ylabel("FOXF1 normalized intensity")
    ax.set_title(marker + " vs FOXF1 pixel density")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default=os.path.join(LANE, "derived"))
    ap.add_argument("--outdir", default=os.path.join(REPO, "figures", "output"))
    args = ap.parse_args()

    xy, summary = load(args.derived)
    foxa2_x, foxf1_y_a, sox2_x, foxf1_y_b = xy[:, 0], xy[:, 1], xy[:, 2], xy[:, 3]
    xscale_foxa2 = float(summary["xscale_foxa2"])
    xscale_sox2 = float(summary["xscale_sox2"])
    yscale = float(summary["yscale_foxf1"])

    xg_foxa2 = round(GATE_X_RAW / xscale_foxa2, 6)
    yg = round(GATE_Y_RAW / yscale, 6)
    xg_sox2 = round(GATE_X_RAW / xscale_sox2, 6)

    s_foxa2 = stats(foxa2_x, foxf1_y_a, xg_foxa2, yg)
    s_sox2 = stats(sox2_x, foxf1_y_b, xg_sox2, yg)

    # A single RNG stream, drawn FOXA2-x, FOXA2-y, SOX2-x, SOX2-y in that order, matching the
    # notebook's call sequence (one `rng` shared across both build_panel() calls).
    rng = np.random.default_rng(SEED)

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(5, 9), facecolor="white")
    im_top = panel(ax_top, "FOXA2", foxa2_x, foxf1_y_a, xscale_foxa2, yscale, rng, s_foxa2)
    fig.colorbar(im_top, ax=ax_top, label="Pixel density")
    im_bottom = panel(ax_bottom, "SOX2", sox2_x, foxf1_y_b, xscale_sox2, yscale, rng, s_sox2)
    fig.colorbar(im_bottom, ax=ax_bottom, label="Pixel density")
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    stem = os.path.join(args.outdir, "ed3g_foxa2_sox2_foxf1_density")
    fig.savefig(stem + ".pdf", format="pdf", dpi=300)
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)

    for marker, x, s in [("FOXA2", foxa2_x, s_foxa2), ("SOX2", sox2_x, s_sox2)]:
        print(f"{marker} vs FOXF1:  pixels {s['n_total']}")
        print(f"  Qll {s['Qll']:.4f}  Qlr {s['Qlr']:.4f}  Qul {s['Qul']:.4f}  Qur {s['Qur']:.4f}"
              f"   (marker+ {s['n_marker_only'] + s['n_double']}, FOXF1+ {s['n_foxf1_only'] + s['n_double']},"
              f" either {s['n_either']}, double {s['n_double']})")
        print(f"  Jaccard {s['jaccard'] * 100:.2f}%   "
              f"P(FOXF1+|{marker}+) {s['p_foxf1_given_marker'] * 100:.2f}%   "
              f"P({marker}+|FOXF1+) {s['p_marker_given_foxf1'] * 100:.2f}%")

    print(f"wrote            {stem}.pdf and .png")


if __name__ == "__main__":
    main()
