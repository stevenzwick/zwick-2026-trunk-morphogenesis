#!/usr/bin/env python
"""07 - assemble the 3 pSMAD-validation panels into ONE supplement figure (a, b, c).

Vertical stack, narrative order:
  a  edge-cell pSMAD by condition, 30 min: BMP4 response + 2 negative controls  (cf. 03_figure.py)
  b  edge/interior pSMAD timecourse, 0/15/30/45 min                             (cf. 05_timecourse.py)
  c  pSMAD vs distance from the colony edge, 30 min                             (cf. 06_edge_profile.py)

Each panel's drawing reproduces its standalone script VERBATIM (same styling) into a shared
axes, so the single source of truth for each panel's look stays 03/05/06; this is only the
assembler. Bold lowercase panel labels a/b/c. Output: 300-dpi vector PDF + PNG.

Reads results/tables/per_cell_dedge.csv. Writes results/figures/pSMAD_validation_figure_v<VERSION>.{pdf,png}.
Env: psmad_quant.
Run: /opt/anaconda3/bin/conda run -n psmad_quant python scripts/07_panel_figure.py

NOT runnable as shipped. This reads `results/tables/per_cell_dedge.csv`, the full per-cell table
produced by `02_dedge_quant.py` from the raw images. That file is not included; what ships is a
single-field QC export (`derived/per_cell_dedge_QC.csv`).

Note what this script draws: the **nine-box** figure — three knockdown conditions x (no BMP4 edge,
BMP4 edge, BMP4 interior) — which is the Supplementary Data 4 display, **not** ED Fig 3k. ED 3k is
the four-box panel (SCRAMBLE KD and BMPR1A KD, each without and with BMP4), and its twelve per-well
values ship as `derived/negctrl_summary_per_field_30min.csv`. See the lane README.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
matplotlib.rcParams["pdf.fonttype"] = 42  # embed editable TrueType text in PDFs
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]  # global convention: Arial
matplotlib.rcParams.update({"font.size": 12, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})  # larger text

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB  = os.path.join(WORK, "results", "tables")
FIG  = os.path.join(WORK, "results", "figures")
CONDS   = ["no_KD", "SCRAMBLE_KD", "BMPR1A_KD"]
DISPLAY = {"no_KD": "no KD", "SCRAMBLE_KD": "scramble KD", "BMPR1A_KD": "BMPR1A KD"}
COLORS  = {"no_KD": "#555555", "SCRAMBLE_KD": "#2a9d5c", "BMPR1A_KD": "#d23b3b"}
VALUE   = "psmad_bgsub"
EDGE_UM = 25.0
VERSION = "5"   # v5: larger text (axis/tick/legend/panel labels) + spacing to keep layout clean. v1-v4 kept.


def field_medians(df, cond, tp, edge):
    g = df[(df.condition == cond) & (df.timepoint == tp)]
    g = g[g.is_edge] if edge else g[~g.is_edge]
    v = g.groupby("field")[VALUE].median().values
    return v[~np.isnan(v)]


def draw_box(ax, df, baseline="no_BMP", stim="30min", conc=10.0):
    """Panel a - reproduced from 03_figure.py Panel A."""
    np.random.seed(0)
    C = f"{conc:g} ng/mL BMP4"
    cats = [("0 ng/mL BMP4,\nedge cells",     baseline, True,  "#cfcfcf"),   # 2-line labels -> narrower legend
            (f"{C},\nedge cells",             stim,     True,  "#2171b5"),
            (f"{C},\ninterior cells",         stim,     False, "#8a8a8a")]
    width = 0.26
    for ci, c in enumerate(CONDS):
        for k, (lab, tp, edge, col) in enumerate(cats):
            x = ci + (k - 1) * width
            v = field_medians(df, c, tp, edge)
            if not len(v):
                continue
            ax.boxplot([v], positions=[x], widths=width * 0.82, patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=0.55), medianprops=dict(color="k"),
                       whiskerprops=dict(color="#777"), capprops=dict(color="#777"), showfliers=False)
            ax.scatter(np.full(len(v), x) + np.random.uniform(-0.05, 0.05, len(v)), v,
                       color=col, s=30, zorder=3, edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([DISPLAY[c] for c in CONDS])
    ax.set_ylabel("pSMAD1/5/9 (a.u.)")
    ax.legend(handles=[mpatches.Patch(facecolor=col, alpha=0.6, label=lab) for lab, _, _, col in cats],
              frameon=True, framealpha=1.0, edgecolor="0.6", fancybox=False, fontsize=11, loc="upper left", bbox_to_anchor=(1.02, 1.0))  # standardized anchor


def draw_timecourse(ax, df):
    """Panel b - reproduced from 05_timecourse.py."""
    ORDER = ["no_BMP", "15min", "30min", "45min"]
    XT = {"no_BMP": 0, "15min": 15, "30min": 30, "45min": 45}
    present = list(pd.unique(df.timepoint))
    xtps = [t for t in ORDER if t in present and t in XT]
    xs = [XT[t] for t in xtps]

    def mow(tp, c, edge):
        v = field_medians(df, c, tp, edge)
        return np.median(v) if len(v) else np.nan

    for c in CONDS:
        ax.plot(xs, [mow(t, c, True) for t in xtps], "-", color=COLORS[c], lw=2.2, zorder=3)
        ax.plot(xs, [mow(t, c, False) for t in xtps], "--", color=COLORS[c], lw=1.3, alpha=0.5, zorder=2)
        for t, x in zip(xtps, xs):
            ve = field_medians(df, c, t, True); vi = field_medians(df, c, t, False)
            ax.scatter([x] * len(ve), ve, color=COLORS[c], s=18, zorder=4, edgecolor="k", linewidth=0.3)
            ax.scatter([x] * len(vi), vi, facecolor="white", edgecolor=COLORS[c], s=14, zorder=3, linewidth=0.8, alpha=0.75)
    ax.set_xlabel("time after BMP4 addition (min)"); ax.set_ylabel("pSMAD1/5/9 (a.u.)")
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.set_ylim(bottom=0); ax.margins(x=0.06)
    cond_h = [Line2D([0], [0], color=COLORS[c], lw=2.4, label=DISPLAY[c]) for c in CONDS]
    reg_h  = [Line2D([0], [0], color="#333", lw=2.2, ls="-", label="edge cells"),
              Line2D([0], [0], color="#333", lw=1.3, ls="--", alpha=0.6, label="interior cells")]
    leg1 = ax.legend(handles=cond_h, frameon=True, framealpha=1.0, edgecolor="0.6", fancybox=False, fontsize=11, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0))  # standardized anchor; titles dropped for uniformity
    ax.add_artist(leg1)
    ax.legend(handles=reg_h, frameon=True, framealpha=1.0, edgecolor="0.6", fancybox=False, fontsize=11, loc="upper left",
              bbox_to_anchor=(1.02, 0.58))


def draw_profile(ax, df, binw=10.0, min_cells=10, xmax=100.0):
    """Panel c - reproduced from 06_edge_profile.py."""
    d = df[(df.timepoint == "30min") & (df.dedge_um >= 0)].copy()
    d["bin"] = (d.dedge_um // binw).astype(int)
    ax.axvspan(0, EDGE_UM, color="0.85", alpha=0.6, zorder=0, linewidth=0)  # edge zone; labeled in caption
    for c in CONDS:
        g = d[d.condition == c]; nwell = g.field.nunique()
        rows = []
        for b, gb in g.groupby("bin"):
            wm = [gf[VALUE].median() for _, gf in gb.groupby("field") if len(gf) >= min_cells]
            if len(wm) == nwell:
                rows.append(((b + 0.5) * binw, float(np.median(wm)), float(np.min(wm)), float(np.max(wm))))
        if not rows:
            continue
        rows.sort()
        xs = [r[0] for r in rows]; med = [r[1] for r in rows]; lo = [r[2] for r in rows]; hi = [r[3] for r in rows]
        ax.fill_between(xs, lo, hi, color=COLORS[c], alpha=0.18, zorder=2, linewidth=0)
        ax.plot(xs, med, color=COLORS[c], lw=2.2, zorder=3, label=DISPLAY[c])
    ax.set_xlabel("distance from colony edge (µm)"); ax.set_ylabel("pSMAD1/5/9 (a.u.)")
    ax.set_xlim(0, xmax); ax.set_ylim(bottom=0)
    ax.legend(frameon=True, framealpha=1.0, edgecolor="0.6", fancybox=False, fontsize=11, loc="upper left", bbox_to_anchor=(1.02, 1.0))  # standardized: outside-right like a/b


def main():
    df = pd.read_csv(os.path.join(TAB, "per_cell_dedge.csv"))
    df["is_edge"] = df.dedge_um < EDGE_UM

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 13.8))
    fig.subplots_adjust(left=0.10, right=0.72, top=0.975, bottom=0.05, hspace=0.40)
    draw_box(axes[0], df)
    draw_timecourse(axes[1], df)
    draw_profile(axes[2], df)
    for ax, lab in zip(axes, "abc"):
        ax.text(-0.13, 1.03, lab, transform=ax.transAxes, fontweight="bold",
                fontsize=17, va="bottom", ha="left")

    base = os.path.join(FIG, f"pSMAD_validation_figure_v{VERSION}")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {base}.pdf  +  {base}.png")


if __name__ == "__main__":
    main()
