#!/usr/bin/env python
"""06 - pSMAD1/5/9 vs distance from the colony edge (spatial gradient profile), 30 min.

The Zhang-2019-style readout: in responsive cells pSMAD1/5/9 is highest at the colony edge
and falls off into the interior. For each condition, cells are binned by distance from the
colony edge (d_edge); the LINE is the median over the 3 wells of each well's per-bin median,
and the shaded BAND spans the 3 wells (min-max). Well-based (unit = well, n = 3), consistent
with the box plot / timecourse. DESCRIPTIVE - the formal 30-min comparison is the box-plot
panel; no fold/ratio (bgsub = pure image background). The 0-25 um edge zone (the box-plot
edge/interior cutoff) is shaded. Only bins where ALL wells have >= min_cells are drawn, so
the band is always a true 3-well span.

Reads results/tables/per_cell_dedge.csv (from 02). Writes:
  results/figures/pSMAD_edge_distance_profile_v<VERSION>.png
  results/tables/edge_distance_profile_<timepoint>.csv
Versioned per the figure-iteration convention (bump VERSION; never overwrite a look).
Env: psmad_quant.
Run: /opt/anaconda3/bin/conda run -n psmad_quant python scripts/06_edge_profile.py [--timepoint 30min]

NOT runnable as shipped. This reads `results/tables/per_cell_dedge.csv`, the full per-cell table
produced by `02_dedge_quant.py` from the raw images. That file is not included; what ships is a
single-field QC export (`derived/per_cell_dedge_QC.csv`).

This script draws `pSMAD_edge_distance_profile_v{VERSION}.png`, not the nine-box Supplementary
Data 4 display and not ED Fig 3k. The nine-box figure is drawn by `03_figure.py` and
`07_panel_figure.py`; ED 3k is the four-box panel
(SCRAMBLE KD and BMPR1A KD, each without and with BMP4), drawn by `08_ed3k_panel.py`, and its
twelve per-well values ship as `derived/negctrl_summary_per_field_30min.csv`. See the lane README.
"""
import os, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42  # embed editable TrueType text in PDFs
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]  # global convention: Arial

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB  = os.path.join(WORK, "results", "tables")
FIG  = os.path.join(WORK, "results", "figures")
CONDS   = ["no_KD", "SCRAMBLE_KD", "BMPR1A_KD"]
DISPLAY = {"no_KD": "no KD", "SCRAMBLE_KD": "scramble KD", "BMPR1A_KD": "BMPR1A KD"}
COLORS  = {"no_KD": "#555555", "SCRAMBLE_KD": "#2a9d5c", "BMPR1A_KD": "#d23b3b"}
VERSION = "2"   # v2: removed on-plot "edge zone" text label (-> caption), gray box kept. bump each iteration; never overwrite a prior render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timepoint", default="30min")
    ap.add_argument("--value", default="psmad_bgsub")
    ap.add_argument("--binw", type=float, default=10.0, help="distance bin width (um)")
    ap.add_argument("--min_cells", type=int, default=10, help="min cells per well per bin to include a bin")
    ap.add_argument("--edge_um", type=float, default=25.0)
    ap.add_argument("--xmax", type=float, default=100.0, help="x-axis cap (um); decay completes by ~60 um, tail is flat interior")
    ap.add_argument("--csv", default=os.path.join(TAB, "per_cell_dedge.csv"))
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)

    df = pd.read_csv(args.csv)
    d = df[(df.timepoint == args.timepoint) & (df.dedge_um >= 0)].copy()
    d["bin"] = (d.dedge_um // args.binw).astype(int)

    rows = []
    for c in CONDS:
        g = d[d.condition == c]
        nwell = g.field.nunique()
        for b, gb in g.groupby("bin"):
            wm = [gf[args.value].median() for _, gf in gb.groupby("field") if len(gf) >= args.min_cells]
            if len(wm) == nwell:                       # all wells present -> true 3-well band
                rows.append(dict(condition=c, dist_um=(b + 0.5) * args.binw,
                                 med_of_wells=float(np.median(wm)),
                                 lo=float(np.min(wm)), hi=float(np.max(wm)), n_wells=len(wm)))
    prof = pd.DataFrame(rows)
    prof.to_csv(os.path.join(TAB, f"edge_distance_profile_{args.timepoint}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axvspan(0, args.edge_um, color="0.85", alpha=0.6, zorder=0, linewidth=0)  # edge zone; labeled in the caption, not on-plot
    for c in CONDS:
        p = prof[prof.condition == c].sort_values("dist_um")
        if not len(p):
            continue
        ax.fill_between(p.dist_um, p.lo, p.hi, color=COLORS[c], alpha=0.18, zorder=2, linewidth=0)
        ax.plot(p.dist_um, p.med_of_wells, color=COLORS[c], lw=2.2, zorder=3, label=DISPLAY[c])
    ax.set_xlabel("distance from colony edge (µm)")
    ax.set_ylabel("pSMAD1/5/9 (a.u.)")
    ax.set_xlim(0, args.xmax); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    figpath = os.path.join(FIG, f"pSMAD_edge_distance_profile_v{VERSION}.png")
    fig.savefig(figpath, dpi=150); fig.savefig(figpath.replace(".png", ".pdf"), dpi=300); plt.close(fig)

    # ---- console summary (edge vs interior extents) ----
    print(f"== pSMAD1/5/9 vs distance from edge ({args.timepoint}); median-of-wells [3-well band] ==")
    for c in CONDS:
        p = prof[prof.condition == c].sort_values("dist_um")
        if not len(p):
            continue
        near = p.iloc[0]; far = p.iloc[-1]
        print(f"  {DISPLAY[c]:12s}  {near.dist_um:3.0f}um={near.med_of_wells:5.0f} [{near.lo:.0f}-{near.hi:.0f}]"
              f"  ...  {far.dist_um:3.0f}um={far.med_of_wells:5.0f} [{far.lo:.0f}-{far.hi:.0f}]"
              f"   (bins {p.dist_um.min():.0f}-{p.dist_um.max():.0f}um)")
    print(f"\nfigure -> {figpath}\ntable  -> {TAB}/edge_distance_profile_{args.timepoint}.csv")


if __name__ == "__main__":
    main()
