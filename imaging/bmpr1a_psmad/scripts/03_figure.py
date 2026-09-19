#!/usr/bin/env python
"""03 - headline figure + per-field summary: edge pSMAD1/5/9 response reduced in BMPR1A-KD.

Reads results/tables/per_cell_dedge.csv (from 02). For the 3 knockdown conditions:
  Panel A  per-field pSMAD with TWO negative controls shown alongside the response:
             no-BMP edge cells (stimulus-negative) | +BMP edge cells (response) |
             +BMP interior cells (within-well, receptor-shielded floor).   n = fields.
  Panel B  pSMAD vs d_edge profile per condition (pooled cells, binned median).
Edge cell = d_edge < --edge_um (default 25 um).

Writes results/figures/*_<timepoint>.png + results/tables/negctrl_summary_per_field_<timepoint>.csv.
Env: psmad_quant.
Run: /opt/anaconda3/bin/conda run -n psmad_quant python scripts/03_figure.py [--timepoint 30min] [--baseline no_BMP] [--edge_um 25]

NOT runnable as shipped. This reads `results/tables/per_cell_dedge.csv`, the full per-cell table
produced by `02_dedge_quant.py` from the raw images. That file is not included; what ships is a
single-field QC export (`derived/per_cell_dedge_QC.csv`).

Note what this script draws: the **nine-box** figure — three knockdown conditions x (no BMP4 edge,
BMP4 edge, BMP4 interior) — which is the Supplementary Data 4 display, **not** ED Fig 3k. ED 3k is
the four-box panel (SCRAMBLE KD and BMPR1A KD, each without and with BMP4), and its twelve per-well
values ship as `derived/negctrl_summary_per_field_30min.csv` and are plotted by
`08_ed3k_panel.py`, which runs as shipped. See the lane README.
"""
import os, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams["pdf.fonttype"] = 42  # embed editable TrueType text in PDFs (Nature-friendly)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]  # global convention: Arial

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB  = os.path.join(WORK, "results", "tables")
FIG  = os.path.join(WORK, "results", "figures")
CONDS   = ["no_KD", "SCRAMBLE_KD", "BMPR1A_KD"]
DISPLAY = {"no_KD": "no KD", "SCRAMBLE_KD": "scramble KD", "BMPR1A_KD": "BMPR1A KD"}
COLORS  = {"no_KD": "#555555", "SCRAMBLE_KD": "#2a9d5c", "BMPR1A_KD": "#d23b3b"}


def field_medians(df, cond, tp, edge, value):
    """Per-field median of `value` for cells of one condition/timepoint/region."""
    g = df[(df.condition == cond) & (df.timepoint == tp)]
    g = g[g.is_edge] if edge else g[~g.is_edge]
    v = g.groupby("field")[value].median().values
    return v[~np.isnan(v)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timepoint", default="30min", help="BMP-stimulated timepoint")
    ap.add_argument("--baseline", default="no_BMP", help="unstimulated negative-control timepoint")
    ap.add_argument("--edge_um", type=float, default=25.0)
    ap.add_argument("--conc", type=float, default=10.0, help="BMP4 conc (ng/mL) at the stimulated timepoint")
    ap.add_argument("--value", default="psmad_bgsub")
    ap.add_argument("--csv", default=os.path.join(TAB, "per_cell_dedge.csv"))
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True); np.random.seed(0)

    df = pd.read_csv(args.csv)
    df["is_edge"] = df.dedge_um < args.edge_um
    BASE, STIM = args.baseline, args.timepoint

    # the three measures per condition: two negatives flanking the response
    C = f"{args.conc:g} ng/mL BMP4"
    cats = [("0 ng/mL BMP4, edge cells", BASE, True,  "#cfcfcf"),   # stimulus-negative (no BMP)
            (f"{C}, edge cells",         STIM, True,  "#2171b5"),   # the BMP4 response
            (f"{C}, interior cells",     STIM, False, "#8a8a8a")]   # within-well floor

    # ---- per-field long table (provenance / the plotted numbers) ----
    rows = []
    for c in CONDS:
        for lab, tp, edge, _ in cats:
            g = df[(df.condition == c) & (df.timepoint == tp)]
            g = g[g.is_edge] if edge else g[~g.is_edge]
            for fld, m in g.groupby("field")[args.value].median().items():
                rows.append(dict(condition=c, category=lab, field=fld, pSMAD_med=round(float(m), 1)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(TAB, f"negctrl_summary_per_field_{STIM}.csv"), index=False)

    # ---- Panel A: response + 2 negative controls, grouped by condition ----
    fig, ax = plt.subplots(figsize=(7.6, 5))
    width = 0.26
    for ci, c in enumerate(CONDS):
        for k, (lab, tp, edge, col) in enumerate(cats):
            x = ci + (k - 1) * width
            v = field_medians(df, c, tp, edge, args.value)
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
              frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"edge_pSMAD_by_condition_{STIM}.png"), dpi=130)
    fig.savefig(os.path.join(FIG, f"edge_pSMAD_by_condition_{STIM}.pdf"), dpi=300)  # 300-dpi vector PDF
    plt.close(fig)

    # ---- Panel B: pSMAD vs d_edge profile (stimulated timepoint) ----
    d = df[df.timepoint == STIM]
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.arange(-10, 160, 10); ctr = (bins[:-1] + bins[1:]) / 2
    for c in CONDS:
        g = d[d.condition == c]
        med = [g[(g.dedge_um >= lo) & (g.dedge_um < hi)][args.value].median() for lo, hi in zip(bins[:-1], bins[1:])]
        ax.plot(ctr, med, color=COLORS[c], label=DISPLAY[c], lw=2)
    ax.axvline(args.edge_um, ls=":", color="gray", lw=1)
    ax.set_xlabel("distance from colony edge, d_edge (um)")
    ax.set_ylabel("pSMAD1/5/9 (a.u.)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"pSMAD_vs_dedge_profile_{STIM}.png"), dpi=130); plt.close(fig)

    # ---- descriptives (n = fields) ----
    print(f"== per-field median pSMAD ({args.value}); n=fields ==")
    for c in CONDS:
        print(f"  {DISPLAY[c]}")
        for lab, tp, edge, _ in cats:
            v = field_medians(df, c, tp, edge, args.value)
            print(f"     {lab:24s} med-of-fields={np.median(v):6.0f}  per-field={np.round(v).astype(int).tolist()}")
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
