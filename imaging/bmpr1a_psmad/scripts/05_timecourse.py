#!/usr/bin/env python
"""05 - BMP4 timecourse of edge vs interior pSMAD1/5/9, per well, all conditions.

Rolls the per-well readout across ALL timepoints (no_BMP / 15min / 30min / 45min),
so the 30-min headline sits in the context of the full BMP4 timecourse. Unit of
replication = well (one 10x field/well); n = 3 wells/condition/timepoint, except no_KD at 45 min,
which has 4 (`derived/timecourse_per_well.csv` is 74 rows, not 72). The script takes n from the
data, so the 45-min comparison is n=4 vs n=3.

Reads results/tables/per_cell_dedge.csv (from 02). Writes:
  results/tables/timecourse_per_well.csv        (tidy: timepoint,condition,region,field,pSMAD_med)
  results/tables/timecourse_median_of_wells.csv (timepoint,condition,region,median,mean,sd,n)
  results/figures/pSMAD_edge_interior_timecourse_v<VERSION>.png  (the timecourse figure)
Prints the edge and interior timecourse (median-of-wells) + the BMPR1A_KD-vs-control
edge difference at each stimulated timepoint (with the exact two-sided Mann-Whitney p).
NO fold/ratio (psmad_bgsub subtracts pure image background, not a cellular baseline);
the difference cancels the shared background.

Also renders the timecourse figure: edge (solid) + interior (dashed)
pSMAD1/5/9 vs time, one colored line per condition, individual wells as points.
Versioned per the figure-iteration convention (bump VERSION; never overwrite a look).
Env: psmad_quant.
Run: /opt/anaconda3/bin/conda run -n psmad_quant python scripts/05_timecourse.py [--edge_um 25]

NOT runnable as shipped. This reads `results/tables/per_cell_dedge.csv`, the full per-cell table
produced by `02_dedge_quant.py` from the raw images. That file is not included; what ships is a
single-field QC export (`derived/per_cell_dedge_QC.csv`).

This script draws `pSMAD_edge_interior_timecourse_v{VERSION}.png`, not the nine-box Supplementary
Data 4 display and not ED Fig 3k. The nine-box figure is drawn by `03_figure.py` and
`07_panel_figure.py`; ED 3k is the four-box panel
(SCRAMBLE KD and BMPR1A KD, each without and with BMP4), drawn by `08_ed3k_panel.py`, and its
twelve per-well values ship as `derived/negctrl_summary_per_field_30min.csv`. See the lane README.
"""
import os, argparse
import numpy as np, pandas as pd
from scipy import stats

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB  = os.path.join(WORK, "results", "tables")
CONDS   = ["no_KD", "SCRAMBLE_KD", "BMPR1A_KD"]
DISPLAY = {"no_KD": "no KD", "SCRAMBLE_KD": "scramble KD", "BMPR1A_KD": "BMPR1A KD"}
ORDER   = ["no_BMP", "15min", "30min", "45min"]
VERSION = "1"   # bump on each figure-style iteration; never overwrite a prior render


def per_well(df, cond, tp, edge, value):
    """The per-well medians for one condition/timepoint/region (edge or interior)."""
    g = df[(df.condition == cond) & (df.timepoint == tp)]
    g = g[g.is_edge] if edge else g[~g.is_edge]
    v = g.groupby("field")[value].median().values
    return v[~np.isnan(v)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_um", type=float, default=25.0)
    ap.add_argument("--value", default="psmad_bgsub")
    ap.add_argument("--csv", default=os.path.join(TAB, "per_cell_dedge.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["is_edge"] = df.dedge_um < args.edge_um
    present = list(pd.unique(df.timepoint))
    tps = [t for t in ORDER if t in present] + [t for t in present if t not in ORDER]
    print(f"[timepoints present] {tps}\n")

    rows, summ = [], []
    for tp in tps:
        for c in CONDS:
            for region, edge in [("edge", True), ("interior", False)]:
                g = df[(df.condition == c) & (df.timepoint == tp)]
                g = g[g.is_edge] if edge else g[~g.is_edge]
                med = g.groupby("field")[args.value].median()
                for fld, m in med.items():
                    rows.append(dict(timepoint=tp, condition=c, region=region,
                                     field=fld, pSMAD_med=round(float(m), 1)))
                v = med.values; v = v[~np.isnan(v)]
                if len(v):
                    summ.append(dict(timepoint=tp, condition=c, region=region,
                                     median=round(float(np.median(v)), 1),
                                     mean=round(float(np.mean(v)), 1),
                                     sd=round(float(np.std(v, ddof=1)), 1) if len(v) > 1 else np.nan,
                                     n=len(v)))
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "timecourse_per_well.csv"), index=False)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(TAB, "timecourse_median_of_wells.csv"), index=False)

    def mow(tp, c, region):
        r = S[(S.timepoint == tp) & (S.condition == c) & (S.region == region)]
        return float(r["median"].iloc[0]) if len(r) else np.nan

    print("== edge-cell pSMAD1/5/9 (median-of-wells, a.u.) across the BMP4 timecourse ==")
    print(f"  {'condition':12s} " + "  ".join(f"{t:>8s}" for t in tps))
    for c in CONDS:
        print(f"  {DISPLAY[c]:12s} " + "  ".join(f"{mow(t, c, 'edge'):8.0f}" for t in tps))
    print("\n== interior-cell pSMAD1/5/9 (median-of-wells, a.u.) - within-well floor ==")
    print(f"  {'condition':12s} " + "  ".join(f"{t:>8s}" for t in tps))
    for c in CONDS:
        print(f"  {DISPLAY[c]:12s} " + "  ".join(f"{mow(t, c, 'interior'):8.0f}" for t in tps))

    print("\n== BMPR1A_KD edge vs each control at each stimulated timepoint (median-of-wells) ==")
    for tp in [t for t in tps if t != "no_BMP"]:
        kd = per_well(df, "BMPR1A_KD", tp, True, args.value)
        for c in ["no_KD", "SCRAMBLE_KD"]:
            ct = per_well(df, c, tp, True, args.value)
            if len(kd) and len(ct):
                U, p = stats.mannwhitneyu(kd, ct, alternative="two-sided", method="exact")
                sep = "sep" if (max(kd) < min(ct) or max(ct) < min(kd)) else "OVERLAP"
                print(f"  {tp:>7s}  BMPR1A_KD {np.median(kd):5.0f} vs {DISPLAY[c]:12s} {np.median(ct):5.0f}:  "
                      f"diff {np.median(ct)-np.median(kd):5.0f} a.u.   MW p={p:.3f} [{sep}]")

    # ---- timecourse figure: edge (solid) + interior (dashed), one line per condition ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams["pdf.fonttype"] = 42  # embed editable TrueType text in PDFs
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]  # global convention: Arial
    FIG = os.path.join(WORK, "results", "figures"); os.makedirs(FIG, exist_ok=True)
    COLORS = {"no_KD": "#555555", "SCRAMBLE_KD": "#2a9d5c", "BMPR1A_KD": "#d23b3b"}
    XT = {"no_BMP": 0, "15min": 15, "30min": 30, "45min": 45}
    xtps = [t for t in tps if t in XT]; xs = [XT[t] for t in xtps]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.subplots_adjust(left=0.11, right=0.70, bottom=0.13, top=0.97)
    for c in CONDS:
        ax.plot(xs, [mow(t, c, "edge") for t in xtps], "-", color=COLORS[c], lw=2.2, zorder=3)
        ax.plot(xs, [mow(t, c, "interior") for t in xtps], "--", color=COLORS[c], lw=1.3, alpha=0.5, zorder=2)
        for t, x in zip(xtps, xs):
            ve = per_well(df, c, t, True, args.value); vi = per_well(df, c, t, False, args.value)
            ax.scatter([x] * len(ve), ve, color=COLORS[c], s=18, zorder=4, edgecolor="k", linewidth=0.3)
            ax.scatter([x] * len(vi), vi, facecolor="white", edgecolor=COLORS[c], s=14, zorder=3, linewidth=0.8, alpha=0.75)
    ax.set_xlabel("time after BMP4 addition (min)"); ax.set_ylabel("pSMAD1/5/9 (a.u.)")
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.set_ylim(bottom=0); ax.margins(x=0.06)
    cond_h = [Line2D([0], [0], color=COLORS[c], lw=2.4, label=DISPLAY[c]) for c in CONDS]
    reg_h  = [Line2D([0], [0], color="#333", lw=2.2, ls="-", label="edge cells"),
              Line2D([0], [0], color="#333", lw=1.3, ls="--", alpha=0.6, label="interior cells")]
    leg1 = ax.legend(handles=cond_h, frameon=False, fontsize=9, loc="upper left",
                     bbox_to_anchor=(1.02, 1.0), title="condition"); leg1.get_title().set_fontsize(9)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=reg_h, frameon=False, fontsize=9, loc="upper left",
                     bbox_to_anchor=(1.02, 0.52), title="region"); leg2.get_title().set_fontsize(9)
    figpath = os.path.join(FIG, f"pSMAD_edge_interior_timecourse_v{VERSION}.png")
    fig.savefig(figpath, dpi=150); fig.savefig(figpath.replace(".png", ".pdf"), dpi=300); plt.close(fig)

    print(f"\ntables -> {TAB}/timecourse_per_well.csv, timecourse_median_of_wells.csv")
    print(f"figure -> {figpath}")


if __name__ == "__main__":
    main()
