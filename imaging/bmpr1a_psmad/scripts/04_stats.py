#!/usr/bin/env python
"""04 - n=3 statistics for the headline figure (edge pSMAD1/5/9 by condition, 30 min).

The unit of replication is the WELL (one 10x field per well); n = 3 wells / condition.
Per-well medians are recomputed from per_cell_dedge.csv with the SAME logic 03_figure.py
uses (edge = dedge_um < edge_um) and cross-checked against the plotted numbers in
negctrl_summary_per_field_<timepoint>.csv, so the stats provably describe what the figure shows.

Reports, for the stimulated timepoint (default 30 min):
  - per condition x region: the 3 per-well medians, median-of-wells, mean +/- s.d., range
  - BMP4 response at the edge (edge 10 vs its two floors: no-BMP edge, +BMP interior) as
    absolute medians + their difference. NO ratio/fold: psmad_bgsub subtracts pure image
    background, not a cellular baseline, so a ratio has no principled denominator; the
    difference cancels the shared background.
  - cross-condition edge response (BMPR1A_KD edge 10 vs no_KD / SCRAMBLE edge 10): median + difference
  - within-BMPR1A_KD: edge 10 vs its no-BMP edge and its interior (is the response gone?)
  - REPORTED TEST: Welch's two-sided t-test on the per-well medians,
    BMPR1A_KD vs each control (uses magnitude, keeps the well as the replicate unit;
    p < 0.001 for each). Mann-Whitney is printed ONLY to show its n=3 rank FLOOR
    (two-sided min = 0.10 regardless of effect size) - not reported. Naive per-cell
    testing = pseudoreplication (p ~ 0, invalid: cells within a well aren't independent).

Env: psmad_quant.
Run: /opt/anaconda3/bin/conda run -n psmad_quant python scripts/04_stats.py [--timepoint 30min] [--edge_um 25]

NOT runnable as shipped. This reads `results/tables/per_cell_dedge.csv`, the full per-cell table
produced by `02_dedge_quant.py` from the raw images. That file is not included; what ships is a
single-field QC export (`derived/per_cell_dedge_QC.csv`).

This script draws nothing; it prints statistics. They describe the **nine-box** grouping — three
knockdown conditions x (no BMP4 edge, BMP4 edge, BMP4 interior) — which is the Supplementary Data 4
display, **not** ED Fig 3k. ED 3k is the four-box panel (SCRAMBLE KD and BMPR1A KD, each without and
with BMP4), and its twelve per-well values ship as `derived/negctrl_summary_per_field_30min.csv`.
The nine-box figure itself is drawn by `03_figure.py` and `07_panel_figure.py`. See the lane README.
"""
import os, argparse
import numpy as np, pandas as pd
from scipy import stats

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB  = os.path.join(WORK, "results", "tables")
CONDS   = ["no_KD", "SCRAMBLE_KD", "BMPR1A_KD"]
DISPLAY = {"no_KD": "no KD", "SCRAMBLE_KD": "scramble KD", "BMPR1A_KD": "BMPR1A KD"}


def per_well(df, cond, tp, edge, value):
    """The 3 per-well medians for one condition/timepoint/region (edge or interior)."""
    g = df[(df.condition == cond) & (df.timepoint == tp)]
    g = g[g.is_edge] if edge else g[~g.is_edge]
    v = g.groupby("field")[value].median().values
    return v[~np.isnan(v)]


def desc(v):
    return dict(median=np.median(v), mean=np.mean(v), sd=np.std(v, ddof=1),
               lo=np.min(v), hi=np.max(v), n=len(v))


def line(label, v):
    d = desc(v)
    vals = ", ".join(f"{x:.0f}" for x in v)
    print(f"    {label:34s} n={d['n']}  median={d['median']:6.0f}  "
          f"mean={d['mean']:6.0f} +/- {d['sd']:4.0f} (s.d.)  range [{d['lo']:.0f}, {d['hi']:.0f}]  "
          f"wells: [{vals}]")


def tests(a, b, la, lb):
    """Two-sided exact MW-U + Welch t; return a one-line string. Descriptive-only decision aid."""
    U, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact")
    t, p_t  = stats.ttest_ind(a, b, equal_var=False)
    sep = "complete separation" if (max(a) < min(b) or max(b) < min(a)) else "overlap"
    return (f"    {la} vs {lb}:  Mann-Whitney U (exact, two-sided) p = {p_mw:.3f}  "
            f"[{sep}; n1=n2=3 floor = 0.100]   Welch t p = {p_t:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timepoint", default="30min")
    ap.add_argument("--baseline", default="no_BMP")
    ap.add_argument("--edge_um", type=float, default=25.0)
    ap.add_argument("--value", default="psmad_bgsub")
    ap.add_argument("--csv", default=os.path.join(TAB, "per_cell_dedge.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["is_edge"] = df.dedge_um < args.edge_um
    BASE, STIM = args.baseline, args.timepoint

    # --- cross-check per-well medians vs the plotted summary (provenance guarantee) ---
    summ = pd.read_csv(os.path.join(TAB, f"negctrl_summary_per_field_{args.timepoint}.csv"))
    recomputed = {}
    for c in CONDS:
        recomputed[(c, "edge0")] = per_well(df, c, BASE, True,  args.value)
        recomputed[(c, "edge1")] = per_well(df, c, STIM, True,  args.value)
        recomputed[(c, "int1")]  = per_well(df, c, STIM, False, args.value)
    catmap = {"edge0": "0 ng/mL BMP4, edge cells",
              "edge1": "10 ng/mL BMP4, edge cells",
              "int1":  "10 ng/mL BMP4, interior cells"}
    mism = 0
    for (c, k), v in recomputed.items():
        plotted = summ[(summ.condition == c) & (summ.category == catmap[k])].pSMAD_med.values
        if not np.allclose(np.sort(np.round(v, 1)), np.sort(plotted), atol=0.15):
            mism += 1
            print(f"  !! MISMATCH {c}/{k}: recomputed {np.round(v,1)} vs plotted {plotted}")
    print(f"[cross-check] per-well medians vs negctrl_summary_per_field.csv: "
          f"{'OK - identical to the plotted numbers' if mism == 0 else f'{mism} MISMATCHES'}\n")

    # --- descriptives, per condition ---
    print(f"== per-well median pSMAD1/5/9 ({args.value}); unit = well; {STIM} timepoint ==")
    for c in CONDS:
        print(f"  {DISPLAY[c]}")
        line("0 ng/mL BMP4, edge",     recomputed[(c, "edge0")])
        line("10 ng/mL BMP4, edge",    recomputed[(c, "edge1")])
        line("10 ng/mL BMP4, interior",recomputed[(c, "int1")])

    # --- BMP4 response at the edge: absolute medians + difference over the two floors ---
    # NO fold/ratio: bgsub removes pure image background, not a cellular baseline, so a ratio
    # has no principled denominator. The DIFFERENCE cancels the shared background.
    print("\n== BMP4 response at the edge (median-of-wells, a.u.); difference over floors ==")
    for c in CONDS:
        e1 = np.median(recomputed[(c, "edge1")])
        e0 = np.median(recomputed[(c, "edge0")])
        it = np.median(recomputed[(c, "int1")])
        print(f"  {DISPLAY[c]:12s}  edge+BMP={e1:6.0f}   no-BMP edge={e0:5.0f} (diff {e1-e0:+5.0f})   "
              f"interior={it:5.0f} (diff {e1-it:+5.0f})")

    # --- cross-condition edge response: absolute medians + difference (no ratio) ---
    print("\n== edge response (10 ng/mL) - BMPR1A_KD vs each control (median-of-wells, a.u.) ==")
    kd = np.median(recomputed[("BMPR1A_KD", "edge1")])
    for c in ["no_KD", "SCRAMBLE_KD"]:
        ctrl = np.median(recomputed[(c, "edge1")])
        print(f"  BMPR1A_KD {kd:.0f} vs {DISPLAY[c]} {ctrl:.0f}:  difference {ctrl-kd:.0f} a.u.")

    # --- formal tests on the key comparisons (DECISION AID; caption may stay descriptive) ---
    print("\n== significance: Welch t on per-well medians = REPORTED (KD vs each control); MW = rank-floor demo only ==")
    print(tests(recomputed[("BMPR1A_KD","edge1")], recomputed[("no_KD","edge1")],
                "BMPR1A_KD edge", "no_KD edge"))
    print(tests(recomputed[("BMPR1A_KD","edge1")], recomputed[("SCRAMBLE_KD","edge1")],
                "BMPR1A_KD edge", "scramble edge"))
    # within-BMPR1A_KD: is the response gone?
    print(tests(recomputed[("BMPR1A_KD","edge1")], recomputed[("BMPR1A_KD","edge0")],
                "BMPR1A_KD edge+BMP", "BMPR1A_KD no-BMP edge"))
    print(tests(recomputed[("no_KD","edge1")], recomputed[("no_KD","edge0")],
                "no_KD edge+BMP", "no_KD no-BMP edge"))


if __name__ == "__main__":
    main()
