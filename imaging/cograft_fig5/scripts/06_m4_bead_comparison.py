#!/usr/bin/env python3
"""m4: co-graft vs BMP4-bead matched-metric comparison (Referee #1).

Bead baseline = the EXACT dataset behind Fig 3c/3f: the day-5 pax8 trunk morphs, BOTH
batches (2026-01-02 + 2026-02-27), restricted to the bead-position-grouped subset that
the canonical Fig-3f script plots (bilateral_plot/plot_bilaterality_with_stats.py). That
subset = 43 morphs across three bead positions (unilateral / medial / bilateral).

Comparison rules:
  - morph length (m4-iii)                : bead pooled over all 3 positions  vs co-graft
  - LPM (FOXF1) bilateral symmetry (m4-ii, m5-iii, "like Fig 3f"): the 3 bead positions
                                           EACH vs co-graft (co-graft = a 4th condition)
  - FOXF1 & MESP2 co-presence (m4-i, m5-ii): bead pooled over all 3 positions vs co-graft

Co-graft numbers -> this module's 05_per_organoid_metrics.tsv (cograft rows).
No PAX8 in the co-graft set -> co-presence scored on FOXF1+MESP2. Descriptive only:
no interpretive annotation of the differences. Deterministic (fixed jitter seed).

NOT runnable as shipped. This stage reads the raw .czi acquisitions, which are not deposited —
they are available from the lead contact on request, per the paper's Data Availability
Statement. The measurements it produces are committed under the lane's `derived/` directory,
which is what the figure scripts read. See `data/DOWNLOAD.md`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
DA = Path("<analysis-root>/")
CG_TABLE = WORK / "results" / "tables" / "05_per_organoid_metrics.tsv"
BEAD_TABLES = {
    "2026-01-02": DA / "bilateral_plot" / "2026-01-02" / "05b_day5_feature_handoff_table.tsv",
    "2026-02-27": DA / "bilateral_plot" / "2026-02-27" / "05b_day5_feature_handoff_table.tsv",
}
FIGDIR = WORK / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Fig-3f bead-position group membership (verbatim from plot_bilaterality_with_stats.py).
PLOT_IDS = {
    "unilateral bead": {
        "2026-01-02": ["30", "3202", "24", "39", "01", "35", "14", "15", "38", "21", "09", "12", "07"],
        "2026-02-27": ["04", "05", "06", "09", "20", "22"],
    },
    "medial bead": {
        "2026-01-02": ["29", "17", "33", "28", "27", "05", "04", "02", "18", "11"],
        "2026-02-27": ["07", "08", "11", "12", "15", "17", "18"],
    },
    "bilateral bead": {
        "2026-01-02": ["06", "20", "31", "10", "34"],
        "2026-02-27": ["19", "13"],
    },
}
BEAD_ORDER = ["unilateral bead", "medial bead", "bilateral bead"]

PRESENCE_MIN_FRACTION = 0.01
C_COGRAFT = "#e6844d"                                   # co-graft = orange (the new condition)
C_BEAD_POOL = "#5b8bbf"                                 # bead pooled = blue
C_BEAD_POS = {"unilateral bead": "#a9c4e0", "medial bead": "#5b8bbf", "bilateral bead": "#2f5f8f"}
RNG = np.random.default_rng(7)


def _coerce_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load_bead() -> pd.DataFrame:
    frames = []
    for exp, path in BEAD_TABLES.items():
        d = pd.read_csv(path, sep="\t", dtype={"trunk_morph_id": str})
        d["experiment"] = exp
        frames.append(d)
    allrows = pd.concat(frames, ignore_index=True)
    recs = []
    for group, exps in PLOT_IDS.items():
        for exp, ids in exps.items():
            for tid in ids:
                m = allrows[(allrows["experiment"] == exp) & (allrows["trunk_morph_id"] == tid)]
                if len(m) != 1:
                    raise RuntimeError(f"{group} {exp} id={tid}: found {len(m)} rows (expected 1)")
                r = m.iloc[0]
                recs.append(dict(
                    group=group, experiment=exp, tid=tid,
                    length=float(r["consensus_axis_length_um"]),
                    foxf1_bilat=float(r["foxf1_bilaterality_index_mean"]),
                    foxf1_frac=float(r["foxf1_fraction_mean"]),
                    mesp2_frac=float(r["mesp2_fraction_mean"]),
                ))
    bead = pd.DataFrame(recs)
    bead["copresent"] = (bead["foxf1_frac"] > PRESENCE_MIN_FRACTION) & (bead["mesp2_frac"] > PRESENCE_MIN_FRACTION)
    return bead


def load_cograft() -> pd.DataFrame:
    cg = pd.read_csv(CG_TABLE, sep="\t")
    cg = cg[cg["condition"] == "cograft"].copy()
    cg["copresent"] = (cg["foxf1_fraction_mean"] > PRESENCE_MIN_FRACTION) & \
                      (cg["mesp2_fraction_mean"] > PRESENCE_MIN_FRACTION)
    return cg


def _strip(ax, x, vals, color, box=False):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    if box:
        bp = ax.boxplot([vals], positions=[x], widths=0.5, showfliers=False, patch_artist=True)
        bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.30)
        for w in bp["whiskers"] + bp["caps"]: w.set_color("#555")
        bp["medians"][0].set_color("black"); bp["medians"][0].set_linewidth(2.0)
    jit = (RNG.random(vals.size) - 0.5) * 0.30
    ax.scatter(np.full(vals.size, x) + jit, vals, s=24, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.4, zorder=3)
    if not box:
        med = float(np.median(vals))
        ax.hlines(med, x - 0.32, x + 0.32, color=color, lw=2.4, zorder=4)
        q1, q3 = np.percentile(vals, [25, 75])
        ax.vlines(x, q1, q3, color=color, lw=1.4, alpha=0.9, zorder=2)
    return float(np.median(vals))


def main():
    bead = load_bead()
    cg = load_cograft()
    cg_len = cg["length_um"].to_numpy(float)
    cg_bilat = cg["foxf1_bilaterality_mean"].to_numpy(float)
    n_cg = len(cg)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    # (iii) morph length — bead pooled over all positions vs co-graft
    ax = axes[0]
    mb = _strip(ax, 0, bead["length"], C_BEAD_POOL)
    mc = _strip(ax, 1, cg_len, C_COGRAFT)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"bead\n(n={len(bead)})", f"co-graft\n(n={n_cg})"])
    ax.set_ylabel("Morph length (µm)"); ax.set_title("Morph length", fontsize=11)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, max(bead["length"].max(), cg_len.max()) * 1.12)

    # (ii) LPM (FOXF1) bilateral symmetry — 3 bead positions each + co-graft (4th)
    ax = axes[1]
    xs, xlabels, bilat_meds = [], [], {}
    for i, g in enumerate(BEAD_ORDER):
        v = bead.loc[bead["group"] == g, "foxf1_bilat"].to_numpy(float)
        bilat_meds[g] = _strip(ax, i, v, C_BEAD_POS[g], box=True)
        xs.append(i); xlabels.append(g.replace(" bead", "\nbead") + f"\n(n={len(v)})")
    bilat_meds["co-graft"] = _strip(ax, 3, cg_bilat, C_COGRAFT, box=True)
    xs.append(3); xlabels.append(f"co-graft\n(n={n_cg})")
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("LPM bilateral symmetry index"); ax.set_title("LPM (FOXF1) bilateral symmetry", fontsize=11)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 1.02)

    # (i) FOXF1+ & MESP2+ co-presence — bead pooled vs co-graft
    ax = axes[2]
    fb = 100.0 * bead["copresent"].mean(); fc = 100.0 * cg["copresent"].mean()
    ax.bar([0, 1], [fb, fc], width=0.6, color=[C_BEAD_POOL, C_COGRAFT], edgecolor="black", linewidth=0.6)
    ax.text(0, fb + 1.5, f"{int(bead['copresent'].sum())}/{len(bead)}", ha="center", fontsize=9)
    ax.text(1, fc + 1.5, f"{int(cg['copresent'].sum())}/{n_cg}", ha="center", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"bead\n(n={len(bead)})", f"co-graft\n(n={n_cg})"])
    ax.set_ylabel("Morphs with FOXF1⁺ & MESP2⁺ (%)"); ax.set_title("FOXF1⁺ & MESP2⁺ co-presence", fontsize=11)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 112)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGDIR / "06_m4_cograft_vs_bead.png"
    fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)

    print(f"BEAD dataset (Fig-3f, both batches): n={len(bead)}  "
          f"[{', '.join(f'{g}={int((bead.group==g).sum())}' for g in BEAD_ORDER)}]")
    print(f"  length_med={np.median(bead['length']):.0f}  copresent={int(bead['copresent'].sum())}/{len(bead)}")
    for g in BEAD_ORDER:
        print(f"  FOXF1 bilat median [{g}] = {bilat_meds[g]:.3f}")
    print(f"COGRAFT n={n_cg}  length_med={np.median(cg_len):.0f}  "
          f"FOXF1 bilat median={bilat_meds['co-graft']:.3f}  copresent={int(cg['copresent'].sum())}/{n_cg}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
