#!/usr/bin/env python3
"""m4-ii / m5-iii (Referee #1) — bilaterality (symmetry) index, co-graft vs the BMP4-bead reference
(Fig-3f style). Same index + R1 method as Fig 3f (`M.project_points_to_centerline` -> 1-|R-L|/(R+L),
0 = unilateral / all one side, 1 = symmetric), computed on the SAME per-organoid segmentation.
Strip plot (points + median + IQR). Deterministic. Descriptive/neutral.

  one-graft -> {marker}_bilaterality_mean, 05_per_organoid_metrics_newog.tsv (20) + _onegraft.tsv (5)
  two-graft -> {marker}_bilaterality_mean, 05_per_organoid_metrics.tsv (cograft representatives)
  bead      -> {marker}_bilaterality_index_mean, Fig-3c/3f handoff, the 43-morph position subset, plotted per position

  Two markers: MESP2 (somite = axial) and FOXF1 (LPM = mediolateral). Grafts carry no PAX8.
  The co-graft LPM is expected UNILATERAL (the donor graft sits on one side -> one-sided induction),
  vs the bilateral bead LPM — the point of the panel.

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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
RNG = np.random.default_rng(7)

WORK = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
DA = Path("<analysis-root>/")
TAB = WORK / "results" / "tables"
FIG = WORK / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

BEAD_TABLES = {"2026-01-02": DA / "bilateral_plot" / "2026-01-02" / "05b_day5_feature_handoff_table.tsv",
               "2026-02-27": DA / "bilateral_plot" / "2026-02-27" / "05b_day5_feature_handoff_table.tsv"}
# Fig-3f bead POSITIONS (split the bead by condition). IDs per bilateral_plot/plot_bilaterality_with_stats.py.
BEAD_POS = {
    "unilateral": {"2026-01-02": ["30", "3202", "24", "39", "01", "35", "14", "15", "38", "21", "09", "12", "07"],
                   "2026-02-27": ["04", "05", "06", "09", "20", "22"]},
    "medial":     {"2026-01-02": ["29", "17", "33", "28", "27", "05", "04", "02", "18", "11"],
                   "2026-02-27": ["07", "08", "11", "12", "15", "17", "18"]},
    "bilateral":  {"2026-01-02": ["06", "20", "31", "10", "34"], "2026-02-27": ["19", "13"]},
}


def _b(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def onegraft(marker: str) -> np.ndarray:
    col = f"{marker}_bilaterality_mean"
    new = pd.read_csv(TAB / "05_per_organoid_metrics_newog.tsv", sep="\t")
    orig = pd.read_csv(TAB / "05_per_organoid_metrics_onegraft.tsv", sep="\t")
    orig = orig[orig["condition"].isin(["onegraft_ctrl", "onegraft_pax8"])]
    return pd.concat([new[col], orig[col]]).to_numpy(float)


def twograft(marker: str) -> np.ndarray:
    reps = pd.read_csv(WORK / "annotations" / "two_graft_morphs.tsv", sep="\t")
    repset = set(reps[_b(reps["representative"])]["image_id"])
    cg = pd.read_csv(TAB / "05_per_organoid_metrics.tsv", sep="\t")
    cg = cg[(cg["condition"] == "cograft") & (cg["image_id"].isin(repset))]
    return cg[f"{marker}_bilaterality_mean"].to_numpy(float)


def bead_pos(marker: str, position: str) -> np.ndarray:
    """Bilaterality for one Fig-3f bead position (unilateral / medial / bilateral)."""
    col = f"{marker}_bilaterality_index_mean"
    out = []
    for date, path in BEAD_TABLES.items():
        df = pd.read_csv(path, sep="\t", dtype={"trunk_morph_id": str})
        ids = set(BEAD_POS[position].get(date, []))
        out.append(df[df["trunk_morph_id"].isin(ids)][col].to_numpy(float))
    return np.concatenate(out) if out else np.array([])


def strip(ax, x, vals, color):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    jit = (RNG.random(vals.size) - 0.5) * 0.30
    ax.scatter(np.full(vals.size, x) + jit, vals, s=26, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.4, zorder=3)
    med = float(np.median(vals))
    ax.hlines(med, x - 0.32, x + 0.32, color=color, lw=2.6, zorder=4)
    q1, q3 = np.percentile(vals, [25, 75])
    ax.vlines(x, q1, q3, color=color, lw=1.5, alpha=0.9, zorder=2)
    return vals.size, med


# split the BMP4-bead into its three Fig-3f positions (unilateral / medial / bilateral).
# two-graft kept but flagged pre-audit (axes not re-tuned, FOXF1 still global threshold — pending audit).
GROUPS = [("one-graft\n(LPM)", lambda m: onegraft(m), "#e6844d"),
          ("two-graft\n(LPM+NMP)", lambda m: twograft(m), "#c0552b"),
          ("bead\nunilateral", lambda m: bead_pos(m, "unilateral"), "#a9c7e0"),
          ("bead\nmedial", lambda m: bead_pos(m, "medial"), "#5b8bbf"),
          ("bead\nbilateral", lambda m: bead_pos(m, "bilateral"), "#2c5985")]
PANELS = [("MESP2 (somite)", "mesp2"), ("FOXF1 (LPM)", "foxf1")]

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharey=True)
for ax, (ptitle, marker) in zip(axes, PANELS):
    labs = []
    for x, (lab, loader, col) in enumerate(GROUPS):
        n, med = strip(ax, x, loader(marker), col)
        labs.append(f"{lab}\n(n={n})")
    ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(labs, fontsize=7.8)
    ax.set_title(ptitle, fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.axvline(1.5, color="0.85", lw=0.8, ls="--", zorder=0)   # co-graft | bead divider
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("bilaterality index\n(0 = unilateral   ·   1 = symmetric)", fontsize=9.5)
fig.tight_layout()
out = FIG / "07d_m4ii_bilaterality.png"
fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)

for ptitle, marker in PANELS:
    print(f"[{ptitle}]")
    for lab, loader, _ in GROUPS:
        v = loader(marker); v = v[np.isfinite(v)]
        print(f"  {lab.splitlines()[0]:10s} n={v.size:2d}  median={np.median(v):.2f}  "
              f"IQR=[{np.percentile(v,25):.2f},{np.percentile(v,75):.2f}]")
print("wrote", out)
