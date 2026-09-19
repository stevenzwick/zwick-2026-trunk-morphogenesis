#!/usr/bin/env python3
"""m4-iii (Referee #1) — morph length (µm) for one-graft / two-graft vs the BMP4-bead reference.
Strip plot (points + median + IQR). Deterministic. Descriptive/neutral (no narration of differences).

  one-graft -> length_um from 05_per_organoid_metrics_newog.tsv (20 new) + _onegraft.tsv (5 ctrl+PAX8)
  two-graft -> length_um from 05_per_organoid_metrics.tsv (cograft, independent representatives)
  bead      -> consensus_axis_length_um, Fig-3c/3f handoff, ALL morphs (n=67), positions pooled

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

BEAD_TABLES = {
    "2026-01-02": DA / "bilateral_plot" / "2026-01-02" / "05b_day5_feature_handoff_table.tsv",
    "2026-02-27": DA / "bilateral_plot" / "2026-02-27" / "05b_day5_feature_handoff_table.tsv",
}
PLOT_IDS = {
    "unilateral bead": {"2026-01-02": ["30", "3202", "24", "39", "01", "35", "14", "15", "38", "21", "09", "12", "07"],
                        "2026-02-27": ["04", "05", "06", "09", "20", "22"]},
    "medial bead":     {"2026-01-02": ["29", "17", "33", "28", "27", "05", "04", "02", "18", "11"],
                        "2026-02-27": ["07", "08", "11", "12", "15", "17", "18"]},
    "bilateral bead":  {"2026-01-02": ["06", "20", "31", "10", "34"], "2026-02-27": ["19", "13"]},
}


def _b(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def bead_length() -> np.ndarray:
    """FULL day-5 BMP4-bead trunk-morph cohort — BOTH batches, ALL morphs (the old
    43 was the Fig-3f bilaterality position-SUBSET; length is position-independent, so use every bead
    morph). n=67 (39 + 28); consensus_axis_length_um = same axis metric as the co-graft."""
    allr = pd.concat([pd.read_csv(p, sep="\t", dtype={"trunk_morph_id": str}) for p in BEAD_TABLES.values()],
                     ignore_index=True)
    return allr["consensus_axis_length_um"].to_numpy(float)


def onegraft_length() -> np.ndarray:
    new = pd.read_csv(TAB / "05_per_organoid_metrics_newog.tsv", sep="\t")
    orig = pd.read_csv(TAB / "05_per_organoid_metrics_onegraft.tsv", sep="\t")
    orig = orig[orig["condition"].isin(["onegraft_ctrl", "onegraft_pax8"])]
    return pd.concat([new["length_um"], orig["length_um"]]).to_numpy(float)


def twograft_length() -> np.ndarray:
    reps = pd.read_csv(WORK / "annotations" / "two_graft_morphs.tsv", sep="\t")
    repset = set(reps[_b(reps["representative"])]["image_id"])
    cg = pd.read_csv(TAB / "05_per_organoid_metrics.tsv", sep="\t")
    cg = cg[(cg["condition"] == "cograft") & (cg["image_id"].isin(repset))]
    return cg["length_um"].to_numpy(float)


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


og = onegraft_length(); tg = twograft_length(); bd = bead_length()
groups = [("one-graft\n(LPM)", og, "#e6844d"), ("two-graft\n(LPM+NMP)", tg, "#c0552b"), ("BMP4-bead", bd, "#5b8bbf")]

fig, ax = plt.subplots(figsize=(4.9, 4.3))
labs = []
for x, (lab, vals, col) in enumerate(groups):
    n, med = strip(ax, x, vals, col)
    labs.append(f"{lab}\n(n={n})")
ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("morph length (µm)", fontsize=10)
ax.set_ylim(0, None)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = FIG / "07c_m4iii_length.png"
fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)

for lab, vals, _ in groups:
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    print(f"{lab.splitlines()[0]:10s} n={v.size:2d}  median={np.median(v):.0f}  IQR=[{np.percentile(v,25):.0f},{np.percentile(v,75):.0f}]  range=[{v.min():.0f},{v.max():.0f}]")
print("wrote", out)
