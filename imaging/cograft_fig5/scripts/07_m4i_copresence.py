#!/usr/bin/env python3
"""m4-i (Referee #1): fraction of morphs co-forming FOXF1+ LPM AND MESP2+ somite domains,
for the one-graft (LPM only) and two-graft (LPM+NMP) transplants vs the BMP4-bead reference.

No PAX8 channel in either graft set -> "all present" is scored on FOXF1+MESP2 (matched across
all three groups). Presence = a positive domain is detected for both markers.
  one-graft -> results/tables/05_per_organoid_metrics_onegraft.tsv (engine *_present)
  two-graft -> results/tables/05_per_organoid_metrics.tsv          (engine *_present)
  bead      -> Fig-3c/3f handoff (43 morphs); present = positive fraction > 0
Fully deterministic (no RNG). Descriptive/neutral, no narration of the differences.

NOT runnable as shipped. This stage reads the raw .czi acquisitions, which are not deposited —
they are available from the lead contact on request, per the paper's Data Availability
Statement. The measurements it produces are committed under the lane's `derived/` directory,
which is what the figure scripts read. See `data/DOWNLOAD.md`.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"                 # SD4 standard: Arial explicit + editable PDF text
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42

WORK = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
DA = Path("<analysis-root>/")
TAB = WORK / "results" / "tables"
FIG = WORK / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

BEAD_TABLES = {
    "2026-01-02": DA / "bilateral_plot" / "2026-01-02" / "05b_day5_feature_handoff_table.tsv",
    "2026-02-27": DA / "bilateral_plot" / "2026-02-27" / "05b_day5_feature_handoff_table.tsv",
}
# Fig-3f bead-position subset (verbatim from ../bilateral_plot/plot_bilaterality_with_stats.py)
PLOT_IDS = {
    "unilateral bead": {"2026-01-02": ["30","3202","24","39","01","35","14","15","38","21","09","12","07"],
                        "2026-02-27": ["04","05","06","09","20","22"]},
    "medial bead":     {"2026-01-02": ["29","17","33","28","27","05","04","02","18","11"],
                        "2026-02-27": ["07","08","11","12","15","17","18"]},
    "bilateral bead":  {"2026-01-02": ["06","20","31","10","34"], "2026-02-27": ["19","13"]},
}


def _b(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load_representatives() -> set:
    """Independent-morph representatives for the two-graft (one stack per organoid); drops
    day4-live paired timepoints + duplicate views. annotations/two_graft_morphs.tsv."""
    d = pd.read_csv(WORK / "annotations" / "two_graft_morphs.tsv", sep="\t")
    return set(d[_b(d["representative"])]["image_id"])


def graft_copresent(path: Path, cond_prefix: str, reps: set | None = None,
                    conditions: set | None = None) -> tuple[int, int]:
    d = pd.read_csv(path, sep="\t")
    if conditions is not None:                             # restrict to specific conditions (m4-i efficiency
        d = d[d["condition"].isin(conditions)]             # = ctrl+PAX8 only; KD/LDN are BMP-dependence panels)
    else:
        d = d[d["condition"].astype(str).str.startswith(cond_prefix)]
    if reps is not None:
        d = d[d["image_id"].isin(reps)]
    pres = _b(d["foxf1_present"]) & _b(d["mesp2_present"])
    return int(pres.sum()), int(len(d))


def bead_copresent() -> tuple[int, int]:
    allr = pd.concat([pd.read_csv(p, sep="\t", dtype={"trunk_morph_id": str}).assign(experiment=e)
                      for e, p in BEAD_TABLES.items()], ignore_index=True)
    k = n = 0
    for exps in PLOT_IDS.values():
        for e, ids in exps.items():
            for tid in ids:
                r = allr[(allr["experiment"] == e) & (allr["trunk_morph_id"] == tid)].iloc[0]
                n += 1
                if float(r["foxf1_fraction_mean"]) > 0 and float(r["mesp2_fraction_mean"]) > 0:
                    k += 1
    return k, n


og_k, og_n = graft_copresent(TAB / "05_per_organoid_metrics_onegraft.tsv", "onegraft",
                             conditions={"onegraft_ctrl", "onegraft_pax8"})   # efficiency baseline (no KD/LDN)
tg_k, tg_n = graft_copresent(TAB / "05_per_organoid_metrics.tsv", "cograft", reps=load_representatives())
bd_k, bd_n = bead_copresent()

groups = [("one-graft\n(LPM)", og_k, og_n), ("two-graft\n(LPM+NMP)", tg_k, tg_n), ("BMP4-bead", bd_k, bd_n)]
colors = ["#e6844d", "#c0552b", "#5b8bbf"]

fig, ax = plt.subplots(figsize=(4.7, 4.1))
xs = list(range(len(groups)))
for x, (lab, k, n), col in zip(xs, groups, colors):
    ax.bar(x, 100.0 * k / n, width=0.62, color=col, edgecolor="black", linewidth=0.7)
    ax.text(x, 100.0 * k / n + 1.6, f"{k}/{n}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(xs); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("FOXF1$^{+}$ LPM & MESP2$^{+}$ somite co-present (% of morphs)", fontsize=10)
ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = FIG / "07_m4i_copresence.png"
fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)

print(f"one-graft {og_k}/{og_n}  |  two-graft {tg_k}/{tg_n}  |  bead {bd_k}/{bd_n}")
print("wrote", out)
