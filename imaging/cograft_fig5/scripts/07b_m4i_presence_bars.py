#!/usr/bin/env python3
"""m4-i (Referee #1) — SEPARATE FOXF1 and MESP2 presence bars (correction to the old
COMBINED co-presence in 07_m4i_copresence.py). Two panels: fraction of morphs forming a MESP2+ somite
domain, and fraction forming a FOXF1+ LPM domain, for one-graft / two-graft vs the BMP4-bead reference.

Cohorts + presence criterion:
  one-graft -> results/tables/05_per_organoid_metrics_newog.tsv  (20 new morphs; per-image thresholds,
               hand-tuned masks; present = engine *_present = a positive region survives threshold+min-size)
  two-graft -> results/tables/05_per_organoid_metrics.tsv         (cograft, independent representatives)
  bead      -> Fig-3c/3f handoff, ALL morphs (n=67); present = fraction_mean > 0
NB the BMP4-bead comparator stays on its ORIGINAL global-pool quant (keep as-is); the
one-graft is the new per-image quant. Deterministic. Descriptive/neutral (no narration of differences).

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

plt.rcParams["font.family"] = "sans-serif"
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
PLOT_IDS = {
    "unilateral bead": {"2026-01-02": ["30", "3202", "24", "39", "01", "35", "14", "15", "38", "21", "09", "12", "07"],
                        "2026-02-27": ["04", "05", "06", "09", "20", "22"]},
    "medial bead":     {"2026-01-02": ["29", "17", "33", "28", "27", "05", "04", "02", "18", "11"],
                        "2026-02-27": ["07", "08", "11", "12", "15", "17", "18"]},
    "bilateral bead":  {"2026-01-02": ["06", "20", "31", "10", "34"], "2026-02-27": ["19", "13"]},
}


def _b(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load_representatives() -> set:
    d = pd.read_csv(WORK / "annotations" / "two_graft_morphs.tsv", sep="\t")
    return set(d[_b(d["representative"])]["image_id"])


def graft_present(path: Path, cond_prefix: str, reps: set | None = None) -> tuple[int, int, int]:
    """(MESP2+ count, HOST-FOXF1+ count, n) from an engine per-organoid table. FOXF1 = HOST-derived
    (FOXF1+ minus the donor graft): the donor IS FOXF1+ LPM, so total FOXF1 is trivially present —
    the induction readout is the host forming a FOXF1+ LPM domain."""
    d = pd.read_csv(path, sep="\t")
    d = d[d["condition"].astype(str).str.startswith(cond_prefix)]
    if reps is not None:
        d = d[d["image_id"].isin(reps)]
    return int(_b(d["mesp2_present"]).sum()), int(_b(d["foxf1_host_present"]).sum()), int(len(d))


def bead_present() -> tuple[int, int, int]:
    """FULL day-5 BMP4-bead trunk-morph cohort — BOTH batches, ALL morphs (the old 43
    was the Fig-3f bilaterality position-SUBSET; presence is position-independent). n=67 (39 + 28)."""
    allr = pd.concat([pd.read_csv(p, sep="\t", dtype={"trunk_morph_id": str}) for p in BEAD_TABLES.values()],
                     ignore_index=True)
    return int((allr["mesp2_fraction_mean"] > 0).sum()), int((allr["foxf1_fraction_mean"] > 0).sum()), int(len(allr))


def onegraft_present() -> tuple[int, int, int]:
    """One-graft cohort = ALL the LPM-induction data: the 20 new per-image-tuned morphs
    + the original 5 ctrl+PAX8 (KD/LDN excluded — those are the BMP-dependence panels, Fig 5 m,n). Small
    pipeline differences between the two batches accepted ("analyze all the data")."""
    new = pd.read_csv(TAB / "05_per_organoid_metrics_newog.tsv", sep="\t")
    orig = pd.read_csv(TAB / "05_per_organoid_metrics_onegraft.tsv", sep="\t")
    orig = orig[orig["condition"].isin(["onegraft_ctrl", "onegraft_pax8"])]
    d = pd.concat([new, orig], ignore_index=True)
    return int(_b(d["mesp2_present"]).sum()), int(_b(d["foxf1_host_present"]).sum()), int(len(d))


og_m, og_f, og_n = onegraft_present()
tg_m, tg_f, tg_n = graft_present(TAB / "05_per_organoid_metrics.tsv", "cograft", reps=load_representatives())
bd_m, bd_f, bd_n = bead_present()

labels = [f"one-graft\n(LPM)\nn={og_n}", f"two-graft\n(LPM+NMP)\nn={tg_n}", f"BMP4-bead\nn={bd_n}"]
colors = ["#e6844d", "#c0552b", "#5b8bbf"]
MESP2 = [(og_m, og_n), (tg_m, tg_n), (bd_m, bd_n)]
FOXF1 = [(og_f, og_n), (tg_f, tg_n), (bd_f, bd_n)]

fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
# Both labeled host-derived: FOXF1 is quantified as host (FOXF1 minus donor graft); MESP2 is host-derived
# by construction (one-graft donor = LPM not somite; bead = no donor) so it is NOT separately subtracted
# (a given, visually obvious — the two-graft NMP contribution is not quantified out).
for ax, (title, data) in zip(axes, [("host-derived MESP2$^{+}$ somite domain", MESP2),
                                    ("host-derived FOXF1$^{+}$ LPM domain", FOXF1)]):
    for x, (k, n), col in zip(range(3), data, colors):
        ax.bar(x, 100.0 * k / n, width=0.62, color=col, edgecolor="black", linewidth=0.7)
        ax.text(x, 100.0 * k / n + 1.6, f"{k}/{n}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("morphs forming the domain (%)", fontsize=10)
fig.tight_layout()
out = FIG / "07b_m4i_presence_bars.png"
fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)

print(f"MESP2 present: one-graft {og_m}/{og_n}  two-graft {tg_m}/{tg_n}  bead {bd_m}/{bd_n}")
print(f"host-FOXF1 present: one-graft {og_f}/{og_n}  two-graft {tg_f}/{tg_n}  bead {bd_f}/{bd_n} (bead=all host, no donor)")
print("wrote", out)
