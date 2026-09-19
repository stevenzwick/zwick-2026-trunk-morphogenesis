#!/usr/bin/env python3
"""m5-iv(b) (Referee #1) — one-graft donor-vs-host of PAX8 (+ FOXF1 for contrast).
The PAX8-stained one-graft morphs (og_pax8_1/2; host UNLABELED, CFP-donor graft on ch5) show that
PAX8+ is almost entirely HOST-induced (the donor LPM graft does not express PAX8), whereas FOXF1+ in
the same morphs is mostly the donor graft (LPM bead is FOXF1+). Metric = % of the marker+ signal
assigned to host (= marker+ OUTSIDE the donor-graft mask / total marker+).

Three bars are drawn, not two: PAX8, FOXF1 and WT1.

  PAX8  = 05_per_organoid_metrics_pax8og.tsv (engine run with the 'foxf1' slot pointed at ch4=PAX8).
  FOXF1 = 05_per_organoid_metrics_onegraft.tsv (the same 2 morphs; donor-subtracted, n_donor=1,
          ch_donor=5) CONCATENATED with 05_per_organoid_metrics_wt1foxf1og.tsv, so the FOXF1 bar
          covers 3 morphs, not 2.
  WT1   = 05_per_organoid_metrics_wt1og.tsv.

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

WORK = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
TAB = WORK / "results" / "tables"
FIG = WORK / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAX8_IDS = ["og_pax8_1", "og_pax8_2"]


def host_pct(table: str, ids=None) -> np.ndarray:
    df = pd.read_csv(TAB / table, sep="\t")
    if ids is not None:
        df = df[df["image_id"].isin(ids)]
    return (100.0 * df["foxf1_host_fraction_mean"] / df["foxf1_fraction_mean"]).to_numpy(float)


# FOXF1 (LPM, donor-dominated) vs the host-induced intermediate-mesoderm markers PAX8 & WT1.
# FOXF1 across all 3 IM-stained organoids (2 CFP-donor PAX8 + 1 H2B-donor WT1; the H2B nuclear label
# gives a tighter donor mask -> higher host%, author-accepted).
foxf1 = np.concatenate([host_pct("05_per_organoid_metrics_onegraft.tsv", PAX8_IDS),
                        host_pct("05_per_organoid_metrics_wt1foxf1og.tsv")])
GROUPS = [("FOXF1\n(LPM)", foxf1, "#c0552b"),
          ("PAX8\n(IM)", host_pct("05_per_organoid_metrics_pax8og.tsv"), "#2e7d32"),
          ("WT1\n(IM)", host_pct("05_per_organoid_metrics_wt1og.tsv"), "#2e7d32")]

fig, ax = plt.subplots(figsize=(3.9, 5.2))
labs = []
for x, (lab, vals, col) in enumerate(GROUPS):
    vals = vals[np.isfinite(vals)]
    m = float(np.mean(vals))
    ax.bar(x, m, width=0.6, color=col, edgecolor="#1b1b1b", linewidth=1.0, zorder=2)
    ax.scatter(np.full(vals.size, x), vals, s=48, color="#1b1b1b", zorder=3, clip_on=False)
    labs.append(f"{lab}\n(n={vals.size})")
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(labs, fontsize=9)
ax.set_ylabel("% marker+ signal assigned to host", fontsize=10)
ax.set_ylim(0, 105); ax.set_yticks(range(0, 101, 20))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = FIG / "07g_m5iv_b_onegraft_pax8.png"
fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)

for lab, vals, _ in GROUPS:
    v = vals[np.isfinite(vals)]
    print(f"{lab.splitlines()[0]:8s} one-graft host%: {[round(x,1) for x in v.tolist()]}  mean={v.mean():.1f}")
print("wrote", out)
