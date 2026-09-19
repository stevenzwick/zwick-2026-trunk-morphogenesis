#!/usr/bin/env python3
"""m5-iv(c) (Referee #1) — host FOXF1 induction: two-graft alongside the full Fig 5n panel
(one-graft LPM-bead ctrl + BMPR1A-KD + LDN), replotted for comparison. Fig-5n-style bar
(mean ± SEM, per-organoid points), metric = % of FOXF1+ signal assigned to host.

  Fig 5n conditions (ctrl / BMPR1A_KD / LDN) = read straight from the LPM_transplant analysis
      (`../LPM_transplant/results/tables/02_foxf1_positive_identity_summary.tsv`, summary_level=file):
      host% = 100 * foxf1_positive_host_partition_fraction  (host/donor reporter LINEAGE partition).
  two-graft = this dir's cograft reps: host% = 100 * foxf1_host_fraction / foxf1_fraction
      (host = FOXF1+ OUTSIDE the donor LPM-bead graft MASK — unlabeled host, no reporter).
  ⚠️ two-graft host scored by graft-exclusion (no host marker), the rest by reporter lineage — the
     two-graft bar is drawn in a distinct colour and the method is noted in the caption.

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
DA = Path("<analysis-root>/")
TAB = WORK / "results" / "tables"
FIG = WORK / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
LPM_SUMMARY = DA / "LPM_transplant" / "results" / "tables" / "02_foxf1_positive_identity_summary.tsv"

FIG5N_GREEN = "#2e7d32"   # Fig 5n bars (reporter lineage partition)
TWOGRAFT = "#c0552b"      # two-graft (graft-exclusion; distinct colour = different method)


def fig5n_hostpct(condition: str) -> np.ndarray:
    """Fig 5n per-file host% of FOXF1+ for a condition (reporter lineage partition)."""
    s = pd.read_csv(LPM_SUMMARY, sep="\t")
    s = s[(s["summary_level"] == "file") & (s["condition"] == condition)]
    return (s.groupby("file_id")["foxf1_positive_host_partition_fraction"].mean() * 100.0).to_numpy(float)


def twograft_hostpct() -> np.ndarray:
    reps = pd.read_csv(WORK / "annotations" / "two_graft_morphs.tsv", sep="\t")
    repset = set(reps[reps["representative"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])]["image_id"])
    cg = pd.read_csv(TAB / "05_per_organoid_metrics.tsv", sep="\t")
    cg = cg[(cg["condition"] == "cograft") & (cg["image_id"].isin(repset))]
    return (100.0 * cg["foxf1_host_fraction_mean"] / cg["foxf1_fraction_mean"]).to_numpy(float)


# two-graft first (new/focus, next to the LPM-bead ctrl), then Fig 5n in its native order
GROUPS = [("two-graft\n(LPM+NMP)", twograft_hostpct(), TWOGRAFT),
          ("LPM bead\n(one-graft)", fig5n_hostpct("ctrl"), FIG5N_GREEN),
          ("BMPR1A-KD", fig5n_hostpct("BMPR1A_KD"), FIG5N_GREEN),
          ("LDN", fig5n_hostpct("LDN"), FIG5N_GREEN)]

fig, ax = plt.subplots(figsize=(5.6, 5.2))
labs = []
for x, (lab, vals, col) in enumerate(GROUPS):
    vals = vals[np.isfinite(vals)]
    m = float(np.mean(vals)); sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    ax.bar(x, m, yerr=sem, width=0.62, color=col, edgecolor="#1b1b1b", linewidth=1.0, capsize=6, zorder=2)
    jit = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
    ax.scatter(x + jit, vals, s=42, color="#1b1b1b", zorder=3)
    labs.append(f"{lab}\n(n={vals.size})")
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(labs, fontsize=9)
ax.set_ylabel("% FOXF1+ signal assigned to host", fontsize=10)
ax.set_ylim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = FIG / "07f_m5iv_c_host_vs_onegraft.png"
fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)

for lab, vals, _ in GROUPS:
    v = vals[np.isfinite(vals)]
    print(f"{lab.splitlines()[0]:12s} n={v.size}  mean={v.mean():.1f}%  SEM={v.std(ddof=1)/np.sqrt(v.size):.1f}")
print("wrote", out)
