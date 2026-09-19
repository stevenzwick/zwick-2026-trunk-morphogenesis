#!/usr/bin/env python3
"""Per-marker (FOXF1, MESP2) positive-domain fraction BY CONDITION, with the LDN and
BMPR1A-KD one-graft perturbations as empirical negatives to CALIBRATE the FOXF1+/MESP2+
cutoff: the perturbations should show ~no induced FOXF1. Two separate
panels (FOXF1, MESP2); bars = conditions, points = per-morph. Deterministic (fixed jitter).

NOT runnable as shipped. This stage reads the raw .czi acquisitions, which are not deposited —
they are available from the lead contact on request, per the paper's Data Availability
Statement. The measurements it produces are committed under the lane's `derived/` directory,
which is what the figure scripts read. See `data/DOWNLOAD.md`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
TAB = WORK / "results" / "tables"
FIG = WORK / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)


def _b(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def reps() -> set:
    d = pd.read_csv(WORK / "annotations" / "two_graft_morphs.tsv", sep="\t")
    return set(d[_b(d["representative"])]["image_id"])


og = pd.read_csv(TAB / "05_per_organoid_metrics_onegraft.tsv", sep="\t")
tg = pd.read_csv(TAB / "05_per_organoid_metrics.tsv", sep="\t")
tg = tg[(tg["condition"] == "cograft") & (tg["image_id"].isin(reps()))]

groups = [
    ("one-graft\nctrl", og[og["condition"].isin(["onegraft_ctrl", "onegraft_pax8"])], "#e6844d"),
    ("one-graft\nBMPR1A-KD", og[og["condition"] == "onegraft_bmpr1a_kd"], "#9a9a9a"),
    ("one-graft\nLDN", og[og["condition"] == "onegraft_ldn"], "#9a9a9a"),
    ("two-graft", tg, "#2c7fb8"),
]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
for ax, (mk, title) in zip(axes, [("foxf1", "FOXF1 (LPM)"), ("mesp2", "MESP2 (somite)")]):
    for x, (lab, df, col) in enumerate(groups):
        v = df[f"{mk}_fraction_mean"].to_numpy(float); v = v[np.isfinite(v)]
        ax.bar(x, v.mean() if v.size else 0.0, width=0.6, color=col, alpha=0.35, edgecolor="black", linewidth=0.6)
        jit = (RNG.random(v.size) - 0.5) * 0.30
        ax.scatter(np.full(v.size, x) + jit, v, s=26, color=col, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel(f"{title.split()[0]}+ positive fraction"); ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(0, None)
fig.suptitle("Marker+ fraction by condition (FOXF1 4σ / MESP2 3σ). LDN + BMPR1A-KD = perturbation negatives to calibrate the cutoff.",
             fontsize=10)
fig.tight_layout()
out = FIG / "08_marker_by_condition.png"
fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)

for lab, df, _ in groups:
    print(f"{lab.replace(chr(10),' '):24s} n={len(df):2d}  "
          f"FOXF1_frac={df['foxf1_fraction_mean'].mean():.3f}  MESP2_frac={df['mesp2_fraction_mean'].mean():.3f}")
print("wrote", out)
