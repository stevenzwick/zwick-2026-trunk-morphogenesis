#!/usr/bin/env python3
"""m5-ii (Referee #1) — success rate of co-grafted morphs producing the stereotyped axial (MESP2+
somite) and mediolateral (FOXF1+ LPM) pattern, over ALL attempted morphs (not the picked subset).
Justifies that the detailed panels (m4-i/ii/iii, m5-iv) quantify the SUCCESSFUL, good-morphology morphs.

Counts = Tianlei He manual scoring, Slack 2026-07-06 (source of truth; NOT in the image tables — these
count every attempted morph, including cells-present-but-poor-morphology ones excluded from the analyzed
set):  one-graft  MESP2 38/38, FOXF1 27/38   ·   two-graft  MESP2 14/21, FOXF1 14/21.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42

FIG = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27/results/figures")

# (numerator, denominator) per group per marker  (Tianlei He, Slack 2026-07-06)
DATA = {"MESP2 (somite)":    {"one-graft\n(LPM)": (38, 38), "two-graft\n(LPM+NMP)": (14, 21)},
        "FOXF1 (LPM)":       {"one-graft\n(LPM)": (27, 38), "two-graft\n(LPM+NMP)": (14, 21)}}
MARK_COLOR = {"MESP2 (somite)": "#b0413e", "FOXF1 (LPM)": "#3a7d44"}
GROUPS = ["one-graft\n(LPM)", "two-graft\n(LPM+NMP)"]
MARKERS = list(DATA.keys())

x = np.arange(len(GROUPS)); w = 0.36
fig, ax = plt.subplots(figsize=(4.8, 4.4))
for j, mk in enumerate(MARKERS):
    off = (j - 0.5) * w
    for i, g in enumerate(GROUPS):
        num, den = DATA[mk][g]; pct = 100.0 * num / den
        ax.bar(x[i] + off, pct, width=w, color=MARK_COLOR[mk], edgecolor="black", linewidth=0.7,
               label=mk if i == 0 else None)
        ax.text(x[i] + off, pct + 1.8, f"{num}/{den}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(GROUPS, fontsize=9)
ax.set_ylabel("morphs forming the domain (%)", fontsize=10)
ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
ax.legend(fontsize=8.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.30))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = FIG / "07h_m5ii_success_rate.png"
fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)
for mk in MARKERS:
    print(mk, {g: f"{DATA[mk][g][0]}/{DATA[mk][g][1]} ({100*DATA[mk][g][0]/DATA[mk][g][1]:.0f}%)" for g in GROUPS})
print("wrote", out)
