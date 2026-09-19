"""
Tier-1 clean re-plot of SD4 §7's four metric panels (7.1a, 7.1b, 7.2a, 7.2b).

The authors approved cleaning the baked internal jargon off these four figures. This is a CACHE
RE-PLOT — NO recompute: it imports the READ-ONLY analysis module, loads the three cached summary tables,
calls the four plot functions, and POST-PROCESSES the returned matplotlib figures (never edits the source):
  - blank the QC-narrative titles/suptitles ("...stage-02 labels...", "Threshold changes shift...");
  - 7.2a: rename the raw code legend (mapped_label_accuracy/raw_ARI/...) -> plain metric names;
  - axis labels -> the manuscript's "SMD Z-score cutoff" (capital Z, "cutoff"); "z > N" ticks -> "Z > N";
  - 7.1b / 7.2b colorbars: add/clean labels ("Recovery of accepted states", "Recall").
Numbers are byte-for-byte the cached values already approved (same tables the original figures used).
Clean PNGs -> figures/*_clean.png; build_07 points its four Tier-1 panels at them. Originals kept.
Env: /opt/anaconda3/bin/python3.

NOT runnable as shipped. The Supplementary Data builders embed pre-rendered figure assets that
are not in this repository, so the build stops at the first missing image. The built PDFs are
published with the paper as Supplementary Data, so a reader already has the output; what is
absent is the ability to rebuild it. This file ships as the record of how the section was
assembled. See `supplementary/README.md`.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import rcParams  # noqa: E402

# Global font convention (Arial, embedded as editable TrueType)
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["pdf.fonttype"] = 42

REPO = Path("<analysis-root>/")
SRC = REPO / "experiments/rq2_6aii_smd_z_score_threshold/src/rq2_6aii_smd_threshold.py"
INT = REPO / "experiments/rq2_6aii_smd_z_score_threshold/results/intermediates/02_trunk_main"
OUT = Path(__file__).resolve().parent / "figures"

# the module does `from src.trunk_morph_ref...` -> needs the repo root on sys.path
sys.path.insert(0, str(REPO))
# import the read-only analysis module by path (defines the plot fns; loads no data)
spec = importlib.util.spec_from_file_location("rq2_6aii_smd_threshold", SRC)
mod = importlib.util.module_from_spec(spec)
sys.modules["rq2_6aii_smd_threshold"] = mod
spec.loader.exec_module(mod)

summary = pd.read_pickle(INT / "morph_sweep_summary.pkl")
scan = pd.read_pickle(INT / "morph_resolution_scan_summary.pkl")
per_label = pd.read_pickle(INT / "morph_per_label_metrics.pkl")
# canonical coarse-state order = per_label row order (matches the original rendered figure)
label_order = per_label["label"].drop_duplicates().tolist()
selection_order = summary.sort_values("z_threshold")["display_name"].tolist()

CUTOFF = "SMD Z-score cutoff"


def _fix_z_ticks(labels):
    return [t.get_text().replace("z >", "Z >") for t in labels]


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", p.name)


# --- 7.1a  selection regimes (gene counts + chosen resolution vs cutoff) ---
fig = mod.plot_morph_selection_regimes(summary)
ax_top, ax_bot = fig.axes[0], fig.axes[1]
ax_top.set_title("")                       # drop QC headline
ax_top.legend(ax_top.get_lines()[:2],      # "above threshold" -> "above cutoff" (match axis)
              ["Scored genes above cutoff", "Final clustering genes"], frameon=False)
ax_bot.set_xlabel(CUTOFF)                   # "SMD z-score threshold" -> manuscript term
save(fig, "02_trunk_main_selection_regimes_clean.png")

# --- 7.1b  per-cutoff resolution scan heatmap ---
fig = mod.plot_morph_resolution_scan_heatmap(scan, summary)
ax = fig.axes[0]
ax.set_title("")
ax.set_ylabel(CUTOFF)                        # "Z-threshold regime" -> cutoff
ax.set_yticklabels(_fix_z_ticks(ax.get_yticklabels()))
cb = ax.collections[0].colorbar
if cb is not None:
    cb.set_label("Recovery of accepted clusters")   # "Composite recovery score"
save(fig, "02_trunk_main_resolution_scan_heatmap_clean.png")

# --- 7.2a  global agreement metrics vs cutoff ---
fig = mod.plot_morph_global_metrics(summary)
ax = fig.axes[0]
ax.set_title("")
ax.set_xlabel(CUTOFF)
# rename the raw code legend (first 4 lines are the metrics, in plot order; a 5th is the reference axvline)
ax.legend(
    ax.get_lines()[:4],
    ["Accuracy", "Macro-averaged recall", "Minimum per-cluster recall", "Adjusted Rand index"],
    frameon=False, ncols=2,
)
save(fig, "02_trunk_main_global_metrics_clean.png")

# --- 7.2b  per-state recall heatmap ---
fig = mod.plot_morph_label_recall_heatmap(per_label, label_order, selection_order=selection_order)
ax = fig.axes[0]
ax.set_title("")
ax.set_xlabel(CUTOFF)                         # "Selection regime" -> cutoff
ax.set_xticklabels(_fix_z_ticks(ax.get_xticklabels()))
cb = ax.collections[0].colorbar
if cb is not None:
    cb.set_label("Recall")
save(fig, "02_trunk_main_label_recall_heatmap_clean.png")

print("Tier-1 clean re-plot complete (4 figures, no recompute).")
