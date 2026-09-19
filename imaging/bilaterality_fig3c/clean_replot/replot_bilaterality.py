"""
Clean re-render of the SD4 Section 4 figure — the bilaterality-index box plot (= main-text Fig. 3c).

FAITHFUL reproduction of the canonical source export
  I/bilateral_plot/Orgs_bilaterality_score_three_channels_with_pooled_stats.png
(same data, same 3-group x 3-marker layout, same FOXF1/PAX8/MESP2 colors, same pooled-comparison
significance brackets + stars). ONLY notebook-export styling is cleaned to Nature house style:
  - Arial set explicitly + embedded as editable TrueType (global figure convention).
  - y-axis label "bilaterality_index_mean" -> "Bilaterality index" (the manuscript's own term).
  - legend title "Channel" -> "Marker".
  - non-significant label "ns" -> "n.s." (manuscript Fig. 3c notation).

Data are RE-LOADED read-only from the source TSVs through the source module's own functions, so the
reconstructed subset (n = 43 morphs; 41 for PAX8, which 2 medial morphs lack) matches Fig. 3c exactly.
Holm-adjusted p-values are the LOCKED canonical values from bilateral_plot/README.md (source of record;
two-sided experiment-stratified permutation test, 1e6 perms, seed 20260412) — not re-run here, so the
stars are byte-faithful to the committed figure.

READ-ONLY: this writes ONLY into this replot/ dir; the source analysis dir is never modified.
Rerun: /opt/anaconda3/bin/python3 replot_bilaterality.py
"""
import sys
sys.dont_write_bytecode = True  # never drop a .pyc into the READ-ONLY source dir on import
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

SRC = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SRC))
# read-only import of the canonical script's pure data/stat/draw helpers (its main() is NOT called)
from plot_bilaterality_with_stats import (  # noqa: E402
    load_rows, build_reconstructed_rows, build_plot_dataframe,
    PALETTE, GROUP_ORDER, hue_offsets, add_pooled_sig_bar, p_to_stars,
)

HERE = Path(__file__).resolve().parent
#: Rendered panels go to the lane's output/ directory, as in every other lane, so that
#: running this script never writes into a source directory.
OUTPUT = HERE.parent / "output"

# LOCKED canonical Holm-adjusted p-values (bilateral_plot/README.md; source_map.md agrees):
# unilateral bead  vs  pooled(medial + bilateral), per marker.
P_HOLM = {"FOXF1": 0.000003, "PAX8": 0.000270, "MESP2": 0.794695}
CHANNEL_ORDER = ["FOXF1", "PAX8", "MESP2"]  # manuscript order (Fig. 3c)

# Arial everywhere; embed as editable TrueType in the vector masters.
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["pdf.fonttype"] = 42
rcParams["svg.fonttype"] = "none"


def main() -> None:
    df = build_plot_dataframe(build_reconstructed_rows(load_rows()))

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    box_width = 0.5

    sns.boxplot(
        data=df, x="Group", y="Score", hue="Channel",
        order=GROUP_ORDER, hue_order=CHANNEL_ORDER, palette=PALETTE,
        width=box_width, boxprops=dict(alpha=0.3), fliersize=0, linewidth=1.5, ax=ax,
    )
    sns.stripplot(
        data=df, x="Group", y="Score", hue="Channel",
        order=GROUP_ORDER, hue_order=CHANNEL_ORDER, palette=PALETTE,
        dodge=True, jitter=0.18, size=4, edgecolor="black", linewidth=0.5, alpha=0.45, ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], title="Marker", frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10.5, title_fontsize=11)
    ax.set_ylabel("Bilaterality index", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.22)
    ax.tick_params(axis="both", labelsize=11)

    offsets = hue_offsets(CHANNEL_ORDER, width=box_width)
    y_levels = {"FOXF1": 1.03, "PAX8": 1.11, "MESP2": 1.19}
    group_centers = {g: i for i, g in enumerate(GROUP_ORDER)}
    for marker in CHANNEL_ORDER:
        label = p_to_stars(P_HOLM[marker])
        if label == "ns":
            label = "n.s."  # match the manuscript's Fig. 3c notation
        add_pooled_sig_bar(
            ax,
            x_uni=group_centers["unilateral bead"] + offsets[marker],
            x_med=group_centers["medial bead"] + offsets[marker],
            x_bil=group_centers["bilateral bead"] + offsets[marker],
            y=y_levels[marker], text=label,
        )

    plt.tight_layout(rect=[0, 0, 0.86, 1])
    for ext in ("png", "svg", "pdf"):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT / f"bilaterality_boxplot_v1.{ext}", dpi=300)
    plt.close(fig)

    # provenance echo (anti-drift check: must read 43/41/43 and ****,***,n.s.)
    n = {m: int((df["Channel"] == m).sum()) for m in CHANNEL_ORDER}
    stars = {m: ("n.s." if p_to_stars(P_HOLM[m]) == "ns" else p_to_stars(P_HOLM[m])) for m in CHANNEL_ORDER}
    print("n per marker:", n)
    print("stars       :", stars)
    print("saved        :", OUTPUT / "bilaterality_boxplot_v1.png")


if __name__ == "__main__":
    main()
