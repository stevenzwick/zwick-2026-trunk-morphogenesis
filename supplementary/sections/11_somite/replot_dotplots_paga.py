"""
Font/legibility re-render for SD4 Section 11 ("font sizes too small ... re-render",
scope = easy panels, NO recompute). Redraws the 3 dot plots + 2 PAGA heatmaps straight from the SAVED
summary tables (no re-quantification) at proper Arial fonts + Fig-4 house style, writing over the cropped
copies in ./derived/ (so build_11_somite.py, which reads ./derived/, picks them up). The UMAP/diffusion-map
embeddings + boxplots are left as crop_panels.py output (their per-cell data isn't saved -> would need a
recompute). RUN AFTER crop_panels.py.

Encoding kept identical to the source (dot color = RAW mean expression, per-plot scale; dot size = % cells
expressing) so only legibility changes, not the science.

NOT runnable as shipped. The Supplementary Data builders embed pre-rendered figure assets that
are not in this repository, so the build stops at the first missing image. The built PDFs are
published with the paper as Supplementary Data, so a reader already has the output; what is
absent is the ability to rebuild it. This file ships as the record of how the section was
assembled. See `supplementary/README.md`.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from pathlib import Path
from cmcrameri.cm import batlow

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["pdf.fonttype"] = 42
rcParams["svg.fonttype"] = "none"

RES = Path("<analysis-root>/"
           "experiments/rq1_4_somite_progression/results")
OUT = Path(__file__).resolve().parent / "derived"
DPI = 200

GENES = ["TBXT", "MSGN1", "DLL1", "RIPPLY2", "MESP2", "HES7",
         "MEOX1", "MEOX2", "TCF15", "PAX3", "FOXC1", "FOXC2",
         "UNCX", "TBX18", "ALDH1A2", "FST", "MEGF10"]
# short group headers (full names overran their brackets and collided); span = gene index range
GROUPS = [("NMP / PSM", 0, 5),
          ("Somite maturation", 6, 11),
          ("Regionalization / subtype", 12, 16)]

SHORT = {"Neuromesodermal Progenitors": "NMP", "Presomitic Mesoderm": "PSM",
         "Anterior Presomitic Mesoderm / Early Somite": "Anterior PSM / Early Somite"}


def short(lbl):
    return SHORT.get(lbl, lbl)


# ---- font sizes (points at render size; panels render close to placed width so ~= printed pt) ----
F_GENE, F_CLUST, F_GROUP, F_CBAR, F_LEG = 11, 11, 9, 10, 10


def dotplot(mean_csv, pct_csv, out_name, fig_w=8.0):
    mean = pd.read_csv(RES / mean_csv, index_col=0)[GENES]
    pct = pd.read_csv(RES / pct_csv, index_col=0)[GENES]
    clusters = list(mean.index)
    nC, nG = len(clusters), len(GENES)
    vmax = float(mean.values.max())

    fig_h = 0.62 + 0.46 * nC + 1.15          # brackets + rows + rotated gene labels
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    xx, yy = np.meshgrid(np.arange(nG), np.arange(nC))
    m = mean.values.ravel()
    p = pct.values.ravel()
    sizes = 12 + (p / 100.0) * 300.0         # marker area in pts^2
    sc = ax.scatter(xx.ravel(), yy.ravel(), s=sizes, c=m, cmap=batlow, vmin=0, vmax=vmax,
                    edgecolors="black", linewidths=0.4, zorder=3)

    ax.set_xlim(-0.6, nG - 0.4)
    ax.set_ylim(nC - 0.5, -0.5)              # first cluster on top
    ax.set_xticks(range(nG))
    ax.set_xticklabels(GENES, rotation=90, fontsize=F_GENE)
    ax.set_yticks(range(nC))
    ax.set_yticklabels([short(c) for c in clusters], fontsize=F_CLUST)
    ax.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.6)
    ax.set_axisbelow(True)
    ax.grid(True, color="0.90", linewidth=0.5, zorder=0)

    # gene-group brackets above the plot
    ytop = -0.75
    for label, a, b in GROUPS:
        ax.plot([a - 0.15, b + 0.15], [ytop, ytop], color="0.25", lw=1.0, clip_on=False)
        for x in (a - 0.15, b + 0.15):
            ax.plot([x, x], [ytop, ytop + 0.12], color="0.25", lw=1.0, clip_on=False)
        ax.text((a + b) / 2, ytop - 0.14, label, ha="center", va="bottom",
                fontsize=F_GROUP, clip_on=False, ma="center")

    # colorbar (Mean expression, top->bottom reading)
    cb = fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.015, aspect=18)
    cb.set_label("Mean expression", rotation=270, labelpad=14, fontsize=F_CBAR)
    cb.ax.tick_params(labelsize=F_CBAR - 1)
    cb.outline.set_linewidth(0.6)

    # size legend "% cells expressing"
    handles = [plt.scatter([], [], s=12 + (q / 100.0) * 300.0, color="0.45",
                           edgecolors="black", linewidths=0.4) for q in (20, 40, 60, 80, 100)]
    leg = ax.legend(handles, ["20", "40", "60", "80", "100"], title="% cells expressing",
                    loc="center left", bbox_to_anchor=(1.11, 0.5), frameon=False,
                    labelspacing=1.1, handletextpad=0.6, fontsize=F_LEG, title_fontsize=F_LEG)
    leg.get_title().set_multialignment("center")

    fig.subplots_adjust(left=0.20, right=0.80, top=0.90, bottom=0.20)
    fig.savefig(OUT / out_name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  dotplot -> {out_name}  ({nC} clusters x {nG} genes; vmax={vmax:.2f})")


def paga(csv, out_name, fig_w=4.6):
    # Rendered small (2-across on the page) -> render at large fonts so they stay legible after the
    # renderer's downscale to ~47% page width; x-labels rotated so the long state names don't collide.
    d = pd.read_csv(RES / csv, index_col=0)
    n = len(d)
    labs = [short(s).replace(" / ", " /\n") for s in d.index]
    vmax = float(d.values.max())
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.9), dpi=DPI)
    im = ax.imshow(d.values, cmap=batlow, vmin=0, vmax=vmax)
    thr = 0.55 * vmax
    for i in range(n):
        for j in range(n):
            v = d.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=15, color="white" if v < thr else "black")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labs, fontsize=13, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labs, fontsize=13)
    ax.set_xlabel("To state", fontsize=13.5); ax.set_ylabel("From state", fontsize=13.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("PAGA connectivity", rotation=270, labelpad=16, fontsize=13)
    cb.ax.tick_params(labelsize=11)
    cb.outline.set_linewidth(0.6)
    fig.savefig(OUT / out_name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  paga -> {out_name}  ({n}x{n}; vmax={vmax:.2f})")


if __name__ == "__main__":
    dotplot("RefQ1_4_markers_coarse_mean.csv", "RefQ1_4_markers_coarse_pct_cells.csv",
            "RefQ1_4_dotplot_somite_progression_coarse.png")
    dotplot("RefQ1_4_markers_subclusters_mean.csv", "RefQ1_4_markers_subclusters_pct_cells.csv",
            "RefQ1_4_dotplot_somite_progression_subclusters.png")
    dotplot("RefQ1_4_markers_superstates_mean.csv", "RefQ1_4_markers_superstates_pct_cells.csv",
            "RefQ1_4_dotplot_somite_superstates.png")
    paga("RefQ1_4_paga_somite_superstate_connectivity.csv",
         "RefQ1_4_paga_somite_superstate_connectivity_heatmap.png")
    paga("RefQ1_4_paga_psm_somite_superstate_connectivity.csv",
         "RefQ1_4_paga_psm_somite_superstate_connectivity_heatmap.png")
    print("re-plotted 5 panels into", OUT)
