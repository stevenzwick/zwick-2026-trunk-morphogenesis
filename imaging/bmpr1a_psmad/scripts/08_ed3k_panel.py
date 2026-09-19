#!/usr/bin/env python
"""08 - ED Fig 3k: pSMAD1/5/9 in colony-edge cells, SCRAMBLE KD vs BMPR1A KD, without / with BMP4.

The four-box panel of Extended Data Fig. 3k, drawn at the published panel's geometry. Note this is
a DIFFERENT figure from the nine-box display `03_figure.py` builds, which adds the no-KD condition
and the BMP4-interior category and belongs to Supplementary Data 4.

Values come from `derived/negctrl_summary_per_field_30min.csv` through
`figures/panel_specs.plotted_rows`, the same call `figures/verify_imaging_panels.py` makes. Going
through one path is deliberate: when the rule that selects plotted rows lives in the renderer and
is restated in the verifier, the two can drift and verification checks a row set the figure was
never drawn from. Here the rule drops the 15 rows that belong to the nine-box figure, leaving the
12 the panel plots - the same 12 as the Source Data sheet 'Extended Data Fig.3k'.

Geometry, colours, font sizes and legend layout were MEASURED off the accepted manuscript figure
(300 ppi) so the output drops into the ED 3 artboard at that scale; axes are 25.3 x 26.3 mm.
Whiskers are the matplotlib default 1.5 x IQR, which at n = 3 always reaches the minimum and
maximum. Points are jittered +/-0.05 under seed 0, so the render is deterministic.

Env: any environment with pandas + matplotlib. Set SOURCE_DATA_DIR to also assert the plotted
values equal the published sheet.
Run: python imaging/bmpr1a_psmad/scripts/08_ed3k_panel.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import ScaledTranslation

matplotlib.rcParams["pdf.fonttype"] = 42  # embed editable TrueType text in PDFs (Nature-friendly)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "figures"))
from panel_specs import (by_panel, plotted_rows,          # noqa: E402
                         ED3K_KNOCKDOWNS, ED3K_CATEGORIES)

OUT = REPO / "figures" / "output"
NAME = "ed3k_psmad_edge_knockdown"

CONDS = [(t, p) for p, t in ED3K_KNOCKDOWNS.items()]                    # table value, sheet label
CATS = [(ED3K_CATEGORIES["without BMP4"], "without BMP4", "#eca146"),
        (ED3K_CATEGORIES["with BMP4"], "with BMP4", "#2171b5")]

# ---- measured off the published panel, in pixels at 300 ppi ----
PPI = 300
ORIGIN = (100, 2995)                  # panel top-left on the reference page
CANVAS = (560, 485)                   # panel size, px
AXES = (237, 3010, 536, 3321)         # spine centres: left, top, right, bottom
LEGEND_CENTRE = (404, 3439.5)
XLIM = (-0.745, 1.317)                # the 'with BMP4' box sits at x = condition index
YLIM = (83.5, 1016.0)
BOX_OFFSET = 0.4                      # the 'without BMP4' box sits this far left of it
BOX_W = 0.293
XTICK_SHIFT = 0.114                   # label anchor, right of the 'with BMP4' box (ticks invisible)
FS = dict(ytick=7.5, xtick=8.0, ylabel=10.0, legend=8.0)
LW = dict(spine=0.5, box=0.25, median=0.3, whisker=0.3, tick=0.35, legend_frame=0.6, marker_edge=0.2)
TICK_LEN, TICK_PAD = 1.4, 0.7         # pt
MARKER_S = 1.5                        # pt^2 (~1.2 pt diameter)
XLABEL_ROT, XLABEL_PAD = 15, 0.5
YLABEL_RIGHT_PX = 168                 # right edge of the y-label ink
LEGEND = dict(borderpad=0.26, handlelength=0.79, handleheight=0.28, handletextpad=0.285,
              columnspacing=0.46)
LEGEND_NUDGE_PX = dict(dx=-3.0, text=4.0, handle=6.5)   # dx rightward; text/handle downward, px


def load_values() -> dict:
    """The 12 plotted per-well medians, keyed by (knockdown, BMP4 category)."""
    table = plotted_rows(by_panel()["ED Fig 3k"])
    values = {}
    for cond, _ in CONDS:
        for cat, _, _ in CATS:
            group = table[(table.condition == cond) & (table.category == cat)].sort_values("field")
            assert len(group) == 3, (cond, cat, len(group))
            values[(cond, cat)] = group.pSMAD_med.to_numpy(float)
    _cross_check(values)
    return values


def _cross_check(values: dict) -> None:
    """Assert the plotted values equal the published sheet, when Source Data is available."""
    directory = os.environ.get("SOURCE_DATA_DIR")
    book = Path(directory or REPO / "source_data") / "Source Data Extended Data Fig.3.xlsx"
    if not book.exists():
        print("[check] Source Data workbook not found; plotted values not cross-checked")
        return
    import openpyxl
    sheet = openpyxl.load_workbook(book, read_only=True, data_only=True)["Extended Data Fig.3k"]
    published = {}
    for knockdown, bmp4, _well, value in list(sheet.iter_rows(values_only=True))[3:]:
        if knockdown:
            published.setdefault((knockdown, bmp4), []).append(float(value))
    for cond, knockdown in CONDS:
        for cat, bmp4, _ in CATS:
            assert np.allclose(sorted(published[(knockdown, bmp4)]),
                               sorted(values[(cond, cat)])), (knockdown, bmp4)
    print("[check] plotted values == Source Data sheet 'Extended Data Fig.3k' (12/12)")


def main() -> int:
    values = load_values()
    width, height = CANVAS
    fig = plt.figure(figsize=(width / PPI, height / PPI), dpi=PPI)
    left, top, right, bottom = AXES
    ox, oy = ORIGIN
    ax = fig.add_axes([(left - ox) / width, 1 - (bottom - oy) / height,
                       (right - left) / width, (bottom - top) / height])

    np.random.seed(0)
    for ci, (cond, _) in enumerate(CONDS):
        for k, (cat, _, colour) in enumerate(CATS):
            x = ci - BOX_OFFSET * (1 - k)
            v = values[(cond, cat)]
            ax.boxplot([v], positions=[x], widths=BOX_W, patch_artist=True, manage_ticks=False,
                       boxprops=dict(facecolor=colour, alpha=0.55, linewidth=LW["box"]),
                       medianprops=dict(color="k", linewidth=LW["median"]),
                       whiskerprops=dict(color="#777", linewidth=LW["whisker"]),
                       capprops=dict(color="#777", linewidth=LW["whisker"]), showfliers=False)
            ax.scatter(np.full(len(v), x) + np.random.uniform(-0.05, 0.05, len(v)), v,
                       color=colour, s=MARKER_S, zorder=3, edgecolor="k",
                       linewidth=LW["marker_edge"])

    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_yticks([200, 400, 600, 800])
    ax.set_xticks([ci + XTICK_SHIFT for ci in range(len(CONDS))])
    ax.set_xticklabels([label for _, label in CONDS], rotation=XLABEL_ROT, ha="right",
                       fontsize=FS["xtick"])
    ax.tick_params(axis="x", length=0, pad=XLABEL_PAD)
    ax.tick_params(axis="y", length=TICK_LEN, width=LW["tick"], pad=TICK_PAD,
                   labelsize=FS["ytick"])
    for spine in ax.spines.values():
        spine.set_linewidth(LW["spine"])
    ax.set_ylabel("pSMAD (a.u.)", fontsize=FS["ylabel"])
    ax.yaxis.set_label_coords((YLABEL_RIGHT_PX - left) / (right - left), 0.5)

    handles = [mpatches.Patch(facecolor=CATS[0][2], label=CATS[0][1]),
               mpatches.Patch(facecolor=CATS[1][2], alpha=0.6, label=CATS[1][1])]
    lx, ly = LEGEND_CENTRE
    legend = fig.legend(handles=handles, loc="center",
                        bbox_to_anchor=((lx - ox) / width, 1 - (ly - oy) / height),
                        ncol=2, fontsize=FS["legend"], frameon=True, fancybox=True, **LEGEND)
    legend.get_frame().set_linewidth(LW["legend_frame"])
    for text in legend.get_texts():
        text.set_transform(text.get_transform() + ScaledTranslation(
            LEGEND_NUDGE_PX["dx"] / PPI, -LEGEND_NUDGE_PX["text"] / PPI, fig.dpi_scale_trans))
    for handle in legend.legend_handles:
        # get_data_transform(), not get_transform(): a Patch's own transform already includes its
        # shape, so adding the nudge to it scales the patch a second time.
        handle.set_transform(handle.get_data_transform() + ScaledTranslation(
            LEGEND_NUDGE_PX["dx"] / PPI, -LEGEND_NUDGE_PX["handle"] / PPI, fig.dpi_scale_trans))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{NAME}.pdf")
    fig.savefig(OUT / f"{NAME}.png", dpi=PPI)
    plt.close(fig)
    print(f"wrote figures/output/{NAME}.pdf + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
