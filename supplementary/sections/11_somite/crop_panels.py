"""
Crop pass for SD4 Section 11 (author-approved). Post-processes the SAVED bundle PNGs — NO analysis
code is rerun. Removes the baked-in matplotlib figure titles (they duplicate the caption), drops the
redundant display-only "rank DPT" 3rd sub-panel from the 11.3 embedding strips (enlarges the rest), and
tight-crops outer whitespace. Sources are READ-ONLY; crops are written to ./derived/.

NOT runnable as shipped. The Supplementary Data builders embed pre-rendered figure assets that
are not in this repository, so the build stops at the first missing image. The built PDFs are
published with the paper as Supplementary Data, so a reader already has the output; what is
absent is the ability to rebuild it. This file ships as the record of how the section was
assembled. See `supplementary/README.md`.
"""
from PIL import Image
import numpy as np
from pathlib import Path

SRC = Path("<analysis-root>/"
           "experiments/rq1_4_somite_progression/results/supplement_display_bundle/figures")
OUT = Path(__file__).resolve().parent / "derived"
OUT.mkdir(exist_ok=True)

INK = 205  # < INK == "ink" (dark) pixel


def title_cut_central(g):
    """Row to crop from top, using only central columns so a side legend/colorbar (full-height) doesn't
    mask the white gap under the centered title."""
    H, W = g.shape
    c0, c1 = int(0.30 * W), int(0.70 * W)
    row_ink = (g[:, c0:c1] < INK).mean(axis=1)
    inked = row_ink > 0.004
    idx = np.where(inked)[0]
    if len(idx) == 0 or idx[0] > 0.30 * H:
        return 0
    top = idx[0]
    gap_h = max(4, int(0.010 * H))
    y, run = top, 0
    while y < 0.42 * H:
        if not inked[y]:
            run += 1
            if run >= gap_h:
                z = y
                while z < H and not inked[z]:
                    z += 1
                return z
        else:
            run = 0
        y += 1
    # fallback (title line hugs the plot, e.g. paga_psm): take the LARGEST white gap that
    # starts within the title zone (top 32% H); the title->plot separation is the biggest gap there.
    lim, best, y = int(0.42 * H), None, top
    while y < lim:
        if not inked[y]:
            z = y
            while z < lim and not inked[z]:
                z += 1
            if y < 0.32 * H and (best is None or (z - y) > best[0]):
                best = (z - y, z)
            y = z
        else:
            y += 1
    return best[1] if best and best[0] >= 6 else 0


def spine_cut(g, min_run_frac=0.15, search_frac=0.35):
    """Row just above the plot's top axes spine (the first long continuous dark horizontal run). Used for
    the multi-line-titled embedding strips where the inter-title-line gap fools the white-gap detector;
    UMAP/diffusion-map panels have no group brackets above the box, so cutting at the spine is safe."""
    H, W = g.shape
    lim, need = int(search_frac * H), int(min_run_frac * W)
    dark = g < INK
    for y in range(lim):
        idx = np.where(~dark[y])[0]
        runmax = W if len(idx) == 0 else int(np.max(np.diff(np.concatenate(([-1], idx, [W])))) - 1)
        if runmax >= need:
            return max(0, y - 4)
    return 0


def internal_white_gutters(g, min_w_frac=0.02):
    """Maximal runs of near-white columns wider than min_w_frac*W, away from the outer margins."""
    H, W = g.shape
    col_ink = (g < INK).mean(axis=0)
    white = col_ink < 0.004
    runs, i = [], 0
    while i < W:
        if white[i]:
            j = i
            while j < W and white[j]:
                j += 1
            if (j - i) >= min_w_frac * W and i > 0.10 * W and j < 0.96 * W:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def tight_bbox(g, pad=8):
    """Content bounding box (non-white), with a small pad."""
    H, W = g.shape
    rows = np.where((g < 250).any(axis=1))[0]
    cols = np.where((g < 250).any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, W, H
    t, b = max(0, rows[0] - pad), min(H, rows[-1] + 1 + pad)
    l, r = max(0, cols[0] - pad), min(W, cols[-1] + 1 + pad)
    return l, t, r, b


DROP_RANK = {"RefQ1_4_umap_somite_progression_dpt", "RefQ1_4_diffmap_somite_progression_dpt"}

PANELS = ["RefQ1_4_umap_lineage_context", "RefQ1_4_dotplot_somite_progression_coarse",
          "RefQ1_4_dotplot_somite_progression_subclusters", "RefQ1_4_dotplot_somite_superstates",
          "RefQ1_4_umap_somite_progression_dpt", "RefQ1_4_diffmap_somite_progression_dpt",
          "RefQ1_4_boxplot_dpt_somite_superstates_rank",
          "RefQ1_4_umap_psm_somite_progression_dpt", "RefQ1_4_diffmap_psm_somite_progression_dpt",
          "RefQ1_4_boxplot_dpt_psm_somite_superstates",
          "RefQ1_4_paga_somite_superstate_connectivity_heatmap",
          "RefQ1_4_paga_psm_somite_superstate_connectivity_heatmap"]

for n in PANELS:
    im = Image.open(SRC / f"{n}.png").convert("RGB")
    g = np.asarray(im.convert("L"))
    H, W = g.shape
    # embedding strips: cut at the top spine (multi-line titles fool the white-gap detector); others:
    # first white gap under the centered title (preserves dot-plot group brackets that sit above the box).
    tcut = spine_cut(g) if n in DROP_RANK else title_cut_central(g)
    note = f"title@{tcut}({100*tcut/H:.0f}%)"
    # drop the redundant 3rd (rank) sub-panel on the 11.3 strips
    right = W
    if n in DROP_RANK:
        gutters = internal_white_gutters(g[tcut:])
        if gutters:
            right = gutters[-1][0]
            note += f"  dropRankAt={right}  gutters={[(a,b) for a,b in gutters]}"
    crop = im.crop((0, tcut, right, H))
    # tight-crop remaining whitespace
    gg = np.asarray(crop.convert("L"))
    l, t, r, b = tight_bbox(gg)
    crop = crop.crop((l, t, r, b))
    crop.save(OUT / f"{n}.png")
    print(f"{im.size} -> {crop.size}   {note}   {n}")

print("\nderived crops written to", OUT)
