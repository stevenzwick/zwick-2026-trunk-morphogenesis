#!/usr/bin/env python3
"""Clean per-channel mask QC for the new one-graft morphs — ONE channel + ONE mask per panel
(the multi-overlay donor review shows too many things at once). Columns: morph mask (on BF/DAPI),
MESP2+ (on MESP2), FOXF1+ (on FOXF1), donor (on donor channel), host FOXF1 (on host channel).
Per-image null thresholds (each morph judged against its OWN background) -> fixes the no-DAPI
under-detection. One figure per reporter-config folder.

NOT runnable as shipped. This stage reads the raw .czi acquisitions, which are not deposited —
they are available from the lead contact on request, per the paper's Data Availability
Statement. The measurements it produces are committed under the lane's `derived/` directory,
which is what the figure scripts read. See `data/DOWNLOAD.md`.
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

WORK = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
spec = importlib.util.spec_from_file_location("eng", WORK / "scripts/05_cograft_quant_perz.py")
eng = importlib.util.module_from_spec(spec); spec.loader.exec_module(eng)
M, czi_io = eng.M, eng.czi_io
QC = WORK / "results" / "qc"
MESP2_SIGMA, FOXF1_SIGMA, FOXF1_SCALE = 3.0, 4.0, 1.3


def perimage_threshold(corr, mask, sigma):
    v = corr[np.asarray(mask, bool) & np.isfinite(corr)].astype(np.float64)
    if v.size < 50:
        return np.inf
    lo, hi = np.percentile(v, [0.1, 99.9]);  hi = hi if hi > lo else lo + 1.0
    edges = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 2049)
    counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
    return float(M.half_gaussian_threshold_from_histogram(counts=counts, edges=edges, z=sigma)["threshold_value"])


def marker_mask(stack, row, dmask, ci_col, sigma, scale):
    mp = np.asarray(stack.data_czyx[int(row[ci_col])].max(axis=0), np.float32)
    corr, _, _ = M.background_correct_signal(mp, dmask, background_estimator="whole_off_morph")
    thr = perimage_threshold(corr, dmask, sigma) * scale
    return M.positive_mask_from_corrected(corr, dmask, thr)


df = pd.read_csv(WORK / "results/tables/00_newonegraft_manifest.tsv", sep="\t")
COLS = ["morph", "MESP2", "FOXF1", "donor", "host"]
for folder in sorted(df.folder.unique()):
    sub = df[df.folder == folder].reset_index(drop=True)
    fig, axes = plt.subplots(len(sub), len(COLS), figsize=(2.6 * len(COLS), 2.7 * len(sub)), squeeze=False)
    for k, row in sub.iterrows():
        stack = czi_io.load_czi_stack(Path(row.file_path))
        di, nz = int(row.ch_dapi), stack.data_czyx.shape[1]
        dmask = eng.domain_mask(stack, row, di, None, 0.5 if di >= 0 else 0.7, 4000, nz, 0)
        mesp2 = marker_mask(stack, row, dmask, "ch_mesp2", MESP2_SIGMA, 1.0)
        foxf1 = marker_mask(stack, row, dmask, "ch_foxf1", FOXF1_SIGMA, FOXF1_SCALE)
        donor_chs = eng.donor_channels_for_row(row)
        dmsk = eng.donor_mask(stack, row, dmask, donor_chs, di, 1.0, 8, 0.005, True)[0] if donor_chs else np.zeros(dmask.shape, bool)
        host = foxf1 & ~dmsk
        ci_host = int(row.ch_host)
        panels = {"morph": (di if di >= 0 else int(row.ch_bf), dmask, "cyan"),
                  "MESP2": (int(row.ch_mesp2), mesp2, "red"),
                  "FOXF1": (int(row.ch_foxf1), foxf1, "lime"),
                  "donor": (int(row.ch_donor), dmsk, "magenta"),
                  "host":  (ci_host, host, "yellow")}
        for j, c in enumerate(COLS):
            ax = axes[k][j]; ax.set_xticks([]); ax.set_yticks([])
            ci, msk, color = panels[c]
            if ci < 0:
                ax.set_facecolor("#111"); ax.text(0.5, 0.5, "n/a", color="#666", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.imshow(M.robust_rescale(np.asarray(stack.data_czyx[ci].max(axis=0), np.float32)), cmap="gray")
                if msk is not None and msk.any():
                    ax.contour(msk, colors=color, linewidths=1.0)
            if k == 0:
                ax.set_title({"morph": "morph mask", "MESP2": "MESP2 + mask", "FOXF1": "FOXF1 + mask",
                              "donor": "donor + mask", "host": "host FOXF1"}[c], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{row.image_id}\n{row.organoid_label[:14]}", fontsize=7.5, rotation=0, ha="right", va="center", labelpad=30)
    fig.suptitle(f"Per-channel masks — {folder}   (one channel + one mask per panel; per-image threshold)",
                 fontsize=12, y=1 - 0.11 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    out = QC / f"00e_perchannel_masks__{folder}.png"; fig.savefig(out, dpi=100); plt.close(fig); print("wrote", out)
