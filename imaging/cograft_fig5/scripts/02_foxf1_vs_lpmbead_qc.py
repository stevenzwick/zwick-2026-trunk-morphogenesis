#!/usr/bin/env python3
"""Focused QC: is EGFP (LPM-bead) the SAME signal as TagYFP (FOXF1), or distinct?

These two reporters are spectrally close (GFP ~509 nm, YFP ~527 nm), so before
trusting a donor-vs-host (m5-iv) split we need to see whether the EGFP channel is
(a) genuine donor-LPM marking that is a SUBSET / different region than total
FOXF1, or (b) a near-copy of FOXF1 (co-expression or bleed-through).

Per organoid: BF | FOXF1 (TagYFP) | LPM-bead (EGFP) | green/magenta merge
(FOXF1=green, LPM-bead=magenta -> co-localized = white). A rough foreground
Pearson r between the two channels is annotated as a quick indicator only.
Interpretation is Steven's.

NOT runnable as shipped. This stage reads the raw .czi acquisitions, which are not deposited —
they are available from the lead contact on request, per the paper's Data Availability
Statement. The measurements it produces are committed under the lane's `derived/` directory,
which is what the figure scripts read. See `data/DOWNLOAD.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# czifile cannot be installed beside this project's pinned numpy, and mis-frames odd-sized
# acquisitions. czi_compat presents czifile's interface over libCZI. See src/trunk_morph_ref/czi_compat.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from src.trunk_morph_ref import czi_compat as czifile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORK_DIR = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
MANIFEST = WORK_DIR / "results" / "tables" / "00_raw_manifest.tsv"
QC_DIR = WORK_DIR / "results" / "qc"
DEFAULT_IDS = ["img01", "img07", "img18"]


def load_cyx_maxproj(path: Path) -> np.ndarray:
    with czifile.CziFile(path) as czi:
        arr = np.asarray(czi.asarray())
        axes = str(czi.axes)
    ci = axes.index("C")
    arr = np.moveaxis(arr, ci, 0)
    axes = "C" + axes[:ci] + axes[ci + 1:]
    while any(c not in {"C", "Y", "X"} for c in axes):
        for j, c in enumerate(axes):
            if c not in {"C", "Y", "X"}:
                arr = np.squeeze(arr, axis=j) if arr.shape[j] == 1 else np.max(arr, axis=j)
                axes = axes[:j] + axes[j + 1:]
                break
    yi, xi = axes.index("Y"), axes.index("X")
    return np.transpose(arr, (0, yi, xi)).astype(np.float32)


def stretch(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def foreground_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over pixels in the brighter foreground of either channel."""
    fg = (a > np.percentile(a, 90)) | (b > np.percentile(b, 90))
    if fg.sum() < 50:
        return float("nan")
    av, bv = a[fg].astype(np.float64), b[fg].astype(np.float64)
    if av.std() < 1e-9 or bv.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def main() -> None:
    df = pd.read_csv(MANIFEST, sep="\t")
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    ids = [s.strip() for s in arg.split(",")] if arg else DEFAULT_IDS
    sel = df[df["image_id"].isin(ids)].sort_values("image_id")

    cols = ["BF", "FOXF1 (TagYFP)", "LPM-bead (EGFP)", "merge: FOXF1=green / LPM-bead=magenta"]
    n = len(sel)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 4, 4.2 * n), squeeze=False)
    for c, t in enumerate(cols):
        axes[0][c].set_title(t, fontsize=11)
    for r, (_, row) in enumerate(sel.iterrows()):
        cyx = load_cyx_maxproj(Path(row["file_path"]))
        bf = stretch(cyx[int(row["ch_bf"])])
        foxf1 = stretch(cyx[int(row["ch_foxf1"])])
        lpm = stretch(cyx[int(row["ch_lpm_bead"])])
        merge = np.dstack([lpm, foxf1, lpm])  # R=lpm, G=foxf1, B=lpm
        r_fg = foreground_pearson(cyx[int(row["ch_foxf1"])], cyx[int(row["ch_lpm_bead"])])
        for c, im, cm in [(0, bf, "gray"), (1, foxf1, "Greens"), (2, lpm, "RdPu")]:
            axes[r][c].imshow(im, cmap=cm, vmin=0, vmax=1)
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        axes[r][3].imshow(merge); axes[r][3].set_xticks([]); axes[r][3].set_yticks([])
        axes[r][3].text(0.02, 0.98, f"foreground r = {r_fg:.2f}", transform=axes[r][3].transAxes,
                        color="white", fontsize=10, va="top",
                        bbox=dict(facecolor="black", alpha=0.5, pad=2))
        axes[r][0].set_ylabel(f"{row['image_id']}\n{row['organoid_label'][:20]}\n{row['group']}",
                              fontsize=9, rotation=0, ha="right", va="center", labelpad=46)
    fig.suptitle("FOXF1 (TagYFP) vs LPM-bead (EGFP) — same signal or distinct?  Z-max-proj",
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=(0.05, 0, 1, 0.985))
    QC_DIR.mkdir(parents=True, exist_ok=True)
    out = QC_DIR / "02_foxf1_vs_lpmbead.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
