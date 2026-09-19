#!/usr/bin/env python3
"""Canonical-marker QC montage for the co-transplantation set.

Eyes-on-data-first: renders, per selected organoid, the Z-max-projection of each
channel placed in a column fixed by ASSIGNED MARKER (bf, dapi, mesp2, foxf1,
lpm_bead, nmp_bead), so columns are biologically aligned across organoids despite
the per-acquisition channel-order swap. Each panel is titled with the actual CZI
dye, so the dye->marker legend can be confirmed visually (does mCherry look like
somites? does TagYFP look like LPM? do EGFP / AF647 sit on the grafts?).

This artifact is FOR STEVEN's image-quality + legend judgement (his lane).

Usage:
    python scripts/01_channel_qc_montage.py            # representative 4
    python scripts/01_channel_qc_montage.py all        # all 18
    python scripts/01_channel_qc_montage.py img01,img07 # specific

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

MARKER_ORDER = ["bf", "dapi", "mesp2", "foxf1", "lpm_bead", "nmp_bead"]
MARKER_DISPLAY = {
    "bf": "BF", "dapi": "DAPI", "mesp2": "MESP2\n(mCherry)", "foxf1": "FOXF1\n(TagYFP)",
    "lpm_bead": "LPM-bead\n(EGFP)", "nmp_bead": "NMP-bead\n(AF647)",
}
MARKER_CMAP = {
    "bf": "gray", "dapi": "gray", "mesp2": "Reds", "foxf1": "Greens",
    "lpm_bead": "YlGn", "nmp_bead": "RdPu",
}
REPRESENTATIVE = ["img01", "img04", "img07", "img18"]


def load_cyx_maxproj(path: Path) -> np.ndarray:
    """Return (C, Y, X) Z-max-projection from a .czi of arbitrary axis order."""
    with czifile.CziFile(path) as czi:
        arr = np.asarray(czi.asarray())
        axes = str(czi.axes)
    ci = axes.index("C")
    arr = np.moveaxis(arr, ci, 0)
    axes = "C" + axes[:ci] + axes[ci + 1:]
    keep = {"C", "Y", "X"}
    while any(c not in keep for c in axes):
        for j, c in enumerate(axes):
            if c not in keep:
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


def render(df: pd.DataFrame, out_path: Path) -> None:
    n = len(df)
    ncol = len(MARKER_ORDER)
    fig, axes = plt.subplots(n, ncol, figsize=(2.5 * ncol, 2.7 * n), squeeze=False)
    for c, m in enumerate(MARKER_ORDER):
        axes[0][c].set_title(MARKER_DISPLAY[m], fontsize=10)
    for r, (_, row) in enumerate(df.iterrows()):
        cyx = load_cyx_maxproj(Path(row["file_path"]))
        for c, m in enumerate(MARKER_ORDER):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            ch = int(row[f"ch_{m}"])
            if ch < 0 or ch >= cyx.shape[0]:
                ax.text(0.5, 0.5, "absent", ha="center", va="center",
                        color="0.6", fontsize=9, transform=ax.transAxes)
                ax.set_facecolor("0.95")
                continue
            ax.imshow(stretch(cyx[ch]), cmap=MARKER_CMAP[m], vmin=0, vmax=1)
        flag = f"  [{row['quality_flag']}]" if row["quality_flag"] else ""
        axes[r][0].set_ylabel(f"{row['image_id']}\n{row['organoid_label'][:22]}\n"
                              f"{row['group']}/{row['condition']}{flag}",
                              fontsize=7.5, rotation=0, ha="right", va="center", labelpad=42)
    fig.suptitle("Co-transplant channel QC — columns = assigned marker (panel title = CZI dye); "
                 "Z-max-proj, per-panel 1–99.5% contrast", fontsize=11, y=0.997)
    fig.tight_layout(rect=(0.04, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    df = pd.read_csv(MANIFEST, sep="\t")
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "all":
        sel, tag = df, "all18"
    elif arg:
        ids = [s.strip() for s in arg.split(",")]
        sel, tag = df[df["image_id"].isin(ids)], "_".join(ids)
    else:
        sel, tag = df[df["image_id"].isin(REPRESENTATIVE)], "representative"
    sel = sel.sort_values("image_id")
    render(sel, QC_DIR / f"01_channel_qc_{tag}.png")


if __name__ == "__main__":
    main()
