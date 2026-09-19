#!/usr/bin/env python3
"""Stage + manifest the co-transplantation (Fig 5 / ED Fig 12) confocal set.

m4/m5 image-quant thread. Builds a raw QC manifest over the locally-staged .czi
files (one row per organoid) recording: experiment/day/condition tags, the
embedded CZI dye names, dimensions, voxel size, and the dye->marker mapping
resolved BY DYE NAME (not channel position -- position is NOT uniform: the two
bead channels are swapped between the day4 and day5 acquisitions, and the
`d4_images/` subset lacks DAPI).

Dye -> target legend for THIS experiment (confirm w/ Steven; folder-name channel
order + reporter-line names corroborate):
    TL Brightfield  -> bf        (morphology)
    DAPI            -> dapi       (nuclei / whole-morph mask; ABSENT in d4_images)
    mCherry         -> mesp2      (somite; MESP2-2A-mCherry reporter)
    TagYFP          -> foxf1      (LPM; FOXF1-YFP reporter -- the induced readout)
    EGFP            -> lpm_bead   (LPM donor population)
    Alexa Fluor 647 -> nmp_bead   (NMP donor population)

Raw images are staged OUTSIDE iCloud; analysis outputs
live in the iCloud-backed work dir. Reads only; never edits the Drive source.

NOT runnable as shipped. This stage reads the raw .czi acquisitions, which are not deposited —
they are available from the lead contact on request, per the paper's Data Availability
Statement. The measurements it produces are committed under the lane's `derived/` directory,
which is what the figure scripts read. See `data/DOWNLOAD.md`.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

# czifile cannot be installed beside this project's pinned numpy, and mis-frames odd-sized
# acquisitions. czi_compat presents czifile's interface over libCZI. See src/trunk_morph_ref/czi_compat.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from src.trunk_morph_ref import czi_compat as czifile
import os
import numpy as np
import pandas as pd

# ---- locations -------------------------------------------------------------
# ---- locations -------------------------------------------------------------
# STAGING_ROOT points at the raw .czi acquisitions. They are NOT deposited: they are available
# from the lead contact on request, per the paper's Data Availability Statement. Set
# COGRAFT_RAW_ROOT to wherever you have put them.
STAGING_ROOT = Path(os.environ.get("COGRAFT_RAW_ROOT", "raw/cotransplant_fig5_quant_2026-06-27"))
WORK_DIR = Path(os.environ.get("COGRAFT_WORK_DIR", "results/cograft_fig5"))
OUT_TSV = WORK_DIR / "results" / "tables" / "00_raw_manifest.tsv"

# ---- channels --------------------------------------------------------------
# Resolved BY DYE NAME, never by channel position: position is not uniform across this set (the
# two bead channels are swapped between the day 4 and day 5 acquisitions, and the `d4_images/`
# subset has no DAPI). The keys are the dye strings the CZI files carry in their own metadata;
# the legend they encode is in this script's docstring. `01_channel_qc_montage.py` uses the same
# marker order.
MARKER_ORDER = ["bf", "dapi", "mesp2", "foxf1", "lpm_bead", "nmp_bead"]
DYE_TO_MARKER = {
    "TL Brightfield": "bf",         # morphology
    "DAPI": "dapi",                 # nuclei / whole-morph mask; absent in `d4_images/`
    "mCherry": "mesp2",             # somite; MESP2-2A-mCherry reporter
    "TagYFP": "foxf1",              # LPM; FOXF1-YFP reporter, the induced readout
    "EGFP": "lpm_bead",             # LPM donor population
    "Alexa Fluor 647": "nmp_bead",  # NMP donor population
}


def _channels_and_scale(path: Path):
    with czifile.CziFile(path) as czi:
        raw = czi.asarray()
        axes = str(czi.axes)
        meta = czi.metadata(raw=False)
    c_count = raw.shape[axes.index("C")] if "C" in axes else 1
    z_count = raw.shape[axes.index("Z")] if "Z" in axes else 1
    y_count = raw.shape[axes.index("Y")]
    x_count = raw.shape[axes.index("X")]
    dims = (meta.get("ImageDocument", {}).get("Metadata", {}).get("Information", {})
            .get("Image", {}).get("Dimensions", {}))
    chan = dims.get("Channels", {}).get("Channel")
    chan = chan if isinstance(chan, list) else ([chan] if chan else [])
    names = []
    for i, ch in enumerate(chan):
        if isinstance(ch, dict):
            names.append(str(ch.get("ShortName") or ch.get("Name") or ch.get("Fluor")
                              or ch.get("DyeName") or f"Channel_{i}"))
        else:
            names.append(f"Channel_{i}")
    names = names[:c_count] + [f"Channel_{i}" for i in range(len(names), c_count)]
    scale = {"X": np.nan, "Y": np.nan, "Z": np.nan}
    items = (meta.get("ImageDocument", {}).get("Metadata", {}).get("Scaling", {})
             .get("Items", {}).get("Distance"))
    for it in (items if isinstance(items, list) else ([items] if items else [])):
        if isinstance(it, dict):
            ax = str(it.get("Id", "")).upper()
            try:
                if ax in scale:
                    scale[ax] = float(it.get("Value")) * 1e6
            except (TypeError, ValueError):
                pass
    return names, int(c_count), int(z_count), int(y_count), int(x_count), scale


def _tags(path: Path):
    rel = path.relative_to(STAGING_ROOT).as_posix()
    name = path.name
    if "d4_images" in rel:
        group, experiment, fix_day, imaged_day = "d4images_exp1", "exp1", "day4", "day4"
    elif rel.startswith("fix_day4_exp2"):
        group, experiment, fix_day, imaged_day = "day4_exp2", "exp2", "day4", "day4"
    else:
        group, experiment, fix_day, imaged_day = "day5_exp1", "exp1", "day5", "day5"
    condition = "neg_ctrl" if name.lower().startswith("neg") else "cograft"
    low = name.lower()
    # negative-quality notes win over a bare "good" substring (e.g. "may not very good")
    if "may not" in low or "maybe" in low or "not very good" in low:
        flag = "flagged_quality"
    elif "different direction" in low:
        flag = "orientation_note"
    elif "good" in low:
        flag = "good"
    else:
        flag = ""
    paren = re.search(r"\(([^)]*)\)", name)
    return group, experiment, fix_day, imaged_day, condition, flag, (paren.group(1) if paren else "")


def main() -> None:
    print("Staging pending downloads ...")
    files = sorted(STAGING_ROOT.rglob("*.czi"))
    if not files:
        # Without this, an absent raw tree produces an empty DataFrame and the sort below raises
        # a bare `KeyError: 'group'` from pandas internals -- which tells a reader nothing about
        # the actual problem, that no input was found.
        raise SystemExit(
            f"No .czi files found under {STAGING_ROOT}\n"
            "The raw acquisitions are not in this repository; see data/DOWNLOAD.md. Point\n"
            "COGRAFT_RAW_ROOT at wherever you have put them:\n"
            "    COGRAFT_RAW_ROOT=/path/to/raw python imaging/cograft_fig5/scripts/00_build_raw_manifest.py")
    print(f"\nFound {len(files)} staged .czi files. Reading metadata ...")
    rows = []
    for path in files:
        names, c, z, y, x, scale = _channels_and_scale(path)
        group, exp, fix_day, imaged, cond, flag, paren = _tags(path)
        marker_idx = {}
        for i, dye in enumerate(names):
            m = DYE_TO_MARKER.get(dye)
            if m and m not in marker_idx:
                marker_idx[m] = i
        row = {
            "organoid_label": path.stem,
            "group": group, "experiment": exp, "fix_day": fix_day,
            "imaged_day": imaged, "condition": cond,
            "quality_flag": flag, "filename_note": paren,
            "size_c": c, "size_z": z, "size_y": y, "size_x": x,
            "pixel_um_x": round(scale["X"], 4), "pixel_um_y": round(scale["Y"], 4),
            "pixel_um_z": round(scale["Z"], 4),
            "raw_dyes": ";".join(names),
            "has_dapi": "dapi" in marker_idx,
        }
        for m in MARKER_ORDER:
            row[f"ch_{m}"] = marker_idx.get(m, -1)
        row["unmapped_dyes"] = ";".join(d for d in names if d not in DYE_TO_MARKER)
        row["file_path"] = str(path)
        rows.append(row)

    df = pd.DataFrame(rows)
    order = {"day4_exp2": 0, "d4images_exp1": 1, "day5_exp1": 2}
    df = df.sort_values(by=["group", "organoid_label"],
                        key=lambda s: s.map(order) if s.name == "group" else s).reset_index(drop=True)
    df.insert(0, "image_id", [f"img{i+1:02d}" for i in range(len(df))])
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(f"\nWrote manifest -> {OUT_TSV}\n")
    cols = ["image_id", "organoid_label", "group", "condition", "quality_flag",
            "size_c", "size_z", "has_dapi", "ch_mesp2", "ch_foxf1", "ch_lpm_bead", "ch_nmp_bead"]
    print(df[cols].to_string(index=False))
    print("\nGroup x condition counts:")
    print(df.groupby(["group", "condition"]).size().to_string())
    print("\nChannel-count / DAPI summary:")
    print(df.groupby(["group", "size_c", "has_dapi"]).size().to_string())
    bad = df[df["unmapped_dyes"] != ""]
    if len(bad):
        print("\n!! files with unmapped dyes:")
        print(bad[["organoid_label", "raw_dyes", "unmapped_dyes"]].to_string(index=False))


if __name__ == "__main__":
    main()
