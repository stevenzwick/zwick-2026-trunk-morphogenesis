#!/usr/bin/env python3
"""Manifest for the ONE-graft (single LPM donor) images = `../LPM_transplant/data/`,
so the co-graft engine (05_cograft_quant_perz.py) can score MATCHED metrics on them
(--manifest=results/tables/00_onegraft_manifest.tsv --outtag=_onegraft).

Fixed channel order (LPM_transplant README): 0 BF · 1 DAPI · 2 MESP2 · 3 FOXF1 · 4 donor · 5 host.
No LPM/NMP beads here -> ch_lpm_bead/ch_nmp_bead = -1 (engine skips bead exclusion).
Some files are ImageJ-TIF-hyperstacks-as-.czi -> handled by czi_io's tifffile fallback.
Pixel size: extracted from CZI scaling / TIF tags where possible, else PLACEHOLDER (flagged);
PLACEHOLDER length/area are provisional — only presence/bilaterality are size-independent.
"""
from pathlib import Path
import sys

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import czi_io  # noqa: E402
from aicspylibczi import CziFile  # noqa: E402
import tifffile  # noqa: E402

LT = Path("<analysis-root>/LPM_transplant/data")
OUT = Path(__file__).resolve().parent.parent / "results" / "tables" / "00_onegraft_manifest.tsv"
CONDS = ["ctrl", "BMPR1A_KD", "LDN"]
PLACEHOLDER_PX = 0.65


def pixel_um(path: Path):
    try:                                              # true CZI scaling (metres -> um)
        meta = CziFile(str(path)).meta
        for dist in meta.findall(".//Scaling/Items/Distance"):
            if dist.get("Id") == "X":
                v = dist.find("Value")
                if v is not None and v.text:
                    return float(v.text) * 1e6, "czi_scaling"
    except Exception:
        pass
    try:                                              # ImageJ TIF resolution tag
        with tifffile.TiffFile(str(path)) as tf:
            tags = tf.pages[0].tags
            unit = str(tags["ResolutionUnit"].value) if "ResolutionUnit" in tags else ""
            if "XResolution" in tags:
                num, den = tags["XResolution"].value
                if num:
                    return den / num, f"tif_xres({unit})"
    except Exception:
        pass
    return None, "PLACEHOLDER"


rows = []
for cond in CONDS:
    for i, f in enumerate(sorted(list((LT / cond).glob("*.czi")) + list((LT / cond).glob("*.tif"))), start=1):
        s = czi_io.load_czi_stack(f)
        C, Z = int(s.data_czyx.shape[0]), int(s.data_czyx.shape[1])
        px, src = pixel_um(f)
        px = px if px else PLACEHOLDER_PX
        rows.append(dict(
            image_id=f"og_{cond.lower()}_{i}", organoid_label=f.stem[:44], group="onegraft",
            experiment="LPM_transplant", condition=f"onegraft_{cond.lower()}", quality_flag="",
            size_c=C, size_z=Z, pixel_um_x=px, pixel_um_y=px, pixel_um_source=src,
            ch_bf=0, ch_dapi=1, ch_mesp2=2, ch_foxf1=3, ch_lpm_bead=-1, ch_nmp_bead=-1,
            file_path=str(f)))
        print(f"{rows[-1]['image_id']:14s} C={C} Z={Z} px={px:.4f}({src})  {f.name[:44]}")

# Extra successful one-graft morphs stained for PAX8 (downloaded from Drive `Ref_1.1b(done) /
# (done) iii(induce LPM) / organoid 3,4 (PAX8 stained)`). "trunk no color" = no host CFP; 6 ch
# BF/DAPI/MESP2/FOXF1/PAX8/CFP-donor (confirmed by eye) -> DAPI/MESP2/FOXF1 at ch1/2/3 like ctrl;
# PAX8(ch4)/donor(ch5) unused in the FOXF1+MESP2 quant.
PAX8_DIR = Path("<imaging-data-root>/cotransplant_fig5_quant_2026-06-27/onegraft_pax8")
for i, f in enumerate(sorted(PAX8_DIR.glob("*.tif")), start=1):
    s = czi_io.load_czi_stack(f)
    C, Z = int(s.data_czyx.shape[0]), int(s.data_czyx.shape[1])
    px, src = pixel_um(f)
    px = px if px else PLACEHOLDER_PX
    rows.append(dict(
        image_id=f"og_pax8_{i}", organoid_label=f.stem[:44], group="onegraft",
        experiment="LPM_transplant_pax8", condition="onegraft_pax8", quality_flag="",
        size_c=C, size_z=Z, pixel_um_x=px, pixel_um_y=px, pixel_um_source=src,
        ch_bf=0, ch_dapi=1, ch_mesp2=2, ch_foxf1=3, ch_lpm_bead=-1, ch_nmp_bead=-1,
        ch_donor="5",  # CFP donor (host FOXF1 = FOXF1+ minus this)
        file_path=str(f)))
    print(f"{rows[-1]['image_id']:14s} C={C} Z={Z} px={px:.4f}({src})  {f.name[:44]}")

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
print("wrote", OUT, "| rows:", len(rows))
