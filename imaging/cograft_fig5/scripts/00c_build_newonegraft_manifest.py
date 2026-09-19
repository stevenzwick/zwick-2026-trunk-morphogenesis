#!/usr/bin/env python3
"""Build the manifest for the newly-picked one-graft LPM-induction morphs (Drive
`2026-07-04-pick-nice-one-graph-morph`, 20 CZIs across 3 reporter-config folders) — for the
m5-ii / m4-i SUCCESS (FOXF1+/MESP2+ presence) count.

Channels resolved BY DYE NAME (consistent within each folder per Tianlei): FOXF1=TagYFP,
MESP2=mCherry, DAPI where present (else BF mask). The donor/host channel (EGFP vs A647) is
recorded but its donor-vs-host role is FLAGGED for review (folder names say CFP/iRFP but the
fluor channels are EGFP/A647) — not needed for the success count, only for later host-FOXF1.
Deterministic. Writes results/tables/00_newonegraft_manifest.tsv.

NOT runnable as shipped. This stage reads the raw .czi acquisitions, which are not deposited —
they are available from the lead contact on request, per the paper's Data Availability
Statement. The measurements it produces are committed under the lane's `derived/` directory,
which is what the figure scripts read. See `data/DOWNLOAD.md`.
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
from aicspylibczi import CziFile

WORK = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
BASE = Path("<imaging-data-root>/2026-07-04-pick-nice-one-graft")
DRIVE_MANIFEST = BASE / "drive_manifest.tsv"
OUT = WORK / "results" / "tables" / "00_newonegraft_manifest.tsv"


def dye_channels(czi: CziFile) -> list[str]:
    dims = dict(czi.get_dims_shape()[0]); C = dims["C"][1]
    return [ (ch.get("Name") or "") for ch in czi.meta.findall(".//Channels/Channel")[:C] ]


def idx(names, *needles):
    for i, n in enumerate(names):
        if any(k.lower() in n.lower() for k in needles):
            return i
    return -1


def pixel_um(czi: CziFile) -> float:
    for it in czi.meta.findall(".//Scaling/Items/Distance"):
        if it.get("Id") == "X":
            v = it.findtext("Value")
            if v:
                return float(v) * 1e6                       # metres -> µm
    return float("nan")


rows_in = [r for r in csv.DictReader(open(DRIVE_MANIFEST), delimiter="\t") if not r["folder"].startswith("#")]
out_rows = []
for i, r in enumerate(sorted(rows_in, key=lambda x: (x["folder"], x["filename"])), start=1):
    p = BASE / r["folder"] / r["filename"]
    czi = CziFile(p); names = dye_channels(czi); dims = dict(czi.get_dims_shape()[0])
    ci_egfp, ci_a647 = idx(names, "EGFP"), idx(names, "647")
    # donor/host channels (author-confirmed): iRFP donor = A647; CFP host = EGFP; CFP donor = EGFP.
    # host-noFP_donor-iRFP EGFP = TagYFP bleed / bead autofluor -> ignore (host unlabeled).
    if r["folder"] == "host-CFP_donor-iRFP":
        donor_ch, host_ch = ci_a647, ci_egfp
    elif r["folder"] == "host-noFP_donor-iRFP":
        donor_ch, host_ch = ci_a647, -1
    else:                                                # host-noFP_donor-CFP
        donor_ch, host_ch = ci_egfp, -1
    out_rows.append(dict(
        image_id=f"ng{i:02d}",
        file_path=str(p),
        group="onegraft_new",
        condition="onegraft_pick",
        organoid_label=r["filename"][:-4],
        quality_flag=r["quality_hint"] or "",
        folder=r["folder"],
        ch_bf=idx(names, "bright"),
        ch_dapi=idx(names, "DAPI"),
        ch_mesp2=idx(names, "mCher"),                       # matches both "mCherry" and truncated "mCher"
        ch_foxf1=idx(names, "TagYFP"),
        ch_egfp=ci_egfp,
        ch_a647=ci_a647,
        ch_donor=donor_ch, ch_host=host_ch,                 # author-confirmed (see donor/host logic above)
        ch_lpm_bead=donor_ch,                               # engine donor-exclusion channel = the donor
        ch_nmp_bead=-1,                                      # one-graft: no second donor
        n_channels=dims["C"][1], n_z=dims.get("Z", (0, 1))[1],
        pixel_um_x=round(pixel_um(czi), 5), pixel_um_y=round(pixel_um(czi), 5),
    ))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), delimiter="\t")
    w.writeheader(); w.writerows(out_rows)

print(f"wrote {OUT}  ({len(out_rows)} morphs)")
print(f"{'id':5}{'folder':22}{'label':26}{'nCh':>4}{'nZ':>4}{'BF':>3}{'DAPI':>5}{'MESP2':>6}{'FOXF1':>6}{'EGFP':>5}{'A647':>5}{'px_um':>7}")
for r in out_rows:
    print(f"{r['image_id']:5}{r['folder']:22}{r['organoid_label'][:25]:26}{r['n_channels']:>4}{r['n_z']:>4}"
          f"{r['ch_bf']:>3}{r['ch_dapi']:>5}{r['ch_mesp2']:>6}{r['ch_foxf1']:>6}{r['ch_egfp']:>5}{r['ch_a647']:>5}{r['pixel_um_x']:>7}")
