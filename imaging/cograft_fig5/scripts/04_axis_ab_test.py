#!/usr/bin/env python3
"""Is (b) [max-proj medial axis] OK vs (a) [per-z consensus medial axis] for bilaterality?

Bilaterality is axis-sensitive (medial-vs-PCA test showed mean|Δ|~0.2, max~0.7),
so we must check the ACTUAL (b)-vs-(a) difference, not the extreme medial-vs-PCA one.
Here we hold the domain masks fixed (max-projection positive masks) and vary ONLY
the axis: (b) centerline of the max-proj mask, vs (a) consensus of per-z centerlines
(the day-5 method). If the per-organoid bilaterality barely moves, (b) is fine.

DAPI-bearing organoids only, to keep masks clean and isolate the axis effect.

NOT runnable as shipped. This stage reads the raw .czi acquisitions, which are not deposited —
they are available from the lead contact on request, per the paper's Data Availability
Statement. The measurements it produces are committed under the lane's `derived/` directory,
which is what the figure scripts read. See `data/DOWNLOAD.md`.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# The morphology helpers ship in this repository, under the morphometry lane.
MORPHO = str(Path(__file__).resolve().parents[2] / "morphometry" / "scripts")
sys.path.insert(0, MORPHO)
import morphology_quantification_helpers as M
import functools as _ft, inspect as _insp
from skimage import morphology as _skm
if "rng" in _insp.signature(_skm.medial_axis).parameters:
    _skm.medial_axis = _ft.partial(_skm.medial_axis, rng=0)

WORK = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
man = pd.read_csv(WORK / "results/tables/00_raw_manifest.tsv", sep="\t")
thr_df = pd.read_csv(WORK / "results/tables/03_marker_thresholds.tsv", sep="\t")
THR = {r["marker"]: float(r["threshold"]) for _, r in thr_df.iterrows()}
MARKERS = ["mesp2", "foxf1"]
SUBSET = ["img01", "img02", "img03", "img12", "img13", "img16", "img17", "img18"]


def bilat(pos, corr, axis_xy):
    ys, xs = np.where(pos)
    if xs.size < 3 or axis_xy is None or len(axis_xy) < 2:
        return np.nan
    s = np.asarray(M.project_points_to_centerline(np.column_stack([xs, ys]).astype(float),
                                                  axis_xy)["signed_transverse_px"])
    lc, rc = int((s < -0.5).sum()), int((s > 0.5).sum())
    return (1.0 - abs(rc - lc) / (lc + rc)) if (lc + rc) else np.nan


rows = []
for iid in SUBSET:
    row = man[man["image_id"] == iid].iloc[0]
    stack = M.load_czi_stack(Path(row["file_path"]))
    di = int(row["ch_dapi"])
    nz = int(stack.data_czyx.shape[1])
    maxmask = M.segment_whole_morph(stack.data_czyx[di].max(axis=0))
    axis_b = M.centerline_from_mask(maxmask)["centerline_xy"]
    per_z = []
    for z in range(nz):
        mz = M.segment_whole_morph(np.asarray(stack.data_czyx[di, z], dtype=np.float32))
        if mz.sum() > 200:
            try:
                per_z.append(M.centerline_from_mask(mz)["centerline_xy"])
            except Exception:
                pass
    axis_a = (M.consensus_centerline_from_paths(per_z, reference_mask=maxmask)["centerline_xy"]
              if per_z else axis_b)
    rec = {"image_id": iid, "group": row["group"], "n_z_used": len(per_z)}
    for m in MARKERS:
        corr, _, _ = M.background_correct_signal(stack.data_czyx[int(row[f"ch_{m}"])].max(axis=0),
                                                 maxmask, background_estimator="whole_off_morph")
        pos = M.positive_mask_from_corrected(corr, maxmask, THR[m])
        rec[f"{m}_b"] = bilat(pos, corr, axis_b)
        rec[f"{m}_a"] = bilat(pos, corr, axis_a)
        rec[f"{m}_d"] = abs(rec[f"{m}_b"] - rec[f"{m}_a"])
    rows.append(rec)

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
print("\n(b)=max-proj medial   (a)=per-z consensus medial")
for m in MARKERS:
    print(f"  {m}: mean|Δ(a,b)| = {df[f'{m}_d'].mean():.3f}   max|Δ| = {df[f'{m}_d'].max():.3f}")
