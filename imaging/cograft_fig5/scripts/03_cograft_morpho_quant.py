#!/usr/bin/env python3
"""Co-transplant morphological quantification engine (m4 + m5 i/ii/iii).

Faithfully REUSES the day-5 morpho pipeline methods (imported from the read-only
sibling `morphological_quantification/scripts`). Operates on the Z-MAX-PROJECTION
of each organoid (one stable mask + axis per morph, matching the original
"single consensus axis per file" intent; far cheaper than per-plane routing):

  whole-morph mask  -> M.segment_whole_morph        (DAPI; inverted-BF surrogate
                                                      for the DAPI-less d4 images)
  morph length      -> M.centerline_from_mask        (centerline arc length)
  background        -> M.background_correct_signal    (off-morph median)
  threshold         -> pooled half-Gaussian null per marker (M.half_gaussian_*)
  positive domain   -> M.positive_mask_from_corrected
  bilaterality      -> M.project_points_to_centerline -> signed_transverse ->
                       1 - |R-L|/(R+L)   (the exact Fig-3f side_balance_index)

Markers: MESP2 (mCherry) + FOXF1 (TagYFP). Autofluorescent beads (bright in every
channel) are detected from the two bead channels and excluded; bead mask is drawn
in the QC overlays for Steven to verify. THRESHOLD_SIGMA is the main calibration
knob (QC'd, not tuned to a desired answer).

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
from scipy import ndimage as ndi
from skimage import morphology

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The morphology helpers ship in this repository, under the morphometry lane.
MORPHO_SCRIPTS = str(Path(__file__).resolve().parents[2] / "morphometry" / "scripts")
sys.path.insert(0, MORPHO_SCRIPTS)
import morphology_quantification_helpers as M  # noqa: E402

# Pin medial_axis RNG so centerline length is deterministic (helper calls it without a seed).
import functools as _ft, inspect as _insp  # noqa: E402
from skimage import morphology as _skm  # noqa: E402
if "rng" in _insp.signature(_skm.medial_axis).parameters:
    _skm.medial_axis = _ft.partial(_skm.medial_axis, rng=0)

WORK = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
MANIFEST = WORK / "results" / "tables" / "00_raw_manifest.tsv"
TAB = WORK / "results" / "tables"
QC = WORK / "results" / "qc"
MARKERS = ["mesp2", "foxf1"]
THRESHOLD_SIGMA = 3.0
PRESENCE_MIN_FRACTION = 0.01


def maxproj(stack, ci):
    return np.asarray(stack.data_czyx[int(ci)].max(axis=0), dtype=np.float32)


def morph_mask(stack, row, dapi_idx):
    if dapi_idx >= 0:
        return M.segment_whole_morph(maxproj(stack, dapi_idx)), "dapi"
    bf = maxproj(stack, int(row["ch_bf"]))
    return M.segment_whole_morph(float(np.nanmax(bf)) - bf), "bf_surrogate"


def bead_mask(stack, row):
    lpm = maxproj(stack, int(row["ch_lpm_bead"]))
    nmp = maxproj(stack, int(row["ch_nmp_bead"]))
    both = (lpm > np.percentile(lpm, 99.7)) & (nmp > np.percentile(nmp, 99.7))
    both = morphology.remove_small_objects(both, min_size=150)
    if both.any():
        both = morphology.binary_closing(both, morphology.disk(3))
        both = ndi.binary_fill_holes(both)
        both = morphology.binary_dilation(both, morphology.disk(2))
    return np.asarray(both, dtype=bool)


def bilaterality(pos, corr, centerline_xy):
    ys, xs = np.where(pos)
    if xs.size < 3 or centerline_xy is None or centerline_xy.shape[0] < 2:
        return np.nan, np.nan
    proj = M.project_points_to_centerline(np.column_stack([xs, ys]).astype(np.float64), centerline_xy)
    s = np.asarray(proj["signed_transverse_px"], dtype=np.float64)
    w = corr[pos].astype(np.float64)
    left, right = s < -0.5, s > 0.5
    lc, rc = int(left.sum()), int(right.sum())
    idx = (1.0 - abs(rc - lc) / (lc + rc)) if (lc + rc) else np.nan
    wl, wr = float(w[left].sum()), float(w[right].sum())
    widx = (1.0 - abs(wr - wl) / (wl + wr)) if (wl + wr) else np.nan
    return idx, widx


def main():
    qc_ids, only_ids = [], None
    contact = "--contact" in sys.argv
    ifsheet = "--ifsheet" in sys.argv
    axissheet = "--axissheet" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--qc="):
            qc_ids = a.split("=", 1)[1].split(",")
        elif a.startswith("--only="):
            only_ids = a.split("=", 1)[1].split(",")
    df = pd.read_csv(MANIFEST, sep="\t")
    if only_ids:
        df = df[df["image_id"].isin(only_ids)]

    cache, pool = {}, {m: [] for m in MARKERS}
    rng = np.random.default_rng(7)
    for _, row in df.iterrows():
        stack = M.load_czi_stack(Path(row["file_path"]))
        mask, msrc = morph_mask(stack, row, int(row["ch_dapi"]))
        beads = bead_mask(stack, row)
        mask = np.asarray(mask, dtype=bool) & ~beads
        cl_xy, length_px = None, np.nan
        if mask.sum() > 200:
            try:
                cl = M.centerline_from_mask(mask)
                cl_xy = np.asarray(cl["centerline_xy"], dtype=np.float64)
                length_px = float(cl["length_px"])
            except Exception as e:
                print(f"  centerline failed {row['image_id']}: {e}")
        mp = {m: maxproj(stack, int(row[f"ch_{m}"])) for m in MARKERS}
        corr = {}
        for m in MARKERS:
            c, _, _ = M.background_correct_signal(mp[m], mask, background_estimator="whole_off_morph")
            corr[m] = c
            v = c[mask & np.isfinite(c)].astype(np.float32)
            if v.size:
                pool[m].append(v if v.size <= 4096 else v[rng.choice(v.size, 4096, replace=False)])
        cache[row["image_id"]] = dict(row=row, mask=mask, beads=beads, msrc=msrc,
                                      cl_xy=cl_xy, length_px=length_px, corr=corr,
                                      bf=maxproj(stack, int(row["ch_bf"])), mp=mp)

    # pooled half-Gaussian threshold per marker
    thr, thr_rows = {}, []
    for m in MARKERS:
        v = np.concatenate(pool[m]).astype(np.float64) if pool[m] else np.array([0.0, 1.0])
        v = v[np.isfinite(v)]
        lo, hi = np.quantile(v, [0.001, 0.999])
        hi = hi if hi > lo else lo + 1.0
        edges = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 2049)
        counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
        d = M.half_gaussian_threshold_from_histogram(counts=counts, edges=edges, z=THRESHOLD_SIGMA)
        thr[m] = float(d["threshold_value"])
        thr_rows.append({"marker": m, "sigma": THRESHOLD_SIGMA, "threshold": thr[m],
                         "baseline_loc": d["baseline_location"], "baseline_scale": d["baseline_scale"],
                         "n_pooled": int(v.size)})
    pd.DataFrame(thr_rows).to_csv(TAB / "03_marker_thresholds.tsv", sep="\t", index=False)

    out = []
    for iid, e in cache.items():
        row = e["row"]
        px, py = float(row["pixel_um_x"]), float(row["pixel_um_y"])
        morph_px = int(e["mask"].sum())
        r = {"image_id": iid, "organoid_label": row["organoid_label"], "group": row["group"],
             "condition": row["condition"], "quality_flag": row["quality_flag"], "mask_source": e["msrc"],
             "n_z": int(row["size_z"]), "bead_px": int(e["beads"].sum()),
             "morph_area_um2": morph_px * px * py,
             "length_um": e["length_px"] * px if np.isfinite(e["length_px"]) else np.nan}
        # PCA straight axis as a 2nd, deliberately-different axis (sensitivity check only)
        try:
            _c, _u, _b = M.major_axis_from_mask(e["mask"])
            _pa, _pb = M.choose_axis_endpoints(_b, _c, _u)
            pca_axis = np.vstack([_pa, _pb]).astype(np.float64)
        except Exception:
            pca_axis = None
        for m in MARKERS:
            pos = M.positive_mask_from_corrected(e["corr"][m], e["mask"], thr[m])
            npos = int(pos.sum())
            frac = (npos / morph_px) if morph_px else np.nan
            bi, wbi = bilaterality(pos, e["corr"][m], e["cl_xy"])
            bip = bilaterality(pos, e["corr"][m], pca_axis)[0] if pca_axis is not None else np.nan
            r[f"{m}_area_um2"] = npos * px * py
            r[f"{m}_fraction"] = frac
            r[f"{m}_present"] = bool(frac >= PRESENCE_MIN_FRACTION)
            r[f"{m}_bilaterality"] = bi
            r[f"{m}_bilaterality_pca"] = bip
            r[f"{m}_weighted_bilaterality"] = wbi
        r["success_mesp2_and_foxf1"] = bool(r["mesp2_present"] and r["foxf1_present"])
        out.append(r)
    out_df = pd.DataFrame(out)
    out_df.to_csv(TAB / "03_per_organoid_metrics.tsv", sep="\t", index=False)

    print("\n=== axis-sensitivity of bilaterality (medial centerline vs PCA straight axis) ===")
    for m in MARKERS:
        d = (out_df[f"{m}_bilaterality"] - out_df[f"{m}_bilaterality_pca"]).abs()
        print(f"  {m}: mean|Δ| = {d.mean():.3f}   median|Δ| = {d.median():.3f}   max|Δ| = {d.max():.3f}")
    worst = (out_df["foxf1_bilaterality"] - out_df["foxf1_bilaterality_pca"]).abs().sort_values(ascending=False)
    print("  most axis-sensitive (FOXF1):")
    for iid in worst.head(5).index:
        rr = out_df.loc[iid]
        print(f"    {rr['image_id']} {rr['group']:14s} medial={rr['foxf1_bilaterality']:.2f} "
              f"pca={rr['foxf1_bilaterality_pca']:.2f}  Δ={worst.loc[iid]:.2f}")

    pd.set_option("display.width", 250, "display.max_columns", 60)
    show = ["image_id", "organoid_label", "group", "condition", "mask_source", "length_um",
            "mesp2_present", "mesp2_fraction", "mesp2_bilaterality",
            "foxf1_present", "foxf1_fraction", "foxf1_bilaterality", "success_mesp2_and_foxf1"]
    print(out_df[show].round(3).to_string(index=False))

    for iid in qc_ids:
        if iid not in cache:
            continue
        e = cache[iid]
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
        bf = M.robust_rescale(e["bf"])
        for a in ax:
            a.imshow(bf, cmap="gray"); a.set_xticks([]); a.set_yticks([])
        M.plot_mask_outline(ax[0], e["mask"], color="yellow", linewidth=1.2)
        if e["beads"].any():
            ax[0].contour(e["beads"], colors="cyan", linewidths=0.8)
        if e["cl_xy"] is not None:
            ax[0].plot(e["cl_xy"][:, 0], e["cl_xy"][:, 1], "-", color="orange", lw=1.5)
        ax[0].set_title(f"{iid} morph(yellow)/bead(cyan)/axis(orange) src={e['msrc']}")
        for j, (m, col) in enumerate([("mesp2", "#ff3030"), ("foxf1", "#28d23a")], 1):
            pos = M.positive_mask_from_corrected(e["corr"][m], e["mask"], thr[m])
            ax[j].imshow(M.robust_rescale(e["mp"][m]), cmap="gray")
            ax[j].contour(pos, colors=col, linewidths=0.8)
            M.plot_mask_outline(ax[j], e["mask"], color="yellow", linewidth=0.6)
            ax[j].set_title(f"{m} positive frac={pos.sum()/max(e['mask'].sum(),1):.3f}")
        fig.tight_layout()
        fig.savefig(QC / f"03_overlay_{iid}.png", dpi=130)
        plt.close(fig)
        print(f"wrote {QC / f'03_overlay_{iid}.png'}")

    if contact:
        ids = list(cache.keys())
        ncol, nrow = 4, int(np.ceil(len(cache) / 4))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.5 * nrow), squeeze=False)
        for k, iid in enumerate(ids):
            e = cache[iid]; rw = e["row"]; ax = axes[k // ncol][k % ncol]
            ax.imshow(M.robust_rescale(e["bf"]), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
            M.plot_mask_outline(ax, e["mask"], color="yellow", linewidth=1.0)
            if e["beads"].any():
                ax.contour(e["beads"], colors="cyan", linewidths=0.6)
            if e["cl_xy"] is not None:
                ax.plot(e["cl_xy"][:, 0], e["cl_xy"][:, 1], "-", color="orange", lw=1.0)
            for m, col in [("mesp2", "#ff3030"), ("foxf1", "#28d23a")]:
                pos = M.positive_mask_from_corrected(e["corr"][m], e["mask"], thr[m])
                ax.contour(pos, colors=col, linewidths=0.5)
            o = out_df[out_df["image_id"] == iid].iloc[0]
            ax.set_title(f"{iid}  {rw['organoid_label'][:16]}\n{rw['group']}/{rw['condition']} [{e['msrc']}]\n"
                         f"L={o['length_um']:.0f}µm  M={o['mesp2_fraction']:.2f}  F={o['foxf1_fraction']:.2f}",
                         fontsize=8)
        for k in range(len(ids), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("Co-transplant mask/marker TRIAGE — BF + morph(yellow) bead(cyan) axis(orange) "
                     "MESP2(red) FOXF1(green).  Mark each: good / re-ROI / drop", fontsize=12, y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(QC / "03_contact_sheet.png", dpi=120)
        plt.close(fig)
        print(f"wrote {QC / '03_contact_sheet.png'}")

    if ifsheet:
        ids = list(cache.keys())
        ncol, nrow = 4, int(np.ceil(len(cache) / 4))
        for marker, col, disp in [("mesp2", "#ff3030", "MESP2 (mCherry) IF — somite"),
                                   ("foxf1", "#28d23a", "FOXF1 (TagYFP) IF — LPM")]:
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.5 * nrow), squeeze=False)
            for k, iid in enumerate(ids):
                e = cache[iid]; rw = e["row"]; ax = axes[k // ncol][k % ncol]
                ax.imshow(M.robust_rescale(e["mp"][marker]), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
                pos = M.positive_mask_from_corrected(e["corr"][marker], e["mask"], thr[marker])
                M.plot_mask_outline(ax, e["mask"], color="yellow", linewidth=0.8)
                ax.contour(pos, colors=col, linewidths=0.7)
                o = out_df[out_df["image_id"] == iid].iloc[0]
                ax.set_title(f"{iid}  {rw['organoid_label'][:16]} [{e['msrc']}]\n"
                             f"frac={o[marker+'_fraction']:.2f}  bilat={o[marker+'_bilaterality']:.2f}", fontsize=8)
            for k in range(len(ids), nrow * ncol):
                axes[k // ncol][k % ncol].axis("off")
            fig.suptitle(f"{disp} — grayscale IF + morph(yellow) + called {marker}+ region. "
                         "THRESHOLD UNCALIBRATED + masks not yet fixed (first pass)", fontsize=12, y=0.998)
            fig.tight_layout(rect=(0, 0, 1, 0.985))
            fig.savefig(QC / f"03_if_{marker}_sheet.png", dpi=120)
            plt.close(fig)
            print(f"wrote {QC / f'03_if_{marker}_sheet.png'}")

    if axissheet:
        ids = list(cache.keys())
        ncol, nrow = 4, int(np.ceil(len(cache) / 4))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.5 * nrow), squeeze=False)
        for k, iid in enumerate(ids):
            e = cache[iid]; rw = e["row"]; ax = axes[k // ncol][k % ncol]
            ax.imshow(M.robust_rescale(e["bf"]), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
            M.plot_mask_outline(ax, e["mask"], color="yellow", linewidth=0.9)
            if e["cl_xy"] is not None:
                ax.plot(e["cl_xy"][:, 0], e["cl_xy"][:, 1], "-", color="cyan", lw=1.8, label="A medial")
            try:
                centroid, axis_u, boundary = M.major_axis_from_mask(e["mask"])
                pa, pb = M.choose_axis_endpoints(boundary, centroid, axis_u)
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "-", color="magenta", lw=1.8, label="B PCA")
            except Exception:
                pass
            ax.set_title(f"{iid}  {rw['organoid_label'][:16]} [{e['msrc']}]", fontsize=8)
        for k in range(len(ids), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        axes[0][0].legend(loc="upper left", fontsize=7, framealpha=0.6)
        fig.suptitle("AXIS CANDIDATES per organoid — A=cyan (medial centerline), B=magenta (PCA straight). "
                     "Reply per organoid: keep+A / keep+B / redraw A→B / drop", fontsize=12, y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(QC / "03_axis_candidates.png", dpi=120)
        plt.close(fig)
        print(f"wrote {QC / '03_axis_candidates.png'}")


if __name__ == "__main__":
    main()
