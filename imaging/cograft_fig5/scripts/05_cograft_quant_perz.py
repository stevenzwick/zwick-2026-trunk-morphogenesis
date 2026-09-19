#!/usr/bin/env python3
"""Co-transplant morphological quantification (m4 + m5 i/ii/iii).

Built to be comparable to the day-5 bead pipeline (same helpers, same bilaterality and
threshold method), with ONE deliberate, author-approved divergence for the noisier co-graft
data: the fluorescent-domain denominator is a single max-projection mask per organoid
(not a per-z mask), which removes per-z mask drift. The AXIS is still per-z.

    AXIS  (the day-5 method, kept exactly):
      per-z whole-morph masks          -> M.segment_whole_morph   (DAPI; inverted-BF
                                          surrogate for the DAPI-less d4 images)
      per-z medial centerlines         -> M.centerline_from_mask
      CONSENSUS axis over retained z   -> M.consensus_centerline_from_paths
      morph length                     =  consensus-axis arc length

    DOMAINS  (single-mask divergence):
      SINGLE max-proj domain mask      -> domain_mask  (DAPI max-proj / inverted-BF
                                          min-proj; beads excluded; = the denominator)
      background / threshold           -> M.background_correct_signal + pooled
                                          half-Gaussian null (M.half_gaussian_*),
                                          sampled INSIDE the single domain mask
      positive domains (per retained z)-> M.positive_mask_from_corrected
      bilaterality (per z, then mean)  -> M.project_points_to_centerline onto the
                                          consensus axis -> 1 - |R-L|/(R+L)

Supersedes the all-max-projection first pass (`03_*`): the (a)-vs-(b) axis test
(`04_axis_ab_test`) showed a max-proj-medial AXIS materially changed bilaterality vs the
per-z consensus, so the axis is per-z; only the domain DENOMINATOR is a single mask.

Manual per-organoid decisions are READ from annotation TSVs (recorded input ->
reproducible): `annotations/roi.tsv` (restrict mask to the intended object) and
`annotations/retained_z.tsv` (which z feed the consensus axis, or `drop`). Absent
-> defaults (auto mask, all valid z). Deterministic given annotations + raw images.

Reuses R1 methods via ONE documented import (vendor at consolidation — see
CONSOLIDATION_NOTES.md). Markers: MESP2 (mCherry) + FOXF1 (TagYFP); no PAX8 here.

NOT runnable as shipped. This stage reads the raw .czi acquisitions, which are not deposited —
they are available from the lead contact on request, per the paper's Data Availability
Statement. The measurements it produces are committed under the lane's `derived/` directory,
which is what the figure scripts read. See `data/DOWNLOAD.md`.
"""
from __future__ import annotations

import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# --- the single R1 dependency (vendor into image_analysis/lib/ at consolidation) ---
# The morphology helpers ship in this repository, under the morphometry lane.
MORPHO = str(Path(__file__).resolve().parents[2] / "morphometry" / "scripts")
sys.path.insert(0, MORPHO)
import morphology_quantification_helpers as M  # noqa: E402

# CZI pixel IO goes through libCZI, NOT M.load_czi_stack's czifile — czifile mis-frames
# the odd-sized files here (img11/13/14/15). See czi_io.py + memory `czifile-odd-czi-bug`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import czi_io  # noqa: E402

# Pin medial_axis RNG so centerline length is deterministic (helper seeds nothing).
import functools as _ft, inspect as _insp  # noqa: E402
from skimage import morphology as _skm  # noqa: E402
if "rng" in _insp.signature(_skm.medial_axis).parameters:
    _skm.medial_axis = _ft.partial(_skm.medial_axis, rng=0)

WORK = Path("<analysis-root>/cotransplant_fig5_quant_2026-06-27")
MANIFEST = WORK / "results" / "tables" / "00_raw_manifest.tsv"
TAB = WORK / "results" / "tables"
QC = WORK / "results" / "qc"
ANNO = WORK / "annotations"

MARKERS = ["mesp2", "foxf1"]
THRESHOLD_SIGMA = 3.0
FOXF1_SIGMA = 4.0            # FOXF1 threshold tightened vs MESP2 (over-broad at 3σ on visual inspection); --foxf1_sigma to tune
FOXF1_THRESH_SCALE = 1.3    # global visual-calibration ×multiplier on the pooled FOXF1 threshold (chosen on img01 — 4σ was slightly broad); --foxf1_thresh_scale to tune
PRESENCE_MIN_FRACTION = 0.01
MIN_MASK_PX = 200
SAMPLE_PER_PLANE = 4096
RNG = np.random.default_rng(7)


def arc_len_px(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else 0.0


def load_annotations():
    """retained_z (per organoid z-list for the consensus axis) + roi bbox + drop set."""
    retained_z, roi, drop = {}, {}, set()
    p = ANNO / "retained_z.tsv"
    if p.exists():
        for _, r in pd.read_csv(p, sep="\t").iterrows():
            iid = str(r["image_id"])
            if str(r.get("drop", "")).strip().lower() in {"1", "true", "yes", "y"}:
                drop.add(iid)
            v = str(r.get("retained_z", "")).strip()
            if v and v.lower() != "all":
                retained_z[iid] = [int(x) for x in v.replace(";", ",").split(",") if x.strip()]
    p = ANNO / "roi.tsv"
    if p.exists():
        for _, r in pd.read_csv(p, sep="\t").iterrows():
            try:
                roi[str(r["image_id"])] = (int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"]))
            except (KeyError, TypeError, ValueError):
                pass
    seg = {}                                        # per-image segmentation overrides
    p = ANNO / "seg_params.tsv"
    if p.exists():
        for _, r in pd.read_csv(p, sep="\t").iterrows():
            d = {}
            if pd.notna(r.get("tscale")):
                d["tscale"] = float(r["tscale"])
            if pd.notna(r.get("min_size")):
                d["min_size"] = int(r["min_size"])
            if pd.notna(r.get("open_radius")):
                d["open_radius"] = int(r["open_radius"])
            if d:
                seg[str(r["image_id"])] = d
    donor = {}                                      # per-image donor-exclusion hand-tuning
    p = ANNO / "donor.tsv"
    if p.exists():
        for _, r in pd.read_csv(p, sep="\t").iterrows():
            d = {}
            if pd.notna(r.get("thresh_scale")):     # multiplier on the log1p-Otsu donor threshold
                d["thresh_scale"] = float(r["thresh_scale"])
            if pd.notna(r.get("close_radius")):      # graft coherence: closing disk radius (px)
                d["close_radius"] = int(r["close_radius"])
            if pd.notna(r.get("min_frac")):          # graft coherence: drop objects < min_frac * morph
                d["min_frac"] = float(r["min_frac"])
            if pd.notna(r.get("signal")):            # 'ratio' (EGFP/DAPI) or 'raw' (brightness)
                d["signal"] = str(r["signal"]).strip().lower()
            if d:
                donor[str(r["image_id"])] = d
    axis_man = {}                                   # manual AP axis (2-pt line or N-pt polyline) -> overrides medial
    p = ANNO / "axis.tsv"
    if p.exists():
        for _, r in pd.read_csv(p, sep="\t").iterrows():
            pts = None
            v = str(r.get("points", "")).strip()
            if v and v.lower() != "nan":            # polyline: "x0,y0;x1,y1;x2,y2;..."
                try:
                    pts = [[float(c) for c in pr.split(",")] for pr in v.split(";") if pr.strip()]
                except ValueError:
                    pts = None
            if pts is None:                         # else a 2-point head->tail line
                try:
                    pts = [[float(r["x0"]), float(r["y0"])], [float(r["x1"]), float(r["y1"])]]
                except (KeyError, TypeError, ValueError):
                    pts = None
            if pts and len(pts) >= 2:
                ext = str(r.get("extend", "")).strip().lower()   # default True; "no"/"false"/"0" -> use polyline AS-IS
                do_ext = ext not in ("false", "0", "no", "n")    # host-trunk axes by a contiguous graft skip extend
                axis_man[str(r["image_id"])] = (np.asarray(pts, dtype=float), do_ext)
    exclude = {}                                    # per-image debris/bead bboxes removed from the domain mask
    p = ANNO / "exclude.tsv"
    if p.exists():
        for _, r in pd.read_csv(p, sep="\t").iterrows():
            try:
                bb = (int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"]))
            except (KeyError, TypeError, ValueError):
                continue
            exclude.setdefault(str(r["image_id"]), []).append(bb)
    foxf1 = {}                                      # per-image MANUAL FOXF1-threshold scale:
    p = ANNO / "foxf1.tsv"                          # ×scale on the per-image half-Gaussian FOXF1 cut — <1 LOOSENS
    if p.exists():                                  # (grows the mask) / >1 TIGHTENS. Needed where a bright FOXF1+
        for _, r in pd.read_csv(p, sep="\t").iterrows():   # donor graft skews the automatic per-image cut.
            try:
                foxf1[str(r["image_id"])] = float(r["scale"])
            except (KeyError, TypeError, ValueError):
                continue
    cut = {}                                        # per-image organoid-splitting 'tangent' line(s): two points
    p = ANNO / "cut.tsv"                            # each severs the domain (infinite line) -> keep the target organoid
    if p.exists():
        for _, r in pd.read_csv(p, sep="\t").iterrows():
            try:
                ln = (int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"]))
            except (KeyError, TypeError, ValueError):
                continue
            cut.setdefault(str(r["image_id"]), []).append(ln)
    return retained_z, roi, drop, seg, donor, exclude, axis_man, foxf1, cut


def bead_mask_plane(stack, row, z) -> np.ndarray:
    """Autofluorescent beads: compact blobs co-bright in BOTH bead channels.
    Skipped (empty mask) when a bead channel is absent (ch < 0, e.g. the one-graft set)."""
    ci_l, ci_n = int(row["ch_lpm_bead"]), int(row["ch_nmp_bead"])
    if ci_l < 0 or ci_n < 0:
        return np.zeros(stack.data_czyx.shape[2:], dtype=bool)
    lpm = np.asarray(stack.data_czyx[ci_l, z], dtype=np.float32)
    nmp = np.asarray(stack.data_czyx[ci_n, z], dtype=np.float32)
    both = (lpm > np.percentile(lpm, 99.7)) & (nmp > np.percentile(nmp, 99.7))
    both = morphology.remove_small_objects(both, min_size=150)
    if both.any():
        both = ndi.binary_fill_holes(morphology.binary_closing(both, morphology.disk(3)))
        both = morphology.binary_dilation(both, morphology.disk(2))
    return np.asarray(both, dtype=bool)


def bead_mask_union(stack, row, zs) -> np.ndarray:
    """Union of per-z bead masks over z (beads to exclude from the max-proj domain mask)."""
    acc = None
    for z in zs:
        b = bead_mask_plane(stack, row, z)
        acc = b if acc is None else (acc | b)
    return np.asarray(acc, dtype=bool) if acc is not None else np.zeros(stack.data_czyx.shape[2:], bool)


def _segment_with_roi(img, roi, tscale, min_size) -> np.ndarray:
    """Otsu whole-morph segmentation. With an ROI bbox, threshold INSIDE it only (so a
    brighter off-target object outside can't bias Otsu) and keep its largest component."""
    if roi is None:
        return np.asarray(M.segment_whole_morph(img, threshold_scale=tscale, min_size_px=min_size), dtype=bool)
    x0, y0, x1, y1 = max(0, roi[0]), max(0, roi[1]), roi[2], roi[3]
    out = np.zeros(img.shape, dtype=bool)
    sub = np.asarray(M.segment_whole_morph(img[y0:y1, x0:x1], threshold_scale=tscale,
                                           min_size_px=min_size), dtype=bool)
    lbl, n = ndi.label(sub)
    if n > 1:                                      # keep the largest object inside the ROI
        counts = np.bincount(lbl.ravel()); counts[0] = 0
        sub = lbl == int(counts.argmax())
    out[y0:y1, x0:x1] = sub
    return out


def source_plane(stack, row, z, dapi_idx) -> np.ndarray:
    """Per-z segmentation source: DAPI (organoid bright) or inverted BF (organoid dark)."""
    if dapi_idx >= 0:
        return np.asarray(stack.data_czyx[dapi_idx, z], dtype=np.float32)
    bf = np.asarray(stack.data_czyx[int(row["ch_bf"]), z], dtype=np.float32)
    return float(np.nanmax(bf)) - bf               # tscale < day-5 default -> looser periphery


def source_projection(stack, row, dapi_idx) -> np.ndarray:
    """SINGLE 2-D source for the whole-organoid domain mask:
       DAPI -> MAX-projection over z (nuclei are bright wherever tissue is present),
       BF   -> invert the MIN-projection (organoid is dark on BF, so the per-pixel
               darkest value over z marks tissue at its best-focus plane)."""
    if dapi_idx >= 0:
        return np.asarray(stack.data_czyx[dapi_idx].max(axis=0), dtype=np.float32)
    bf_min = np.asarray(stack.data_czyx[int(row["ch_bf"])].min(axis=0), dtype=np.float32)
    return float(np.nanmax(bf_min)) - bf_min


def morph_mask_plane(stack, row, z, dapi_idx, roi, tscale=0.5, min_size=4000, cuts=None) -> np.ndarray:
    """Per-z mask — used ONLY to derive the per-z medial centerlines for the consensus axis.
    `cuts` are the SAME hand-drawn organoid-splitting tangent lines as the domain mask: apply them
    here too and keep the largest component, so the axis (hence morph length) stays on the target
    organoid. Without this the cut reached only the domain denominator, and the per-z centerlines
    still spanned BOTH touching organoids -> inflated length (first-version finding: ng05, ng10)."""
    m = np.asarray(_segment_with_roi(source_plane(stack, row, z, dapi_idx), roi, tscale, min_size),
                   dtype=bool) & ~bead_mask_plane(stack, row, z)
    if cuts:
        m = _apply_cut(m, cuts)
        lbl, n = ndi.label(m)                       # keep the target organoid only (matches domain_mask)
        if n > 1:
            counts = np.bincount(lbl.ravel()); counts[0] = 0
            m = lbl == int(counts.argmax())
    return np.asarray(m, dtype=bool)


def _apply_cut(m, cuts, width=2.5):
    """Sever the domain along one or more hand-drawn 'tangent' lines (each = two points), so the
    single-largest-component step then keeps only the target organoid. Each line is treated as
    INFINITE (extended across the whole frame) so it fully divides the mask — used to split two
    organoids that are broadly touching (a bbox ROI can't cut on a diagonal; opening only removes
    thin necks)."""
    if not cuts:
        return np.asarray(m, dtype=bool)
    H, W = m.shape
    ys, xs = np.mgrid[0:H, 0:W]
    keep = np.ones((H, W), dtype=bool)
    for (x0, y0, x1, y1) in cuts:
        dx, dy = float(x1 - x0), float(y1 - y0)
        norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
        dist = ((xs - x0) * dy - (ys - y0) * dx) / norm     # signed distance to the infinite line
        keep &= (np.abs(dist) >= width)
    return np.asarray(m, dtype=bool) & keep


def domain_mask(stack, row, dapi_idx, roi, tscale, min_size, nz, open_radius=0, cuts=None) -> np.ndarray:
    """SINGLE max-projection whole-morph mask per organoid = the consistent denominator for
    the fluorescent-domain quant. One region for every z -> removes per-z mask drift (the
    AXIS still comes from the per-z centerlines). Beads excluded (union over all z).
    Minor, deliberate divergence from day-5's per-z domain masks (author-approved),
    justified by the noisier co-graft data; the bilaterality formula + per-z-consensus axis
    stay matched to day-5.

    open_radius (annotation, per image) > 0 applies a morphological opening that severs a
    thin attachment (a fiber, or a neck to an adjacent organoid) BEFORE the single-largest-
    component step removes the detached junk/neighbor. Only features thinner than the disk
    are removed; the organoid bulk boundary is preserved."""
    m = _segment_with_roi(source_projection(stack, row, dapi_idx), roi, tscale, min_size)
    if open_radius and open_radius > 0:
        m = morphology.binary_opening(np.asarray(m, dtype=bool), morphology.disk(int(open_radius)))
    m = _apply_cut(m, cuts)                         # sever hand-drawn organoid-splitting tangent line(s)
    lbl, n = ndi.label(m)                          # keep the target organoid only
    if n > 1:
        counts = np.bincount(lbl.ravel()); counts[0] = 0
        m = lbl == int(counts.argmax())
    return np.asarray(m, dtype=bool) & ~bead_mask_union(stack, row, range(nz))


def _march_out(start, direction, mask, max_steps=4000):
    """From `start`, step 1 px along `direction` while inside `mask`; return the last in-mask
    point (or None if we leave immediately). Extends a hand-drawn AP-axis endpoint out to the
    mask boundary so morph length spans the whole domain. Stops at the first exit (never jumps)."""
    d = np.asarray(direction, dtype=float)
    n = float(np.hypot(d[0], d[1]))
    if n == 0:
        return None
    d = d / n
    H, W = mask.shape
    p = np.asarray(start, dtype=float).copy()
    last = None
    for _ in range(int(max_steps)):
        p = p + d
        xi, yi = int(round(p[0])), int(round(p[1]))
        if xi < 0 or yi < 0 or xi >= W or yi >= H or not mask[yi, xi]:
            break
        last = p.copy()
    return last


def extend_polyline_to_mask(poly, mask):
    """Prepend/append points so a manual AP polyline reaches the mask ends, marching out along
    each end segment's own direction. Endpoints already outside the mask are left as-is (no trim)."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return poly
    head = _march_out(poly[0], poly[0] - poly[1], mask)
    tail = _march_out(poly[-1], poly[-1] - poly[-2], mask)
    return np.asarray(([head] if head is not None else []) + list(poly)
                      + ([tail] if tail is not None else []), dtype=float)


def donor_channels_for_row(row) -> list:
    """Donor channel(s) to subtract for HOST FOXF1. Two-graft: the LPM donor (ch_lpm_bead / EGFP) ONLY.
    FOXF1 is the LPM marker, so the FOXF1+ donor to remove is the donor LPM; the NMP donor (ch_nmp_bead
    / A647) is FOXF1-negative and its weak, broad A647 signal yields a huge noisy mask (63% of the morph
    on img01) that wrongly eats real host FOXF1. `ch_donor` override (semicolon list)
    still honoured. ctrl/KD/LDN host FOXF1 comes from LPM_transplant, not here."""
    if "neg_ctrl" in (str(row.get("condition", "")).lower() if hasattr(row, "get") else ""):
        return []                                # negative control (neg_ctrl): pluripotent, no beads, no donor
        # NB match "neg_ctrl" not bare "neg" — "onegraft" contains the substring "neg" (o-NEG-raft)!
    val = str(row.get("ch_donor", "")) if hasattr(row, "get") else ""
    if val.strip() not in ("", "nan", "-1"):
        return [int(float(x)) for x in val.split(";") if x.strip() and int(float(x)) >= 0]
    return [int(row["ch_lpm_bead"])] if int(row["ch_lpm_bead"]) >= 0 else []


def _dapi_floor(corr_dapi, mask, floor_quantile=0.10, min_floor=1.0) -> float:
    """Protected DAPI floor = low-quantile of corrected DAPI inside the morph (ports LPM_transplant
    compute_dapi_floor) — stops low-DAPI pixels from exploding the reporter/DAPI ratio."""
    v = np.asarray(corr_dapi, dtype=np.float32)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(v) & (v > 0)
    if not ok.any():
        return float(min_floor)
    fv = float(np.quantile(v[ok], float(floor_quantile)))
    return float(max(float(min_floor), fv if np.isfinite(fv) else min_floor))


def _reporter_over_dapi(corr_rep, corr_dapi, mask, floor) -> np.ndarray:
    """Donor-reporter / DAPI ratio with a protected floor (ports LPM_transplant reporter_over_dapi):
    dividing by DAPI removes the tissue pedestal (autofluor baseline that scales with tissue density),
    so the diffuse donor baseline collapses and Otsu can isolate the focal donor cells. NaN off-morph."""
    rep = np.clip(np.asarray(corr_rep, dtype=np.float32), 0, None)
    dp = np.clip(np.asarray(corr_dapi, dtype=np.float32), 0, None)
    denom = np.maximum(dp, np.float32(floor))
    out = np.full_like(rep, np.nan, dtype=np.float32)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(rep) & np.isfinite(denom)
    out[ok] = rep[ok] / denom[ok]
    return out


def _donor_threshold_log1p_otsu(vals) -> float:
    """Threshold that peels the BRIGHT donor clump off the dimmer tissue pedestal: 3-class multi-Otsu in
    log1p space, keep the UPPER threshold (classes = background / tissue / donor-clump -> top class).
    2-class Otsu (raw OR ratio) split tissue-vs-background on the broadly-EGFP+ morphs and flagged
    ~75-90% of the tissue as donor (verified via the close=0 pure-seed diagnostic); the 3rd class isolates
    the focal clump. Falls back to 2-class Otsu if multi-Otsu can't fit (too few distinct levels)."""
    from skimage.filters import threshold_otsu, threshold_multiotsu
    v = np.clip(np.asarray(vals, dtype=np.float64), 0, None); v = v[np.isfinite(v)]
    if v.size < 10 or float(v.max()) <= 0:
        return np.inf
    lv = np.log1p(v)
    try:
        return float(np.expm1(threshold_multiotsu(lv, classes=3)[-1]))
    except Exception:
        try:
            return float(np.expm1(threshold_otsu(lv)))
        except Exception:
            return np.inf


def donor_signal_map(stack, row, dmask, ci, dapi_idx, use_ratio=True):
    """Per donor channel -> the scalar field the donor threshold + mask act on.
    `ratio` (default): background-corrected max-proj / corrected DAPI (LPM_transplant / Fig-5n recipe) —
      good when the donor is reporter+ tissue over a diffuse pedestal (host-reporter transplants).
    `raw` (annotations/donor.tsv signal=raw): background-corrected max-proj, NO DAPI division — the
      bead-labelled two-graft donor CLUMP is identified by BRIGHTNESS; the ratio divides out that
      brightness (and inflates low-DAPI regions) so it has no distinct clump mode (verified: ratio
      threshold jumps 65%->0% with nothing in between), while raw intensity shows the clump cleanly.
    No DAPI (day4-live img05-09) -> raw regardless. Returns (field, is_ratio)."""
    mp = np.asarray(stack.data_czyx[ci].max(axis=0), dtype=np.float32)
    corr, _, _ = M.background_correct_signal(mp, dmask, background_estimator="whole_off_morph")
    if use_ratio and dapi_idx is not None and int(dapi_idx) >= 0:
        dmp = np.asarray(stack.data_czyx[int(dapi_idx)].max(axis=0), dtype=np.float32)
        dcorr, _, _ = M.background_correct_signal(dmp, dmask, background_estimator="whole_off_morph")
        return _reporter_over_dapi(corr, dcorr, dmask, _dapi_floor(dcorr, dmask)), True
    return corr, False


def _coherent_region(seed, dmask, close_radius, min_size, fill_max_frac=0.05):
    """Turn a noisy pixel seed into the coherent grafted clump: morphological closing (bridge gaps) +
    fill BOUNDED holes (recover the dense graft core the EGFP/DAPI ratio under-weights, WITHOUT letting
    a sparse ring fill a whole-morph interior of non-donor tissue) + drop objects below a size floor
    (kill scattered noise). The graft stays mostly coherent with only limited boundary
    intermixing, so donor IS a connected region — this removes the scattered-pixel noise that made
    pixel-only host FOXF1 wildly threshold-sensitive."""
    s = np.asarray(seed, dtype=bool)
    if not s.any():
        return s
    dm = np.asarray(dmask, dtype=bool)
    if close_radius and close_radius > 0:
        s = morphology.binary_closing(s, morphology.disk(int(close_radius)))
    fill_area = max(1, int(float(fill_max_frac) * int(dm.sum())))    # only fill holes < fill_max_frac of morph
    s = morphology.remove_small_holes(np.asarray(s, dtype=bool), area_threshold=fill_area)
    s = np.asarray(s, dtype=bool) & dm
    if min_size and min_size > 0:
        s = morphology.remove_small_objects(s, min_size=int(min_size))
    return np.asarray(s, dtype=bool)


def donor_seed(stack, row, dmask, ci, dapi_idx, thr_scale, use_ratio=True):
    """Per-channel pixel seed for the donor graft: signal (ratio or raw) > 3-class multi-Otsu top *
    thr_scale. Returns (seed_mask, is_ratio, threshold)."""
    val, is_ratio = donor_signal_map(stack, row, dmask, int(ci), dapi_idx, use_ratio)
    ok = np.asarray(dmask, dtype=bool) & np.isfinite(val)
    t = _donor_threshold_log1p_otsu(val[ok]) * float(thr_scale)
    return (ok & (val > t)), is_ratio, float(t)


def donor_mask(stack, row, dmask, donor_chs, dapi_idx, thr_scale=1.0, close_radius=8, min_frac=0.005,
               use_ratio=True):
    """Union donor-GRAFT mask: per donor channel a signal/Otsu pixel seed (donor_seed) -> spatial
    coherence (_coherent_region) -> the connected grafted clump. HOST FOXF1 = FOXF1+ NOT in this mask.
    Per-image hand-tuning via annotations/donor.tsv (signal, thresh_scale, close_radius, min_frac) —
    legitimate here: measuring our own data, gold standard = matches by eye. Returns (mask, is_ratio,
    {ch: thr})."""
    acc = np.zeros(dmask.shape, dtype=bool)
    thr_used, has_dapi_any = {}, False
    min_size = max(50, int(float(min_frac) * int(np.asarray(dmask, dtype=bool).sum())))
    for ci in donor_chs:
        seed, has_dapi, t = donor_seed(stack, row, dmask, int(ci), dapi_idx, thr_scale, use_ratio)
        has_dapi_any = has_dapi_any or has_dapi
        thr_used[int(ci)] = t
        acc |= _coherent_region(seed, dmask, close_radius, min_size)
    return np.asarray(acc, dtype=bool), bool(has_dapi_any), thr_used


def bilaterality(pos: np.ndarray, corr: np.ndarray, axis_xy) -> tuple[float, float]:
    ys, xs = np.where(pos)
    if xs.size < 3 or axis_xy is None or len(axis_xy) < 2:
        return np.nan, np.nan
    proj = M.project_points_to_centerline(np.column_stack([xs, ys]).astype(float), axis_xy)
    s = np.asarray(proj["signed_transverse_px"], dtype=np.float64)
    w = corr[pos].astype(np.float64)
    left, right = s < -0.5, s > 0.5
    lc, rc = int(left.sum()), int(right.sum())
    idx = (1.0 - abs(rc - lc) / (lc + rc)) if (lc + rc) else np.nan
    wl, wr = float(w[left].sum()), float(w[right].sum())
    widx = (1.0 - abs(wr - wl) / (wl + wr)) if (wl + wr) else np.nan
    return idx, widx


def _outline_mask(ax, mask, rgb, extent=None, width=2):
    """Draw a boolean mask as a thin PERIMETER overlay — a crisp boundary that does NOT obscure the
    underlying signal (the mask must be judged against the signal; a translucent fill hides it), and
    is memory-cheap: no ax.contour path explosion on the big ragged 0.65 um masks (that spiked >2.8 GB).
    `mask` may be downsampled; `extent` (true-pixel [0,W,H,0]) registers it to the full-res grid."""
    if not mask.any():
        return
    edge = ndi.binary_dilation(mask, iterations=int(width)) & ~mask
    ov = np.zeros(mask.shape + (4,), dtype=np.float32)
    ov[..., 0], ov[..., 1], ov[..., 2] = rgb
    ov[..., 3] = np.where(edge, 1.0, 0.0)
    ax.imshow(ov, extent=extent)


def _half_gauss_thr(vals, sigma) -> float:
    """Half-Gaussian noise threshold from a 1-D array of (background-corrected) pixel values —
    the SAME computation as the pooled block below, factored out so MESP2 can be thresholded
    PER IMAGE (each morph's own in-mask distribution) instead of on the global cohort pool."""
    v = np.asarray(vals, dtype=np.float64); v = v[np.isfinite(v)]
    if v.size < 2:
        v = np.array([0.0, 1.0])
    lo, hi = np.quantile(v, [0.001, 0.999]); hi = hi if hi > lo else lo + 1.0
    edges = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 2049)
    counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
    d = M.half_gaussian_threshold_from_histogram(counts=counts, edges=edges, z=sigma)
    return float(d["threshold_value"])


def main():
    only, tscale, tscale_bf = None, 0.5, 0.7
    manifest_path, outtag = MANIFEST, ""      # override to run on another cohort (e.g. one-graft)
    foxf1_sigma = FOXF1_SIGMA
    foxf1_thresh_scale = FOXF1_THRESH_SCALE
    review = "--review" in sys.argv
    zreview = "--zreview" in sys.argv
    maskreview = "--maskreview" in sys.argv
    ifreview = "--ifreview" in sys.argv
    axisreview = "--axisreview" in sys.argv
    donorreview = "--donorreview" in sys.argv
    beadreview = "--beadreview" in sys.argv
    threshsweep = "--threshsweep" in sys.argv
    maskqc = "--maskqc" in sys.argv
    sweepscales = [1.0, 0.8, 0.65, 0.5, 0.4]
    sweepbase = None                       # explicit MESP2 base threshold for the sweep (canonical full-set value)
    zresolve = "--zresolve" in sys.argv
    zmarkers = "--zmarkers" in sys.argv
    axisqc = "--axisqc" in sys.argv          # compact 1-panel/morph axis overlay (length verification)
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = a.split("=", 1)[1].split(",")
        elif a.startswith("--tscale="):
            tscale = float(a.split("=", 1)[1])
        elif a.startswith("--tscale_bf="):
            tscale_bf = float(a.split("=", 1)[1])
        elif a.startswith("--manifest="):
            manifest_path = Path(a.split("=", 1)[1])
        elif a.startswith("--outtag="):
            outtag = a.split("=", 1)[1]
        elif a.startswith("--foxf1_sigma="):
            foxf1_sigma = float(a.split("=", 1)[1])
        elif a.startswith("--foxf1_thresh_scale="):
            foxf1_thresh_scale = float(a.split("=", 1)[1])
        elif a.startswith("--sweepscales="):
            sweepscales = [float(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--sweepbase="):
            sweepbase = float(a.split("=", 1)[1])
    df = pd.read_csv(manifest_path, sep="\t")
    if only:
        df = df[df["image_id"].isin(only)]
    # A subset run (--only) recomputes a NON-canonical pooled threshold -> never let it overwrite
    # the canonical full-set tables; route subset outputs to a "_sub" suffix instead.
    otag = outtag + ("_sub" if only else "")
    retained_anno, roi_anno, drop, seg_anno, donor_anno, exclude_anno, axis_anno, foxf1_anno, cut_anno = load_annotations()

    # ---- PASS 1: per-z centerlines -> consensus axis; single domain mask; threshold pool ----
    cache, pool = {}, {m: [] for m in MARKERS}
    for _, row in df.iterrows():
        iid = row["image_id"]
        if iid in drop:
            continue
        stack = czi_io.load_czi_stack(Path(row["file_path"]))
        di, nz, roi = int(row["ch_dapi"]), int(stack.data_czyx.shape[1]), roi_anno.get(iid)
        ts = tscale if di >= 0 else tscale_bf      # DAPI (clean) looser; inverted-BF (noisy) tighter
        sp = seg_anno.get(iid, {})
        ts, ms = sp.get("tscale", ts), sp.get("min_size", 4000)   # per-image overrides (annotation)
        oradius = sp.get("open_radius", 0)
        # (1) per-z masks -> per-z medial centerlines (these feed the consensus AXIS only)
        zmasks, zcl = {}, {}
        for z in range(nz):
            m = morph_mask_plane(stack, row, z, di, roi, ts, ms, cuts=cut_anno.get(iid))
            if int(m.sum()) < MIN_MASK_PX:
                continue
            zmasks[z] = m
            try:
                zcl[z] = np.asarray(M.centerline_from_mask(m)["centerline_xy"], dtype=float)
            except Exception:
                pass
        valid_z = sorted(zcl.keys())
        ret = [z for z in retained_anno.get(iid, valid_z) if z in zcl] or valid_z
        axis, length_px = None, np.nan
        if ret:
            ref = zmasks[ret[len(ret) // 2]]
            try:
                cons = M.consensus_centerline_from_paths([zcl[z] for z in ret], reference_mask=ref)
                axis, length_px = np.asarray(cons["centerline_xy"], dtype=float), float(cons["length_px"])
            except Exception:
                z0 = max(zcl, key=lambda z: arc_len_px(zcl[z]))
                axis, length_px = zcl[z0], arc_len_px(zcl[z0])
        # (2) SINGLE domain mask (max-proj); beads excluded. Built BEFORE the manual axis so a
        #     hand-drawn AP axis can be extended out to the mask ends (full morph-length span).
        dmask = domain_mask(stack, row, di, roi, ts, ms, nz, oradius, cuts=cut_anno.get(iid))
        for (ex0, ey0, ex1, ey1) in exclude_anno.get(iid, []):    # remove annotated debris/bead regions
            dmask[max(0, ey0):ey1, max(0, ex0):ex1] = False
        am = axis_anno.get(iid)                     # manual AP axis (pts, extend?) — overrides medial
        if am is not None and len(am[0]) >= 2:
            am_pts, am_ext = am
            poly = extend_polyline_to_mask(am_pts, dmask) if am_ext else am_pts  # no-extend keeps the tail OUT of the contiguous graft
            seglen = np.linalg.norm(np.diff(poly, axis=0), axis=1); total = float(seglen.sum())
            if total > 0:                           # densify the polyline to 100 pts along arc length
                cum = np.concatenate([[0.0], np.cumsum(seglen)]); s = np.linspace(0, total, 100)
                axis = np.column_stack([np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])])
                length_px = total
        img_pix = {m: [] for m in MARKERS}           # this morph's own pixels per marker -> per-image threshold
        if not (maskreview or zreview or axisreview or zresolve or zmarkers or axisqc):
            for z in (ret or valid_z):
                for mk in MARKERS:
                    c, _, _ = M.background_correct_signal(
                        np.asarray(stack.data_czyx[int(row[f"ch_{mk}"]), z], dtype=np.float32),
                        dmask, background_estimator="whole_off_morph")
                    v = c[dmask & np.isfinite(c)].astype(np.float32)
                    if v.size:
                        vv = (v if v.size <= SAMPLE_PER_PLANE
                              else v[RNG.choice(v.size, SAMPLE_PER_PLANE, replace=False)])
                        pool[mk].append(vv)          # global cohort pool (reference / fallback only)
                        img_pix[mk].append(v)        # per-image pool = ALL in-mask pixels (NOT the
                        #                              RNG-subsample) -> per-image threshold is
                        #                              deterministic + independent of run composition
        # MESP2 + FOXF1 = PER-IMAGE half-Gaussian thresholds. The global cohort pool is dragged by
        # bright / high-background morphs -> too tight for individual morphs (ng12 MESP2
        # global 581 -> own-pool 213 = the ×0.37 chosen; ng15 FOXF1 global 425 -> per-image 126
        # captures a real domain the global cut called 0.000). Each morph's OWN in-mask distribution;
        # MESP2 sigma=3, FOXF1 sigma=4 ×1.3 (same method + convention as the global, just per-image).
        mesp2_thr_img = _half_gauss_thr(np.concatenate(img_pix["mesp2"]), THRESHOLD_SIGMA) if img_pix["mesp2"] else None
        foxf1_thr_img = (_half_gauss_thr(np.concatenate(img_pix["foxf1"]), foxf1_sigma)
                         * foxf1_thresh_scale * float(foxf1_anno.get(iid, 1.0))    # per-image manual tune
                         if img_pix["foxf1"] else None)
        # Retain per-z masks ONLY for the z-review viz modes — they are ~1.4 GB across a full
        # cohort (every z of every morph) and are not needed by the quant or the other reviews.
        cache[iid] = dict(row=row, zmasks=(zmasks if (zreview or maskreview) else {}),
                          zcl=zcl, ret=ret, valid_z=valid_z, axis=axis, length_px=length_px,
                          dmask=dmask, mesp2_thr=mesp2_thr_img, foxf1_thr=foxf1_thr_img)
        del stack; gc.collect()          # free the CZI stack each iteration (peak-memory guard, <2 GB)

    if zreview:
        # one z per panel so each per-z centerline is unambiguous (for retained-z selection)
        ids = list(cache.keys())
        maxz = max((len(cache[i]["zcl"]) for i in ids), default=1)
        fig, axes = plt.subplots(len(ids), maxz, figsize=(1.8 * maxz, 1.85 * len(ids)), squeeze=False)
        for r, iid in enumerate(ids):
            e = cache[iid]; zs = sorted(e["zcl"].keys())
            for c in range(maxz):
                ax = axes[r][c]; ax.set_xticks([]); ax.set_yticks([])
                if c >= len(zs):
                    ax.axis("off"); continue
                z = zs[c]
                ax.imshow(e["zmasks"][z], cmap="gray_r", vmin=0, vmax=1)
                cl = e["zcl"][z]; ax.plot(cl[:, 0], cl[:, 1], "-", color="red", lw=1.4)
                ax.set_title(f"{iid}  z{z}", fontsize=7)
        for r, iid in enumerate(ids):
            axes[r][0].set_ylabel(iid, fontsize=7, rotation=0, ha="right", va="center", labelpad=18)
        fig.suptitle("Per-z medial centerlines — ONE z per panel (mask=gray, centerline=red). "
                     "Per organoid, tell me which z to DROP (keep the good ones).", fontsize=12, y=0.999)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.savefig(QC / "05_perz_zpanels.png", dpi=110); plt.close(fig)
        print(f"wrote {QC / '05_perz_zpanels.png'}")
        return

    if maskreview:
        # the SINGLE max-proj domain mask per organoid (ONE outline to sign off) on the
        # exact source it was segmented from -> judge right object + right boundary.
        ids = list(cache.keys()); ncol, nrow = 4, int(np.ceil(len(ids) / 4))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.6 * nrow), squeeze=False)
        for k, iid in enumerate(ids):
            e = cache[iid]; row = e["row"]; ax = axes[k // ncol][k % ncol]
            stack = czi_io.load_czi_stack(Path(row["file_path"]))
            di = int(row["ch_dapi"])
            src = source_projection(stack, row, di)
            src_name = "DAPI max-proj" if di >= 0 else "BF inverted min-proj (surrogate)"
            px, py = float(row["pixel_um_x"]), float(row["pixel_um_y"])
            area = int(e["dmask"].sum()) * px * py
            ax.imshow(M.robust_rescale(src), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
            ax.contour(e["dmask"], colors="lime", linewidths=1.1)
            ax.set_title(f"{iid}  {row['organoid_label'][:15]}\n{src_name}  ·  {area/1e3:.0f}e3 µm²",
                         fontsize=7.5)
        for k in range(len(ids), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("DOMAIN MASK CHECK — ONE max-projection mask per organoid (lime) on its segmentation "
                     "source. This single region is the denominator for all fluorescent-domain quant. OK?",
                     fontsize=12, y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(QC / "05_domain_mask_check.png", dpi=125); plt.close(fig)
        print(f"wrote {QC / '05_domain_mask_check.png'}")
        return

    if axisreview:
        # DAPI/BF + MESP2 (somite=axial cue) + FOXF1 (LPM=lateral), with the current consensus
        # axis (cyan) + per-z centerlines (#=z) + pixel ticks -> pick retained_z, or read off
        # head->tail (x,y) endpoints for a manual axis (annotations/axis.tsv).
        ids = list(cache.keys()); ncol = 4
        fig, axes = plt.subplots(len(ids), ncol, figsize=(4.7 * ncol, 4.7 * len(ids)), squeeze=False)
        for r, iid in enumerate(ids):
            e = cache[iid]; row = e["row"]; stack = czi_io.load_czi_stack(Path(row["file_path"]))
            di = int(row["ch_dapi"])
            _d = row.get("ch_donor", None)              # bead graft = region to AVOID (ch_donor, else LPM-bead)
            dch = int(_d) if (_d is not None and pd.notna(_d) and int(_d) >= 0) else int(row["ch_lpm_bead"])
            panels = [("DAPI" if di >= 0 else "BF", di if di >= 0 else int(row["ch_bf"])),
                      ("MESP2 host-somite", int(row["ch_mesp2"])), ("FOXF1 LPM", int(row["ch_foxf1"])),
                      ("DONOR graft (avoid)", dch)]
            for c, (nm, ci) in enumerate(panels):
                ax = axes[r][c]
                if ci < 0:
                    ax.axis("off"); ax.set_title(f"{iid} {nm} n/a", fontsize=8); continue
                img = stack.data_czyx[ci].max(axis=0)
                ax.imshow(M.robust_rescale(img), cmap="gray")
                ax.contour(e["dmask"], colors="lime", linewidths=0.8, alpha=0.6)  # domain mask = extent the axis must span
                zs = sorted(e["zcl"].keys())
                for zi, z in enumerate(zs):
                    cl = e["zcl"][z]; col = plt.cm.autumn(zi / max(len(zs) - 1, 1))
                    ax.plot(cl[:, 0], cl[:, 1], "-", color=col, lw=0.7, alpha=0.65)
                    mid = cl[len(cl) // 2]; ax.text(mid[0], mid[1], str(z), color=col, fontsize=7)
                if e["axis"] is not None:
                    ax.plot(e["axis"][:, 0], e["axis"][:, 1], "-", color="cyan", lw=2.5)
                ax.set_xticks(range(0, img.shape[1], 128)); ax.set_yticks(range(0, img.shape[0], 128))
                ax.grid(True, color="white", alpha=0.25, lw=0.5); ax.tick_params(labelsize=6)
                ax.set_title(f"{iid} {nm}  retained_z={e['ret']}", fontsize=8)
        fig.suptitle("AXIS RESOLVE — DAPI/BF + MESP2 (somite=axial) + FOXF1 (LPM=lateral). cyan = current "
                     "consensus axis; thin warm = per-z centerlines (#=z); grid = px. Reply retained_z to keep, "
                     "or head->tail (x,y) endpoints for a manual axis.", fontsize=11, y=0.999)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        for iid in ids:                              # confirm retained-z / manual-axis span without reloading the image
            e = cache[iid]; extra = ""
            if iid in axis_anno and e["axis"] is not None:
                extra = f" manual head={e['axis'][0].round(1)} tail={e['axis'][-1].round(1)}"
            print(f"  {iid}: retained_z={e['ret']} length_px={e['length_px']:.0f}{extra}")
        fig.savefig(QC / "05_axis_resolve.png", dpi=120); plt.close(fig)
        print(f"wrote {QC / '05_axis_resolve.png'}")
        return

    if axisqc:
        # COMPACT axis verification (m4-iii length): ONE panel per morph — source projection +
        # domain-mask outline (green) + consensus axis (cyan) + endpoints (magenta) + length label.
        # Plus a numeric triage printout: tortuosity = arc / straight-chord (>>1 = a wandering /
        # protrusion-seeking skeleton -> length likely inflated), footprint, n_z. Memory-safe reload.
        ids = list(cache.keys()); ncol = 5; nrow = int(np.ceil(len(ids) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.7 * nrow), squeeze=False)
        print("axis QC (length verification):")
        print("  %-6s %8s %8s %6s %8s %4s  %s" % ("id", "len_um", "len_px", "tort", "foot_e3", "nz", "src"))
        for k, iid in enumerate(ids):
            e = cache[iid]; row = e["row"]; ax = axes[k // ncol][k % ncol]
            stack = czi_io.load_czi_stack(Path(row["file_path"])); di = int(row["ch_dapi"])
            src = source_projection(stack, row, di)
            px, py = float(row["pixel_um_x"]), float(row["pixel_um_y"])
            H, W = e["dmask"].shape; st = max(1, int(np.ceil(max(H, W) / 900))); ext = [0, W, H, 0]
            ax.imshow(M.robust_rescale(src)[::st, ::st], cmap="gray", extent=ext)
            _outline_mask(ax, e["dmask"][::st, ::st], (0.4, 1.0, 0.4), extent=ext)
            axy = e["axis"]; tort = np.nan
            length_um = e["length_px"] * px if np.isfinite(e["length_px"]) else np.nan
            if axy is not None and len(axy) >= 2:
                ax.plot(axy[:, 0], axy[:, 1], "-", color="cyan", lw=2.2)
                ax.plot([axy[0, 0], axy[-1, 0]], [axy[0, 1], axy[-1, 1]], "o", color="magenta", ms=4)
                chord = float(np.linalg.norm(axy[-1] - axy[0]))
                tort = (e["length_px"] / chord) if chord > 0 else np.nan
            ax.set_xticks([]); ax.set_yticks([])
            src_tag = "DAPI" if di >= 0 else "BF"
            ax.set_title(f"{iid}  L={length_um:.0f}µm\ntort={tort:.2f}  {src_tag} {px:g}µm/px", fontsize=8)
            foot = int(e["dmask"].sum()) * px * py
            print("  %-6s %8.0f %8.0f %6.2f %8.0f %4d  %s" % (
                iid, length_um, e["length_px"], tort, foot / 1e3, len(e["ret"] or e["valid_z"]), src_tag))
            del stack; gc.collect()
        for k in range(len(ids), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("AXIS QC (m4-iii morph length) — source + domain mask (green) + consensus axis (cyan) "
                     "+ endpoints (magenta). High tortuosity / oversized footprint = length to verify.",
                     fontsize=11, y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = QC / f"05_axisqc{otag}.png"; fig.savefig(out, dpi=130); plt.close(fig)
        print(f"wrote {out}")
        return

    if zresolve:
        # ONE organoid per file: each z's centerline on its own panel (over DAPI/BF max-proj +
        # that z's mask + px grid), plus a summary panel -> pick which z hold the true midline.
        for iid in list(cache.keys()):
            e = cache[iid]; row = e["row"]; stack = czi_io.load_czi_stack(Path(row["file_path"]))
            di = int(row["ch_dapi"]); bg = stack.data_czyx[di if di >= 0 else int(row["ch_bf"])].max(axis=0)
            zs = sorted(e["zcl"].keys()); n = len(zs) + 1; ncol = 4; nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.4 * nrow), squeeze=False)
            for k, z in enumerate(zs):
                ax = axes[k // ncol][k % ncol]
                ax.imshow(M.robust_rescale(bg), cmap="gray")
                ax.contour(e["zmasks"][z], colors="yellow", linewidths=0.5, alpha=0.4)
                cl = e["zcl"][z]; ax.plot(cl[:, 0], cl[:, 1], "-", color="cyan", lw=2.0)
                ax.set_xticks(range(0, bg.shape[1], 128)); ax.set_yticks(range(0, bg.shape[0], 128))
                ax.grid(True, color="white", alpha=0.25, lw=0.5); ax.tick_params(labelsize=6)
                ax.set_title(f"{iid} z{z}" + ("  [in retained_z]" if z in e["ret"] else ""), fontsize=8)
            axc = axes[len(zs) // ncol][len(zs) % ncol]         # summary: all z (#) + consensus (cyan)
            axc.imshow(M.robust_rescale(bg), cmap="gray")
            for zi, z in enumerate(zs):
                cl = e["zcl"][z]; col = plt.cm.autumn(zi / max(len(zs) - 1, 1))
                axc.plot(cl[:, 0], cl[:, 1], "-", color=col, lw=0.9)
                mid = cl[len(cl) // 2]; axc.text(mid[0], mid[1], str(z), color=col, fontsize=7)
            if e["axis"] is not None:
                axc.plot(e["axis"][:, 0], e["axis"][:, 1], "-", color="cyan", lw=2.2)
            axc.set_xticks(range(0, bg.shape[1], 128)); axc.set_yticks(range(0, bg.shape[0], 128))
            axc.grid(True, color="white", alpha=0.25, lw=0.5); axc.tick_params(labelsize=6)
            axc.set_title(f"{iid} ALL z (#) + consensus (cyan)", fontsize=8)
            for k in range(n, nrow * ncol):
                axes[k // ncol][k % ncol].axis("off")
            fig.suptitle(f"{iid} — per-z centerlines (cyan), one z per panel + that z's mask (yellow) + px grid. "
                         f"current retained_z={e['ret']}. Tell me which z to keep.", fontsize=11, y=0.999)
            fig.tight_layout(rect=(0, 0, 1, 0.99))
            out = QC / f"05_zresolve_{iid}.png"; fig.savefig(out, dpi=120); plt.close(fig)
            print(f"wrote {out}")
        return

    if zmarkers:
        # each z's centerline (cyan) on all 3 channels (DAPI/BF, MESP2=somite, FOXF1=LPM) + px grid,
        # so the marker layout can decide the axis / which lobe is the graft. One file per organoid.
        for iid in list(cache.keys()):
            e = cache[iid]; row = e["row"]; stack = czi_io.load_czi_stack(Path(row["file_path"]))
            di = int(row["ch_dapi"])
            chans = [("DAPI" if di >= 0 else "BF", di if di >= 0 else int(row["ch_bf"])),
                     ("MESP2", int(row["ch_mesp2"])), ("FOXF1", int(row["ch_foxf1"]))]
            proj = {ci: M.robust_rescale(stack.data_czyx[ci].max(axis=0)) for _, ci in chans}
            zs = sorted(e["zcl"].keys()); nrow = len(zs); ncol = 3
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow), squeeze=False)
            for r, z in enumerate(zs):
                cl = e["zcl"][z]
                for c, (nm, ci) in enumerate(chans):
                    ax = axes[r][c]; im = proj[ci]
                    ax.imshow(im, cmap="gray"); ax.plot(cl[:, 0], cl[:, 1], "-", color="cyan", lw=2.0)
                    ax.set_xticks(range(0, im.shape[1], 128)); ax.set_yticks(range(0, im.shape[0], 128))
                    ax.grid(True, color="white", alpha=0.25, lw=0.5); ax.tick_params(labelsize=6)
                    ax.set_title(f"{iid} z{z} {nm}" + ("  [retained]" if z in e["ret"] else ""), fontsize=8)
            fig.suptitle(f"{iid} — each z's centerline (cyan) on DAPI/BF + MESP2 (somite) + FOXF1 (LPM) + px grid. "
                         f"Pick which z to keep, or give head->tail endpoints.", fontsize=11, y=0.999)
            fig.tight_layout(rect=(0, 0, 1, 0.995))
            out = QC / f"05_zmarkers_{iid}.png"; fig.savefig(out, dpi=118); plt.close(fig)
            print(f"wrote {out}")
        return

    # pooled half-Gaussian threshold per marker (same method as day-5)
    thr, trows = {}, []
    for mk in MARKERS:
        v = np.concatenate(pool[mk]).astype(np.float64) if pool[mk] else np.array([0.0, 1.0])
        v = v[np.isfinite(v)]
        lo, hi = np.quantile(v, [0.001, 0.999]); hi = hi if hi > lo else lo + 1.0
        edges = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 2049)
        counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
        sig = foxf1_sigma if mk == "foxf1" else THRESHOLD_SIGMA
        d = M.half_gaussian_threshold_from_histogram(counts=counts, edges=edges, z=sig)
        thr[mk] = float(d["threshold_value"])
        tscl = foxf1_thresh_scale if mk == "foxf1" else 1.0          # global FOXF1 visual-calibration tightening
        thr[mk] *= tscl
        trows.append({"marker": mk, "sigma": sig, "thresh_scale": tscl, "threshold": thr[mk], "n_pooled": int(v.size)})
    TAB.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trows).to_csv(TAB / f"05_marker_thresholds{otag}.tsv", sep="\t", index=False)
    for iid in cache:                                # per-image thresholds; fall back to global if a morph had no pool
        if cache[iid].get("mesp2_thr") is None:
            cache[iid]["mesp2_thr"] = thr["mesp2"]
        if cache[iid].get("foxf1_thr") is None:
            cache[iid]["foxf1_thr"] = thr["foxf1"]
    print("per-image thresholds (global pool: MESP2=%.0f FOXF1=%.0f):" % (thr["mesp2"], thr["foxf1"]))
    print("  MESP2: " + ", ".join(f"{i}={cache[i]['mesp2_thr']:.0f}" for i in cache))
    print("  FOXF1: " + ", ".join(f"{i}={cache[i]['foxf1_thr']:.0f}" for i in cache))
    if only:
        # A subset run's GLOBAL pooled threshold is non-canonical (FOXF1 esp.: the pool is dominated
        # by whatever few morphs are in --only). Override the global thresholds with the saved FULL-set
        # values so review modes render the SAME FOXF1 mask the real quant does. (MESP2 is per-image, so
        # it is already correct regardless of the run set.)
        ctf = TAB / f"05_marker_thresholds{outtag}.tsv"       # canonical full-set table (NOT the _sub subset)
        if ctf.exists():
            ct = pd.read_csv(ctf, sep="\t")
            for _, rr in ct.iterrows():
                if str(rr["marker"]) in thr:
                    thr[str(rr["marker"])] = float(rr["threshold"])
            print("[--only] using canonical global thresholds from %s: %s" % (
                ctf.name, ", ".join(f"{k}={thr[k]:.0f}" for k in thr)))

    # ---- PASS 2: fluorescent domains + bilaterality, inside the SINGLE domain mask,
    #              averaged over the retained (in-focus) planes; axis = per-z consensus ----
    perz, perorg = [], []
    for iid, e in cache.items():
        row = e["row"]; stack = czi_io.load_czi_stack(Path(row["file_path"]))
        px, py, axis = float(row["pixel_um_x"]), float(row["pixel_um_y"]), e["axis"]
        dmask = e["dmask"]; mpx = int(dmask.sum())
        agg = {m: {"area": [], "frac": [], "bilat": [], "wbilat": []} for m in MARKERS}
        donor_chs = donor_channels_for_row(row)                      # HOST FOXF1 = FOXF1+ minus donor
        da = donor_anno.get(iid, {})
        dscale = float(da.get("thresh_scale", 1.0))
        dclose = int(da.get("close_radius", 8)); dminf = float(da.get("min_frac", 0.005))
        duse_ratio = str(da.get("signal", "ratio")) != "raw"
        if donor_chs:
            dmsk, donor_is_ratio, donor_thr = donor_mask(stack, row, dmask, donor_chs,
                                                         int(row["ch_dapi"]), dscale, dclose, dminf, duse_ratio)
        else:
            dmsk, donor_is_ratio, donor_thr = np.zeros(dmask.shape, dtype=bool), False, {}
        agg_host = {"frac": [], "bilat": []}
        for z in (e["ret"] or e["valid_z"]):
            rec = {"image_id": iid, "group": row["group"], "z": z, "morph_area_um2": mpx * px * py}
            for mk in MARKERS:
                c, _, _ = M.background_correct_signal(
                    np.asarray(stack.data_czyx[int(row[f"ch_{mk}"]), z], dtype=np.float32),
                    dmask, background_estimator="whole_off_morph")
                t_mk = e["mesp2_thr"] if mk == "mesp2" else e["foxf1_thr"]   # both markers per-image
                pos = M.positive_mask_from_corrected(c, dmask, t_mk); npos = int(pos.sum())
                frac = (npos / mpx) if mpx else np.nan
                bi, wbi = bilaterality(pos, c, axis)
                rec[f"{mk}_area_um2"], rec[f"{mk}_fraction"] = npos * px * py, frac
                rec[f"{mk}_bilaterality"], rec[f"{mk}_weighted_bilaterality"] = bi, wbi
                agg[mk]["area"].append(npos * px * py); agg[mk]["frac"].append(frac)
                agg[mk]["bilat"].append(bi); agg[mk]["wbilat"].append(wbi)
                if mk == "foxf1":                                    # host = FOXF1+ not overlapping the donor
                    hpos = pos & ~dmsk
                    hfrac = (int(hpos.sum()) / mpx) if mpx else np.nan
                    hbi, _ = bilaterality(hpos, c, axis)
                    rec["foxf1_host_fraction"], rec["foxf1_host_bilaterality"] = hfrac, hbi
                    agg_host["frac"].append(hfrac); agg_host["bilat"].append(hbi)
            perz.append(rec)
        r = {"image_id": iid, "organoid_label": row["organoid_label"], "group": row["group"],
             "condition": row["condition"], "quality_flag": row["quality_flag"],
             "mask_source": "dapi_maxproj" if int(row["ch_dapi"]) >= 0 else "bf_surrogate_minproj",
             "morph_footprint_um2": mpx * px * py,
             "n_z_domain": len(e["ret"] or e["valid_z"]), "n_z_axis": len(e["ret"]),
             "length_um": e["length_px"] * px if np.isfinite(e["length_px"]) else np.nan}
        for mk in MARKERS:
            fr = np.array(agg[mk]["frac"], dtype=float)
            r[f"{mk}_area_um2_mean"] = float(np.nanmean(agg[mk]["area"])) if agg[mk]["area"] else np.nan
            r[f"{mk}_fraction_mean"] = float(np.nanmean(fr)) if fr.size else np.nan
            r[f"{mk}_fraction_max"] = float(np.nanmax(fr)) if fr.size else np.nan
            # present = a bona-fide positive region exists (same segmentation that feeds the
            # bilaterality index), NOT an arbitrary area cutoff. positive_mask already applies
            # the fixed threshold + min_object_size, so any surviving domain is real.
            r[f"{mk}_present"] = bool(np.nanmax(fr) > 0) if fr.size else False
            bvals = np.array(agg[mk]["bilat"], dtype=float)              # NaN where a plane had <3 pos px
            has = bool(np.isfinite(bvals).any())                        # neg-ctrl has none -> NaN (no warning)
            r[f"{mk}_bilaterality_mean"] = float(np.nanmean(bvals)) if has else np.nan
            r[f"{mk}_bilaterality_median"] = float(np.nanmedian(bvals)) if has else np.nan
        hf = np.array(agg_host["frac"], dtype=float); hb = np.array(agg_host["bilat"], dtype=float)
        r["foxf1_host_fraction_mean"] = float(np.nanmean(hf)) if hf.size and np.isfinite(hf).any() else np.nan
        r["foxf1_host_present"] = bool(np.nanmax(hf) > 0) if hf.size and np.isfinite(hf).any() else False
        r["foxf1_host_bilaterality_mean"] = float(np.nanmean(hb)) if np.isfinite(hb).any() else np.nan
        r["n_donor_channels"] = len(donor_chs)
        r["foxf1_host_method"] = ("none" if not donor_chs else
                                  ("dapi_ratio" if donor_is_ratio else
                                   ("raw_noDAPI" if int(row["ch_dapi"]) < 0 else "raw_bychoice")))
        r["donor_thresh_scale"] = dscale
        r["donor_close_radius"] = dclose
        r["donor_min_frac"] = dminf
        r["donor_signal"] = "ratio" if donor_is_ratio else "raw"   # ACTUAL (no-DAPI requests ratio but falls back to raw)
        r["mesp2_threshold"] = float(e["mesp2_thr"]) if e.get("mesp2_thr") is not None else np.nan
        r["foxf1_threshold"] = float(e["foxf1_thr"]) if e.get("foxf1_thr") is not None else np.nan
        r["foxf1_manual_scale"] = float(foxf1_anno.get(iid, 1.0))
        r["mesp2_threshold_method"] = r["foxf1_threshold_method"] = "per_image_halfgauss"
        r["success_mesp2_and_foxf1"] = bool(r["mesp2_present"] and r["foxf1_present"])
        perorg.append(r)
        del stack; gc.collect()          # free the CZI stack each iteration (peak-memory guard, <2 GB)

    pd.DataFrame(perz).to_csv(TAB / f"05_per_z_metrics{otag}.tsv", sep="\t", index=False)
    od = pd.DataFrame(perorg); od.to_csv(TAB / f"05_per_organoid_metrics{otag}.tsv", sep="\t", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 60)
    show = ["image_id", "organoid_label", "group", "condition", "mask_source", "n_z_domain", "n_z_axis",
            "length_um", "mesp2_present", "mesp2_fraction_mean", "mesp2_bilaterality_mean",
            "foxf1_present", "foxf1_fraction_mean", "foxf1_bilaterality_mean", "success_mesp2_and_foxf1"]
    print(od[show].round(3).to_string(index=False))

    if review:
        ids = list(cache.keys()); ncol, nrow = 4, int(np.ceil(len(ids) / 4))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.5 * nrow), squeeze=False)
        for k, iid in enumerate(ids):
            e = cache[iid]; row = e["row"]; ax = axes[k // ncol][k % ncol]
            stack = czi_io.load_czi_stack(Path(row["file_path"]))
            ax.imshow(M.robust_rescale(stack.data_czyx[int(row["ch_bf"])].max(axis=0)), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            zs = sorted(e["zcl"].keys())
            for zi, z in enumerate(zs):
                cl = e["zcl"][z]; col = plt.cm.viridis(zi / max(len(zs) - 1, 1))
                ax.plot(cl[:, 0], cl[:, 1], "-", color=col, lw=0.8, alpha=0.85)
                mid = cl[len(cl) // 2]; ax.text(mid[0], mid[1], str(z), color=col, fontsize=6)
            if e["axis"] is not None:
                ax.plot(e["axis"][:, 0], e["axis"][:, 1], "-", color="white", lw=2.2)
            ax.set_title(f"{iid} {row['organoid_label'][:14]}  z={zs}", fontsize=7)
        for k in range(len(ids), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("Per-z centerlines (colored by z, # = z index) + consensus axis (white). "
                     "Reply per organoid: retained_z=[list] / 'all' / drop", fontsize=11, y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(QC / "05_perz_centerline_review.png", dpi=120); plt.close(fig)
        print(f"wrote {QC / '05_perz_centerline_review.png'}")

    if ifreview:
        # eyes-on-data: marker signal (max-proj) + called-positive domain (union over the
        # retained planes) + domain-mask boundary + axis -> are the domains real / thresholds sane?
        ids = list(cache.keys()); per_org = 2; ncol = 2 * per_org
        nrow = int(np.ceil(len(ids) / per_org))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.9 * nrow), squeeze=False)
        mcolor = {"mesp2": "red", "foxf1": "lime"}
        for k, iid in enumerate(ids):
            e = cache[iid]; row = e["row"]; dmask = e["dmask"]; axis = e["axis"]
            stack = czi_io.load_czi_stack(Path(row["file_path"]))
            for j, mk in enumerate(MARKERS):
                ax = axes[k // per_org][(k % per_org) * 2 + j]
                ax.imshow(M.robust_rescale(stack.data_czyx[int(row[f"ch_{mk}"])].max(axis=0)), cmap="gray")
                ax.set_xticks([]); ax.set_yticks([])
                posu = np.zeros(dmask.shape, dtype=bool); fracs = []
                for z in (e["ret"] or e["valid_z"]):
                    c, _, _ = M.background_correct_signal(
                        np.asarray(stack.data_czyx[int(row[f"ch_{mk}"]), z], dtype=np.float32),
                        dmask, background_estimator="whole_off_morph")
                    tmk = e["mesp2_thr"] if mk == "mesp2" else e["foxf1_thr"]   # both per-image
                    p = M.positive_mask_from_corrected(c, dmask, tmk)
                    posu |= p; fracs.append(int(p.sum()) / max(int(dmask.sum()), 1))
                ax.contour(dmask, colors="cyan", linewidths=0.6, alpha=0.5)
                if posu.any():
                    ax.contour(posu, colors=mcolor[mk], linewidths=0.8)
                if axis is not None:
                    ax.plot(axis[:, 0], axis[:, 1], "-", color="white", lw=1.0, alpha=0.85)
                ax.set_title(f"{iid} {mk.upper()}  frac~{np.mean(fracs):.2f}", fontsize=7.5)
        for s in range(len(ids), nrow * per_org):
            cb = (s % per_org) * 2; axes[s // per_org][cb].axis("off"); axes[s // per_org][cb + 1].axis("off")
        fig.suptitle("IF DOMAIN CHECK — marker max-proj (gray) + called-positive domain (MESP2 red / FOXF1 lime, "
                     "union over retained z) + morph mask (cyan) + consensus axis (white). Are the domains real?",
                     fontsize=12, y=0.999)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.savefig(QC / f"05_if_domain_check{outtag}.png", dpi=115); plt.close(fig)
        print(f"wrote {QC / '05_if_domain_check.png'}")

    if donorreview:
        # eyes-on-data for HOST FOXF1 = FOXF1+ minus donor. Per organoid: (A) donor channel (LPM-bead
        # EGFP) max-proj + the donor mask that gets SUBTRACTED (magenta); (B) FOXF1 max-proj + FOXF1+
        # domain (lime) with the donor mask (magenta) removed -> what survives = host FOXF1 (yellow).
        # Judge: does the donor mask sit on real donor cells, and is the leftover host FOXF1 believable?
        ids = list(cache.keys()); nrow = len(ids)
        fig, axes = plt.subplots(nrow, 2, figsize=(7.8, 3.3 * nrow), squeeze=False)
        for k, iid in enumerate(ids):
            e = cache[iid]; row = e["row"]; dmask = e["dmask"]
            stack = czi_io.load_czi_stack(Path(row["file_path"]))
            donor_chs = donor_channels_for_row(row)
            da = donor_anno.get(iid, {})
            dscale = float(da.get("thresh_scale", 1.0))
            dclose = int(da.get("close_radius", 8)); dminf = float(da.get("min_frac", 0.005))
            duse_ratio = str(da.get("signal", "ratio")) != "raw"
            ci_l = int(row["ch_lpm_bead"])
            if donor_chs:
                dmsk, is_ratio, _ = donor_mask(stack, row, dmask, donor_chs, int(row["ch_dapi"]),
                                               dscale, dclose, dminf, duse_ratio)
                # LPM-channel raw seed (before coherence cleanup) for the diagnostic overlay
                seed_l = (donor_seed(stack, row, dmask, ci_l, int(row["ch_dapi"]), dscale, duse_ratio)[0]
                          if ci_l >= 0 else np.zeros(dmask.shape, dtype=bool))
            else:
                dmsk, is_ratio, seed_l = np.zeros(dmask.shape, dtype=bool), False, np.zeros(dmask.shape, dtype=bool)
            posu = np.zeros(dmask.shape, dtype=bool)                  # FOXF1+ union over retained planes
            for z in (e["ret"] or e["valid_z"]):
                c, _, _ = M.background_correct_signal(
                    np.asarray(stack.data_czyx[int(row["ch_foxf1"]), z], dtype=np.float32),
                    dmask, background_estimator="whole_off_morph")
                posu |= M.positive_mask_from_corrected(c, dmask, e["foxf1_thr"])
            host = posu & ~dmsk
            mpx = max(int(dmask.sum()), 1)
            dfrac, hfrac = int(dmsk.sum()) / mpx, int(host.sum()) / mpx
            donor_disp = (stack.data_czyx[ci_l].max(axis=0) if ci_l >= 0
                          else stack.data_czyx[int(row["ch_foxf1"])].max(axis=0))
            axA = axes[k][0]
            axA.imshow(M.robust_rescale(donor_disp), cmap="gray")
            axA.contour(dmask, colors="cyan", linewidths=0.6, alpha=0.5)
            if seed_l.any():                                         # raw seed (pre-cleanup) = white dotted
                axA.contour(seed_l, colors="white", linewidths=0.5, linestyles="dotted", alpha=0.9)
            if dmsk.any():                                           # final donor graft = solid magenta
                axA.contour(dmsk, colors="magenta", linewidths=1.1)
            axA.set_xticks([]); axA.set_yticks([])
            tag = "ratio" if is_ratio else "raw"
            axA.set_title(f"{iid}  EGFP — graft = {dfrac:.0%} of morph  [{tag} ×{dscale:g}]", fontsize=7.5)
            axB = axes[k][1]
            axB.imshow(M.robust_rescale(stack.data_czyx[int(row["ch_foxf1"])].max(axis=0)), cmap="gray")
            axB.contour(dmask, colors="cyan", linewidths=0.6, alpha=0.5)
            if posu.any():                               # green (all FOXF1+) first + THICK so it shows under/around yellow
                axB.contour(posu, colors="lime", linewidths=1.8, alpha=0.9)
            if dmsk.any():
                axB.contour(dmsk, colors="magenta", linewidths=0.6, alpha=0.5)
            if host.any():                               # yellow (host) drawn LAST -> on top; thick green peeks out where they overlap
                axB.contour(host, colors="yellow", linewidths=0.9)
            axB.set_xticks([]); axB.set_yticks([])
            axB.set_title(f"{iid}  FOXF1 — host FOXF1 = {hfrac:.0%} of morph", fontsize=7.5)
        handles = [
            Line2D([0], [0], color="cyan", lw=1.4, label="morph outline"),
            Line2D([0], [0], color="white", lw=1.2, ls=":", label="raw seed (ignore)"),
            Line2D([0], [0], color="magenta", lw=1.8, label="donor GRAFT (subtracted)"),
            Line2D([0], [0], color="lime", lw=1.8, label="all FOXF1+"),
            Line2D([0], [0], color="yellow", lw=2.0, label="HOST FOXF1  (← the number)"),
        ]
        legy = 1.0 - 0.55 / fig.get_figheight()                      # ~0.55 inch below the top, height-independent
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, legy), ncol=3,
                   fontsize=8.5, framealpha=0.95)
        fig.suptitle("DONOR REVIEW — LEFT: does magenta GRAFT match the donor clump?   RIGHT: yellow = host FOXF1",
                     fontsize=9.5, y=1.0 - 0.13 / fig.get_figheight())
        fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.9 / fig.get_figheight()))
        out = QC / f"05_donor_review{outtag}.png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"wrote {out}")

    if beadreview:
        # eyes-on-data VIEWER (no auto-detection) for the MESP2 bead false-positive fix. Beads are
        # physical objects that block light -> DARK spots in brightfield (the discriminator used here); some
        # also autofluoresce in mCherry -> spurious MESP2+ blobs. Per morph, 3 panels with a pixel
        # grid so we can read off exclude.tsv bboxes by eye, image by image:
        #   A: BF MIN-proj (beads = darkest over z -> pop as dark spots)
        #   B: MESP2 max-proj + the CURRENT called-positive (red) = exactly what the quant counts
        #   C: the MESP2 called-positive (red) over the BF min-proj -> does a red blob sit on a bead?
        beaddir = QC / f"beadreview{outtag}"; beaddir.mkdir(parents=True, exist_ok=True)
        def _grid(ax, shp):
            step = 200 if max(shp) > 1200 else 100
            ax.set_xticks(np.arange(0, shp[1], step)); ax.set_yticks(np.arange(0, shp[0], step))
            ax.tick_params(labelsize=6, length=2)
            ax.grid(True, color="yellow", alpha=0.28, lw=0.4)
        for iid in cache:
            e = cache[iid]; row = e["row"]; dmask = e["dmask"]; axis = e["axis"]
            stack = czi_io.load_czi_stack(Path(row["file_path"]))
            bf_min = np.asarray(stack.data_czyx[int(row["ch_bf"])].min(axis=0), dtype=np.float32)
            mesp = np.asarray(stack.data_czyx[int(row["ch_mesp2"])].max(axis=0), dtype=np.float32)
            posu = np.zeros(dmask.shape, dtype=bool)
            for z in (e["ret"] or e["valid_z"]):
                c, _, _ = M.background_correct_signal(
                    np.asarray(stack.data_czyx[int(row["ch_mesp2"]), z], dtype=np.float32),
                    dmask, background_estimator="whole_off_morph")
                posu |= M.positive_mask_from_corrected(c, dmask, e["mesp2_thr"])
            frac = int(posu.sum()) / max(int(dmask.sum()), 1)
            # Downsample display for memory safety on the big 0.65 um (2062^2) morphs; `extent` keeps
            # the pixel grid + bboxes in TRUE pixel coords so exclude.tsv can still be read off here.
            H, W = bf_min.shape
            st = max(1, int(np.ceil(max(H, W) / 1100)))
            ext = [0, W, H, 0]
            bfd = M.robust_rescale(bf_min)[::st, ::st]
            mspd = M.robust_rescale(mesp)[::st, ::st]
            dmd = dmask[::st, ::st].astype(float)
            posd = posu[::st, ::st]
            xs = np.arange(dmd.shape[1]) * st; ys = np.arange(dmd.shape[0]) * st
            fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.7))
            ax[0].imshow(bfd, cmap="gray", extent=ext); _grid(ax[0], bf_min.shape)
            ax[0].contour(xs, ys, dmd, levels=[0.5], colors="cyan", linewidths=0.5, alpha=0.5)
            ax[0].set_title(f"{iid}  BF min-proj  (beads = dark spots)", fontsize=9)
            ax[1].imshow(mspd, cmap="gray", extent=ext); _grid(ax[1], mesp.shape)
            ax[1].contour(xs, ys, dmd, levels=[0.5], colors="cyan", linewidths=0.5, alpha=0.5)
            _outline_mask(ax[1], posd, (1.0, 0.0, 0.0), extent=ext)   # called MESP2 = red outline
            if axis is not None:
                ax[1].plot(axis[:, 0], axis[:, 1], "-", color="white", lw=0.9, alpha=0.8)
            ax[1].set_title(f"{iid}  MESP2 max-proj + called-positive (red)  frac~{frac:.3f}", fontsize=9)
            ax[2].imshow(bfd, cmap="gray", extent=ext); _grid(ax[2], bf_min.shape)
            ax[2].contour(xs, ys, dmd, levels=[0.5], colors="cyan", linewidths=0.5, alpha=0.5)
            _outline_mask(ax[2], posd, (1.0, 0.0, 0.0), extent=ext)   # called MESP2 = red outline
            for (ex0, ey0, ex1, ey1) in exclude_anno.get(iid, []):   # existing bboxes (if any) = orange
                ax[2].add_patch(Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, fill=False,
                                          edgecolor="orange", lw=1.2, ls="--"))
            ax[2].set_title(f"{iid}  MESP2-called (red) over BF  — does red sit on a dark bead?", fontsize=9)
            fig.suptitle(f"{iid}  {row['organoid_label']}  [{row.get('folder', '')}]  mask={('DAPI' if int(row['ch_dapi'])>=0 else 'BF-invert (no DAPI)')}"
                         f"   — beads block light (dark in BF); flag red MESP2 blobs sitting on a bead", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            out = beaddir / f"05_beadreview_{iid}.png"
            fig.savefig(out, dpi=125); plt.close(fig)
            print(f"wrote {out}")
            del stack; gc.collect()          # free the CZI stack each iteration (peak-memory guard, <2 GB)

    if threshsweep:
        # MESP2 threshold-calibration VIEWER: show the called-positive domain at a range of scalings
        # of the pooled MESP2 threshold, so a reviewer can pick where the mask matches the true somite domain
        # by eye. Base = the canonical FULL-set pooled threshold (read from the saved thresholds table
        # if present, since an --only subset recomputes a NON-canonical pool). Same per-z union method
        # as the quant. Existing exclude.tsv bead bboxes drawn (orange) for reference.
        base = thr["mesp2"]
        tf = TAB / f"05_marker_thresholds{outtag}.tsv"
        if tf.exists():
            _td = pd.read_csv(tf, sep="\t"); _m = _td[_td["marker"] == "mesp2"]
            if len(_m):
                base = float(_m["threshold"].iloc[0])
        if sweepbase is not None:                 # explicit canonical base (immune to --only pool recompute)
            base = sweepbase
        swdir = QC / f"threshsweep{outtag}"; swdir.mkdir(parents=True, exist_ok=True)
        for iid in cache:
            e = cache[iid]; row = e["row"]; dmask = e["dmask"]; axis = e["axis"]
            stack = czi_io.load_czi_stack(Path(row["file_path"]))
            mesp = np.asarray(stack.data_czyx[int(row["ch_mesp2"])].max(axis=0), dtype=np.float32)
            corrs = [M.background_correct_signal(
                        np.asarray(stack.data_czyx[int(row["ch_mesp2"]), z], dtype=np.float32),
                        dmask, background_estimator="whole_off_morph")[0]
                     for z in (e["ret"] or e["valid_z"])]
            n = len(sweepscales)
            fig, ax = plt.subplots(1, n, figsize=(4.7 * n, 5.4), squeeze=False); ax = ax[0]
            step = 200 if max(mesp.shape) > 1200 else 100
            for j, s in enumerate(sweepscales):
                t = base * s
                posu = np.zeros(dmask.shape, dtype=bool)
                for c in corrs:
                    posu |= M.positive_mask_from_corrected(c, dmask, t)
                frac = int(posu.sum()) / max(int(dmask.sum()), 1)
                ax[j].imshow(M.robust_rescale(mesp), cmap="gray")
                ax[j].set_xticks(np.arange(0, mesp.shape[1], step)); ax[j].set_yticks(np.arange(0, mesp.shape[0], step))
                ax[j].tick_params(labelsize=6, length=2); ax[j].grid(True, color="yellow", alpha=0.28, lw=0.4)
                ax[j].contour(dmask, colors="cyan", linewidths=0.5, alpha=0.5)
                _outline_mask(ax[j], posu, (1.0, 0.0, 0.0))        # called MESP2 = red outline (signal stays visible)
                if axis is not None:
                    ax[j].plot(axis[:, 0], axis[:, 1], "-", color="white", lw=0.8, alpha=0.75)
                for (ex0, ey0, ex1, ey1) in exclude_anno.get(iid, []):
                    ax[j].add_patch(Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, fill=False,
                                              edgecolor="orange", lw=1.2, ls="--"))
                tag = "  (current)" if abs(s - 1.0) < 1e-9 else ""
                ax[j].set_title(f"×{s:g}  thr={t:.0f}{tag}\nfrac~{frac:.3f}", fontsize=8.5)
            fig.suptitle(f"{iid}  {row['organoid_label']}  — MESP2 threshold sweep (base pooled thr={base:.0f})   "
                         f"pick where red matches the true somite domain", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            out = swdir / f"05_threshsweep_{iid}.png"
            fig.savefig(out, dpi=130); plt.close(fig)
            print(f"wrote {out}")
            del stack, corrs; gc.collect()   # free the CZI stack + per-z buffers (peak-memory guard, <2 GB)

    if maskqc:
        # Per-morph MASK QC, to judge each mask boundary against its OWN signal. Each mask is a
        # THIN outline (not a fill -> the signal stays visible) over the grayscale channel it was called
        # on. Downsampled + memory-safe (extent keeps the pixel grid in true coords). Panels:
        #   domain (denominator) | MESP2 (per-image thr) | FOXF1 (global thr) | donor graft | host FOXF1.
        # Masks are computed with the SAME method + thresholds as the quant, so this shows exactly what is counted.
        qcdir = QC / f"maskqc{outtag}"; qcdir.mkdir(parents=True, exist_ok=True)
        for iid in cache:
            e = cache[iid]; row = e["row"]; dmask = e["dmask"]; di = int(row["ch_dapi"])
            stack = czi_io.load_czi_stack(Path(row["file_path"]))
            mesp = np.asarray(stack.data_czyx[int(row["ch_mesp2"])].max(axis=0), dtype=np.float32)
            foxf = np.asarray(stack.data_czyx[int(row["ch_foxf1"])].max(axis=0), dtype=np.float32)
            src = source_projection(stack, row, di)
            posM = np.zeros(dmask.shape, dtype=bool); posF = np.zeros(dmask.shape, dtype=bool)
            for z in (e["ret"] or e["valid_z"]):
                cM, _, _ = M.background_correct_signal(
                    np.asarray(stack.data_czyx[int(row["ch_mesp2"]), z], dtype=np.float32),
                    dmask, background_estimator="whole_off_morph")
                posM |= M.positive_mask_from_corrected(cM, dmask, e["mesp2_thr"])
                cF, _, _ = M.background_correct_signal(
                    np.asarray(stack.data_czyx[int(row["ch_foxf1"]), z], dtype=np.float32),
                    dmask, background_estimator="whole_off_morph")
                posF |= M.positive_mask_from_corrected(cF, dmask, e["foxf1_thr"])
            donor_chs = donor_channels_for_row(row)
            da = donor_anno.get(iid, {})
            dsc = float(da.get("thresh_scale", 1.0)); dcl = int(da.get("close_radius", 8))
            dmf = float(da.get("min_frac", 0.005)); dur = str(da.get("signal", "ratio")) != "raw"
            dmsk = (donor_mask(stack, row, dmask, donor_chs, di, dsc, dcl, dmf, dur)[0]
                    if donor_chs else np.zeros(dmask.shape, dtype=bool))
            host = posF & ~dmsk
            ci_d = int(row["ch_lpm_bead"])
            donor_sig = np.asarray(stack.data_czyx[ci_d].max(axis=0), dtype=np.float32) if ci_d >= 0 else foxf
            mpx = max(int(dmask.sum()), 1)
            panels = [("domain (denominator)", src, dmask, (0.0, 1.0, 1.0), None),
                      (f"MESP2 (somite)  thr={e['mesp2_thr']:.0f}", mesp, posM, (1.0, 0.0, 0.0), int(posM.sum()) / mpx),
                      (f"FOXF1 (LPM)  thr={e['foxf1_thr']:.0f}", foxf, posF, (0.2, 1.0, 0.2), int(posF.sum()) / mpx),
                      ("donor graft", donor_sig, dmsk, (1.0, 0.0, 1.0), int(dmsk.sum()) / mpx),
                      ("host FOXF1 = FOXF1 − donor", foxf, host, (1.0, 1.0, 0.0), int(host.sum()) / mpx)]
            H, W = dmask.shape; st = max(1, int(np.ceil(max(H, W) / 1100))); ext = [0, W, H, 0]
            gstep = 200 if max(H, W) > 1200 else 100
            fig, ax = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 5.3))
            for j, (ttl, sig, msk, rgb, fr) in enumerate(panels):
                ax[j].imshow(M.robust_rescale(sig)[::st, ::st], cmap="gray", extent=ext)
                ax[j].set_xticks(np.arange(0, W, gstep)); ax[j].set_yticks(np.arange(0, H, gstep))
                ax[j].tick_params(labelsize=6, length=2); ax[j].grid(True, color="yellow", alpha=0.25, lw=0.4)
                _outline_mask(ax[j], msk[::st, ::st], rgb, extent=ext)
                ax[j].set_title(ttl + (f"   frac~{fr:.3f}" if fr is not None else ""), fontsize=8)
            fig.suptitle(f"{iid}  {row['organoid_label']}  [{row.get('folder', '')}]  mask={('DAPI' if di >= 0 else 'BF-invert')}"
                         f"  — MASK QC: thin outline over the signal each mask was called on", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            out = qcdir / f"05_maskqc_{iid}.png"; fig.savefig(out, dpi=125); plt.close(fig)
            print(f"wrote {out}")
            del stack; gc.collect()          # free the CZI stack each iteration (peak-memory guard, <2 GB)


if __name__ == "__main__":
    main()
