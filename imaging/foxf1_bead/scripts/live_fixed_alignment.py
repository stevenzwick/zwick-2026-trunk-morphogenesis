#!/usr/bin/env python3
"""
Cross-dataset alignment utilities for live day2 and fixed day2-fix data.

Reference space:
    registered LIVE large-image coordinate system

Primary derived transform:
    fixed small -> live small
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import feature, transform as sk_transform
from skimage.registration import phase_cross_correlation

# `scripts` is a package under the assay directory; put that directory on
# the path so this file runs from anywhere, as the READMEs show it being run.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from scripts import manual_well_annotation as mwa
from scripts import pipeline_common as common


LIVE_FIXED_ALIGNMENT_COLUMNS = [
    "canonical_position",
    "live_image_id",
    "fixed_image_id",
    "live_small_file_path",
    "live_large_file_path",
    "fixed_small_file_path",
    "fixed_large_file_path",
    "status",
    "fixed_large_to_live_large_theta_deg",
    "fixed_large_to_live_large_shift_x_px",
    "fixed_large_to_live_large_shift_y_px",
    "fixed_large_to_live_large_score",
    "live_large_focus_z_index",
    "fixed_large_focus_z_index",
    "fixed_small_to_live_small_m00",
    "fixed_small_to_live_small_m01",
    "fixed_small_to_live_small_m02",
    "fixed_small_to_live_small_m10",
    "fixed_small_to_live_small_m11",
    "fixed_small_to_live_small_m12",
    "fixed_small_to_live_large_m00",
    "fixed_small_to_live_large_m01",
    "fixed_small_to_live_large_m02",
    "fixed_small_to_live_large_m10",
    "fixed_small_to_live_large_m11",
    "fixed_small_to_live_large_m12",
    "well_refine_status",
    "well_refine_n_pairs",
    "well_refine_n_live_candidates",
    "well_refine_n_fixed_candidates",
    "well_refine_candidate_scope",
    "well_refine_force_relaxed_detection",
    "well_refine_delta_theta_deg",
    "well_refine_delta_shift_x_px",
    "well_refine_delta_shift_y_px",
    "well_refine_mean_residual_px",
    "well_refine_max_residual_px",
    "updated_at",
]

# Tuned to the observed large-well diameter of about 110 px in the raw images.
DEFAULT_LARGE_WELL_RADII_PX = tuple(range(50, 61, 2))
DEFAULT_LARGE_WELL_MIN_CENTER_DISTANCE_PX = 80
LIVE_FIXED_LARGE_FOCUS_Z_OVERRIDE_BY_POSITION = {
    # 2-6 large-large registration is more stable on live large z=0 than on the
    # inherited small-large focus plane.
    "2-6": 0,
}


@dataclass(frozen=True)
class LiveFixedPairRecord:
    canonical_position: str
    live_pair: mwa.PairedImageRecord
    fixed_pair: mwa.PairedImageRecord


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _affine_matrix_to_row_prefix(mat: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_m00": float(mat[0, 0]),
        f"{prefix}_m01": float(mat[0, 1]),
        f"{prefix}_m02": float(mat[0, 2]),
        f"{prefix}_m10": float(mat[1, 0]),
        f"{prefix}_m11": float(mat[1, 1]),
        f"{prefix}_m12": float(mat[1, 2]),
    }


def affine_matrix_from_row_prefix(row: Union[pd.Series, Dict[str, object]], prefix: str) -> np.ndarray:
    return np.array(
        [
            [float(row[f"{prefix}_m00"]), float(row[f"{prefix}_m01"]), float(row[f"{prefix}_m02"])],
            [float(row[f"{prefix}_m10"]), float(row[f"{prefix}_m11"]), float(row[f"{prefix}_m12"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _translation_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, float(tx)],
            [0.0, 1.0, float(ty)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_matrix(theta_deg: float) -> np.ndarray:
    ang = np.deg2rad(float(theta_deg))
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rigid_about_center_matrix(
    shape_yx: Tuple[int, int],
    theta_deg: float = 0.0,
    shift_x_px: float = 0.0,
    shift_y_px: float = 0.0,
) -> np.ndarray:
    h, w = shape_yx
    cx = 0.5 * (float(w) - 1.0)
    cy = 0.5 * (float(h) - 1.0)
    return (
        _translation_matrix(float(shift_x_px), float(shift_y_px))
        @ _translation_matrix(cx, cy)
        @ _rotation_matrix(float(theta_deg))
        @ _translation_matrix(-cx, -cy)
    )


def small_to_large_affine_matrix(
    small_shape_yx: Tuple[int, int],
    top_left_x_px: float,
    top_left_y_px: float,
    theta_deg: float,
) -> np.ndarray:
    return (
        _translation_matrix(float(top_left_x_px), float(top_left_y_px))
        @ rigid_about_center_matrix(
            shape_yx=small_shape_yx,
            theta_deg=float(theta_deg),
            shift_x_px=0.0,
            shift_y_px=0.0,
        )
    )


def inverse_affine_matrix(mat: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(mat, dtype=np.float64))


def apply_affine_to_xy(x_px: float, y_px: float, mat: np.ndarray) -> Tuple[float, float]:
    vec = np.array([float(x_px), float(y_px), 1.0], dtype=np.float64)
    out = np.asarray(mat, dtype=np.float64) @ vec
    return float(out[0]), float(out[1])


def warp_image_with_affine(
    image: np.ndarray,
    forward_mat: np.ndarray,
    output_shape_yx: Tuple[int, int],
    order: int = 1,
    cval: float = np.nan,
) -> np.ndarray:
    tform = sk_transform.AffineTransform(matrix=np.asarray(forward_mat, dtype=np.float64))
    warped = sk_transform.warp(
        image.astype(np.float32),
        inverse_map=tform.inverse,
        output_shape=tuple(int(v) for v in output_shape_yx),
        order=int(order),
        mode="constant",
        cval=float(cval),
        preserve_range=True,
    )
    return warped.astype(np.float32)


def _fit_rigid_transform_2d(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"Expected matching (N,2) arrays; got src={src.shape} dst={dst.shape}")
    n = int(src.shape[0])
    if n < 1:
        raise ValueError("Need at least one point for rigid fit.")
    if n == 1:
        dx = float(dst[0, 0] - src[0, 0])
        dy = float(dst[0, 1] - src[0, 1])
        return _translation_matrix(dx, dy)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    H = src0.T @ dst0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = dst_mean - (R @ src_mean)
    mat = np.array(
        [
            [float(R[0, 0]), float(R[0, 1]), float(t[0])],
            [float(R[1, 0]), float(R[1, 1]), float(t[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return mat


def _rotation_deg_from_matrix(mat: np.ndarray) -> float:
    M = np.asarray(mat, dtype=np.float64)
    return float(np.rad2deg(np.arctan2(M[1, 0], M[0, 0])))


def _load_live_large_bead_xy(
    record: LiveFixedPairRecord,
    root: Union[str, Path],
) -> np.ndarray:
    root = Path(root).resolve()
    large_centroid_path = root / "results/annotations/well_centroids_large.tsv"
    if not large_centroid_path.exists():
        return np.empty((0, 2), dtype=np.float64)

    large_cent_df = mwa.load_large_centroid_table(large_centroid_path)
    large_cent_df = large_cent_df[
        (large_cent_df["large_file_path"].astype(str) == str(record.live_pair.large_file_path))
        & (large_cent_df["annotation_status"].astype(str).str.lower() == "annotated")
    ].copy()
    if large_cent_df.empty:
        return np.empty((0, 2), dtype=np.float64)
    return large_cent_df[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=np.float64)


def _detect_local_well_circle_center(
    image: np.ndarray,
    approx_x_px: float,
    approx_y_px: float,
    half_window_px: int = 110,
    radii_px: Sequence[int] = tuple(range(18, 36, 2)),
    canny_sigma: float = 2.0,
    total_num_peaks: int = 5,
) -> Optional[Dict[str, float]]:
    arr = np.asarray(image, dtype=np.float32)
    H, W = arr.shape
    x = float(approx_x_px)
    y = float(approx_y_px)
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if x < 0.0 or x >= float(W) or y < 0.0 or y >= float(H):
        return None

    xi = int(round(x))
    yi = int(round(y))
    half = int(max(24, half_window_px))
    x0 = max(0, xi - half)
    x1 = min(W, xi + half)
    y0 = max(0, yi - half)
    y1 = min(H, yi + half)
    patch = arr[y0:y1, x0:x1]
    finite = patch[np.isfinite(patch)]
    if finite.size < 64:
        return None

    lo, hi = np.quantile(finite, [0.01, 0.99])
    patch_n = np.clip((patch - float(lo)) / (float(hi - lo) + 1e-6), 0.0, 1.0)
    edges = feature.canny(patch_n, sigma=float(canny_sigma))
    radii = np.asarray(list(radii_px), dtype=int)
    if radii.size == 0:
        return None
    hough = sk_transform.hough_circle(edges, radii)
    accums, cx, cy, rad = sk_transform.hough_circle_peaks(
        hough,
        radii,
        total_num_peaks=int(max(1, total_num_peaks)),
    )
    if len(cx) == 0:
        return None

    patch_center = np.array([x - float(x0), y - float(y0)], dtype=np.float64)
    pts = np.column_stack([cx, cy]).astype(np.float64)
    d2 = np.sum((pts - patch_center[None, :]) ** 2, axis=1)
    best_idx = int(np.argmin(d2))
    return {
        "x_px": float(x0 + float(cx[best_idx])),
        "y_px": float(y0 + float(cy[best_idx])),
        "radius_px": float(rad[best_idx]),
        "accum": float(accums[best_idx]),
        "search_x_px": float(x),
        "search_y_px": float(y),
    }


def detect_large_well_candidates(
    large_bf: np.ndarray,
    small_to_large_row: pd.Series,
    radii_px: Sequence[int] = DEFAULT_LARGE_WELL_RADII_PX,
    canny_sigma: float = 2.0,
    total_num_peaks: int = 96,
    min_center_distance_px: int = DEFAULT_LARGE_WELL_MIN_CENTER_DISTANCE_PX,
    min_accum: float = 0.18,
    inner_radius_frac: float = 0.65,
    inner_std_max: float = 0.07,
    border_pad_px: int = 12,
) -> pd.DataFrame:
    return run_large_well_detection_debug(
        large_bf=large_bf,
        small_to_large_row=small_to_large_row,
        radii_px=radii_px,
        canny_sigma=canny_sigma,
        total_num_peaks=total_num_peaks,
        min_center_distance_px=min_center_distance_px,
        min_accum=min_accum,
        inner_radius_frac=inner_radius_frac,
        inner_std_max=inner_std_max,
        border_pad_px=border_pad_px,
    )["candidate_df"].copy()


def run_large_well_detection_debug(
    large_bf: np.ndarray,
    small_to_large_row: pd.Series,
    radii_px: Sequence[int] = DEFAULT_LARGE_WELL_RADII_PX,
    canny_sigma: float = 2.0,
    total_num_peaks: int = 96,
    min_center_distance_px: int = DEFAULT_LARGE_WELL_MIN_CENTER_DISTANCE_PX,
    min_accum: float = 0.18,
    inner_radius_frac: float = 0.65,
    inner_std_max: float = 0.07,
    border_pad_px: int = 12,
) -> Dict[str, object]:
    img = np.asarray(large_bf, dtype=np.float32)
    finite = img[np.isfinite(img)]
    if finite.size < 256:
        empty = pd.DataFrame(columns=["x_px", "y_px", "radius_px", "accum", "inner_std", "inner_mean", "outer_mean"])
        z = np.zeros_like(img, dtype=np.float32)
        return {
            "image_norm": z,
            "edges": np.zeros_like(img, dtype=bool),
            "hough_max": z,
            "raw_peaks_df": empty.copy(),
            "candidate_df": empty.copy(),
        }
    lo, hi = np.quantile(finite, [0.01, 0.99])
    imgn = np.clip((img - float(lo)) / (float(hi - lo) + 1e-6), 0.0, 1.0)
    edges = feature.canny(imgn, sigma=float(canny_sigma))
    radii = np.asarray(list(radii_px), dtype=int)
    if radii.size == 0:
        empty = pd.DataFrame(columns=["x_px", "y_px", "radius_px", "accum", "inner_std", "inner_mean", "outer_mean"])
        return {
            "image_norm": imgn.astype(np.float32),
            "edges": edges,
            "hough_max": np.zeros_like(imgn, dtype=np.float32),
            "raw_peaks_df": empty.copy(),
            "candidate_df": empty.copy(),
        }
    hough = sk_transform.hough_circle(edges, radii)
    hough_max = np.max(hough, axis=0).astype(np.float32) if hough.size else np.zeros_like(imgn, dtype=np.float32)
    accums, cx, cy, rad = sk_transform.hough_circle_peaks(
        hough,
        radii,
        total_num_peaks=int(max(1, total_num_peaks)),
        min_xdistance=int(max(8, min_center_distance_px)),
        min_ydistance=int(max(8, min_center_distance_px)),
    )
    raw_peak_df = pd.DataFrame(
        {
            "x_px": np.asarray(cx, dtype=np.float64),
            "y_px": np.asarray(cy, dtype=np.float64),
            "radius_px": np.asarray(rad, dtype=np.float64),
            "accum": np.asarray(accums, dtype=np.float64),
        }
    )
    if len(cx) == 0:
        empty = pd.DataFrame(columns=["x_px", "y_px", "radius_px", "accum", "inner_std", "inner_mean", "outer_mean"])
        return {
            "image_norm": imgn.astype(np.float32),
            "edges": edges,
            "hough_max": hough_max,
            "raw_peaks_df": raw_peak_df,
            "candidate_df": empty.copy(),
        }

    poly = np.asarray(mwa.build_small_fov_polygon_in_large(pd.Series(small_to_large_row)), dtype=np.float64)
    poly_path = MplPath(poly)
    H, W = imgn.shape
    Y, X = np.indices(imgn.shape)

    rows: List[Dict[str, float]] = []
    for x, y, r, a in zip(cx, cy, rad, accums):
        x = float(x)
        y = float(y)
        r = float(r)
        a = float(a)
        if a < float(min_accum):
            continue
        if (
            x < (r + float(border_pad_px))
            or x > (float(W) - r - float(border_pad_px))
            or y < (r + float(border_pad_px))
            or y > (float(H) - r - float(border_pad_px))
        ):
            continue
        d = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        inner = imgn[d <= float(inner_radius_frac) * r]
        if inner.size < 32:
            continue
        inner_std = float(np.nanstd(inner))
        if inner_std > float(inner_std_max):
            continue
        rows.append(
            {
                "x_px": x,
                "y_px": y,
                "radius_px": r,
                "accum": a,
                "inner_std": inner_std,
                "inner_mean": float(np.nanmean(inner)),
                "outside_small_fov": bool(not poly_path.contains_point((x, y), radius=40.0)),
            }
        )

    if not rows:
        empty = pd.DataFrame(columns=["x_px", "y_px", "radius_px", "accum", "inner_std", "inner_mean", "outside_small_fov"])
        return {
            "image_norm": imgn.astype(np.float32),
            "edges": edges,
            "hough_max": hough_max,
            "raw_peaks_df": raw_peak_df,
            "candidate_df": empty.copy(),
        }

    cand_df = pd.DataFrame(rows)
    cand_df = cand_df.sort_values(
        by=["outside_small_fov", "accum", "radius_px"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return {
        "image_norm": imgn.astype(np.float32),
        "edges": edges,
        "hough_max": hough_max,
        "raw_peaks_df": raw_peak_df,
        "candidate_df": cand_df,
    }


def refine_fixed_large_to_live_large_with_well_pattern(
    live_large_bf: np.ndarray,
    fixed_large_bf: np.ndarray,
    live_small_to_large_row: pd.Series,
    fixed_small_to_large_row: pd.Series,
    fixed_large_to_live_large: np.ndarray,
    position_key: Optional[str] = None,
    live_bead_xy: Optional[np.ndarray] = None,
    radii_px: Sequence[int] = DEFAULT_LARGE_WELL_RADII_PX,
    canny_sigma: float = 2.0,
    total_num_peaks: int = 96,
    min_center_distance_px: int = DEFAULT_LARGE_WELL_MIN_CENTER_DISTANCE_PX,
    min_accum: float = 0.18,
    inner_std_max: float = 0.07,
    relaxed_min_accum: float = 0.15,
    relaxed_inner_std_max: float = 0.10,
    force_relaxed_detection: bool = False,
    candidate_scope: str = "all",
    max_match_distance_px: float = 140.0,
    max_correction_shift_px: float = 180.0,
    max_correction_abs_theta_deg: float = 8.0,
    n_refine_iters: int = 2,
) -> Dict[str, object]:
    current = np.asarray(fixed_large_to_live_large, dtype=np.float64)
    detect_min_accum = float(relaxed_min_accum) if bool(force_relaxed_detection) else float(min_accum)
    detect_inner_std_max = float(relaxed_inner_std_max) if bool(force_relaxed_detection) else float(inner_std_max)
    live_cand_df = detect_large_well_candidates(
        large_bf=live_large_bf,
        small_to_large_row=live_small_to_large_row,
        radii_px=radii_px,
        canny_sigma=float(canny_sigma),
        total_num_peaks=int(total_num_peaks),
        min_center_distance_px=int(min_center_distance_px),
        min_accum=detect_min_accum,
        inner_std_max=detect_inner_std_max,
    )
    fixed_cand_df = detect_large_well_candidates(
        large_bf=fixed_large_bf,
        small_to_large_row=fixed_small_to_large_row,
        radii_px=radii_px,
        canny_sigma=float(canny_sigma),
        total_num_peaks=int(total_num_peaks),
        min_center_distance_px=int(min_center_distance_px),
        min_accum=detect_min_accum,
        inner_std_max=detect_inner_std_max,
    )
    if len(live_cand_df) < 2 and not bool(force_relaxed_detection):
        live_cand_df = detect_large_well_candidates(
            large_bf=live_large_bf,
            small_to_large_row=live_small_to_large_row,
            radii_px=radii_px,
            canny_sigma=float(canny_sigma),
            total_num_peaks=int(total_num_peaks),
            min_center_distance_px=int(min_center_distance_px),
            min_accum=float(relaxed_min_accum),
            inner_std_max=float(relaxed_inner_std_max),
        )
    if len(fixed_cand_df) < 2 and not bool(force_relaxed_detection):
        fixed_cand_df = detect_large_well_candidates(
            large_bf=fixed_large_bf,
            small_to_large_row=fixed_small_to_large_row,
            radii_px=radii_px,
            canny_sigma=float(canny_sigma),
            total_num_peaks=int(total_num_peaks),
            min_center_distance_px=int(min_center_distance_px),
            min_accum=float(relaxed_min_accum),
            inner_std_max=float(relaxed_inner_std_max),
        )
    candidate_scope_used = str(candidate_scope)
    if str(candidate_scope_used) == "outside_small_fov_only":
        live_out_df = live_cand_df[live_cand_df["outside_small_fov"].astype(bool)].copy().reset_index(drop=True)
        fixed_out_df = fixed_cand_df[fixed_cand_df["outside_small_fov"].astype(bool)].copy().reset_index(drop=True)
        if len(live_out_df) >= 2 and len(fixed_out_df) >= 2:
            live_cand_df = live_out_df
            fixed_cand_df = fixed_out_df
        else:
            candidate_scope_used = "all"
    live_cand_df, fixed_cand_df, position_filter_tag = _apply_position_specific_large_well_candidate_filters(
        position_key=position_key,
        live_cand_df=live_cand_df,
        fixed_cand_df=fixed_cand_df,
        live_bead_xy=live_bead_xy,
    )
    if position_filter_tag:
        candidate_scope_used = f"{candidate_scope_used}+{position_filter_tag}"
    if live_cand_df.empty or fixed_cand_df.empty:
        return {
            "status": "no_large_well_candidates",
            "n_pairs": 0,
            "n_live_candidates": int(len(live_cand_df)),
            "n_fixed_candidates": int(len(fixed_cand_df)),
            "candidate_scope_used": candidate_scope_used,
            "force_relaxed_detection": bool(force_relaxed_detection),
            "matrix": current,
            "delta_matrix": np.eye(3, dtype=np.float64),
            "delta_theta_deg": 0.0,
            "delta_shift_x_px": 0.0,
            "delta_shift_y_px": 0.0,
            "mean_residual_px": np.nan,
            "max_residual_px": np.nan,
            "matches_df": pd.DataFrame(),
            "live_candidates_df": live_cand_df,
            "fixed_candidates_df": fixed_cand_df,
        }

    live_xy = live_cand_df[["x_px", "y_px"]].to_numpy(dtype=np.float64)
    fixed_xy = fixed_cand_df[["x_px", "y_px"]].to_numpy(dtype=np.float64)
    last_matches = pd.DataFrame()

    for _ in range(int(max(1, n_refine_iters))):
        pred_live_xy = np.array(
            [apply_affine_to_xy(float(x), float(y), current) for x, y in fixed_xy],
            dtype=np.float64,
        )
        dists = np.sqrt(((pred_live_xy[:, None, :] - live_xy[None, :, :]) ** 2).sum(axis=2))
        cost = dists.copy()
        cost[cost > float(max_match_distance_px)] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_rows: List[Dict[str, float]] = []
        for ii, jj in zip(row_ind, col_ind):
            dist = float(dists[ii, jj])
            if dist > float(max_match_distance_px):
                continue
            matched_rows.append(
                {
                    "fixed_x_px": float(fixed_xy[ii, 0]),
                    "fixed_y_px": float(fixed_xy[ii, 1]),
                    "pred_live_x_px": float(pred_live_xy[ii, 0]),
                    "pred_live_y_px": float(pred_live_xy[ii, 1]),
                    "live_x_px": float(live_xy[jj, 0]),
                    "live_y_px": float(live_xy[jj, 1]),
                    "match_distance_px": dist,
                }
            )
        last_matches = pd.DataFrame(matched_rows)
        if last_matches.empty:
            return {
                "status": "no_matched_large_wells",
                "n_pairs": 0,
                "n_live_candidates": int(len(live_cand_df)),
                "n_fixed_candidates": int(len(fixed_cand_df)),
                "candidate_scope_used": candidate_scope_used,
                "force_relaxed_detection": bool(force_relaxed_detection),
                "matrix": current,
                "delta_matrix": np.eye(3, dtype=np.float64),
                "delta_theta_deg": 0.0,
                "delta_shift_x_px": 0.0,
                "delta_shift_y_px": 0.0,
                "mean_residual_px": np.nan,
                "max_residual_px": np.nan,
                "matches_df": last_matches,
                "live_candidates_df": live_cand_df,
                "fixed_candidates_df": fixed_cand_df,
            }
        delta_mat = _fit_rigid_transform_2d(
            src_xy=last_matches[["pred_live_x_px", "pred_live_y_px"]].to_numpy(dtype=np.float64),
            dst_xy=last_matches[["live_x_px", "live_y_px"]].to_numpy(dtype=np.float64),
        )
        current = delta_mat @ current

    delta_mat = current @ inverse_affine_matrix(np.asarray(fixed_large_to_live_large, dtype=np.float64))
    delta_theta_deg = float(_rotation_deg_from_matrix(delta_mat))
    delta_shift_x_px = float(delta_mat[0, 2])
    delta_shift_y_px = float(delta_mat[1, 2])
    if (
        abs(delta_theta_deg) > float(max_correction_abs_theta_deg)
        or np.hypot(delta_shift_x_px, delta_shift_y_px) > float(max_correction_shift_px)
    ):
        return {
            "status": "correction_rejected_too_large",
            "n_pairs": int(len(last_matches)),
            "n_live_candidates": int(len(live_cand_df)),
            "n_fixed_candidates": int(len(fixed_cand_df)),
            "candidate_scope_used": candidate_scope_used,
            "force_relaxed_detection": bool(force_relaxed_detection),
            "matrix": np.asarray(fixed_large_to_live_large, dtype=np.float64),
            "delta_matrix": np.eye(3, dtype=np.float64),
            "delta_theta_deg": 0.0,
            "delta_shift_x_px": 0.0,
            "delta_shift_y_px": 0.0,
            "mean_residual_px": np.nan,
            "max_residual_px": np.nan,
            "matches_df": last_matches,
            "live_candidates_df": live_cand_df,
            "fixed_candidates_df": fixed_cand_df,
        }

    fixed_match_xy = last_matches[["fixed_x_px", "fixed_y_px"]].to_numpy(dtype=np.float64)
    refined_live_xy = np.array(
        [apply_affine_to_xy(float(x), float(y), current) for x, y in fixed_match_xy],
        dtype=np.float64,
    )
    obs_live_xy = last_matches[["live_x_px", "live_y_px"]].to_numpy(dtype=np.float64)
    residuals = np.sqrt(np.sum((refined_live_xy - obs_live_xy) ** 2, axis=1))
    return {
        "status": "ok",
        "n_pairs": int(len(last_matches)),
        "n_live_candidates": int(len(live_cand_df)),
        "n_fixed_candidates": int(len(fixed_cand_df)),
        "candidate_scope_used": candidate_scope_used,
        "force_relaxed_detection": bool(force_relaxed_detection),
        "matrix": current,
        "delta_matrix": delta_mat,
        "delta_theta_deg": delta_theta_deg,
        "delta_shift_x_px": delta_shift_x_px,
        "delta_shift_y_px": delta_shift_y_px,
        "mean_residual_px": float(np.mean(residuals)) if residuals.size else np.nan,
        "max_residual_px": float(np.max(residuals)) if residuals.size else np.nan,
        "matches_df": last_matches,
        "live_candidates_df": live_cand_df,
        "fixed_candidates_df": fixed_cand_df,
    }


def refine_fixed_small_to_live_small_with_wells(
    live_small_bf: np.ndarray,
    fixed_small_bf: np.ndarray,
    fixed_small_to_live_small: np.ndarray,
    live_bead_rows: pd.DataFrame,
    detect_half_window_px: int = 110,
    radii_px: Sequence[int] = tuple(range(18, 36, 2)),
    canny_sigma: float = 2.0,
    max_correction_shift_px: float = 180.0,
    max_correction_abs_theta_deg: float = 8.0,
) -> Dict[str, object]:
    if live_bead_rows.empty:
        return {
            "status": "no_inside_live_beads",
            "n_pairs": 0,
            "matrix": np.asarray(fixed_small_to_live_small, dtype=np.float64),
            "delta_matrix": np.eye(3, dtype=np.float64),
            "delta_theta_deg": 0.0,
            "delta_shift_x_px": 0.0,
            "delta_shift_y_px": 0.0,
            "mean_residual_px": np.nan,
            "max_residual_px": np.nan,
            "matches_df": pd.DataFrame(),
        }

    current = np.asarray(fixed_small_to_live_small, dtype=np.float64)
    live_to_fixed = inverse_affine_matrix(current)

    matched_rows: List[Dict[str, object]] = []
    for bead in live_bead_rows.itertuples(index=False):
        lx = float(bead.centroid_small_x_px)
        ly = float(bead.centroid_small_y_px)
        fx_approx, fy_approx = apply_affine_to_xy(lx, ly, live_to_fixed)

        live_det = _detect_local_well_circle_center(
            image=live_small_bf,
            approx_x_px=lx,
            approx_y_px=ly,
            half_window_px=int(detect_half_window_px),
            radii_px=radii_px,
            canny_sigma=float(canny_sigma),
        )
        fixed_det = _detect_local_well_circle_center(
            image=fixed_small_bf,
            approx_x_px=fx_approx,
            approx_y_px=fy_approx,
            half_window_px=int(detect_half_window_px),
            radii_px=radii_px,
            canny_sigma=float(canny_sigma),
        )
        if live_det is None or fixed_det is None:
            continue

        pred_live_x, pred_live_y = apply_affine_to_xy(
            x_px=float(fixed_det["x_px"]),
            y_px=float(fixed_det["y_px"]),
            mat=current,
        )
        matched_rows.append(
            {
                "centroid_index": int(bead.centroid_index),
                "live_click_x_px": lx,
                "live_click_y_px": ly,
                "live_detect_x_px": float(live_det["x_px"]),
                "live_detect_y_px": float(live_det["y_px"]),
                "live_radius_px": float(live_det["radius_px"]),
                "fixed_approx_x_px": float(fx_approx),
                "fixed_approx_y_px": float(fy_approx),
                "fixed_detect_x_px": float(fixed_det["x_px"]),
                "fixed_detect_y_px": float(fixed_det["y_px"]),
                "fixed_radius_px": float(fixed_det["radius_px"]),
                "pred_live_x_px": float(pred_live_x),
                "pred_live_y_px": float(pred_live_y),
            }
        )

    if not matched_rows:
        return {
            "status": "no_detected_well_pairs",
            "n_pairs": 0,
            "matrix": current,
            "delta_matrix": np.eye(3, dtype=np.float64),
            "delta_theta_deg": 0.0,
            "delta_shift_x_px": 0.0,
            "delta_shift_y_px": 0.0,
            "mean_residual_px": np.nan,
            "max_residual_px": np.nan,
            "matches_df": pd.DataFrame(),
        }

    matches_df = pd.DataFrame(matched_rows)
    pred_live_xy = matches_df[["pred_live_x_px", "pred_live_y_px"]].to_numpy(dtype=np.float64)
    obs_live_xy = matches_df[["live_detect_x_px", "live_detect_y_px"]].to_numpy(dtype=np.float64)
    delta_mat = _fit_rigid_transform_2d(pred_live_xy, obs_live_xy)
    delta_theta_deg = float(_rotation_deg_from_matrix(delta_mat))
    delta_shift_x_px = float(delta_mat[0, 2])
    delta_shift_y_px = float(delta_mat[1, 2])
    if (
        abs(delta_theta_deg) > float(max_correction_abs_theta_deg)
        or np.hypot(delta_shift_x_px, delta_shift_y_px) > float(max_correction_shift_px)
    ):
        return {
            "status": "correction_rejected_too_large",
            "n_pairs": int(len(matches_df)),
            "matrix": current,
            "delta_matrix": np.eye(3, dtype=np.float64),
            "delta_theta_deg": 0.0,
            "delta_shift_x_px": 0.0,
            "delta_shift_y_px": 0.0,
            "mean_residual_px": np.nan,
            "max_residual_px": np.nan,
            "matches_df": matches_df,
        }

    refined = delta_mat @ current
    fixed_detect_xy = matches_df[["fixed_detect_x_px", "fixed_detect_y_px"]].to_numpy(dtype=np.float64)
    refined_live_xy = np.array(
        [apply_affine_to_xy(float(x), float(y), refined) for x, y in fixed_detect_xy],
        dtype=np.float64,
    )
    residuals = np.sqrt(np.sum((refined_live_xy - obs_live_xy) ** 2, axis=1))
    return {
        "status": "ok",
        "n_pairs": int(len(matches_df)),
        "matrix": refined,
        "delta_matrix": delta_mat,
        "delta_theta_deg": delta_theta_deg,
        "delta_shift_x_px": delta_shift_x_px,
        "delta_shift_y_px": delta_shift_y_px,
        "mean_residual_px": float(np.mean(residuals)) if residuals.size else np.nan,
        "max_residual_px": float(np.max(residuals)) if residuals.size else np.nan,
        "matches_df": matches_df,
    }


def build_live_fixed_pair_records(
    root: Union[str, Path],
    live_subdir: Union[str, Path] = "data/2026-01-22_PDMS/day2/single",
    fixed_subdir: Union[str, Path] = "data/2026-01-22_PDMS/day2-fix-well2-568SOX2-647T/single",
    live_cohort_id: str = "2026-01-22_day2_live",
    fixed_cohort_id: str = "2026-01-22_day2_fix",
    limit: Optional[int] = None,
) -> List[LiveFixedPairRecord]:
    root = Path(root).resolve()
    live_pairs = mwa.paired_records_from_directory(
        root=root,
        target_subdir=str(live_subdir),
        cohort_id=str(live_cohort_id),
        limit=None,
    )
    fixed_pairs = mwa.paired_records_from_directory(
        root=root,
        target_subdir=str(fixed_subdir),
        cohort_id=str(fixed_cohort_id),
        limit=None,
    )
    fixed_by_pos = {p.canonical_position: p for p in fixed_pairs}
    live_positions = {p.canonical_position for p in live_pairs}
    fixed_positions = set(fixed_by_pos)
    if live_positions != fixed_positions:
        missing_in_fixed = sorted(live_positions - fixed_positions)
        missing_in_live = sorted(fixed_positions - live_positions)
        raise RuntimeError(
            "Live/fixed canonical position mismatch. "
            f"missing_in_fixed={missing_in_fixed[:8]} missing_in_live={missing_in_live[:8]}"
        )

    records = [
        LiveFixedPairRecord(
            canonical_position=p.canonical_position,
            live_pair=p,
            fixed_pair=fixed_by_pos[p.canonical_position],
        )
        for p in live_pairs
    ]
    if limit is not None:
        records = records[: int(limit)]
    return records


def _overlap_corrcoef(reference: np.ndarray, moved: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(moved)
    if int(valid.sum()) < 64:
        return float(-np.inf)
    a = reference[valid].astype(np.float64)
    b = moved[valid].astype(np.float64)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = np.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denom <= 0:
        return float(-np.inf)
    return float(np.sum(a * b) / denom)


def _center_pad_to_shape(
    image: np.ndarray,
    target_shape_yx: Tuple[int, int],
    fill_value: Optional[float] = None,
) -> Tuple[np.ndarray, int, int]:
    h, w = image.shape
    H, W = target_shape_yx
    if H < h or W < w:
        raise ValueError(f"Target shape {target_shape_yx} smaller than image shape {image.shape}")
    if fill_value is None:
        finite = image[np.isfinite(image)]
        fill_value = float(np.median(finite)) if finite.size else 0.0
    out = np.full((H, W), float(fill_value), dtype=np.float32)
    top = int((H - h) // 2)
    left = int((W - w) // 2)
    out[top : top + h, left : left + w] = image.astype(np.float32)
    return out, top, left


def register_large_to_large_rigid(
    reference_bf: np.ndarray,
    moving_bf: np.ndarray,
    allow_rotation: bool = True,
    max_abs_angle_deg: float = 6.0,
    coarse_step_deg: float = 0.5,
    fine_step_deg: float = 0.1,
    upsample_factor: int = 20,
    search_radius_px: int = 240,
) -> Dict[str, float]:
    target_shape = (
        int(max(reference_bf.shape[0], moving_bf.shape[0])),
        int(max(reference_bf.shape[1], moving_bf.shape[1])),
    )
    ref_pad, ref_top, ref_left = _center_pad_to_shape(reference_bf, target_shape)
    mov_pad, mov_top, mov_left = _center_pad_to_shape(moving_bf, target_shape)
    ref_norm = mwa._zscore_norm(ref_pad)
    mov_norm = mwa._zscore_norm(mov_pad)
    if allow_rotation:
        coarse_angles = np.arange(
            -float(max_abs_angle_deg),
            float(max_abs_angle_deg) + 0.5 * float(coarse_step_deg),
            float(coarse_step_deg),
            dtype=np.float32,
        )
    else:
        coarse_angles = np.array([0.0], dtype=np.float32)

    def _evaluate(angle_deg: float) -> Dict[str, float]:
        rotated = mwa._rotate_image_no_resize(mov_norm, float(angle_deg))
        shift_yx, _, _ = phase_cross_correlation(
            reference_image=ref_norm,
            moving_image=rotated,
            upsample_factor=int(max(1, upsample_factor)),
        )
        shift_y = float(shift_yx[0])
        shift_x = float(shift_yx[1])
        if abs(shift_x) > float(search_radius_px) or abs(shift_y) > float(search_radius_px):
            score = float(-np.inf)
        else:
            moved = ndi.shift(
                rotated,
                shift=(shift_y, shift_x),
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            ).astype(np.float32)
            score = _overlap_corrcoef(ref_norm, moved)
        return {
            "theta_deg": float(angle_deg),
            "shift_x_px": float(shift_x),
            "shift_y_px": float(shift_y),
            "score": float(score),
        }

    best = None
    for angle in coarse_angles:
        cand = _evaluate(float(angle))
        if best is None or cand["score"] > best["score"]:
            best = cand

    if best is None:
        raise RuntimeError("Failed rigid registration evaluation.")

    if allow_rotation and float(fine_step_deg) > 0 and float(coarse_step_deg) > float(fine_step_deg):
        fine_angles = np.arange(
            float(best["theta_deg"]) - float(coarse_step_deg),
            float(best["theta_deg"]) + float(coarse_step_deg) + 0.5 * float(fine_step_deg),
            float(fine_step_deg),
            dtype=np.float32,
        )
        for angle in fine_angles:
            cand = _evaluate(float(angle))
            if cand["score"] > best["score"]:
                best = cand

    padded_mat = rigid_about_center_matrix(
        shape_yx=target_shape,
        theta_deg=float(best["theta_deg"]),
        shift_x_px=float(best["shift_x_px"]),
        shift_y_px=float(best["shift_y_px"]),
    )
    full_mat = (
        _translation_matrix(-float(ref_left), -float(ref_top))
        @ padded_mat
        @ _translation_matrix(float(mov_left), float(mov_top))
    )
    return {
        **best,
        "reference_padded_shape_y_px": int(target_shape[0]),
        "reference_padded_shape_x_px": int(target_shape[1]),
        "reference_pad_top_px": int(ref_top),
        "reference_pad_left_px": int(ref_left),
        "moving_pad_top_px": int(mov_top),
        "moving_pad_left_px": int(mov_left),
        "matrix_full": full_mat,
    }


def large_to_large_rigid_matrix_from_params(
    reference_shape_yx: Tuple[int, int],
    moving_shape_yx: Tuple[int, int],
    theta_deg: float,
    shift_x_px: float,
    shift_y_px: float,
) -> np.ndarray:
    ref_h, ref_w = (int(reference_shape_yx[0]), int(reference_shape_yx[1]))
    mov_h, mov_w = (int(moving_shape_yx[0]), int(moving_shape_yx[1]))
    target_shape = (int(max(ref_h, mov_h)), int(max(ref_w, mov_w)))
    ref_top = int((target_shape[0] - ref_h) // 2)
    ref_left = int((target_shape[1] - ref_w) // 2)
    mov_top = int((target_shape[0] - mov_h) // 2)
    mov_left = int((target_shape[1] - mov_w) // 2)
    padded_mat = rigid_about_center_matrix(
        shape_yx=target_shape,
        theta_deg=float(theta_deg),
        shift_x_px=float(shift_x_px),
        shift_y_px=float(shift_y_px),
    )
    return (
        _translation_matrix(-float(ref_left), -float(ref_top))
        @ padded_mat
        @ _translation_matrix(float(mov_left), float(mov_top))
    )


def compute_fixed_to_live_alignment_table(
    records: Sequence[LiveFixedPairRecord],
    root: Union[str, Path],
    live_transform_path: Union[str, Path],
    fixed_transform_path: Union[str, Path],
    out_path: Union[str, Path],
    force_recompute: bool = False,
    allow_rotation: bool = True,
    max_abs_angle_deg: float = 6.0,
    coarse_step_deg: float = 0.5,
    fine_step_deg: float = 0.1,
    upsample_factor: int = 20,
    max_shift_px: int = 180,
    well_detect_radii_px: Sequence[int] = DEFAULT_LARGE_WELL_RADII_PX,
    well_detect_canny_sigma: float = 2.0,
    well_detect_total_num_peaks: int = 96,
    well_detect_min_center_distance_px: int = DEFAULT_LARGE_WELL_MIN_CENTER_DISTANCE_PX,
    well_detect_min_accum: float = 0.18,
    well_detect_inner_std_max: float = 0.07,
    well_refine_outside_small_fov_only_positions: Sequence[str] = (),
    well_refine_force_relaxed_detection_positions: Sequence[str] = (),
    well_refine_skip_positions: Sequence[str] = (),
    well_match_max_distance_px: float = 140.0,
    well_refine_max_shift_px: float = 180.0,
    well_refine_max_abs_theta_deg: float = 8.0,
    verbose: bool = True,
) -> pd.DataFrame:
    root = Path(root).resolve()
    out_path = Path(out_path)
    skip_refine_positions = {str(pos) for pos in well_refine_skip_positions}
    if out_path.exists() and not bool(force_recompute):
        existing_df = pd.read_csv(out_path, sep="\t")
        needed = {str(r.canonical_position) for r in records}
        have = set(existing_df["canonical_position"].astype(str))
        has_refine_cols = {
            "well_refine_status",
            "well_refine_n_pairs",
            "well_refine_n_live_candidates",
            "well_refine_n_fixed_candidates",
            "well_refine_candidate_scope",
            "well_refine_force_relaxed_detection",
        }.issubset(existing_df.columns)
        has_requested_skip_behavior = True
        if skip_refine_positions and "well_refine_status" in existing_df.columns:
            skip_rows = existing_df[existing_df["canonical_position"].astype(str).isin(skip_refine_positions)].copy()
            if len(skip_rows) < len(skip_refine_positions):
                has_requested_skip_behavior = False
            elif not skip_rows["well_refine_status"].astype(str).eq("skipped_by_position").all():
                has_requested_skip_behavior = False
        if needed.issubset(have) and has_refine_cols:
            if skip_refine_positions and not has_requested_skip_behavior:
                if verbose:
                    print(
                        f"Existing live-fixed alignment table at {out_path} does not reflect "
                        f"requested skip-refine positions {sorted(skip_refine_positions)}; recomputing."
                    )
            else:
                out_df = (
                    existing_df[existing_df["canonical_position"].astype(str).isin(needed)]
                    .copy()
                    .reset_index(drop=True)
                )
                if verbose:
                    print(
                        f"Loading existing live-fixed alignment table from {out_path} "
                        f"for {len(out_df)} position(s). force_recompute={bool(force_recompute)}"
                    )
                return out_df
        if verbose and needed.issubset(have) and not has_refine_cols:
            print(
                f"Existing live-fixed alignment table at {out_path} lacks well-refinement columns; "
                "recomputing."
            )
    live_tr_df = mwa.load_pair_transform_table(Path(live_transform_path))
    fixed_tr_df = mwa.load_pair_transform_table(Path(fixed_transform_path))
    live_by_large = {str(r.large_file_path): pd.Series(r._asdict()) for r in live_tr_df.itertuples(index=False)}
    fixed_by_large = {str(r.large_file_path): pd.Series(r._asdict()) for r in fixed_tr_df.itertuples(index=False)}
    outside_only_positions = {str(pos) for pos in well_refine_outside_small_fov_only_positions}
    force_relaxed_positions = {str(pos) for pos in well_refine_force_relaxed_detection_positions}

    rows: List[Dict[str, object]] = []
    total = len(records)
    t0 = time.perf_counter()
    for idx, rec in enumerate(records, start=1):
        label = f"[{idx}/{total}] {rec.canonical_position}"
        pair_t0 = time.perf_counter()
        live_tr = live_by_large.get(str(rec.live_pair.large_file_path))
        fixed_tr = fixed_by_large.get(str(rec.fixed_pair.large_file_path))
        base_row = {
            "canonical_position": rec.canonical_position,
            "live_image_id": rec.live_pair.image_id,
            "fixed_image_id": rec.fixed_pair.image_id,
            "live_small_file_path": rec.live_pair.small_file_path,
            "live_large_file_path": rec.live_pair.large_file_path,
            "fixed_small_file_path": rec.fixed_pair.small_file_path,
            "fixed_large_file_path": rec.fixed_pair.large_file_path,
        }

        if live_tr is None or str(live_tr.get("registration_status", "")) != "ok":
            row = {
                **base_row,
                "status": "missing_live_small_large_transform",
                "well_refine_status": "skipped",
                "well_refine_n_pairs": 0,
                "well_refine_n_live_candidates": 0,
                "well_refine_n_fixed_candidates": 0,
                "well_refine_candidate_scope": "skipped",
                "well_refine_force_relaxed_detection": False,
                "well_refine_delta_theta_deg": 0.0,
                "well_refine_delta_shift_x_px": 0.0,
                "well_refine_delta_shift_y_px": 0.0,
                "well_refine_mean_residual_px": np.nan,
                "well_refine_max_residual_px": np.nan,
                "updated_at": _iso_now(),
            }
            row.update(_affine_matrix_to_row_prefix(np.eye(3), "fixed_small_to_live_small"))
            row.update(_affine_matrix_to_row_prefix(np.eye(3), "fixed_small_to_live_large"))
            rows.append(row)
            if verbose:
                elapsed = time.perf_counter() - t0
                avg = elapsed / float(idx)
                eta = max(0.0, avg * float(total - idx))
                print(f"{label} missing live transform | elapsed={elapsed:.1f}s eta={eta:.1f}s")
            continue

        if fixed_tr is None or str(fixed_tr.get("registration_status", "")) != "ok":
            row = {
                **base_row,
                "status": "missing_fixed_small_large_transform",
                "well_refine_status": "skipped",
                "well_refine_n_pairs": 0,
                "well_refine_n_live_candidates": 0,
                "well_refine_n_fixed_candidates": 0,
                "well_refine_candidate_scope": "skipped",
                "well_refine_force_relaxed_detection": False,
                "well_refine_delta_theta_deg": 0.0,
                "well_refine_delta_shift_x_px": 0.0,
                "well_refine_delta_shift_y_px": 0.0,
                "well_refine_mean_residual_px": np.nan,
                "well_refine_max_residual_px": np.nan,
                "updated_at": _iso_now(),
            }
            row.update(_affine_matrix_to_row_prefix(np.eye(3), "fixed_small_to_live_small"))
            row.update(_affine_matrix_to_row_prefix(np.eye(3), "fixed_small_to_live_large"))
            rows.append(row)
            if verbose:
                elapsed = time.perf_counter() - t0
                avg = elapsed / float(idx)
                eta = max(0.0, avg * float(total - idx))
                print(f"{label} missing fixed transform | elapsed={elapsed:.1f}s eta={eta:.1f}s")
            continue

        try:
            live_large = mwa.load_large_registered_stack(pair=rec.live_pair, root=root)
            fixed_large = mwa.load_large_registered_stack(pair=rec.fixed_pair, root=root)
            live_bead_xy = _load_live_large_bead_xy(record=rec, root=root)
            live_focus_z = int(round(float(live_tr.get("large_focus_z_index", live_tr.get("small_to_large_best_z_index", 0)))))
            fixed_focus_z = int(round(float(fixed_tr.get("large_focus_z_index", fixed_tr.get("small_to_large_best_z_index", 0)))))
            if str(rec.canonical_position) in LIVE_FIXED_LARGE_FOCUS_Z_OVERRIDE_BY_POSITION:
                live_focus_z = int(LIVE_FIXED_LARGE_FOCUS_Z_OVERRIDE_BY_POSITION[str(rec.canonical_position)])
            live_focus_z = int(np.clip(live_focus_z, 0, live_large.bf_stack.shape[0] - 1))
            fixed_focus_z = int(np.clip(fixed_focus_z, 0, fixed_large.bf_stack.shape[0] - 1))
            reg = register_large_to_large_rigid(
                reference_bf=live_large.bf_stack[live_focus_z],
                moving_bf=fixed_large.bf_stack[fixed_focus_z],
                allow_rotation=bool(allow_rotation),
                max_abs_angle_deg=float(max_abs_angle_deg),
                coarse_step_deg=float(coarse_step_deg),
                fine_step_deg=float(fine_step_deg),
                upsample_factor=int(upsample_factor),
                search_radius_px=int(max_shift_px),
            )

            live_small_to_live_large = small_to_large_affine_matrix(
                small_shape_yx=(int(live_tr["small_shape_y_px"]), int(live_tr["small_shape_x_px"])),
                top_left_x_px=float(live_tr["small_to_large_top_left_x_px"]),
                top_left_y_px=float(live_tr["small_to_large_top_left_y_px"]),
                theta_deg=float(live_tr["small_to_large_theta_deg"]),
            )
            fixed_small_to_fixed_large = small_to_large_affine_matrix(
                small_shape_yx=(int(fixed_tr["small_shape_y_px"]), int(fixed_tr["small_shape_x_px"])),
                top_left_x_px=float(fixed_tr["small_to_large_top_left_x_px"]),
                top_left_y_px=float(fixed_tr["small_to_large_top_left_y_px"]),
                theta_deg=float(fixed_tr["small_to_large_theta_deg"]),
            )
            fixed_large_to_live_large = np.asarray(reg["matrix_full"], dtype=np.float64)
            candidate_scope = (
                "outside_small_fov_only"
                if str(rec.canonical_position) in outside_only_positions
                else "all"
            )
            force_relaxed_detection = bool(str(rec.canonical_position) in force_relaxed_positions)
            if str(rec.canonical_position) in skip_refine_positions:
                well_refine = {
                    "status": "skipped_by_position",
                    "n_pairs": 0,
                    "n_live_candidates": 0,
                    "n_fixed_candidates": 0,
                    "candidate_scope_used": "skipped_by_position",
                    "force_relaxed_detection": False,
                    "matrix": fixed_large_to_live_large,
                    "delta_matrix": np.eye(3, dtype=np.float64),
                    "delta_theta_deg": 0.0,
                    "delta_shift_x_px": 0.0,
                    "delta_shift_y_px": 0.0,
                    "mean_residual_px": np.nan,
                    "max_residual_px": np.nan,
                    "matches_df": pd.DataFrame(),
                    "live_candidates_df": pd.DataFrame(),
                    "fixed_candidates_df": pd.DataFrame(),
                }
            else:
                well_refine = refine_fixed_large_to_live_large_with_well_pattern(
                    live_large_bf=live_large.bf_stack[live_focus_z],
                    fixed_large_bf=fixed_large.bf_stack[fixed_focus_z],
                    live_small_to_large_row=live_tr,
                    fixed_small_to_large_row=fixed_tr,
                    fixed_large_to_live_large=fixed_large_to_live_large,
                    position_key=str(rec.canonical_position),
                    live_bead_xy=live_bead_xy,
                    radii_px=well_detect_radii_px,
                    canny_sigma=float(well_detect_canny_sigma),
                    total_num_peaks=int(well_detect_total_num_peaks),
                    min_center_distance_px=int(well_detect_min_center_distance_px),
                    min_accum=float(well_detect_min_accum),
                    inner_std_max=float(well_detect_inner_std_max),
                    force_relaxed_detection=force_relaxed_detection,
                    candidate_scope=str(candidate_scope),
                    max_match_distance_px=float(well_match_max_distance_px),
                    max_correction_shift_px=float(well_refine_max_shift_px),
                    max_correction_abs_theta_deg=float(well_refine_max_abs_theta_deg),
                )
            fixed_large_to_live_large = np.asarray(well_refine["matrix"], dtype=np.float64)
            fixed_small_to_live_large = fixed_large_to_live_large @ fixed_small_to_fixed_large
            live_large_to_live_small = inverse_affine_matrix(live_small_to_live_large)
            fixed_small_to_live_small = live_large_to_live_small @ fixed_small_to_live_large

            row = {
                **base_row,
                "status": "ok",
                "fixed_large_to_live_large_theta_deg": float(reg["theta_deg"]),
                "fixed_large_to_live_large_shift_x_px": float(reg["shift_x_px"]),
                "fixed_large_to_live_large_shift_y_px": float(reg["shift_y_px"]),
                "fixed_large_to_live_large_score": float(reg["score"]),
                "live_large_focus_z_index": int(live_focus_z),
                "fixed_large_focus_z_index": int(fixed_focus_z),
                "well_refine_status": str(well_refine["status"]),
                "well_refine_n_pairs": int(well_refine["n_pairs"]),
                "well_refine_n_live_candidates": int(well_refine["n_live_candidates"]),
                "well_refine_n_fixed_candidates": int(well_refine["n_fixed_candidates"]),
                "well_refine_candidate_scope": str(well_refine.get("candidate_scope_used", candidate_scope)),
                "well_refine_force_relaxed_detection": bool(well_refine.get("force_relaxed_detection", force_relaxed_detection)),
                "well_refine_delta_theta_deg": float(well_refine["delta_theta_deg"]),
                "well_refine_delta_shift_x_px": float(well_refine["delta_shift_x_px"]),
                "well_refine_delta_shift_y_px": float(well_refine["delta_shift_y_px"]),
                "well_refine_mean_residual_px": float(well_refine["mean_residual_px"]) if pd.notna(well_refine["mean_residual_px"]) else np.nan,
                "well_refine_max_residual_px": float(well_refine["max_residual_px"]) if pd.notna(well_refine["max_residual_px"]) else np.nan,
                "updated_at": _iso_now(),
            }
            row.update(_affine_matrix_to_row_prefix(fixed_small_to_live_small, "fixed_small_to_live_small"))
            row.update(_affine_matrix_to_row_prefix(fixed_small_to_live_large, "fixed_small_to_live_large"))
            rows.append(row)
            if verbose:
                elapsed = time.perf_counter() - t0
                avg = elapsed / float(idx)
                eta = max(0.0, avg * float(total - idx))
                pair_dt = time.perf_counter() - pair_t0
                print(
                    f"{label} ok score={float(reg['score']):.4f} "
                    f"theta={float(reg['theta_deg']):.2f}deg "
                    f"shift=({float(reg['shift_x_px']):.1f}, {float(reg['shift_y_px']):.1f}) "
                    f"| well_refine={str(well_refine['status'])} n={int(well_refine['n_pairs'])} "
                    f"cand=({int(well_refine['n_live_candidates'])},{int(well_refine['n_fixed_candidates'])}) "
                    f"dtheta={float(well_refine['delta_theta_deg']):.2f} "
                    f"dshift=({float(well_refine['delta_shift_x_px']):.1f}, {float(well_refine['delta_shift_y_px']):.1f}) "
                    f"| pair={pair_dt:.1f}s elapsed={elapsed:.1f}s eta={eta:.1f}s"
                )
        except Exception as exc:
            row = {
                **base_row,
                "status": f"error:{exc}",
                "well_refine_status": "error",
                "well_refine_n_pairs": 0,
                "well_refine_n_live_candidates": 0,
                "well_refine_n_fixed_candidates": 0,
                "well_refine_candidate_scope": "error",
                "well_refine_force_relaxed_detection": False,
                "well_refine_delta_theta_deg": 0.0,
                "well_refine_delta_shift_x_px": 0.0,
                "well_refine_delta_shift_y_px": 0.0,
                "well_refine_mean_residual_px": np.nan,
                "well_refine_max_residual_px": np.nan,
                "updated_at": _iso_now(),
            }
            row.update(_affine_matrix_to_row_prefix(np.eye(3), "fixed_small_to_live_small"))
            row.update(_affine_matrix_to_row_prefix(np.eye(3), "fixed_small_to_live_large"))
            rows.append(row)
            if verbose:
                elapsed = time.perf_counter() - t0
                avg = elapsed / float(idx)
                eta = max(0.0, avg * float(total - idx))
                print(f"{label} error: {exc} | elapsed={elapsed:.1f}s eta={eta:.1f}s")

    out_df = pd.DataFrame(rows)
    for col in LIVE_FIXED_ALIGNMENT_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = np.nan
    out_df = out_df[LIVE_FIXED_ALIGNMENT_COLUMNS].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    if verbose:
        print(f"Wrote {out_path}")
    return out_df


def _classify_candidates_by_manual_beads(
    cand_df: pd.DataFrame,
    anchor_xy: np.ndarray,
    max_match_distance_px: float = 80.0,
) -> pd.DataFrame:
    out = cand_df.copy()
    out["bead_annotated"] = False
    out["bead_anchor_index"] = -1
    out["bead_anchor_distance_px"] = np.nan
    if out.empty:
        return out

    anchors = np.asarray(anchor_xy, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[0] == 0 or anchors.shape[1] != 2:
        return out

    cand_xy = out[["x_px", "y_px"]].to_numpy(dtype=np.float64)
    dists = np.sqrt(((anchors[:, None, :] - cand_xy[None, :, :]) ** 2).sum(axis=2))
    cost = dists.copy()
    cost[cost > float(max_match_distance_px)] = 1e6
    row_ind, col_ind = linear_sum_assignment(cost)
    for aa, cc in zip(row_ind, col_ind):
        dist = float(dists[aa, cc])
        if dist > float(max_match_distance_px):
            continue
        out.at[int(cc), "bead_annotated"] = True
        out.at[int(cc), "bead_anchor_index"] = int(aa)
        out.at[int(cc), "bead_anchor_distance_px"] = dist
    return out


def _apply_position_specific_large_well_candidate_filters(
    position_key: Optional[str],
    live_cand_df: pd.DataFrame,
    fixed_cand_df: pd.DataFrame,
    live_bead_xy: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    pos = str(position_key or "")
    if pos != "2-6":
        return live_cand_df.copy(), fixed_cand_df.copy(), None

    live_out = live_cand_df.copy()
    fixed_out = fixed_cand_df.copy()
    if live_out.empty or fixed_out.empty:
        return live_out, fixed_out, None

    # For 2-6, keep the three bead-linked live wells plus the two trusted
    # live de novo wells (top and bottom-right-most), and on the fixed side
    # keep the trusted de novo set (topmost and bottom three).
    live_out = _classify_candidates_by_manual_beads(
        live_out,
        np.asarray(live_bead_xy if live_bead_xy is not None else np.empty((0, 2)), dtype=np.float64),
        max_match_distance_px=80.0,
    )
    live_keep_idx = set(live_out.index[live_out["bead_annotated"].astype(bool)])
    live_denovo = live_out[~live_out["bead_annotated"].astype(bool)].copy()
    if not live_denovo.empty:
        live_top_idx = int(
            live_denovo.sort_values(["y_px", "x_px"], ascending=[True, True]).index[0]
        )
        live_br_idx = int(
            live_denovo.assign(_diag_score=live_denovo["x_px"] + live_denovo["y_px"])
            .sort_values(["_diag_score", "x_px", "y_px"], ascending=[False, False, False])
            .index[0]
        )
        live_keep_idx.update([live_top_idx, live_br_idx])
    live_out = live_out.loc[sorted(live_keep_idx)].copy().reset_index(drop=True)

    fixed_sorted = fixed_out.sort_values(["y_px", "x_px"], ascending=[True, True])
    fixed_keep_idx = {int(fixed_sorted.index[0])}
    fixed_keep_idx.update(int(ii) for ii in fixed_sorted.tail(3).index.tolist())
    fixed_out = fixed_out.loc[sorted(fixed_keep_idx)].copy().reset_index(drop=True)

    if len(live_out) < 2 or len(fixed_out) < 2:
        return live_cand_df.copy(), fixed_cand_df.copy(), None
    return live_out, fixed_out, "trusted_candidates_2-6"


def _draw_candidate_circles(
    ax: plt.Axes,
    cand_df: pd.DataFrame,
    bead_color: str,
    denovo_color: str,
    center_marker: str = ".",
    center_size: float = 8.0,
    linewidth: float = 1.1,
    alpha: float = 0.95,
) -> None:
    if cand_df is None or cand_df.empty:
        return
    for rr in cand_df.itertuples(index=False):
        color = bead_color if bool(getattr(rr, "bead_annotated", False)) else denovo_color
        circ = plt.Circle(
            (float(rr.x_px), float(rr.y_px)),
            float(rr.radius_px),
            edgecolor=color,
            facecolor="none",
            linewidth=float(linewidth),
            alpha=float(alpha),
            linestyle="-",
        )
        ax.add_patch(circ)
        ax.plot(
            float(rr.x_px),
            float(rr.y_px),
            marker=center_marker,
            markersize=float(center_size),
            color=color,
            alpha=float(alpha),
        )


def render_live_fixed_large_well_detection_qc(
    record: LiveFixedPairRecord,
    align_row: pd.Series,
    live_small_to_large_row: pd.Series,
    fixed_small_to_large_row: pd.Series,
    root: Union[str, Path],
    well_detect_radii_px: Sequence[int] = DEFAULT_LARGE_WELL_RADII_PX,
    well_detect_canny_sigma: float = 2.0,
    well_detect_total_num_peaks: int = 96,
    well_detect_min_center_distance_px: int = DEFAULT_LARGE_WELL_MIN_CENTER_DISTANCE_PX,
    well_detect_min_accum: float = 0.18,
    well_detect_inner_std_max: float = 0.07,
    well_match_max_distance_px: float = 140.0,
    well_refine_max_shift_px: float = 180.0,
    well_refine_max_abs_theta_deg: float = 8.0,
) -> plt.Figure:
    if str(align_row.get("status", "")) != "ok":
        raise RuntimeError(
            f"Alignment row not ok for {record.canonical_position}: {align_row.get('status')}"
        )

    root = Path(root).resolve()
    live_large = mwa.load_large_registered_stack(pair=record.live_pair, root=root)
    fixed_large = mwa.load_large_registered_stack(pair=record.fixed_pair, root=root)

    z_live = int(
        np.clip(
            int(round(float(align_row["live_large_focus_z_index"]))),
            0,
            live_large.bf_stack.shape[0] - 1,
        )
    )
    z_fixed = int(
        np.clip(
            int(round(float(align_row["fixed_large_focus_z_index"]))),
            0,
            fixed_large.bf_stack.shape[0] - 1,
        )
    )
    live_large_bf = live_large.bf_stack[z_live]
    fixed_large_bf = fixed_large.bf_stack[z_fixed]

    coarse_matrix = large_to_large_rigid_matrix_from_params(
        reference_shape_yx=live_large_bf.shape,
        moving_shape_yx=fixed_large_bf.shape,
        theta_deg=float(align_row["fixed_large_to_live_large_theta_deg"]),
        shift_x_px=float(align_row["fixed_large_to_live_large_shift_x_px"]),
        shift_y_px=float(align_row["fixed_large_to_live_large_shift_y_px"]),
    )
    candidate_scope = str(align_row.get("well_refine_candidate_scope", "all") or "all")
    force_relaxed_detection = bool(align_row.get("well_refine_force_relaxed_detection", False))
    live_bead_xy = _load_live_large_bead_xy(record=record, root=root)
    if str(align_row.get("well_refine_status", "")) == "skipped_by_position":
        live_cand_df = detect_large_well_candidates(
            large_bf=live_large_bf,
            small_to_large_row=live_small_to_large_row,
            radii_px=well_detect_radii_px,
            canny_sigma=float(well_detect_canny_sigma),
            total_num_peaks=int(well_detect_total_num_peaks),
            min_center_distance_px=int(well_detect_min_center_distance_px),
            min_accum=float(well_detect_min_accum),
            inner_std_max=float(well_detect_inner_std_max),
        )
        fixed_cand_df = detect_large_well_candidates(
            large_bf=fixed_large_bf,
            small_to_large_row=fixed_small_to_large_row,
            radii_px=well_detect_radii_px,
            canny_sigma=float(well_detect_canny_sigma),
            total_num_peaks=int(well_detect_total_num_peaks),
            min_center_distance_px=int(well_detect_min_center_distance_px),
            min_accum=float(well_detect_min_accum),
            inner_std_max=float(well_detect_inner_std_max),
        )
        refine = {
            "status": "skipped_by_position",
            "n_pairs": 0,
            "n_live_candidates": int(len(live_cand_df)),
            "n_fixed_candidates": int(len(fixed_cand_df)),
            "candidate_scope_used": "skipped_by_position",
            "force_relaxed_detection": False,
            "matrix": coarse_matrix,
            "delta_matrix": np.eye(3, dtype=np.float64),
            "delta_theta_deg": 0.0,
            "delta_shift_x_px": 0.0,
            "delta_shift_y_px": 0.0,
            "mean_residual_px": np.nan,
            "max_residual_px": np.nan,
            "matches_df": pd.DataFrame(),
            "live_candidates_df": live_cand_df,
            "fixed_candidates_df": fixed_cand_df,
        }
    else:
        refine = refine_fixed_large_to_live_large_with_well_pattern(
            live_large_bf=live_large_bf,
            fixed_large_bf=fixed_large_bf,
            live_small_to_large_row=live_small_to_large_row,
            fixed_small_to_large_row=fixed_small_to_large_row,
            fixed_large_to_live_large=coarse_matrix,
            position_key=str(record.canonical_position),
            live_bead_xy=live_bead_xy,
            radii_px=well_detect_radii_px,
            canny_sigma=float(well_detect_canny_sigma),
            total_num_peaks=int(well_detect_total_num_peaks),
            min_center_distance_px=int(well_detect_min_center_distance_px),
            min_accum=float(well_detect_min_accum),
            inner_std_max=float(well_detect_inner_std_max),
            force_relaxed_detection=force_relaxed_detection,
            candidate_scope=candidate_scope,
            max_match_distance_px=float(well_match_max_distance_px),
            max_correction_shift_px=float(well_refine_max_shift_px),
            max_correction_abs_theta_deg=float(well_refine_max_abs_theta_deg),
        )

    live_cand_df = refine["live_candidates_df"].copy()
    fixed_cand_df = refine["fixed_candidates_df"].copy()
    matches_df = refine["matches_df"].copy()
    refined_matrix = np.asarray(refine["matrix"], dtype=np.float64)
    refined_inv = inverse_affine_matrix(refined_matrix)

    fixed_bead_xy = np.empty((0, 2), dtype=np.float64)
    if live_bead_xy.size:
        fixed_bead_xy = np.array(
            [apply_affine_to_xy(float(x), float(y), refined_inv) for x, y in live_bead_xy],
            dtype=np.float64,
        )

    live_poly = np.asarray(
        mwa.build_small_fov_polygon_in_large(pd.Series(live_small_to_large_row)),
        dtype=np.float64,
    )
    fixed_poly = np.asarray(
        mwa.build_small_fov_polygon_in_large(pd.Series(fixed_small_to_large_row)),
        dtype=np.float64,
    )

    coarse_poly = np.array(
        [apply_affine_to_xy(float(x), float(y), coarse_matrix) for x, y in fixed_poly],
        dtype=np.float64,
    )
    refined_poly = np.array(
        [apply_affine_to_xy(float(x), float(y), refined_matrix) for x, y in fixed_poly],
        dtype=np.float64,
    )

    live_cand_df = _classify_candidates_by_manual_beads(
        live_cand_df,
        anchor_xy=live_bead_xy,
        max_match_distance_px=80.0,
    )
    fixed_cand_df = _classify_candidates_by_manual_beads(
        fixed_cand_df,
        anchor_xy=fixed_bead_xy,
        max_match_distance_px=80.0,
    )

    if not fixed_cand_df.empty:
        fixed_xy = fixed_cand_df[["x_px", "y_px"]].to_numpy(dtype=np.float64)
        coarse_xy = np.array(
            [apply_affine_to_xy(float(x), float(y), coarse_matrix) for x, y in fixed_xy],
            dtype=np.float64,
        )
        refined_xy = np.array(
            [apply_affine_to_xy(float(x), float(y), refined_matrix) for x, y in fixed_xy],
            dtype=np.float64,
        )
        fixed_cand_df["coarse_live_x_px"] = coarse_xy[:, 0]
        fixed_cand_df["coarse_live_y_px"] = coarse_xy[:, 1]
        fixed_cand_df["refined_live_x_px"] = refined_xy[:, 0]
        fixed_cand_df["refined_live_y_px"] = refined_xy[:, 1]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), constrained_layout=True)

    axes[0].imshow(mwa.robust_rescale(live_large_bf), cmap="gray")
    axes[0].plot(
        np.r_[live_poly[:, 0], live_poly[0, 0]],
        np.r_[live_poly[:, 1], live_poly[0, 1]],
        color="cyan",
        linestyle="--",
        linewidth=1.2,
    )
    _draw_candidate_circles(
        axes[0],
        live_cand_df,
        bead_color="deepskyblue",
        denovo_color="gold",
    )
    if live_bead_xy.size:
        axes[0].scatter(live_bead_xy[:, 0], live_bead_xy[:, 1], s=24, c="deepskyblue", marker="+", linewidths=1.0, alpha=0.9)
    axes[0].set_title(
        f"Live large BF\nbead-linked={int(live_cand_df['bead_annotated'].sum())} de novo={int((~live_cand_df['bead_annotated']).sum())}"
    )

    axes[1].imshow(mwa.robust_rescale(fixed_large_bf), cmap="gray")
    axes[1].plot(
        np.r_[fixed_poly[:, 0], fixed_poly[0, 0]],
        np.r_[fixed_poly[:, 1], fixed_poly[0, 1]],
        color="deepskyblue",
        linestyle="--",
        linewidth=1.2,
    )
    _draw_candidate_circles(
        axes[1],
        fixed_cand_df,
        bead_color="deepskyblue",
        denovo_color="gold",
    )
    if fixed_bead_xy.size:
        axes[1].scatter(fixed_bead_xy[:, 0], fixed_bead_xy[:, 1], s=24, c="deepskyblue", marker="+", linewidths=1.0, alpha=0.9)
    axes[1].set_title(
        f"Fixed large BF\nbead-linked={int(fixed_cand_df['bead_annotated'].sum())} de novo={int((~fixed_cand_df['bead_annotated']).sum())}"
    )

    axes[2].imshow(mwa.robust_rescale(live_large_bf), cmap="gray")
    axes[2].plot(
        np.r_[live_poly[:, 0], live_poly[0, 0]],
        np.r_[live_poly[:, 1], live_poly[0, 1]],
        color="cyan",
        linestyle="--",
        linewidth=1.0,
    )
    axes[2].plot(
        np.r_[coarse_poly[:, 0], coarse_poly[0, 0]],
        np.r_[coarse_poly[:, 1], coarse_poly[0, 1]],
        color="magenta",
        linestyle="-",
        linewidth=1.0,
    )
    _draw_candidate_circles(
        axes[2],
        live_cand_df,
        bead_color="deepskyblue",
        denovo_color="gold",
        center_size=6.0,
        linewidth=0.9,
        alpha=0.85,
    )
    if not fixed_cand_df.empty:
        fixed_colors = [
            "deepskyblue" if bool(v) else "gold"
            for v in fixed_cand_df["bead_annotated"].tolist()
        ]
        axes[2].scatter(
            fixed_cand_df["coarse_live_x_px"],
            fixed_cand_df["coarse_live_y_px"],
            s=18,
            c=fixed_colors,
            marker="x",
            alpha=0.9,
        )
    if not matches_df.empty:
        for rr in matches_df.itertuples(index=False):
            axes[2].plot(
                [float(rr.pred_live_x_px), float(rr.live_x_px)],
                [float(rr.pred_live_y_px), float(rr.live_y_px)],
                color="white",
                linewidth=0.7,
                alpha=0.9,
            )
    axes[2].set_title(
        f"Coarse mapping\nscore={float(align_row['fixed_large_to_live_large_score']):.4f}"
    )

    axes[3].imshow(mwa.robust_rescale(live_large_bf), cmap="gray")
    axes[3].plot(
        np.r_[live_poly[:, 0], live_poly[0, 0]],
        np.r_[live_poly[:, 1], live_poly[0, 1]],
        color="cyan",
        linestyle="--",
        linewidth=1.0,
    )
    axes[3].plot(
        np.r_[refined_poly[:, 0], refined_poly[0, 0]],
        np.r_[refined_poly[:, 1], refined_poly[0, 1]],
        color="magenta",
        linestyle="-",
        linewidth=1.0,
    )
    _draw_candidate_circles(
        axes[3],
        live_cand_df,
        bead_color="deepskyblue",
        denovo_color="gold",
        center_size=6.0,
        linewidth=0.9,
        alpha=0.85,
    )
    if not fixed_cand_df.empty:
        fixed_colors = [
            "deepskyblue" if bool(v) else "gold"
            for v in fixed_cand_df["bead_annotated"].tolist()
        ]
        axes[3].scatter(
            fixed_cand_df["refined_live_x_px"],
            fixed_cand_df["refined_live_y_px"],
            s=18,
            c=fixed_colors,
            marker="x",
            alpha=0.9,
        )
    if not matches_df.empty:
        for rr in matches_df.itertuples(index=False):
            refined_x, refined_y = apply_affine_to_xy(
                float(rr.fixed_x_px),
                float(rr.fixed_y_px),
                refined_matrix,
            )
            axes[3].plot(
                [float(refined_x), float(rr.live_x_px)],
                [float(refined_y), float(rr.live_y_px)],
                color="white",
                linewidth=0.8,
                alpha=0.9,
            )
            axes[3].plot(float(rr.live_x_px), float(rr.live_y_px), marker="o", markersize=3.0, color="cyan")
    axes[3].set_title(
        f"Refined mapping\nstatus={refine['status']} matches={int(refine['n_pairs'])}"
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    fig.suptitle(
        f"{record.canonical_position} | live/fixed large well-detection QC | "
        f"cand=({int(refine['n_live_candidates'])},{int(refine['n_fixed_candidates'])}) "
        f"| dtheta={float(refine['delta_theta_deg']):.2f} "
        f"| dshift=({float(refine['delta_shift_x_px']):.1f}, {float(refine['delta_shift_y_px']):.1f}) "
        f"| residual={float(refine['mean_residual_px']):.1f}px"
        if pd.notna(refine["mean_residual_px"])
        else f"{record.canonical_position} | live/fixed large well-detection QC | "
        f"cand=({int(refine['n_live_candidates'])},{int(refine['n_fixed_candidates'])}) "
        f"| dtheta={float(refine['delta_theta_deg']):.2f} "
        f"| dshift=({float(refine['delta_shift_x_px']):.1f}, {float(refine['delta_shift_y_px']):.1f})",
        fontsize=11,
    )
    return fig


def render_live_fixed_large_well_detection_qc_batch(
    records: Sequence[LiveFixedPairRecord],
    align_path: Union[str, Path],
    live_transform_path: Union[str, Path],
    fixed_transform_path: Union[str, Path],
    root: Union[str, Path],
    out_dir: Union[str, Path],
    well_detect_radii_px: Sequence[int] = DEFAULT_LARGE_WELL_RADII_PX,
    well_detect_canny_sigma: float = 2.0,
    well_detect_total_num_peaks: int = 96,
    well_detect_min_center_distance_px: int = DEFAULT_LARGE_WELL_MIN_CENTER_DISTANCE_PX,
    well_detect_min_accum: float = 0.18,
    well_detect_inner_std_max: float = 0.07,
    well_match_max_distance_px: float = 140.0,
    well_refine_max_shift_px: float = 180.0,
    well_refine_max_abs_theta_deg: float = 8.0,
    show_inline: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    align_df = pd.read_csv(Path(align_path), sep="\t")
    live_tr_df = mwa.load_pair_transform_table(Path(live_transform_path))
    fixed_tr_df = mwa.load_pair_transform_table(Path(fixed_transform_path))
    align_by_pos = {str(r.canonical_position): pd.Series(r._asdict()) for r in align_df.itertuples(index=False)}
    live_by_large = {str(r.large_file_path): pd.Series(r._asdict()) for r in live_tr_df.itertuples(index=False)}
    fixed_by_large = {str(r.large_file_path): pd.Series(r._asdict()) for r in fixed_tr_df.itertuples(index=False)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    total = len(records)
    for idx, rec in enumerate(records, start=1):
        align_row = align_by_pos.get(str(rec.canonical_position))
        live_row = live_by_large.get(str(rec.live_pair.large_file_path))
        fixed_row = fixed_by_large.get(str(rec.fixed_pair.large_file_path))
        out_png = out_dir / f"{rec.live_pair.cohort_id}_{rec.canonical_position}_live_fixed_well_detection_qc.png"

        if align_row is None or live_row is None or fixed_row is None:
            rows.append(
                {
                    "canonical_position": rec.canonical_position,
                    "status": "missing_inputs",
                    "qc_status": "missing_inputs",
                    "qc_png_path": str(out_png),
                }
            )
            if verbose:
                print(f"[{idx}/{total}] {rec.canonical_position} well-detection qc missing inputs")
            continue

        try:
            fig = render_live_fixed_large_well_detection_qc(
                record=rec,
                align_row=align_row,
                live_small_to_large_row=live_row,
                fixed_small_to_large_row=fixed_row,
                root=root,
                well_detect_radii_px=well_detect_radii_px,
                well_detect_canny_sigma=float(well_detect_canny_sigma),
                well_detect_total_num_peaks=int(well_detect_total_num_peaks),
                well_detect_min_center_distance_px=int(well_detect_min_center_distance_px),
                well_detect_min_accum=float(well_detect_min_accum),
                well_detect_inner_std_max=float(well_detect_inner_std_max),
                well_match_max_distance_px=float(well_match_max_distance_px),
                well_refine_max_shift_px=float(well_refine_max_shift_px),
                well_refine_max_abs_theta_deg=float(well_refine_max_abs_theta_deg),
            )
            fig.savefig(out_png, dpi=160, bbox_inches="tight")
            if show_inline:
                plt.show()
            plt.close(fig)
            rows.append(
                {
                    "canonical_position": rec.canonical_position,
                    "status": str(align_row["status"]),
                    "well_refine_status": str(align_row.get("well_refine_status", "")),
                    "well_refine_n_pairs": int(float(align_row.get("well_refine_n_pairs", 0) or 0)),
                    "well_refine_n_live_candidates": int(float(align_row.get("well_refine_n_live_candidates", 0) or 0)),
                    "well_refine_n_fixed_candidates": int(float(align_row.get("well_refine_n_fixed_candidates", 0) or 0)),
                    "qc_status": "rendered",
                    "qc_png_path": str(out_png),
                }
            )
            if verbose:
                print(f"[{idx}/{total}] {rec.canonical_position} well-detection qc rendered")
        except Exception as exc:
            rows.append(
                {
                    "canonical_position": rec.canonical_position,
                    "status": str(align_row.get("status", "")),
                    "well_refine_status": str(align_row.get("well_refine_status", "")),
                    "well_refine_n_pairs": int(float(align_row.get("well_refine_n_pairs", 0) or 0)),
                    "well_refine_n_live_candidates": int(float(align_row.get("well_refine_n_live_candidates", 0) or 0)),
                    "well_refine_n_fixed_candidates": int(float(align_row.get("well_refine_n_fixed_candidates", 0) or 0)),
                    "qc_status": f"error:{exc}",
                    "qc_png_path": str(out_png),
                }
            )
            if verbose:
                print(f"[{idx}/{total}] {rec.canonical_position} well-detection qc error: {exc}")

    return pd.DataFrame(rows)


def _load_live_fixed_large_bead_anchor_xy(
    record: LiveFixedPairRecord,
    refined_matrix: np.ndarray,
    root: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    live_bead_xy = _load_live_large_bead_xy(record=record, root=root)
    fixed_bead_xy = np.empty((0, 2), dtype=np.float64)
    if live_bead_xy.size == 0:
        return live_bead_xy, fixed_bead_xy
    refined_inv = inverse_affine_matrix(np.asarray(refined_matrix, dtype=np.float64))
    fixed_bead_xy = np.array(
        [apply_affine_to_xy(float(x), float(y), refined_inv) for x, y in live_bead_xy],
        dtype=np.float64,
    )
    return live_bead_xy, fixed_bead_xy


def _draw_peak_circles(
    ax: plt.Axes,
    peak_df: pd.DataFrame,
    color: str = "tomato",
    linewidth: float = 0.7,
    alpha: float = 0.7,
    max_draw: int = 32,
) -> None:
    if peak_df is None or peak_df.empty:
        return
    draw_df = peak_df.sort_values("accum", ascending=False).head(int(max_draw)).copy()
    for rr in draw_df.itertuples(index=False):
        circ = plt.Circle(
            (float(rr.x_px), float(rr.y_px)),
            float(rr.radius_px),
            edgecolor=color,
            facecolor="none",
            linewidth=float(linewidth),
            alpha=float(alpha),
            linestyle="-",
        )
        ax.add_patch(circ)
        ax.plot(float(rr.x_px), float(rr.y_px), marker=".", markersize=3.0, color=color, alpha=float(alpha))


def render_live_fixed_large_well_processing_qc(
    record: LiveFixedPairRecord,
    align_row: pd.Series,
    live_small_to_large_row: pd.Series,
    fixed_small_to_large_row: pd.Series,
    root: Union[str, Path],
    well_detect_radii_px: Sequence[int] = DEFAULT_LARGE_WELL_RADII_PX,
    well_detect_canny_sigma: float = 2.0,
    well_detect_total_num_peaks: int = 96,
    well_detect_min_center_distance_px: int = DEFAULT_LARGE_WELL_MIN_CENTER_DISTANCE_PX,
    well_detect_min_accum: float = 0.18,
    well_detect_inner_std_max: float = 0.07,
    well_match_max_distance_px: float = 140.0,
    well_refine_max_shift_px: float = 180.0,
    well_refine_max_abs_theta_deg: float = 8.0,
) -> plt.Figure:
    if str(align_row.get("status", "")) != "ok":
        raise RuntimeError(
            f"Alignment row not ok for {record.canonical_position}: {align_row.get('status')}"
        )

    root = Path(root).resolve()
    live_large = mwa.load_large_registered_stack(pair=record.live_pair, root=root)
    fixed_large = mwa.load_large_registered_stack(pair=record.fixed_pair, root=root)

    z_live = int(
        np.clip(
            int(round(float(align_row["live_large_focus_z_index"]))),
            0,
            live_large.bf_stack.shape[0] - 1,
        )
    )
    z_fixed = int(
        np.clip(
            int(round(float(align_row["fixed_large_focus_z_index"]))),
            0,
            fixed_large.bf_stack.shape[0] - 1,
        )
    )
    live_large_bf = live_large.bf_stack[z_live]
    fixed_large_bf = fixed_large.bf_stack[z_fixed]

    coarse_matrix = large_to_large_rigid_matrix_from_params(
        reference_shape_yx=live_large_bf.shape,
        moving_shape_yx=fixed_large_bf.shape,
        theta_deg=float(align_row["fixed_large_to_live_large_theta_deg"]),
        shift_x_px=float(align_row["fixed_large_to_live_large_shift_x_px"]),
        shift_y_px=float(align_row["fixed_large_to_live_large_shift_y_px"]),
    )
    candidate_scope = str(align_row.get("well_refine_candidate_scope", "all") or "all")
    force_relaxed_detection = bool(align_row.get("well_refine_force_relaxed_detection", False))
    live_bead_xy = _load_live_large_bead_xy(record=record, root=root)
    if str(align_row.get("well_refine_status", "")) == "skipped_by_position":
        refine = {
            "status": "skipped_by_position",
            "n_pairs": 0,
            "n_live_candidates": 0,
            "n_fixed_candidates": 0,
            "candidate_scope_used": "skipped_by_position",
            "force_relaxed_detection": False,
            "matrix": coarse_matrix,
            "delta_matrix": np.eye(3, dtype=np.float64),
            "delta_theta_deg": 0.0,
            "delta_shift_x_px": 0.0,
            "delta_shift_y_px": 0.0,
            "mean_residual_px": np.nan,
            "max_residual_px": np.nan,
            "matches_df": pd.DataFrame(),
        }
    else:
        refine = refine_fixed_large_to_live_large_with_well_pattern(
            live_large_bf=live_large_bf,
            fixed_large_bf=fixed_large_bf,
            live_small_to_large_row=live_small_to_large_row,
            fixed_small_to_large_row=fixed_small_to_large_row,
            fixed_large_to_live_large=coarse_matrix,
            position_key=str(record.canonical_position),
            live_bead_xy=live_bead_xy,
            radii_px=well_detect_radii_px,
            canny_sigma=float(well_detect_canny_sigma),
            total_num_peaks=int(well_detect_total_num_peaks),
            min_center_distance_px=int(well_detect_min_center_distance_px),
            min_accum=float(well_detect_min_accum),
            inner_std_max=float(well_detect_inner_std_max),
            force_relaxed_detection=force_relaxed_detection,
            candidate_scope=candidate_scope,
            max_match_distance_px=float(well_match_max_distance_px),
            max_correction_shift_px=float(well_refine_max_shift_px),
            max_correction_abs_theta_deg=float(well_refine_max_abs_theta_deg),
        )
    refined_matrix = np.asarray(refine["matrix"], dtype=np.float64)
    _, fixed_bead_xy = _load_live_fixed_large_bead_anchor_xy(
        record=record,
        refined_matrix=refined_matrix,
        root=root,
    )

    live_dbg = run_large_well_detection_debug(
        large_bf=live_large_bf,
        small_to_large_row=live_small_to_large_row,
        radii_px=well_detect_radii_px,
        canny_sigma=float(well_detect_canny_sigma),
        total_num_peaks=int(well_detect_total_num_peaks),
        min_center_distance_px=int(well_detect_min_center_distance_px),
        min_accum=float(well_detect_min_accum),
        inner_std_max=float(well_detect_inner_std_max),
    )
    fixed_dbg = run_large_well_detection_debug(
        large_bf=fixed_large_bf,
        small_to_large_row=fixed_small_to_large_row,
        radii_px=well_detect_radii_px,
        canny_sigma=float(well_detect_canny_sigma),
        total_num_peaks=int(well_detect_total_num_peaks),
        min_center_distance_px=int(well_detect_min_center_distance_px),
        min_accum=float(well_detect_min_accum),
        inner_std_max=float(well_detect_inner_std_max),
    )
    live_cand_df = _classify_candidates_by_manual_beads(
        live_dbg["candidate_df"],
        anchor_xy=live_bead_xy,
        max_match_distance_px=80.0,
    )
    fixed_cand_df = _classify_candidates_by_manual_beads(
        fixed_dbg["candidate_df"],
        anchor_xy=fixed_bead_xy,
        max_match_distance_px=80.0,
    )

    live_poly = np.asarray(
        mwa.build_small_fov_polygon_in_large(pd.Series(live_small_to_large_row)),
        dtype=np.float64,
    )
    fixed_poly = np.asarray(
        mwa.build_small_fov_polygon_in_large(pd.Series(fixed_small_to_large_row)),
        dtype=np.float64,
    )

    fig, axes = plt.subplots(2, 5, figsize=(24, 10), constrained_layout=True)

    def _plot_row(
        row_axes: Sequence[plt.Axes],
        label: str,
        dbg: Dict[str, object],
        cand_df: pd.DataFrame,
        bf: np.ndarray,
        poly: np.ndarray,
        bead_xy: np.ndarray,
    ) -> None:
        bf_disp = mwa.robust_rescale(bf)
        norm_disp = np.asarray(dbg["image_norm"], dtype=np.float32)
        edges_disp = np.asarray(dbg["edges"], dtype=bool)
        hough_disp = mwa.robust_rescale(np.asarray(dbg["hough_max"], dtype=np.float32))
        raw_df = dbg["raw_peaks_df"]

        row_axes[0].imshow(bf_disp, cmap="gray")
        row_axes[0].plot(
            np.r_[poly[:, 0], poly[0, 0]],
            np.r_[poly[:, 1], poly[0, 1]],
            color="cyan",
            linestyle="--",
            linewidth=1.0,
        )
        if bead_xy.size:
            row_axes[0].scatter(
                bead_xy[:, 0],
                bead_xy[:, 1],
                s=28,
                c="deepskyblue",
                marker="+",
                linewidths=1.0,
            )
        row_axes[0].set_title(f"{label} BF")

        row_axes[1].imshow(norm_disp, cmap="gray", vmin=0.0, vmax=1.0)
        row_axes[1].set_title(f"{label} normalized")

        row_axes[2].imshow(edges_disp, cmap="gray")
        row_axes[2].set_title(f"{label} Canny edges")

        row_axes[3].imshow(hough_disp, cmap="magma", vmin=0.0, vmax=1.0)
        _draw_peak_circles(
            row_axes[3],
            raw_df,
            color="tomato",
            linewidth=0.6,
            alpha=0.55,
            max_draw=32,
        )
        row_axes[3].set_title(
            f"{label} Hough response\nraw peaks shown={min(len(raw_df), 32)}/{len(raw_df)}"
        )

        row_axes[4].imshow(bf_disp, cmap="gray")
        row_axes[4].plot(
            np.r_[poly[:, 0], poly[0, 0]],
            np.r_[poly[:, 1], poly[0, 1]],
            color="cyan",
            linestyle="--",
            linewidth=1.0,
        )
        _draw_candidate_circles(
            row_axes[4],
            cand_df,
            bead_color="deepskyblue",
            denovo_color="gold",
            center_size=7.0,
            linewidth=1.0,
            alpha=0.95,
        )
        if bead_xy.size:
            row_axes[4].scatter(
                bead_xy[:, 0],
                bead_xy[:, 1],
                s=28,
                c="deepskyblue",
                marker="+",
                linewidths=1.0,
            )
        row_axes[4].set_title(
            f"{label} kept wells\nbead-linked={int(cand_df['bead_annotated'].sum())} de novo={int((~cand_df['bead_annotated']).sum())}"
        )

        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

    _plot_row(axes[0], "Live", live_dbg, live_cand_df, live_large_bf, live_poly, live_bead_xy)
    _plot_row(axes[1], "Fixed", fixed_dbg, fixed_cand_df, fixed_large_bf, fixed_poly, fixed_bead_xy)

    rmin = int(min(well_detect_radii_px)) if len(well_detect_radii_px) else -1
    rmax = int(max(well_detect_radii_px)) if len(well_detect_radii_px) else -1
    fig.suptitle(
        f"{record.canonical_position} | large-well processing debug | "
        f"radii={rmin}-{rmax}px | canny_sigma={float(well_detect_canny_sigma):.1f} | "
        f"min_accum={float(well_detect_min_accum):.2f} | inner_std_max={float(well_detect_inner_std_max):.2f}",
        fontsize=12,
    )
    return fig


def render_live_fixed_large_well_processing_qc_batch(
    records: Sequence[LiveFixedPairRecord],
    align_path: Union[str, Path],
    live_transform_path: Union[str, Path],
    fixed_transform_path: Union[str, Path],
    root: Union[str, Path],
    out_dir: Union[str, Path],
    positions: Optional[Sequence[str]] = None,
    well_detect_radii_px: Sequence[int] = DEFAULT_LARGE_WELL_RADII_PX,
    well_detect_canny_sigma: float = 2.0,
    well_detect_total_num_peaks: int = 96,
    well_detect_min_center_distance_px: int = DEFAULT_LARGE_WELL_MIN_CENTER_DISTANCE_PX,
    well_detect_min_accum: float = 0.18,
    well_detect_inner_std_max: float = 0.07,
    well_match_max_distance_px: float = 140.0,
    well_refine_max_shift_px: float = 180.0,
    well_refine_max_abs_theta_deg: float = 8.0,
    show_inline: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    align_df = pd.read_csv(Path(align_path), sep="\t")
    live_tr_df = mwa.load_pair_transform_table(Path(live_transform_path))
    fixed_tr_df = mwa.load_pair_transform_table(Path(fixed_transform_path))
    align_by_pos = {str(r.canonical_position): pd.Series(r._asdict()) for r in align_df.itertuples(index=False)}
    live_by_large = {str(r.large_file_path): pd.Series(r._asdict()) for r in live_tr_df.itertuples(index=False)}
    fixed_by_large = {str(r.large_file_path): pd.Series(r._asdict()) for r in fixed_tr_df.itertuples(index=False)}
    requested = {str(p) for p in positions} if positions is not None else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    selected_records = [r for r in records if requested is None or str(r.canonical_position) in requested]
    total = len(selected_records)
    for idx, rec in enumerate(selected_records, start=1):
        align_row = align_by_pos.get(str(rec.canonical_position))
        live_row = live_by_large.get(str(rec.live_pair.large_file_path))
        fixed_row = fixed_by_large.get(str(rec.fixed_pair.large_file_path))
        out_png = out_dir / f"{rec.live_pair.cohort_id}_{rec.canonical_position}_live_fixed_well_processing_qc.png"

        if align_row is None or live_row is None or fixed_row is None:
            rows.append(
                {
                    "canonical_position": rec.canonical_position,
                    "status": "missing_inputs",
                    "qc_status": "missing_inputs",
                    "qc_png_path": str(out_png),
                }
            )
            if verbose:
                print(f"[{idx}/{total}] {rec.canonical_position} well-processing qc missing inputs")
            continue

        try:
            fig = render_live_fixed_large_well_processing_qc(
                record=rec,
                align_row=align_row,
                live_small_to_large_row=live_row,
                fixed_small_to_large_row=fixed_row,
                root=root,
                well_detect_radii_px=well_detect_radii_px,
                well_detect_canny_sigma=float(well_detect_canny_sigma),
                well_detect_total_num_peaks=int(well_detect_total_num_peaks),
                well_detect_min_center_distance_px=int(well_detect_min_center_distance_px),
                well_detect_min_accum=float(well_detect_min_accum),
                well_detect_inner_std_max=float(well_detect_inner_std_max),
                well_match_max_distance_px=float(well_match_max_distance_px),
                well_refine_max_shift_px=float(well_refine_max_shift_px),
                well_refine_max_abs_theta_deg=float(well_refine_max_abs_theta_deg),
            )
            fig.savefig(out_png, dpi=160, bbox_inches="tight")
            if show_inline:
                plt.show()
            plt.close(fig)
            rows.append(
                {
                    "canonical_position": rec.canonical_position,
                    "status": str(align_row["status"]),
                    "qc_status": "rendered",
                    "qc_png_path": str(out_png),
                }
            )
            if verbose:
                print(f"[{idx}/{total}] {rec.canonical_position} well-processing qc rendered")
        except Exception as exc:
            rows.append(
                {
                    "canonical_position": rec.canonical_position,
                    "status": str(align_row.get("status", "")),
                    "qc_status": f"error:{exc}",
                    "qc_png_path": str(out_png),
                }
            )
            if verbose:
                print(f"[{idx}/{total}] {rec.canonical_position} well-processing qc error: {exc}")

    return pd.DataFrame(rows)


def _load_small_channel_image(record: mwa.PairedImageRecord, root: Path, channel_keywords: Sequence[str]) -> np.ndarray:
    abs_path = (root / record.small_file_path).resolve()
    img = common.read_czi(abs_path)
    idx = common.find_channel_index(img.channels, list(channel_keywords))
    if idx is None:
        raise RuntimeError(f"Missing channel {list(channel_keywords)} in {abs_path}")
    return img.channel_images[idx].astype(np.float32)


def render_live_fixed_alignment_qc(
    record: LiveFixedPairRecord,
    align_row: pd.Series,
    root: Union[str, Path],
) -> plt.Figure:
    if str(align_row.get("status", "")) != "ok":
        raise RuntimeError(f"Alignment row not ok for {record.canonical_position}: {align_row.get('status')}")

    root = Path(root).resolve()
    live_pair = record.live_pair
    fixed_pair = record.fixed_pair
    live_large = mwa.load_large_registered_stack(pair=live_pair, root=root)
    fixed_large = mwa.load_large_registered_stack(pair=fixed_pair, root=root)
    fixed_large_dapi_stack = mwa.load_large_registered_channel_stack(
        pair=fixed_pair,
        root=root,
        channel_keywords=["dapi"],
    )

    z_live = int(np.clip(int(round(float(align_row["live_large_focus_z_index"]))), 0, live_large.bf_stack.shape[0] - 1))
    z_fixed = int(np.clip(int(round(float(align_row["fixed_large_focus_z_index"]))), 0, fixed_large.bf_stack.shape[0] - 1))
    live_large_bf = live_large.bf_stack[z_live]
    fixed_large_bf = fixed_large.bf_stack[z_fixed]
    fixed_large_dapi = fixed_large_dapi_stack[int(np.clip(z_fixed, 0, fixed_large_dapi_stack.shape[0] - 1))]

    live_tr_df = mwa.load_pair_transform_table(root / "results/annotations/small_to_large_pair_transforms.tsv")
    fixed_tr_df = mwa.load_pair_transform_table(root / "results/annotations/fixed_small_to_large_pair_transforms.tsv")
    live_tr = live_tr_df.loc[lambda d: d["large_file_path"] == live_pair.large_file_path].iloc[0]
    fixed_tr = fixed_tr_df.loc[lambda d: d["large_file_path"] == fixed_pair.large_file_path].iloc[0]

    live_small_to_live_large = small_to_large_affine_matrix(
        small_shape_yx=(int(live_tr["small_shape_y_px"]), int(live_tr["small_shape_x_px"])),
        top_left_x_px=float(live_tr["small_to_large_top_left_x_px"]),
        top_left_y_px=float(live_tr["small_to_large_top_left_y_px"]),
        theta_deg=float(live_tr["small_to_large_theta_deg"]),
    )
    fixed_small_to_fixed_large = small_to_large_affine_matrix(
        small_shape_yx=(int(fixed_tr["small_shape_y_px"]), int(fixed_tr["small_shape_x_px"])),
        top_left_x_px=float(fixed_tr["small_to_large_top_left_x_px"]),
        top_left_y_px=float(fixed_tr["small_to_large_top_left_y_px"]),
        theta_deg=float(fixed_tr["small_to_large_theta_deg"]),
    )
    fixed_small_to_live_large = affine_matrix_from_row_prefix(align_row, "fixed_small_to_live_large")
    fixed_large_to_live_large = fixed_small_to_live_large @ inverse_affine_matrix(fixed_small_to_fixed_large)

    warped_fixed_large_bf = warp_image_with_affine(
        image=fixed_large_bf,
        forward_mat=fixed_large_to_live_large,
        output_shape_yx=live_large_bf.shape,
        order=1,
        cval=np.nan,
    )
    warped_fixed_large_dapi = warp_image_with_affine(
        image=fixed_large_dapi,
        forward_mat=fixed_large_to_live_large,
        output_shape_yx=live_large_bf.shape,
        order=1,
        cval=np.nan,
    )

    fixed_large_fill = float(np.nanmedian(warped_fixed_large_bf)) if np.isfinite(warped_fixed_large_bf).any() else 0.0
    fixed_dapi_fill = float(np.nanmedian(warped_fixed_large_dapi)) if np.isfinite(warped_fixed_large_dapi).any() else 0.0
    warped_fixed_large_bf_fill = np.nan_to_num(warped_fixed_large_bf, nan=fixed_large_fill)
    warped_fixed_large_dapi_fill = np.nan_to_num(warped_fixed_large_dapi, nan=fixed_dapi_fill)

    overlay = np.zeros((*live_large_bf.shape, 3), dtype=np.float32)
    overlay[..., 0] = mwa.robust_rescale(warped_fixed_large_bf_fill)
    overlay[..., 1] = mwa.robust_rescale(live_large_bf)
    overlay[..., 2] = mwa.robust_rescale(live_large_bf)

    diff = np.abs(
        mwa._zscore_norm(live_large_bf)
        - mwa._zscore_norm(warped_fixed_large_bf_fill)
    )

    live_fov_poly = mwa.build_small_fov_polygon_in_large(pd.Series(live_tr))
    live_fov_poly = np.vstack([live_fov_poly, live_fov_poly[0]])
    fixed_fov_poly_native = mwa.build_small_fov_polygon_in_large(pd.Series(fixed_tr))
    fixed_fov_poly_native = np.vstack([fixed_fov_poly_native, fixed_fov_poly_native[0]])
    fixed_fov_poly_live = np.array(
        [apply_affine_to_xy(x_px=float(x), y_px=float(y), mat=fixed_large_to_live_large) for x, y in fixed_fov_poly_native[:-1]],
        dtype=np.float64,
    )
    fixed_fov_poly_live = np.vstack([fixed_fov_poly_live, fixed_fov_poly_live[0]])

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2), constrained_layout=True)
    axes[0].imshow(mwa.robust_rescale(live_large_bf), cmap="gray")
    axes[0].plot(live_fov_poly[:, 0], live_fov_poly[:, 1], color="cyan", linestyle="--", linewidth=1.2, label="live small FOV")
    axes[0].plot(fixed_fov_poly_live[:, 0], fixed_fov_poly_live[:, 1], color="magenta", linestyle="-", linewidth=1.2, label="fixed small FOV -> live")
    axes[0].set_title("Live large BF")
    axes[0].axis("off")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")

    axes[1].imshow(mwa.robust_rescale(fixed_large_bf), cmap="gray")
    axes[1].plot(fixed_fov_poly_native[:, 0], fixed_fov_poly_native[:, 1], color="deepskyblue", linestyle="--", linewidth=1.2)
    axes[1].set_title("Fixed large BF (native)")
    axes[1].axis("off")

    axes[2].imshow(mwa.robust_rescale(warped_fixed_large_bf_fill), cmap="gray")
    axes[2].plot(live_fov_poly[:, 0], live_fov_poly[:, 1], color="cyan", linestyle="--", linewidth=1.2)
    axes[2].plot(fixed_fov_poly_live[:, 0], fixed_fov_poly_live[:, 1], color="magenta", linestyle="-", linewidth=1.2)
    axes[2].set_title("Fixed large BF -> live large")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].plot(live_fov_poly[:, 0], live_fov_poly[:, 1], color="cyan", linestyle="--", linewidth=1.0)
    axes[3].plot(fixed_fov_poly_live[:, 0], fixed_fov_poly_live[:, 1], color="magenta", linestyle="-", linewidth=1.0)
    axes[3].set_title("Overlay: fixed large BF (R) vs live large BF (G)")
    axes[3].axis("off")

    axes[4].imshow(mwa.robust_rescale(diff), cmap="inferno", vmin=0.0, vmax=1.0)
    axes[4].contour(mwa.robust_rescale(warped_fixed_large_dapi_fill), levels=[0.5], colors="white", linewidths=0.5, alpha=0.75)
    axes[4].plot(live_fov_poly[:, 0], live_fov_poly[:, 1], color="cyan", linestyle="--", linewidth=1.0)
    axes[4].plot(fixed_fov_poly_live[:, 0], fixed_fov_poly_live[:, 1], color="magenta", linestyle="-", linewidth=1.0)
    axes[4].set_title("|live - fixed large BF| + fixed DAPI contour")
    axes[4].axis("off")

    fig.suptitle(
        f"{record.canonical_position} | score={float(align_row['fixed_large_to_live_large_score']):.4f} "
        f"| theta={float(align_row['fixed_large_to_live_large_theta_deg']):.2f}deg "
        f"| shift=({float(align_row['fixed_large_to_live_large_shift_x_px']):.1f}, "
        f"{float(align_row['fixed_large_to_live_large_shift_y_px']):.1f}) "
        f"| refine={str(align_row.get('well_refine_status', 'na'))} "
        f"n={int(float(align_row.get('well_refine_n_pairs', 0) or 0))} "
        f"cand=({int(float(align_row.get('well_refine_n_live_candidates', 0) or 0))},"
        f"{int(float(align_row.get('well_refine_n_fixed_candidates', 0) or 0))})",
        fontsize=11,
    )
    return fig


def render_live_fixed_alignment_qc_batch(
    records: Sequence[LiveFixedPairRecord],
    align_path: Union[str, Path],
    root: Union[str, Path],
    out_dir: Union[str, Path],
    show_inline: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    align_df = pd.read_csv(Path(align_path), sep="\t")
    align_by_pos = {str(r.canonical_position): pd.Series(r._asdict()) for r in align_df.itertuples(index=False)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    total = len(records)
    for idx, rec in enumerate(records, start=1):
        row = align_by_pos.get(str(rec.canonical_position))
        if row is None:
            rows.append(
                {
                    "canonical_position": rec.canonical_position,
                    "status": "missing_alignment_row",
                    "qc_status": "missing_alignment_row",
                    "qc_png_path": "",
                }
            )
            continue
        out_png = out_dir / f"{rec.live_pair.cohort_id}_{rec.canonical_position}_live_fixed_alignment_qc.png"
        try:
            fig = render_live_fixed_alignment_qc(record=rec, align_row=row, root=root)
            fig.savefig(out_png, dpi=160, bbox_inches="tight")
            if show_inline:
                plt.show()
            plt.close(fig)
            rows.append(
                {
                    "canonical_position": rec.canonical_position,
                    "status": str(row["status"]),
                    "qc_status": "rendered",
                    "qc_png_path": str(out_png),
                }
            )
            if verbose:
                print(f"[{idx}/{total}] {rec.canonical_position} rendered")
        except Exception as exc:
            rows.append(
                {
                    "canonical_position": rec.canonical_position,
                    "status": str(row["status"]),
                    "qc_status": f"error:{exc}",
                    "qc_png_path": str(out_png),
                }
            )
            if verbose:
                print(f"[{idx}/{total}] {rec.canonical_position} qc error: {exc}")

    return pd.DataFrame(rows)
