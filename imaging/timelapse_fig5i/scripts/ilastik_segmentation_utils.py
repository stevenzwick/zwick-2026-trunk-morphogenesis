from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation


THREE_CLASS_DEFAULTS = {
    "support_sigma": 1.0,
    "artifact_sigma": 1.0,
    "support_low_thr": 0.30,
    "support_high_thr": 0.52,
    "seed_local_max_thr": 0.50,
    "seed_min_distance": 3,
    "weak_seed_weight": 0.0,
    "artifact_penalty_weight": 0.18,
    "open_radius": 0,
    "close_radius": 0,
    "fill_hole_area": 48,
    "min_area": 18,
    "max_area": 240,
    "min_solidity": 0.45,
    "max_eccentricity": 0.995,
    "min_circularity": 0.0,
    "min_mean_pos_prob": 0.34,
    "max_mean_art_prob": 0.62,
}

FOUR_CLASS_DEFAULTS = {
    "support_sigma": 1.0,
    "artifact_sigma": 1.0,
    "support_low_thr": 0.30,
    "support_high_thr": 0.52,
    "seed_local_max_thr": 0.40,
    "seed_min_distance": 5,
    "weak_seed_weight": 0.55,
    "artifact_penalty_weight": 0.35,
    "open_radius": 1,
    "close_radius": 1,
    "fill_hole_area": 64,
    "min_area": 24,
    "max_area": 320,
    "min_solidity": 0.55,
    "max_eccentricity": 0.985,
    "min_circularity": 0.0,
    "min_mean_pos_prob": 0.28,
    "max_mean_art_prob": 0.60,
}

THRESHOLD_WATERSHED_DEFAULTS = {
    "method": "threshold_watershed",
    "sigma": 1.0,
    "art_penalty": 0.0,
    "mask_thr": 0.18,
    "peak_thr": 0.15,
    "min_distance": 3,
    "min_size": 8,
    "fill_hole_area": 0,
    "ensure_seed_per_component": True,
}


def default_segmentation_cfg(class_names: list[str] | tuple[str, ...]) -> dict:
    class_names = list(class_names)
    if len(class_names) == 3:
        cfg = dict(THREE_CLASS_DEFAULTS)
        cfg["method"] = "legacy_multistage"
        return cfg
    if len(class_names) == 4:
        cfg = dict(FOUR_CLASS_DEFAULTS)
        cfg["method"] = "legacy_multistage"
        return cfg
    raise ValueError(f"Unsupported number of classes: {len(class_names)}")


def default_threshold_watershed_cfg() -> dict:
    return dict(THRESHOLD_WATERSHED_DEFAULTS)


def recommended_segmentation_cfg(dataset_id: str, class_names: list[str] | tuple[str, ...]) -> dict:
    dataset_id = str(dataset_id)
    if dataset_id == "20260325":
        cfg = default_threshold_watershed_cfg()
        cfg.update(
            {
                "sigma": 0.5,
                "art_penalty": 0.30,
                "mask_thr": 0.24,
                "peak_thr": 0.35,
                "min_distance": 2,
                "min_size": 12,
                "fill_hole_area": 0,
                "ensure_seed_per_component": False,
            }
        )
        return cfg
    return default_segmentation_cfg(class_names)


def robust_limits(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> tuple[float, float]:
    lo, hi = np.percentile(img.astype(np.float32), [low_pct, high_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(img))
        hi = float(np.max(img))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def clamp_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(np.float32, copy=False), 0.0, 1.0)


def _class_maps(prob_frame: np.ndarray, class_names: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
    probs = clamp_probs(prob_frame)
    class_names = list(class_names)
    if probs.ndim != 3 or probs.shape[0] != len(class_names):
        raise ValueError(f"Expected probability frame shape (C,Y,X) with C={len(class_names)}, got {prob_frame.shape}")

    if len(class_names) == 3:
        p_pos, p_art, p_other = probs
        return {
            "p_pos": p_pos,
            "p_weak": np.zeros_like(p_pos),
            "p_art": p_art,
            "p_other": p_other,
            "p_support": p_pos,
        }

    if len(class_names) == 4:
        p_pos, p_weak, p_art, p_other = probs
        return {
            "p_pos": p_pos,
            "p_weak": p_weak,
            "p_art": p_art,
            "p_other": p_other,
            "p_support": np.clip(p_pos + p_weak, 0.0, 1.0),
        }

    raise ValueError(f"Unsupported number of classes: {len(class_names)}")


def region_df(labels: np.ndarray, p_pos: np.ndarray, p_art: np.ndarray) -> pd.DataFrame:
    if int(labels.max()) == 0:
        return pd.DataFrame(
            columns=[
                "label", "area", "solidity", "eccentricity", "perimeter", "circularity",
                "centroid_y", "centroid_x", "mean_pos_prob", "max_pos_prob", "mean_art_prob",
            ]
        )
    rows = []
    for r in measure.regionprops(labels, intensity_image=p_pos.astype(np.float32)):
        rr, cc = r.coords[:, 0], r.coords[:, 1]
        perim = float(r.perimeter) if r.perimeter is not None else 0.0
        circ = 0.0 if perim <= 0 else float((4.0 * np.pi * float(r.area)) / (perim ** 2))
        rows.append(
            {
                "label": int(r.label),
                "area": float(r.area),
                "solidity": float(r.solidity) if r.solidity is not None else np.nan,
                "eccentricity": float(r.eccentricity) if r.eccentricity is not None else np.nan,
                "perimeter": perim,
                "circularity": circ,
                "centroid_y": float(r.centroid[0]),
                "centroid_x": float(r.centroid[1]),
                "mean_pos_prob": float(np.mean(p_pos[rr, cc])) if rr.size else np.nan,
                "max_pos_prob": float(np.max(p_pos[rr, cc])) if rr.size else np.nan,
                "mean_art_prob": float(np.mean(p_art[rr, cc])) if rr.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ensure_seed_per_component(seed_labels: np.ndarray, mask: np.ndarray, seed_driver: np.ndarray | None = None) -> np.ndarray:
    out = seed_labels.copy()
    cc, n_cc = ndi.label(mask)
    next_label = int(out.max()) + 1
    for cc_id in range(1, n_cc + 1):
        cc_mask = cc == cc_id
        if np.any(out[cc_mask] > 0):
            continue
        ys, xs = np.where(cc_mask)
        if not len(ys):
            continue
        if seed_driver is not None:
            vals = seed_driver[cc_mask]
            idx = int(np.argmax(vals))
            y = int(ys[idx])
            x = int(xs[idx])
        else:
            idx = len(ys) // 2
            y = int(ys[idx])
            x = int(xs[idx])
        out[y, x] = next_label
        next_label += 1
    return out


def segment_probability_frame(
    raw_frame: np.ndarray,
    prob_frame: np.ndarray,
    cfg: dict,
    class_names: list[str] | tuple[str, ...],
    return_debug: bool = False,
):
    maps = _class_maps(prob_frame, class_names)
    p_pos = maps["p_pos"]
    p_weak = maps["p_weak"]
    p_art = maps["p_art"]
    p_other = maps["p_other"]
    p_support = maps["p_support"]

    support_sm = filters.gaussian(p_support, sigma=float(cfg["support_sigma"]), preserve_range=True)
    art_sm = filters.gaussian(p_art, sigma=float(cfg["artifact_sigma"]), preserve_range=True)

    seed_driver = np.clip(
        p_pos + float(cfg.get("weak_seed_weight", 0.0)) * p_weak - float(cfg.get("artifact_penalty_weight", 0.0)) * art_sm,
        0.0,
        1.0,
    )

    low = support_sm >= float(cfg["support_low_thr"])
    high = seed_driver >= float(cfg["support_high_thr"])
    if np.any(high):
        hyst = ndi.binary_propagation(high, mask=low)
    else:
        hyst = low.copy()

    post = hyst.copy()
    open_radius = int(cfg.get("open_radius", 0))
    close_radius = int(cfg.get("close_radius", 0))
    if open_radius > 0:
        post = morphology.binary_opening(post, morphology.disk(open_radius))
    if close_radius > 0:
        post = morphology.binary_closing(post, morphology.disk(close_radius))
    post = ndi.binary_fill_holes(post)
    post = morphology.remove_small_holes(post, area_threshold=int(cfg["fill_hole_area"]))
    post = morphology.remove_small_objects(post, min_size=max(4, int(cfg["min_area"] * 0.5)))

    peaks = feature.peak_local_max(
        seed_driver,
        labels=post.astype(np.uint8),
        min_distance=int(cfg["seed_min_distance"]),
        threshold_abs=float(cfg["seed_local_max_thr"]),
        exclude_border=False,
    )
    seed_mask = np.zeros_like(post, dtype=bool)
    if len(peaks):
        seed_mask[tuple(peaks.T)] = True
    seed_labels, _ = ndi.label(seed_mask)
    seed_labels = ensure_seed_per_component(seed_labels, post, seed_driver=seed_driver)

    labels_ws = segmentation.watershed(-seed_driver, markers=seed_labels, mask=post)
    labels_ws = morphology.remove_small_objects(labels_ws, min_size=int(cfg["min_area"]))

    keep = np.zeros(int(labels_ws.max()) + 1, dtype=bool)
    stats = region_df(labels_ws, p_pos=p_pos, p_art=p_art)
    for row in stats.itertuples(index=False):
        if row.area < float(cfg["min_area"]) or row.area > float(cfg["max_area"]):
            continue
        if not np.isfinite(row.solidity) or row.solidity < float(cfg["min_solidity"]):
            continue
        if not np.isfinite(row.eccentricity) or row.eccentricity > float(cfg["max_eccentricity"]):
            continue
        if not np.isfinite(row.circularity) or row.circularity < float(cfg.get("min_circularity", 0.0)):
            continue
        if not np.isfinite(row.mean_pos_prob) or row.mean_pos_prob < float(cfg["min_mean_pos_prob"]):
            continue
        if np.isfinite(row.mean_art_prob) and row.mean_art_prob > float(cfg["max_mean_art_prob"]):
            continue
        keep[int(row.label)] = True

    final = labels_ws.copy()
    if len(keep):
        final[~keep[final]] = 0
    # Preserve watershed instance boundaries after filtering; binarizing here
    # would merge adjacent kept nuclei back together.
    final, _, _ = segmentation.relabel_sequential(final)
    final = final.astype(np.int32, copy=False)
    stats_final = region_df(final, p_pos=p_pos, p_art=p_art)

    if not return_debug:
        return final

    debug = {
        "phase": raw_frame[0].astype(np.float32, copy=False),
        "rfp": raw_frame[1].astype(np.float32, copy=False),
        "p_pos": p_pos,
        "p_weak": p_weak,
        "p_art": p_art,
        "p_other": p_other,
        "p_support": p_support,
        "support_sm": support_sm,
        "seed_driver": seed_driver,
        "low": low,
        "high": high,
        "hyst": hyst,
        "post": post,
        "seed_points": peaks,
        "seed_labels": seed_labels,
        "labels_ws": labels_ws,
        "final": final,
        "stats_ws": stats,
        "stats_final": stats_final,
        "stats": stats_final,
    }
    return final, debug


def segment_probability_frame_threshold_watershed(
    raw_frame: np.ndarray,
    prob_frame: np.ndarray,
    cfg: dict,
    class_names: list[str] | tuple[str, ...],
    return_debug: bool = False,
):
    maps = _class_maps(prob_frame, class_names)
    p_pos = maps["p_pos"]
    p_art = maps["p_art"]
    p_other = maps["p_other"]

    sigma = float(cfg.get("sigma", 1.0))
    if sigma > 0:
        support = filters.gaussian(p_pos, sigma=sigma, preserve_range=True)
        art = filters.gaussian(p_art, sigma=sigma, preserve_range=True)
    else:
        support = p_pos.astype(np.float32, copy=False)
        art = p_art.astype(np.float32, copy=False)

    seed_driver = np.clip(support - float(cfg.get("art_penalty", 0.0)) * art, 0.0, 1.0)
    mask = support >= float(cfg.get("mask_thr", 0.18))
    mask = ndi.binary_fill_holes(mask)
    fill_hole_area = int(cfg.get("fill_hole_area", 0))
    if fill_hole_area > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=fill_hole_area)
    min_size = int(cfg.get("min_size", 8))
    mask = morphology.remove_small_objects(mask, min_size=min_size)

    peaks = feature.peak_local_max(
        seed_driver,
        labels=mask.astype(np.uint8),
        min_distance=int(cfg.get("min_distance", 3)),
        threshold_abs=float(cfg.get("peak_thr", 0.15)),
        exclude_border=False,
    )
    seed_mask = np.zeros_like(mask, dtype=bool)
    if len(peaks):
        seed_mask[tuple(peaks.T)] = True
    seed_labels, _ = ndi.label(seed_mask)
    if bool(cfg.get("ensure_seed_per_component", True)):
        seed_labels = ensure_seed_per_component(seed_labels, mask, seed_driver=seed_driver)

    labels_ws = segmentation.watershed(-seed_driver, markers=seed_labels, mask=mask)
    final = morphology.remove_small_objects(labels_ws, min_size=min_size)
    final, _, _ = segmentation.relabel_sequential(final)
    final = final.astype(np.int32, copy=False)
    stats_final = region_df(final, p_pos=p_pos, p_art=p_art)

    if not return_debug:
        return final

    debug = {
        "phase": raw_frame[0].astype(np.float32, copy=False),
        "rfp": raw_frame[1].astype(np.float32, copy=False),
        "p_pos": p_pos,
        "p_weak": maps["p_weak"],
        "p_art": p_art,
        "p_other": p_other,
        "p_support": maps["p_support"],
        "support_sm": support,
        "seed_driver": seed_driver,
        "mask": mask,
        "low": mask,
        "high": seed_driver >= float(cfg.get("peak_thr", 0.15)),
        "hyst": mask,
        "post": mask,
        "seed_points": peaks,
        "seed_labels": seed_labels,
        "labels_ws": labels_ws,
        "final": final,
        "stats_ws": region_df(labels_ws, p_pos=p_pos, p_art=p_art),
        "stats_final": stats_final,
        "stats": stats_final,
    }
    return final, debug


def segment_probability_frame_auto(
    raw_frame: np.ndarray,
    prob_frame: np.ndarray,
    cfg: dict,
    class_names: list[str] | tuple[str, ...],
    return_debug: bool = False,
):
    method = str(cfg.get("method", "legacy_multistage"))
    if method == "threshold_watershed":
        return segment_probability_frame_threshold_watershed(
            raw_frame,
            prob_frame,
            cfg,
            class_names,
            return_debug=return_debug,
        )
    return segment_probability_frame(
        raw_frame,
        prob_frame,
        cfg,
        class_names,
        return_debug=return_debug,
    )


def segmentation_overlay(gray: np.ndarray, labels: np.ndarray, color=(1.0, 0.15, 0.15)) -> np.ndarray:
    lo, hi = robust_limits(gray)
    base = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = np.dstack([base, base, base])
    mask = labels > 0
    return segmentation.mark_boundaries(rgb, mask, color=color, mode="thick")


def seed_overlay(gray: np.ndarray, peaks: np.ndarray, color=(0.0, 1.0, 1.0)) -> np.ndarray:
    lo, hi = robust_limits(gray)
    base = np.clip((gray.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = np.dstack([base, base, base])
    if len(peaks):
        for y, x in peaks:
            y = int(y)
            x = int(x)
            rr0 = max(0, y - 2)
            rr1 = min(rgb.shape[0], y + 3)
            cc0 = max(0, x - 2)
            cc1 = min(rgb.shape[1], x + 3)
            rgb[rr0:rr1, x:x+1] = color
            rgb[y:y+1, cc0:cc1] = color
    return np.clip(rgb, 0.0, 1.0)


def labels_to_detection_table(labels_stack: np.ndarray, raw_mm: np.ndarray, prob_mm: np.ndarray, class_names: list[str] | tuple[str, ...]) -> pd.DataFrame:
    rows = []
    det_id = 1
    for t in range(int(labels_stack.shape[0])):
        labels = labels_stack[t]
        maps = _class_maps(prob_mm[t], class_names)
        p_pos = maps["p_pos"]
        p_art = maps["p_art"]
        rfp = raw_mm[t, 1].astype(np.float32)
        for r in measure.regionprops(labels, intensity_image=p_pos.astype(np.float32)):
            rr, cc = r.coords[:, 0], r.coords[:, 1]
            rows.append(
                {
                    "det_id": int(det_id),
                    "time": int(t),
                    "label": int(r.label),
                    "cy": float(r.centroid[0]),
                    "cx": float(r.centroid[1]),
                    "area": float(r.area),
                    "mean_prob": float(np.mean(p_pos[rr, cc])) if rr.size else np.nan,
                    "max_prob": float(np.max(p_pos[rr, cc])) if rr.size else np.nan,
                    "mean_art_prob": float(np.mean(p_art[rr, cc])) if rr.size else np.nan,
                    "mean_rfp": float(np.mean(rfp[rr, cc])) if rr.size else np.nan,
                    "solidity": float(r.solidity) if r.solidity is not None else np.nan,
                    "eccentricity": float(r.eccentricity) if r.eccentricity is not None else np.nan,
                    "bbox_minr": int(r.bbox[0]),
                    "bbox_minc": int(r.bbox[1]),
                    "bbox_maxr": int(r.bbox[2]),
                    "bbox_maxc": int(r.bbox[3]),
                    "radius": float(max(2.0, math.sqrt(float(r.area) / math.pi))),
                }
            )
            det_id += 1
    if not rows:
        return pd.DataFrame(columns=[
            "det_id", "time", "label", "cy", "cx", "area", "mean_prob", "max_prob", "mean_art_prob", "mean_rfp",
            "solidity", "eccentricity", "bbox_minr", "bbox_minc", "bbox_maxr", "bbox_maxc", "radius"
        ])
    return pd.DataFrame(rows).sort_values(["time", "label"]).reset_index(drop=True)


def segment_stack(raw_mm: np.ndarray, prob_mm: np.ndarray, cfg: dict, class_names: list[str] | tuple[str, ...]) -> tuple[np.ndarray, pd.DataFrame]:
    n_time = int(raw_mm.shape[0])
    h = int(raw_mm.shape[-2])
    w = int(raw_mm.shape[-1])
    labels_stack = np.zeros((n_time, h, w), dtype=np.uint16)
    frame_rows = []
    for t in range(n_time):
        labels = segment_probability_frame_auto(raw_mm[t], prob_mm[t], cfg, class_names, return_debug=False)
        labels_stack[t] = labels.astype(np.uint16, copy=False)
        maps = _class_maps(prob_mm[t], class_names)
        p_pos = maps["p_pos"]
        p_art = maps["p_art"]
        stats = region_df(labels, p_pos=p_pos, p_art=p_art)
        frame_rows.append(
            {
                "time": int(t),
                "n_objects": int(labels.max()),
                "median_area": float(stats["area"].median()) if len(stats) else np.nan,
                "median_solidity": float(stats["solidity"].median()) if len(stats) else np.nan,
                "median_circularity": float(stats["circularity"].median()) if len(stats) else np.nan,
                "median_mean_pos_prob": float(stats["mean_pos_prob"].median()) if len(stats) else np.nan,
                "median_mean_art_prob": float(stats["mean_art_prob"].median()) if len(stats) else np.nan,
                "frame_mean_pos_prob": float(np.mean(p_pos)),
                "frame_mean_art_prob": float(np.mean(p_art)),
                "total_positive_mass": float(np.sum(p_pos)),
                "captured_positive_mass": float(np.sum(p_pos[labels > 0])),
                "captured_positive_frac": float(np.sum(p_pos[labels > 0]) / max(np.sum(p_pos), 1e-6)),
            }
        )
    return labels_stack, pd.DataFrame(frame_rows)
