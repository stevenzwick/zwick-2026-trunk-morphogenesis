"""Helpers for day 5 trunk-morph morphological quantification.

This module keeps the notebooks focused on analysis flow while reusable CZI
loading, manifest building, whole-morph geometry, and manual posterior-click
annotation live in normal Python code under ``scripts/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import heapq
import json
import os
from pathlib import Path
import re
from typing import Any, Optional, Sequence

# czifile cannot be installed beside this project's pinned numpy, and mis-frames odd-sized
# acquisitions. czi_compat presents czifile's interface over libCZI. See src/trunk_morph_ref/czi_compat.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from src.trunk_morph_ref import czi_compat as czifile
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, graph, measure, morphology
import tifffile

try:
    from .pipeline_contract import CURRENT_DATASET_PROFILE
except ImportError:
    from pipeline_contract import CURRENT_DATASET_PROFILE  # type: ignore

DEFAULT_MPLCONFIGDIR = str(
    Path(__file__).resolve().parents[1] / "results" / "tmp" / "mplconfig"
)
os.environ.setdefault("MPLCONFIGDIR", DEFAULT_MPLCONFIGDIR)

import matplotlib.pyplot as plt


MANIFEST_COLUMNS = [
    "image_id",
    "cohort_id",
    "canonical_position",
    "file_id",
    "organoid_id",
    "file_variant_index",
    "file_name",
    "file_stem",
    "filename_suffix",
    "file_path",
    "duplicate_organoid_flag",
    "name_review_flag",
    "name_review_note",
    "source_dir",
    "acquisition_timestamp_utc",
    "acquisition_date",
    "acquisition_batch_label",
    "batch_warning_flag",
    "batch_warning_reason",
    "objective_name",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "pixel_size_z_um",
    "size_c",
    "size_z",
    "size_y",
    "size_x",
    "czi_axes",
    "dtype",
    "channel_names",
    "brightfield_channel_name",
    "dapi_channel_name",
    "mesp2_channel_name",
    "foxf1_channel_name",
    "pax8_channel_name",
]

MANUAL_CONDITION_COLUMNS = [
    "image_id",
    "cohort_id",
    "canonical_position",
    "file_id",
    "file_path",
    "acquisition_date",
    "acquisition_batch_label",
    "condition_label",
    "condition_group",
    "condition_order",
    "include_in_analysis",
    "exclusion_reason",
    "notes",
]

POSTERIOR_COLUMNS = [
    "image_id",
    "cohort_id",
    "canonical_position",
    "file_id",
    "file_path",
    "posterior_click_x_px",
    "posterior_click_y_px",
    "updated_at",
]

MARKER_KEYS = ["mesp2", "foxf1", "pax8"]
MARKER_DISPLAY_NAMES = {
    "mesp2": "MESP2-mCherry",
    "foxf1": "FOXF1-YFP",
    "pax8": "PAX8-647",
}
MARKER_CHANNEL_KEYWORDS = {
    "mesp2": ["mesp2", "mcherry"],
    "foxf1": ["foxf1", "tagyfp", "yfp"],
    "pax8": ["pax8", "alexa fluor 647", "647"],
}
MARKER_OUTLINE_COLORS = {
    "mesp2": "#ff4040",
    "foxf1": "#ffd84a",
    "pax8": "#36d7ff",
}

DEFAULT_COHORT_ID = CURRENT_DATASET_PROFILE.cohort_id

MASK_REVIEW_DECISION_KEYS = [
    "accepted_as_is",
    "needs_manual_outline",
    "exclude_from_analysis",
    "notes",
]

@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    cohort_id: str
    canonical_position: str
    file_id: int
    file_path: str
    acquisition_date: str = ""
    acquisition_batch_label: str = ""


@dataclass
class CziStack:
    path: Path
    data_czyx: np.ndarray
    channels: list[str]
    metadata: dict[str, Any]
    scale_um: dict[str, float]
    acquisition_timestamp_utc: str
    objective_name: str
    axes: str = "CZYX"


_LEADING_INT_STEM_RE = re.compile(r"^(?P<base>\d+)(?P<suffix>.*)$")


def _ensure_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _resolve_local_path(path_like: str | Path, root: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (Path(root) / path)


def _parse_manifest_stem(path: Path) -> tuple[int, str] | None:
    match = _LEADING_INT_STEM_RE.match(path.stem)
    if match is None:
        return None
    organoid_id = int(match.group("base"))
    suffix = str(match.group("suffix") or "").lstrip("-_ ").strip()
    return (organoid_id, suffix)


def _safe_int_stem(path: Path) -> tuple[int, str]:
    parsed = _parse_manifest_stem(path)
    if parsed is None:
        return (10**9, path.name)
    return (int(parsed[0]), path.name)


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if "Name" in value and value["Name"] is not None:
            return str(value["Name"])
        if "Id" in value and value["Id"] is not None:
            return str(value["Id"])
    return str(value)


def _name_review_note(
    organoid_id: int,
    filename_suffix: str,
    duplicate_organoid_count: int,
) -> str:
    notes: list[str] = []
    normalized = filename_suffix.replace("-", " ").replace("_", " ").strip().lower()
    if filename_suffix:
        notes.append(f"descriptive filename suffix: {filename_suffix}")
    if duplicate_organoid_count > 1:
        notes.append(
            f"{duplicate_organoid_count} files share organoid_id {int(organoid_id)}; review duplicate acquisition handling"
        )
    if "no pax8" in normalized:
        notes.append("filename indicates no PAX8 stain")
    if "phalloidin" in normalized:
        notes.append("filename indicates 647 channel may be phalloidin instead of PAX8")
    if "previous stain" in normalized:
        notes.append("filename indicates prior stain / potentially different exposure")
    if "direction" in normalized:
        notes.append("filename indicates alternate acquisition direction")
    return "; ".join(notes)


def extract_channels(metadata: dict[str, Any], c_count: int | None = None) -> list[str]:
    dims = (
        metadata.get("ImageDocument", {})
        .get("Metadata", {})
        .get("Information", {})
        .get("Image", {})
        .get("Dimensions", {})
    )
    channel_obj = dims.get("Channels", {}).get("Channel")
    channels = _ensure_list(channel_obj)
    names: list[str] = []
    for idx, ch in enumerate(channels):
        if isinstance(ch, dict):
            names.append(
                str(
                    ch.get("ShortName")
                    or ch.get("Name")
                    or ch.get("Fluor")
                    or ch.get("DyeName")
                    or f"Channel_{idx}"
                )
            )
        else:
            names.append(f"Channel_{idx}")
    if c_count is not None and len(names) < int(c_count):
        names.extend(f"Channel_{idx}" for idx in range(len(names), int(c_count)))
    return names[: int(c_count)] if c_count is not None else names


def extract_scale_um(metadata: dict[str, Any]) -> dict[str, float]:
    scale = {"X": np.nan, "Y": np.nan, "Z": np.nan}
    items = (
        metadata.get("ImageDocument", {})
        .get("Metadata", {})
        .get("Scaling", {})
        .get("Items", {})
        .get("Distance")
    )
    for item in _ensure_list(items):
        if not isinstance(item, dict):
            continue
        axis = str(item.get("Id", "")).upper()
        value = item.get("Value")
        try:
            um = float(value) * 1e6
        except Exception:
            continue
        if axis in scale:
            scale[axis] = um
    return scale


def extract_objective_name(metadata: dict[str, Any]) -> str:
    scaling = metadata.get("ImageDocument", {}).get("Metadata", {}).get("Scaling", {})
    autoscaling = scaling.get("AutoScaling", {})
    if isinstance(autoscaling, dict):
        name = autoscaling.get("ObjectiveName")
        if name:
            return str(name)

    objective = (
        metadata.get("ImageDocument", {})
        .get("Metadata", {})
        .get("Information", {})
        .get("Image", {})
        .get("ObjectiveSettings", {})
    )
    if isinstance(objective, dict):
        ref = objective.get("ObjectiveRef")
        return _string_or_empty(ref or objective.get("Name"))
    return ""


def extract_acquisition_timestamp_utc(metadata: dict[str, Any]) -> str:
    raw = (
        metadata.get("ImageDocument", {})
        .get("Metadata", {})
        .get("Information", {})
        .get("Image", {})
        .get("AcquisitionDateAndTime")
    )
    if raw is None or str(raw).strip() == "":
        return ""
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return str(raw)
    return ts.isoformat()


def normalize_to_czyx(array: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Collapse an arbitrary CZI array into canonical ``(C, Z, Y, X)`` order."""
    out = np.asarray(array)
    out_axes = str(axes)

    changed = True
    while changed:
        changed = False
        for axis_index, (axis_name, axis_size) in enumerate(zip(out_axes, out.shape)):
            if axis_size == 1 and axis_name not in {"Y", "X"}:
                out = np.take(out, indices=0, axis=axis_index)
                out_axes = out_axes[:axis_index] + out_axes[axis_index + 1 :]
                changed = True
                break

    if "C" not in out_axes or "Y" not in out_axes or "X" not in out_axes:
        raise RuntimeError(f"Expected axes to include C/Y/X, got {out_axes}")

    if "Z" not in out_axes:
        out = np.expand_dims(out, axis=1)
        out_axes = "CZ" + out_axes[1:]
    else:
        z_index = out_axes.index("Z")
        out = np.moveaxis(out, z_index, 1)
        out_axes = "CZ" + out_axes[1:z_index] + out_axes[z_index + 1 :]

    for axis_name in list(out_axes):
        if axis_name in {"C", "Z", "Y", "X"}:
            continue
        axis_index = out_axes.index(axis_name)
        out = out.max(axis=axis_index)
        out_axes = out_axes.replace(axis_name, "")

    perm = [
        out_axes.index("C"),
        out_axes.index("Z"),
        out_axes.index("Y"),
        out_axes.index("X"),
    ]
    out = np.transpose(out, axes=perm)
    return out.astype(np.float32), "CZYX"


def find_channel_index(channels: Sequence[str], aliases: Sequence[str]) -> Optional[int]:
    if not channels:
        return None
    normalized = [str(ch).strip().lower() for ch in channels]
    for alias in aliases:
        alias_norm = str(alias).strip().lower()
        for idx, channel_name in enumerate(normalized):
            if alias_norm in channel_name:
                return int(idx)
    return None


def load_czi_stack(path: Path) -> CziStack:
    with czifile.CziFile(path) as czi:
        raw = czi.asarray()
        raw_axes = str(czi.axes)
        metadata = czi.metadata(raw=False)
        dtype = str(np.dtype(raw.dtype).name)

    czyx, _ = normalize_to_czyx(raw, raw_axes)
    channels = extract_channels(metadata, c_count=czyx.shape[0])
    scale_um = extract_scale_um(metadata)
    acquisition_timestamp_utc = extract_acquisition_timestamp_utc(metadata)
    objective_name = extract_objective_name(metadata)

    stack = CziStack(
        path=Path(path),
        data_czyx=czyx,
        channels=list(channels),
        metadata=metadata,
        scale_um=scale_um,
        acquisition_timestamp_utc=acquisition_timestamp_utc,
        objective_name=objective_name,
    )
    stack.metadata["_dtype_name"] = dtype
    return stack


def center_crop_bounds(
    shape_yx: tuple[int, int],
    crop_fraction: float = 0.5,
    min_side_px: int = 64,
) -> tuple[int, int, int, int]:
    height, width = int(shape_yx[0]), int(shape_yx[1])
    crop_h = max(int(min_side_px), int(round(height * float(crop_fraction))))
    crop_w = max(int(min_side_px), int(round(width * float(crop_fraction))))
    crop_h = min(crop_h, height)
    crop_w = min(crop_w, width)
    y0 = max(0, (height - crop_h) // 2)
    x0 = max(0, (width - crop_w) // 2)
    return y0, y0 + crop_h, x0, x0 + crop_w


def focus_score_for_slice(
    image: np.ndarray,
    crop_fraction: float = 0.5,
    min_side_px: int = 64,
) -> float:
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={img.shape}")
    y0, y1, x0, x1 = center_crop_bounds(
        shape_yx=img.shape,
        crop_fraction=crop_fraction,
        min_side_px=min_side_px,
    )
    crop = img[y0:y1, x0:x1].astype(np.float32)
    crop = crop - float(crop.mean())
    grad = filters.sobel(crop)
    return float(np.mean(grad**2))


def choose_focus_z_index(
    stack_zyx: np.ndarray,
    crop_fraction: float = 0.5,
    min_side_px: int = 64,
) -> dict[str, Any]:
    stack = np.asarray(stack_zyx, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Expected ZYX stack, got shape={stack.shape}")

    scores: list[float] = []
    best: Optional[dict[str, Any]] = None
    for z_idx in range(int(stack.shape[0])):
        score = focus_score_for_slice(
            image=stack[z_idx],
            crop_fraction=crop_fraction,
            min_side_px=min_side_px,
        )
        scores.append(float(score))
        candidate = {
            "focus_z_index": int(z_idx),
            "focus_score": float(score),
        }
        if best is None or candidate["focus_score"] > best["focus_score"]:
            best = candidate

    if best is None:
        raise RuntimeError("Failed to choose focus z index.")
    best["focus_scores"] = np.asarray(scores, dtype=np.float32)
    best["z_count"] = int(stack.shape[0])
    return best


def project_array(array: np.ndarray, projection: str = "max") -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape={arr.shape}")
    if projection == "max":
        return arr.max(axis=0).astype(np.float32)
    if projection == "mean":
        return arr.mean(axis=0).astype(np.float32)
    if projection == "mid":
        return arr[arr.shape[0] // 2].astype(np.float32)
    raise ValueError(f"Unknown projection={projection!r}")


def load_plane_channels(
    path: Path,
    z_index: int,
) -> dict[str, Any]:
    stack = load_czi_stack(Path(path))
    channels = stack.channels
    bf_idx = find_channel_index(channels, ["bright"])
    dapi_idx = find_channel_index(channels, ["dapi"])
    mesp2_idx = find_channel_index(channels, ["mesp2", "mcherry"])
    foxf1_idx = find_channel_index(channels, ["foxf1", "tagyfp", "yfp"])
    pax8_idx = find_channel_index(channels, ["pax8", "alexa fluor 647", "647"])

    z_idx = int(np.clip(int(z_index), 0, stack.data_czyx.shape[1] - 1))

    def _channel_plane(idx: Optional[int]) -> Optional[np.ndarray]:
        if idx is None:
            return None
        return np.asarray(stack.data_czyx[idx, z_idx], dtype=np.float32)

    bf = _channel_plane(bf_idx)
    dapi = _channel_plane(dapi_idx)
    mesp2 = _channel_plane(mesp2_idx)
    foxf1 = _channel_plane(foxf1_idx)
    pax8 = _channel_plane(pax8_idx)
    return {
        "stack": stack,
        "selected_z": int(z_idx),
        "brightfield": bf,
        "dapi": dapi,
        "mesp2": mesp2,
        "foxf1": foxf1,
        "pax8": pax8,
        "marker_rgb": build_marker_rgb(mesp2=mesp2, foxf1=foxf1, pax8=pax8),
        "overlay_rgb": build_overlay_rgb(dapi=dapi, mesp2=mesp2, foxf1=foxf1, pax8=pax8),
        "channel_index_map": {
            "brightfield": bf_idx,
            "dapi": dapi_idx,
            "mesp2": mesp2_idx,
            "foxf1": foxf1_idx,
            "pax8": pax8_idx,
        },
    }


def load_projected_channels(
    path: Path,
    projection: str = "max",
) -> dict[str, Any]:
    stack = load_czi_stack(Path(path))
    channels = stack.channels
    bf_idx = find_channel_index(channels, ["bright"])
    dapi_idx = find_channel_index(channels, ["dapi"])
    mesp2_idx = find_channel_index(channels, ["mesp2", "mcherry"])
    foxf1_idx = find_channel_index(channels, ["foxf1", "tagyfp", "yfp"])
    pax8_idx = find_channel_index(channels, ["pax8", "alexa fluor 647", "647"])

    def _channel_projection(idx: Optional[int]) -> Optional[np.ndarray]:
        if idx is None:
            return None
        return project_array(stack.data_czyx[idx], projection=projection)

    bf = _channel_projection(bf_idx)
    dapi = _channel_projection(dapi_idx)
    mesp2 = _channel_projection(mesp2_idx)
    foxf1 = _channel_projection(foxf1_idx)
    pax8 = _channel_projection(pax8_idx)
    return {
        "stack": stack,
        "brightfield": bf,
        "dapi": dapi,
        "mesp2": mesp2,
        "foxf1": foxf1,
        "pax8": pax8,
        "marker_rgb": build_marker_rgb(mesp2=mesp2, foxf1=foxf1, pax8=pax8),
        "overlay_rgb": build_overlay_rgb(dapi=dapi, mesp2=mesp2, foxf1=foxf1, pax8=pax8),
        "channel_index_map": {
            "brightfield": bf_idx,
            "dapi": dapi_idx,
            "mesp2": mesp2_idx,
            "foxf1": foxf1_idx,
            "pax8": pax8_idx,
        },
    }


def robust_rescale(
    image: Optional[np.ndarray],
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    if image is None:
        return np.zeros((1, 1), dtype=np.float32)
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.quantile(arr[finite], [p_low / 100.0, p_high / 100.0])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def load_binary_mask(path: Path) -> np.ndarray:
    mask = tifffile.imread(Path(path))
    return np.asarray(mask, dtype=np.float32) > 0


def build_marker_rgb(
    mesp2: Optional[np.ndarray],
    foxf1: Optional[np.ndarray],
    pax8: Optional[np.ndarray],
) -> np.ndarray:
    channels = [mesp2, foxf1, pax8]
    shape = None
    for image in channels:
        if image is not None:
            shape = np.asarray(image).shape
            break
    if shape is None:
        return np.zeros((1, 1, 3), dtype=np.float32)
    rgb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    if mesp2 is not None:
        rgb[..., 0] = robust_rescale(mesp2)
    if foxf1 is not None:
        rgb[..., 1] = robust_rescale(foxf1)
    if pax8 is not None:
        rgb[..., 2] = robust_rescale(pax8)
    return np.clip(rgb, 0.0, 1.0)


def build_overlay_rgb(
    dapi: Optional[np.ndarray],
    mesp2: Optional[np.ndarray],
    foxf1: Optional[np.ndarray],
    pax8: Optional[np.ndarray],
) -> np.ndarray:
    marker_rgb = build_marker_rgb(mesp2=mesp2, foxf1=foxf1, pax8=pax8)
    if dapi is None:
        return marker_rgb
    dapi_gray = robust_rescale(dapi)
    base = np.stack([dapi_gray, dapi_gray, dapi_gray], axis=-1)
    out = 0.35 * base + 0.85 * marker_rgb
    return np.clip(out, 0.0, 1.0)


def segment_whole_morph(
    dapi_projection: np.ndarray,
    gaussian_sigma: float = 2.0,
    baseline_percentile: float = 5.0,
    threshold_scale: float = 0.85,
    min_size_px: int = 4000,
    opening_radius_px: int = 3,
    closing_radius_px: int = 7,
    dilation_radius_px: int = 2,
) -> np.ndarray:
    mask, _ = segment_whole_morph_with_debug(
        dapi_projection=dapi_projection,
        gaussian_sigma=gaussian_sigma,
        baseline_percentile=baseline_percentile,
        threshold_scale=threshold_scale,
        min_size_px=min_size_px,
        opening_radius_px=opening_radius_px,
        closing_radius_px=closing_radius_px,
        dilation_radius_px=dilation_radius_px,
    )
    return mask


def segment_whole_morph_with_debug(
    dapi_projection: np.ndarray,
    gaussian_sigma: float = 2.0,
    baseline_percentile: float = 5.0,
    threshold_scale: float = 0.85,
    min_size_px: int = 4000,
    opening_radius_px: int = 3,
    closing_radius_px: int = 7,
    dilation_radius_px: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = np.asarray(dapi_projection, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        empty = np.zeros_like(image, dtype=bool)
        return empty, {
            "baseline_value": np.nan,
            "flat": np.zeros_like(image, dtype=np.float32),
            "smoothed": np.zeros_like(image, dtype=np.float32),
            "mask_threshold": empty.astype(np.uint8),
            "mask_after_morphology": empty.astype(np.uint8),
            "labels_raw": np.zeros_like(image, dtype=np.int32),
            "threshold_info": {
                "method": "otsu",
                "candidate_thresholds": {"otsu": np.nan},
                "raw_threshold": np.nan,
                "threshold_scale": float(threshold_scale),
                "threshold_value": np.nan,
                "selected_variant": "empty_input",
            },
            "n_raw_components": 0,
            "n_kept_components": 0,
        }

    baseline = float(np.nanpercentile(image[finite], float(baseline_percentile)))
    flat = np.clip(image - baseline, a_min=0.0, a_max=None).astype(np.float32)
    smoothed = filters.gaussian(flat, sigma=float(gaussian_sigma), preserve_range=True)
    values = smoothed[np.isfinite(smoothed)]
    if values.size < 64:
        empty = np.zeros_like(image, dtype=bool)
        return empty, {
            "baseline_value": baseline,
            "flat": flat,
            "smoothed": smoothed.astype(np.float32),
            "mask_threshold": empty.astype(np.uint8),
            "mask_after_morphology": empty.astype(np.uint8),
            "labels_raw": np.zeros_like(image, dtype=np.int32),
            "threshold_info": {
                "method": "otsu",
                "candidate_thresholds": {"otsu": np.nan},
                "raw_threshold": np.nan,
                "threshold_scale": float(threshold_scale),
                "threshold_value": np.nan,
                "selected_variant": "too_few_values",
            },
            "n_raw_components": 0,
            "n_kept_components": 0,
        }

    raw_threshold = float(filters.threshold_otsu(values))
    threshold = float(raw_threshold * float(threshold_scale))
    candidate_thresholds: dict[str, float] = {"otsu": raw_threshold}

    mask = smoothed >= threshold
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=int(min_size_px))
    if int(dilation_radius_px) > 0:
        mask = morphology.binary_dilation(mask, footprint=morphology.disk(int(dilation_radius_px)))
    if int(mask.sum()) == 0:
        empty = mask.astype(bool)
        return empty, {
            "baseline_value": baseline,
            "flat": flat,
            "smoothed": smoothed.astype(np.float32),
            "mask_threshold": (smoothed >= threshold).astype(np.uint8),
            "mask_after_morphology": empty.astype(np.uint8),
            "labels_raw": np.zeros_like(image, dtype=np.int32),
            "threshold_info": {
                "method": "otsu",
                "candidate_thresholds": candidate_thresholds,
                "raw_threshold": raw_threshold,
                "threshold_scale": float(threshold_scale),
                "threshold_value": threshold,
                "selected_variant": f"otsu x{float(threshold_scale):.2f}",
            },
            "n_raw_components": 0,
            "n_kept_components": 0,
        }
    mask = morphology.binary_closing(mask, footprint=morphology.disk(int(closing_radius_px)))
    mask = ndi.binary_fill_holes(mask)
    mask_after_morphology = mask.astype(bool)

    labels = measure.label(mask)
    if labels.max() <= 0:
        empty = np.zeros_like(mask, dtype=bool)
        return empty, {
            "baseline_value": baseline,
            "flat": flat,
            "smoothed": smoothed.astype(np.float32),
            "mask_threshold": (smoothed >= threshold).astype(np.uint8),
            "mask_after_morphology": mask_after_morphology.astype(np.uint8),
            "labels_raw": labels.astype(np.int32),
            "threshold_info": {
                "method": "otsu",
                "candidate_thresholds": candidate_thresholds,
                "raw_threshold": raw_threshold,
                "threshold_scale": float(threshold_scale),
                "threshold_value": threshold,
                "selected_variant": f"otsu x{float(threshold_scale):.2f}",
            },
            "n_raw_components": 0,
            "n_kept_components": 0,
        }
    props = measure.regionprops(labels)
    largest = max(props, key=lambda prop: float(prop.area))
    mask = labels == largest.label
    mask = morphology.binary_opening(mask, footprint=morphology.disk(int(opening_radius_px)))
    mask = morphology.binary_closing(mask, footprint=morphology.disk(int(closing_radius_px)))
    mask = ndi.binary_fill_holes(mask)
    final_mask = mask.astype(bool)
    return final_mask, {
        "baseline_value": baseline,
        "flat": flat,
        "smoothed": smoothed.astype(np.float32),
        "mask_threshold": (smoothed >= threshold).astype(np.uint8),
        "mask_after_morphology": mask_after_morphology.astype(np.uint8),
        "labels_raw": labels.astype(np.int32),
        "threshold_info": {
            "method": "otsu",
            "candidate_thresholds": candidate_thresholds,
            "raw_threshold": raw_threshold,
            "threshold_scale": float(threshold_scale),
            "threshold_value": threshold,
            "selected_variant": f"otsu x{float(threshold_scale):.2f}",
        },
        "n_raw_components": int(labels.max()),
        "n_kept_components": int(final_mask.any()),
    }


def plot_whole_morph_segmentation_debug_figure(
    brightfield: np.ndarray | None,
    dapi: np.ndarray,
    mask: np.ndarray,
    debug: dict[str, Any],
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 5, figsize=(21, 9))
    axes = axes.ravel()

    bf_display = robust_rescale(brightfield) if brightfield is not None else np.zeros_like(dapi, dtype=np.float32)
    dapi_display = robust_rescale(dapi)
    flat_display = robust_rescale(debug.get("flat"))
    smoothed_display = robust_rescale(debug.get("smoothed"))

    axes[0].imshow(bf_display, cmap="gray")
    axes[0].set_title("Brightfield")
    axes[0].axis("off")

    axes[1].imshow(dapi_display, cmap="gray")
    axes[1].set_title("DAPI raw")
    axes[1].axis("off")

    axes[2].imshow(flat_display, cmap="gray")
    axes[2].set_title(f"Flattened DAPI (raw - {debug.get('baseline_value', np.nan):.1f})")
    axes[2].axis("off")

    axes[3].imshow(smoothed_display, cmap="gray")
    axes[3].set_title("Smoothed flattened DAPI")
    axes[3].axis("off")

    axes[4].imshow(debug.get("mask_threshold"), cmap="gray")
    axes[4].set_title(
        f"Threshold mask (thr={debug.get('threshold_info', {}).get('threshold_value', np.nan):.1f})"
    )
    axes[4].axis("off")

    axes[5].imshow(bf_display, cmap="gray")
    plot_mask_outline(axes[5], mask, color="yellow", linewidth=1.2)
    axes[5].set_title("BF + final outline")
    axes[5].axis("off")

    axes[6].imshow(dapi_display, cmap="gray")
    plot_mask_outline(axes[6], mask, color="yellow", linewidth=1.2)
    axes[6].set_title("DAPI + final outline")
    axes[6].axis("off")

    axes[7].imshow(debug.get("mask_after_morphology"), cmap="gray")
    axes[7].set_title("Mask after morphology")
    axes[7].axis("off")

    raw_overlay = np.asarray(debug.get("labels_raw", np.zeros_like(mask, dtype=np.int32)), dtype=np.int32)
    raw_rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    if raw_overlay.max() > 0:
        raw_rgb = plt.cm.nipy_spectral((raw_overlay % 255) / 255.0)[..., :3]
    axes[8].imshow(dapi_display, cmap="gray")
    axes[8].imshow(raw_rgb, alpha=(raw_overlay > 0).astype(np.float32) * 0.35)
    plot_mask_outline(axes[8], mask, color="yellow", linewidth=1.2)
    axes[8].set_title("Raw components + final")
    axes[8].axis("off")

    flat_vals = np.asarray(debug.get("flat", np.zeros_like(dapi)), dtype=np.float32).ravel()
    flat_vals = flat_vals[np.isfinite(flat_vals)]
    axes[9].hist(flat_vals, bins=256, color="#808080")
    threshold_info = debug.get("threshold_info", {})
    candidate_thresholds = threshold_info.get("candidate_thresholds", {})
    otsu_value = float(candidate_thresholds.get("otsu", np.nan))
    selected_value = float(threshold_info.get("threshold_value", np.nan))
    handles = []
    labels = []
    if np.isfinite(otsu_value):
        handles.append(axes[9].axvline(otsu_value, color="#f58518", linestyle="--", linewidth=1.4))
        labels.append("otsu")
    if np.isfinite(selected_value):
        handles.append(axes[9].axvline(selected_value, color="red", linewidth=2.2))
        labels.append(f"selected: {threshold_info.get('selected_variant', 'current')}")
    axes[9].set_title("Flattened DAPI histogram")
    axes[9].set_yscale("log")
    if handles:
        axes[9].legend(handles, labels, fontsize=7, loc="upper right")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def unit_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Cannot normalize zero-length vector")
    return arr / norm


SKELETON_NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _step_distance_rc(a_rc: tuple[int, int], b_rc: tuple[int, int]) -> float:
    return float(np.hypot(float(b_rc[0] - a_rc[0]), float(b_rc[1] - a_rc[1])))


def _skeleton_neighbors(skeleton: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    coords = [tuple(int(v) for v in rc) for rc in np.argwhere(np.asarray(skeleton, dtype=bool))]
    coord_set = set(coords)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, col in coords:
        nbrs = []
        for d_row, d_col in SKELETON_NEIGHBOR_OFFSETS:
            candidate = (row + d_row, col + d_col)
            if candidate in coord_set:
                nbrs.append(candidate)
        neighbors[(row, col)] = nbrs
    return neighbors


def _trace_endpoint_branch(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    endpoint_rc: tuple[int, int],
) -> tuple[list[tuple[int, int]], float, int]:
    path = [endpoint_rc]
    prev = None
    current = endpoint_rc
    length_px = 0.0

    while True:
        current_neighbors = neighbors.get(current, [])
        next_neighbors = [nbr for nbr in current_neighbors if nbr != prev]
        if not next_neighbors:
            break
        if len(next_neighbors) > 1:
            break

        nxt = next_neighbors[0]
        path.append(nxt)
        length_px += _step_distance_rc(current, nxt)
        prev, current = current, nxt
        if len(neighbors.get(current, [])) != 2:
            break

    terminal_degree = len(neighbors.get(current, []))
    return path, float(length_px), int(terminal_degree)


def _prune_skeleton_spurs(skeleton: np.ndarray, max_branch_length_px: float) -> np.ndarray:
    pruned = np.asarray(skeleton, dtype=bool).copy()
    max_branch_length_px = float(max_branch_length_px)

    while True:
        neighbors = _skeleton_neighbors(pruned)
        endpoints = [rc for rc, nbrs in neighbors.items() if len(nbrs) == 1]
        to_remove: set[tuple[int, int]] = set()

        for endpoint_rc in endpoints:
            branch_path, branch_length_px, terminal_degree = _trace_endpoint_branch(neighbors, endpoint_rc)
            if terminal_degree > 2 and branch_length_px <= max_branch_length_px:
                to_remove.update(branch_path[:-1])

        if not to_remove:
            break

        rows = np.array([rc[0] for rc in to_remove], dtype=int)
        cols = np.array([rc[1] for rc in to_remove], dtype=int)
        pruned[rows, cols] = False

    return pruned


def _dijkstra_skeleton(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    start_rc: tuple[int, int],
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, int]]]:
    distances = {start_rc: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, start_rc)]

    while heap:
        current_distance, current_rc = heapq.heappop(heap)
        if current_distance > distances.get(current_rc, np.inf):
            continue

        for neighbor_rc in neighbors.get(current_rc, []):
            step_px = _step_distance_rc(current_rc, neighbor_rc)
            candidate_distance = current_distance + step_px
            if candidate_distance < distances.get(neighbor_rc, np.inf):
                distances[neighbor_rc] = candidate_distance
                previous[neighbor_rc] = current_rc
                heapq.heappush(heap, (candidate_distance, neighbor_rc))

    return distances, previous


def _reconstruct_path_rc(
    previous: dict[tuple[int, int], tuple[int, int]],
    start_rc: tuple[int, int],
    end_rc: tuple[int, int],
) -> list[tuple[int, int]]:
    if start_rc == end_rc:
        return [start_rc]

    path = [end_rc]
    current = end_rc
    while current != start_rc:
        current = previous[current]
        path.append(current)
    path.reverse()
    return path


def _polyline_arc_length_xy(points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[0] == 0:
        return np.array([0.0], dtype=np.float64)
    if points_xy.shape[0] == 1:
        return np.array([0.0], dtype=np.float64)
    step_lengths = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step_lengths)])


def _interpolate_point_along_polyline(points_xy: np.ndarray, target_length_px: float) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    arc_lengths = _polyline_arc_length_xy(points_xy)
    if points_xy.shape[0] == 1 or arc_lengths[-1] <= 0:
        return points_xy[0].copy()

    target = float(np.clip(target_length_px, 0.0, arc_lengths[-1]))
    idx = int(np.searchsorted(arc_lengths, target, side="right"))
    if idx <= 0:
        return points_xy[0].copy()
    if idx >= len(points_xy):
        return points_xy[-1].copy()

    left_length = float(arc_lengths[idx - 1])
    right_length = float(arc_lengths[idx])
    if right_length <= left_length:
        return points_xy[idx].copy()

    alpha = (target - left_length) / (right_length - left_length)
    return (1.0 - alpha) * points_xy[idx - 1] + alpha * points_xy[idx]


def resample_polyline_xy(points_xy: np.ndarray, n_points: int = 101) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    n_points = max(int(n_points), 2)
    if points_xy.ndim != 2 or points_xy.shape[0] == 0:
        raise ValueError("Polyline must be a non-empty Nx2 array")
    if points_xy.shape[0] == 1:
        return np.repeat(points_xy[:1], n_points, axis=0)

    arc_lengths = _polyline_arc_length_xy(points_xy)
    total_length = float(arc_lengths[-1])
    if total_length <= 0:
        return np.repeat(points_xy[:1], n_points, axis=0)

    targets = np.linspace(0.0, total_length, n_points, dtype=np.float64)
    return np.vstack([_interpolate_point_along_polyline(points_xy, t) for t in targets]).astype(np.float64)


def _endpoint_tangent_xy(points_xy: np.ndarray, at_start: bool, min_separation_px: float = 5.0) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.shape[0] < 2:
        return np.array([np.nan, np.nan], dtype=np.float64)

    if at_start:
        ref = points_xy[0]
        candidate_indices = range(1, points_xy.shape[0])
    else:
        ref = points_xy[-1]
        candidate_indices = range(points_xy.shape[0] - 2, -1, -1)

    for idx in candidate_indices:
        vec = points_xy[idx] - ref
        if float(np.linalg.norm(vec)) >= float(min_separation_px):
            return unit_vector(vec)

    fallback_vec = points_xy[1] - points_xy[0] if at_start else points_xy[-2] - points_xy[-1]
    return unit_vector(fallback_vec)


def _rotate_vector_xy(vec_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    vec_xy = unit_vector(vec_xy)
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return unit_vector(rot @ vec_xy)


def _mask_contains_xy(mask: np.ndarray, point_xy: np.ndarray) -> bool:
    point_xy = np.asarray(point_xy, dtype=np.float64)
    row = int(np.round(point_xy[1]))
    col = int(np.round(point_xy[0]))
    if row < 0 or row >= mask.shape[0] or col < 0 or col >= mask.shape[1]:
        return False
    return bool(mask[row, col])


def _march_to_mask_boundary(
    mask: np.ndarray,
    start_xy: np.ndarray,
    direction_xy: np.ndarray,
    step_px: float = 0.5,
) -> tuple[np.ndarray, float]:
    start_xy = np.asarray(start_xy, dtype=np.float64)
    direction_xy = unit_vector(direction_xy)
    max_distance_px = float(np.hypot(mask.shape[0], mask.shape[1])) + 5.0

    last_inside_xy = start_xy.copy()
    distance_px = 0.0
    while distance_px <= max_distance_px:
        candidate_xy = start_xy + direction_xy * distance_px
        if not _mask_contains_xy(mask, candidate_xy):
            break
        last_inside_xy = candidate_xy
        distance_px += float(step_px)

    return last_inside_xy.astype(np.float64), float(np.linalg.norm(last_inside_xy - start_xy))


def extend_endpoint_to_tip(
    mask: np.ndarray,
    endpoint_xy: np.ndarray,
    inward_tangent_xy: np.ndarray,
    step_px: float = 0.5,
) -> tuple[np.ndarray, float]:
    endpoint_xy = np.asarray(endpoint_xy, dtype=np.float64)
    outward_xy = unit_vector(-np.asarray(inward_tangent_xy, dtype=np.float64))
    tip_xy, distance_px = _march_to_mask_boundary(mask, endpoint_xy, outward_xy, step_px=step_px)
    return tip_xy.astype(np.float64), float(distance_px)


def orient_polyline_to_reference(points_xy: np.ndarray, reference_xy: np.ndarray, n_compare: int = 61) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    reference_xy = np.asarray(reference_xy, dtype=np.float64)
    if points_xy.ndim != 2 or reference_xy.ndim != 2:
        raise ValueError("Both polyline inputs must be Nx2 arrays")

    n_compare = max(int(n_compare), 11)
    compare_target = min(n_compare, max(points_xy.shape[0], 2), max(reference_xy.shape[0], 2))
    points_cmp = resample_polyline_xy(points_xy, n_points=compare_target)
    reference_cmp = resample_polyline_xy(reference_xy, n_points=compare_target)

    score_forward = float(np.mean(np.linalg.norm(points_cmp - reference_cmp, axis=1)))
    score_reverse = float(np.mean(np.linalg.norm(points_cmp[::-1] - reference_cmp, axis=1)))
    if score_reverse < score_forward:
        return points_xy[::-1].copy()
    return points_xy.copy()


def project_points_inside_mask(mask: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[0] == 0:
        raise ValueError("Points must be a non-empty Nx2 array")

    if mask.sum() == 0:
        return points_xy.copy()

    _, nearest_inside_idx = ndi.distance_transform_edt(~mask, return_indices=True)
    projected_xy = points_xy.copy()

    for idx, point_xy in enumerate(projected_xy):
        row = int(np.clip(np.round(point_xy[1]), 0, mask.shape[0] - 1))
        col = int(np.clip(np.round(point_xy[0]), 0, mask.shape[1] - 1))
        if mask[row, col]:
            continue
        nearest_row = int(nearest_inside_idx[0, row, col])
        nearest_col = int(nearest_inside_idx[1, row, col])
        projected_xy[idx] = np.array([nearest_col, nearest_row], dtype=np.float64)

    return projected_xy.astype(np.float64)


def centerline_dict_from_polyline(
    points_xy: np.ndarray,
    mask: np.ndarray | None = None,
    method: str = "polyline",
) -> dict[str, Any]:
    centerline_xy = np.asarray(points_xy, dtype=np.float64)
    if centerline_xy.ndim != 2 or centerline_xy.shape[0] < 2:
        raise ValueError("Polyline centerline must contain at least two points")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        centerline_xy = project_points_inside_mask(mask, centerline_xy)

    base_endpoint_a_xy = centerline_xy[0].astype(np.float64).copy()
    base_endpoint_b_xy = centerline_xy[-1].astype(np.float64).copy()
    start_tangent_xy = _endpoint_tangent_xy(centerline_xy, at_start=True)
    end_tangent_xy = _endpoint_tangent_xy(centerline_xy, at_start=False)

    tip_extension_a_px = 0.0
    tip_extension_b_px = 0.0
    if mask is not None:
        tip_a_xy, tip_extension_a_px = extend_endpoint_to_tip(mask, centerline_xy[0], start_tangent_xy)
        tip_b_xy, tip_extension_b_px = extend_endpoint_to_tip(mask, centerline_xy[-1], end_tangent_xy)
        if tip_extension_a_px > 0.5:
            centerline_xy = np.vstack([tip_a_xy, centerline_xy])
        if tip_extension_b_px > 0.5:
            centerline_xy = np.vstack([centerline_xy, tip_b_xy])

    arc_lengths = _polyline_arc_length_xy(centerline_xy)
    midpoint_xy = _interpolate_point_along_polyline(centerline_xy, 0.5 * float(arc_lengths[-1]))
    endpoint_a_xy = centerline_xy[0].astype(np.float64)
    endpoint_b_xy = centerline_xy[-1].astype(np.float64)
    chord_length_px = float(np.linalg.norm(endpoint_b_xy - endpoint_a_xy))

    return {
        "centerline_xy": centerline_xy.astype(np.float64),
        "midpoint_xy": midpoint_xy.astype(np.float64),
        "endpoint_a_xy": endpoint_a_xy.astype(np.float64),
        "endpoint_b_xy": endpoint_b_xy.astype(np.float64),
        "base_endpoint_a_xy": base_endpoint_a_xy.astype(np.float64),
        "base_endpoint_b_xy": base_endpoint_b_xy.astype(np.float64),
        "start_tangent_xy": _endpoint_tangent_xy(centerline_xy, at_start=True),
        "end_tangent_xy": _endpoint_tangent_xy(centerline_xy, at_start=False),
        "length_px": float(arc_lengths[-1]),
        "chord_length_px": chord_length_px,
        "tortuosity": float(arc_lengths[-1] / chord_length_px) if chord_length_px > 0 else np.nan,
        "method": str(method),
        "n_points": int(centerline_xy.shape[0]),
        "prune_length_px": np.nan,
        "skeleton_px": np.nan,
        "tip_extension_a_px": float(tip_extension_a_px),
        "tip_extension_b_px": float(tip_extension_b_px),
    }


def consensus_centerline_from_paths(
    centerlines_xy: Sequence[np.ndarray],
    reference_mask: np.ndarray | None = None,
    n_points: int = 121,
    average_mode: str = "mean",
) -> dict[str, Any]:
    valid_paths = [np.asarray(path, dtype=np.float64) for path in centerlines_xy if np.asarray(path).shape[0] >= 2]
    if not valid_paths:
        raise ValueError("No valid centerlines were provided for consensus estimation")

    lengths = [float(_polyline_arc_length_xy(path)[-1]) for path in valid_paths]
    reference_xy = valid_paths[int(np.argmax(lengths))]
    oriented_paths = [orient_polyline_to_reference(path, reference_xy) for path in valid_paths]
    resampled = np.stack([resample_polyline_xy(path, n_points=n_points) for path in oriented_paths], axis=0)

    if str(average_mode).lower() == "median":
        consensus_xy = np.median(resampled, axis=0)
        method = f"consensus_median_n{len(oriented_paths)}"
    else:
        consensus_xy = np.mean(resampled, axis=0)
        method = f"consensus_mean_n{len(oriented_paths)}"

    return centerline_dict_from_polyline(consensus_xy, mask=reference_mask, method=method)


def centerline_from_mask(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask)
    if len(xs) < 10:
        raise ValueError("Mask too small for centerline estimation")

    labels = measure.label(mask)
    props = measure.regionprops(labels)
    region = max(props, key=lambda prop: float(prop.area))
    prune_length_px = float(np.clip(0.15 * float(region.minor_axis_length), 10.0, 60.0))

    raw_skeleton = morphology.medial_axis(mask)
    pruned_skeleton = _prune_skeleton_spurs(raw_skeleton, max_branch_length_px=prune_length_px)
    neighbors = _skeleton_neighbors(pruned_skeleton)
    endpoints = [rc for rc, nbrs in neighbors.items() if len(nbrs) == 1]

    if len(endpoints) < 2:
        raw_skeleton = morphology.skeletonize(mask)
        pruned_skeleton = _prune_skeleton_spurs(raw_skeleton, max_branch_length_px=prune_length_px)
        neighbors = _skeleton_neighbors(pruned_skeleton)
        endpoints = [rc for rc, nbrs in neighbors.items() if len(nbrs) == 1]

    if len(endpoints) < 2:
        centroid_xy, axis_u_xy, boundary_xy = major_axis_from_mask(mask)
        endpoint_a_xy, endpoint_b_xy = choose_axis_endpoints(boundary_xy, centroid_xy, axis_u_xy)
        base_endpoint_a_xy = np.asarray(endpoint_a_xy, dtype=np.float64).copy()
        base_endpoint_b_xy = np.asarray(endpoint_b_xy, dtype=np.float64).copy()
        centerline_xy = np.vstack([endpoint_a_xy, centroid_xy, endpoint_b_xy]).astype(np.float64)
        start_tangent_xy = _endpoint_tangent_xy(centerline_xy, at_start=True)
        end_tangent_xy = _endpoint_tangent_xy(centerline_xy, at_start=False)
        tip_a_xy, tip_extension_a_px = extend_endpoint_to_tip(mask, endpoint_a_xy, start_tangent_xy)
        tip_b_xy, tip_extension_b_px = extend_endpoint_to_tip(mask, endpoint_b_xy, end_tangent_xy)
        if tip_extension_a_px > 0.5:
            centerline_xy = np.vstack([tip_a_xy, centerline_xy])
        if tip_extension_b_px > 0.5:
            centerline_xy = np.vstack([centerline_xy, tip_b_xy])
        arc_lengths = _polyline_arc_length_xy(centerline_xy)
        midpoint_xy = _interpolate_point_along_polyline(centerline_xy, 0.5 * float(arc_lengths[-1]))
        endpoint_a_xy = centerline_xy[0].astype(np.float64)
        endpoint_b_xy = centerline_xy[-1].astype(np.float64)
        chord_length_px = float(np.linalg.norm(endpoint_b_xy - endpoint_a_xy))
        return {
            "centerline_xy": centerline_xy,
            "midpoint_xy": midpoint_xy.astype(np.float64),
            "endpoint_a_xy": endpoint_a_xy.astype(np.float64),
            "endpoint_b_xy": endpoint_b_xy.astype(np.float64),
            "base_endpoint_a_xy": base_endpoint_a_xy.astype(np.float64),
            "base_endpoint_b_xy": base_endpoint_b_xy.astype(np.float64),
            "start_tangent_xy": _endpoint_tangent_xy(centerline_xy, at_start=True),
            "end_tangent_xy": _endpoint_tangent_xy(centerline_xy, at_start=False),
            "length_px": float(arc_lengths[-1]),
            "chord_length_px": chord_length_px,
            "tortuosity": float(arc_lengths[-1] / chord_length_px) if chord_length_px > 0 else np.nan,
            "method": "major_axis_fallback_tip_extended",
            "n_points": int(centerline_xy.shape[0]),
            "prune_length_px": prune_length_px,
            "skeleton_px": int(pruned_skeleton.sum()),
            "tip_extension_a_px": float(tip_extension_a_px),
            "tip_extension_b_px": float(tip_extension_b_px),
        }

    best_start = endpoints[0]
    best_end = endpoints[1]
    best_previous: dict[tuple[int, int], tuple[int, int]] | None = None
    best_length_px = -np.inf

    for start_rc in endpoints:
        distances, previous = _dijkstra_skeleton(neighbors, start_rc)
        for end_rc in endpoints:
            if end_rc == start_rc or end_rc not in distances:
                continue
            if distances[end_rc] > best_length_px:
                best_start = start_rc
                best_end = end_rc
                best_previous = previous
                best_length_px = float(distances[end_rc])

    if best_previous is None:
        raise RuntimeError("Failed to reconstruct centerline path")

    path_rc = _reconstruct_path_rc(best_previous, best_start, best_end)
    distance_to_boundary = ndi.distance_transform_edt(mask)
    route_cost = np.full(mask.shape, 1e9, dtype=np.float64)
    route_cost[mask] = 1.0 / np.maximum(distance_to_boundary[mask], 1.0)

    try:
        routed_path_rc, _ = graph.route_through_array(
            route_cost,
            start=best_start,
            end=best_end,
            fully_connected=True,
            geometric=True,
        )
        if len(routed_path_rc) >= 2:
            path_rc = [tuple(int(v) for v in rc) for rc in routed_path_rc]
            centerline_method = "medial_axis_endpoint_route"
        else:
            centerline_method = "medial_axis_longest_path"
    except Exception:
        centerline_method = "medial_axis_longest_path"

    centerline_xy = np.column_stack(
        [
            np.array([rc[1] for rc in path_rc], dtype=np.float64),
            np.array([rc[0] for rc in path_rc], dtype=np.float64),
        ]
    )
    base_endpoint_a_xy = centerline_xy[0].astype(np.float64).copy()
    base_endpoint_b_xy = centerline_xy[-1].astype(np.float64).copy()
    start_tangent_xy = _endpoint_tangent_xy(centerline_xy, at_start=True)
    end_tangent_xy = _endpoint_tangent_xy(centerline_xy, at_start=False)
    tip_a_xy, tip_extension_a_px = extend_endpoint_to_tip(mask, centerline_xy[0], start_tangent_xy)
    tip_b_xy, tip_extension_b_px = extend_endpoint_to_tip(mask, centerline_xy[-1], end_tangent_xy)
    if tip_extension_a_px > 0.5:
        centerline_xy = np.vstack([tip_a_xy, centerline_xy])
    if tip_extension_b_px > 0.5:
        centerline_xy = np.vstack([centerline_xy, tip_b_xy])
    arc_lengths = _polyline_arc_length_xy(centerline_xy)
    midpoint_xy = _interpolate_point_along_polyline(centerline_xy, 0.5 * float(arc_lengths[-1]))
    endpoint_a_xy = centerline_xy[0].astype(np.float64)
    endpoint_b_xy = centerline_xy[-1].astype(np.float64)
    chord_length_px = float(np.linalg.norm(endpoint_b_xy - endpoint_a_xy))
    return {
        "centerline_xy": centerline_xy,
        "midpoint_xy": midpoint_xy.astype(np.float64),
        "endpoint_a_xy": endpoint_a_xy,
        "endpoint_b_xy": endpoint_b_xy,
        "base_endpoint_a_xy": base_endpoint_a_xy,
        "base_endpoint_b_xy": base_endpoint_b_xy,
        "start_tangent_xy": _endpoint_tangent_xy(centerline_xy, at_start=True),
        "end_tangent_xy": _endpoint_tangent_xy(centerline_xy, at_start=False),
        "length_px": float(arc_lengths[-1]),
        "chord_length_px": chord_length_px,
        "tortuosity": float(arc_lengths[-1] / chord_length_px) if chord_length_px > 0 else np.nan,
        "method": centerline_method,
        "n_points": int(centerline_xy.shape[0]),
        "prune_length_px": prune_length_px,
        "skeleton_px": int(pruned_skeleton.sum()),
        "tip_extension_a_px": float(tip_extension_a_px),
        "tip_extension_b_px": float(tip_extension_b_px),
    }


def extract_boundary_xy(mask: np.ndarray) -> np.ndarray:
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    if not contours:
        raise ValueError("No contour found for mask")
    contour = max(contours, key=lambda arr: arr.shape[0])
    return np.column_stack([contour[:, 1], contour[:, 0]]).astype(np.float64)


def major_axis_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) < 10:
        raise ValueError("Mask too small for axis estimation")

    centroid_xy = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
    points_xy = np.column_stack([xs, ys]).astype(np.float64)
    centered = points_xy - centroid_xy[None, :]
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_u_xy = unit_vector(eigenvectors[:, int(np.argmax(eigenvalues))])
    boundary_xy = extract_boundary_xy(mask)
    return centroid_xy, axis_u_xy, boundary_xy


def choose_axis_endpoints(
    boundary_xy: np.ndarray,
    centroid_xy: np.ndarray,
    axis_u_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    projection = (boundary_xy - centroid_xy[None, :]) @ axis_u_xy
    pt_plus = boundary_xy[int(np.argmax(projection))]
    pt_minus = boundary_xy[int(np.argmin(projection))]
    return pt_plus.astype(np.float64), pt_minus.astype(np.float64)


def orient_axis_endpoints(
    pt_a_xy: np.ndarray,
    pt_b_xy: np.ndarray,
    reference_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference_xy, dtype=np.float64)
    pt_a = np.asarray(pt_a_xy, dtype=np.float64)
    pt_b = np.asarray(pt_b_xy, dtype=np.float64)
    da = float(np.linalg.norm(pt_a - ref))
    db = float(np.linalg.norm(pt_b - ref))
    if da <= db:
        return pt_a, pt_b
    return pt_b, pt_a


def orient_centerline_to_reference_end(
    centerline_xy: np.ndarray,
    reference_xy: np.ndarray,
) -> np.ndarray:
    centerline_xy = np.asarray(centerline_xy, dtype=np.float64)
    if centerline_xy.ndim != 2 or centerline_xy.shape[0] < 2:
        raise ValueError("Centerline must be an Nx2 array with at least two points")
    endpoint_a_xy, endpoint_b_xy = orient_axis_endpoints(
        centerline_xy[0],
        centerline_xy[-1],
        reference_xy,
    )
    if np.allclose(endpoint_a_xy, centerline_xy[0]):
        return centerline_xy.copy()
    return centerline_xy[::-1].copy()


def project_points_to_centerline(
    points_xy: np.ndarray,
    centerline_xy: np.ndarray,
    n_samples: int = 241,
) -> dict[str, np.ndarray | float]:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    centerline_xy = np.asarray(centerline_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must be an Nx2 array")
    if centerline_xy.ndim != 2 or centerline_xy.shape[0] < 2:
        raise ValueError("centerline_xy must be an Nx2 array with at least two points")

    sample_count = max(int(n_samples), max(int(centerline_xy.shape[0]), 11))
    sampled_xy = resample_polyline_xy(centerline_xy, n_points=sample_count)
    sampled_arc_px = _polyline_arc_length_xy(sampled_xy)
    total_length_px = float(sampled_arc_px[-1]) if sampled_arc_px.size else 0.0

    tree = cKDTree(sampled_xy)
    _, nearest_idx = tree.query(points_xy, k=1)
    nearest_idx = np.asarray(nearest_idx, dtype=int)

    prev_idx = np.clip(nearest_idx - 1, 0, sampled_xy.shape[0] - 1)
    next_idx = np.clip(nearest_idx + 1, 0, sampled_xy.shape[0] - 1)
    seg_start = sampled_xy[prev_idx]
    seg_end = sampled_xy[next_idx]
    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_len_sq = np.sum(seg_vec**2, axis=1)

    rel = points_xy - seg_start
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.divide(np.sum(rel * seg_vec, axis=1), seg_len_sq, out=np.zeros_like(seg_len_sq), where=seg_len_sq > 0)
    u = np.clip(u, 0.0, 1.0)
    projected_xy = seg_start + seg_vec * u[:, None]
    arc_start_px = sampled_arc_px[prev_idx]
    projected_arc_px = arc_start_px + u * seg_len

    tangent_xy = np.divide(seg_vec, seg_len[:, None], out=np.zeros_like(seg_vec), where=seg_len[:, None] > 0)
    fallback = np.array([1.0, 0.0], dtype=np.float64)
    zero_tangent = ~np.isfinite(tangent_xy).all(axis=1) | (np.linalg.norm(tangent_xy, axis=1) <= 0)
    if np.any(zero_tangent):
        tangent_xy[zero_tangent] = fallback

    offset_xy = points_xy - projected_xy
    signed_transverse_px = tangent_xy[:, 0] * offset_xy[:, 1] - tangent_xy[:, 1] * offset_xy[:, 0]
    transverse_abs_px = np.linalg.norm(offset_xy, axis=1)

    if total_length_px > 0:
        projected_fraction = projected_arc_px / total_length_px
    else:
        projected_fraction = np.full(points_xy.shape[0], np.nan, dtype=np.float64)

    return {
        "projected_xy": projected_xy.astype(np.float64),
        "projected_arc_px": projected_arc_px.astype(np.float64),
        "projected_fraction": projected_fraction.astype(np.float64),
        "signed_transverse_px": signed_transverse_px.astype(np.float64),
        "transverse_abs_px": transverse_abs_px.astype(np.float64),
        "tangent_xy": tangent_xy.astype(np.float64),
        "sampled_centerline_xy": sampled_xy.astype(np.float64),
        "sampled_centerline_arc_px": sampled_arc_px.astype(np.float64),
        "total_length_px": float(total_length_px),
    }


def measure_axis_perpendicular_width_profile(
    mask: np.ndarray,
    centerline_xy: np.ndarray,
    pixel_size_um: float,
    n_samples: int = 121,
    edge_trim_fraction: float = 0.05,
    min_pixels_per_sample: int = 24,
    span_percentiles: tuple[float, float] = (2.0, 98.0),
) -> dict[str, object]:
    mask = np.asarray(mask, dtype=bool)
    centerline_xy = np.asarray(centerline_xy, dtype=np.float64)
    if mask.sum() <= 0:
        return {
            "mean_width_px": np.nan,
            "mean_width_um": np.nan,
            "median_width_px": np.nan,
            "median_width_um": np.nan,
            "n_valid_samples": 0,
            "span_segments_xy": [],
            "representative_span_xy": None,
        }

    ys, xs = np.where(mask)
    points_xy = np.column_stack([xs, ys]).astype(np.float64)
    projection = project_points_to_centerline(points_xy, centerline_xy, n_samples=n_samples)
    sampled_xy = np.asarray(projection["sampled_centerline_xy"], dtype=np.float64)
    projected_fraction = np.asarray(projection["projected_fraction"], dtype=np.float64)
    signed_px = np.asarray(projection["signed_transverse_px"], dtype=np.float64)

    if sampled_xy.ndim != 2 or sampled_xy.shape[0] < 3 or projected_fraction.size == 0:
        return {
            "mean_width_px": np.nan,
            "mean_width_um": np.nan,
            "median_width_px": np.nan,
            "median_width_um": np.nan,
            "n_valid_samples": 0,
            "span_segments_xy": [],
            "representative_span_xy": None,
        }

    sample_count = int(sampled_xy.shape[0])
    nearest_idx = np.clip(np.round(projected_fraction * float(sample_count - 1)).astype(int), 0, sample_count - 1)

    span_segments_xy: list[tuple[np.ndarray, np.ndarray]] = []
    span_widths_px: list[float] = []
    keep_low = float(edge_trim_fraction)
    keep_high = 1.0 - float(edge_trim_fraction)

    for sample_idx in range(sample_count):
        sample_fraction = float(sample_idx) / float(max(sample_count - 1, 1))
        if sample_fraction < keep_low or sample_fraction > keep_high:
            continue
        sample_signed = signed_px[nearest_idx == sample_idx]
        sample_signed = sample_signed[np.isfinite(sample_signed)]
        if sample_signed.size < int(min_pixels_per_sample):
            continue

        lo = float(np.nanpercentile(sample_signed, span_percentiles[0]))
        hi = float(np.nanpercentile(sample_signed, span_percentiles[1]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue

        prev_idx = max(sample_idx - 1, 0)
        next_idx = min(sample_idx + 1, sample_count - 1)
        tangent_xy = sampled_xy[next_idx] - sampled_xy[prev_idx]
        tangent_norm = float(np.linalg.norm(tangent_xy))
        if tangent_norm <= 0:
            continue
        tangent_xy = tangent_xy / tangent_norm
        normal_xy = np.array([-tangent_xy[1], tangent_xy[0]], dtype=np.float64)

        center_xy = sampled_xy[sample_idx]
        seg_a_xy = center_xy + normal_xy * lo
        seg_b_xy = center_xy + normal_xy * hi
        span_segments_xy.append((seg_a_xy.astype(np.float64), seg_b_xy.astype(np.float64)))
        span_widths_px.append(hi - lo)

    if not span_widths_px:
        return {
            "mean_width_px": np.nan,
            "mean_width_um": np.nan,
            "median_width_px": np.nan,
            "median_width_um": np.nan,
            "n_valid_samples": 0,
            "span_segments_xy": [],
            "representative_span_xy": None,
        }

    span_widths_px_arr = np.asarray(span_widths_px, dtype=np.float64)
    mean_width_px = float(np.nanmean(span_widths_px_arr))
    median_width_px = float(np.nanmedian(span_widths_px_arr))
    representative_idx = int(np.nanargmin(np.abs(span_widths_px_arr - mean_width_px)))
    representative_span_xy = span_segments_xy[representative_idx]

    px = float(pixel_size_um) if np.isfinite(float(pixel_size_um)) else np.nan
    return {
        "mean_width_px": mean_width_px,
        "mean_width_um": mean_width_px * px if np.isfinite(px) else np.nan,
        "median_width_px": median_width_px,
        "median_width_um": median_width_px * px if np.isfinite(px) else np.nan,
        "n_valid_samples": int(len(span_widths_px)),
        "span_segments_xy": span_segments_xy,
        "representative_span_xy": representative_span_xy,
    }


def mask_touches_image_border(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def measure_mask_geometry(mask: np.ndarray, pixel_size_um: float | None = None) -> dict[str, Any]:
    labels = measure.label(mask.astype(bool))
    if labels.max() <= 0:
        raise ValueError("Mask is empty")
    props = measure.regionprops(labels)
    region = max(props, key=lambda prop: float(prop.area))

    px = float(pixel_size_um) if pixel_size_um is not None and np.isfinite(pixel_size_um) else np.nan
    area_um2 = float(region.area) * (px**2) if np.isfinite(px) else np.nan
    perimeter_um = float(region.perimeter) * px if np.isfinite(px) else np.nan
    length_um = float(region.major_axis_length) * px if np.isfinite(px) else np.nan
    width_um = float(region.minor_axis_length) * px if np.isfinite(px) else np.nan

    min_row, min_col, max_row, max_col = region.bbox
    bbox_height_px = int(max_row - min_row)
    bbox_width_px = int(max_col - min_col)
    return {
        "area_px": int(region.area),
        "area_fraction": float(region.area) / float(mask.size),
        "area_um2": area_um2,
        "perimeter_px": float(region.perimeter),
        "perimeter_um": perimeter_um,
        "major_axis_length_px": float(region.major_axis_length),
        "major_axis_length_um": length_um,
        "minor_axis_length_px": float(region.minor_axis_length),
        "minor_axis_length_um": width_um,
        "aspect_ratio": (
            float(region.major_axis_length) / float(region.minor_axis_length)
            if float(region.minor_axis_length) > 0
            else np.nan
        ),
        "eccentricity": float(region.eccentricity),
        "solidity": float(region.solidity),
        "extent": float(region.extent),
        "bbox_height_px": bbox_height_px,
        "bbox_width_px": bbox_width_px,
        "centroid_x_px": float(region.centroid[1]),
        "centroid_y_px": float(region.centroid[0]),
        "touches_border": mask_touches_image_border(mask),
    }


def quantify_whole_morph_geometry(
    record: ImageRecord,
    root: Path,
    projection: str = "max",
    selected_z: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    abs_path = _resolve_local_path(record.file_path, root=root)
    if selected_z is None:
        projected = load_projected_channels(abs_path, projection=projection)
    else:
        projected = load_plane_channels(abs_path, z_index=int(selected_z))
    dapi = projected["dapi"]
    if dapi is None:
        raise RuntimeError(f"Missing DAPI channel for {record.file_path}")
    mask = segment_whole_morph(dapi)
    centroid_xy, axis_u_xy, boundary_xy = major_axis_from_mask(mask)
    pt_plus_xy, pt_minus_xy = choose_axis_endpoints(
        boundary_xy=boundary_xy,
        centroid_xy=centroid_xy,
        axis_u_xy=axis_u_xy,
    )

    scale_um = projected["stack"].scale_um
    geometry = measure_mask_geometry(mask, pixel_size_um=scale_um.get("X"))
    geometry.update(
        {
            "image_id": record.image_id,
            "cohort_id": record.cohort_id,
            "canonical_position": record.canonical_position,
            "file_id": int(record.file_id),
            "file_path": record.file_path,
            "projection_mode": projection,
            "selected_z": int(projected.get("selected_z", -1))
            if selected_z is not None
            else np.nan,
            "axis_u_x": float(axis_u_xy[0]),
            "axis_u_y": float(axis_u_xy[1]),
            "axis_endpoint_a_x_px": float(pt_plus_xy[0]),
            "axis_endpoint_a_y_px": float(pt_plus_xy[1]),
            "axis_endpoint_b_x_px": float(pt_minus_xy[0]),
            "axis_endpoint_b_y_px": float(pt_minus_xy[1]),
            "pixel_size_x_um": float(scale_um.get("X", np.nan)),
            "pixel_size_y_um": float(scale_um.get("Y", np.nan)),
        }
    )
    return geometry, projected, mask


def plot_mask_outline(ax: plt.Axes, mask: np.ndarray, color: str = "yellow", linewidth: float = 1.2) -> None:
    for contour in measure.find_contours(mask.astype(np.uint8), level=0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def plot_major_axis_overlay(
    ax: plt.Axes,
    centroid_xy: np.ndarray,
    pt_a_xy: np.ndarray,
    pt_b_xy: np.ndarray,
    posterior_click_xy: np.ndarray | None = None,
) -> None:
    ax.plot(
        [pt_a_xy[0], pt_b_xy[0]],
        [pt_a_xy[1], pt_b_xy[1]],
        color="white",
        linewidth=1.3,
        alpha=0.9,
    )
    ax.scatter([centroid_xy[0]], [centroid_xy[1]], c="cyan", s=16, marker="x")
    ax.scatter([pt_a_xy[0], pt_b_xy[0]], [pt_a_xy[1], pt_b_xy[1]], c="white", s=18)

    if posterior_click_xy is None:
        return
    posterior_xy, anterior_xy = orient_axis_endpoints(pt_a_xy, pt_b_xy, posterior_click_xy)
    ax.scatter([posterior_click_xy[0]], [posterior_click_xy[1]], c="yellow", s=26, marker="o")
    ax.scatter([posterior_xy[0]], [posterior_xy[1]], c="yellow", s=50, marker="*")
    ax.scatter([anterior_xy[0]], [anterior_xy[1]], c="magenta", s=30, marker="^")


def plot_centerline_overlay(
    ax: plt.Axes,
    centerline_xy: np.ndarray,
    midpoint_xy: np.ndarray,
    endpoint_a_xy: np.ndarray,
    endpoint_b_xy: np.ndarray,
    base_endpoint_a_xy: np.ndarray | None = None,
    base_endpoint_b_xy: np.ndarray | None = None,
    posterior_click_xy: np.ndarray | None = None,
    line_color: str = "white",
    line_width: float = 1.4,
) -> None:
    centerline_xy = np.asarray(centerline_xy, dtype=np.float64)
    midpoint_xy = np.asarray(midpoint_xy, dtype=np.float64)
    endpoint_a_xy = np.asarray(endpoint_a_xy, dtype=np.float64)
    endpoint_b_xy = np.asarray(endpoint_b_xy, dtype=np.float64)

    ax.plot(
        centerline_xy[:, 0],
        centerline_xy[:, 1],
        color=line_color,
        linewidth=line_width,
        alpha=0.95,
    )
    ax.scatter([midpoint_xy[0]], [midpoint_xy[1]], c="cyan", s=16, marker="x")
    ax.scatter([endpoint_a_xy[0], endpoint_b_xy[0]], [endpoint_a_xy[1], endpoint_b_xy[1]], c=line_color, s=18)

    if base_endpoint_a_xy is not None and base_endpoint_b_xy is not None:
        base_endpoint_a_xy = np.asarray(base_endpoint_a_xy, dtype=np.float64)
        base_endpoint_b_xy = np.asarray(base_endpoint_b_xy, dtype=np.float64)
        ax.scatter(
            [base_endpoint_a_xy[0], base_endpoint_b_xy[0]],
            [base_endpoint_a_xy[1], base_endpoint_b_xy[1]],
            facecolors="none",
            edgecolors="#ff4fd8",
            s=38,
            linewidths=1.1,
        )

    if posterior_click_xy is None:
        return
    posterior_xy, anterior_xy = orient_axis_endpoints(endpoint_a_xy, endpoint_b_xy, posterior_click_xy)
    ax.scatter([posterior_click_xy[0]], [posterior_click_xy[1]], c="yellow", s=26, marker="o")
    ax.scatter([posterior_xy[0]], [posterior_xy[1]], c="yellow", s=50, marker="*")
    ax.scatter([anterior_xy[0]], [anterior_xy[1]], c="magenta", s=30, marker="^")


def render_whole_morph_qc_panel(
    projections: dict[str, Any],
    mask: np.ndarray,
    posterior_click_xy: np.ndarray | None = None,
    title: str | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    dapi = projections["dapi"]
    marker_rgb = projections["marker_rgb"]
    overlay_rgb = projections["overlay_rgb"]

    centerline = centerline_from_mask(mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(robust_rescale(dapi), cmap="gray")
    axes[0].set_title("DAPI Max Projection")
    plot_mask_outline(axes[0], mask)

    axes[1].imshow(marker_rgb)
    axes[1].set_title("MESP2 / FOXF1 / PAX8")
    plot_mask_outline(axes[1], mask)

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Mask + Centerline")
    plot_mask_outline(axes[2], mask)
    plot_centerline_overlay(
        axes[2],
        centerline_xy=centerline["centerline_xy"],
        midpoint_xy=centerline["midpoint_xy"],
        endpoint_a_xy=centerline["endpoint_a_xy"],
        endpoint_b_xy=centerline["endpoint_b_xy"],
        base_endpoint_a_xy=centerline["base_endpoint_a_xy"],
        base_endpoint_b_xy=centerline["base_endpoint_b_xy"],
        posterior_click_xy=posterior_click_xy,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=12)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def add_whole_morph_review_triptych(
    axes_row: Sequence[plt.Axes],
    projections: dict[str, Any],
    mask: np.ndarray,
    title: str,
    subtitle: str | None = None,
) -> None:
    if len(axes_row) != 3:
        raise ValueError("axes_row must contain exactly three axes")

    bf = projections.get("brightfield")
    dapi = projections.get("dapi")
    overlay_rgb = projections.get("overlay_rgb")

    bf_display = robust_rescale(bf) if bf is not None else np.zeros_like(dapi, dtype=np.float32)
    dapi_display = robust_rescale(dapi) if dapi is not None else np.zeros(mask.shape, dtype=np.float32)
    overlay_display = (
        np.asarray(overlay_rgb, dtype=np.float32)
        if overlay_rgb is not None
        else np.stack([dapi_display, dapi_display, dapi_display], axis=-1)
    )

    axes_row[0].imshow(bf_display, cmap="gray")
    axes_row[0].set_title("Brightfield")
    plot_mask_outline(axes_row[0], mask)

    axes_row[1].imshow(dapi_display, cmap="gray")
    axes_row[1].set_title("DAPI")
    plot_mask_outline(axes_row[1], mask)

    axes_row[2].imshow(overlay_display)
    axes_row[2].set_title("Markers + Outline")
    plot_mask_outline(axes_row[2], mask)

    for ax in axes_row:
        ax.set_xticks([])
        ax.set_yticks([])

    axes_row[1].text(
        0.5,
        1.10,
        title,
        transform=axes_row[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    if subtitle:
        axes_row[1].text(
            0.5,
            1.03,
            subtitle,
            transform=axes_row[1].transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def load_mask_review_decisions_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "accepted_as_is": [],
            "needs_manual_outline": [],
            "exclude_from_analysis": [],
            "notes": {},
        }
    payload = json.loads(path.read_text())
    for key in MASK_REVIEW_DECISION_KEYS:
        if key not in payload:
            payload[key] = {} if key == "notes" else []
    if not isinstance(payload["notes"], dict):
        payload["notes"] = {}
    return payload


def save_mask_review_decisions_json(
    path: Path,
    accepted_as_is: Sequence[int | str],
    needs_manual_outline: Sequence[int | str],
    exclude_from_analysis: Sequence[int | str],
    notes: dict[Any, Any] | None = None,
    source_geometry_path: Path | None = None,
) -> dict[str, Any]:
    def _normalize_id_list(values: Sequence[int | str]) -> list[int]:
        out = []
        for value in values:
            out.append(int(value))
        return sorted(set(out))

    notes = {} if notes is None else dict(notes)
    payload = {
        "accepted_as_is": _normalize_id_list(accepted_as_is),
        "needs_manual_outline": _normalize_id_list(needs_manual_outline),
        "exclude_from_analysis": _normalize_id_list(exclude_from_analysis),
        "notes": {str(int(key)): str(value) for key, value in notes.items() if str(value).strip() != ""},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if source_geometry_path is not None:
        payload["source_geometry_path"] = str(Path(source_geometry_path))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_raw_czi_manifest(
    data_root: Path,
    project_root: Path,
    cohort_id: str = DEFAULT_COHORT_ID,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    manifest_entries: list[tuple[Path, int, str]] = []
    for path in sorted(Path(data_root).rglob("*.czi"), key=_safe_int_stem):
        parsed = _parse_manifest_stem(path)
        if parsed is None:
            raise ValueError(f"Expected CZI stem to start with an integer organoid ID, got {path.name}")
        organoid_id, filename_suffix = parsed
        manifest_entries.append((path, int(organoid_id), str(filename_suffix)))

    organoid_counts: dict[int, int] = {}
    for _, organoid_id, _ in manifest_entries:
        organoid_counts[int(organoid_id)] = organoid_counts.get(int(organoid_id), 0) + 1

    variant_index_by_organoid: dict[int, int] = {}
    for path, organoid_id, filename_suffix in manifest_entries:
        variant_index_by_organoid[int(organoid_id)] = variant_index_by_organoid.get(int(organoid_id), 0) + 1
        file_variant_index = int(variant_index_by_organoid[int(organoid_id)])
        duplicate_organoid_count = int(organoid_counts[int(organoid_id)])
        if duplicate_organoid_count > 1:
            file_id = int(organoid_id) * 100 + int(file_variant_index)
        else:
            file_id = int(organoid_id)

        with czifile.CziFile(path) as czi:
            axes = str(czi.axes)
            shape = tuple(int(v) for v in czi.shape)
            metadata = czi.metadata(raw=False)
            dtype_name = str(np.dtype(czi.dtype).name) if hasattr(czi, "dtype") else ""

        axis_to_size = {axis_name: shape[idx] for idx, axis_name in enumerate(axes)}
        c_count = int(axis_to_size.get("C", 1))
        channels = extract_channels(metadata, c_count=c_count)
        acquisition_timestamp_utc = extract_acquisition_timestamp_utc(metadata)
        acquisition_date = ""
        if acquisition_timestamp_utc:
            ts = pd.to_datetime(acquisition_timestamp_utc, utc=True, errors="coerce")
            if not pd.isna(ts):
                acquisition_date = str(ts.date())

        scale_um = extract_scale_um(metadata)
        objective_name = extract_objective_name(metadata)
        rel_path = path.relative_to(project_root).as_posix()
        name_review_note = _name_review_note(
            organoid_id=int(organoid_id),
            filename_suffix=str(filename_suffix),
            duplicate_organoid_count=int(duplicate_organoid_count),
        )

        brightfield_idx = find_channel_index(channels, ["bright"])
        dapi_idx = find_channel_index(channels, ["dapi"])
        mesp2_idx = find_channel_index(channels, ["mesp2", "mcherry"])
        foxf1_idx = find_channel_index(channels, ["foxf1", "tagyfp", "yfp"])
        pax8_idx = find_channel_index(channels, ["pax8", "alexa fluor 647", "647"])

        rows.append(
            {
                "image_id": f"{cohort_id}_{file_id}",
                "cohort_id": str(cohort_id),
                "canonical_position": str(int(organoid_id)),
                "file_id": int(file_id),
                "organoid_id": int(organoid_id),
                "file_variant_index": int(file_variant_index),
                "file_name": path.name,
                "file_stem": path.stem,
                "filename_suffix": str(filename_suffix),
                "file_path": rel_path,
                "duplicate_organoid_flag": bool(duplicate_organoid_count > 1),
                "name_review_flag": bool(name_review_note),
                "name_review_note": str(name_review_note),
                "source_dir": path.parent.relative_to(project_root).as_posix(),
                "acquisition_timestamp_utc": acquisition_timestamp_utc,
                "acquisition_date": acquisition_date,
                "acquisition_batch_label": acquisition_date,
                "batch_warning_flag": False,
                "batch_warning_reason": "",
                "objective_name": objective_name,
                "pixel_size_x_um": float(scale_um.get("X", np.nan)),
                "pixel_size_y_um": float(scale_um.get("Y", np.nan)),
                "pixel_size_z_um": float(scale_um.get("Z", np.nan)),
                "size_c": int(axis_to_size.get("C", 1)),
                "size_z": int(axis_to_size.get("Z", 1)),
                "size_y": int(axis_to_size.get("Y", 0)),
                "size_x": int(axis_to_size.get("X", 0)),
                "czi_axes": axes,
                "dtype": dtype_name,
                "channel_names": ";".join(channels),
                "brightfield_channel_name": channels[brightfield_idx] if brightfield_idx is not None else "",
                "dapi_channel_name": channels[dapi_idx] if dapi_idx is not None else "",
                "mesp2_channel_name": channels[mesp2_idx] if mesp2_idx is not None else "",
                "foxf1_channel_name": channels[foxf1_idx] if foxf1_idx is not None else "",
                "pax8_channel_name": channels[pax8_idx] if pax8_idx is not None else "",
            }
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if manifest.empty:
        return manifest

    valid_dates = manifest["acquisition_date"].fillna("").astype(str)
    valid_dates = valid_dates.loc[valid_dates.str.strip().ne("")]
    predominant_date = str(valid_dates.value_counts().idxmax()) if len(valid_dates) else ""
    if predominant_date:
        warning_mask = manifest["acquisition_date"].fillna("").astype(str) != predominant_date
        manifest.loc[warning_mask, "batch_warning_flag"] = True
        manifest.loc[warning_mask, "batch_warning_reason"] = (
            f"acquisition_date differs from predominant date {predominant_date}"
        )

    manifest = manifest.sort_values(["file_id", "file_name"]).reset_index(drop=True)
    return manifest


def _default_manual_condition_template(manifest_df: pd.DataFrame) -> pd.DataFrame:
    out = manifest_df[
        [
            "image_id",
            "cohort_id",
            "canonical_position",
            "file_id",
            "file_path",
            "acquisition_date",
            "acquisition_batch_label",
        ]
    ].copy()
    out["condition_label"] = ""
    out["condition_group"] = ""
    out["condition_order"] = np.nan
    out["include_in_analysis"] = True
    out["exclusion_reason"] = ""
    out["notes"] = ""
    note_parts: list[pd.Series] = []
    if "name_review_note" in manifest_df.columns:
        note_parts.append(manifest_df["name_review_note"].fillna("").astype(str).str.strip())
    if "batch_warning_reason" in manifest_df.columns:
        note_parts.append(manifest_df["batch_warning_reason"].fillna("").astype(str).str.strip())
    if note_parts:
        note_df = pd.concat(note_parts, axis=1).fillna("")
        out["notes"] = note_df.apply(
            lambda row: " | ".join(part for part in row.astype(str) if part.strip()),
            axis=1,
        )
    return out[MANUAL_CONDITION_COLUMNS].copy()


def load_manual_condition_table(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=MANUAL_CONDITION_COLUMNS)
    df = pd.read_csv(path, sep="\t")
    for col in MANUAL_CONDITION_COLUMNS:
        if col not in df.columns:
            if col == "include_in_analysis":
                df[col] = True
            else:
                df[col] = np.nan
    return df[MANUAL_CONDITION_COLUMNS].copy()


def sync_manual_condition_template(
    manifest_df: pd.DataFrame,
    template_path: Path,
) -> pd.DataFrame:
    template_path = Path(template_path)
    default_df = _default_manual_condition_template(manifest_df)
    if not template_path.exists():
        template_path.parent.mkdir(parents=True, exist_ok=True)
        default_df.to_csv(template_path, sep="\t", index=False)
        return default_df

    existing = load_manual_condition_table(template_path)
    merged = default_df.merge(
        existing.drop(columns=["image_id", "cohort_id", "canonical_position", "file_id"], errors="ignore"),
        on="file_path",
        how="left",
        suffixes=("", "_existing"),
    )

    for column in [
        "condition_label",
        "condition_group",
        "condition_order",
        "include_in_analysis",
        "exclusion_reason",
        "notes",
    ]:
        existing_col = f"{column}_existing"
        if existing_col in merged.columns:
            if column == "include_in_analysis":
                merged[column] = merged[existing_col].where(
                    merged[existing_col].notna(),
                    merged[column],
                )
                merged[column] = merged[column].astype(bool)
            else:
                merged[column] = merged[existing_col].where(
                    merged[existing_col].notna(),
                    merged[column],
                )
            merged = merged.drop(columns=[existing_col])

    merged = merged[MANUAL_CONDITION_COLUMNS].copy()
    merged = merged.sort_values(["file_id", "file_path"]).reset_index(drop=True)
    merged.to_csv(template_path, sep="\t", index=False)
    return merged


def build_analysis_manifest(
    manifest_df: pd.DataFrame,
    manual_condition_df: pd.DataFrame,
) -> pd.DataFrame:
    manual = manual_condition_df.copy()
    for col in MANUAL_CONDITION_COLUMNS:
        if col not in manual.columns:
            manual[col] = np.nan
    merged = manifest_df.merge(
        manual[
            [
                "file_path",
                "condition_label",
                "condition_group",
                "condition_order",
                "include_in_analysis",
                "exclusion_reason",
                "notes",
            ]
        ],
        on="file_path",
        how="left",
    )
    if "include_in_analysis" in merged.columns:
        merged["include_in_analysis"] = merged["include_in_analysis"].fillna(True).astype(bool)
    return merged.sort_values(["file_id", "file_path"]).reset_index(drop=True)


def records_from_manifest(manifest_df: pd.DataFrame) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for row in manifest_df.sort_values(["file_id", "file_path"]).itertuples(index=False):
        records.append(
            ImageRecord(
                image_id=str(row.image_id),
                cohort_id=str(row.cohort_id),
                canonical_position=str(row.canonical_position),
                file_id=int(row.file_id),
                file_path=str(row.file_path),
                acquisition_date=str(getattr(row, "acquisition_date", "") or ""),
                acquisition_batch_label=str(getattr(row, "acquisition_batch_label", "") or ""),
            )
        )
    return records


def parse_int_list_field(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        tokens = [tok.strip() for tok in text.replace(";", ",").split(",")]
        return [int(tok) for tok in tokens if tok]
    if np.isscalar(value):
        try:
            if not np.isfinite(value):
                return []
        except Exception:
            pass
        return [int(value)]
    out: list[int] = []
    for item in value:
        try:
            if np.isfinite(item):
                out.append(int(item))
        except Exception:
            continue
    return out


def _normalized_marker_key(marker_key: Any) -> str:
    text = str(marker_key or "").strip().lower()
    if text in {"pax8", "647", "647/pax8", "pax8-647"}:
        return "pax8"
    if text in {"foxf1", "foxf1-yfp", "yfp"}:
        return "foxf1"
    if text in {"mesp2", "mesp2-mcherry", "mcherry"}:
        return "mesp2"
    return text


def parse_marker_key_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        tokens = [tok.strip() for tok in text.replace(";", ",").split(",")]
        return [marker_key for marker_key in (_normalized_marker_key(tok) for tok in tokens) if marker_key]
    if np.isscalar(value):
        marker_key = _normalized_marker_key(value)
        return [marker_key] if marker_key else []
    out: list[str] = []
    for item in value:
        marker_key = _normalized_marker_key(item)
        if marker_key:
            out.append(marker_key)
    return out


def marker_exclusion_map_from_manifest(manifest_df: pd.DataFrame) -> dict[str, set[str]]:
    excluded: dict[str, set[str]] = {}
    if manifest_df is None or len(manifest_df) == 0:
        return excluded

    for row in manifest_df.itertuples(index=False):
        file_path = str(getattr(row, "file_path", "") or "").strip()
        if not file_path:
            continue
        markers: set[str] = set()
        explicit_value = getattr(row, "excluded_marker_keys", "")
        markers.update(parse_marker_key_list_field(explicit_value))

        notes_text = str(getattr(row, "notes", "") or "").strip().lower()
        if "do not analyze 647/pax8 channel" in notes_text or "do not analyze pax8" in notes_text:
            markers.add("pax8")
        if markers:
            excluded[file_path] = {marker for marker in markers if marker in MARKER_KEYS}
    return excluded


def marker_is_available_for_file(
    file_path: Any,
    marker_key: Any,
    excluded_markers_by_file_path: Optional[dict[str, set[str]]] = None,
) -> bool:
    marker_key_norm = _normalized_marker_key(marker_key)
    file_key = str(file_path or "")
    if marker_key_norm not in MARKER_KEYS:
        return False
    if not excluded_markers_by_file_path:
        return True
    return marker_key_norm not in set(excluded_markers_by_file_path.get(file_key, set()))


def marker_display_name(marker_key: str) -> str:
    return MARKER_DISPLAY_NAMES.get(str(marker_key), str(marker_key))


def chunked(items: Sequence[Any], chunk_size: int) -> list[list[Any]]:
    items = list(items)
    size = max(1, int(chunk_size))
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def marker_channel_index_map(channels: Sequence[str]) -> dict[str, Optional[int]]:
    return {
        marker_key: find_channel_index(channels, MARKER_CHANNEL_KEYWORDS[marker_key])
        for marker_key in MARKER_KEYS
    }


def annulus_mask(
    organoid_mask: np.ndarray,
    inner_radius_px: int,
    outer_radius_px: int,
    min_ring_pixels: int,
) -> np.ndarray:
    organoid_mask = np.asarray(organoid_mask, dtype=bool)
    inner = morphology.binary_dilation(organoid_mask, morphology.disk(max(0, int(inner_radius_px))))
    outer = morphology.binary_dilation(organoid_mask, morphology.disk(max(0, int(outer_radius_px))))
    ring = outer & (~inner)
    if int(ring.sum()) < int(min_ring_pixels):
        ring = ~outer
    if int(ring.sum()) < int(min_ring_pixels):
        ring = ~organoid_mask
    return np.asarray(ring, dtype=bool)


def background_reference_mask(
    organoid_mask: np.ndarray,
    background_estimator: str = "whole_off_morph",
    annulus_inner_radius_px: int = 6,
    annulus_outer_radius_px: int = 20,
    min_ring_pixels: int = 2000,
) -> np.ndarray:
    mode = str(background_estimator).strip().lower()
    mask = np.asarray(organoid_mask, dtype=bool)
    if mode in {"none", "raw", "no_subtraction", "global_raw_null"}:
        return np.zeros_like(mask, dtype=bool)
    if mode in {"whole_off_morph", "whole_off_organoid", "whole_off_mask"}:
        return ~mask
    return annulus_mask(
        organoid_mask=mask,
        inner_radius_px=int(annulus_inner_radius_px),
        outer_radius_px=int(annulus_outer_radius_px),
        min_ring_pixels=int(min_ring_pixels),
    )


def background_correct_signal(
    signal: np.ndarray,
    organoid_mask: np.ndarray,
    background_estimator: str = "whole_off_morph",
    annulus_inner_radius_px: int = 6,
    annulus_outer_radius_px: int = 20,
    min_ring_pixels: int = 2000,
) -> tuple[np.ndarray, float, np.ndarray]:
    signal_f = np.asarray(signal, dtype=np.float32)
    organoid_mask = np.asarray(organoid_mask, dtype=bool)
    mode = str(background_estimator).strip().lower()
    if mode in {"none", "raw", "no_subtraction", "global_raw_null"}:
        empty_bg = np.zeros_like(organoid_mask, dtype=bool)
        return signal_f.astype(np.float32), float("nan"), empty_bg
    bg_mask = background_reference_mask(
        organoid_mask=organoid_mask,
        background_estimator=background_estimator,
        annulus_inner_radius_px=int(annulus_inner_radius_px),
        annulus_outer_radius_px=int(annulus_outer_radius_px),
        min_ring_pixels=int(min_ring_pixels),
    )
    valid_bg = bg_mask & np.isfinite(signal_f)
    if np.any(valid_bg):
        background_value = float(np.median(signal_f[valid_bg]))
    else:
        outside = (~organoid_mask) & np.isfinite(signal_f)
        background_value = float(np.median(signal_f[outside])) if np.any(outside) else 0.0
    corrected = signal_f.astype(np.float32) - np.float32(background_value)
    return corrected.astype(np.float32), float(background_value), np.asarray(bg_mask, dtype=bool)


def smooth_histogram_counts(counts: np.ndarray, sigma_bins: float) -> np.ndarray:
    counts_f = np.asarray(counts, dtype=np.float64)
    sigma = float(sigma_bins)
    if sigma <= 0:
        return counts_f
    return ndi.gaussian_filter1d(counts_f, sigma=sigma, mode="nearest")


def half_gaussian_threshold_from_histogram(
    counts: np.ndarray,
    edges: np.ndarray,
    z: float,
    smooth_sigma_bins: float = 4.0,
) -> dict[str, Any]:
    counts_f = np.asarray(counts, dtype=np.float64)
    edges_f = np.asarray(edges, dtype=np.float64)
    if counts_f.ndim != 1 or edges_f.ndim != 1 or len(edges_f) != len(counts_f) + 1:
        raise ValueError("Histogram shapes are inconsistent.")
    if not np.any(counts_f > 0):
        raise ValueError("Histogram is empty.")

    centers = 0.5 * (edges_f[:-1] + edges_f[1:])
    smooth_counts = smooth_histogram_counts(counts_f, sigma_bins=float(smooth_sigma_bins))
    mode_idx = int(np.nanargmax(smooth_counts))
    baseline_location = float(centers[mode_idx])
    left = centers <= baseline_location
    left_counts = counts_f[left].astype(np.float64)
    left_centers = centers[left].astype(np.float64)

    if np.sum(left_counts) <= 0:
        bin_width = float(np.median(np.diff(centers))) if len(centers) > 1 else 1.0
        baseline_scale = max(abs(bin_width), 1.0)
    else:
        baseline_scale = float(
            np.sqrt(
                np.sum(left_counts * (left_centers - baseline_location) ** 2)
                / np.sum(left_counts)
            )
        )
        baseline_scale = max(float(baseline_scale), 1.0)

    threshold_value = float(baseline_location + float(z) * baseline_scale)
    fit_counts = smooth_counts[mode_idx] * np.exp(
        -0.5 * ((centers - baseline_location) / max(float(baseline_scale), 1e-6)) ** 2
    )
    return {
        "threshold_value": float(threshold_value),
        "baseline_location": float(baseline_location),
        "baseline_scale": float(baseline_scale),
        "null_partition_value": float(baseline_location),
        "hist_edges": edges_f,
        "hist_centers": centers,
        "hist_counts": counts_f,
        "smooth_counts": smooth_counts,
        "fit_counts": fit_counts.astype(np.float64),
        "mode_index": int(mode_idx),
        "smooth_sigma_bins": float(smooth_sigma_bins),
        "threshold_z": float(z),
    }


def positive_mask_from_corrected(
    corrected: np.ndarray,
    organoid_mask: np.ndarray,
    threshold_value: float,
    closing_radius: int = 1,
    min_object_size: int = 16,
    hole_area: int = 16,
) -> np.ndarray:
    positive = np.asarray(organoid_mask, dtype=bool) & (
        np.asarray(corrected, dtype=np.float32) > float(threshold_value)
    )
    if int(closing_radius) > 0:
        positive = morphology.binary_closing(positive, morphology.disk(int(closing_radius)))
    if int(min_object_size) > 1:
        positive = morphology.remove_small_objects(positive, min_size=int(min_object_size))
    if int(hole_area) > 1:
        positive = morphology.remove_small_holes(positive, area_threshold=int(hole_area))
    return np.asarray(positive, dtype=bool) & np.asarray(organoid_mask, dtype=bool)


def corrected_display_limits_from_arrays(
    arrays: Sequence[np.ndarray],
    low_q: float = 0.01,
    high_q: float = 0.995,
) -> tuple[float, float]:
    pooled: list[np.ndarray] = []
    for array in arrays:
        arr = np.asarray(array, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            pooled.append(finite.astype(np.float32))
    if not pooled:
        return 0.0, 1.0
    values = np.concatenate(pooled).astype(np.float32)
    vmin = float(np.quantile(values, float(low_q)))
    vmax = float(np.quantile(values, float(high_q)))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def estimate_marker_thresholds_from_per_z_geometry(
    per_z_geometry_df: pd.DataFrame,
    root: Path,
    excluded_markers_by_file_path: Optional[dict[str, set[str]]] = None,
    threshold_sigma_choices: Sequence[float] = (2.0, 3.0, 4.0, 5.0),
    default_sigma_by_marker: Optional[dict[str, float]] = None,
    hist_bins: int = 2048,
    smooth_sigma_bins: float = 4.0,
    background_estimator: str = "whole_off_morph",
    annulus_inner_radius_px: int = 6,
    annulus_outer_radius_px: int = 20,
    min_ring_pixels: int = 2000,
    sample_values_per_plane: int = 2048,
    random_seed: int = 7,
    hist_quantile_low: float = 0.001,
    hist_quantile_high: float = 0.999,
    hist_edge_padding_fraction: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    root = Path(root)
    sigma_choices = [float(v) for v in threshold_sigma_choices]
    default_sigma = {
        marker_key: float(
            default_sigma_by_marker.get(marker_key, sigma_choices[0])
            if default_sigma_by_marker is not None
            else sigma_choices[0]
        )
        for marker_key in MARKER_KEYS
    }
    plane_df = per_z_geometry_df.sort_values(["file_id", "z_index"]).reset_index(drop=True)
    rng = np.random.default_rng(int(random_seed))

    marker_state = {
        marker_key: {
            "min_value": np.inf,
            "max_value": -np.inf,
            "pooled_pixel_count": 0,
            "background_values": [],
            "sample_values": [],
            "included_plane_count": 0,
        }
        for marker_key in MARKER_KEYS
    }

    grouped = plane_df.groupby("file_path", sort=True)
    for file_path, group in grouped:
        stack = load_czi_stack(_resolve_local_path(file_path, root))
        channel_idx_map = marker_channel_index_map(stack.channels)
        excluded_markers = set(excluded_markers_by_file_path.get(str(file_path), set())) if excluded_markers_by_file_path else set()
        for row in group.itertuples(index=False):
            mask = load_binary_mask(_resolve_local_path(str(row.mask_path), root))
            if not np.any(mask):
                continue
            z_index = int(row.z_index)
            for marker_key in MARKER_KEYS:
                if marker_key in excluded_markers:
                    continue
                ch_idx = channel_idx_map.get(marker_key)
                if ch_idx is None:
                    continue
                signal = np.asarray(stack.data_czyx[int(ch_idx), z_index], dtype=np.float32)
                corrected, background_value, _ = background_correct_signal(
                    signal=signal,
                    organoid_mask=mask,
                    background_estimator=background_estimator,
                    annulus_inner_radius_px=int(annulus_inner_radius_px),
                    annulus_outer_radius_px=int(annulus_outer_radius_px),
                    min_ring_pixels=int(min_ring_pixels),
                )
                values = corrected[mask & np.isfinite(corrected)].astype(np.float32)
                if values.size == 0:
                    continue
                state = marker_state[marker_key]
                state["min_value"] = min(float(state["min_value"]), float(np.min(values)))
                state["max_value"] = max(float(state["max_value"]), float(np.max(values)))
                state["pooled_pixel_count"] += int(values.size)
                state["included_plane_count"] += 1
                state["background_values"].append(float(background_value))
                if int(sample_values_per_plane) > 0:
                    take_n = min(int(sample_values_per_plane), int(values.size))
                    if take_n >= int(values.size):
                        sample = values
                    else:
                        sample = values[rng.choice(int(values.size), size=take_n, replace=False)]
                    state["sample_values"].append(np.asarray(sample, dtype=np.float32))

    hist_edges_by_marker: dict[str, np.ndarray] = {}
    hist_counts_by_marker: dict[str, np.ndarray] = {}
    for marker_key, state in marker_state.items():
        sample_arrays = state["sample_values"]
        pooled_sample = (
            np.concatenate(sample_arrays).astype(np.float32)
            if sample_arrays
            else np.array([], dtype=np.float32)
        )
        finite_sample = pooled_sample[np.isfinite(pooled_sample)]
        if finite_sample.size:
            min_value = float(np.quantile(finite_sample, float(hist_quantile_low)))
            max_value = float(np.quantile(finite_sample, float(hist_quantile_high)))
            if max_value <= min_value:
                min_value = float(np.min(finite_sample))
                max_value = float(np.max(finite_sample))
            span = float(max_value - min_value)
            if np.isfinite(span) and span > 0:
                pad = float(hist_edge_padding_fraction) * span
                min_value -= pad
                max_value += pad
        else:
            min_value = float(state["min_value"])
            max_value = float(state["max_value"])
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            continue
        if max_value <= min_value:
            max_value = min_value + 1.0
        edges = np.linspace(min_value, max_value, max(32, int(hist_bins)) + 1, dtype=np.float64)
        hist_edges_by_marker[marker_key] = edges
        hist_counts_by_marker[marker_key] = np.zeros(len(edges) - 1, dtype=np.float64)

    for file_path, group in plane_df.groupby("file_path", sort=True):
        stack = load_czi_stack(_resolve_local_path(file_path, root))
        channel_idx_map = marker_channel_index_map(stack.channels)
        excluded_markers = set(excluded_markers_by_file_path.get(str(file_path), set())) if excluded_markers_by_file_path else set()
        for row in group.itertuples(index=False):
            mask = load_binary_mask(_resolve_local_path(str(row.mask_path), root))
            if not np.any(mask):
                continue
            z_index = int(row.z_index)
            for marker_key in MARKER_KEYS:
                if marker_key in excluded_markers:
                    continue
                if marker_key not in hist_edges_by_marker:
                    continue
                ch_idx = channel_idx_map.get(marker_key)
                if ch_idx is None:
                    continue
                signal = np.asarray(stack.data_czyx[int(ch_idx), z_index], dtype=np.float32)
                corrected, _, _ = background_correct_signal(
                    signal=signal,
                    organoid_mask=mask,
                    background_estimator=background_estimator,
                    annulus_inner_radius_px=int(annulus_inner_radius_px),
                    annulus_outer_radius_px=int(annulus_outer_radius_px),
                    min_ring_pixels=int(min_ring_pixels),
                )
                values = corrected[mask & np.isfinite(corrected)].astype(np.float32)
                if values.size == 0:
                    continue
                clipped = values[
                    np.isfinite(values)
                    & (values >= hist_edges_by_marker[marker_key][0])
                    & (values <= hist_edges_by_marker[marker_key][-1])
                ]
                if clipped.size == 0:
                    continue
                hist_counts_by_marker[marker_key] += np.histogram(
                    clipped,
                    bins=hist_edges_by_marker[marker_key],
                )[0].astype(np.float64)

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    mode = str(background_estimator).strip().lower()
    threshold_scope = (
        "pooled_all_files_all_z_planes_raw_within_mask"
        if mode in {"none", "raw", "no_subtraction", "global_raw_null"}
        else "pooled_all_files_all_z_planes_within_mask"
    )
    for marker_key in MARKER_KEYS:
        if marker_key not in hist_edges_by_marker:
            continue
        state = marker_state[marker_key]
        sample_arrays = state["sample_values"]
        pooled_sample = (
            np.concatenate(sample_arrays).astype(np.float32)
            if sample_arrays
            else np.array([], dtype=np.float32)
        )
        base_diag = half_gaussian_threshold_from_histogram(
            counts=hist_counts_by_marker[marker_key],
            edges=hist_edges_by_marker[marker_key],
            z=float(default_sigma[marker_key]),
            smooth_sigma_bins=float(smooth_sigma_bins),
        )
        diagnostics[marker_key] = {
            **base_diag,
            "marker_key": marker_key,
            "marker_display_name": marker_display_name(marker_key),
            "pooled_pixel_count": int(state["pooled_pixel_count"]),
            "included_plane_count": int(state["included_plane_count"]),
            "background_values": np.asarray(state["background_values"], dtype=np.float32),
            "sample_values": pooled_sample,
            "hist_quantile_low": float(hist_quantile_low),
            "hist_quantile_high": float(hist_quantile_high),
            "hist_edge_padding_fraction": float(hist_edge_padding_fraction),
        }
        for sigma_value in sigma_choices:
            threshold_value = float(
                base_diag["baseline_location"] + float(sigma_value) * float(base_diag["baseline_scale"])
            )
            rows.append(
                {
                    "marker_key": marker_key,
                    "marker_display_name": marker_display_name(marker_key),
                    "sigma_multiple": float(sigma_value),
                    "selected_as_default": bool(np.isclose(float(sigma_value), float(default_sigma[marker_key]))),
                    "threshold_value": float(threshold_value),
                    "baseline_location": float(base_diag["baseline_location"]),
                    "baseline_scale": float(base_diag["baseline_scale"]),
                    "null_partition_value": float(base_diag["null_partition_value"]),
                    "threshold_scope": threshold_scope,
                    "background_estimator": str(background_estimator),
                    "baseline_location_method": "smoothed_histogram_peak_of_full_pooled_histogram",
                    "baseline_scale_method": "left_half_weighted_second_moment",
                    "peak_hist_bins": int(len(hist_edges_by_marker[marker_key]) - 1),
                    "peak_smooth_sigma_bins": float(smooth_sigma_bins),
                    "pooled_pixel_count": int(state["pooled_pixel_count"]),
                    "included_plane_count": int(state["included_plane_count"]),
                    "sample_pixel_count": int(pooled_sample.size),
                    "pooled_background_median": (
                        float(np.median(np.asarray(state["background_values"], dtype=float)[np.isfinite(np.asarray(state["background_values"], dtype=float))]))
                        if np.isfinite(np.asarray(state["background_values"], dtype=float)).any()
                        else np.nan
                    ),
                }
            )
    threshold_df = pd.DataFrame(rows).sort_values(["marker_key", "sigma_multiple"]).reset_index(drop=True)
    return threshold_df, diagnostics


def quantify_marker_positive_regions(
    per_z_geometry_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    root: Path,
    positive_mask_root: Path,
    excluded_markers_by_file_path: Optional[dict[str, set[str]]] = None,
    background_estimator: str = "whole_off_morph",
    annulus_inner_radius_px: int = 6,
    annulus_outer_radius_px: int = 20,
    min_ring_pixels: int = 2000,
    closing_radius: int = 1,
    min_object_size: int = 16,
    hole_area: int = 16,
) -> pd.DataFrame:
    root = Path(root)
    positive_mask_root = Path(positive_mask_root)
    threshold_use = threshold_df.loc[threshold_df["selected_as_default"].astype(bool)].copy()
    if threshold_use.empty:
        raise ValueError("No default thresholds were marked in threshold_df.")
    threshold_lookup = {
        str(row.marker_key): {
            "sigma_multiple": float(row.sigma_multiple),
            "threshold_value": float(row.threshold_value),
        }
        for row in threshold_use.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    plane_df = per_z_geometry_df.sort_values(["file_id", "z_index"]).reset_index(drop=True)
    for file_path, group in plane_df.groupby("file_path", sort=True):
        stack = load_czi_stack(_resolve_local_path(file_path, root))
        channel_idx_map = marker_channel_index_map(stack.channels)
        excluded_markers = set(excluded_markers_by_file_path.get(str(file_path), set())) if excluded_markers_by_file_path else set()
        for row in group.itertuples(index=False):
            mask = load_binary_mask(_resolve_local_path(str(row.mask_path), root))
            if not np.any(mask):
                continue
            z_index = int(row.z_index)
            organoid_pixel_count = int(np.sum(mask))
            for marker_key in MARKER_KEYS:
                if marker_key in excluded_markers:
                    continue
                if marker_key not in threshold_lookup:
                    continue
                ch_idx = channel_idx_map.get(marker_key)
                if ch_idx is None:
                    continue
                signal = np.asarray(stack.data_czyx[int(ch_idx), z_index], dtype=np.float32)
                corrected, background_value, _ = background_correct_signal(
                    signal=signal,
                    organoid_mask=mask,
                    background_estimator=background_estimator,
                    annulus_inner_radius_px=int(annulus_inner_radius_px),
                    annulus_outer_radius_px=int(annulus_outer_radius_px),
                    min_ring_pixels=int(min_ring_pixels),
                )
                threshold_value = float(threshold_lookup[marker_key]["threshold_value"])
                positive_mask = positive_mask_from_corrected(
                    corrected=corrected,
                    organoid_mask=mask,
                    threshold_value=threshold_value,
                    closing_radius=int(closing_radius),
                    min_object_size=int(min_object_size),
                    hole_area=int(hole_area),
                )
                marker_dir = positive_mask_root / marker_key
                marker_dir.mkdir(parents=True, exist_ok=True)
                positive_mask_path = marker_dir / f"{int(row.file_id):02d}_z{int(z_index):02d}_{marker_key}_positive_mask.tif"
                tifffile.imwrite(positive_mask_path, positive_mask.astype(np.uint8))

                positive_values = corrected[positive_mask & np.isfinite(corrected)].astype(np.float32)
                organoid_values = corrected[mask & np.isfinite(corrected)].astype(np.float32)
                rows.append(
                    {
                        "image_id": str(row.image_id),
                        "cohort_id": str(row.cohort_id),
                        "canonical_position": str(row.canonical_position),
                        "file_id": int(row.file_id),
                        "file_path": str(row.file_path),
                        "acquisition_date": str(getattr(row, "acquisition_date", "") or ""),
                        "acquisition_batch_label": str(getattr(row, "acquisition_batch_label", "") or ""),
                        "z_index": int(z_index),
                        "z_count": int(row.z_count),
                        "mask_path": str(row.mask_path),
                        "marker_key": marker_key,
                        "marker_display_name": marker_display_name(marker_key),
                        "threshold_sigma_multiple": float(threshold_lookup[marker_key]["sigma_multiple"]),
                        "threshold_value": float(threshold_value),
                        "background_estimator": str(background_estimator),
                        "background_value": float(background_value),
                        "organoid_pixel_count": int(organoid_pixel_count),
                        "positive_pixels": int(np.sum(positive_mask)),
                        "positive_fraction": float(np.sum(positive_mask) / organoid_pixel_count)
                        if organoid_pixel_count > 0
                        else np.nan,
                        "organoid_mean_intensity": float(np.mean(organoid_values)) if organoid_values.size else np.nan,
                        "organoid_median_intensity": float(np.median(organoid_values)) if organoid_values.size else np.nan,
                        "positive_mean_intensity": float(np.mean(positive_values)) if positive_values.size else np.nan,
                        "positive_median_intensity": float(np.median(positive_values)) if positive_values.size else np.nan,
                        "positive_integrated_intensity": float(np.sum(positive_values)) if positive_values.size else 0.0,
                        "positive_mask_path": positive_mask_path.relative_to(root).as_posix(),
                    }
                )
    return pd.DataFrame(rows).sort_values(["file_id", "z_index", "marker_key"]).reset_index(drop=True)


def build_posterior_annotation_input_table(
    consensus_df: pd.DataFrame,
    per_z_geometry_df: pd.DataFrame,
) -> pd.DataFrame:
    left = consensus_df.copy()
    right = per_z_geometry_df.copy()
    merged = left.merge(
        right[
            [
                "file_path",
                "z_index",
                "mask_path",
                "acquisition_date",
                "acquisition_batch_label",
            ]
        ].rename(columns={"z_index": "display_z_index", "mask_path": "display_mask_path"}),
        on=["file_path", "display_z_index"],
        how="left",
    )
    missing = merged["display_mask_path"].isna()
    if missing.any():
        missing_ids = merged.loc[missing, "file_id"].astype(int).tolist()
        raise ValueError(
            "Missing display-mask rows for consensus posterior annotation input: "
            + ", ".join(str(v) for v in missing_ids[:10])
        )
    return merged.sort_values(["file_id", "file_path"]).reset_index(drop=True)


def load_posterior_table(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=POSTERIOR_COLUMNS)
    df = pd.read_csv(path, sep="\t")
    for col in POSTERIOR_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[POSTERIOR_COLUMNS].copy()


def upsert_posterior_row(
    posterior_path: Path,
    record: ImageRecord,
    point_xy: tuple[float, float],
) -> None:
    posterior_path = Path(posterior_path)
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_posterior_table(posterior_path)
    if not df.empty:
        df = df[df["file_path"] != record.file_path].copy()

    row = pd.DataFrame(
        [
            {
                "image_id": record.image_id,
                "cohort_id": record.cohort_id,
                "canonical_position": record.canonical_position,
                "file_id": int(record.file_id),
                "file_path": record.file_path,
                "posterior_click_x_px": float(point_xy[0]),
                "posterior_click_y_px": float(point_xy[1]),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        ],
        columns=POSTERIOR_COLUMNS,
    )
    if df.empty:
        out = row
    else:
        out = pd.concat([df, row], ignore_index=True)
    out = out.sort_values(["file_id", "file_path"]).reset_index(drop=True)
    out.to_csv(posterior_path, sep="\t", index=False)


def posterior_progress_for_records(
    records: Sequence[ImageRecord],
    posterior_path: Path,
) -> pd.DataFrame:
    base = pd.DataFrame(
        [
            {
                "image_id": record.image_id,
                "cohort_id": record.cohort_id,
                "canonical_position": record.canonical_position,
                "file_id": int(record.file_id),
                "file_path": record.file_path,
            }
            for record in records
        ]
    )
    if base.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "cohort_id",
                "canonical_position",
                "file_id",
                "file_path",
                "has_posterior_click",
            ]
        )
    posterior_df = load_posterior_table(posterior_path)
    posterior_df["has_posterior_click"] = True
    merged = base.merge(
        posterior_df[["file_path", "has_posterior_click"]],
        on="file_path",
        how="left",
    )
    merged["has_posterior_click"] = merged["has_posterior_click"].fillna(False).astype(bool)
    return merged.sort_values(["file_id", "file_path"]).reset_index(drop=True)


def consensus_geometry_from_annotation_row(
    row: Any,
    per_z_geometry_df: pd.DataFrame,
    root: Path,
    average_mode: str = "mean",
    n_points: int = 121,
) -> dict[str, Any]:
    def _row_get(name: str) -> Any:
        if hasattr(row, name):
            return getattr(row, name)
        return row[name]

    file_path = str(_row_get("file_path"))
    display_z_index = int(_row_get("display_z_index"))
    display_mask_path = str(_row_get("display_mask_path"))
    retained_z = parse_int_list_field(_row_get("retained_z_indices"))
    if not retained_z:
        retained_z = [display_z_index]

    file_sub = per_z_geometry_df[per_z_geometry_df["file_path"].astype(str) == file_path].copy()
    retained_sub = file_sub[file_sub["z_index"].isin(retained_z)].sort_values("z_index").reset_index(drop=True)
    if retained_sub.empty:
        retained_sub = file_sub[file_sub["z_index"] == display_z_index].sort_values("z_index").reset_index(drop=True)
    if retained_sub.empty:
        raise ValueError(f"No retained geometry rows found for posterior annotation file_path={file_path!r}")

    centerlines_xy: list[np.ndarray] = []
    for item in retained_sub.itertuples(index=False):
        retained_mask = load_binary_mask(_resolve_local_path(str(item.mask_path), root))
        centerlines_xy.append(centerline_from_mask(retained_mask)["centerline_xy"])

    display_mask = load_binary_mask(_resolve_local_path(display_mask_path, root))
    return consensus_centerline_from_paths(
        centerlines_xy=centerlines_xy,
        reference_mask=display_mask,
        n_points=int(n_points),
        average_mode=str(average_mode),
    )


class ManualPosteriorPointSession:
    """Interactive widget for one manual posterior-near click per morph."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        root: Path,
        posterior_path: Path,
        projection: str = "max",
        allow_add: bool = True,
    ):
        self.records = list(records)
        if len(self.records) == 0:
            raise ValueError("No image records provided for posterior annotation.")
        self.root = Path(root)
        self.posterior_path = Path(posterior_path)
        self.projection = str(projection)
        self.allow_add = bool(allow_add)

        self.posterior_df = load_posterior_table(self.posterior_path)
        self.index = 0
        self.current_record: Optional[ImageRecord] = None
        self.current_projections: Optional[dict[str, Any]] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_geometry: Optional[dict[str, np.ndarray]] = None
        self.current_point_xy: Optional[tuple[float, float]] = None
        self.cache: dict[str, tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]] = {}

        with plt.ioff():
            self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        try:
            self.fig.canvas.toolbar_visible = True
            self.fig.canvas.header_visible = False
        except Exception:
            pass
        self._click_cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self.btn_prev = widgets.Button(description="Previous")
        self.btn_pass = widgets.Button(description="Pass")
        self.btn_save_next = widgets.Button(description="Save Point + Next", button_style="success")
        self.btn_clear = widgets.Button(description="Clear Point")
        self.progress = widgets.HTML()
        self.status = widgets.HTML()

        self.btn_prev.on_click(self._on_prev)
        self.btn_pass.on_click(self._on_pass)
        self.btn_save_next.on_click(self._on_save_next)
        self.btn_clear.on_click(self._on_clear)

        self._load_current()

    def _set_status(self, text: str) -> None:
        self.status.value = f"<pre style='margin:0'>{text}</pre>"

    def _update_progress(self) -> None:
        record = self.records[self.index]
        self.progress.value = (
            f"<b>Image {self.index + 1} / {len(self.records)}</b> | "
            f"{record.canonical_position} | {record.acquisition_date} | {record.file_path}"
        )

    def _existing_point_for_current(self) -> Optional[tuple[float, float]]:
        record = self.records[self.index]
        if self.posterior_df.empty:
            return None
        sub = self.posterior_df[self.posterior_df["file_path"] == record.file_path].copy()
        if sub.empty:
            return None
        row = sub.sort_values("updated_at").iloc[-1]
        x = pd.to_numeric(row["posterior_click_x_px"], errors="coerce")
        y = pd.to_numeric(row["posterior_click_y_px"], errors="coerce")
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return float(x), float(y)

    def _load_cached_or_compute(
        self,
        record: ImageRecord,
    ) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
        if record.file_path in self.cache:
            return self.cache[record.file_path]

        abs_path = _resolve_local_path(record.file_path, self.root)
        projections = load_projected_channels(abs_path, projection=self.projection)
        dapi = projections["dapi"]
        if dapi is None:
            raise RuntimeError(f"Missing DAPI channel for {record.file_path}")
        mask = segment_whole_morph(dapi)
        geometry = centerline_from_mask(mask)
        self.cache[record.file_path] = (projections, mask, geometry)
        return projections, mask, geometry

    def _load_current(self) -> None:
        self.current_record = self.records[self.index]
        self.current_projections, self.current_mask, self.current_geometry = self._load_cached_or_compute(
            self.current_record
        )
        self.current_point_xy = self._existing_point_for_current()
        self._update_progress()
        if self.current_point_xy is None:
            if self.allow_add:
                self._set_status("Click near the posterior end, then Save Point + Next.")
            else:
                self._set_status("Review mode: no saved posterior click for this image.")
        else:
            if self.allow_add:
                self._set_status("Existing posterior click loaded. Click to replace it, then save.")
            else:
                self._set_status("Review mode: existing posterior click loaded.")
        self._redraw()

    def _redraw(self) -> None:
        if self.current_projections is None or self.current_mask is None or self.current_geometry is None:
            return

        dapi = robust_rescale(self.current_projections["dapi"])
        marker_rgb = self.current_projections["marker_rgb"]
        overlay_rgb = self.current_projections["overlay_rgb"]
        centerline_xy = self.current_geometry["centerline_xy"]
        midpoint_xy = self.current_geometry["midpoint_xy"]
        endpoint_a_xy = self.current_geometry["endpoint_a_xy"]
        endpoint_b_xy = self.current_geometry["endpoint_b_xy"]
        base_endpoint_a_xy = self.current_geometry["base_endpoint_a_xy"]
        base_endpoint_b_xy = self.current_geometry["base_endpoint_b_xy"]

        for ax in self.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        self.axes[0].imshow(dapi, cmap="gray")
        self.axes[0].set_title("DAPI Max Projection")
        plot_mask_outline(self.axes[0], self.current_mask)

        self.axes[1].imshow(marker_rgb)
        self.axes[1].set_title("MESP2 / FOXF1 / PAX8")
        plot_mask_outline(self.axes[1], self.current_mask)

        self.axes[2].imshow(overlay_rgb)
        self.axes[2].set_title("Centerline + Posterior Click")
        plot_mask_outline(self.axes[2], self.current_mask)
        posterior_xy = None if self.current_point_xy is None else np.asarray(self.current_point_xy, dtype=np.float64)
        plot_centerline_overlay(
            self.axes[2],
            centerline_xy=centerline_xy,
            midpoint_xy=midpoint_xy,
            endpoint_a_xy=endpoint_a_xy,
            endpoint_b_xy=endpoint_b_xy,
            base_endpoint_a_xy=base_endpoint_a_xy,
            base_endpoint_b_xy=base_endpoint_b_xy,
            posterior_click_xy=posterior_xy,
        )

        if self.current_point_xy is not None:
            for ax in self.axes:
                ax.scatter(
                    [self.current_point_xy[0]],
                    [self.current_point_xy[1]],
                    c="yellow",
                    s=24,
                    marker="o",
                )

        self.fig.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if self.current_record is None or not self.allow_add:
            return
        if event.inaxes not in list(self.axes):
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        self.current_point_xy = (float(event.xdata), float(event.ydata))
        self._set_status(
            f"Posterior click set to x={self.current_point_xy[0]:.1f}, y={self.current_point_xy[1]:.1f}. Save to keep it."
        )
        self._redraw()

    def _on_prev(self, _btn) -> None:
        if self.index == 0:
            self._set_status("Already at first image.")
            return
        self.index -= 1
        self._load_current()

    def _on_pass(self, _btn) -> None:
        if self.index >= len(self.records) - 1:
            self._set_status("Pass applied. Reached last image.")
            return
        self.index += 1
        self._load_current()

    def _on_clear(self, _btn) -> None:
        if not self.allow_add:
            self._set_status("Review mode: editing is disabled.")
            return
        self.current_point_xy = None
        self._set_status("Cleared local posterior click. Save is needed to replace an existing entry.")
        self._redraw()

    def _on_save_next(self, _btn) -> None:
        if not self.allow_add:
            self._set_status("Review mode: editing is disabled.")
            return
        if self.current_record is None:
            self._set_status("No image loaded.")
            return
        if self.current_point_xy is None:
            self._set_status("No posterior click set for this image.")
            return
        upsert_posterior_row(
            posterior_path=self.posterior_path,
            record=self.current_record,
            point_xy=self.current_point_xy,
        )
        self.posterior_df = load_posterior_table(self.posterior_path)
        if self.index >= len(self.records) - 1:
            self._set_status("Saved posterior click. Reached last image.")
            return
        self.index += 1
        self._load_current()

    def render(self) -> None:
        if self.allow_add:
            controls = widgets.HBox(
                [
                    self.btn_prev,
                    self.btn_pass,
                    self.btn_save_next,
                    self.btn_clear,
                ]
            )
        else:
            controls = widgets.HBox([self.btn_prev, self.btn_pass])
        display(widgets.VBox([self.progress, controls, self.status]))
        display(self.fig.canvas)


class ManualConsensusPosteriorPointSession:
    """Interactive widget for one posterior-near click per file using consensus geometry."""

    def __init__(
        self,
        annotation_df: pd.DataFrame,
        per_z_geometry_df: pd.DataFrame,
        root: Path,
        posterior_path: Path,
        allow_add: bool = True,
    ):
        self.annotation_df = pd.DataFrame(annotation_df).sort_values(["file_id", "file_path"]).reset_index(drop=True)
        if self.annotation_df.empty:
            raise ValueError("No annotation rows provided for posterior annotation.")
        self.per_z_geometry_df = pd.DataFrame(per_z_geometry_df).copy()
        self.root = Path(root)
        self.posterior_path = Path(posterior_path)
        self.allow_add = bool(allow_add)

        self.records = records_from_manifest(self.annotation_df)
        self.posterior_df = load_posterior_table(self.posterior_path)
        progress_df = posterior_progress_for_records(self.records, self.posterior_path)
        remaining = progress_df.index[~progress_df["has_posterior_click"].astype(bool)].tolist()
        self.index = int(remaining[0]) if remaining else 0

        self.current_record: Optional[ImageRecord] = None
        self.current_row: Optional[pd.Series] = None
        self.current_plane: Optional[dict[str, Any]] = None
        self.current_mask: Optional[np.ndarray] = None
        self.current_geometry: Optional[dict[str, Any]] = None
        self.current_point_xy: Optional[tuple[float, float]] = None
        self.cache: dict[str, tuple[dict[str, Any], np.ndarray, dict[str, Any]]] = {}

        with plt.ioff():
            self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        try:
            self.fig.canvas.toolbar_visible = True
            self.fig.canvas.header_visible = False
        except Exception:
            pass
        self._click_cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self.btn_prev = widgets.Button(description="Previous")
        self.btn_pass = widgets.Button(description="Pass")
        self.btn_save_next = widgets.Button(description="Save Point + Next", button_style="success")
        self.btn_clear = widgets.Button(description="Clear Point")
        self.progress = widgets.HTML()
        self.status = widgets.HTML()

        self.btn_prev.on_click(self._on_prev)
        self.btn_pass.on_click(self._on_pass)
        self.btn_save_next.on_click(self._on_save_next)
        self.btn_clear.on_click(self._on_clear)

        self._load_current()

    def _set_status(self, text: str) -> None:
        self.status.value = f"<pre style='margin:0'>{text}</pre>"

    def _existing_point_for_current(self) -> Optional[tuple[float, float]]:
        record = self.records[self.index]
        if self.posterior_df.empty:
            return None
        sub = self.posterior_df[self.posterior_df["file_path"] == record.file_path].copy()
        if sub.empty:
            return None
        row = sub.sort_values("updated_at").iloc[-1]
        x = pd.to_numeric(row["posterior_click_x_px"], errors="coerce")
        y = pd.to_numeric(row["posterior_click_y_px"], errors="coerce")
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return float(x), float(y)

    def _update_progress(self) -> None:
        row = self.annotation_df.iloc[self.index]
        retained_z = parse_int_list_field(row["retained_z_indices"])
        retained_label = ",".join(str(v) for v in retained_z) if retained_z else str(int(row["display_z_index"]))
        self.progress.value = (
            f"<b>Image {self.index + 1} / {len(self.records)}</b> | "
            f"file {int(row['file_id']):02d} | display z={int(row['display_z_index'])} | retained z={retained_label}"
        )

    def _load_cached_or_compute(
        self,
        row: pd.Series,
    ) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
        cache_key = (
            f"{row['file_path']}|z{int(row['display_z_index'])}|"
            f"{str(row.get('retained_z_indices', ''))}|{str(row.get('display_mask_path', ''))}"
        )
        if cache_key in self.cache:
            return self.cache[cache_key]

        plane = load_plane_channels(
            _resolve_local_path(str(row["file_path"]), self.root),
            z_index=int(row["display_z_index"]),
        )
        mask = load_binary_mask(_resolve_local_path(str(row["display_mask_path"]), self.root))
        geometry = consensus_geometry_from_annotation_row(
            row=row,
            per_z_geometry_df=self.per_z_geometry_df,
            root=self.root,
            average_mode="mean",
            n_points=121,
        )
        self.cache[cache_key] = (plane, mask, geometry)
        return plane, mask, geometry

    def _load_current(self) -> None:
        self.current_record = self.records[self.index]
        self.current_row = self.annotation_df.iloc[self.index].copy()
        self.current_plane, self.current_mask, self.current_geometry = self._load_cached_or_compute(self.current_row)
        self.current_point_xy = self._existing_point_for_current()
        self._update_progress()

        if self.current_point_xy is None:
            if self.allow_add:
                self._set_status("Click near the posterior end, then Save Point + Next.")
            else:
                self._set_status("Review mode: no saved posterior click for this image.")
        else:
            if self.allow_add:
                self._set_status("Existing posterior click loaded. Click to replace it, then save.")
            else:
                self._set_status("Review mode: existing posterior click loaded.")
        self._redraw()

    def _redraw(self) -> None:
        if self.current_plane is None or self.current_mask is None or self.current_geometry is None:
            return

        dapi = robust_rescale(self.current_plane["dapi"])
        marker_rgb = self.current_plane["marker_rgb"]
        overlay_rgb = self.current_plane["overlay_rgb"]

        for ax in self.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        self.axes[0].imshow(dapi, cmap="gray", vmin=0.0, vmax=1.0)
        self.axes[0].set_title("DAPI + Display Mask")
        plot_mask_outline(self.axes[0], self.current_mask)

        self.axes[1].imshow(marker_rgb)
        self.axes[1].set_title("MESP2 / FOXF1 / PAX8")
        plot_mask_outline(self.axes[1], self.current_mask)

        self.axes[2].imshow(overlay_rgb)
        self.axes[2].set_title("Consensus Axis + Posterior Click")
        plot_mask_outline(self.axes[2], self.current_mask)
        posterior_xy = None if self.current_point_xy is None else np.asarray(self.current_point_xy, dtype=np.float64)
        plot_centerline_overlay(
            self.axes[2],
            centerline_xy=self.current_geometry["centerline_xy"],
            midpoint_xy=self.current_geometry["midpoint_xy"],
            endpoint_a_xy=self.current_geometry["endpoint_a_xy"],
            endpoint_b_xy=self.current_geometry["endpoint_b_xy"],
            base_endpoint_a_xy=None,
            base_endpoint_b_xy=None,
            posterior_click_xy=posterior_xy,
        )

        if self.current_point_xy is not None:
            x, y = self.current_point_xy
            for ax in self.axes:
                ax.scatter(
                    [x],
                    [y],
                    s=90,
                    facecolors="none",
                    edgecolors="magenta",
                    linewidths=1.8,
                    marker="o",
                    zorder=10,
                )
                ax.scatter(
                    [x],
                    [y],
                    s=65,
                    c="white",
                    linewidths=1.2,
                    marker="+",
                    zorder=11,
                )
                ax.text(
                    x + 6.0,
                    y - 6.0,
                    f"({x:.0f}, {y:.0f})",
                    color="magenta",
                    fontsize=8,
                    fontweight="bold",
                    zorder=12,
                )
        self.fig.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if self.current_record is None or not self.allow_add:
            return
        if event.inaxes not in list(self.axes):
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        self.current_point_xy = (float(event.xdata), float(event.ydata))
        self._set_status(
            f"Posterior click set to x={self.current_point_xy[0]:.1f}, y={self.current_point_xy[1]:.1f}. Save to keep it."
        )
        self._redraw()

    def _on_prev(self, _btn) -> None:
        if self.index == 0:
            self._set_status("Already at first image.")
            return
        self.index -= 1
        self._load_current()

    def _on_pass(self, _btn) -> None:
        if self.index >= len(self.records) - 1:
            self._set_status("Pass applied. Reached last image.")
            return
        self.index += 1
        self._load_current()

    def _on_clear(self, _btn) -> None:
        if not self.allow_add:
            self._set_status("Review mode: editing is disabled.")
            return
        self.current_point_xy = None
        self._set_status("Cleared local posterior click. Save is needed to replace an existing entry.")
        self._redraw()

    def _on_save_next(self, _btn) -> None:
        if not self.allow_add:
            self._set_status("Review mode: editing is disabled.")
            return
        if self.current_record is None:
            self._set_status("No image loaded.")
            return
        if self.current_point_xy is None:
            self._set_status("No posterior click set for this image.")
            return
        upsert_posterior_row(
            posterior_path=self.posterior_path,
            record=self.current_record,
            point_xy=self.current_point_xy,
        )
        self.posterior_df = load_posterior_table(self.posterior_path)
        if self.index >= len(self.records) - 1:
            self._set_status("Saved posterior click. Reached last image.")
            return
        self.index += 1
        self._load_current()

    def render(self) -> None:
        if self.allow_add:
            controls = widgets.HBox(
                [
                    self.btn_prev,
                    self.btn_pass,
                    self.btn_save_next,
                    self.btn_clear,
                ]
            )
        else:
            controls = widgets.HBox([self.btn_prev, self.btn_pass])
        display(widgets.VBox([self.progress, controls, self.status]))
        display(self.fig.canvas)
