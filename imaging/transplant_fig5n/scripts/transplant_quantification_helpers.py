"""Helpers for LPM transplant pixel-level quantification.

This module is intentionally lightweight and mirrors the structure of the
neighboring manuscript-analysis workspaces: raw-image loading, DAPI-based mask
generation, reporter normalization, and host/donor pixel classification live in
plain Python so notebooks can stay focused on interpretation and figure making.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# czifile cannot be installed beside this project's pinned numpy, and mis-frames odd-sized
# acquisitions. czi_compat presents czifile's interface over libCZI. See src/trunk_morph_ref/czi_compat.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from src.trunk_morph_ref import czi_compat as czifile
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
import tifffile


CANONICAL_CHANNEL_ORDER = [
    "brightfield",
    "dapi",
    "mesp2",
    "foxf1",
    "donor",
    "host",
]

CHANNEL_DISPLAY_NAMES = {
    "brightfield": "Brightfield",
    "dapi": "DAPI",
    "mesp2": "MESP2-RFP",
    "foxf1": "FOXF1-YFP",
    "donor": "Donor Reporter",
    "host": "Host Reporter",
}

PIXEL_CLASS_LABELS = {
    0: "unlabeled",
    1: "host",
    2: "donor",
    3: "mixed",
}

MANIFEST_COLUMNS = [
    "image_id",
    "file_id",
    "condition",
    "position_label",
    "file_name",
    "file_path",
    "source_format",
    "exists",
    "size_bytes",
    "modified_at",
    "axes",
    "size_c",
    "size_z",
    "size_y",
    "size_x",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "pixel_size_z_um",
    "raw_channel_names",
    "canonical_channel_names",
    "acquisition_timestamp_utc",
]

CONDITION_SORT_PRIORITY = {
    "ctrl": 0,
}


@dataclass
class ImageStack:
    path: Path
    data_czyx: np.ndarray
    raw_channel_names: list[str]
    canonical_channel_names: list[str]
    scale_um: dict[str, float]
    acquisition_timestamp_utc: str
    axes: str
    source_format: str
    metadata: dict[str, Any]


def _ensure_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _canonical_channels_for_count(c_count: int) -> list[str]:
    channels = list(CANONICAL_CHANNEL_ORDER[: int(c_count)])
    if int(c_count) > len(CANONICAL_CHANNEL_ORDER):
        channels.extend(
            f"extra_channel_{idx:02d}"
            for idx in range(len(CANONICAL_CHANNEL_ORDER), int(c_count))
        )
    return channels


def normalize_to_czyx(array: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Normalize a stack to CZYX, reducing any other axes by max projection."""
    out = np.asarray(array)
    out_axes = str(axes)
    if len(out_axes) != out.ndim:
        raise ValueError(
            f"Axes string length does not match ndim. axes={out_axes!r}, ndim={out.ndim}"
        )
    if "C" not in out_axes or "Y" not in out_axes or "X" not in out_axes:
        raise ValueError(f"Missing required axes for CZYX normalization. axes={out_axes!r}")

    c_idx = out_axes.index("C")
    out = np.moveaxis(out, c_idx, 0)
    out_axes = "C" + out_axes[:c_idx] + out_axes[c_idx + 1 :]

    if "Z" in out_axes:
        z_idx = out_axes.index("Z")
        out = np.moveaxis(out, z_idx, 1)
        out_axes = "CZ" + out_axes[1:z_idx] + out_axes[z_idx + 1 :]
    else:
        out = np.expand_dims(out, axis=1)
        out_axes = "CZ" + out_axes[1:]

    while any(ax not in {"C", "Z", "Y", "X"} for ax in out_axes):
        for axis_name in list(out_axes):
            if axis_name in {"C", "Z", "Y", "X"}:
                continue
            axis = out_axes.index(axis_name)
            if out.shape[axis] == 1:
                out = np.squeeze(out, axis=axis)
            else:
                out = np.max(out, axis=axis)
            out_axes = out_axes[:axis] + out_axes[axis + 1 :]
            break

    perm = [
        out_axes.index("C"),
        out_axes.index("Z"),
        out_axes.index("Y"),
        out_axes.index("X"),
    ]
    out = np.transpose(out, axes=perm)
    return np.asarray(out, dtype=np.float32), "CZYX"


def extract_czi_channels(metadata: dict[str, Any], c_count: int) -> list[str]:
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
    if len(names) < int(c_count):
        names.extend(f"Channel_{idx}" for idx in range(len(names), int(c_count)))
    return names[: int(c_count)]


def extract_czi_scale_um(metadata: dict[str, Any]) -> dict[str, float]:
    out = {"X": np.nan, "Y": np.nan, "Z": np.nan}
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
            value_um = float(value) * 1e6
        except Exception:
            continue
        if axis in out:
            out[axis] = value_um
    return out


def extract_czi_acquisition_timestamp_utc(metadata: dict[str, Any]) -> str:
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


def load_czi_stack(path: Path) -> ImageStack:
    with czifile.CziFile(path) as czi:
        raw = czi.asarray()
        raw_axes = str(czi.axes)
        metadata = czi.metadata(raw=False)
    data_czyx, axes = normalize_to_czyx(raw, raw_axes)
    raw_channel_names = extract_czi_channels(metadata, c_count=data_czyx.shape[0])
    canonical_channel_names = _canonical_channels_for_count(data_czyx.shape[0])
    return ImageStack(
        path=Path(path),
        data_czyx=data_czyx,
        raw_channel_names=raw_channel_names,
        canonical_channel_names=canonical_channel_names,
        scale_um=extract_czi_scale_um(metadata),
        acquisition_timestamp_utc=extract_czi_acquisition_timestamp_utc(metadata),
        axes=axes,
        source_format="czi",
        metadata={"czi_axes": raw_axes},
    )


def _extract_imagej_scale_um(tif: tifffile.TiffFile) -> dict[str, float]:
    imagej_meta = tif.imagej_metadata or {}
    out = {"X": np.nan, "Y": np.nan, "Z": np.nan}
    if "spacing" in imagej_meta:
        out["Z"] = _safe_float(imagej_meta.get("spacing"))
    return out


def load_tiff_like_stack(path: Path) -> ImageStack:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        raw = series.asarray()
        raw_axes = getattr(series, "axes", "")
        imagej_meta = tif.imagej_metadata or {}
        scale_um = _extract_imagej_scale_um(tif)
    if not raw_axes:
        if raw.ndim == 4:
            raw_axes = "ZCYX"
        elif raw.ndim == 3:
            raw_axes = "CYX"
        else:
            raise ValueError(f"Could not infer TIFF axes for {path}")
    data_czyx, axes = normalize_to_czyx(raw, raw_axes)
    c_count = int(data_czyx.shape[0])
    raw_channel_names = _canonical_channels_for_count(c_count)
    return ImageStack(
        path=Path(path),
        data_czyx=data_czyx,
        raw_channel_names=raw_channel_names,
        canonical_channel_names=_canonical_channels_for_count(c_count),
        scale_um=scale_um,
        acquisition_timestamp_utc="",
        axes=axes,
        source_format="tiff_hyperstack",
        metadata={"tiff_axes": raw_axes, "imagej_metadata": imagej_meta},
    )


def load_transplant_stack(path: Path) -> ImageStack:
    path = Path(path)
    try:
        return load_czi_stack(path)
    except Exception:
        return load_tiff_like_stack(path)


def build_raw_input_manifest(data_dir: Path, root: Optional[Path] = None) -> pd.DataFrame:
    root = Path(root or data_dir.parent)
    rows: list[dict[str, Any]] = []
    file_id = 1
    image_paths = [
        path
        for path in Path(data_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in {".czi", ".tif", ".tiff"}
    ]

    def _condition_for_path(path: Path) -> str:
        rel_parts = path.resolve().relative_to(Path(data_dir).resolve()).parts
        if len(rel_parts) >= 2:
            return str(rel_parts[0])
        return "unspecified"

    def _sort_key(path: Path) -> tuple[Any, ...]:
        condition = _condition_for_path(path)
        priority = CONDITION_SORT_PRIORITY.get(str(condition).lower(), 1)
        return (
            int(priority),
            str(condition).lower(),
            path.name.lower(),
            path.as_posix().lower(),
        )

    for path in sorted(image_paths, key=_sort_key):
        condition = _condition_for_path(path)
        stack = load_transplant_stack(path)
        rel_path = path.resolve().relative_to(root.resolve()).as_posix()
        rows.append(
            {
                "image_id": f"img{file_id:02d}",
                "file_id": int(file_id),
                "condition": str(condition),
                "position_label": path.stem,
                "file_name": path.name,
                "file_path": rel_path,
                "source_format": stack.source_format,
                "exists": True,
                "size_bytes": int(path.stat().st_size),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "axes": stack.axes,
                "size_c": int(stack.data_czyx.shape[0]),
                "size_z": int(stack.data_czyx.shape[1]),
                "size_y": int(stack.data_czyx.shape[2]),
                "size_x": int(stack.data_czyx.shape[3]),
                "pixel_size_x_um": float(stack.scale_um.get("X", np.nan)),
                "pixel_size_y_um": float(stack.scale_um.get("Y", np.nan)),
                "pixel_size_z_um": float(stack.scale_um.get("Z", np.nan)),
                "raw_channel_names": ";".join(stack.raw_channel_names),
                "canonical_channel_names": ";".join(stack.canonical_channel_names),
                "acquisition_timestamp_utc": stack.acquisition_timestamp_utc,
            }
        )
        file_id += 1
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def channel_index(channel_names: Sequence[str], key: str) -> int:
    key = str(key)
    if key not in channel_names:
        raise KeyError(f"Channel {key!r} not found in {list(channel_names)!r}")
    return int(list(channel_names).index(key))


def segment_whole_organoid_from_dapi_with_debug(
    dapi_image: np.ndarray,
    gaussian_sigma: float = 2.0,
    baseline_percentile: float = 5.0,
    threshold_scale: float = 0.50,
    min_size_px: int = 4000,
    opening_radius_px: int = 3,
    closing_radius_px: int = 7,
    dilation_radius_px: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = np.asarray(dapi_image, dtype=np.float32)
    finite = np.isfinite(image)
    empty = np.zeros_like(image, dtype=bool)
    if not finite.any():
        return empty, {
            "baseline_value": np.nan,
            "raw_threshold": np.nan,
            "threshold_value": np.nan,
            "selected_variant": "empty_input",
            "n_raw_components": 0,
            "mask_area_px": 0,
        }

    baseline_value = float(np.nanpercentile(image[finite], float(baseline_percentile)))
    flattened = np.clip(image - baseline_value, a_min=0.0, a_max=None).astype(np.float32)
    smoothed = filters.gaussian(flattened, sigma=float(gaussian_sigma), preserve_range=True)
    values = smoothed[np.isfinite(smoothed)]
    if values.size < 64:
        return empty, {
            "baseline_value": baseline_value,
            "raw_threshold": np.nan,
            "threshold_value": np.nan,
            "selected_variant": "too_few_values",
            "n_raw_components": 0,
            "mask_area_px": 0,
        }

    raw_threshold = float(filters.threshold_otsu(values))
    threshold_value = float(raw_threshold * float(threshold_scale))
    mask = smoothed >= threshold_value
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=int(min_size_px))
    if int(dilation_radius_px) > 0:
        mask = morphology.binary_dilation(mask, footprint=morphology.disk(int(dilation_radius_px)))
    mask = morphology.binary_closing(mask, footprint=morphology.disk(int(closing_radius_px)))
    mask = ndi.binary_fill_holes(mask)
    labels = measure.label(mask)
    n_raw_components = int(labels.max())
    if n_raw_components == 0:
        return empty, {
            "baseline_value": baseline_value,
            "raw_threshold": raw_threshold,
            "threshold_value": threshold_value,
            "selected_variant": f"otsu x{float(threshold_scale):.2f}",
            "n_raw_components": 0,
            "mask_area_px": 0,
        }
    props = measure.regionprops(labels)
    largest = max(props, key=lambda item: float(item.area))
    mask = labels == largest.label
    if int(opening_radius_px) > 0:
        mask = morphology.binary_opening(mask, footprint=morphology.disk(int(opening_radius_px)))
    mask = morphology.binary_closing(mask, footprint=morphology.disk(int(closing_radius_px)))
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool), {
        "baseline_value": baseline_value,
        "raw_threshold": raw_threshold,
        "threshold_value": threshold_value,
        "selected_variant": f"otsu x{float(threshold_scale):.2f}",
        "n_raw_components": n_raw_components,
        "mask_area_px": int(np.sum(mask)),
    }


def background_reference_mask(
    organoid_mask: np.ndarray,
    background_estimator: str = "whole_off_organoid",
    annulus_inner_radius_px: int = 6,
    annulus_outer_radius_px: int = 20,
    min_ring_pixels: int = 2000,
) -> np.ndarray:
    mask = np.asarray(organoid_mask, dtype=bool)
    mode = str(background_estimator).strip().lower()
    if mode in {"none", "raw", "no_subtraction"}:
        return np.zeros_like(mask, dtype=bool)
    if mode in {"whole_off_organoid", "whole_off_mask", "whole_off_tissue"}:
        return ~mask
    inner = morphology.binary_dilation(mask, morphology.disk(max(0, int(annulus_inner_radius_px))))
    outer = morphology.binary_dilation(mask, morphology.disk(max(0, int(annulus_outer_radius_px))))
    ring = outer & (~inner)
    if int(ring.sum()) < int(min_ring_pixels):
        ring = ~outer
    if int(ring.sum()) < int(min_ring_pixels):
        ring = ~mask
    return ring.astype(bool)


def background_correct_signal(
    signal: np.ndarray,
    organoid_mask: np.ndarray,
    background_estimator: str = "whole_off_organoid",
    annulus_inner_radius_px: int = 6,
    annulus_outer_radius_px: int = 20,
    min_ring_pixels: int = 2000,
) -> tuple[np.ndarray, float, np.ndarray]:
    signal_f = np.asarray(signal, dtype=np.float32)
    bg_mask = background_reference_mask(
        organoid_mask=organoid_mask,
        background_estimator=background_estimator,
        annulus_inner_radius_px=int(annulus_inner_radius_px),
        annulus_outer_radius_px=int(annulus_outer_radius_px),
        min_ring_pixels=int(min_ring_pixels),
    )
    valid_bg = bg_mask & np.isfinite(signal_f)
    if np.any(valid_bg):
        bg_value = float(np.median(signal_f[valid_bg]))
    else:
        valid_off = (~np.asarray(organoid_mask, dtype=bool)) & np.isfinite(signal_f)
        bg_value = float(np.median(signal_f[valid_off])) if np.any(valid_off) else 0.0
    corrected = signal_f - np.float32(bg_value)
    corrected = np.where(np.isfinite(corrected), corrected, np.nan).astype(np.float32)
    return corrected, float(bg_value), bg_mask.astype(bool)


def compute_dapi_floor(
    corrected_dapi: np.ndarray,
    organoid_mask: np.ndarray,
    floor_quantile: float = 0.10,
    min_floor: float = 1.0,
) -> float:
    valid = (
        np.asarray(organoid_mask, dtype=bool)
        & np.isfinite(corrected_dapi)
        & (np.asarray(corrected_dapi, dtype=np.float32) > 0)
    )
    if not np.any(valid):
        return float(min_floor)
    floor_value = float(np.quantile(np.asarray(corrected_dapi, dtype=np.float32)[valid], float(floor_quantile)))
    if not np.isfinite(floor_value):
        floor_value = float(min_floor)
    return float(max(float(min_floor), floor_value))


def reporter_over_dapi(
    corrected_reporter: np.ndarray,
    corrected_dapi: np.ndarray,
    organoid_mask: np.ndarray,
    dapi_floor: float,
) -> np.ndarray:
    reporter = np.clip(np.asarray(corrected_reporter, dtype=np.float32), a_min=0.0, a_max=None)
    dapi = np.clip(np.asarray(corrected_dapi, dtype=np.float32), a_min=0.0, a_max=None)
    denom = np.maximum(dapi, float(dapi_floor)).astype(np.float32)
    ratio = np.full_like(reporter, np.nan, dtype=np.float32)
    valid = np.asarray(organoid_mask, dtype=bool) & np.isfinite(reporter) & np.isfinite(denom)
    ratio[valid] = reporter[valid] / denom[valid]
    return ratio


def normalize_signal_with_null(
    signal: np.ndarray,
    background_mu: float,
    background_sigma: float,
) -> np.ndarray:
    signal_f = np.asarray(signal, dtype=np.float32)
    sigma = float(max(float(background_sigma), 1.0e-6))
    normalized = (signal_f - np.float32(float(background_mu))) / np.float32(sigma)
    normalized = np.where(np.isfinite(normalized), normalized, np.nan).astype(np.float32)
    return normalized


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
    smooth_counts = ndi.gaussian_filter1d(counts_f, sigma=float(smooth_sigma_bins), mode="nearest")
    mode_idx = int(np.nanargmax(smooth_counts))
    baseline_location = float(centers[mode_idx])
    left = centers <= baseline_location
    left_counts = counts_f[left]
    left_centers = centers[left]
    if np.sum(left_counts) <= 0:
        baseline_scale = 1.0
    else:
        baseline_scale = float(
            np.sqrt(np.sum(left_counts * (left_centers - baseline_location) ** 2) / np.sum(left_counts))
        )
        baseline_scale = max(baseline_scale, 1.0e-6)
    threshold_value = float(baseline_location + float(z) * baseline_scale)
    return {
        "threshold_value": float(threshold_value),
        "baseline_location": float(baseline_location),
        "baseline_scale": float(baseline_scale),
        "null_partition_value": float(baseline_location),
        "hist_edges": edges_f,
        "hist_centers": centers,
        "hist_counts": counts_f,
        "smooth_counts": smooth_counts,
        "mode_index": int(mode_idx),
        "threshold_z": float(z),
    }


def half_gaussian_threshold_from_values(
    values: np.ndarray,
    z: float,
    hist_bins: int = 2048,
    smooth_sigma_bins: float = 4.0,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
    edge_padding_fraction: float = 0.05,
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if arr.size == 0:
        raise ValueError("No finite nonnegative values were available for threshold estimation.")
    lo = float(np.quantile(arr, float(quantile_low)))
    hi = float(np.quantile(arr, float(quantile_high)))
    if not np.isfinite(lo):
        lo = float(np.min(arr))
    if not np.isfinite(hi):
        hi = float(np.max(arr))
    if hi <= lo:
        hi = lo + 1.0
    pad = float(edge_padding_fraction) * float(hi - lo)
    edge_min = max(0.0, lo - pad)
    edge_max = hi + pad
    edges = np.linspace(edge_min, edge_max, max(32, int(hist_bins)) + 1, dtype=np.float64)
    clipped = np.clip(arr.astype(np.float64), edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    diag = half_gaussian_threshold_from_histogram(
        counts=counts,
        edges=edges,
        z=float(z),
        smooth_sigma_bins=float(smooth_sigma_bins),
    )
    diag["sample_count"] = int(arr.size)
    return diag


def half_gaussian_null_from_values(
    values: np.ndarray,
    hist_bins: int = 2048,
    smooth_sigma_bins: float = 4.0,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
    edge_padding_fraction: float = 0.05,
) -> dict[str, Any]:
    diag = half_gaussian_threshold_from_values(
        values=values,
        z=0.0,
        hist_bins=int(hist_bins),
        smooth_sigma_bins=float(smooth_sigma_bins),
        quantile_low=float(quantile_low),
        quantile_high=float(quantile_high),
        edge_padding_fraction=float(edge_padding_fraction),
    )
    diag["null_mu"] = float(diag["baseline_location"])
    diag["null_sigma"] = float(diag["baseline_scale"])
    diag["method"] = "half_gaussian_left_null"
    return diag


def otsu_threshold_from_values(
    values: np.ndarray,
    transform: str = "log1p",
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if arr.size == 0:
        raise ValueError("No finite nonnegative values were available for Otsu threshold estimation.")
    if str(transform) == "log1p":
        transformed = np.log1p(arr.astype(np.float64))
        threshold_transformed = float(filters.threshold_otsu(transformed))
        threshold_value = float(np.expm1(threshold_transformed))
    else:
        transformed = arr.astype(np.float64)
        threshold_transformed = float(filters.threshold_otsu(transformed))
        threshold_value = float(threshold_transformed)
    return {
        "threshold_value": float(threshold_value),
        "threshold_transformed": float(threshold_transformed),
        "sample_count": int(arr.size),
        "method": f"otsu_{str(transform)}",
    }


def classify_host_donor_pixels(
    host_ratio: np.ndarray,
    donor_ratio: np.ndarray,
    host_threshold: float,
    donor_threshold: float,
    dominance_margin_log2: float = 0.75,
    organoid_mask: Optional[np.ndarray] = None,
    eps: float = 1.0e-6,
) -> np.ndarray:
    host_ratio = np.asarray(host_ratio, dtype=np.float32)
    donor_ratio = np.asarray(donor_ratio, dtype=np.float32)
    classes = np.zeros(host_ratio.shape, dtype=np.uint8)
    valid = np.isfinite(host_ratio) & np.isfinite(donor_ratio)
    if organoid_mask is not None:
        valid &= np.asarray(organoid_mask, dtype=bool)
    host_positive = valid & (host_ratio > float(host_threshold))
    donor_positive = valid & (donor_ratio > float(donor_threshold))
    dominance = np.log2((host_ratio + float(eps)) / (donor_ratio + float(eps)))

    host_only = host_positive & (~donor_positive)
    donor_only = donor_positive & (~host_positive)
    both = host_positive & donor_positive
    host_dominant = both & (dominance >= float(dominance_margin_log2))
    donor_dominant = both & (dominance <= -float(dominance_margin_log2))
    mixed = both & (~host_dominant) & (~donor_dominant)

    classes[host_only | host_dominant] = 1
    classes[donor_only | donor_dominant] = 2
    classes[mixed] = 3
    return classes


def classify_host_donor_from_binary_calls(
    host_positive: np.ndarray,
    donor_positive: np.ndarray,
    host_ratio: np.ndarray,
    donor_ratio: np.ndarray,
    organoid_mask: np.ndarray,
    dominance_margin_log2: float = 0.75,
    eps: float = 1.0e-6,
) -> np.ndarray:
    host_positive = np.asarray(host_positive, dtype=bool)
    donor_positive = np.asarray(donor_positive, dtype=bool)
    host_ratio = np.asarray(host_ratio, dtype=np.float32)
    donor_ratio = np.asarray(donor_ratio, dtype=np.float32)
    valid = np.asarray(organoid_mask, dtype=bool) & np.isfinite(host_ratio) & np.isfinite(donor_ratio)

    classes = np.zeros(valid.shape, dtype=np.uint8)
    if not np.any(valid):
        return classes

    host_positive = valid & host_positive
    donor_positive = valid & donor_positive
    dominance = np.log2((host_ratio + float(eps)) / (donor_ratio + float(eps)))

    host_only = host_positive & (~donor_positive)
    donor_only = donor_positive & (~host_positive)
    both = host_positive & donor_positive
    host_dominant = both & (dominance >= float(dominance_margin_log2))
    donor_dominant = both & (dominance <= -float(dominance_margin_log2))
    mixed = both & (~host_dominant) & (~donor_dominant)

    classes[host_only | host_dominant] = 1
    classes[donor_only | donor_dominant] = 2
    classes[mixed] = 3
    return classes


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    labels, n_labels = ndi.label(mask)
    if int(n_labels) <= 1:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep_label = int(np.argmax(counts))
    return (labels == keep_label).astype(bool)


def donor_graft_component(
    donor_positive: np.ndarray,
    organoid_mask: np.ndarray,
    fill_holes: bool = True,
) -> np.ndarray:
    donor_positive = np.asarray(donor_positive, dtype=bool)
    organoid_mask = np.asarray(organoid_mask, dtype=bool)
    graft = largest_connected_component(donor_positive & organoid_mask)
    if bool(fill_holes) and np.any(graft):
        graft = ndi.binary_fill_holes(graft) & organoid_mask
    return graft.astype(bool)


def connected_components_touching_seed(
    mask: np.ndarray,
    seed: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    seed = np.asarray(seed, dtype=bool) & mask
    if not np.any(mask) or not np.any(seed):
        return np.zeros_like(mask, dtype=bool)
    labels, _ = ndi.label(mask)
    keep_labels = np.unique(labels[seed & (labels > 0)])
    if keep_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, keep_labels).astype(bool)


def partition_domain_by_seeds(
    domain_mask: np.ndarray,
    primary_seed: np.ndarray,
    secondary_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    domain_mask = np.asarray(domain_mask, dtype=bool)
    primary_seed = np.asarray(primary_seed, dtype=bool) & domain_mask
    secondary_seed = np.asarray(secondary_seed, dtype=bool) & domain_mask & (~primary_seed)

    if not np.any(primary_seed) and not np.any(secondary_seed):
        empty = np.zeros_like(domain_mask, dtype=bool)
        return empty, empty
    if not np.any(primary_seed):
        return np.zeros_like(domain_mask, dtype=bool), domain_mask.astype(bool)
    if not np.any(secondary_seed):
        return domain_mask.astype(bool), np.zeros_like(domain_mask, dtype=bool)

    dist_primary = ndi.distance_transform_edt(~primary_seed)
    dist_secondary = ndi.distance_transform_edt(~secondary_seed)
    primary_region = domain_mask & (dist_primary <= dist_secondary)
    secondary_region = domain_mask & (~primary_region)
    return primary_region.astype(bool), secondary_region.astype(bool)


def assign_domain_by_intensity_with_seed_connectivity(
    domain_mask: np.ndarray,
    primary_seed: np.ndarray,
    secondary_seed: np.ndarray,
    primary_score: np.ndarray,
    secondary_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    domain_mask = np.asarray(domain_mask, dtype=bool)
    primary_seed = np.asarray(primary_seed, dtype=bool) & domain_mask
    secondary_seed = np.asarray(secondary_seed, dtype=bool) & domain_mask & (~primary_seed)
    primary_score = np.asarray(primary_score, dtype=np.float32)
    secondary_score = np.asarray(secondary_score, dtype=np.float32)

    if not np.any(primary_seed) and not np.any(secondary_seed):
        empty = np.zeros_like(domain_mask, dtype=bool)
        return empty, empty
    if not np.any(primary_seed):
        return np.zeros_like(domain_mask, dtype=bool), domain_mask.astype(bool)
    if not np.any(secondary_seed):
        return domain_mask.astype(bool), np.zeros_like(domain_mask, dtype=bool)

    secondary_candidate = (
        domain_mask
        & np.isfinite(primary_score)
        & np.isfinite(secondary_score)
        & (secondary_score > primary_score)
    )
    secondary_kept = connected_components_touching_seed(
        mask=(secondary_candidate | secondary_seed),
        seed=secondary_seed,
    )
    primary_kept = connected_components_touching_seed(
        mask=((domain_mask & (~secondary_kept)) | primary_seed),
        seed=primary_seed,
    )
    secondary_final = domain_mask & (~primary_kept)
    return primary_kept.astype(bool), secondary_final.astype(bool)


def partition_mask_by_intensity_with_seed_components(
    mask: np.ndarray,
    primary_seed: np.ndarray,
    secondary_seed: np.ndarray,
    primary_score: np.ndarray,
    secondary_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    primary_seed = np.asarray(primary_seed, dtype=bool) & mask
    secondary_seed = np.asarray(secondary_seed, dtype=bool) & mask & (~primary_seed)
    primary_score = np.asarray(primary_score, dtype=np.float32)
    secondary_score = np.asarray(secondary_score, dtype=np.float32)

    if not np.any(primary_seed) and not np.any(secondary_seed):
        empty = np.zeros_like(mask, dtype=bool)
        return empty, empty
    if not np.any(primary_seed):
        return np.zeros_like(mask, dtype=bool), mask.astype(bool)
    if not np.any(secondary_seed):
        return mask.astype(bool), np.zeros_like(mask, dtype=bool)

    secondary_wins = (
        mask
        & np.isfinite(primary_score)
        & np.isfinite(secondary_score)
        & (secondary_score > primary_score)
        & (~primary_seed)
    )
    primary_wins = mask & (~secondary_wins)

    secondary_core = connected_components_touching_seed(
        mask=(secondary_wins | secondary_seed),
        seed=secondary_seed,
    )
    primary_core = connected_components_touching_seed(
        mask=(primary_wins | primary_seed),
        seed=primary_seed,
    )

    unlabeled = mask & (~secondary_core) & (~primary_core)
    secondary_fill = (
        unlabeled
        & np.isfinite(primary_score)
        & np.isfinite(secondary_score)
        & (secondary_score > primary_score)
    )

    secondary_region = connected_components_touching_seed(
        mask=(secondary_core | secondary_fill | secondary_seed),
        seed=secondary_seed,
    )
    primary_region = mask & (~secondary_region)
    primary_region = connected_components_touching_seed(
        mask=(primary_region | primary_seed),
        seed=primary_seed,
    )
    secondary_region = mask & (~primary_region)

    return primary_region.astype(bool), secondary_region.astype(bool)


def donor_halo_mask(
    donor_positive: np.ndarray,
    organoid_mask: np.ndarray,
    radius_px: float = 80.0,
) -> np.ndarray:
    donor_positive = np.asarray(donor_positive, dtype=bool)
    organoid_mask = np.asarray(organoid_mask, dtype=bool)
    if not np.any(donor_positive):
        return np.zeros_like(organoid_mask, dtype=bool)
    distances = ndi.distance_transform_edt(~donor_positive)
    halo = organoid_mask & (distances <= float(radius_px))
    return halo.astype(bool)


def donor_adjacent_foxf1_component(
    foxf1_positive: np.ndarray,
    donor_positive: np.ndarray,
    organoid_mask: np.ndarray,
    radius_px: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    foxf1_positive = np.asarray(foxf1_positive, dtype=bool)
    halo = donor_halo_mask(
        donor_positive=np.asarray(donor_positive, dtype=bool),
        organoid_mask=np.asarray(organoid_mask, dtype=bool),
        radius_px=float(radius_px),
    )
    if not np.any(foxf1_positive) or not np.any(halo):
        return np.zeros_like(foxf1_positive, dtype=bool), halo.astype(bool)
    labels, _ = ndi.label(foxf1_positive)
    keep_labels = np.unique(labels[halo & (labels > 0)])
    if keep_labels.size == 0:
        return np.zeros_like(foxf1_positive, dtype=bool), halo.astype(bool)
    filtered = np.isin(labels, keep_labels)
    return filtered.astype(bool), halo.astype(bool)


def pixel_class_label(code: int) -> str:
    return PIXEL_CLASS_LABELS.get(int(code), f"unknown_{int(code)}")


def sample_masked_coordinates(
    mask: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0 or int(sample_size) <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    if len(coords) <= int(sample_size):
        return coords.astype(np.int32)
    choice = rng.choice(len(coords), size=int(sample_size), replace=False)
    return coords[np.asarray(choice, dtype=np.int64)].astype(np.int32)


def sample_masked_values(
    values: np.ndarray,
    mask: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coords = sample_masked_coordinates(mask=mask, sample_size=sample_size, rng=rng)
    if coords.size == 0:
        return np.zeros((0,), dtype=np.float32)
    vals = np.asarray(values, dtype=np.float32)[coords[:, 0], coords[:, 1]]
    vals = vals[np.isfinite(vals)]
    return vals.astype(np.float32)


def density_map_table(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    map_name: str,
    group_col: str = "pixel_class_label",
    bins: int = 96,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
) -> pd.DataFrame:
    finite = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.DataFrame(
            columns=[
                "map_name",
                "group_label",
                "x_bin_index",
                "y_bin_index",
                "x_bin_start",
                "x_bin_end",
                "y_bin_start",
                "y_bin_end",
                "count",
                "x_transform",
                "y_transform",
            ]
        )

    x_all = np.log1p(np.clip(finite[x_col].to_numpy(dtype=np.float64), a_min=0.0, a_max=None))
    y_all = np.log1p(np.clip(finite[y_col].to_numpy(dtype=np.float64), a_min=0.0, a_max=None))
    x_lo = float(np.quantile(x_all, float(quantile_low)))
    x_hi = float(np.quantile(x_all, float(quantile_high)))
    y_lo = float(np.quantile(y_all, float(quantile_low)))
    y_hi = float(np.quantile(y_all, float(quantile_high)))
    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0
    x_edges = np.linspace(x_lo, x_hi, max(16, int(bins)) + 1, dtype=np.float64)
    y_edges = np.linspace(y_lo, y_hi, max(16, int(bins)) + 1, dtype=np.float64)

    out_rows: list[dict[str, Any]] = []
    for group_label in ["all_masked", *sorted(df[group_col].dropna().unique().tolist())]:
        if group_label == "all_masked":
            sub = df.copy()
        else:
            sub = df.loc[df[group_col] == group_label].copy()
        if sub.empty:
            continue
        x = np.log1p(np.clip(sub[x_col].to_numpy(dtype=np.float64), a_min=0.0, a_max=None))
        y = np.log1p(np.clip(sub[y_col].to_numpy(dtype=np.float64), a_min=0.0, a_max=None))
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite_mask):
            continue
        counts, _, _ = np.histogram2d(x[finite_mask], y[finite_mask], bins=[x_edges, y_edges])
        nz = np.argwhere(counts > 0)
        for idx in nz:
            x_idx = int(idx[0])
            y_idx = int(idx[1])
            out_rows.append(
                {
                    "map_name": str(map_name),
                    "group_label": str(group_label),
                    "x_bin_index": x_idx,
                    "y_bin_index": y_idx,
                    "x_bin_start": float(x_edges[x_idx]),
                    "x_bin_end": float(x_edges[x_idx + 1]),
                    "y_bin_start": float(y_edges[y_idx]),
                    "y_bin_end": float(y_edges[y_idx + 1]),
                    "count": int(counts[x_idx, y_idx]),
                    "x_transform": f"log1p({x_col})",
                    "y_transform": f"log1p({y_col})",
                }
            )
    return pd.DataFrame(out_rows)


__all__ = [
    "CANONICAL_CHANNEL_ORDER",
    "CHANNEL_DISPLAY_NAMES",
    "ImageStack",
    "MANIFEST_COLUMNS",
    "PIXEL_CLASS_LABELS",
    "background_correct_signal",
    "background_reference_mask",
    "build_raw_input_manifest",
    "channel_index",
    "classify_host_donor_from_binary_calls",
    "classify_host_donor_pixels",
    "connected_components_touching_seed",
    "compute_dapi_floor",
    "density_map_table",
    "assign_domain_by_intensity_with_seed_connectivity",
    "donor_graft_component",
    "half_gaussian_null_from_values",
    "half_gaussian_threshold_from_values",
    "largest_connected_component",
    "load_transplant_stack",
    "normalize_signal_with_null",
    "otsu_threshold_from_values",
    "partition_mask_by_intensity_with_seed_components",
    "partition_domain_by_seeds",
    "pixel_class_label",
    "reporter_over_dapi",
    "sample_masked_coordinates",
    "sample_masked_values",
    "segment_whole_organoid_from_dapi_with_debug",
]
