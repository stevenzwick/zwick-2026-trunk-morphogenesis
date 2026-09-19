"""Helpers for the FOXF1/BMP4 day-2 pixel-quantification pipeline.

This workspace intentionally follows the same broad shape as the neighboring
LPM transplant analysis, but the biological readout here is simpler:

- canonical raw inputs are the `.czi` files under `data/with DAPI/`
- DAPI defines the analyzable tissue mask on each z plane
- FOXF1 and BMP4 are quantified as background-corrected reporter intensities
- positivity review compares raw corrected intensity thresholds with
  background-sigma-standardized thresholds
- all z planes are used, while final density outputs weight each image stack
  equally so thicker stacks do not dominate manuscript-facing summaries
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Callable, Optional, Sequence

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
    "foxf1",
    "bmp4",
]

RAW_TO_CANONICAL_CHANNELS = {
    "TL Brightfield": "brightfield",
    "DAPI": "dapi",
    "mCherry": "foxf1",
    "TagYFP": "bmp4",
}

CHANNEL_DISPLAY_NAMES = {
    "brightfield": "Brightfield",
    "dapi": "DAPI",
    "foxf1": "FOXF1-RFP",
    "bmp4": "BMP4-YFP",
}

MANIFEST_COLUMNS = [
    "image_id",
    "file_id",
    "position_label",
    "file_name",
    "file_path",
    "paired_review_tif_name",
    "paired_review_tif_path",
    "selected_review_z_1based",
    "selected_review_z_0based",
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
    "paired_tif_axes",
    "paired_tif_size_c",
    "paired_tif_size_z",
    "paired_tif_size_y",
    "paired_tif_size_x",
]

REVIEW_TIF_RE = re.compile(r"^(?P<image_id>\d+)\(z(?P<z_index_1based>\d+)\)\.tif$")


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


def _sort_key(path: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.stem):08d}")
    except Exception:
        return (1, path.stem)


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


def map_raw_channels_to_canonical(raw_channel_names: Sequence[str]) -> list[str]:
    out: list[str] = []
    for idx, raw_name in enumerate(raw_channel_names):
        mapped = RAW_TO_CANONICAL_CHANNELS.get(str(raw_name))
        if mapped is None:
            fallback = _canonical_channels_for_count(len(raw_channel_names))[idx]
            mapped = fallback
        out.append(str(mapped))
    return out


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
    canonical_channel_names = map_raw_channels_to_canonical(raw_channel_names)
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


def paired_review_tif_info(path: Path, data_dir: Path) -> dict[str, Any]:
    match_path = None
    selected_review_z_1based = np.nan
    selected_review_z_0based = np.nan
    for tif_path in sorted(data_dir.glob(f"{path.stem}(z*).tif")):
        match = REVIEW_TIF_RE.match(tif_path.name)
        if match:
            match_path = tif_path
            selected_review_z_1based = int(match.group("z_index_1based"))
            selected_review_z_0based = int(selected_review_z_1based - 1)
            break

    if match_path is None:
        return {
            "paired_review_tif_name": "",
            "paired_review_tif_path": "",
            "selected_review_z_1based": np.nan,
            "selected_review_z_0based": np.nan,
            "paired_tif_axes": "",
            "paired_tif_size_c": np.nan,
            "paired_tif_size_z": np.nan,
            "paired_tif_size_y": np.nan,
            "paired_tif_size_x": np.nan,
        }

    with tifffile.TiffFile(match_path) as tif:
        series = tif.series[0]
        tif_shape = tuple(int(x) for x in series.shape)
        tif_axes = str(getattr(series, "axes", ""))

    c_count = np.nan
    z_count = np.nan
    y_count = np.nan
    x_count = np.nan
    if len(tif_shape) == 4:
        z_count, c_count, y_count, x_count = tif_shape
    elif len(tif_shape) == 3:
        c_count, y_count, x_count = tif_shape

    return {
        "paired_review_tif_name": match_path.name,
        "paired_review_tif_path": match_path.as_posix(),
        "selected_review_z_1based": selected_review_z_1based,
        "selected_review_z_0based": selected_review_z_0based,
        "paired_tif_axes": tif_axes,
        "paired_tif_size_c": c_count,
        "paired_tif_size_z": z_count,
        "paired_tif_size_y": y_count,
        "paired_tif_size_x": x_count,
    }


def build_raw_input_manifest(data_dir: Path, root: Optional[Path] = None) -> pd.DataFrame:
    root = Path(root or data_dir.parent)
    rows: list[dict[str, Any]] = []
    file_id = 1
    for path in sorted(Path(data_dir).glob("*.czi"), key=_sort_key):
        if not path.is_file():
            continue
        stack = load_czi_stack(path)
        rel_path = path.resolve().relative_to(root.resolve()).as_posix()
        row = {
            "image_id": f"img{file_id:02d}",
            "file_id": int(file_id),
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
        row.update(paired_review_tif_info(path=path, data_dir=Path(data_dir)))
        rows.append(row)
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
    inner_radius = max(0.0, float(annulus_inner_radius_px))
    outer_radius = max(inner_radius, float(annulus_outer_radius_px))
    background = ~mask
    distance_to_mask = ndi.distance_transform_edt(background)
    ring = background & (distance_to_mask > inner_radius) & (distance_to_mask <= outer_radius)
    if int(ring.sum()) < int(min_ring_pixels):
        ring = background & (distance_to_mask > outer_radius)
    if int(ring.sum()) < int(min_ring_pixels):
        ring = background
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


def robust_sigma_from_values(
    values: np.ndarray,
    min_sigma: float = 1.0e-3,
) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(min_sigma)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(arr))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(min_sigma)
    return float(max(float(min_sigma), sigma))


def estimate_background_sigma(
    corrected_signal: np.ndarray,
    background_mask: np.ndarray,
    min_sigma: float = 1.0e-3,
) -> float:
    signal = np.asarray(corrected_signal, dtype=np.float32)
    bg_mask = np.asarray(background_mask, dtype=bool)
    return robust_sigma_from_values(signal[bg_mask], min_sigma=float(min_sigma))


def standardize_by_background_sigma(
    corrected_signal: np.ndarray,
    background_sigma: float,
) -> np.ndarray:
    signal = np.asarray(corrected_signal, dtype=np.float32)
    sigma = float(background_sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    out = np.full_like(signal, np.nan, dtype=np.float32)
    valid = np.isfinite(signal)
    out[valid] = signal[valid] / np.float32(sigma)
    return out


def estimate_half_gaussian_null_from_values(
    values: np.ndarray,
    hist_bins: int = 2048,
    smooth_sigma_bins: float = 4.0,
    quantile_low: float = 0.001,
    quantile_high: float = 0.999,
    edge_padding_fraction: float = 0.05,
    min_sigma: float = 1.0e-6,
) -> dict[str, Any]:
    diag = half_gaussian_threshold_from_values(
        values=np.asarray(values, dtype=np.float32),
        z=0.0,
        hist_bins=int(hist_bins),
        smooth_sigma_bins=float(smooth_sigma_bins),
        quantile_low=float(quantile_low),
        quantile_high=float(quantile_high),
        edge_padding_fraction=float(edge_padding_fraction),
    )
    mu = float(diag["baseline_location"])
    sigma = float(max(float(min_sigma), float(diag["baseline_scale"])))
    out = dict(diag)
    out["mu"] = mu
    out["sigma"] = sigma
    return out


def positive_signal(
    corrected_signal: np.ndarray,
) -> np.ndarray:
    signal = np.asarray(corrected_signal, dtype=np.float32)
    return np.clip(signal, a_min=0.0, a_max=None).astype(np.float32)


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


def reporter_over_dapi_signed(
    corrected_reporter: np.ndarray,
    corrected_dapi: np.ndarray,
    organoid_mask: np.ndarray,
    dapi_floor: float,
) -> np.ndarray:
    reporter = np.asarray(corrected_reporter, dtype=np.float32)
    dapi = np.clip(np.asarray(corrected_dapi, dtype=np.float32), a_min=0.0, a_max=None)
    denom = np.maximum(dapi, float(dapi_floor)).astype(np.float32)
    ratio = np.full_like(reporter, np.nan, dtype=np.float32)
    valid = np.asarray(organoid_mask, dtype=bool) & np.isfinite(reporter) & np.isfinite(denom)
    ratio[valid] = reporter[valid] / denom[valid]
    return ratio


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


def allocate_stack_sample_counts(mask_sizes: Sequence[int], target_total: int) -> list[int]:
    sizes = np.asarray(mask_sizes, dtype=np.int64)
    target = int(max(0, target_total))
    if target <= 0 or sizes.size == 0:
        return [0 for _ in sizes]
    total_available = int(np.sum(sizes))
    if total_available <= target:
        return [int(x) for x in sizes]

    fractions = sizes.astype(np.float64) / float(total_available)
    raw_alloc = fractions * float(target)
    alloc = np.floor(raw_alloc).astype(np.int64)
    alloc = np.minimum(alloc, sizes)

    remaining = target - int(np.sum(alloc))
    if remaining > 0:
        remainders = raw_alloc - alloc.astype(np.float64)
        order = np.argsort(-remainders)
        for idx in order:
            if remaining <= 0:
                break
            if alloc[idx] >= sizes[idx]:
                continue
            alloc[idx] += 1
            remaining -= 1
    return [int(x) for x in alloc]


def compute_axis_scale(
    values: np.ndarray,
    quantile: float = 0.995,
    min_scale: float = 1.0e-6,
) -> float:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if arr.size == 0:
        return float(min_scale)
    scale = float(np.quantile(arr, float(quantile)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanmax(arr)) if arr.size else float(min_scale)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(min_scale)
    return float(max(float(min_scale), scale))


def _hist2d_counts(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=[int(bins), int(bins)],
        range=[list(x_range), list(y_range)],
    )
    return counts.astype(np.float64), x_edges, y_edges


def balanced_density_map_table(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    map_name: str,
    stack_col: str = "image_id",
    group_specs: Optional[dict[str, Callable[[pd.DataFrame], pd.Series]]] = None,
    bins: int = 96,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    if group_specs is None:
        group_specs = {"all_masked": lambda frame: pd.Series(True, index=frame.index)}

    rows: list[dict[str, Any]] = []
    for group_label, selector in group_specs.items():
        group_mask = selector(df)
        sub = df.loc[group_mask].copy()
        if sub.empty:
            continue

        finite_mask = (
            np.isfinite(sub[x_col].to_numpy(dtype=np.float64))
            & np.isfinite(sub[y_col].to_numpy(dtype=np.float64))
        )
        sub = sub.loc[finite_mask].copy()
        if sub.empty:
            continue

        x_all = np.clip(sub[x_col].to_numpy(dtype=np.float64), x_range[0], x_range[1])
        y_all = np.clip(sub[y_col].to_numpy(dtype=np.float64), y_range[0], y_range[1])

        pooled_counts, x_edges, y_edges = _hist2d_counts(
            x=x_all,
            y=y_all,
            bins=int(bins),
            x_range=x_range,
            y_range=y_range,
        )
        pooled_density = pooled_counts / pooled_counts.sum() if pooled_counts.sum() > 0 else pooled_counts

        equal_density_sum = np.zeros_like(pooled_density, dtype=np.float64)
        equal_presence = np.zeros_like(pooled_density, dtype=np.int64)
        equal_counts = np.zeros_like(pooled_density, dtype=np.float64)
        stack_count = 0
        for _, stack_df in sub.groupby(stack_col, sort=True):
            x_stack = np.clip(stack_df[x_col].to_numpy(dtype=np.float64), x_range[0], x_range[1])
            y_stack = np.clip(stack_df[y_col].to_numpy(dtype=np.float64), y_range[0], y_range[1])
            stack_counts, _, _ = _hist2d_counts(
                x=x_stack,
                y=y_stack,
                bins=int(bins),
                x_range=x_range,
                y_range=y_range,
            )
            if stack_counts.sum() <= 0:
                continue
            equal_density_sum += stack_counts / stack_counts.sum()
            equal_presence += (stack_counts > 0).astype(np.int64)
            equal_counts += stack_counts
            stack_count += 1
        if stack_count > 0:
            equal_density = equal_density_sum / float(stack_count)
        else:
            equal_density = np.zeros_like(pooled_density, dtype=np.float64)

        density_payloads = [
            ("raw_pooled", pooled_counts, pooled_density, np.where(pooled_counts > 0, 1, 0), 1),
            ("equal_stack_weight", equal_counts, equal_density, equal_presence, stack_count),
        ]

        for weighting_mode, counts_arr, density_arr, presence_arr, n_stacks in density_payloads:
            nz = np.argwhere(density_arr > 0)
            for idx in nz:
                x_idx = int(idx[0])
                y_idx = int(idx[1])
                rows.append(
                    {
                        "map_name": str(map_name),
                        "weighting_mode": str(weighting_mode),
                        "group_label": str(group_label),
                        "x_bin_index": x_idx,
                        "y_bin_index": y_idx,
                        "x_bin_start": float(x_edges[x_idx]),
                        "x_bin_end": float(x_edges[x_idx + 1]),
                        "y_bin_start": float(y_edges[y_idx]),
                        "y_bin_end": float(y_edges[y_idx + 1]),
                        "count": int(round(float(counts_arr[x_idx, y_idx]))),
                        "density_value": float(density_arr[x_idx, y_idx]),
                        "stack_presence_count": int(presence_arr[x_idx, y_idx]),
                        "n_stacks": int(n_stacks),
                        "x_transform": str(x_col),
                        "y_transform": str(y_col),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "map_name",
                "weighting_mode",
                "group_label",
                "x_bin_index",
                "y_bin_index",
                "x_bin_start",
                "x_bin_end",
                "y_bin_start",
                "y_bin_end",
                "count",
                "density_value",
                "density_value_maxnorm",
                "stack_presence_count",
                "n_stacks",
                "x_transform",
                "y_transform",
            ]
        )

    out["density_value_maxnorm"] = (
        out.groupby(["map_name", "weighting_mode", "group_label"])["density_value"]
        .transform(lambda col: col / col.max() if float(col.max()) > 0 else col)
        .astype(float)
    )
    return out


__all__ = [
    "CANONICAL_CHANNEL_ORDER",
    "CHANNEL_DISPLAY_NAMES",
    "ImageStack",
    "MANIFEST_COLUMNS",
    "RAW_TO_CANONICAL_CHANNELS",
    "REVIEW_TIF_RE",
    "allocate_stack_sample_counts",
    "background_correct_signal",
    "estimate_background_sigma",
    "background_reference_mask",
    "balanced_density_map_table",
    "build_raw_input_manifest",
    "channel_index",
    "compute_axis_scale",
    "compute_dapi_floor",
    "extract_czi_acquisition_timestamp_utc",
    "extract_czi_channels",
    "extract_czi_scale_um",
    "estimate_half_gaussian_null_from_values",
    "half_gaussian_threshold_from_values",
    "load_czi_stack",
    "map_raw_channels_to_canonical",
    "normalize_to_czyx",
    "otsu_threshold_from_values",
    "paired_review_tif_info",
    "positive_signal",
    "reporter_over_dapi",
    "reporter_over_dapi_signed",
    "robust_sigma_from_values",
    "sample_masked_coordinates",
    "sample_masked_values",
    "segment_whole_organoid_from_dapi_with_debug",
    "standardize_by_background_sigma",
]
