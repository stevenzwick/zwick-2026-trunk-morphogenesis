#!/usr/bin/env python3
"""
Quantification helpers for aligned live/fixed FOXF1 bead experiments.

Reference logic:
  - bead annotations are authored in LIVE large-image coordinates
  - live and fixed SMALL images are measured in their own native pixel grids
  - fixed DAPI defines the cyst mask
  - the fixed mask is rigidly transferred into LIVE small-image space
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# czifile cannot be installed beside this project's pinned numpy, and mis-frames odd-sized
# acquisitions. czi_compat presents czifile's interface over libCZI. See src/trunk_morph_ref/czi_compat.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from src.trunk_morph_ref import czi_compat as czifile
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

# `scripts` is a package under the assay directory; put that directory on
# the path so this file runs from anywhere, as the READMEs show it being run.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from scripts import live_fixed_alignment as lfa
from scripts import manual_well_annotation as mwa
from scripts import pipeline_common as common


EPS = 1e-6
DEFAULT_FIXED_DAPI_MASK_BLUR_SIGMA_PX = 2.0
DEFAULT_FIXED_DAPI_THRESHOLD_SIGMA = 4.0
DEFAULT_FIXED_DAPI_CORE_BLUR_SIGMA_PX = 12.0
DEFAULT_FIXED_DAPI_CORE_THRESHOLD_SCALE = 0.90
DEFAULT_FIXED_DAPI_CORE_THRESHOLD_SIGMA = 6.0
DEFAULT_FIXED_DAPI_MONOLAYER_TOUCH_RADIUS_PX = 18
FIXED_SMALL_SELECTED_Z_BY_POSITION = {
    # Selected from the round-2 residual-focus QC; all other fixed small scenes are single-plane.
    "1-1": 0,
}
FIXED_DAPI_POSITION_EXCLUSION_BOXES = {
    "2-6": [
        {
            "row_min": 60,
            "row_max": 240,
            "col_min": 0,
            "col_max": 260,
            "reason": "top_left_artifact",
        }
    ],
}
APPROVED_FINAL04_FIXED_DAPI_CORE_THRESHOLD_SCALE = 0.80
APPROVED_FINAL04_FIXED_DAPI_CORE_FILL_HOLES = False
APPROVED_FINAL04_FIXED_DAPI_CORE_HOLE_AREA_PX = 0
APPROVED_FINAL04_FIXED_DAPI_ROUND2_BLUR_SIGMA_PX = 1.0
APPROVED_FINAL04_FIXED_DAPI_ROUND2_THRESHOLD_SIGMA = 4.0
APPROVED_FINAL04_FIXED_DAPI_ROUND2_MIN_AREA_PX = 40
APPROVED_FINAL04_FIXED_DAPI_ROUND2_CLOSING_RADIUS_PX = 1
APPROVED_FINAL04_FIXED_DAPI_ROUND2_BLUR_MODE = "reflect"
APPROVED_FINAL04_FIXED_DAPI_ROUND2_SUPPRESS_POSITIONS = {
    "3-3": "No evident monolayer cells. Round 2 is intentionally skipped.",
}
APPROVED_FINAL04_FIXED_DAPI_ROUND2_MANUAL_ARTIFACT_EXCLUSION = {
    "2-6": {
        "dilation_radius_px": 40,
        "reason": "top_left_artifact_shape_dilation",
    },
}
APPROVED_FINAL04_FIXED_DAPI_ROUND2_MANUAL_COMPONENT_ROIS = {
    "1-3": [
        {
            "row_min": 0,
            "row_max": 150,
            "col_min": 0,
            "col_max": 210,
            "reason": "top_left_false_positive_region",
            "mode": "mask_intersection",
        }
    ],
}

FIXED_SMALL_CENTROID_COLUMNS = [
    "image_id",
    "cohort_id",
    "canonical_position",
    "fixed_small_file_path",
    "fixed_large_file_path",
    "live_large_file_path",
    "annotation_status",
    "mapping_status",
    "centroid_index",
    "centroid_live_large_x_px",
    "centroid_live_large_y_px",
    "centroid_fixed_small_x_px",
    "centroid_fixed_small_y_px",
    "inside_fixed_small_fov",
    "fixed_small_shape_y_px",
    "fixed_small_shape_x_px",
    "updated_at",
]

MASK_SUMMARY_COLUMNS = [
    "canonical_position",
    "live_image_id",
    "fixed_image_id",
    "status",
    "fixed_mask_area_px",
    "fixed_mask_fraction",
    "live_mask_area_px",
    "live_mask_fraction",
    "fixed_mask_components",
    "fixed_dapi_threshold",
    "fixed_dapi_bg_mu",
    "fixed_dapi_bg_sigma",
    "fixed_dapi_threshold_sigma",
    "fixed_dapi_core_threshold",
    "fixed_dapi_core_bg_mu",
    "fixed_dapi_core_bg_sigma",
    "fixed_dapi_core_threshold_sigma",
    "fixed_dapi_core_blur_sigma_px",
    "fixed_dapi_blur_sigma_px",
    "fixed_artifact_exclusion_applied",
    "fixed_artifact_exclusion_area_px",
    "fixed_artifact_exclusion_reasons",
    "n_live_beads_total",
    "n_live_beads_inside",
    "n_fixed_beads_total",
    "n_fixed_beads_inside",
    "live_qc_png_path",
    "fixed_qc_png_path",
    "combined_qc_png_path",
    "updated_at",
]


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_channel_from_background(
    image: np.ndarray,
    mu_bg_raw: float,
    sigma_bg_raw: float,
) -> np.ndarray:
    sigma = max(float(EPS), float(sigma_bg_raw))
    out = (np.asarray(image, dtype=np.float32) - float(mu_bg_raw)) / sigma
    return np.maximum(out, 0.0).astype(np.float32)


def prepare_fixed_dapi_for_mask(
    dapi_raw: np.ndarray,
    blur_sigma_px: float = DEFAULT_FIXED_DAPI_MASK_BLUR_SIGMA_PX,
    **_ignored,
) -> np.ndarray:
    return filters.gaussian(
        np.asarray(dapi_raw, dtype=np.float32),
        sigma=float(blur_sigma_px),
        preserve_range=True,
    ).astype(np.float32)


def _position_specific_exclusion_mask(
    shape_yx: Tuple[int, int],
    canonical_position: Optional[Union[str, int]],
) -> Tuple[np.ndarray, List[str]]:
    mask = np.zeros(shape_yx, dtype=bool)
    reasons: List[str] = []
    if canonical_position is None:
        return mask, reasons
    specs = FIXED_DAPI_POSITION_EXCLUSION_BOXES.get(str(canonical_position), [])
    if not specs:
        return mask, reasons
    h, w = shape_yx
    for spec in specs:
        r0 = max(0, int(spec["row_min"]))
        r1 = min(h, int(spec["row_max"]))
        c0 = max(0, int(spec["col_min"]))
        c1 = min(w, int(spec["col_max"]))
        if r0 < r1 and c0 < c1:
            mask[r0:r1, c0:c1] = True
            reasons.append(str(spec.get("reason", "manual_exclusion")))
    return mask, reasons


def _collapse_to_czyx(arr: np.ndarray, axes: str) -> Tuple[np.ndarray, str]:
    """Convert any CZI stack variant to (C, Z, Y, X), max-projecting over non-C/Z/Y/X axes."""
    out = arr
    out_axes = axes

    if "C" not in out_axes or "Y" not in out_axes or "X" not in out_axes:
        raise ValueError(f"Missing required axes in CZI stack. axes={axes}")

    c_idx = out_axes.index("C")
    out = np.moveaxis(out, c_idx, 0)
    out_axes = "C" + out_axes[:c_idx] + out_axes[c_idx + 1 :]

    if "Z" not in out_axes:
        out = np.expand_dims(out, axis=1)
        out_axes = "CZ" + out_axes[1:]
    else:
        z_idx = out_axes.index("Z")
        out = np.moveaxis(out, z_idx, 1)
        out_axes = "CZ" + out_axes[1:z_idx] + out_axes[z_idx + 1 :]

    for ax_name in list(out_axes):
        if ax_name in ("C", "Z", "Y", "X"):
            continue
        axis = out_axes.index(ax_name)
        out = out.max(axis=axis)
        out_axes = out_axes.replace(ax_name, "")

    perm = [out_axes.index("C"), out_axes.index("Z"), out_axes.index("Y"), out_axes.index("X")]
    out = np.transpose(out, axes=perm)
    return out.astype(np.float32), "CZYX"


def read_czi_with_optional_fixed_small_plane_selection(
    path: Union[str, Path],
    canonical_position: Optional[Union[str, int]] = None,
) -> common.CziImage:
    path = Path(path)
    position_key = None if canonical_position is None else str(canonical_position)
    selected_z = FIXED_SMALL_SELECTED_Z_BY_POSITION.get(position_key)
    if selected_z is None:
        return common.read_czi(path)

    with czifile.CziFile(path) as czi:
        raw = czi.asarray()
        raw_axes = str(czi.axes)
        metadata = czi.metadata(raw=False)

    squeezed, squeezed_axes = common._squeeze_axes(raw, raw_axes)
    czyx, _ = _collapse_to_czyx(squeezed, squeezed_axes)
    z_idx = int(np.clip(int(selected_z), 0, czyx.shape[1] - 1))
    cyx = np.asarray(czyx[:, z_idx], dtype=np.float32)
    channels = common._extract_channels(metadata, c_count=cyx.shape[0])
    pixel_um_x, pixel_um_y = common._extract_pixel_um(metadata)

    return common.CziImage(
        path=path,
        channels=channels,
        channel_images=cyx,
        pixel_um_x=float(pixel_um_x),
        pixel_um_y=float(pixel_um_y),
        raw_axes=raw_axes,
        raw_shape=tuple(raw.shape),
    )


def load_czi_bundle_with_z(
    path: Union[str, Path],
) -> dict:
    """Load a CZI file while preserving Z and exposing max projections for context/QC."""
    path = Path(path)
    with czifile.CziFile(path) as czi:
        raw = czi.asarray()
        raw_axes = str(czi.axes)
        metadata = czi.metadata(raw=False)

    squeezed, squeezed_axes = common._squeeze_axes(raw, raw_axes)
    czyx, _ = _collapse_to_czyx(squeezed, squeezed_axes)
    channels = common._extract_channels(metadata, c_count=czyx.shape[0])
    pixel_um_x, pixel_um_y = common._extract_pixel_um(metadata)

    return {
        "path": path,
        "channels": list(channels),
        "channel_stack_czyx": czyx.astype(np.float32),
        "channel_max_cyx": np.max(czyx, axis=1).astype(np.float32),
        "pixel_um_x": float(pixel_um_x),
        "pixel_um_y": float(pixel_um_y),
        "raw_axes": raw_axes,
        "raw_shape": tuple(raw.shape),
    }


def selected_fixed_small_dapi_plane(
    dapi_stack: np.ndarray,
    canonical_position: Optional[Union[str, int]] = None,
) -> Tuple[int, np.ndarray]:
    """Return the single DAPI plane approved for downstream small-image analysis."""
    stack = np.asarray(dapi_stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"dapi_stack must have shape (Z, Y, X); got {stack.shape}")
    if stack.shape[0] <= 1:
        return 0, stack[0].astype(np.float32)
    position_key = None if canonical_position is None else str(canonical_position)
    selected_z = FIXED_SMALL_SELECTED_Z_BY_POSITION.get(position_key, 0)
    selected_z = int(np.clip(int(selected_z), 0, stack.shape[0] - 1))
    return selected_z, stack[selected_z].astype(np.float32)


def background_histogram_from_pixels(
    image: np.ndarray,
    n_bins: int = 2048,
    exclude_exact_zero_pixels: bool = True,
) -> dict:
    finite = np.asarray(image, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if bool(exclude_exact_zero_pixels):
        finite = finite[finite != 0]
    if finite.size == 0:
        raise RuntimeError("No finite pixels available for background histogram.")

    sample = finite if finite.size <= 4096 else finite[:: max(1, finite.size // 4096)]
    integer_like = bool(np.all(np.abs(sample - np.rint(sample)) <= 1e-6))
    global_min = float(np.min(finite))
    global_max = float(np.max(finite))
    if global_max <= global_min:
        global_max = global_min + 1.0

    if bool(integer_like):
        lo_i = int(np.floor(global_min))
        hi_i = int(np.ceil(global_max))
        edges = np.arange(lo_i - 0.5, hi_i + 1.5, 1.0, dtype=np.float64)
    else:
        edges = np.linspace(global_min, global_max, int(n_bins) + 1, dtype=np.float64)
    counts, _ = np.histogram(finite, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "edges": edges,
        "centers": centers,
        "counts": counts.astype(np.int64),
        "global_min": float(global_min),
        "global_max": float(global_max),
        "n_pixels": int(finite.size),
        "n_images": 1,
        "n_bins": int(len(counts)),
        "integer_like_bins": bool(integer_like),
    }


def fit_background_from_image_pixels(
    image: np.ndarray,
    n_bins: int = 2048,
    smooth_sigma_bins: float = 6.0,
    exclude_exact_zero_pixels: bool = True,
) -> dict:
    hist = background_histogram_from_pixels(
        image=image,
        n_bins=int(n_bins),
        exclude_exact_zero_pixels=bool(exclude_exact_zero_pixels),
    )
    fit = common.estimate_background_half_gaussian(
        hist=hist,
        smooth_sigma_bins=float(smooth_sigma_bins),
    )
    fit["exclude_exact_zero_pixels"] = bool(exclude_exact_zero_pixels)
    if bool(exclude_exact_zero_pixels):
        fit["fit_method"] = f"{fit['fit_method']};exclude_exact_zero_pixels"
    return fit


def _connected_components_touching(
    mask: np.ndarray,
    anchor_mask: np.ndarray,
) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    anchor_bool = np.asarray(anchor_mask, dtype=bool)
    labels, n_labels = ndi.label(mask_bool)
    if int(n_labels) <= 0:
        return np.zeros_like(mask_bool, dtype=bool)
    keep = np.zeros(int(n_labels) + 1, dtype=bool)
    touched = np.unique(labels[anchor_bool])
    keep[touched] = True
    keep[0] = False
    return keep[labels].astype(bool)


def apply_illumination_field(
    image: np.ndarray,
    illumination_field: Optional[np.ndarray],
) -> np.ndarray:
    if illumination_field is None:
        return np.asarray(image, dtype=np.float32)
    field = _match_field_shape_to_image(
        illumination_field=np.asarray(illumination_field, dtype=np.float32),
        image_shape_yx=np.asarray(image).shape,
    )
    return (
        np.asarray(image, dtype=np.float32)
        / np.clip(field, float(EPS), None)
    ).astype(np.float32)


def smooth_masked_image(
    masked_mean: np.ndarray,
    observed_mask: np.ndarray,
    sigma_px: float,
    mode: str = "nearest",
) -> np.ndarray:
    weighted = ndi.gaussian_filter(
        np.asarray(masked_mean, dtype=np.float32) * np.asarray(observed_mask, dtype=np.float32),
        sigma=float(sigma_px),
        mode=str(mode),
    )
    weight_sum = ndi.gaussian_filter(
        np.asarray(observed_mask, dtype=np.float32),
        sigma=float(sigma_px),
        mode=str(mode),
    )
    return (weighted / np.maximum(weight_sum, float(EPS))).astype(np.float32)


def _match_field_shape_to_image(
    illumination_field: np.ndarray,
    image_shape_yx: Tuple[int, int],
) -> np.ndarray:
    field = np.asarray(illumination_field, dtype=np.float32)
    target_h, target_w = map(int, image_shape_yx)
    if field.shape == (target_h, target_w):
        return field
    out = np.ones((target_h, target_w), dtype=np.float32)
    copy_h = min(target_h, int(field.shape[0]))
    copy_w = min(target_w, int(field.shape[1]))
    out[:copy_h, :copy_w] = field[:copy_h, :copy_w]
    return out


def _pad_to_shape(
    image: np.ndarray,
    target_shape_yx: Tuple[int, int],
    fill_value: float = 0.0,
) -> np.ndarray:
    arr = np.asarray(image)
    target_h, target_w = map(int, target_shape_yx)
    if arr.shape == (target_h, target_w):
        return arr
    out = np.full((target_h, target_w), fill_value, dtype=arr.dtype)
    copy_h = min(target_h, int(arr.shape[0]))
    copy_w = min(target_w, int(arr.shape[1]))
    out[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
    return out


def _resolve_project_root(
    root: Optional[Union[str, Path]],
    position_manifest: Optional[Union[str, Path, pd.DataFrame]] = None,
) -> Path:
    if root is not None:
        return Path(root).resolve()
    if position_manifest is not None:
        return common._infer_project_root_from_manifest(position_manifest)
    return Path.cwd().resolve()


def _resolve_project_path(path_like: Union[str, Path], project_root: Path) -> Path:
    return common._resolve_project_path(path_like, project_root)


def _filter_positions(
    position_manifest: Union[str, Path, pd.DataFrame],
    cohort_ids: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    pos_df = common.load_position_manifest(position_manifest)
    return common.filter_position_manifest(
        pos_df=pos_df,
        cohort_ids=cohort_ids,
        conditions=conditions,
        canonical_positions=canonical_positions,
    )


def estimate_background_for_manifest_subset(
    position_manifest: Union[str, Path, pd.DataFrame],
    cohort_ids: Sequence[str],
    channel_specs: Dict[str, Sequence[str]],
    conditions: Optional[Sequence[str]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    n_bins: int = 16384,
    smooth_sigma_bins: float = 6.0,
    exclude_exact_zero_pixels: bool = True,
    transformers: Optional[Dict[str, object]] = None,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    rows: List[Dict[str, object]] = []
    payload: Dict[str, dict] = {}
    for channel_name, keywords in channel_specs.items():
        if transformers is not None and str(channel_name) in transformers:
            hist = pooled_histogram_manifest_channel(
                position_manifest=position_manifest,
                cohort_ids=list(cohort_ids),
                channel_keywords=list(keywords),
                transform_fn=transformers[str(channel_name)],
                conditions=list(conditions) if conditions is not None else None,
                canonical_positions=list(canonical_positions) if canonical_positions is not None else None,
                n_bins=int(n_bins),
                exclude_exact_zero_pixels=bool(exclude_exact_zero_pixels),
            )
            fit = common.estimate_background_half_gaussian(
                hist=hist,
                smooth_sigma_bins=float(smooth_sigma_bins),
            )
            fit["exclude_exact_zero_pixels"] = bool(exclude_exact_zero_pixels)
            if bool(exclude_exact_zero_pixels):
                fit["fit_method"] = f"{fit['fit_method']};exclude_exact_zero_pixels"
        else:
            fit = common.estimate_global_channel_background_from_manifest(
                position_manifest=position_manifest,
                channel_keywords=list(keywords),
                cohort_ids=list(cohort_ids),
                conditions=list(conditions) if conditions is not None else None,
                canonical_positions=list(canonical_positions) if canonical_positions is not None else None,
                n_bins=int(n_bins),
                smooth_sigma_bins=float(smooth_sigma_bins),
                exclude_exact_zero_pixels=bool(exclude_exact_zero_pixels),
            )
        payload[str(channel_name)] = fit
        rows.append(
            {
                "channel": str(channel_name),
                "mu_bg_raw": float(fit["mu_bg_raw"]),
                "sigma_bg_raw": float(fit["sigma_bg_raw"]),
                "mode_idx": int(fit["mode_idx"]),
                "n_images": int(fit["n_images"]),
                "n_pixels": int(fit["n_pixels"]),
                "global_min": float(fit["global_min"]),
                "global_max": float(fit["global_max"]),
                "fit_method": str(fit["fit_method"]),
            }
        )
    return pd.DataFrame(rows), payload


def plot_background_fit_panels(
    fit_payloads: Dict[str, dict],
    channel_order: Optional[Sequence[str]] = None,
    xlim_by_channel: Optional[Dict[str, Tuple[float, float]]] = None,
    title_prefix: str = "",
    ncols: Optional[int] = None,
    panel_width: float = 6.0,
    panel_height: float = 4.4,
    show_sigma_guides: bool = True,
) -> plt.Figure:
    keys = list(channel_order) if channel_order is not None else list(fit_payloads.keys())
    n = max(1, len(keys))
    if ncols is None:
        ncols = n
    ncols = int(max(1, min(ncols, n)))
    nrows = int(np.ceil(n / float(ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, keys):
        fit = fit_payloads[str(key)]
        x = np.asarray(fit["centers"], dtype=float)
        counts = np.asarray(fit["counts"], dtype=float)
        smooth = np.asarray(fit["smooth_counts"], dtype=float)
        fit_counts = np.asarray(fit["fit_counts"], dtype=float)
        edges = np.asarray(fit["edges"], dtype=float)
        mu = float(fit["mu_bg_raw"])
        sigma = float(fit["sigma_bg_raw"])
        widths = np.diff(edges)
        ax.bar(
            x,
            counts,
            width=widths,
            align="center",
            color="0.85",
            edgecolor="0.70",
            linewidth=0.2,
            label="pooled histogram",
        )
        ax.plot(x, smooth, color="k", lw=1.5, label="smoothed histogram")
        ax.axvline(mu, color="tab:red", ls="--", lw=1.4, label="mu_bg")
        if bool(show_sigma_guides) and np.isfinite(sigma) and sigma > 0:
            ax.axvline(mu - sigma, color="tab:red", ls=":", lw=1.1, alpha=0.9, label="mu_bg ± sigma_bg")
            ax.axvline(mu + sigma, color="tab:red", ls=":", lw=1.1, alpha=0.9)
        ax.plot(x, fit_counts, color="tab:blue", lw=1.8, label="left-half Gaussian")
        ax.set_title(f"{title_prefix}{key}")
        ax.set_xlabel("Raw intensity")
        ax.set_ylabel("Pixel count")
        if xlim_by_channel and key in xlim_by_channel:
            ax.set_xlim(*xlim_by_channel[key])
        stats_text = f"mu_bg={mu:.1f}\nsigma_bg={sigma:.1f}"
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.25", "alpha": 0.9},
        )
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes[len(keys):]:
        ax.axis("off")
    return fig


def histogram_display_xlim_by_quantile(
    fit_payloads: Dict[str, dict],
    channel_order: Optional[Sequence[str]] = None,
    lower_quantile: float = 0.0,
    upper_quantile: float = 0.99,
) -> Dict[str, Tuple[float, float]]:
    keys = list(channel_order) if channel_order is not None else list(fit_payloads.keys())
    lo_q = float(np.clip(lower_quantile, 0.0, 1.0))
    hi_q = float(np.clip(upper_quantile, 0.0, 1.0))
    if hi_q <= lo_q:
        raise ValueError(f"upper_quantile must be > lower_quantile; got {lo_q} {hi_q}")

    out: Dict[str, Tuple[float, float]] = {}
    for key in keys:
        fit = fit_payloads[str(key)]
        edges = np.asarray(fit["edges"], dtype=float)
        counts = np.asarray(fit["counts"], dtype=float)
        if edges.ndim != 1 or counts.ndim != 1 or len(edges) != len(counts) + 1:
            raise ValueError(f"Unexpected histogram payload for {key}")
        total = float(np.sum(counts))
        if not np.isfinite(total) or total <= 0:
            out[str(key)] = (float(edges[0]), float(edges[-1]))
            continue

        cdf = np.cumsum(counts) / total
        lo_idx = int(np.searchsorted(cdf, lo_q, side="left"))
        hi_idx = int(np.searchsorted(cdf, hi_q, side="left"))
        lo_idx = int(np.clip(lo_idx, 0, len(counts) - 1))
        hi_idx = int(np.clip(max(lo_idx + 1, hi_idx), 1, len(edges) - 1))
        x0 = float(edges[lo_idx])
        x1 = float(edges[hi_idx])
        if x1 <= x0:
            x0 = float(edges[0])
            x1 = float(edges[-1])
        out[str(key)] = (x0, x1)
    return out


def pooled_histogram_transformed_channel_images(
    image_paths: Sequence[Union[str, Path]],
    channel_keywords: Sequence[str],
    transform_fn,
    canonical_positions: Optional[Sequence[Union[str, int]]] = None,
    n_bins: int = 2048,
    exclude_exact_zero_pixels: bool = True,
) -> dict:
    """Accumulate a pooled histogram after applying a channel-wise transform."""
    if len(image_paths) == 0:
        raise ValueError("image_paths is empty")

    global_min = np.inf
    global_max = -np.inf
    n_pixels = 0
    n_images = 0
    integer_like = True

    for idx, path_like in enumerate(image_paths):
        canonical_position = None if canonical_positions is None else canonical_positions[idx]
        img = read_czi_with_optional_fixed_small_plane_selection(
            Path(path_like),
            canonical_position=canonical_position,
        )
        ch_idx = common.find_channel_index(img.channels, list(channel_keywords))
        if ch_idx is None:
            raise RuntimeError(
                f"Missing channel matching {list(channel_keywords)} in {Path(path_like)}"
            )
        raw = img.channel_images[int(ch_idx)].astype(np.float32)
        transformed = np.asarray(transform_fn(raw), dtype=np.float32)
        valid = np.isfinite(raw) & np.isfinite(transformed)
        if bool(exclude_exact_zero_pixels):
            valid &= raw != 0
        finite = transformed[valid]
        if finite.size == 0:
            continue
        sample = finite if finite.size <= 4096 else finite[:: max(1, finite.size // 4096)]
        integer_like = bool(
            integer_like
            and np.all(np.abs(sample - np.rint(sample)) <= 1e-6)
        )
        global_min = min(global_min, float(np.min(finite)))
        global_max = max(global_max, float(np.max(finite)))
        n_pixels += int(finite.size)
        n_images += 1

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("Could not determine finite histogram range for transformed images")
    if global_max <= global_min:
        global_max = global_min + 1.0

    if bool(integer_like):
        lo_i = int(np.floor(global_min))
        hi_i = int(np.ceil(global_max))
        edges = np.arange(lo_i - 0.5, hi_i + 1.5, 1.0, dtype=np.float64)
    else:
        edges = np.linspace(global_min, global_max, int(n_bins) + 1, dtype=np.float64)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)

    for idx, path_like in enumerate(image_paths):
        canonical_position = None if canonical_positions is None else canonical_positions[idx]
        img = read_czi_with_optional_fixed_small_plane_selection(
            Path(path_like),
            canonical_position=canonical_position,
        )
        ch_idx = common.find_channel_index(img.channels, list(channel_keywords))
        if ch_idx is None:
            raise RuntimeError(
                f"Missing channel matching {list(channel_keywords)} in {Path(path_like)}"
            )
        raw = img.channel_images[int(ch_idx)].astype(np.float32)
        transformed = np.asarray(transform_fn(raw), dtype=np.float32)
        valid = np.isfinite(raw) & np.isfinite(transformed)
        if bool(exclude_exact_zero_pixels):
            valid &= raw != 0
        finite = transformed[valid]
        if finite.size == 0:
            continue
        h, _ = np.histogram(finite, bins=edges)
        counts += h.astype(np.int64)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "edges": edges,
        "centers": centers,
        "counts": counts,
        "global_min": float(global_min),
        "global_max": float(global_max),
        "n_pixels": int(n_pixels),
        "n_images": int(n_images),
        "n_bins": int(len(counts)),
        "integer_like_bins": bool(integer_like),
    }


def pooled_histogram_manifest_channel(
    position_manifest: Union[str, Path, pd.DataFrame],
    cohort_ids: Sequence[str],
    channel_keywords: Sequence[str],
    transform_fn=None,
    conditions: Optional[Sequence[str]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    max_positions: Optional[int] = None,
    n_bins: int = 2048,
    exclude_exact_zero_pixels: bool = True,
) -> dict:
    pos_df = _filter_positions(
        position_manifest=position_manifest,
        cohort_ids=list(cohort_ids),
        conditions=list(conditions) if conditions is not None else None,
        canonical_positions=list(canonical_positions) if canonical_positions is not None else None,
    )
    if max_positions is not None:
        keep_positions = _canonical_sort_values(pos_df["canonical_position"])[: int(max_positions)]
        pos_df = pos_df[pos_df["canonical_position"].isin(keep_positions)].copy()
    image_paths = [str(p) for p in pos_df["primary_analysis_file"].tolist()]
    position_keys = [str(p) for p in pos_df["canonical_position"].tolist()]
    if transform_fn is None:
        return pooled_histogram_transformed_channel_images(
            image_paths=image_paths,
            canonical_positions=position_keys,
            channel_keywords=list(channel_keywords),
            transform_fn=lambda arr: np.asarray(arr, dtype=np.float32),
            n_bins=int(n_bins),
            exclude_exact_zero_pixels=bool(exclude_exact_zero_pixels),
        )
    return pooled_histogram_transformed_channel_images(
        image_paths=image_paths,
        canonical_positions=position_keys,
        channel_keywords=list(channel_keywords),
        transform_fn=transform_fn,
        n_bins=int(n_bins),
        exclude_exact_zero_pixels=bool(exclude_exact_zero_pixels),
    )


def pooled_histogram_manifest_ratio(
    position_manifest: Union[str, Path, pd.DataFrame],
    cohort_ids: Sequence[str],
    numerator_keywords: Sequence[str],
    denominator_keywords: Sequence[str],
    conditions: Optional[Sequence[str]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    max_positions: Optional[int] = None,
    n_bins: int = 2048,
    exclude_exact_zero_pixels: bool = True,
) -> dict:
    pos_df = _filter_positions(
        position_manifest=position_manifest,
        cohort_ids=list(cohort_ids),
        conditions=list(conditions) if conditions is not None else None,
        canonical_positions=list(canonical_positions) if canonical_positions is not None else None,
    )
    if max_positions is not None:
        keep_positions = _canonical_sort_values(pos_df["canonical_position"])[: int(max_positions)]
        pos_df = pos_df[pos_df["canonical_position"].isin(keep_positions)].copy()
    image_paths = [str(p) for p in pos_df["primary_analysis_file"].tolist()]
    position_keys = [str(p) for p in pos_df["canonical_position"].tolist()]
    if len(image_paths) == 0:
        raise ValueError("No image paths available for ratio histogram")

    global_min = np.inf
    global_max = -np.inf
    n_pixels = 0
    n_images = 0

    for idx, path_like in enumerate(image_paths):
        img = read_czi_with_optional_fixed_small_plane_selection(
            Path(path_like),
            canonical_position=position_keys[idx],
        )
        num_idx = common.find_channel_index(img.channels, list(numerator_keywords))
        den_idx = common.find_channel_index(img.channels, list(denominator_keywords))
        if num_idx is None or den_idx is None:
            raise RuntimeError(
                f"Missing ratio channels numerator={list(numerator_keywords)} denominator={list(denominator_keywords)} "
                f"in {Path(path_like)}"
            )
        numerator = img.channel_images[int(num_idx)].astype(np.float32)
        denominator = img.channel_images[int(den_idx)].astype(np.float32)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
        if bool(exclude_exact_zero_pixels):
            valid &= (numerator != 0) & (denominator != 0)
        ratio = numerator[valid] / np.maximum(denominator[valid], float(EPS))
        if ratio.size == 0:
            continue
        global_min = min(global_min, float(np.min(ratio)))
        global_max = max(global_max, float(np.max(ratio)))
        n_pixels += int(ratio.size)
        n_images += 1

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("Could not determine finite histogram range for ratio images")
    if global_max <= global_min:
        global_max = global_min + 1.0

    edges = np.linspace(global_min, global_max, int(n_bins) + 1, dtype=np.float64)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)

    for idx, path_like in enumerate(image_paths):
        img = read_czi_with_optional_fixed_small_plane_selection(
            Path(path_like),
            canonical_position=position_keys[idx],
        )
        num_idx = common.find_channel_index(img.channels, list(numerator_keywords))
        den_idx = common.find_channel_index(img.channels, list(denominator_keywords))
        numerator = img.channel_images[int(num_idx)].astype(np.float32)
        denominator = img.channel_images[int(den_idx)].astype(np.float32)
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
        if bool(exclude_exact_zero_pixels):
            valid &= (numerator != 0) & (denominator != 0)
        ratio = numerator[valid] / np.maximum(denominator[valid], float(EPS))
        if ratio.size == 0:
            continue
        h, _ = np.histogram(ratio, bins=edges)
        counts += h.astype(np.int64)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "edges": edges,
        "centers": centers,
        "counts": counts,
        "global_min": float(global_min),
        "global_max": float(global_max),
        "n_pixels": int(n_pixels),
        "n_images": int(n_images),
        "n_bins": int(len(counts)),
        "integer_like_bins": False,
    }


def plot_raw_processed_distribution_panels(
    raw_payloads: Dict[str, dict],
    processed_payloads: Dict[str, dict],
    channel_order: Optional[Sequence[str]] = None,
    raw_xlim_by_channel: Optional[Dict[str, Tuple[float, float]]] = None,
    processed_xlim_by_channel: Optional[Dict[str, Tuple[float, float]]] = None,
    processed_thresholds_by_channel: Optional[Dict[str, Sequence[float]]] = None,
    processed_labels_by_channel: Optional[Dict[str, str]] = None,
    title_prefix: str = "",
    panel_width: float = 6.2,
    panel_height: float = 3.0,
) -> plt.Figure:
    keys = list(channel_order) if channel_order is not None else list(raw_payloads.keys())
    n = max(1, len(keys))
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(panel_width * 2, panel_height * n),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for row_idx, key in enumerate(keys):
        raw = raw_payloads[str(key)]
        proc = processed_payloads[str(key)]
        raw_ax, proc_ax = axes[row_idx, 0], axes[row_idx, 1]

        for ax, payload, xlim_map, subtitle in [
            (raw_ax, raw, raw_xlim_by_channel, "Raw"),
            (proc_ax, proc, processed_xlim_by_channel, processed_labels_by_channel.get(str(key), "Processed") if processed_labels_by_channel else "Processed"),
        ]:
            x = np.asarray(payload["centers"], dtype=float)
            counts = np.asarray(payload["counts"], dtype=float)
            edges = np.asarray(payload["edges"], dtype=float)
            widths = np.diff(edges)
            ax.bar(
                x,
                counts,
                width=widths,
                align="center",
                color="0.85",
                edgecolor="0.70",
                linewidth=0.2,
            )
            ax.set_title(f"{title_prefix}{key} {subtitle}")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Pixel count")
            if xlim_map and key in xlim_map:
                ax.set_xlim(*xlim_map[key])

        if processed_thresholds_by_channel and str(key) in processed_thresholds_by_channel:
            thr = np.asarray(processed_thresholds_by_channel[str(key)], dtype=float)
            thr = thr[np.isfinite(thr)]
            if thr.size:
                q25, q50, q75 = np.quantile(thr, [0.25, 0.50, 0.75])
                proc_ax.axvspan(float(q25), float(q75), color="tab:red", alpha=0.10, lw=0, label="threshold IQR")
                proc_ax.axvline(float(q50), color="tab:red", lw=1.3, ls="--", label="median threshold")
                proc_ax.legend(frameon=False, fontsize=8)

    return fig


def estimate_masked_illumination_fields_for_manifest_subset(
    position_manifest: Union[str, Path, pd.DataFrame],
    cohort_id: str,
    dapi_bg_params: dict,
    channel_specs: Dict[str, Sequence[str]],
    root: Optional[Union[str, Path]] = None,
    conditions: Optional[Sequence[str]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    max_positions: Optional[int] = None,
    bootstrap_mask_dilation_px: int = 20,
    bootstrap_blur_sigma_px: float = 12.0,
    bootstrap_threshold_scale: float = 0.90,
    field_gaussian_sigma_px: float = 40.0,
    min_background_pixels: int = 50000,
) -> Dict[str, object]:
    project_root = _resolve_project_root(root=root, position_manifest=position_manifest)
    pos_df = _filter_positions(
        position_manifest=position_manifest,
        cohort_ids=[cohort_id],
        conditions=conditions,
        canonical_positions=canonical_positions,
    )
    if max_positions is not None:
        keep_positions = _canonical_sort_values(pos_df["canonical_position"])[: int(max_positions)]
        pos_df = pos_df[pos_df["canonical_position"].isin(keep_positions)].copy()
    if pos_df.empty:
        raise ValueError(f"No rows found for cohort {cohort_id}")

    channel_accumulators: Dict[str, Dict[str, object]] = {}
    bootstrap_rows: List[Dict[str, object]] = []
    for channel_name in channel_specs:
        channel_accumulators[str(channel_name)] = {
            "sum_image": None,
            "count_image": None,
            "used_positions": 0,
            "used_background_pixels": 0,
        }

    for row in pos_df.itertuples(index=False):
        img = read_czi_with_optional_fixed_small_plane_selection(
            _resolve_project_path(row.primary_analysis_file, project_root),
            canonical_position=str(row.canonical_position),
        )
        dapi_idx = common.find_channel_index(img.channels, ["dapi"])
        if dapi_idx is None:
            raise RuntimeError(
                f"Missing DAPI channel in {row.primary_analysis_file}; channels={img.channels}"
            )
        dapi_raw = img.channel_images[int(dapi_idx)].astype(np.float32)
        provisional = segment_fixed_dapi_cyst_mask(
            normalize_channel_from_background(
                image=dapi_raw,
                mu_bg_raw=float(dapi_bg_params["mu_bg_raw"]),
                sigma_bg_raw=float(dapi_bg_params["sigma_bg_raw"]),
            ),
            canonical_position=str(row.canonical_position),
            blur_sigma_px=float(bootstrap_blur_sigma_px),
            core_threshold_scale=float(bootstrap_threshold_scale),
        )
        provisional_mask = np.asarray(provisional["mask"], dtype=bool)
        if int(bootstrap_mask_dilation_px) > 0:
            exclude_mask = morphology.binary_dilation(
                provisional_mask,
                morphology.disk(int(bootstrap_mask_dilation_px)),
            )
        else:
            exclude_mask = provisional_mask

        valid_dapi = np.isfinite(dapi_raw) & (dapi_raw > 0)
        background_mask = (~exclude_mask) & valid_dapi
        bootstrap_rows.append(
            {
                "canonical_position": str(row.canonical_position),
                "image_id": str(getattr(row, "image_id", f"{cohort_id}_{row.canonical_position}")),
                "primary_analysis_file": str(row.primary_analysis_file),
                "bootstrap_mask_area_px": int(np.sum(provisional_mask)),
                "bootstrap_mask_fraction": float(np.mean(provisional_mask)),
                "bootstrap_background_area_px": int(np.sum(background_mask)),
                "bootstrap_background_fraction": float(np.mean(background_mask)),
                "bootstrap_mask_dilation_px": int(bootstrap_mask_dilation_px),
                "bootstrap_blur_sigma_px": float(bootstrap_blur_sigma_px),
                "bootstrap_threshold_scale": float(bootstrap_threshold_scale),
            }
        )

        for channel_name, keywords in channel_specs.items():
            ch_idx = common.find_channel_index(img.channels, list(keywords))
            if ch_idx is None:
                raise RuntimeError(
                    f"Missing channel matching {list(keywords)} in {row.primary_analysis_file}; "
                    f"channels={img.channels}"
                )
            raw = img.channel_images[int(ch_idx)].astype(np.float32)
            valid = background_mask & np.isfinite(raw) & (raw > 0)
            n_bg = int(np.sum(valid))
            if n_bg < int(min_background_pixels):
                continue
            background_scale = float(np.median(raw[valid]))
            if not np.isfinite(background_scale) or background_scale <= 0:
                continue

            normalized = raw / background_scale
            acc = channel_accumulators[str(channel_name)]
            if acc["sum_image"] is None:
                acc["sum_image"] = np.zeros_like(normalized, dtype=np.float64)
                acc["count_image"] = np.zeros_like(normalized, dtype=np.float64)
            else:
                current_shape = np.asarray(acc["sum_image"]).shape
                target_shape = (
                    max(int(current_shape[0]), int(normalized.shape[0])),
                    max(int(current_shape[1]), int(normalized.shape[1])),
                )
                if current_shape != target_shape:
                    acc["sum_image"] = _pad_to_shape(
                        np.asarray(acc["sum_image"], dtype=np.float64),
                        target_shape,
                        fill_value=0.0,
                    )
                    acc["count_image"] = _pad_to_shape(
                        np.asarray(acc["count_image"], dtype=np.float64),
                        target_shape,
                        fill_value=0.0,
                    )
                if normalized.shape != target_shape:
                    normalized = _pad_to_shape(normalized, target_shape, fill_value=0.0)
                    valid = _pad_to_shape(valid.astype(np.uint8), target_shape, fill_value=0).astype(bool)
            acc["sum_image"] += normalized * valid
            acc["count_image"] += valid.astype(np.float64)
            acc["used_positions"] = int(acc["used_positions"]) + 1
            acc["used_background_pixels"] = int(acc["used_background_pixels"]) + n_bg

    field_payloads: Dict[str, dict] = {}
    summary_rows: List[Dict[str, object]] = []
    for channel_name, acc in channel_accumulators.items():
        if acc["sum_image"] is None or acc["count_image"] is None or int(acc["used_positions"]) == 0:
            raise RuntimeError(f"No valid bootstrap background frames were available for channel {channel_name}")
        count_image = np.asarray(acc["count_image"], dtype=np.float32)
        observed_mask = count_image > 0
        masked_mean = np.asarray(acc["sum_image"], dtype=np.float32) / np.maximum(count_image, float(EPS))
        field = smooth_masked_image(
            masked_mean=masked_mean,
            observed_mask=observed_mask,
            sigma_px=float(field_gaussian_sigma_px),
        )
        finite = np.isfinite(field) & observed_mask
        if not np.any(finite):
            raise RuntimeError(f"Estimated illumination field is not finite for channel {channel_name}")
        field = (field / np.median(field[finite])).astype(np.float32)
        field_payloads[str(channel_name)] = {
            "field": field,
            "sample_count": count_image,
            "observed_mask": observed_mask.astype(bool),
            "used_positions": int(acc["used_positions"]),
            "used_background_pixels": int(acc["used_background_pixels"]),
        }
        summary_rows.append(
            {
                "channel": str(channel_name),
                "used_positions": int(acc["used_positions"]),
                "used_background_pixels": int(acc["used_background_pixels"]),
                "field_min": float(np.nanmin(field)),
                "field_p01": float(np.nanpercentile(field, 1.0)),
                "field_median": float(np.nanmedian(field)),
                "field_p99": float(np.nanpercentile(field, 99.0)),
                "field_max": float(np.nanmax(field)),
                "field_center_value": float(field[field.shape[0] // 2, field.shape[1] // 2]),
                "sample_coverage_fraction": float(np.mean(observed_mask)),
                "field_gaussian_sigma_px": float(field_gaussian_sigma_px),
                "bootstrap_mask_dilation_px": int(bootstrap_mask_dilation_px),
            }
        )

    return {
        "field_payloads": field_payloads,
        "summary_df": pd.DataFrame(summary_rows),
        "bootstrap_df": pd.DataFrame(bootstrap_rows),
        "parameters": {
            "cohort_id": str(cohort_id),
            "field_gaussian_sigma_px": float(field_gaussian_sigma_px),
            "bootstrap_mask_dilation_px": int(bootstrap_mask_dilation_px),
            "bootstrap_blur_sigma_px": float(bootstrap_blur_sigma_px),
            "bootstrap_threshold_scale": float(bootstrap_threshold_scale),
            "min_background_pixels": int(min_background_pixels),
        },
    }


def plot_illumination_field_panels(
    field_payloads: Dict[str, dict],
    channel_order: Optional[Sequence[str]] = None,
    title_prefix: str = "",
    panel_width: float = 4.6,
    panel_height: float = 3.8,
) -> plt.Figure:
    keys = list(channel_order) if channel_order is not None else list(field_payloads.keys())
    n = max(1, len(keys))
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(panel_width * n, panel_height * 2),
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)
    for col_idx, key in enumerate(keys):
        payload = field_payloads[str(key)]
        field = np.asarray(payload["field"], dtype=np.float32)
        count = np.asarray(payload["sample_count"], dtype=np.float32)
        ax0 = axes[0, col_idx]
        ax1 = axes[1, col_idx]
        im0 = ax0.imshow(field, cmap="magma", vmin=np.nanpercentile(field, 1.0), vmax=np.nanpercentile(field, 99.0))
        ax0.set_title(f"{title_prefix}{key} field")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        cov = count / np.maximum(float(np.nanmax(count)), float(EPS))
        im1 = ax1.imshow(cov, cmap="viridis", vmin=0.0, vmax=1.0)
        ax1.set_title(f"{title_prefix}{key} sample coverage")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        for ax in (ax0, ax1):
            ax.set_xticks([])
            ax.set_yticks([])
    return fig


def build_fixed_small_mapped_centroid_table(
    large_centroid_path: Union[str, Path] = "results/annotations/well_centroids_large.tsv",
    live_fixed_alignment_path: Union[str, Path] = "results/annotations/live_fixed_alignment_transforms.tsv",
    fixed_transform_path: Union[str, Path] = "results/annotations/fixed_small_to_large_pair_transforms.tsv",
    out_path: Optional[Union[str, Path]] = "results/annotations/well_centroids_fixed_small_mapped.tsv",
    canonical_positions: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    cent_df = pd.read_csv(Path(large_centroid_path), sep="\t")
    align_df = pd.read_csv(Path(live_fixed_alignment_path), sep="\t")
    fixed_tr_df = pd.read_csv(Path(fixed_transform_path), sep="\t")

    if canonical_positions is not None:
        keep = set(map(str, canonical_positions))
        cent_df = cent_df[cent_df["canonical_position"].astype(str).isin(keep)].copy()
        align_df = align_df[align_df["canonical_position"].astype(str).isin(keep)].copy()
        fixed_tr_df = fixed_tr_df[fixed_tr_df["canonical_position"].astype(str).isin(keep)].copy()

    align_by_pos = {
        str(r.canonical_position): pd.Series(r._asdict()) for r in align_df.itertuples(index=False)
    }
    fixed_by_pos = {
        str(r.canonical_position): pd.Series(r._asdict()) for r in fixed_tr_df.itertuples(index=False)
    }

    rows: List[Dict[str, object]] = []
    for _, row in cent_df.iterrows():
        pos = str(row["canonical_position"])
        align_row = align_by_pos.get(pos)
        fixed_row = fixed_by_pos.get(pos)
        base = {
            "image_id": align_row["fixed_image_id"] if align_row is not None else np.nan,
            "cohort_id": "2026-01-22_day2_fix",
            "canonical_position": pos,
            "fixed_small_file_path": align_row["fixed_small_file_path"] if align_row is not None else np.nan,
            "fixed_large_file_path": align_row["fixed_large_file_path"] if align_row is not None else np.nan,
            "live_large_file_path": row["large_file_path"],
            "annotation_status": row["annotation_status"],
            "centroid_index": int(row["centroid_index"]),
            "centroid_live_large_x_px": float(row["centroid_x_px"]) if pd.notna(row["centroid_x_px"]) else np.nan,
            "centroid_live_large_y_px": float(row["centroid_y_px"]) if pd.notna(row["centroid_y_px"]) else np.nan,
            "updated_at": row.get("updated_at", ""),
        }

        if row["annotation_status"] != "annotated":
            rows.append(
                {
                    **base,
                    "mapping_status": "no_bead",
                    "centroid_fixed_small_x_px": np.nan,
                    "centroid_fixed_small_y_px": np.nan,
                    "inside_fixed_small_fov": False,
                    "fixed_small_shape_y_px": fixed_row.get("small_shape_y_px", np.nan) if fixed_row is not None else np.nan,
                    "fixed_small_shape_x_px": fixed_row.get("small_shape_x_px", np.nan) if fixed_row is not None else np.nan,
                }
            )
            continue

        if align_row is None or str(align_row.get("status", "")) != "ok":
            rows.append(
                {
                    **base,
                    "mapping_status": "missing_live_fixed_alignment",
                    "centroid_fixed_small_x_px": np.nan,
                    "centroid_fixed_small_y_px": np.nan,
                    "inside_fixed_small_fov": False,
                    "fixed_small_shape_y_px": fixed_row.get("small_shape_y_px", np.nan) if fixed_row is not None else np.nan,
                    "fixed_small_shape_x_px": fixed_row.get("small_shape_x_px", np.nan) if fixed_row is not None else np.nan,
                }
            )
            continue

        if fixed_row is None or str(fixed_row.get("registration_status", "")) != "ok":
            rows.append(
                {
                    **base,
                    "mapping_status": "missing_fixed_small_large_transform",
                    "centroid_fixed_small_x_px": np.nan,
                    "centroid_fixed_small_y_px": np.nan,
                    "inside_fixed_small_fov": False,
                    "fixed_small_shape_y_px": fixed_row.get("small_shape_y_px", np.nan) if fixed_row is not None else np.nan,
                    "fixed_small_shape_x_px": fixed_row.get("small_shape_x_px", np.nan) if fixed_row is not None else np.nan,
                }
            )
            continue

        fixed_small_to_live_large = lfa.affine_matrix_from_row_prefix(
            align_row,
            "fixed_small_to_live_large",
        )
        live_large_to_fixed_small = lfa.inverse_affine_matrix(fixed_small_to_live_large)
        xf, yf = lfa.apply_affine_to_xy(
            x_px=float(row["centroid_x_px"]),
            y_px=float(row["centroid_y_px"]),
            mat=live_large_to_fixed_small,
        )
        h = int(fixed_row["small_shape_y_px"])
        w = int(fixed_row["small_shape_x_px"])
        inside = bool((0.0 <= xf < float(w)) and (0.0 <= yf < float(h)))
        rows.append(
            {
                **base,
                "mapping_status": "ok",
                "centroid_fixed_small_x_px": float(xf),
                "centroid_fixed_small_y_px": float(yf),
                "inside_fixed_small_fov": inside,
                "fixed_small_shape_y_px": h,
                "fixed_small_shape_x_px": w,
            }
        )

    out_df = pd.DataFrame(rows, columns=FIXED_SMALL_CENTROID_COLUMNS)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, sep="\t", index=False)
        if verbose:
            print(f"Wrote {out_path}")
    return out_df


def segment_fixed_dapi_cyst_mask(
    dapi_norm: np.ndarray,
    canonical_position: Optional[Union[str, int]] = None,
    blur_sigma_px: float = DEFAULT_FIXED_DAPI_MASK_BLUR_SIGMA_PX,
    threshold_sigma: float = DEFAULT_FIXED_DAPI_THRESHOLD_SIGMA,
    min_area_px: int = 40,
    closing_radius_px: int = 1,
    dilation_radius_px: int = 0,
    hole_area_px: int = 0,
    core_blur_sigma_px: float = DEFAULT_FIXED_DAPI_CORE_BLUR_SIGMA_PX,
    core_threshold_scale: float = DEFAULT_FIXED_DAPI_CORE_THRESHOLD_SCALE,
    core_threshold_sigma: float = DEFAULT_FIXED_DAPI_CORE_THRESHOLD_SIGMA,
    core_min_area_px: int = 400,
    core_closing_radius_px: int = 2,
    core_fill_holes: bool = True,
    core_hole_area_px: int = 4000,
    monolayer_touch_radius_px: int = DEFAULT_FIXED_DAPI_MONOLAYER_TOUCH_RADIUS_PX,
) -> Dict[str, object]:
    img = np.asarray(dapi_norm, dtype=np.float32)
    artifact_exclusion_roi_mask, artifact_exclusion_reasons = _position_specific_exclusion_mask(
        img.shape,
        canonical_position,
    )
    core_blur = filters.gaussian(
        img,
        sigma=float(core_blur_sigma_px),
        preserve_range=True,
    ).astype(np.float32)
    core_hist = background_histogram_from_pixels(
        image=core_blur,
        n_bins=2048,
        exclude_exact_zero_pixels=True,
    )
    core_counts = np.asarray(core_hist["counts"], dtype=np.float32)
    core_hist["smooth_counts"] = ndi.gaussian_filter1d(core_counts, sigma=6.0, mode="nearest").astype(np.float32)
    core_valid = np.asarray(core_blur, dtype=np.float32)
    core_valid = core_valid[np.isfinite(core_valid) & (core_valid > 0)]
    if core_valid.size == 0:
        raise RuntimeError("No valid DAPI pixels were available for core thresholding.")
    core_otsu_threshold = float(filters.threshold_otsu(core_valid))
    core_thresh = float(core_otsu_threshold * float(core_threshold_scale))
    core_mask_raw = np.asarray(core_blur > core_thresh, dtype=bool)
    core_mask_small_removed = morphology.remove_small_objects(
        core_mask_raw.astype(bool),
        min_size=int(core_min_area_px),
    )
    core_mask = np.asarray(core_mask_small_removed, dtype=bool)
    if int(core_closing_radius_px) > 0:
        core_mask_closed = morphology.binary_closing(
            core_mask,
            morphology.disk(int(core_closing_radius_px)),
        )
    else:
        core_mask_closed = np.asarray(core_mask, dtype=bool)
    core_mask = np.asarray(core_mask_closed, dtype=bool)
    if bool(core_fill_holes):
        core_mask_filled = ndi.binary_fill_holes(core_mask)
    else:
        core_mask_filled = np.asarray(core_mask, dtype=bool)
    core_mask = np.asarray(core_mask_filled, dtype=bool)
    if int(core_hole_area_px) > 0:
        core_mask_holes_removed = morphology.remove_small_holes(
            core_mask.astype(bool),
            area_threshold=int(core_hole_area_px),
        )
    else:
        core_mask_holes_removed = np.asarray(core_mask, dtype=bool)
    core_mask = np.asarray(core_mask_holes_removed, dtype=bool)
    core_mask = morphology.remove_small_objects(
        core_mask.astype(bool),
        min_size=int(core_min_area_px),
    )
    artifact_exclusion_mask = np.zeros_like(core_mask, dtype=bool)
    if np.any(artifact_exclusion_roi_mask):
        core_labels = measure.label(core_mask.astype(bool))
        props = sorted(measure.regionprops(core_labels), key=lambda p: int(p.area), reverse=True)
        largest_label = int(props[0].label) if props else 0
        for prop in props:
            label_id = int(prop.label)
            if label_id == largest_label:
                continue
            component_mask = np.asarray(core_labels == label_id, dtype=bool)
            if np.any(component_mask & artifact_exclusion_roi_mask):
                artifact_exclusion_mask |= component_mask

    blur = filters.gaussian(img, sigma=float(blur_sigma_px), preserve_range=True).astype(np.float32)
    fit = fit_background_from_image_pixels(
        image=blur,
        n_bins=2048,
        smooth_sigma_bins=6.0,
        exclude_exact_zero_pixels=True,
    )
    mu_bg = float(fit["mu_bg_raw"])
    sigma_bg = float(max(EPS, float(fit["sigma_bg_raw"])))
    thresh = float(mu_bg + float(threshold_sigma) * sigma_bg)
    mask_raw = np.asarray(blur > thresh, dtype=bool)
    mono_mask_outside_core = np.asarray(mask_raw & (~core_mask), dtype=bool)
    mask_small_removed = morphology.remove_small_objects(
        mono_mask_outside_core.astype(bool),
        min_size=int(min_area_px),
    )
    mask = np.asarray(mask_small_removed, dtype=bool)
    if int(closing_radius_px) > 0:
        mask_closed = morphology.binary_closing(mask, morphology.disk(int(closing_radius_px)))
    else:
        mask_closed = np.asarray(mask, dtype=bool)
    mask = np.asarray(mask_closed, dtype=bool)
    if int(hole_area_px) > 0:
        mask_filled = ndi.binary_fill_holes(mask)
    else:
        mask_filled = np.asarray(mask, dtype=bool)
    mask = np.asarray(mask_filled, dtype=bool)
    if int(hole_area_px) > 0:
        mask_holes_removed = morphology.remove_small_holes(mask.astype(bool), area_threshold=int(hole_area_px))
    else:
        mask_holes_removed = np.asarray(mask, dtype=bool)
    mask = np.asarray(mask_holes_removed, dtype=bool)
    if int(monolayer_touch_radius_px) > 0:
        mono_anchor = morphology.binary_dilation(
            core_mask.astype(bool),
            morphology.disk(int(monolayer_touch_radius_px)),
        )
    else:
        mono_anchor = np.asarray(core_mask, dtype=bool)
    mono_mask_connected = _connected_components_touching(mask, mono_anchor)
    if int(dilation_radius_px) > 0:
        mask_dilated = morphology.binary_dilation(
            mono_mask_connected.astype(bool),
            morphology.disk(int(dilation_radius_px)),
        )
    else:
        mask_dilated = np.asarray(mono_mask_connected, dtype=bool)
    mono_mask = morphology.remove_small_objects(mask_dilated.astype(bool), min_size=int(min_area_px))
    pre_manual_correction_mask = np.asarray(core_mask | mono_mask, dtype=bool)
    final_mask = np.asarray(
        pre_manual_correction_mask & (~artifact_exclusion_mask),
        dtype=bool,
    )
    comp_labels = measure.label(final_mask)
    return {
        "mask": final_mask.astype(bool),
        "mask_pre_manual_correction": pre_manual_correction_mask.astype(bool),
        "mask_raw": mask_raw.astype(bool),
        "core_mask_raw": core_mask_raw.astype(bool),
        "core_mask_small_removed": np.asarray(core_mask_small_removed, dtype=bool),
        "core_mask_closed": np.asarray(core_mask_closed, dtype=bool),
        "core_mask_filled": np.asarray(core_mask_filled, dtype=bool),
        "core_mask_holes_removed": np.asarray(core_mask_holes_removed, dtype=bool),
        "core_mask": core_mask.astype(bool),
        "mono_mask_outside_core": mono_mask_outside_core.astype(bool),
        "mask_small_removed": np.asarray(mask_small_removed, dtype=bool),
        "mask_closed": np.asarray(mask_closed, dtype=bool),
        "mask_filled": np.asarray(mask_filled, dtype=bool),
        "mask_holes_removed": np.asarray(mask_holes_removed, dtype=bool),
        "mask_dilated": np.asarray(mask_dilated, dtype=bool),
        "mono_anchor": np.asarray(mono_anchor, dtype=bool),
        "mono_mask_connected": np.asarray(mono_mask_connected, dtype=bool),
        "mono_mask": np.asarray(mono_mask, dtype=bool),
        "final_mask": final_mask.astype(bool),
        "artifact_exclusion_roi_mask": np.asarray(artifact_exclusion_roi_mask, dtype=bool),
        "artifact_exclusion_mask": np.asarray(artifact_exclusion_mask, dtype=bool),
        "artifact_exclusion_applied": bool(np.any(artifact_exclusion_mask)),
        "artifact_exclusion_area_px": int(np.sum(artifact_exclusion_mask)),
        "artifact_exclusion_reasons": list(artifact_exclusion_reasons),
        "core_blur": core_blur.astype(np.float32),
        "blur": blur.astype(np.float32),
        "threshold": float(thresh),
        "mu_bg": float(mu_bg),
        "sigma_bg": float(sigma_bg),
        "threshold_sigma": float(threshold_sigma),
        "core_threshold": float(core_thresh),
        "core_otsu_threshold": float(core_otsu_threshold),
        "core_threshold_scale": float(core_threshold_scale),
        "core_mu_bg": float("nan"),
        "core_sigma_bg": float("nan"),
        "core_threshold_sigma": float(core_threshold_sigma),
        "core_blur_sigma_px": float(core_blur_sigma_px),
        "blur_sigma_px": float(blur_sigma_px),
        "n_components": int(comp_labels.max()),
        "background_fit": fit,
        "core_background_fit": core_hist,
    }


def masked_gaussian_blur_excluding_mask(
    image: np.ndarray,
    exclude_mask: np.ndarray,
    sigma_px: float,
    mode: str = "reflect",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.asarray(image, dtype=np.float32)
    excluded = np.asarray(exclude_mask, dtype=bool)
    observed = np.isfinite(img) & (img > 0) & (~excluded)
    masked = np.where(observed, img, 0.0).astype(np.float32)
    blurred = smooth_masked_image(
        masked_mean=masked,
        observed_mask=observed.astype(np.float32),
        sigma_px=float(sigma_px),
        mode=str(mode),
    )
    support = ndi.gaussian_filter(
        observed.astype(np.float32),
        sigma=float(sigma_px),
        mode=str(mode),
    ).astype(np.float32)
    blurred = np.where(observed, blurred, np.nan).astype(np.float32)
    support = np.where(observed, support, np.nan).astype(np.float32)
    return blurred, support, observed


def cleanup_binary_mask(
    mask: np.ndarray,
    min_area_px: int,
    closing_radius_px: int,
) -> np.ndarray:
    out = morphology.remove_small_objects(np.asarray(mask, dtype=bool), min_size=int(min_area_px))
    if int(closing_radius_px) > 0:
        out = morphology.binary_closing(out, morphology.disk(int(closing_radius_px)))
    out = morphology.remove_small_objects(np.asarray(out, dtype=bool), min_size=int(min_area_px))
    return np.asarray(out, dtype=bool)


def build_manual_roi_mask(
    shape_yx: Tuple[int, int],
    specs: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, List[str]]:
    roi = np.zeros(shape_yx, dtype=bool)
    reasons: List[str] = []
    h, w = map(int, shape_yx)
    for spec in specs:
        r0 = max(0, int(spec["row_min"]))
        r1 = min(h, int(spec["row_max"]))
        c0 = max(0, int(spec["col_min"]))
        c1 = min(w, int(spec["col_max"]))
        if r0 < r1 and c0 < c1:
            roi[r0:r1, c0:c1] = True
            reasons.append(str(spec.get("reason", "manual_component_exclusion")))
    return roi.astype(bool), reasons


def build_artifact_shaped_exclusion(
    mask: np.ndarray,
    artifact_mask: np.ndarray,
    dilation_radius_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    artifact = np.asarray(artifact_mask, dtype=bool)
    if int(dilation_radius_px) > 0:
        roi_mask = morphology.binary_dilation(artifact, morphology.disk(int(dilation_radius_px)))
    else:
        roi_mask = artifact.copy()
    exclusion = np.asarray(mask, dtype=bool) & np.asarray(roi_mask, dtype=bool)
    return exclusion.astype(bool), np.asarray(roi_mask, dtype=bool)


def manual_component_exclusion_from_rois(
    mask: np.ndarray,
    specs: Sequence[Dict[str, object]],
    artifact_mask: Optional[np.ndarray] = None,
    artifact_spec: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    if artifact_spec is not None and artifact_mask is not None:
        exclusion, roi_mask = build_artifact_shaped_exclusion(
            mask,
            artifact_mask,
            dilation_radius_px=int(artifact_spec.get("dilation_radius_px", 0)),
        )
        if np.any(exclusion):
            reasons = [str(artifact_spec.get("reason", "manual_artifact_exclusion"))]
            return exclusion.astype(bool), reasons, roi_mask.astype(bool)
    if not specs:
        empty = np.zeros_like(mask, dtype=bool)
        return empty, [], empty
    roi_mask, reasons = build_manual_roi_mask(mask.shape, specs)
    modes = {str(spec.get("mode", "component_overlap")) for spec in specs}
    if "mask_intersection" in modes:
        exclusion = np.asarray(mask, dtype=bool) & roi_mask
        return exclusion.astype(bool), reasons, roi_mask.astype(bool)

    labels = measure.label(np.asarray(mask, dtype=bool))
    exclusion = np.zeros_like(mask, dtype=bool)
    for prop in measure.regionprops(labels):
        component = labels == int(prop.label)
        if np.any(component & roi_mask):
            exclusion |= component
    return exclusion.astype(bool), reasons, roi_mask.astype(bool)


def build_final04_fixed_dapi_masks(
    position_manifest: Union[str, Path, pd.DataFrame],
    cohort_id: str,
    root: Optional[Union[str, Path]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    round2_threshold_sigma: float = APPROVED_FINAL04_FIXED_DAPI_ROUND2_THRESHOLD_SIGMA,
) -> Dict[str, object]:
    """Rebuild the accepted final fixed DAPI masks from notebook 04."""
    project_root = _resolve_project_root(root=root, position_manifest=position_manifest)
    pos_df = _filter_positions(
        position_manifest=position_manifest,
        cohort_ids=[cohort_id],
        conditions=["fixed"],
        canonical_positions=list(canonical_positions) if canonical_positions is not None else None,
    )
    if pos_df.empty:
        raise ValueError(f"No fixed positions found for cohort {cohort_id}")
    pos_df = pos_df.sort_values("canonical_position").reset_index(drop=True)

    cache: Dict[str, Dict[str, object]] = {}
    pooled_values: List[np.ndarray] = []
    summary_rows: List[Dict[str, object]] = []

    for row in pos_df.itertuples(index=False):
        posid = str(row.canonical_position)
        bundle = load_czi_bundle_with_z(_resolve_project_path(row.primary_analysis_file, project_root))
        channels = bundle["channels"]
        bf_idx = common.find_channel_index(channels, ["bright"])
        dapi_idx = common.find_channel_index(channels, ["dapi"])
        if bf_idx is None or dapi_idx is None:
            raise RuntimeError(f"Missing BF or DAPI in {row.primary_analysis_file}; channels={channels}")

        bf_stack = np.asarray(bundle["channel_stack_czyx"][int(bf_idx)], dtype=np.float32)
        dapi_stack = np.asarray(bundle["channel_stack_czyx"][int(dapi_idx)], dtype=np.float32)
        bf_max = np.max(bf_stack, axis=0).astype(np.float32)
        dapi_max = np.max(dapi_stack, axis=0).astype(np.float32)

        seg = segment_fixed_dapi_cyst_mask(
            dapi_max,
            canonical_position=posid,
            blur_sigma_px=float(DEFAULT_FIXED_DAPI_MASK_BLUR_SIGMA_PX),
            threshold_sigma=float(DEFAULT_FIXED_DAPI_THRESHOLD_SIGMA),
            min_area_px=int(APPROVED_FINAL04_FIXED_DAPI_ROUND2_MIN_AREA_PX),
            closing_radius_px=int(APPROVED_FINAL04_FIXED_DAPI_ROUND2_CLOSING_RADIUS_PX),
            dilation_radius_px=0,
            hole_area_px=0,
            core_blur_sigma_px=float(DEFAULT_FIXED_DAPI_CORE_BLUR_SIGMA_PX),
            core_threshold_scale=float(APPROVED_FINAL04_FIXED_DAPI_CORE_THRESHOLD_SCALE),
            core_min_area_px=400,
            core_closing_radius_px=2,
            core_fill_holes=bool(APPROVED_FINAL04_FIXED_DAPI_CORE_FILL_HOLES),
            core_hole_area_px=int(APPROVED_FINAL04_FIXED_DAPI_CORE_HOLE_AREA_PX),
            monolayer_touch_radius_px=int(DEFAULT_FIXED_DAPI_MONOLAYER_TOUCH_RADIUS_PX),
        )

        selected_z, selected_plane = selected_fixed_small_dapi_plane(
            dapi_stack=dapi_stack,
            canonical_position=posid,
        )
        core_exclude_mask = np.asarray(seg["core_mask"], dtype=bool)
        round1_core_mask = np.asarray(
            seg["core_mask"] & (~np.asarray(seg["artifact_exclusion_mask"], dtype=bool)),
            dtype=bool,
        )
        round2_blur, round2_support, observed = masked_gaussian_blur_excluding_mask(
            image=selected_plane,
            exclude_mask=core_exclude_mask,
            sigma_px=float(APPROVED_FINAL04_FIXED_DAPI_ROUND2_BLUR_SIGMA_PX),
            mode=str(APPROVED_FINAL04_FIXED_DAPI_ROUND2_BLUR_MODE),
        )
        observed_vals = round2_blur[np.isfinite(round2_blur) & observed]
        if observed_vals.size:
            pooled_values.append(np.asarray(observed_vals, dtype=np.float32))

        cache[posid] = {
            "canonical_position": posid,
            "row": pd.Series(row._asdict()),
            "bf_max": bf_max,
            "dapi_max": dapi_max,
            "selected_z": int(selected_z),
            "selected_plane": np.asarray(selected_plane, dtype=np.float32),
            "core_exclude_mask": core_exclude_mask.astype(bool),
            "round1_core_mask": round1_core_mask.astype(bool),
            "round2_blur": np.asarray(round2_blur, dtype=np.float32),
            "round2_support": np.asarray(round2_support, dtype=np.float32),
            "round2_observed": np.asarray(observed, dtype=bool),
            "artifact_exclusion_mask": np.asarray(seg["artifact_exclusion_mask"], dtype=bool),
            "artifact_exclusion_applied": bool(seg.get("artifact_exclusion_applied", False)),
            "artifact_exclusion_area_px": int(seg.get("artifact_exclusion_area_px", 0)),
            "artifact_exclusion_reasons": list(seg.get("artifact_exclusion_reasons", [])),
            "core_segmentation": seg,
        }
        summary_rows.append(
            {
                "canonical_position": posid,
                "image_id": str(getattr(row, "image_id", f"{cohort_id}_{posid}")),
                "selected_z": int(selected_z),
                "z_count": int(dapi_stack.shape[0]),
                "round1_core_area_px": int(np.sum(round1_core_mask)),
                "round1_core_fraction": float(np.mean(round1_core_mask)),
                "round2_residual_area_px": int(np.sum(observed)),
                "round2_residual_fraction": float(np.mean(observed)),
                "artifact_exclusion_applied": bool(seg.get("artifact_exclusion_applied", False)),
                "artifact_exclusion_area_px": int(seg.get("artifact_exclusion_area_px", 0)),
                "artifact_exclusion_reasons": ";".join(seg.get("artifact_exclusion_reasons", [])),
            }
        )

    pooled = np.concatenate(pooled_values).astype(np.float32) if pooled_values else np.array([], dtype=np.float32)
    if pooled.size == 0:
        raise RuntimeError("No residual pixels were available to fit the approved round-2 DAPI threshold.")
    round2_fit = fit_background_from_image_pixels(
        image=pooled,
        n_bins=2048,
        smooth_sigma_bins=6.0,
        exclude_exact_zero_pixels=True,
    )
    round2_threshold = float(
        float(round2_fit["mu_bg_raw"]) + float(round2_threshold_sigma) * float(round2_fit["sigma_bg_raw"])
    )

    final_rows: List[Dict[str, object]] = []
    for posid, payload in cache.items():
        raw_added = np.asarray(
            (payload["round2_blur"] > round2_threshold) & payload["round2_observed"],
            dtype=bool,
        )
        cleaned = cleanup_binary_mask(
            raw_added,
            min_area_px=int(APPROVED_FINAL04_FIXED_DAPI_ROUND2_MIN_AREA_PX),
            closing_radius_px=int(APPROVED_FINAL04_FIXED_DAPI_ROUND2_CLOSING_RADIUS_PX),
        ) & np.asarray(payload["round2_observed"], dtype=bool)
        manual_exclusion_mask, manual_reasons, manual_roi_mask = manual_component_exclusion_from_rois(
            cleaned,
            APPROVED_FINAL04_FIXED_DAPI_ROUND2_MANUAL_COMPONENT_ROIS.get(posid, []),
            artifact_mask=np.asarray(payload["artifact_exclusion_mask"], dtype=bool),
            artifact_spec=APPROVED_FINAL04_FIXED_DAPI_ROUND2_MANUAL_ARTIFACT_EXCLUSION.get(posid),
        )
        if posid in APPROVED_FINAL04_FIXED_DAPI_ROUND2_SUPPRESS_POSITIONS:
            round2_added_mask = np.zeros_like(cleaned, dtype=bool)
            round2_status = "suppressed"
        else:
            round2_added_mask = np.asarray(cleaned & (~manual_exclusion_mask), dtype=bool)
            round2_status = "thresholded_with_manual_component_exclusion" if np.any(manual_exclusion_mask) else "thresholded"
        final_mask = np.asarray(payload["round1_core_mask"] | round2_added_mask, dtype=bool)
        payload["round2_added_mask"] = round2_added_mask.astype(bool)
        payload["round2_manual_exclusion_mask"] = manual_exclusion_mask.astype(bool)
        payload["round2_manual_roi_mask"] = manual_roi_mask.astype(bool)
        payload["final_mask"] = final_mask.astype(bool)
        payload["round2_status"] = str(round2_status)

        final_rows.append(
            {
                "canonical_position": posid,
                "round2_threshold": float(round2_threshold),
                "round2_threshold_sigma": float(round2_threshold_sigma),
                "round2_mu_bg": float(round2_fit["mu_bg_raw"]),
                "round2_sigma_bg": float(round2_fit["sigma_bg_raw"]),
                "round2_added_area_px": int(np.sum(round2_added_mask)),
                "round2_added_fraction": float(np.mean(round2_added_mask)),
                "round2_status": str(round2_status),
                "round2_manual_exclusion_area_px": int(np.sum(manual_exclusion_mask)),
                "round2_manual_exclusion_reasons": ";".join(manual_reasons),
                "final_mask_area_px": int(np.sum(final_mask)),
                "final_mask_fraction": float(np.mean(final_mask)),
            }
        )

    summary_df = (
        pd.DataFrame(summary_rows)
        .merge(pd.DataFrame(final_rows), on="canonical_position", how="left")
        .sort_values("canonical_position")
        .reset_index(drop=True)
    )
    return {
        "summary_df": summary_df,
        "mask_payloads": cache,
        "round2_fit": round2_fit,
        "round2_threshold": float(round2_threshold),
        "round2_threshold_sigma": float(round2_threshold_sigma),
    }


def estimate_live_masked_illumination_field(
    position_manifest: Union[str, Path, pd.DataFrame],
    cohort_id: str,
    live_masks_by_position: Dict[str, np.ndarray],
    channel_keywords: Sequence[str],
    root: Optional[Union[str, Path]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    field_gaussian_sigma_px: float = 40.0,
    min_background_pixels: int = 50000,
    smoothing_mode: str = "reflect",
) -> Dict[str, object]:
    """Estimate a single smooth illumination field from off-mask live pixels."""
    project_root = _resolve_project_root(root=root, position_manifest=position_manifest)
    pos_df = _filter_positions(
        position_manifest=position_manifest,
        cohort_ids=[cohort_id],
        conditions=["live"],
        canonical_positions=list(canonical_positions) if canonical_positions is not None else None,
    )
    if pos_df.empty:
        raise ValueError(f"No live positions found for cohort {cohort_id}")
    pos_df = pos_df.sort_values("canonical_position").reset_index(drop=True)

    sum_image: Optional[np.ndarray] = None
    count_image: Optional[np.ndarray] = None
    position_rows: List[Dict[str, object]] = []

    for row in pos_df.itertuples(index=False):
        posid = str(row.canonical_position)
        if posid not in live_masks_by_position:
            continue
        img = common.read_czi(_resolve_project_path(row.primary_analysis_file, project_root))
        ch_idx = common.find_channel_index(img.channels, list(channel_keywords))
        if ch_idx is None:
            raise RuntimeError(
                f"Missing channel matching {list(channel_keywords)} in {row.primary_analysis_file}; "
                f"channels={img.channels}"
            )
        raw = img.channel_images[int(ch_idx)].astype(np.float32)
        live_mask = np.asarray(live_masks_by_position[posid], dtype=bool)
        if live_mask.shape != raw.shape:
            raise ValueError(
                f"Mask/image shape mismatch for {posid}: mask={live_mask.shape} raw={raw.shape}"
            )
        background_mask = (~live_mask) & np.isfinite(raw) & (raw > 0)
        n_bg = int(np.sum(background_mask))
        background_fraction = float(np.mean(background_mask))
        background_scale = float(np.median(raw[background_mask])) if n_bg > 0 else float("nan")
        use_image = (
            n_bg >= int(min_background_pixels)
            and np.isfinite(background_scale)
            and float(background_scale) > 0
        )
        position_rows.append(
            {
                "canonical_position": posid,
                "image_id": str(getattr(row, "image_id", f"{cohort_id}_{posid}")),
                "background_pixels": n_bg,
                "background_fraction": background_fraction,
                "background_median_raw": float(background_scale) if np.isfinite(background_scale) else np.nan,
                "used_for_field": bool(use_image),
            }
        )
        if not use_image:
            continue

        normalized = raw / float(background_scale)
        valid = np.asarray(background_mask, dtype=bool)
        if sum_image is None:
            sum_image = np.zeros_like(normalized, dtype=np.float64)
            count_image = np.zeros_like(normalized, dtype=np.float64)
        else:
            current_shape = np.asarray(sum_image).shape
            target_shape = (
                max(int(current_shape[0]), int(normalized.shape[0])),
                max(int(current_shape[1]), int(normalized.shape[1])),
            )
            if current_shape != target_shape:
                sum_image = _pad_to_shape(np.asarray(sum_image, dtype=np.float64), target_shape, fill_value=0.0)
                count_image = _pad_to_shape(np.asarray(count_image, dtype=np.float64), target_shape, fill_value=0.0)
            if normalized.shape != target_shape:
                normalized = _pad_to_shape(normalized, target_shape, fill_value=0.0)
                valid = _pad_to_shape(valid.astype(np.uint8), target_shape, fill_value=0).astype(bool)

        sum_image += normalized * valid
        count_image += valid.astype(np.float64)

    if sum_image is None or count_image is None:
        raise RuntimeError("No live images had enough off-mask background pixels for illumination estimation.")

    sample_count = np.asarray(count_image, dtype=np.float32)
    observed_mask = sample_count > 0
    masked_mean = np.asarray(sum_image, dtype=np.float32) / np.maximum(sample_count, float(EPS))
    field = smooth_masked_image(
        masked_mean=masked_mean,
        observed_mask=observed_mask.astype(np.float32),
        sigma_px=float(field_gaussian_sigma_px),
        mode=str(smoothing_mode),
    )
    field = np.asarray(field, dtype=np.float32)
    if not np.any(np.isfinite(field) & observed_mask):
        raise RuntimeError("Estimated live illumination field has no finite supported pixels.")
    field = field / np.median(field[np.isfinite(field) & observed_mask])

    summary_df = pd.DataFrame(
        [
            {
                "channel": "live_tagyfp",
                "used_positions": int(np.sum(pd.DataFrame(position_rows)["used_for_field"].astype(bool))),
                "used_background_pixels": int(np.nansum(pd.DataFrame(position_rows).loc[pd.DataFrame(position_rows)["used_for_field"].astype(bool), "background_pixels"].astype(float))),
                "field_min": float(np.nanmin(field)),
                "field_p01": float(np.nanpercentile(field, 1.0)),
                "field_median": float(np.nanmedian(field)),
                "field_p99": float(np.nanpercentile(field, 99.0)),
                "field_max": float(np.nanmax(field)),
                "field_center_value": float(field[field.shape[0] // 2, field.shape[1] // 2]),
                "sample_coverage_fraction": float(np.mean(observed_mask)),
                "field_gaussian_sigma_px": float(field_gaussian_sigma_px),
                "min_background_pixels": int(min_background_pixels),
                "smoothing_mode": str(smoothing_mode),
            }
        ]
    )

    return {
        "field": field.astype(np.float32),
        "sample_count": sample_count.astype(np.float32),
        "observed_mask": observed_mask.astype(bool),
        "masked_mean": masked_mean.astype(np.float32),
        "position_df": pd.DataFrame(position_rows).sort_values("canonical_position").reset_index(drop=True),
        "summary_df": summary_df,
        "parameters": {
            "field_gaussian_sigma_px": float(field_gaussian_sigma_px),
            "min_background_pixels": int(min_background_pixels),
            "smoothing_mode": str(smoothing_mode),
        },
    }


def transfer_fixed_masks_to_live_space(
    position_manifest: Union[str, Path, pd.DataFrame],
    live_fixed_alignment_path: Union[str, Path, pd.DataFrame],
    fixed_mask_payloads: Dict[str, Dict[str, object]],
    root: Optional[Union[str, Path]] = None,
    live_cohort_id: str = "2026-01-22_day2_live",
    fixed_cohort_id: str = "2026-01-22_day2_fix",
    write_fixed_mask_dir: Optional[Union[str, Path]] = None,
    write_live_mask_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Transfer fixed-space masks into live small-image coordinates."""
    project_root = _resolve_project_root(root=root, position_manifest=position_manifest)
    pos_df = common.load_position_manifest(position_manifest)
    align_df = pd.read_csv(live_fixed_alignment_path, sep="\t") if not isinstance(live_fixed_alignment_path, pd.DataFrame) else live_fixed_alignment_path.copy()
    align_df["canonical_position"] = align_df["canonical_position"].astype(str)
    live_df = common.filter_position_manifest(
        pos_df=pos_df,
        cohort_ids=[live_cohort_id],
        conditions=["live"],
    ).copy()
    fixed_df = common.filter_position_manifest(
        pos_df=pos_df,
        cohort_ids=[fixed_cohort_id],
        conditions=["fixed"],
    ).copy()
    live_df["canonical_position"] = live_df["canonical_position"].astype(str)
    fixed_df["canonical_position"] = fixed_df["canonical_position"].astype(str)
    live_by_pos = {str(r.canonical_position): pd.Series(r._asdict()) for r in live_df.itertuples(index=False)}
    fixed_by_pos = {str(r.canonical_position): pd.Series(r._asdict()) for r in fixed_df.itertuples(index=False)}
    align_by_pos = {str(r.canonical_position): pd.Series(r._asdict()) for r in align_df.itertuples(index=False)}

    fixed_masks_by_position: Dict[str, np.ndarray] = {}
    live_masks_by_position: Dict[str, np.ndarray] = {}
    summary_rows: List[Dict[str, object]] = []

    fixed_mask_dir = Path(write_fixed_mask_dir) if write_fixed_mask_dir is not None else None
    live_mask_dir = Path(write_live_mask_dir) if write_live_mask_dir is not None else None
    for out_dir in [fixed_mask_dir, live_mask_dir]:
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

    all_positions = _canonical_sort_values(
        set(live_by_pos.keys()) | set(fixed_by_pos.keys()) | set(fixed_mask_payloads.keys())
    )
    for pos in all_positions:
        live_row = live_by_pos.get(pos)
        fixed_row = fixed_by_pos.get(pos)
        align_row = align_by_pos.get(pos)
        fixed_payload = fixed_mask_payloads.get(pos)
        if live_row is None or fixed_row is None or fixed_payload is None:
            summary_rows.append(
                {
                    "canonical_position": pos,
                    "status": "missing_manifest_or_fixed_mask",
                    "live_image_id": str(live_row.get("image_id", f"{live_cohort_id}_{pos}")) if live_row is not None else np.nan,
                    "fixed_image_id": str(fixed_row.get("image_id", f"{fixed_cohort_id}_{pos}")) if fixed_row is not None else np.nan,
                    "fixed_mask_area_px": np.nan,
                    "fixed_mask_fraction": np.nan,
                    "live_mask_area_px": np.nan,
                    "live_mask_fraction": np.nan,
                    "selected_z": np.nan,
                    "updated_at": _iso_now(),
                }
            )
            continue
        if align_row is None or str(align_row.get("status", "")) != "ok":
            summary_rows.append(
                {
                    "canonical_position": pos,
                    "status": "missing_live_fixed_alignment",
                    "live_image_id": str(live_row.get("image_id", f"{live_cohort_id}_{pos}")),
                    "fixed_image_id": str(fixed_row.get("image_id", f"{fixed_cohort_id}_{pos}")),
                    "fixed_mask_area_px": np.nan,
                    "fixed_mask_fraction": np.nan,
                    "live_mask_area_px": np.nan,
                    "live_mask_fraction": np.nan,
                    "selected_z": int(fixed_payload.get("selected_z", 0)),
                    "updated_at": _iso_now(),
                }
            )
            continue

        live_img = common.read_czi(_resolve_project_path(live_row["primary_analysis_file"], project_root))
        live_bf_idx = common.find_channel_index(live_img.channels, ["bright"])
        live_yfp_idx = common.find_channel_index(live_img.channels, ["tagyfp", "foxf1", "yfp"])
        if live_bf_idx is None or live_yfp_idx is None:
            raise RuntimeError(
                f"Missing live BF/TagYFP channels in {live_row['primary_analysis_file']}; channels={live_img.channels}"
            )
        live_yfp_raw = live_img.channel_images[int(live_yfp_idx)].astype(np.float32)
        fixed_mask = np.asarray(fixed_payload["final_mask"], dtype=bool)
        fixed_small_to_live_small = lfa.affine_matrix_from_row_prefix(
            align_row,
            "fixed_small_to_live_small",
        )
        live_mask = lfa.warp_image_with_affine(
            image=fixed_mask.astype(np.float32),
            forward_mat=fixed_small_to_live_small,
            output_shape_yx=live_yfp_raw.shape,
            order=0,
            cval=0.0,
        ) > 0.5
        live_valid = np.isfinite(live_yfp_raw) & (live_yfp_raw > 0)
        live_mask &= live_valid

        fixed_masks_by_position[pos] = fixed_mask.astype(bool)
        live_masks_by_position[pos] = live_mask.astype(bool)
        if fixed_mask_dir is not None:
            np.savez_compressed(fixed_mask_dir / f"{pos}_fixed_mask_final04.npz", mask=fixed_mask.astype(np.uint8))
        if live_mask_dir is not None:
            np.savez_compressed(live_mask_dir / f"{pos}_live_mask_from_fixed_final04.npz", mask=live_mask.astype(np.uint8))

        summary_rows.append(
            {
                "canonical_position": pos,
                "status": "ok",
                "live_image_id": str(live_row.get("image_id", f"{live_cohort_id}_{pos}")),
                "fixed_image_id": str(fixed_row.get("image_id", f"{fixed_cohort_id}_{pos}")),
                "fixed_mask_area_px": int(np.sum(fixed_mask)),
                "fixed_mask_fraction": float(np.mean(fixed_mask)),
                "live_mask_area_px": int(np.sum(live_mask)),
                "live_mask_fraction": float(np.mean(live_mask)),
                "selected_z": int(fixed_payload.get("selected_z", 0)),
                "updated_at": _iso_now(),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("canonical_position").reset_index(drop=True)
    return {
        "summary_df": summary_df,
        "fixed_masks_by_position": fixed_masks_by_position,
        "live_masks_by_position": live_masks_by_position,
    }


def masked_distance_bin_stats(
    value_img: np.ndarray,
    distance_um_map: np.ndarray,
    mask: np.ndarray,
    bin_um: float,
) -> pd.DataFrame:
    values = np.asarray(value_img, dtype=np.float32)
    dist = np.asarray(distance_um_map, dtype=np.float32)
    keep = np.asarray(mask, dtype=bool) & np.isfinite(values) & np.isfinite(dist)
    if not np.any(keep):
        return pd.DataFrame(
            columns=[
                "bin_idx",
                "bin_start_um",
                "bin_end_um",
                "bin_mid_um",
                "count_px",
                "mean_value",
                "median_value",
                "std_value",
                "sem_value",
            ]
        )

    bin_idx = np.floor(dist[keep] / float(bin_um)).astype(np.int32)
    flat = pd.DataFrame(
        {
            "bin_idx": bin_idx,
            "value": values[keep].astype(np.float32),
        }
    )
    grouped = flat.groupby("bin_idx", sort=True)["value"].agg(["count", "mean", "median", "std"]).reset_index()
    grouped = grouped.rename(
        columns={
            "count": "count_px",
            "mean": "mean_value",
            "median": "median_value",
            "std": "std_value",
        }
    )
    grouped["sem_value"] = grouped["std_value"] / np.sqrt(np.maximum(grouped["count_px"], 1))
    grouped["bin_start_um"] = grouped["bin_idx"] * float(bin_um)
    grouped["bin_end_um"] = grouped["bin_start_um"] + float(bin_um)
    grouped["bin_mid_um"] = grouped["bin_start_um"] + 0.5 * float(bin_um)
    return grouped[
        [
            "bin_idx",
            "bin_start_um",
            "bin_end_um",
            "bin_mid_um",
            "count_px",
            "mean_value",
            "median_value",
            "std_value",
            "sem_value",
        ]
    ].copy()


def _mask_contour(ax: plt.Axes, mask: np.ndarray, color: str = "cyan") -> None:
    if np.any(mask):
        ax.contour(mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=0.8)


def _scatter_beads(ax: plt.Axes, xy: np.ndarray, color: str = "cyan") -> None:
    if xy.size:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=34,
            facecolors="none",
            edgecolors=color,
            linewidths=1.1,
        )


def save_live_fixed_quantification_qc(
    out_path: Path,
    canonical_position: str,
    live_bf: np.ndarray,
    live_yfp_bgsub: np.ndarray,
    live_mask: np.ndarray,
    live_beads_xy: np.ndarray,
    live_distance_um: np.ndarray,
    fixed_bf: np.ndarray,
    fixed_dapi_raw: np.ndarray,
    fixed_mask: np.ndarray,
    fixed_beads_xy: np.ndarray,
    fixed_yfp_ratio: np.ndarray,
    fixed_sox2_ratio: np.ndarray,
    fixed_t_ratio: np.ndarray,
    warped_fixed_bf_to_live: np.ndarray,
    warped_fixed_dapi_to_live: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    live_bf_n = common._robust_rescale(live_bf)
    live_yfp_n = common._robust_rescale(live_yfp_bgsub)
    fixed_bf_n = common._robust_rescale(fixed_bf)
    fixed_dapi_n = common._robust_rescale(fixed_dapi_raw)
    fixed_yfp_ratio_n = common._robust_rescale(fixed_yfp_ratio)
    fixed_sox2_ratio_n = common._robust_rescale(fixed_sox2_ratio)
    fixed_t_ratio_n = common._robust_rescale(fixed_t_ratio)
    warped_fixed_bf_n = common._robust_rescale(np.nan_to_num(warped_fixed_bf_to_live, nan=np.nanmedian(warped_fixed_bf_to_live)))
    warped_fixed_dapi_n = common._robust_rescale(np.nan_to_num(warped_fixed_dapi_to_live, nan=np.nanmedian(warped_fixed_dapi_to_live)))
    live_dist = np.asarray(live_distance_um, dtype=np.float32)
    finite_dist = live_dist[np.isfinite(live_dist)]
    dist_vmax = float(np.quantile(finite_dist, 0.98)) if finite_dist.size else 1.0

    overlay = np.zeros((*live_bf.shape, 3), dtype=np.float32)
    overlay[..., 0] = warped_fixed_bf_n
    overlay[..., 1] = live_bf_n
    overlay[..., 2] = live_bf_n

    fig, axes = plt.subplots(2, 5, figsize=(21, 8.2), constrained_layout=True)
    axs = axes.ravel()

    axs[0].imshow(live_bf_n, cmap="gray")
    _mask_contour(axs[0], live_mask, color="cyan")
    _scatter_beads(axs[0], live_beads_xy, color="magenta")
    axs[0].set_title("Live BF + transferred mask")

    axs[1].imshow(live_yfp_n, cmap="Greens")
    _mask_contour(axs[1], live_mask, color="white")
    _scatter_beads(axs[1], live_beads_xy, color="magenta")
    axs[1].set_title("Live TagYFP bg-subtracted")

    im2 = axs[2].imshow(live_dist, cmap="viridis", vmin=0.0, vmax=max(dist_vmax, 1.0))
    _mask_contour(axs[2], live_mask, color="white")
    _scatter_beads(axs[2], live_beads_xy, color="white")
    axs[2].set_title("Live nearest-bead distance (um)")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    axs[3].imshow(overlay)
    _mask_contour(axs[3], live_mask, color="yellow")
    axs[3].set_title("Overlay: fixed BF (R) vs live BF (G/B)")

    axs[4].imshow(warped_fixed_dapi_n, cmap="magma")
    _mask_contour(axs[4], live_mask, color="cyan")
    axs[4].set_title("Warped fixed DAPI raw in live frame")

    axs[5].imshow(fixed_bf_n, cmap="gray")
    _mask_contour(axs[5], fixed_mask, color="cyan")
    _scatter_beads(axs[5], fixed_beads_xy, color="magenta")
    axs[5].set_title("Fixed BF + DAPI mask")

    axs[6].imshow(fixed_dapi_n, cmap="magma")
    _mask_contour(axs[6], fixed_mask, color="cyan")
    _scatter_beads(axs[6], fixed_beads_xy, color="white")
    axs[6].set_title("Fixed DAPI raw")

    axs[7].imshow(fixed_yfp_ratio_n, cmap="Greens")
    _mask_contour(axs[7], fixed_mask, color="white")
    axs[7].set_title("Fixed TagYFP / DAPI (raw ratio)")

    axs[8].imshow(fixed_sox2_ratio_n, cmap="Oranges")
    _mask_contour(axs[8], fixed_mask, color="white")
    axs[8].set_title("Fixed SOX2 / DAPI (raw ratio)")

    axs[9].imshow(fixed_t_ratio_n, cmap="Purples")
    _mask_contour(axs[9], fixed_mask, color="white")
    axs[9].set_title("Fixed T / DAPI (raw ratio)")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(canonical_position, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _canonical_sort_values(values: Iterable[str]) -> List[str]:
    def _key(raw: str) -> Tuple[int, int]:
        s = str(raw)
        try:
            a, b = s.split("-", 1)
            return int(a), int(b)
        except Exception:
            return (10**9, 10**9)

    return sorted({str(v) for v in values}, key=_key)


def run_live_fixed_small_quantification_pipeline(
    position_manifest: Union[str, Path, pd.DataFrame] = "results/manifests/analysis_position_manifest.tsv",
    large_centroid_path: Union[str, Path] = "results/annotations/well_centroids_large.tsv",
    live_small_centroid_path: Union[str, Path] = "results/annotations/well_centroids_small_mapped.tsv",
    fixed_small_centroid_path: Union[str, Path] = "results/annotations/well_centroids_fixed_small_mapped.tsv",
    fixed_transform_path: Union[str, Path] = "results/annotations/fixed_small_to_large_pair_transforms.tsv",
    live_fixed_alignment_path: Union[str, Path] = "results/annotations/live_fixed_alignment_transforms.tsv",
    outdir: Union[str, Path] = "results/measurements/live_fixed_small",
    qc_dir: Union[str, Path] = "results/qc/05_live_quantification",
    mask_dir: Union[str, Path] = "results/masks/live_fixed_small",
    root: Optional[Union[str, Path]] = None,
    bin_um: float = 35.0,
    live_cohort_id: str = "2026-01-22_day2_live",
    fixed_cohort_id: str = "2026-01-22_day2_fix",
    canonical_positions: Optional[Sequence[str]] = None,
    max_positions: Optional[int] = None,
    live_bg_params: Optional[dict] = None,
    fixed_bg_params: Optional[Dict[str, dict]] = None,
    fixed_illumination_fields: Optional[Dict[str, np.ndarray]] = None,
    write_outputs: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    project_root = _resolve_project_root(root=root, position_manifest=position_manifest)
    pos_df = _filter_positions(
        position_manifest=position_manifest,
        cohort_ids=[live_cohort_id, fixed_cohort_id],
        canonical_positions=canonical_positions,
    )
    if max_positions is not None:
        keep_positions = _canonical_sort_values(pos_df["canonical_position"])[: int(max_positions)]
        pos_df = pos_df[pos_df["canonical_position"].isin(keep_positions)].copy()

    live_pos = pos_df[pos_df["cohort_id"] == live_cohort_id].copy()
    fixed_pos = pos_df[pos_df["cohort_id"] == fixed_cohort_id].copy()
    common_positions = _canonical_sort_values(set(live_pos["canonical_position"]).intersection(set(fixed_pos["canonical_position"])))
    if canonical_positions is not None:
        keep = set(map(str, canonical_positions))
        common_positions = [p for p in common_positions if p in keep]
    if max_positions is not None:
        common_positions = common_positions[: int(max_positions)]

    if isinstance(live_small_centroid_path, pd.DataFrame):
        live_cent_df = live_small_centroid_path.copy()
    else:
        live_cent_df = pd.read_csv(Path(live_small_centroid_path), sep="\t")
    live_cent_df = live_cent_df[
        live_cent_df["canonical_position"].astype(str).isin(common_positions)
    ].copy()

    if isinstance(fixed_small_centroid_path, pd.DataFrame):
        fixed_cent_df = fixed_small_centroid_path.copy()
    else:
        fixed_small_centroid_file = Path(fixed_small_centroid_path)
        if fixed_small_centroid_file.exists():
            fixed_cent_df = pd.read_csv(fixed_small_centroid_file, sep="\t")
        else:
            fixed_cent_df = build_fixed_small_mapped_centroid_table(
                large_centroid_path=large_centroid_path,
                live_fixed_alignment_path=live_fixed_alignment_path,
                fixed_transform_path=fixed_transform_path,
                out_path=fixed_small_centroid_file,
                canonical_positions=common_positions,
                verbose=verbose,
            )
    fixed_cent_df = fixed_cent_df[
        fixed_cent_df["canonical_position"].astype(str).isin(common_positions)
    ].copy()

    align_df = pd.read_csv(Path(live_fixed_alignment_path), sep="\t")
    align_df = align_df[align_df["canonical_position"].astype(str).isin(common_positions)].copy()
    align_by_pos = {str(r.canonical_position): pd.Series(r._asdict()) for r in align_df.itertuples(index=False)}

    if live_bg_params is None:
        _, live_bg_payload = estimate_background_for_manifest_subset(
            position_manifest=live_pos,
            cohort_ids=[live_cohort_id],
            channel_specs={"live_tagyfp": ["tagyfp", "foxf1", "yfp"]},
        )
        live_bg_params = live_bg_payload["live_tagyfp"]

    # Fixed-image quantification now uses raw marker/raw DAPI ratios.
    # We keep the parameter for compatibility but do not use it.

    outdir = Path(outdir)
    qc_dir = Path(qc_dir)
    mask_dir = Path(mask_dir)
    live_mask_dir = mask_dir / "live_mask_npz"
    fixed_mask_dir = mask_dir / "fixed_mask_npz"
    for p in [outdir, qc_dir, live_mask_dir, fixed_mask_dir]:
        p.mkdir(parents=True, exist_ok=True)

    mask_rows: List[Dict[str, object]] = []
    live_rows: List[Dict[str, object]] = []
    fixed_rows: List[Dict[str, object]] = []

    live_pos_by_cp = {str(r.canonical_position): pd.Series(r._asdict()) for r in live_pos.itertuples(index=False)}
    fixed_pos_by_cp = {str(r.canonical_position): pd.Series(r._asdict()) for r in fixed_pos.itertuples(index=False)}

    total = len(common_positions)
    for idx, pos in enumerate(common_positions, start=1):
        label = f"[{idx}/{total}] {pos}"
        live_row = live_pos_by_cp.get(str(pos))
        fixed_row = fixed_pos_by_cp.get(str(pos))
        align_row = align_by_pos.get(str(pos))
        live_image_id = f"{live_cohort_id}_{pos}"
        fixed_image_id = f"{fixed_cohort_id}_{pos}"
        qc_png = qc_dir / f"{pos}_live_fixed_quantification_qc.png"
        fixed_mask_npz = fixed_mask_dir / f"{pos}_fixed_mask.npz"
        live_mask_npz = live_mask_dir / f"{pos}_live_mask_from_fixed.npz"

        if live_row is None or fixed_row is None:
            mask_rows.append(
                {
                    "canonical_position": pos,
                    "live_image_id": live_image_id if live_row is not None else np.nan,
                    "fixed_image_id": fixed_image_id if fixed_row is not None else np.nan,
                    "status": "missing_manifest_rows",
                    "fixed_mask_area_px": np.nan,
                    "fixed_mask_fraction": np.nan,
                    "live_mask_area_px": np.nan,
                    "live_mask_fraction": np.nan,
                    "fixed_mask_components": np.nan,
                    "fixed_dapi_threshold": np.nan,
                    "fixed_dapi_bg_mu": np.nan,
                    "fixed_dapi_bg_sigma": np.nan,
                    "fixed_dapi_threshold_sigma": np.nan,
                    "fixed_dapi_core_threshold": np.nan,
                    "fixed_dapi_core_bg_mu": np.nan,
                    "fixed_dapi_core_bg_sigma": np.nan,
                    "fixed_dapi_core_threshold_sigma": np.nan,
                    "fixed_dapi_core_blur_sigma_px": np.nan,
                    "fixed_dapi_blur_sigma_px": np.nan,
                    "n_live_beads_total": np.nan,
                    "n_live_beads_inside": np.nan,
                    "n_fixed_beads_total": np.nan,
                    "n_fixed_beads_inside": np.nan,
                    "live_qc_png_path": np.nan,
                    "fixed_qc_png_path": np.nan,
                    "combined_qc_png_path": np.nan,
                    "updated_at": _iso_now(),
                }
            )
            if verbose:
                print(f"{label} missing manifest rows")
            continue

        if align_row is None or str(align_row.get("status", "")) != "ok":
            mask_rows.append(
                {
                    "canonical_position": pos,
                    "live_image_id": live_image_id,
                    "fixed_image_id": fixed_image_id,
                    "status": "missing_live_fixed_alignment",
                    "fixed_mask_area_px": np.nan,
                    "fixed_mask_fraction": np.nan,
                    "live_mask_area_px": np.nan,
                    "live_mask_fraction": np.nan,
                    "fixed_mask_components": np.nan,
                    "fixed_dapi_threshold": np.nan,
                    "fixed_dapi_bg_mu": np.nan,
                    "fixed_dapi_bg_sigma": np.nan,
                    "fixed_dapi_threshold_sigma": np.nan,
                    "fixed_dapi_core_threshold": np.nan,
                    "fixed_dapi_core_bg_mu": np.nan,
                    "fixed_dapi_core_bg_sigma": np.nan,
                    "fixed_dapi_core_threshold_sigma": np.nan,
                    "fixed_dapi_core_blur_sigma_px": np.nan,
                    "fixed_dapi_blur_sigma_px": np.nan,
                    "n_live_beads_total": np.nan,
                    "n_live_beads_inside": np.nan,
                    "n_fixed_beads_total": np.nan,
                    "n_fixed_beads_inside": np.nan,
                    "live_qc_png_path": np.nan,
                    "fixed_qc_png_path": np.nan,
                    "combined_qc_png_path": np.nan,
                    "updated_at": _iso_now(),
                }
            )
            if verbose:
                print(f"{label} missing live-fixed alignment")
            continue

        try:
            live_img = common.read_czi(_resolve_project_path(live_row["primary_analysis_file"], project_root))
            fixed_img = read_czi_with_optional_fixed_small_plane_selection(
                _resolve_project_path(fixed_row["primary_analysis_file"], project_root),
                canonical_position=str(pos),
            )

            live_bf_idx = common.find_channel_index(live_img.channels, ["bright"])
            live_yfp_idx = common.find_channel_index(live_img.channels, ["tagyfp", "foxf1", "yfp"])
            fixed_bf_idx = common.find_channel_index(fixed_img.channels, ["bright"])
            fixed_dapi_idx = common.find_channel_index(fixed_img.channels, ["dapi"])
            fixed_yfp_idx = common.find_channel_index(fixed_img.channels, ["tagyfp", "foxf1", "yfp"])
            fixed_t_idx = common.find_channel_index(fixed_img.channels, ["647", "alexa fluor 647"])
            fixed_sox2_idx = common.find_channel_index(fixed_img.channels, ["568", "alexa fluor 568"])
            needed = [live_bf_idx, live_yfp_idx, fixed_bf_idx, fixed_dapi_idx, fixed_yfp_idx, fixed_t_idx, fixed_sox2_idx]
            if any(v is None for v in needed):
                raise RuntimeError(
                    "Missing required channels. "
                    f"live_channels={live_img.channels} fixed_channels={fixed_img.channels}"
                )

            live_bf = live_img.channel_images[int(live_bf_idx)].astype(np.float32)
            live_yfp_raw = live_img.channel_images[int(live_yfp_idx)].astype(np.float32)
            fixed_bf = fixed_img.channel_images[int(fixed_bf_idx)].astype(np.float32)
            fixed_dapi_raw = fixed_img.channel_images[int(fixed_dapi_idx)].astype(np.float32)
            fixed_yfp_raw = fixed_img.channel_images[int(fixed_yfp_idx)].astype(np.float32)
            fixed_t_raw = fixed_img.channel_images[int(fixed_t_idx)].astype(np.float32)
            fixed_sox2_raw = fixed_img.channel_images[int(fixed_sox2_idx)].astype(np.float32)

            live_yfp_bgsub = common.subtract_uniform_background(
                live_yfp_raw,
                mu_bg_raw=float(live_bg_params["mu_bg_raw"]),
                clip_below_zero=True,
            )
            fixed_yfp_ratio = fixed_yfp_raw.astype(np.float32) / np.maximum(fixed_dapi_raw.astype(np.float32), float(EPS))
            fixed_t_ratio = fixed_t_raw.astype(np.float32) / np.maximum(fixed_dapi_raw.astype(np.float32), float(EPS))
            fixed_sox2_ratio = fixed_sox2_raw.astype(np.float32) / np.maximum(fixed_dapi_raw.astype(np.float32), float(EPS))

            seg = segment_fixed_dapi_cyst_mask(
                fixed_dapi_raw,
                canonical_position=str(pos),
            )
            fixed_mask = np.asarray(seg["mask"], dtype=bool)
            fixed_valid = (
                np.isfinite(fixed_dapi_raw)
                & np.isfinite(fixed_yfp_raw)
                & np.isfinite(fixed_t_raw)
                & np.isfinite(fixed_sox2_raw)
                & (fixed_dapi_raw > 0)
                & (fixed_yfp_raw > 0)
                & (fixed_t_raw > 0)
                & (fixed_sox2_raw > 0)
            )
            fixed_mask &= fixed_valid
            fixed_small_to_live_small = lfa.affine_matrix_from_row_prefix(
                align_row,
                "fixed_small_to_live_small",
            )
            live_mask = lfa.warp_image_with_affine(
                image=fixed_mask.astype(np.float32),
                forward_mat=fixed_small_to_live_small,
                output_shape_yx=live_bf.shape,
                order=0,
                cval=0.0,
            ) > 0.5
            live_valid = np.isfinite(live_yfp_raw) & (live_yfp_raw > 0)
            live_mask &= live_valid

            live_cent_sub = live_cent_df[
                (live_cent_df["canonical_position"].astype(str) == str(pos))
                & (live_cent_df["mapping_status"] == "ok")
                & (live_cent_df["annotation_status"] == "annotated")
            ].copy()
            fixed_cent_sub = fixed_cent_df[
                (fixed_cent_df["canonical_position"].astype(str) == str(pos))
                & (fixed_cent_df["mapping_status"] == "ok")
                & (fixed_cent_df["annotation_status"] == "annotated")
            ].copy()

            live_beads_xy = live_cent_sub[["centroid_small_x_px", "centroid_small_y_px"]].to_numpy(dtype=np.float32)
            fixed_beads_xy = fixed_cent_sub[["centroid_fixed_small_x_px", "centroid_fixed_small_y_px"]].to_numpy(dtype=np.float32)
            if live_beads_xy.size == 0 or fixed_beads_xy.size == 0:
                raise RuntimeError("Missing mapped bead centroids in live or fixed small space.")

            live_dist_um = common.nearest_bead_distance_map_um(
                image_shape_yx=live_bf.shape,
                centroid_xy_px=live_beads_xy,
                pixel_um_x=float(live_img.pixel_um_x),
                pixel_um_y=float(live_img.pixel_um_y),
            )
            fixed_dist_um = common.nearest_bead_distance_map_um(
                image_shape_yx=fixed_bf.shape,
                centroid_xy_px=fixed_beads_xy,
                pixel_um_x=float(fixed_img.pixel_um_x),
                pixel_um_y=float(fixed_img.pixel_um_y),
            )

            live_stats = masked_distance_bin_stats(
                value_img=live_yfp_bgsub,
                distance_um_map=live_dist_um,
                mask=live_mask,
                bin_um=float(bin_um),
            )
            for _, tr in live_stats.iterrows():
                live_rows.append(
                    {
                            "canonical_position": pos,
                            "image_id": live_image_id,
                        "cohort_id": live_cohort_id,
                        "small_file_path": live_row["primary_analysis_file"],
                        "measurement_name": "live_tagyfp_bgsub",
                        "mask_source": "fixed_dapi_transferred",
                        "bead_distance_mode": "nearest",
                        "bin_idx": int(tr["bin_idx"]),
                        "bin_start_um": float(tr["bin_start_um"]),
                        "bin_end_um": float(tr["bin_end_um"]),
                        "bin_mid_um": float(tr["bin_mid_um"]),
                        "count_px": int(tr["count_px"]),
                        "mean_value": float(tr["mean_value"]),
                        "median_value": float(tr["median_value"]),
                        "std_value": float(tr["std_value"]) if pd.notna(tr["std_value"]) else np.nan,
                        "sem_value": float(tr["sem_value"]) if pd.notna(tr["sem_value"]) else np.nan,
                        "updated_at": _iso_now(),
                    }
                )

            fixed_signal_map = {
                "fixed_dapi_raw": fixed_dapi_raw,
                "fixed_tagyfp_over_dapi_raw": fixed_yfp_ratio,
                "fixed_sox2_over_dapi_raw": fixed_sox2_ratio,
                "fixed_t_over_dapi_raw": fixed_t_ratio,
            }
            for meas_name, arr in fixed_signal_map.items():
                stats = masked_distance_bin_stats(
                    value_img=arr,
                    distance_um_map=fixed_dist_um,
                    mask=fixed_mask,
                    bin_um=float(bin_um),
                )
                for _, tr in stats.iterrows():
                    fixed_rows.append(
                        {
                            "canonical_position": pos,
                            "image_id": fixed_image_id,
                            "cohort_id": fixed_cohort_id,
                            "small_file_path": fixed_row["primary_analysis_file"],
                            "measurement_name": meas_name,
                            "mask_source": "fixed_dapi_native",
                            "bead_distance_mode": "nearest",
                            "bin_idx": int(tr["bin_idx"]),
                            "bin_start_um": float(tr["bin_start_um"]),
                            "bin_end_um": float(tr["bin_end_um"]),
                            "bin_mid_um": float(tr["bin_mid_um"]),
                            "count_px": int(tr["count_px"]),
                            "mean_value": float(tr["mean_value"]),
                            "median_value": float(tr["median_value"]),
                            "std_value": float(tr["std_value"]) if pd.notna(tr["std_value"]) else np.nan,
                            "sem_value": float(tr["sem_value"]) if pd.notna(tr["sem_value"]) else np.nan,
                            "updated_at": _iso_now(),
                        }
                    )

            warped_fixed_bf_to_live = lfa.warp_image_with_affine(
                image=fixed_bf,
                forward_mat=fixed_small_to_live_small,
                output_shape_yx=live_bf.shape,
                order=1,
                cval=np.nan,
            )
            warped_fixed_dapi_to_live = lfa.warp_image_with_affine(
                image=fixed_dapi_raw,
                forward_mat=fixed_small_to_live_small,
                output_shape_yx=live_bf.shape,
                order=1,
                cval=np.nan,
            )

            save_live_fixed_quantification_qc(
                out_path=qc_png,
                canonical_position=str(pos),
                live_bf=live_bf,
                live_yfp_bgsub=live_yfp_bgsub,
                live_mask=live_mask,
                live_beads_xy=live_beads_xy,
                live_distance_um=live_dist_um,
                fixed_bf=fixed_bf,
                fixed_dapi_raw=fixed_dapi_raw,
                fixed_mask=fixed_mask,
                fixed_beads_xy=fixed_beads_xy,
                fixed_yfp_ratio=fixed_yfp_ratio,
                fixed_sox2_ratio=fixed_sox2_ratio,
                fixed_t_ratio=fixed_t_ratio,
                warped_fixed_bf_to_live=warped_fixed_bf_to_live,
                warped_fixed_dapi_to_live=warped_fixed_dapi_to_live,
            )

            np.savez_compressed(fixed_mask_npz, mask=fixed_mask.astype(np.uint8))
            np.savez_compressed(live_mask_npz, mask=live_mask.astype(np.uint8))

            mask_rows.append(
                {
                    "canonical_position": pos,
                    "live_image_id": live_image_id,
                    "fixed_image_id": fixed_image_id,
                    "status": "ok",
                    "fixed_mask_area_px": int(np.sum(fixed_mask)),
                    "fixed_mask_fraction": float(np.mean(fixed_mask)),
                    "live_mask_area_px": int(np.sum(live_mask)),
                    "live_mask_fraction": float(np.mean(live_mask)),
                    "fixed_mask_components": int(seg["n_components"]),
                    "fixed_dapi_threshold": float(seg["threshold"]),
                    "fixed_dapi_bg_mu": float(seg["mu_bg"]),
                    "fixed_dapi_bg_sigma": float(seg["sigma_bg"]),
                    "fixed_dapi_threshold_sigma": float(seg["threshold_sigma"]),
                    "fixed_dapi_core_threshold": float(seg["core_threshold"]),
                    "fixed_dapi_core_bg_mu": float(seg["core_mu_bg"]),
                    "fixed_dapi_core_bg_sigma": float(seg["core_sigma_bg"]),
                    "fixed_dapi_core_threshold_sigma": float(seg["core_threshold_sigma"]),
                    "fixed_dapi_core_blur_sigma_px": float(seg["core_blur_sigma_px"]),
                    "fixed_dapi_blur_sigma_px": float(seg["blur_sigma_px"]),
                    "fixed_artifact_exclusion_applied": bool(seg.get("artifact_exclusion_applied", False)),
                    "fixed_artifact_exclusion_area_px": int(seg.get("artifact_exclusion_area_px", 0)),
                    "fixed_artifact_exclusion_reasons": ";".join(seg.get("artifact_exclusion_reasons", [])),
                    "n_live_beads_total": int(len(live_cent_sub)),
                    "n_live_beads_inside": int(live_cent_sub["inside_small_fov"].fillna(False).astype(bool).sum()),
                    "n_fixed_beads_total": int(len(fixed_cent_sub)),
                    "n_fixed_beads_inside": int(fixed_cent_sub["inside_fixed_small_fov"].fillna(False).astype(bool).sum()),
                    "live_qc_png_path": str(qc_png.relative_to(project_root)),
                    "fixed_qc_png_path": str(qc_png.relative_to(project_root)),
                    "combined_qc_png_path": str(qc_png.relative_to(project_root)),
                    "updated_at": _iso_now(),
                }
            )
            if verbose:
                print(
                    f"{label} ok live_mask={int(np.sum(live_mask))} fixed_mask={int(np.sum(fixed_mask))} "
                    f"live_beads={len(live_cent_sub)} fixed_beads={len(fixed_cent_sub)}"
                )
        except Exception as exc:
            mask_rows.append(
                {
                    "canonical_position": pos,
                    "live_image_id": live_image_id,
                    "fixed_image_id": fixed_image_id,
                    "status": f"error:{exc}",
                    "fixed_mask_area_px": np.nan,
                    "fixed_mask_fraction": np.nan,
                    "live_mask_area_px": np.nan,
                    "live_mask_fraction": np.nan,
                    "fixed_mask_components": np.nan,
                    "fixed_dapi_threshold": np.nan,
                    "fixed_dapi_bg_mu": np.nan,
                    "fixed_dapi_bg_sigma": np.nan,
                    "fixed_dapi_threshold_sigma": np.nan,
                    "fixed_dapi_core_threshold": np.nan,
                    "fixed_dapi_core_bg_mu": np.nan,
                    "fixed_dapi_core_bg_sigma": np.nan,
                    "fixed_dapi_core_threshold_sigma": np.nan,
                    "fixed_dapi_core_blur_sigma_px": np.nan,
                    "fixed_dapi_blur_sigma_px": np.nan,
                    "fixed_artifact_exclusion_applied": np.nan,
                    "fixed_artifact_exclusion_area_px": np.nan,
                    "fixed_artifact_exclusion_reasons": np.nan,
                    "n_live_beads_total": np.nan,
                    "n_live_beads_inside": np.nan,
                    "n_fixed_beads_total": np.nan,
                    "n_fixed_beads_inside": np.nan,
                    "live_qc_png_path": np.nan,
                    "fixed_qc_png_path": np.nan,
                    "combined_qc_png_path": np.nan,
                    "updated_at": _iso_now(),
                }
            )
            if verbose:
                print(f"{label} error: {exc}")

    mask_df = pd.DataFrame(mask_rows, columns=MASK_SUMMARY_COLUMNS)
    live_stats_df = pd.DataFrame(live_rows)
    fixed_stats_df = pd.DataFrame(fixed_rows)

    live_bg_df = pd.DataFrame(
        [
            {
                "channel": "live_tagyfp",
                "mu_bg_raw": float(live_bg_params["mu_bg_raw"]),
                "sigma_bg_raw": float(live_bg_params["sigma_bg_raw"]),
                "mode_idx": int(live_bg_params["mode_idx"]),
                "n_images": int(live_bg_params["n_images"]),
                "n_pixels": int(live_bg_params["n_pixels"]),
                "global_min": float(live_bg_params["global_min"]),
                "global_max": float(live_bg_params["global_max"]),
                "fit_method": str(live_bg_params["fit_method"]),
            }
        ]
    )
    fixed_bg_df = pd.DataFrame(
        [
            {
                "channel": "fixed_raw_ratio_strategy",
                "mu_bg_raw": np.nan,
                "sigma_bg_raw": np.nan,
                "mode_idx": np.nan,
                "n_images": len(common_positions),
                "n_pixels": np.nan,
                "global_min": np.nan,
                "global_max": np.nan,
                "fit_method": "not_used;fixed_measurements_use_raw_marker_over_raw_dapi",
            }
        ]
    )

    summary_txt = (
        f"positions_total\t{len(common_positions)}\n"
        f"positions_ok\t{int((mask_df['status'] == 'ok').sum())}\n"
        f"positions_error\t{int((mask_df['status'] != 'ok').sum())}\n"
        f"live_bin_rows\t{len(live_stats_df)}\n"
        f"fixed_bin_rows\t{len(fixed_stats_df)}\n"
        f"live_bg_mu_raw\t{float(live_bg_params['mu_bg_raw']):.6f}\n"
        f"live_bg_sigma_raw\t{float(live_bg_params['sigma_bg_raw']):.6f}\n"
        f"fixed_measurement_strategy\traw_marker_over_raw_dapi\n"
    )

    if write_outputs:
        fixed_centroid_out = Path(fixed_small_centroid_path) if not isinstance(fixed_small_centroid_path, pd.DataFrame) else None
        if fixed_centroid_out is not None and not fixed_centroid_out.exists():
            fixed_cent_df.to_csv(fixed_centroid_out, sep="\t", index=False)

        live_bg_df.to_csv(outdir / "global_live_background_params.tsv", sep="\t", index=False)
        fixed_bg_df.to_csv(outdir / "global_fixed_background_params.tsv", sep="\t", index=False)
        mask_df.to_csv(outdir / "mask_summary.tsv", sep="\t", index=False)
        live_stats_df.to_csv(outdir / "live_foxf1_pixel_bin_stats.tsv", sep="\t", index=False)
        fixed_stats_df.to_csv(outdir / "fixed_marker_pixel_bin_stats.tsv", sep="\t", index=False)
        (outdir / "measurement_summary.txt").write_text(summary_txt)

    return {
        "common_positions": common_positions,
        "live_small_centroids_df": live_cent_df,
        "fixed_small_centroids_df": fixed_cent_df,
        "mask_summary_df": mask_df,
        "live_stats_df": live_stats_df,
        "fixed_stats_df": fixed_stats_df,
        "live_bg_df": live_bg_df,
        "fixed_bg_df": fixed_bg_df,
        "live_bg_params": live_bg_params,
        "fixed_bg_params": fixed_bg_params,
    }


__all__ = [
    "FIXED_SMALL_CENTROID_COLUMNS",
    "MASK_SUMMARY_COLUMNS",
    "DEFAULT_FIXED_DAPI_MASK_BLUR_SIGMA_PX",
    "DEFAULT_FIXED_DAPI_THRESHOLD_SIGMA",
    "DEFAULT_FIXED_DAPI_CORE_BLUR_SIGMA_PX",
    "DEFAULT_FIXED_DAPI_CORE_THRESHOLD_SCALE",
    "DEFAULT_FIXED_DAPI_CORE_THRESHOLD_SIGMA",
    "DEFAULT_FIXED_DAPI_MONOLAYER_TOUCH_RADIUS_PX",
    "APPROVED_FINAL04_FIXED_DAPI_CORE_THRESHOLD_SCALE",
    "APPROVED_FINAL04_FIXED_DAPI_CORE_FILL_HOLES",
    "APPROVED_FINAL04_FIXED_DAPI_CORE_HOLE_AREA_PX",
    "APPROVED_FINAL04_FIXED_DAPI_ROUND2_BLUR_SIGMA_PX",
    "APPROVED_FINAL04_FIXED_DAPI_ROUND2_THRESHOLD_SIGMA",
    "APPROVED_FINAL04_FIXED_DAPI_ROUND2_MIN_AREA_PX",
    "APPROVED_FINAL04_FIXED_DAPI_ROUND2_CLOSING_RADIUS_PX",
    "APPROVED_FINAL04_FIXED_DAPI_ROUND2_BLUR_MODE",
    "apply_illumination_field",
    "background_histogram_from_pixels",
    "build_final04_fixed_dapi_masks",
    "fit_background_from_image_pixels",
    "build_fixed_small_mapped_centroid_table",
    "cleanup_binary_mask",
    "estimate_live_masked_illumination_field",
    "estimate_background_for_manifest_subset",
    "estimate_masked_illumination_fields_for_manifest_subset",
    "load_czi_bundle_with_z",
    "masked_distance_bin_stats",
    "masked_gaussian_blur_excluding_mask",
    "normalize_channel_from_background",
    "plot_raw_processed_distribution_panels",
    "plot_background_fit_panels",
    "plot_illumination_field_panels",
    "pooled_histogram_manifest_channel",
    "pooled_histogram_manifest_ratio",
    "pooled_histogram_transformed_channel_images",
    "prepare_fixed_dapi_for_mask",
    "selected_fixed_small_dapi_plane",
    "run_live_fixed_small_quantification_pipeline",
    "save_live_fixed_quantification_qc",
    "segment_fixed_dapi_cyst_mask",
    "smooth_masked_image",
    "transfer_fixed_masks_to_live_space",
]
