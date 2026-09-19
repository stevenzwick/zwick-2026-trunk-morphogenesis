"""Shared helpers for the active FOXF1 bead notebook pipeline.

This module contains the manifest, CZI I/O, background, and distance-map
helpers used by the current notebook-driven workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

# czifile cannot be installed beside this project's pinned numpy, and mis-frames odd-sized
# acquisitions. czi_compat presents czifile's interface over libCZI. See src/trunk_morph_ref/czi_compat.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from src.trunk_morph_ref import czi_compat as czifile
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


EPS = 1e-9


@dataclass
class CziImage:
    path: Path
    channels: List[str]
    channel_images: np.ndarray  # shape: (C, Y, X), max-projected over Z/T if present
    pixel_um_x: float
    pixel_um_y: float
    raw_axes: str
    raw_shape: Tuple[int, ...]


def _squeeze_axes(arr: np.ndarray, axes: str) -> Tuple[np.ndarray, str]:
    """Drop singleton and synthetic '0' axes while keeping axis labels aligned."""
    out = arr
    out_axes = list(axes)
    for idx in range(len(out_axes) - 1, -1, -1):
        ax = out_axes[idx]
        if ax == "0" or out.shape[idx] == 1:
            out = np.take(out, indices=0, axis=idx)
            out_axes.pop(idx)
    return out, "".join(out_axes)


def _collapse_to_cyx(arr: np.ndarray, axes: str) -> Tuple[np.ndarray, str]:
    """Convert any CZI stack variant to (C, Y, X), max-projecting over Z/T."""
    out = arr
    out_axes = axes

    if "C" not in out_axes:
        raise ValueError(f"Missing channel axis. axes={axes}")

    cidx = out_axes.index("C")
    out = np.moveaxis(out, cidx, 0)
    out_axes = "C" + out_axes[:cidx] + out_axes[cidx + 1 :]

    for project_axis in ("Z", "T"):
        if project_axis in out_axes:
            idx = out_axes.index(project_axis)
            out = out.max(axis=idx)
            out_axes = out_axes.replace(project_axis, "")

    for ax in list(out_axes):
        if ax not in ("C", "Y", "X"):
            idx = out_axes.index(ax)
            out = out.max(axis=idx)
            out_axes = out_axes.replace(ax, "")

    if out_axes != "CYX":
        raise ValueError(f"Unexpected collapsed axes={out_axes}, shape={out.shape}")

    return out.astype(np.float32), out_axes


def _extract_channels(metadata: dict, c_count: int) -> List[str]:
    info = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]
    ch = info["Dimensions"]["Channels"]["Channel"]
    if isinstance(ch, dict):
        ch = [ch]

    names = []
    for idx in range(min(len(ch), c_count)):
        names.append(str(ch[idx].get("Name", f"Channel_{idx}")))

    while len(names) < c_count:
        names.append(f"Channel_{len(names)}")
    return names


def _extract_pixel_um(metadata: dict) -> Tuple[float, float]:
    scaling = metadata["ImageDocument"]["Metadata"].get("Scaling", {})
    items = scaling.get("Items", {})
    dist = items.get("Distance", [])
    if isinstance(dist, dict):
        dist = [dist]

    x_um = np.nan
    y_um = np.nan
    for d in dist:
        did = str(d.get("Id", "")).upper()
        val_m = float(d.get("Value", np.nan))
        if did == "X":
            x_um = val_m * 1e6
        elif did == "Y":
            y_um = val_m * 1e6

    if not np.isfinite(x_um):
        x_um = 1.0
    if not np.isfinite(y_um):
        y_um = x_um
    return float(x_um), float(y_um)


def read_czi(path: Path) -> CziImage:
    with czifile.CziFile(path) as czi:
        raw = czi.asarray()
        raw_axes = str(czi.axes)
        metadata = czi.metadata(raw=False)

    squeezed, squeezed_axes = _squeeze_axes(raw, raw_axes)
    cyx, _ = _collapse_to_cyx(squeezed, squeezed_axes)
    channels = _extract_channels(metadata, c_count=cyx.shape[0])
    pixel_um_x, pixel_um_y = _extract_pixel_um(metadata)

    return CziImage(
        path=path,
        channels=channels,
        channel_images=cyx,
        pixel_um_x=pixel_um_x,
        pixel_um_y=pixel_um_y,
        raw_axes=raw_axes,
        raw_shape=tuple(raw.shape),
    )


def find_channel_index(channels: List[str], keywords: List[str]) -> Optional[int]:
    lowered = [c.lower() for c in channels]
    for kw in keywords:
        kw_l = kw.lower()
        for idx, ch in enumerate(lowered):
            if kw_l in ch:
                return idx
    return None


def _robust_rescale(img: np.ndarray, low_q: float = 0.01, high_q: float = 0.995) -> np.ndarray:
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo = float(np.quantile(vals, low_q))
    hi = float(np.quantile(vals, high_q))
    if hi <= lo + EPS:
        hi = lo + 1.0
    out = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def pooled_histogram_channel_images(
    image_paths: Sequence[Union[str, Path]],
    channel_keywords: Sequence[str],
    n_bins: int = 16384,
    exclude_exact_zero_pixels: bool = False,
) -> dict:
    """Accumulate an exact pooled histogram for one channel across many CZI files."""
    if len(image_paths) == 0:
        raise ValueError("image_paths is empty")

    global_min = np.inf
    global_max = -np.inf
    n_pixels = 0
    n_images = 0
    integer_like = True

    for path_like in image_paths:
        img = read_czi(Path(path_like))
        ch_idx = find_channel_index(img.channels, list(channel_keywords))
        if ch_idx is None:
            raise RuntimeError(
                f"Missing channel matching {list(channel_keywords)} in {Path(path_like)}"
            )
        arr = img.channel_images[ch_idx].astype(np.float32)
        finite = arr[np.isfinite(arr)]
        if bool(exclude_exact_zero_pixels):
            finite = finite[finite != 0]
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
        raise RuntimeError("Could not determine finite histogram range")
    if global_max <= global_min:
        global_max = global_min + 1.0

    if bool(integer_like):
        lo_i = int(np.floor(global_min))
        hi_i = int(np.ceil(global_max))
        edges = np.arange(lo_i - 0.5, hi_i + 1.5, 1.0, dtype=np.float64)
    else:
        edges = np.linspace(global_min, global_max, int(n_bins) + 1, dtype=np.float64)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)

    for path_like in image_paths:
        img = read_czi(Path(path_like))
        ch_idx = find_channel_index(img.channels, list(channel_keywords))
        if ch_idx is None:
            raise RuntimeError(
                f"Missing channel matching {list(channel_keywords)} in {Path(path_like)}"
            )
        arr = img.channel_images[ch_idx].astype(np.float32)
        finite = arr[np.isfinite(arr)]
        if bool(exclude_exact_zero_pixels):
            finite = finite[finite != 0]
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


def estimate_background_half_gaussian(
    hist: dict,
    smooth_sigma_bins: float = 6.0,
) -> dict:
    """Estimate one global background model from a pooled histogram."""
    centers = np.asarray(hist["centers"], dtype=np.float64)
    counts = np.asarray(hist["counts"], dtype=np.float64)
    if not np.any(counts > 0):
        raise RuntimeError("Histogram is empty")

    smooth = ndi.gaussian_filter1d(counts, sigma=float(smooth_sigma_bins), mode="nearest")
    mode_idx = int(np.nanargmax(smooth))
    mu_bg = float(centers[mode_idx])

    left = centers <= mu_bg
    left_counts = smooth[left]
    left_centers = centers[left]
    if np.sum(left_counts) <= 0:
        raise RuntimeError("No left-half counts available for sigma estimate")

    amp = float(smooth[mode_idx])
    sigma_init = float(
        np.sqrt(np.sum(left_counts * (left_centers - mu_bg) ** 2) / np.sum(left_counts))
    )
    sigma_init = max(float(EPS), sigma_init)

    bin_width = float(np.median(np.diff(centers))) if len(centers) > 1 else 1.0
    sigma_min = max(float(bin_width), float(EPS))
    sigma_max = max(float(sigma_min) * 1.5, float(sigma_init) * 3.0)
    sigma_grid = np.linspace(sigma_min, sigma_max, 400, dtype=np.float64)
    sse = np.empty_like(sigma_grid)
    for ii, sigma in enumerate(sigma_grid):
        pred_left = amp * np.exp(-0.5 * ((left_centers - mu_bg) / float(sigma)) ** 2)
        sse[ii] = float(np.mean((left_counts - pred_left) ** 2))
    sigma_bg = float(sigma_grid[int(np.argmin(sse))])

    fit_counts = amp * np.exp(-0.5 * ((centers - mu_bg) / sigma_bg) ** 2)

    return {
        **hist,
        "mu_bg_raw": mu_bg,
        "sigma_bg_raw": sigma_bg,
        "mode_idx": mode_idx,
        "smooth_counts": smooth,
        "fit_counts": fit_counts,
        "fit_method": "hist_mode_plus_left_half_gaussian_sse_fit_on_smoothed_histogram",
    }


def subtract_uniform_background(
    image: np.ndarray,
    mu_bg_raw: float,
    clip_below_zero: bool = True,
) -> np.ndarray:
    out = image.astype(np.float32) - float(mu_bg_raw)
    if clip_below_zero:
        out = np.maximum(out, 0.0)
    return out.astype(np.float32)


def nearest_bead_distance_map_um(
    image_shape_yx: Tuple[int, int],
    centroid_xy_px: np.ndarray,
    pixel_um_x: float,
    pixel_um_y: float,
) -> np.ndarray:
    h, w = image_shape_yx
    if centroid_xy_px.size == 0:
        return np.full((h, w), np.nan, dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    dist2_min = np.full((h, w), np.inf, dtype=np.float32)
    for x_px, y_px in np.asarray(centroid_xy_px, dtype=np.float32):
        dx_um = (xx - float(x_px)) * float(pixel_um_x)
        dy_um = (yy - float(y_px)) * float(pixel_um_y)
        dist2 = dx_um * dx_um + dy_um * dy_um
        np.minimum(dist2_min, dist2, out=dist2_min)
    return np.sqrt(dist2_min).astype(np.float32)


def load_position_manifest(
    position_manifest: Union[str, Path, pd.DataFrame]
) -> pd.DataFrame:
    if isinstance(position_manifest, pd.DataFrame):
        return position_manifest.copy()
    return pd.read_csv(Path(position_manifest), sep="\t")


def _infer_project_root_from_manifest(
    position_manifest: Union[str, Path, pd.DataFrame]
) -> Path:
    if isinstance(position_manifest, pd.DataFrame):
        cwd = Path.cwd().resolve()
        if "primary_analysis_file" in position_manifest.columns:
            rel_paths = (
                position_manifest["primary_analysis_file"]
                .dropna()
                .astype(str)
                .tolist()
            )
            for candidate_root in [cwd, *cwd.parents]:
                if (candidate_root / "scripts").exists():
                    for rel_path in rel_paths[:5]:
                        if (candidate_root / Path(rel_path)).exists():
                            return candidate_root
        return cwd
    manifest_path = Path(position_manifest).resolve()
    parts = manifest_path.parts
    if len(parts) >= 3 and manifest_path.parent.name == "manifests":
        return manifest_path.parent.parent.parent
    return manifest_path.parent


def _resolve_project_path(path_like: Union[str, Path], project_root: Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def filter_position_manifest(
    pos_df: pd.DataFrame,
    cohort_ids: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    max_positions: Optional[int] = None,
) -> pd.DataFrame:
    out = pos_df.copy()
    if cohort_ids:
        out = out[out["cohort_id"].isin(list(cohort_ids))]
    if conditions:
        out = out[out["condition"].isin(list(conditions))]
    if canonical_positions:
        out = out[out["canonical_position"].isin(list(canonical_positions))]
    out = out.sort_values(["cohort_id", "canonical_position"]).reset_index(drop=True)
    if max_positions is not None:
        out = out.head(int(max_positions)).copy()
    return out


def primary_image_paths_from_manifest(
    position_manifest: Union[str, Path, pd.DataFrame] = "results/manifests/analysis_position_manifest.tsv",
    cohort_ids: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    max_positions: Optional[int] = None,
) -> List[Path]:
    project_root = _infer_project_root_from_manifest(position_manifest)
    pos_df = load_position_manifest(position_manifest)
    pos_df = filter_position_manifest(
        pos_df=pos_df,
        cohort_ids=cohort_ids,
        conditions=conditions,
        canonical_positions=canonical_positions,
        max_positions=max_positions,
    )
    return [
        _resolve_project_path(row["primary_analysis_file"], project_root)
        for _, row in pos_df.iterrows()
    ]


def estimate_global_channel_background_from_manifest(
    position_manifest: Union[str, Path, pd.DataFrame] = "results/manifests/analysis_position_manifest.tsv",
    channel_keywords: Sequence[str] = ("tagyfp", "foxf1", "yfp"),
    cohort_ids: Optional[Sequence[str]] = None,
    conditions: Optional[Sequence[str]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    max_positions: Optional[int] = None,
    n_bins: int = 16384,
    smooth_sigma_bins: float = 6.0,
    exclude_exact_zero_pixels: bool = False,
) -> dict:
    image_paths = primary_image_paths_from_manifest(
        position_manifest=position_manifest,
        cohort_ids=cohort_ids,
        conditions=conditions,
        canonical_positions=canonical_positions,
        max_positions=max_positions,
    )
    hist = pooled_histogram_channel_images(
        image_paths=image_paths,
        channel_keywords=channel_keywords,
        n_bins=int(n_bins),
        exclude_exact_zero_pixels=bool(exclude_exact_zero_pixels),
    )
    fit = estimate_background_half_gaussian(
        hist=hist,
        smooth_sigma_bins=float(smooth_sigma_bins),
    )
    fit["exclude_exact_zero_pixels"] = bool(exclude_exact_zero_pixels)
    if bool(exclude_exact_zero_pixels):
        fit["fit_method"] = f"{fit['fit_method']};exclude_exact_zero_pixels"
    return fit


__all__ = [
    "CziImage",
    "_extract_channels",
    "_extract_pixel_um",
    "_infer_project_root_from_manifest",
    "_resolve_project_path",
    "_robust_rescale",
    "_squeeze_axes",
    "estimate_background_half_gaussian",
    "estimate_global_channel_background_from_manifest",
    "filter_position_manifest",
    "find_channel_index",
    "load_position_manifest",
    "nearest_bead_distance_map_um",
    "pooled_histogram_channel_images",
    "primary_image_paths_from_manifest",
    "read_czi",
    "subtract_uniform_background",
]
