"""Helper functions extracted from 03_gradient_quantification.ipynb.

This module keeps the notebook focused on analysis flow and outputs while the
reusable geometry, quantification, and diagnostic helpers live in normal Python
code under scripts/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import tifffile
from IPython.display import display
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import ndimage as ndi
from skimage import measure
from skimage.draw import polygon2mask
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Runtime context injected by the notebook. Many plotting / diagnostic helpers
# were originally written against notebook globals, so we keep a narrow explicit
# context setter rather than rewriting the whole analysis object model in one pass.
ROOT = None
PROJECTION = 'max'
EPS = 1e-6
CHANNEL_BG_PARAMS = None
DISTANCE_TRACE_BIN_UM = 10.0
DISTANCE_TRACE_MIN_UM = None
DISTANCE_TRACE_MAX_UM = None
DISTANCE_PLOT_MAX_POINTS = 120000
ORIENTATION_TRACE_BIN_DEG = None
ORIENTATION_SMOOTH_WINDOW_BINS = None
ORIENTATION_NORMALIZATION_PERCENTILE = None
ORIENTATION_MOMENT_BASELINE_PERCENTILE = None
PEAKLIKE_TOP_FRACTION = None
PIXEL_SIZE_UM_FALLBACK = None
SHELL_DEPTH_PX = None
SHELL_MIN_PIXELS = None
PROBLEM_EDGE_BIN_FRACTION = None
PROBLEM_MID_BIN_FRACTION = None
PROBLEM_BACKGROUND_RING_PX = None
DISTANCE_SUPPORT_WINDOW_BINS = None
DISTANCE_SUPPORT_MIN_PIXELS = None
DISTANCE_SUPPORT_EDGE_ONLY = None
MERGED_AXIS_AREA_RATIO = None
MERGED_AXIS_MIN_FRACTION = None
MERGED_AXIS_MIN_OUTER_RADIUS_FRACTION = None
MERGED_AXIS_MIN_PEAK_DISTANCE_PX = None
MERGED_AXIS_MIN_RADIUS_RATIO = None
bead_df = None
summary_df = None
roi_array_center_map = None
FIG_DIR = None
SAVE_FIGURES = False


def set_runtime_context(**kwargs):
    """Update module-level runtime context from notebook variables."""
    globals().update(kwargs)



def get_runtime_context():
    """Return the currently configured runtime context keys used by helpers."""
    keys = [
        'ROOT', 'PROJECTION', 'EPS', 'CHANNEL_BG_PARAMS', 'DISTANCE_TRACE_BIN_UM',
        'DISTANCE_PLOT_MAX_POINTS', 'bead_df', 'summary_df', 'roi_array_center_map',
        'FIG_DIR', 'SAVE_FIGURES'
    ]
    return {k: globals().get(k) for k in keys}



def clear_diag_cache():
    """Reset cached diagnostic image loads."""
    if '_diag_image_cache' in globals():
        _diag_image_cache.clear()


def list_roi_files_from_summary(roi_summary_tsv: Path, project_root: Path) -> list[Path]:
    """Load the canonical ROI JSON list from the 02c summary TSV instead of globbing the directory."""
    roi_summary_df = pd.read_csv(roi_summary_tsv, sep="	")
    required_cols = {"cyst_id", "roi_json_path"}
    missing = required_cols.difference(roi_summary_df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required columns in ROI summary TSV: {sorted(missing)}"
        )

    dup = roi_summary_df.loc[roi_summary_df["cyst_id"].duplicated(keep=False)].copy()
    if len(dup):
        preview = dup.head(20).to_string(index=False)
        raise RuntimeError(
            "Duplicate cyst_id detected in ROI summary TSV. "
            f"Fix summary before quantification.\nTotal duplicate rows: {len(dup)}\nPreview:\n{preview}"
        )

    roi_files = [resolve_path(str(raw), project_root) for raw in roi_summary_df["roi_json_path"]]
    missing_paths = [str(p) for p in roi_files if not p.exists()]
    if missing_paths:
        preview = "\n".join(missing_paths[:20])
        raise RuntimeError(
            "Some ROI JSON paths listed in ROI summary TSV do not exist. "
            f"Missing count: {len(missing_paths)}\nPreview:\n{preview}"
        )

    return sorted(roi_files)


def resolve_path(raw: str, project_root: Path) -> Path:
    p = Path(raw).expanduser()
    candidates = [p] if p.is_absolute() else [project_root / p, p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"Could not resolve path '{raw}'")


def to_2d(array: np.ndarray, projection: str = "max") -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array.astype(np.float32)
    planes = array.reshape((-1, array.shape[-2], array.shape[-1]))
    if projection == "max":
        out = planes.max(axis=0)
    elif projection == "mean":
        out = planes.mean(axis=0)
    elif projection == "first":
        out = planes[0]
    else:
        raise ValueError(f"Unknown projection mode: {projection}")
    return out.astype(np.float32)


def load_image_2d(path: Path, projection: str = "max") -> np.ndarray:
    return to_2d(tifffile.imread(path), projection=projection)


def polygon_xy_to_mask(
    shape_yx: tuple[int, int], polygon_xy: list[list[float]]
) -> np.ndarray:
    if len(polygon_xy) < 3:
        raise ValueError("Polygon needs at least 3 points")
    poly = np.asarray(polygon_xy, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("polygon_xy must have shape (N,2)")
    poly_yx = poly[:, [1, 0]]
    return polygon2mask(shape_yx, poly_yx).astype(bool)


def psmad_over_dapi(
    psmad: np.ndarray, dapi: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    if psmad.shape != dapi.shape:
        raise ValueError(f"Shape mismatch: pSMAD={psmad.shape}, DAPI={dapi.shape}")
    return psmad.astype(np.float32) / (dapi.astype(np.float32) + eps)


def unique_scene_image_paths(scene_df: pd.DataFrame, column: str, project_root: Path) -> list[Path]:
    """Resolve unique scene image paths in dataframe order."""
    seen = set()
    out = []
    for raw in scene_df[column].tolist():
        p = resolve_path(str(raw), project_root)
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def pooled_histogram_all_pixels(
    image_paths: list[Path],
    projection: str = "max",
    n_bins: int = 16384,
) -> dict:
    """Accumulate an exact pooled histogram over all pixels from all images."""
    if len(image_paths) == 0:
        raise ValueError('image_paths is empty')

    global_min = np.inf
    global_max = -np.inf
    n_pixels = 0
    for path in image_paths:
        img = load_image_2d(path, projection=projection)
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            continue
        global_min = min(global_min, float(np.min(finite)))
        global_max = max(global_max, float(np.max(finite)))
        n_pixels += int(finite.size)

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError('Could not determine finite histogram range')
    if global_max <= global_min:
        global_max = global_min + 1.0

    edges = np.linspace(global_min, global_max, int(n_bins) + 1, dtype=np.float64)
    counts = np.zeros(int(n_bins), dtype=np.int64)
    for path in image_paths:
        img = load_image_2d(path, projection=projection)
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            continue
        h, _ = np.histogram(finite, bins=edges)
        counts += h.astype(np.int64)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        'edges': edges,
        'centers': centers,
        'counts': counts,
        'global_min': float(global_min),
        'global_max': float(global_max),
        'n_pixels': int(n_pixels),
        'n_images': int(len(image_paths)),
        'n_bins': int(n_bins),
    }


def estimate_background_half_gaussian(
    hist: dict,
    smooth_sigma_bins: float = 2.0,
) -> dict:
    """
    Estimate background mode and sigma from the pooled histogram.

    Method:
    1. smooth the pooled histogram slightly to stabilize the mode estimate
    2. take the background mode mu_bg as the smoothed histogram maximum
    3. estimate sigma_bg from the left half (x <= mu_bg) using the weighted second moment
    """
    centers = np.asarray(hist['centers'], dtype=np.float64)
    counts = np.asarray(hist['counts'], dtype=np.float64)
    if not np.any(counts > 0):
        raise RuntimeError('Histogram is empty')

    smooth = ndi.gaussian_filter1d(counts, sigma=float(smooth_sigma_bins), mode='nearest')
    mode_idx = int(np.nanargmax(smooth))
    mu_bg = float(centers[mode_idx])

    left = centers <= mu_bg
    left_counts = counts[left]
    left_centers = centers[left]
    if np.sum(left_counts) <= 0:
        raise RuntimeError('No left-half counts available for sigma estimate')

    sigma_bg = float(
        np.sqrt(np.sum(left_counts * (left_centers - mu_bg) ** 2) / np.sum(left_counts))
    )
    sigma_bg = max(float(EPS), sigma_bg)

    amp = float(smooth[mode_idx])
    fit_counts = amp * np.exp(-0.5 * ((centers - mu_bg) / sigma_bg) ** 2)

    return {
        **hist,
        'mu_bg_raw': mu_bg,
        'sigma_bg_raw': sigma_bg,
        'mode_idx': mode_idx,
        'smooth_counts': smooth,
        'fit_counts': fit_counts,
        'fit_method': 'hist_mode_plus_left_half_second_moment',
    }


def normalize_channel_from_background(
    image: np.ndarray, mu_bg_raw: float, sigma_bg_raw: float
) -> np.ndarray:
    """Normalize raw image values to background-SD units and clip below background to zero."""
    sigma = max(float(EPS), float(sigma_bg_raw))
    out = (image.astype(np.float32) - float(mu_bg_raw)) / sigma
    return np.maximum(out, 0.0).astype(np.float32)


def normalize_two_channels_and_ratio(
    psmad_raw: np.ndarray,
    dapi_raw: np.ndarray,
    bg_params: dict,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one dataset-wide background normalization per channel, then take the ratio."""
    psmad_norm = normalize_channel_from_background(
        psmad_raw,
        bg_params['psmad']['mu_bg_raw'],
        bg_params['psmad']['sigma_bg_raw'],
    )
    dapi_norm = normalize_channel_from_background(
        dapi_raw,
        bg_params['dapi']['mu_bg_raw'],
        bg_params['dapi']['sigma_bg_raw'],
    )
    ratio = psmad_over_dapi(psmad_norm, dapi_norm, eps=eps)
    return psmad_norm, dapi_norm, ratio


def signed_angle_deg(v_ref: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Signed angle from v_ref to v in degrees (vectorized for v)."""
    v_ref = np.asarray(v_ref, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if v.ndim == 1:
        v = v[None, :]

    dot = v[:, 0] * v_ref[0] + v[:, 1] * v_ref[1]
    cross = v_ref[0] * v[:, 1] - v_ref[1] * v[:, 0]
    return np.degrees(np.arctan2(cross, dot))


def unit_vector(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0:
        raise ValueError("Zero-length vector")
    return v / n


def extract_boundary_xy(mask: np.ndarray) -> np.ndarray:
    """Return longest contour as Nx2 array in (x,y) float coordinates."""
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    if not contours:
        raise ValueError("No contour found for mask")
    c = max(contours, key=lambda a: a.shape[0])
    # contour is (row=y, col=x)
    return np.column_stack([c[:, 1], c[:, 0]]).astype(np.float64)


def sample_image_at_xy(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = image.shape
    xx = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yy = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return image[yy, xx]


def major_axis_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (centroid_xy, major_axis_unit_xy, boundary_xy).
    Major axis computed by PCA on mask pixel coordinates.
    """
    ys, xs = np.where(mask)
    if len(xs) < 10:
        raise ValueError("Mask too small for axis estimation")

    centroid = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts0 = pts - centroid[None, :]

    cov = np.cov(pts0.T)
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, int(np.argmax(evals))]
    u = unit_vector(u)

    boundary_xy = extract_boundary_xy(mask)
    return centroid, u, boundary_xy


def choose_axis_endpoints(
    boundary_xy: np.ndarray, centroid_xy: np.ndarray, axis_u_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary endpoints (pt_plus, pt_minus) at max/min projection on axis."""
    proj = (boundary_xy - centroid_xy[None, :]) @ axis_u_xy
    pt_plus = boundary_xy[int(np.argmax(proj))]
    pt_minus = boundary_xy[int(np.argmin(proj))]
    return pt_plus, pt_minus


def mask_pixels_xy(mask: np.ndarray) -> np.ndarray:
    """Return all mask pixels as Nx2 float coordinates in (x, y) order."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack([xs, ys]).astype(np.float64)


def wrap_angle_deg(angle_deg: np.ndarray | float) -> np.ndarray:
    """Wrap angles to [-180, 180)."""
    angle_deg = np.asarray(angle_deg, dtype=np.float64)
    return (angle_deg + 180.0) % 360.0 - 180.0


def peripheral_shell_mask(
    mask: np.ndarray,
    band_depth_px: float,
    min_pixels: int = 120,
) -> np.ndarray:
    """
    Return a thin shell just inside the cyst boundary.

    This is the main sampling region for orientation analysis. Using a shell keeps the
    quantification near the epithelium, where BMP is expected to act first, while
    remaining much more robust than a hand-drawn tangent line.
    """
    if float(band_depth_px) <= 0:
        raise ValueError(f"band_depth_px must be > 0, got {band_depth_px}")

    mask = mask.astype(bool)
    if int(mask.sum()) == 0:
        raise ValueError("Mask is empty")

    dist_interior = ndi.distance_transform_edt(mask)
    shell = mask & (dist_interior <= float(band_depth_px))
    if int(shell.sum()) < int(min_pixels):
        shell = mask.copy()
    return shell


def bead_facing_crescent_mask(
    shell_mask: np.ndarray,
    centroid_xy: np.ndarray,
    midline_u_xy: np.ndarray,
    bead_xy: np.ndarray,
    half_angle_deg: float,
    min_pixels: int = 120,
) -> np.ndarray:
    """
    Subset the peripheral shell to the sector that faces the bead.

    This helper is retained only as an alternative for future comparisons. The current
    notebook path does not use it; both distance and orientation use the full shell.
    """
    shell_xy = mask_pixels_xy(shell_mask)
    if len(shell_xy) == 0:
        raise ValueError("Shell mask has zero pixels")

    theta_abs_deg = signed_angle_deg(midline_u_xy, shell_xy - centroid_xy[None, :])
    theta_b_deg = float(signed_angle_deg(midline_u_xy, bead_xy - centroid_xy)[0])
    theta_rel_bead_deg = wrap_angle_deg(theta_abs_deg - theta_b_deg)

    keep = np.abs(theta_rel_bead_deg) <= float(half_angle_deg)
    if int(np.sum(keep)) < int(min_pixels):
        keep = np.ones(len(shell_xy), dtype=bool)

    out = np.zeros_like(shell_mask, dtype=bool)
    out[
        np.round(shell_xy[keep, 1]).astype(int), np.round(shell_xy[keep, 0]).astype(int)
    ] = True
    return out


def top_fraction_mean(values: np.ndarray, frac: float = 0.10) -> float:
    """Mean of the top fraction of finite values; used as a robust peak-like summary."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    frac = float(frac)
    if frac <= 0 or frac > 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    k = max(1, int(np.ceil(values.size * frac)))
    top = np.partition(values, -k)[-k:]
    return float(np.nanmean(top))


def finite_percentile(values: np.ndarray, q: float) -> float:
    """Percentile on finite values only; returns NaN if empty."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.nanpercentile(values, float(q)))


def lower_fraction_mean(values: np.ndarray, order_by: np.ndarray, frac: float = 0.10) -> float:
    """Mean of values in the lowest fraction of an ordering variable."""
    values = np.asarray(values, dtype=np.float64)
    order_by = np.asarray(order_by, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(order_by)
    values = values[keep]
    order_by = order_by[keep]
    if values.size == 0:
        return np.nan
    frac = float(frac)
    if frac <= 0 or frac > 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    k = max(1, int(np.ceil(values.size * frac)))
    idx = np.argsort(order_by)[:k]
    return float(np.nanmean(values[idx]))


def angular_window_mean(values: np.ndarray, theta_rel_deg: np.ndarray, half_angle_deg: float) -> float:
    """Mean of values inside a bead-centered angular window."""
    values = np.asarray(values, dtype=np.float64)
    theta_rel_deg = np.asarray(theta_rel_deg, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(theta_rel_deg)
    values = values[keep]
    theta_rel_deg = theta_rel_deg[keep]
    if values.size == 0:
        return np.nan
    win = np.abs(theta_rel_deg) <= float(half_angle_deg)
    if int(np.sum(win)) == 0:
        return np.nan
    return float(np.nanmean(values[win]))


def trace_table_from_edges(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    center_col: str,
    value_prefix: str,
) -> pd.DataFrame:
    """
    Bin x/y samples onto fixed edges and return one row per bin.

    These are cyst-level traces. They are the correct unit for across-cyst comparisons;
    we do not treat the underlying pixels as independent biological replicates.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a 1D array with at least two values")

    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    centers = 0.5 * (edges[:-1] + edges[1:])
    out = pd.DataFrame(
        {
            center_col: centers,
            f"{value_prefix}_mean": np.nan,
            f"{value_prefix}_median": np.nan,
            f"{value_prefix}_std": np.nan,
            "n_pixels": 0,
        }
    )
    if x.size == 0:
        return out

    idx = np.digitize(x, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges) - 2)

    for i in range(len(centers)):
        sel = idx == i
        if not np.any(sel):
            continue
        yi = y[sel]
        out.loc[i, f"{value_prefix}_mean"] = float(np.nanmean(yi))
        out.loc[i, f"{value_prefix}_median"] = float(np.nanmedian(yi))
        out.loc[i, f"{value_prefix}_std"] = (
            float(np.nanstd(yi, ddof=1)) if int(np.sum(sel)) > 1 else 0.0
        )
        out.loc[i, "n_pixels"] = int(np.sum(sel))
    return out


def xy_bool_to_mask(shape: tuple[int, int], xy: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Rasterize a boolean keep-vector on XY pixel coordinates back to an image mask."""
    out = np.zeros(shape, dtype=bool)
    if xy.size == 0:
        return out
    keep = np.asarray(keep, dtype=bool)
    if keep.size != len(xy):
        raise ValueError("keep vector length must match xy rows")
    xx = np.clip(np.round(xy[keep, 0]).astype(int), 0, shape[1] - 1)
    yy = np.clip(np.round(xy[keep, 1]).astype(int), 0, shape[0] - 1)
    out[yy, xx] = True
    return out


def simple_rolling_mean(values: np.ndarray, window_bins: int) -> np.ndarray:
    """Simple moving average over adjacent occupied bins."""
    values = np.asarray(values, dtype=np.float64)
    if int(window_bins) <= 1:
        return values.copy()
    if int(window_bins) % 2 == 0:
        raise ValueError("window_bins must be odd")

    out = np.full(values.shape, np.nan, dtype=np.float64)
    half = int(window_bins) // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        valid = np.isfinite(values[lo:hi])
        if not np.any(valid):
            continue
        vals = values[lo:hi][valid]
        out[i] = float(np.mean(vals))
    return out


def build_distance_support_trace(
    local_distance_um: np.ndarray,
    dapi_raw: np.ndarray,
    dist_edges: np.ndarray,
    support_window_bins: int,
    support_min_pixels: float,
) -> pd.DataFrame:
    """Build per-distance-bin count support and trim only contiguous low-count edge bins."""
    support_df = trace_table_from_edges(
        local_distance_um,
        dapi_raw,
        dist_edges,
        center_col="distance_local_um",
        value_prefix="dapi_raw",
    )
    support_df["count_support_n_pixels"] = simple_rolling_mean(
        support_df["n_pixels"].to_numpy(dtype=np.float64),
        window_bins=int(support_window_bins),
    )

    occupied = support_df["n_pixels"].to_numpy(dtype=int) > 0
    support = support_df["count_support_n_pixels"].to_numpy(dtype=np.float64)
    keep_bin = occupied.copy()
    trim_reason = np.full(len(support_df), "empty", dtype=object)
    trim_reason[occupied] = "kept"

    valid_bins = np.where(occupied & np.isfinite(support))[0]
    left_trim = []
    right_trim = []

    if DISTANCE_SUPPORT_EDGE_ONLY and len(valid_bins) > 0:
        for i in valid_bins:
            if support[i] < float(support_min_pixels):
                keep_bin[i] = False
                trim_reason[i] = "left_edge_low_count"
                left_trim.append(int(i))
            else:
                break
        for i in valid_bins[::-1]:
            if support[i] < float(support_min_pixels):
                keep_bin[i] = False
                trim_reason[i] = "right_edge_low_count"
                right_trim.append(int(i))
            else:
                break

    if not np.any(keep_bin & occupied):
        keep_bin = occupied.copy()
        trim_reason[occupied] = "kept_fallback"
        left_trim = []
        right_trim = []

    support_df["keep_bin"] = keep_bin
    support_df["trim_reason"] = trim_reason
    support_df["trimmed_left"] = False
    support_df["trimmed_right"] = False
    if left_trim:
        support_df.loc[left_trim, "trimmed_left"] = True
    if right_trim:
        support_df.loc[right_trim, "trimmed_right"] = True
    return support_df


def circular_smooth(values: np.ndarray, window_bins: int = 5) -> np.ndarray:
    """Circular moving-average smoothing for angular traces."""
    values = np.asarray(values, dtype=np.float64)
    if int(window_bins) <= 1:
        return values.copy()
    if int(window_bins) % 2 == 0:
        raise ValueError("window_bins must be odd for symmetric smoothing")

    pad = int(window_bins) // 2
    data = np.nan_to_num(values, nan=0.0)
    valid = np.isfinite(values).astype(np.float64)

    data_pad = np.r_[data[-pad:], data, data[:pad]]
    valid_pad = np.r_[valid[-pad:], valid, valid[:pad]]
    kernel = np.ones(int(window_bins), dtype=np.float64)

    num = np.convolve(data_pad, kernel, mode="same")
    den = np.convolve(valid_pad, kernel, mode="same")
    out = num[pad:-pad]
    den = den[pad:-pad]

    smoothed = np.full_like(out, np.nan, dtype=np.float64)
    ok = den > 0
    smoothed[ok] = out[ok] / den[ok]
    return smoothed


def circular_first_moment_deg(
    theta_deg: np.ndarray,
    values: np.ndarray,
    baseline_percentile: float = 20.0,
) -> tuple[float, float]:
    """
    Return the direction and strength of the shell signal using a circular first moment.

    We subtract a low percentile baseline first so the vector reflects the polarized part
    of the signal rather than the roughly uniform background around the shell.
    """
    theta_deg = np.asarray(theta_deg, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(theta_deg) & np.isfinite(values)
    theta_deg = theta_deg[keep]
    values = values[keep]
    if theta_deg.size == 0:
        return np.nan, np.nan

    baseline = float(np.nanpercentile(values, baseline_percentile))
    weights = np.clip(values - baseline, 0.0, None)
    if float(np.nansum(weights)) <= 0:
        weights = np.clip(values, 0.0, None)
    if float(np.nansum(weights)) <= 0:
        return np.nan, np.nan

    ang = np.deg2rad(theta_deg)
    vx = float(np.sum(weights * np.cos(ang)))
    vy = float(np.sum(weights * np.sin(ang)))
    mag = float(np.hypot(vx, vy))
    if mag <= 0:
        return np.nan, np.nan
    theta = float(np.degrees(np.arctan2(vy, vx)))
    strength = float(mag / np.sum(weights))
    return theta, strength


def aggregate_trace_across_cysts(
    trace_df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    """Aggregate one trace point per cyst/bin into a mean ± SD/SEM table."""
    if len(trace_df) == 0:
        return pd.DataFrame(columns=[x_col, "mean", "sd", "sem", "n_cysts"])

    df = trace_df[np.isfinite(trace_df[x_col]) & np.isfinite(trace_df[y_col])].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=[x_col, "mean", "sd", "sem", "n_cysts"])

    rows = []
    for xval, sub in df.groupby(x_col, sort=True):
        vals = sub[y_col].to_numpy(dtype=float)
        n = len(vals)
        sd = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
        sem = float(sd / np.sqrt(n)) if n > 1 else 0.0
        rows.append(
            {
                x_col: float(xval),
                "mean": float(np.nanmean(vals)),
                "sd": sd,
                "sem": sem,
                "n_cysts": int(n),
            }
        )
    return pd.DataFrame(rows).sort_values(x_col).reset_index(drop=True)


def binned_mean_sem(x: np.ndarray, y: np.ndarray, bin_width: float) -> pd.DataFrame:
    """Return binned mean and SEM table for x/y points (used in diagnostics only)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return pd.DataFrame(columns=["x_center", "mean", "sem", "n"])

    xmin, xmax = float(np.min(x)), float(np.max(x))
    edges = np.arange(xmin, xmax + bin_width, bin_width)
    if edges.size < 2:
        edges = np.array([xmin - 0.5 * bin_width, xmax + 0.5 * bin_width])

    idx = np.digitize(x, edges) - 1
    rows = []
    for i in range(edges.size - 1):
        sel = idx == i
        if not np.any(sel):
            continue
        yi = y[sel]
        n = int(np.sum(sel))
        m = float(np.nanmean(yi))
        s = float(np.nanstd(yi, ddof=1)) if n > 1 else 0.0
        sem = float(s / np.sqrt(n)) if n > 1 else 0.0
        rows.append(
            {
                "x_center": float(0.5 * (edges[i] + edges[i + 1])),
                "mean": m,
                "sem": sem,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def linear_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for finite x/y with n>=2."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, deg=1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
    return float(slope), float(intercept), float(r2)


def check_unique_cyst_ids_or_fail(roi_files: list[Path]) -> pd.DataFrame:
    rows = []
    for rf in roi_files:
        rec = json.loads(rf.read_text(encoding="utf-8"))
        cid = str(rec.get("cyst_id", rf.stem))
        rows.append({"roi_file": str(rf), "cyst_id": cid})
    idx_df = pd.DataFrame(rows)

    dup_mask = idx_df.duplicated("cyst_id", keep=False)
    if dup_mask.any():
        dup = idx_df.loc[dup_mask].sort_values("cyst_id").reset_index(drop=True)
        preview = dup.head(20).to_string(index=False)
        msg = (
            "Duplicate cyst_id detected in ROI files. "
            "Fix duplicates before quantification to avoid double-counting."
            + "\nTotal duplicate rows: {}\nPreview:\n{}".format(len(dup), preview)
        )
        raise RuntimeError(msg)
    return idx_df


def measure_bead_present_arrays(
    record: dict,
    roi_file: Path,
    psmad_raw: np.ndarray,
    dapi_raw: np.ndarray,
    psmad_norm: np.ndarray,
    dapi_norm: np.ndarray,
    array_center_xy: np.ndarray,
) -> dict:
    """
    Measure one bead-present cyst.

    The key objects returned here drive the entire notebook and are intentionally
    independent of the distance-support trim floor so they can be re-summarized
    cheaply if that floor changes later in the workflow:
    1. a full peripheral shell around the cyst
    2. local distance and angular coordinates defined on that same shell
    """
    cyst_id = str(record.get("cyst_id", roi_file.stem))

    if not (psmad_raw.shape == dapi_raw.shape == psmad_norm.shape == dapi_norm.shape):
        raise ValueError(
            f"Shape mismatch for {cyst_id}: raw pSMAD={psmad_raw.shape}, raw DAPI={dapi_raw.shape}, "
            f"norm pSMAD={psmad_norm.shape}, norm DAPI={dapi_norm.shape}"
        )

    array_center_xy = np.asarray(array_center_xy, dtype=np.float64)
    if array_center_xy.shape != (2,) or not np.all(np.isfinite(array_center_xy)):
        raise RuntimeError(
            f"Invalid array-center coordinates for {cyst_id}: {array_center_xy}"
        )

    polygon_xy = record["polygon_xy_px"]
    bead_xy = np.asarray(record["bead_xy_px"], dtype=np.float64)
    pixel_um = float(record.get("pixel_size_um", PIXEL_SIZE_UM_FALLBACK))

    ratio = psmad_over_dapi(psmad_norm, dapi_norm, eps=EPS)
    mask = polygon_xy_to_mask(psmad_raw.shape, polygon_xy)
    if int(mask.sum()) == 0:
        raise ValueError(f"ROI mask has zero area: {cyst_id}")

    geom = geometry_from_mask_and_array_center(mask, array_center_xy=array_center_xy)
    if not bool(geom["ap_defined_from_array_center"]):
        raise RuntimeError(f"Array-center AP assignment failed for {cyst_id}")

    centroid_xy = geom["centroid_xy"]
    boundary_xy = geom["boundary_xy"]
    anterior_xy = geom["anterior_xy"]
    posterior_xy = geom["posterior_xy"]
    midline_u = geom["midline_u"]
    tangent_u = geom["tangent_u"]

    boundary_dist_px = np.linalg.norm(boundary_xy - bead_xy[None, :], axis=1)
    i_nearest = int(np.argmin(boundary_dist_px))
    nearest_boundary_xy = boundary_xy[i_nearest]
    distance_d_um = float(boundary_dist_px[i_nearest] * pixel_um)
    distance_centroid_bead_um = float(np.linalg.norm(bead_xy - centroid_xy) * pixel_um)

    # Full shell for orientation.
    shell_mask = peripheral_shell_mask(
        mask,
        band_depth_px=SHELL_DEPTH_PX,
        min_pixels=SHELL_MIN_PIXELS,
    )
    shell_xy = mask_pixels_xy(shell_mask)
    if len(shell_xy) == 0:
        raise RuntimeError(f"Shell sampling produced zero pixels for {cyst_id}")

    ratio_shell = sample_image_at_xy(ratio, shell_xy)
    dapi_shell = sample_image_at_xy(dapi_raw, shell_xy)
    psmad_shell = sample_image_at_xy(psmad_raw, shell_xy)
    dapi_shell_norm = sample_image_at_xy(dapi_norm, shell_xy)
    psmad_shell_norm = sample_image_at_xy(psmad_norm, shell_xy)
    theta_abs_shell_deg = signed_angle_deg(midline_u, shell_xy - centroid_xy[None, :])
    theta_b_deg = float(signed_angle_deg(midline_u, bead_xy - centroid_xy)[0])
    theta_rel_bead_shell_deg = wrap_angle_deg(theta_abs_shell_deg - theta_b_deg)
    local_distance_shell_um = (
        np.linalg.norm(shell_xy - bead_xy[None, :], axis=1) * pixel_um
    )
    return {
        "cyst_id": cyst_id,
        "source_record": record,
        "roi_file_path": str(roi_file),
        "pixel_um": pixel_um,
        "ratio": ratio,
        "mask": mask,
        "shell_mask": shell_mask,
        "boundary_xy": boundary_xy,
        "shell_xy": shell_xy,
        "centroid_xy": centroid_xy,
        "array_center_xy": array_center_xy,
        "anterior_xy": anterior_xy,
        "posterior_xy": posterior_xy,
        "nearest_boundary_xy": nearest_boundary_xy,
        "bead_xy": bead_xy,
        "midline_u": midline_u,
        "tangent_u": tangent_u,
        "ratio_shell": ratio_shell,
        "dapi_shell": dapi_shell,
        "psmad_shell": psmad_shell,
        "dapi_shell_norm": dapi_shell_norm,
        "psmad_shell_norm": psmad_shell_norm,
        "theta_abs_shell_deg": theta_abs_shell_deg,
        "theta_rel_bead_shell_deg": theta_rel_bead_shell_deg,
        "local_distance_shell_um": local_distance_shell_um,
        "distance_centroid_bead_um": distance_centroid_bead_um,
        "distance_d_um": distance_d_um,
        "theta_b_deg": theta_b_deg,
    }


def summarize_bead_present_measurement(measurement: dict) -> dict:
    """Derive trim-dependent summaries/traces from a cached bead-present measurement."""
    record = measurement["source_record"]
    roi_file = Path(str(measurement["roi_file_path"]))
    cyst_id = str(measurement["cyst_id"])
    pixel_um = float(measurement["pixel_um"])

    ratio_shell = np.asarray(measurement["ratio_shell"], dtype=np.float64)
    dapi_shell = np.asarray(measurement["dapi_shell"], dtype=np.float64)
    psmad_shell = np.asarray(measurement["psmad_shell"], dtype=np.float64)
    dapi_shell_norm = np.asarray(measurement["dapi_shell_norm"], dtype=np.float64)
    psmad_shell_norm = np.asarray(measurement["psmad_shell_norm"], dtype=np.float64)
    shell_xy = np.asarray(measurement["shell_xy"], dtype=np.float64)
    theta_abs_shell_deg = np.asarray(measurement["theta_abs_shell_deg"], dtype=np.float64)
    theta_rel_bead_shell_deg = np.asarray(measurement["theta_rel_bead_shell_deg"], dtype=np.float64)
    local_distance_shell_um = np.asarray(measurement["local_distance_shell_um"], dtype=np.float64)

    mask = np.asarray(measurement["mask"], dtype=bool)
    bead_xy = np.asarray(measurement["bead_xy"], dtype=np.float64)
    centroid_xy = np.asarray(measurement["centroid_xy"], dtype=np.float64)
    array_center_xy = np.asarray(measurement["array_center_xy"], dtype=np.float64)
    anterior_xy = np.asarray(measurement["anterior_xy"], dtype=np.float64)
    posterior_xy = np.asarray(measurement["posterior_xy"], dtype=np.float64)
    nearest_boundary_xy = np.asarray(measurement["nearest_boundary_xy"], dtype=np.float64)
    midline_u = np.asarray(measurement["midline_u"], dtype=np.float64)
    theta_b_deg = float(measurement["theta_b_deg"])

    dist_max = max(
        DISTANCE_TRACE_MAX_UM,
        float(np.nanmax(local_distance_shell_um) + DISTANCE_TRACE_BIN_UM),
    )
    dist_edges = np.arange(
        DISTANCE_TRACE_MIN_UM,
        dist_max + DISTANCE_TRACE_BIN_UM,
        DISTANCE_TRACE_BIN_UM,
    )
    distance_support_df = build_distance_support_trace(
        local_distance_shell_um,
        dapi_shell,
        dist_edges,
        support_window_bins=DISTANCE_SUPPORT_WINDOW_BINS,
        support_min_pixels=DISTANCE_SUPPORT_MIN_PIXELS,
    )
    pixel_bin_idx = np.digitize(local_distance_shell_um, dist_edges, right=False) - 1
    pixel_bin_idx = np.clip(pixel_bin_idx, 0, len(dist_edges) - 2)
    keep_bin_lookup = distance_support_df["keep_bin"].to_numpy(dtype=bool)
    keep_distance_pixel = keep_bin_lookup[pixel_bin_idx]
    if not np.any(keep_distance_pixel):
        keep_distance_pixel = np.ones(len(shell_xy), dtype=bool)

    distance_mask = xy_bool_to_mask(mask.shape, shell_xy, keep_distance_pixel)
    distance_xy = shell_xy[keep_distance_pixel]
    distance_trimmed_xy = shell_xy[~keep_distance_pixel]
    ratio_distance = ratio_shell[keep_distance_pixel]
    dapi_distance = dapi_shell[keep_distance_pixel]
    psmad_distance = psmad_shell[keep_distance_pixel]
    dapi_distance_norm = dapi_shell_norm[keep_distance_pixel]
    psmad_distance_norm = psmad_shell_norm[keep_distance_pixel]
    theta_abs_distance_deg = theta_abs_shell_deg[keep_distance_pixel]
    theta_rel_bead_distance_deg = theta_rel_bead_shell_deg[keep_distance_pixel]
    local_distance_distance_um = local_distance_shell_um[keep_distance_pixel]

    norm_denom = float(
        np.nanpercentile(ratio_shell, ORIENTATION_NORMALIZATION_PERCENTILE)
    )
    if not np.isfinite(norm_denom) or norm_denom <= EPS:
        norm_denom = float(np.nanmax(ratio_shell))
    norm_denom = max(EPS, norm_denom)
    ratio_norm_shell = ratio_shell / norm_denom
    ratio_norm_distance = ratio_distance / norm_denom

    distance_trace_df = trace_table_from_edges(
        local_distance_distance_um,
        ratio_distance,
        dist_edges,
        center_col="distance_local_um",
        value_prefix="ratio",
    )
    distance_trace_df = distance_trace_df.merge(
        distance_support_df[
            [
                "distance_local_um",
                "dapi_raw_mean",
                "count_support_n_pixels",
                "keep_bin",
                "trim_reason",
            ]
        ],
        on="distance_local_um",
        how="left",
    )
    distance_trace_df = distance_trace_df.loc[
        distance_trace_df["n_pixels"] > 0
    ].reset_index(drop=True)

    theta_edges = np.arange(
        -180.0, 180.0 + ORIENTATION_TRACE_BIN_DEG, ORIENTATION_TRACE_BIN_DEG
    )
    theta_abs_trace_df = trace_table_from_edges(
        theta_abs_shell_deg,
        ratio_shell,
        theta_edges,
        center_col="theta_abs_deg",
        value_prefix="ratio",
    )
    theta_rel_trace_df = trace_table_from_edges(
        theta_rel_bead_shell_deg,
        ratio_norm_shell,
        theta_edges,
        center_col="theta_rel_bead_deg",
        value_prefix="ratio_norm",
    )
    theta_rel_trace_df = theta_rel_trace_df.loc[
        theta_rel_trace_df["n_pixels"] > 0
    ].reset_index(drop=True)

    # Save the stored per-cyst direction summaries used downstream.
    # theta_moment_deg is currently the selected shell-pixel first-moment
    # definition used by the final Orientation Plot 2, while theta_peak_deg is
    # still used by some QC mismatch views.
    theta_abs_centers = theta_abs_trace_df["theta_abs_deg"].to_numpy(dtype=float)
    theta_abs_mean = theta_abs_trace_df["ratio_mean"].to_numpy(dtype=float)
    theta_abs_smooth = circular_smooth(
        theta_abs_mean,
        window_bins=ORIENTATION_SMOOTH_WINDOW_BINS,
    )
    if np.any(np.isfinite(theta_abs_smooth)):
        theta_peak_deg = float(theta_abs_centers[int(np.nanargmax(theta_abs_smooth))])
    else:
        theta_peak_deg = float(theta_abs_shell_deg[int(np.nanargmax(ratio_shell))])
    theta_peak_rel_bead_deg = float(wrap_angle_deg(theta_peak_deg - theta_b_deg))

    theta_moment_deg, theta_moment_strength = circular_first_moment_deg(
        theta_abs_shell_deg,
        ratio_shell,
        baseline_percentile=ORIENTATION_MOMENT_BASELINE_PERCENTILE,
    )
    theta_moment_rel_bead_deg = (
        float(wrap_angle_deg(theta_moment_deg - theta_b_deg))
        if np.isfinite(theta_moment_deg)
        else np.nan
    )

    ratio_peak = top_fraction_mean(ratio_distance, frac=PEAKLIKE_TOP_FRACTION)
    ratio_peak_top01 = top_fraction_mean(ratio_distance, frac=0.01)
    ratio_peak_top05 = top_fraction_mean(ratio_distance, frac=0.05)
    ratio_peak_top20 = top_fraction_mean(ratio_distance, frac=0.20)
    ratio_peak_top30 = top_fraction_mean(ratio_distance, frac=0.30)
    ratio_p99 = finite_percentile(ratio_distance, 99.0)
    ratio_p95 = finite_percentile(ratio_distance, 95.0)
    ratio_p90 = finite_percentile(ratio_distance, 90.0)
    ratio_raw_max = float(np.nanmax(ratio_distance))
    ratio_median_distance_region = float(np.nanmedian(ratio_distance))
    ratio_near_q05_mean = lower_fraction_mean(
        ratio_distance, local_distance_distance_um, frac=0.05
    )
    ratio_near_q10_mean = lower_fraction_mean(
        ratio_distance, local_distance_distance_um, frac=0.10
    )
    ratio_near_q20_mean = lower_fraction_mean(
        ratio_distance, local_distance_distance_um, frac=0.20
    )
    ratio_near_q30_mean = lower_fraction_mean(
        ratio_distance, local_distance_distance_um, frac=0.30
    )
    ratio_bead_crescent_30_mean = angular_window_mean(
        ratio_distance, theta_rel_bead_distance_deg, half_angle_deg=30.0
    )
    ratio_bead_crescent_45_mean = angular_window_mean(
        ratio_distance, theta_rel_bead_distance_deg, half_angle_deg=45.0
    )
    ratio_bead_crescent_60_mean = angular_window_mean(
        ratio_distance, theta_rel_bead_distance_deg, half_angle_deg=60.0
    )
    ratio_bead_crescent_90_mean = angular_window_mean(
        ratio_distance, theta_rel_bead_distance_deg, half_angle_deg=90.0
    )

    summary = {
        "cyst_id": cyst_id,
        "roi_file": str(roi_file),
        "scene": record.get("scene", ""),
        "scene_index": int(record.get("scene_index", -1)),
        "source_file": record.get("source_file", ""),
        "source_mode": "roi_json_bead_present",
        "bead_present": True,
        "distance_mode": "bead_center_to_nearest_surface",
        "angle_mode": "centroid_bead_vector_vs_array_center_ap_axis",
        "pixel_size_um": pixel_um,
        "psmad_image_path": str(record.get("psmad_image_path", "")),
        "dapi_image_path": str(record.get("dapi_image_path", "")),
        "array_center_x_px": float(array_center_xy[0]),
        "array_center_y_px": float(array_center_xy[1]),
        "ap_defined_from_array_center": True,
        "sample_region_distance": "full_peripheral_shell",
        "sample_region_orientation": "full_peripheral_shell",
        "shell_depth_px": float(SHELL_DEPTH_PX),
        "distance_crescent_half_angle_deg": np.nan,
        "shell_n_px": int(shell_xy.shape[0]),
        "distance_region_n_px": int(distance_xy.shape[0]),
        "bead_x_px": float(bead_xy[0]),
        "bead_y_px": float(bead_xy[1]),
        "centroid_x_px": float(centroid_xy[0]),
        "centroid_y_px": float(centroid_xy[1]),
        "posterior_x_px": float(posterior_xy[0]),
        "posterior_y_px": float(posterior_xy[1]),
        "anterior_x_px": float(anterior_xy[0]),
        "anterior_y_px": float(anterior_xy[1]),
        "nearest_surface_x_px": float(nearest_boundary_xy[0]),
        "nearest_surface_y_px": float(nearest_boundary_xy[1]),
        "distance_centroid_bead_um": float(measurement["distance_centroid_bead_um"]),
        "distance_d_um": float(measurement["distance_d_um"]),
        "theta_b_deg": theta_b_deg,
        "theta_peak_deg": theta_peak_deg,
        "theta_peak_rel_bead_deg": theta_peak_rel_bead_deg,
        "theta_moment_deg": float(theta_moment_deg),
        "theta_moment_rel_bead_deg": theta_moment_rel_bead_deg,
        "theta_moment_strength": float(theta_moment_strength),
        "roi_area_px": int(mask.sum()),
        "ratio_peak": float(ratio_peak),
        "ratio_peak_top01": float(ratio_peak_top01),
        "ratio_peak_top05": float(ratio_peak_top05),
        "ratio_peak_top20": float(ratio_peak_top20),
        "ratio_peak_top30": float(ratio_peak_top30),
        "ratio_p99": float(ratio_p99),
        "ratio_p95": float(ratio_p95),
        "ratio_p90": float(ratio_p90),
        "ratio_raw_max": ratio_raw_max,
        "ratio_mean_shell": float(np.nanmean(ratio_shell)),
        "ratio_median_shell": float(np.nanmedian(ratio_shell)),
        "ratio_mean_distance_region": float(np.nanmean(ratio_distance)),
        "ratio_median_distance_region": ratio_median_distance_region,
        "ratio_near_q05_mean": float(ratio_near_q05_mean),
        "ratio_near_q10_mean": float(ratio_near_q10_mean),
        "ratio_near_q20_mean": float(ratio_near_q20_mean),
        "ratio_near_q30_mean": float(ratio_near_q30_mean),
        "ratio_bead_crescent_30_mean": float(ratio_bead_crescent_30_mean),
        "ratio_bead_crescent_45_mean": float(ratio_bead_crescent_45_mean),
        "ratio_bead_crescent_60_mean": float(ratio_bead_crescent_60_mean),
        "ratio_bead_crescent_90_mean": float(ratio_bead_crescent_90_mean),
        "ratio_norm_denom": float(norm_denom),
        "psmad_bg_mu_raw": float(CHANNEL_BG_PARAMS["psmad"]["mu_bg_raw"]),
        "psmad_bg_sigma_raw": float(CHANNEL_BG_PARAMS["psmad"]["sigma_bg_raw"]),
        "dapi_bg_mu_raw": float(CHANNEL_BG_PARAMS["dapi"]["mu_bg_raw"]),
        "dapi_bg_sigma_raw": float(CHANNEL_BG_PARAMS["dapi"]["sigma_bg_raw"]),
        "distance_support_min_pixels": float(DISTANCE_SUPPORT_MIN_PIXELS),
        "distance_support_window_bins": int(DISTANCE_SUPPORT_WINDOW_BINS),
        "n_distance_bins_kept": int(np.sum(distance_support_df["keep_bin"].to_numpy(dtype=bool))),
        "n_distance_bins_trimmed_left": int(np.sum(distance_support_df["trimmed_left"].to_numpy(dtype=bool))),
        "n_distance_bins_trimmed_right": int(np.sum(distance_support_df["trimmed_right"].to_numpy(dtype=bool))),
    }

    distance_trace_rows = []
    for _, tr in distance_trace_df.iterrows():
        distance_trace_rows.append(
            {
                "cyst_id": cyst_id,
                "scene_index": summary["scene_index"],
                "scene": summary["scene"],
                "bead_present": True,
                "distance_local_um": float(tr["distance_local_um"]),
                "ratio_mean": float(tr["ratio_mean"]),
                "ratio_median": float(tr["ratio_median"]),
                "ratio_std": float(tr["ratio_std"]),
                "n_pixels": int(tr["n_pixels"]),
                "dapi_raw_mean": float(tr["dapi_raw_mean"]),
                "count_support_n_pixels": float(tr["count_support_n_pixels"]),
                "distance_bin_kept": bool(tr["keep_bin"]),
                "trim_reason": str(tr["trim_reason"]),
            }
        )

    orientation_trace_rows = []
    for _, tr in theta_rel_trace_df.iterrows():
        orientation_trace_rows.append(
            {
                "cyst_id": cyst_id,
                "scene_index": summary["scene_index"],
                "scene": summary["scene"],
                "bead_present": True,
                "theta_rel_bead_deg": float(tr["theta_rel_bead_deg"]),
                "ratio_norm_mean": float(tr["ratio_norm_mean"]),
                "ratio_norm_median": float(tr["ratio_norm_median"]),
                "ratio_norm_std": float(tr["ratio_norm_std"]),
                "n_pixels": int(tr["n_pixels"]),
            }
        )

    return {
        "summary": summary,
        "distance_trace_rows": distance_trace_rows,
        "orientation_trace_rows": orientation_trace_rows,
        "distance_mask": distance_mask,
        "distance_xy": distance_xy,
        "distance_trimmed_xy": distance_trimmed_xy,
        "ratio_distance": ratio_distance,
        "ratio_norm_shell": ratio_norm_shell,
        "ratio_norm_distance": ratio_norm_distance,
        "dapi_distance": dapi_distance,
        "psmad_distance": psmad_distance,
        "dapi_distance_norm": dapi_distance_norm,
        "psmad_distance_norm": psmad_distance_norm,
        "theta_abs_distance_deg": theta_abs_distance_deg,
        "theta_rel_bead_distance_deg": theta_rel_bead_distance_deg,
        "local_distance_distance_um": local_distance_distance_um,
        "distance_support_df": distance_support_df,
        "keep_distance_pixel": keep_distance_pixel,
        "theta_abs_trace_df": theta_abs_trace_df,
        "theta_abs_trace_smooth": theta_abs_smooth,
        "theta_rel_trace_df": theta_rel_trace_df,
    }


def quantify_bead_present_cyst(
    record: dict,
    roi_file: Path,
    project_root: Path,
    image_cache: dict[tuple[str, str], np.ndarray],
    array_center_xy: np.ndarray | None,
) -> tuple[dict, list[dict], list[dict]]:
    """Load images for one ROI, then call the shell/crescent quantifier above."""
    summary, distance_rows, orientation_rows, _ = quantify_bead_present_cyst_with_measurement(
        record=record,
        roi_file=roi_file,
        project_root=project_root,
        image_cache=image_cache,
        array_center_xy=array_center_xy,
    )
    return summary, distance_rows, orientation_rows


def quantify_bead_present_cyst_with_measurement(
    record: dict,
    roi_file: Path,
    project_root: Path,
    image_cache: dict[tuple[str, str], np.ndarray],
    array_center_xy: np.ndarray | None,
    cached_measurement: dict | None = None,
) -> tuple[dict, list[dict], list[dict], dict]:
    """Load or reuse one bead-present measurement, then derive the current summaries."""
    cyst_id = str(record.get("cyst_id", roi_file.stem))

    if array_center_xy is None:
        raise RuntimeError(f"Missing array-center coordinates for {cyst_id}")

    if cached_measurement is None:
        psmad_path = resolve_path(
            str(record["psmad_image_path"]), project_root=project_root
        )
        dapi_path = resolve_path(str(record["dapi_image_path"]), project_root=project_root)

        p_key = (str(psmad_path), PROJECTION, 'raw')
        d_key = (str(dapi_path), PROJECTION, 'raw')
        pn_key = (str(psmad_path), PROJECTION, 'bg_norm')
        dn_key = (str(dapi_path), PROJECTION, 'bg_norm')
        if p_key not in image_cache:
            image_cache[p_key] = load_image_2d(psmad_path, projection=PROJECTION)
        if d_key not in image_cache:
            image_cache[d_key] = load_image_2d(dapi_path, projection=PROJECTION)
        if pn_key not in image_cache:
            image_cache[pn_key] = normalize_channel_from_background(
                image_cache[p_key],
                CHANNEL_BG_PARAMS['psmad']['mu_bg_raw'],
                CHANNEL_BG_PARAMS['psmad']['sigma_bg_raw'],
            )
        if dn_key not in image_cache:
            image_cache[dn_key] = normalize_channel_from_background(
                image_cache[d_key],
                CHANNEL_BG_PARAMS['dapi']['mu_bg_raw'],
                CHANNEL_BG_PARAMS['dapi']['sigma_bg_raw'],
            )

        measurement = measure_bead_present_arrays(
            record=record,
            roi_file=roi_file,
            psmad_raw=image_cache[p_key],
            dapi_raw=image_cache[d_key],
            psmad_norm=image_cache[pn_key],
            dapi_norm=image_cache[dn_key],
            array_center_xy=np.asarray(array_center_xy, dtype=np.float64),
        )
    else:
        measurement = cached_measurement

    measurement.update(summarize_bead_present_measurement(measurement))
    return (
        measurement["summary"],
        measurement["distance_trace_rows"],
        measurement["orientation_trace_rows"],
        measurement,
    )


def quantify_no_bead_control_scene(
    scene_row: pd.Series,
    project_root: Path,
    image_cache: dict[tuple[str, str], np.ndarray],
    excluded_label_id: int | None = None,
) -> list[dict]:
    """
    Quantify no-bead controls.

    Controls do not contribute to the orientation analysis, but they do provide a useful
    far-right reference for the distance/amplitude plot.
    """
    dapi_path = resolve_path(str(scene_row["dapi_image_path"]), project_root)
    psmad_path = resolve_path(str(scene_row["psmad_image_path"]), project_root)
    labels_path = resolve_path(str(scene_row["cyst_labels_path"]), project_root)

    p_key = (str(psmad_path), PROJECTION, 'raw')
    d_key = (str(dapi_path), PROJECTION, 'raw')
    pn_key = (str(psmad_path), PROJECTION, 'bg_norm')
    dn_key = (str(dapi_path), PROJECTION, 'bg_norm')
    if p_key not in image_cache:
        image_cache[p_key] = load_image_2d(psmad_path, projection=PROJECTION)
    if d_key not in image_cache:
        image_cache[d_key] = load_image_2d(dapi_path, projection=PROJECTION)
    if pn_key not in image_cache:
        image_cache[pn_key] = normalize_channel_from_background(
            image_cache[p_key],
            CHANNEL_BG_PARAMS['psmad']['mu_bg_raw'],
            CHANNEL_BG_PARAMS['psmad']['sigma_bg_raw'],
        )
    if dn_key not in image_cache:
        image_cache[dn_key] = normalize_channel_from_background(
            image_cache[d_key],
            CHANNEL_BG_PARAMS['dapi']['mu_bg_raw'],
            CHANNEL_BG_PARAMS['dapi']['sigma_bg_raw'],
        )

    psmad_raw = image_cache[p_key]
    dapi_raw = image_cache[d_key]
    psmad = image_cache[pn_key]
    dapi = image_cache[dn_key]
    labels = tifffile.imread(labels_path).astype(np.int32)

    if psmad.shape != dapi.shape or psmad.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch scene {scene_row['scene_index']}: pSMAD={psmad.shape}, DAPI={dapi.shape}, labels={labels.shape}"
        )

    ratio = psmad_over_dapi(psmad, dapi, eps=EPS)
    pixel_um = float(scene_row.get("pixel_size_um", PIXEL_SIZE_UM_FALLBACK))

    out = []
    props = measure.regionprops(labels)
    for i, region in enumerate(props, start=1):
        if excluded_label_id is not None and int(region.label) == int(
            excluded_label_id
        ):
            continue
        mask = labels == int(region.label)
        if int(mask.sum()) == 0:
            continue

        shell_mask = peripheral_shell_mask(
            mask,
            band_depth_px=SHELL_DEPTH_PX,
            min_pixels=SHELL_MIN_PIXELS,
        )
        shell_xy = mask_pixels_xy(shell_mask)
        if len(shell_xy) == 0:
            continue

        ratio_shell = sample_image_at_xy(ratio, shell_xy)
        dapi_shell_raw = sample_image_at_xy(dapi_raw, shell_xy)
        psmad_shell_raw = sample_image_at_xy(psmad_raw, shell_xy)
        ys, xs = np.where(mask)
        cyst_id = f"manual_s{int(scene_row['scene_index']):02d}_cyst_{i:04d}"

        ratio_peak_top01 = top_fraction_mean(ratio_shell, frac=0.01)
        ratio_peak_top05 = top_fraction_mean(ratio_shell, frac=0.05)
        ratio_peak_top10 = top_fraction_mean(ratio_shell, frac=PEAKLIKE_TOP_FRACTION)
        ratio_peak_top20 = top_fraction_mean(ratio_shell, frac=0.20)
        ratio_peak_top30 = top_fraction_mean(ratio_shell, frac=0.30)
        ratio_p99 = finite_percentile(ratio_shell, 99.0)
        ratio_p95 = finite_percentile(ratio_shell, 95.0)
        ratio_p90 = finite_percentile(ratio_shell, 90.0)

        out.append(
            {
                "cyst_id": cyst_id,
                "roi_file": f"<from_label_scene_{int(scene_row['scene_index']):02d}>",
                "scene": scene_row.get("scene", ""),
                "scene_index": int(scene_row["scene_index"]),
                "source_file": scene_row.get("source_file", ""),
                "source_mode": "label_mask_no_bead_scene",
                "bead_present": False,
                "distance_mode": "no_bead_control",
                "angle_mode": "no_bead_control",
                "pixel_size_um": pixel_um,
                "psmad_image_path": str(psmad_path),
                "dapi_image_path": str(dapi_path),
                "array_center_x_px": np.nan,
                "array_center_y_px": np.nan,
                "ap_defined_from_array_center": np.nan,
                "sample_region_distance": "peripheral_shell_control",
                "sample_region_orientation": "peripheral_shell_control",
                "shell_depth_px": float(SHELL_DEPTH_PX),
                "distance_crescent_half_angle_deg": np.nan,
                "shell_n_px": int(shell_xy.shape[0]),
                "distance_region_n_px": int(shell_xy.shape[0]),
                "bead_x_px": np.nan,
                "bead_y_px": np.nan,
                "centroid_x_px": float(np.mean(xs)),
                "centroid_y_px": float(np.mean(ys)),
                "posterior_x_px": np.nan,
                "posterior_y_px": np.nan,
                "anterior_x_px": np.nan,
                "anterior_y_px": np.nan,
                "nearest_surface_x_px": np.nan,
                "nearest_surface_y_px": np.nan,
                "distance_centroid_bead_um": np.nan,
                "distance_d_um": np.nan,
                "theta_b_deg": np.nan,
                "theta_peak_deg": np.nan,
                "theta_peak_rel_bead_deg": np.nan,
                "theta_moment_deg": np.nan,
                "theta_moment_rel_bead_deg": np.nan,
                "theta_moment_strength": np.nan,
                "roi_area_px": int(mask.sum()),
                "ratio_peak": ratio_peak_top10,
                "ratio_peak_top01": ratio_peak_top01,
                "ratio_peak_top05": ratio_peak_top05,
                "ratio_peak_top20": ratio_peak_top20,
                "ratio_peak_top30": ratio_peak_top30,
                "ratio_p99": ratio_p99,
                "ratio_p95": ratio_p95,
                "ratio_p90": ratio_p90,
                "ratio_raw_max": float(np.nanmax(ratio_shell)),
                "ratio_mean_shell": float(np.nanmean(ratio_shell)),
                "ratio_median_shell": float(np.nanmedian(ratio_shell)),
                "ratio_mean_distance_region": float(np.nanmean(ratio_shell)),
                "ratio_median_distance_region": float(np.nanmedian(ratio_shell)),
                "ratio_near_q05_mean": np.nan,
                "ratio_near_q10_mean": np.nan,
                "ratio_near_q20_mean": np.nan,
                "ratio_near_q30_mean": np.nan,
                "ratio_bead_crescent_30_mean": np.nan,
                "ratio_bead_crescent_45_mean": np.nan,
                "ratio_bead_crescent_60_mean": np.nan,
                "ratio_bead_crescent_90_mean": np.nan,
                "ratio_norm_denom": np.nan,
                "psmad_bg_mu_raw": float(CHANNEL_BG_PARAMS['psmad']['mu_bg_raw']),
                "psmad_bg_sigma_raw": float(CHANNEL_BG_PARAMS['psmad']['sigma_bg_raw']),
                "dapi_bg_mu_raw": float(CHANNEL_BG_PARAMS['dapi']['mu_bg_raw']),
                "dapi_bg_sigma_raw": float(CHANNEL_BG_PARAMS['dapi']['sigma_bg_raw']),
            }
        )

    return out


# -------------------------------
# Center-cyst exclusion + pre-analysis geometry montage
# -------------------------------
def infer_center_candidate_from_features(
    feature_df: pd.DataFrame,
    id_col: str,
    area_col: str = "area_px",
    cx_col: str = "centroid_x_px",
    cy_col: str = "centroid_y_px",
    area_ratio_max: float = 0.65,
    dist_max_px: float = 140.0,
) -> dict:
    """
    Pick center candidate as object closest to scene centroid, then mark as omittable
    only if it is sufficiently small relative to scene median area.
    """
    if len(feature_df) == 0:
        return {
            "candidate_id": None,
            "candidate_dist_px": np.nan,
            "candidate_area_ratio": np.nan,
            "scene_median_area_px": np.nan,
            "omit": False,
            "reason": "empty_scene",
        }

    scene_cx = float(np.mean(feature_df[cx_col].to_numpy(dtype=float)))
    scene_cy = float(np.mean(feature_df[cy_col].to_numpy(dtype=float)))

    d = np.sqrt(
        (feature_df[cx_col].to_numpy(dtype=float) - scene_cx) ** 2
        + (feature_df[cy_col].to_numpy(dtype=float) - scene_cy) ** 2
    )
    i = int(np.argmin(d))
    cand = feature_df.iloc[i]

    med_area = float(np.median(feature_df[area_col].to_numpy(dtype=float)))
    ratio = float(cand[area_col] / med_area) if med_area > 0 else np.nan
    dmin = float(d[i])

    omit = bool(
        np.isfinite(ratio)
        and ratio <= float(area_ratio_max)
        and dmin <= float(dist_max_px)
    )
    reason = "omit_center_cyst" if omit else "keep_all"

    return {
        "candidate_id": cand[id_col],
        "candidate_dist_px": dmin,
        "candidate_area_ratio": ratio,
        "scene_median_area_px": med_area,
        "omit": omit,
        "reason": reason,
    }


def build_roi_feature_table(roi_files: list[Path]) -> pd.DataFrame:
    rows = []
    for rf in roi_files:
        rec = json.loads(rf.read_text(encoding="utf-8"))
        cxy = rec.get("roi_centroid_xy_px", [np.nan, np.nan])
        rows.append(
            {
                "scene_index": int(rec.get("scene_index", -1)),
                "scene": rec.get("scene", ""),
                "cyst_id": str(rec.get("cyst_id", rf.stem)),
                "area_px": float(rec.get("roi_area_px", np.nan)),
                "centroid_x_px": float(cxy[0]) if len(cxy) >= 2 else np.nan,
                "centroid_y_px": float(cxy[1]) if len(cxy) >= 2 else np.nan,
                "roi_file": str(rf),
            }
        )
    return pd.DataFrame(rows)


def build_label_feature_table(scene_row: pd.Series, project_root: Path) -> pd.DataFrame:
    labels_path = resolve_path(str(scene_row["cyst_labels_path"]), project_root)
    labels = tifffile.imread(labels_path).astype(np.int32)
    props = measure.regionprops(labels)

    rows = []
    for r in props:
        rows.append(
            {
                "scene_index": int(scene_row["scene_index"]),
                "scene": scene_row.get("scene", ""),
                "label_id": int(r.label),
                "area_px": float(r.area),
                "centroid_x_px": float(r.centroid[1]),
                "centroid_y_px": float(r.centroid[0]),
            }
        )
    return pd.DataFrame(rows)


def build_scene_center_exclusion_map(
    feature_df: pd.DataFrame,
    id_col: str,
    area_ratio_max: float,
    dist_max_px: float,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    out_map = {}

    for scene_index, g in feature_df.groupby("scene_index"):
        info = infer_center_candidate_from_features(
            g,
            id_col=id_col,
            area_ratio_max=area_ratio_max,
            dist_max_px=dist_max_px,
        )
        row = {
            "scene_index": int(scene_index),
            "n_objects": int(len(g)),
            "candidate_id": info["candidate_id"],
            "candidate_dist_px": float(info["candidate_dist_px"]),
            "candidate_area_ratio": float(info["candidate_area_ratio"]),
            "scene_median_area_px": float(info["scene_median_area_px"]),
            "omit": bool(info["omit"]),
            "reason": info["reason"],
        }
        rows.append(row)
        if info["omit"]:
            out_map[int(scene_index)] = info["candidate_id"]

    out_df = pd.DataFrame(rows).sort_values("scene_index").reset_index(drop=True)
    return out_df, out_map


def infer_array_center_from_features(
    feature_df: pd.DataFrame,
    id_col: str,
    center_object_id,
    cx_col: str = "centroid_x_px",
    cy_col: str = "centroid_y_px",
) -> tuple[np.ndarray | None, str]:
    """
    Use the center-cyst centroid when available; otherwise use the mean centroid of the scene.
    This matches the geometry montage logic.
    """
    if len(feature_df) == 0:
        return None, "empty_scene"

    if center_object_id is not None:
        hit = feature_df.loc[feature_df[id_col].astype(str) == str(center_object_id)]
        if len(hit) >= 1:
            row = hit.iloc[0]
            xy = np.array([float(row[cx_col]), float(row[cy_col])], dtype=np.float64)
            if np.all(np.isfinite(xy)):
                return xy, "center_cyst_centroid"

    cx = feature_df[cx_col].to_numpy(dtype=float)
    cy = feature_df[cy_col].to_numpy(dtype=float)
    finite = np.isfinite(cx) & np.isfinite(cy)
    if not np.any(finite):
        return None, "no_finite_centroids"

    xy = np.array(
        [float(np.mean(cx[finite])), float(np.mean(cy[finite]))], dtype=np.float64
    )
    return xy, "mean_scene_centroid"


def build_scene_array_center_map(
    feature_df: pd.DataFrame,
    id_col: str,
    center_exclusion_map: dict,
    cx_col: str = "centroid_x_px",
    cy_col: str = "centroid_y_px",
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows = []
    out_map = {}

    for scene_index, g in feature_df.groupby("scene_index"):
        center_object_id = center_exclusion_map.get(int(scene_index), None)
        array_center_xy, source = infer_array_center_from_features(
            g,
            id_col=id_col,
            center_object_id=center_object_id,
            cx_col=cx_col,
            cy_col=cy_col,
        )

        row = {
            "scene_index": int(scene_index),
            "center_object_id": center_object_id,
            "array_center_source": source,
            "array_center_x_px": (
                float(array_center_xy[0]) if array_center_xy is not None else np.nan
            ),
            "array_center_y_px": (
                float(array_center_xy[1]) if array_center_xy is not None else np.nan
            ),
        }
        rows.append(row)

        if array_center_xy is not None and np.all(np.isfinite(array_center_xy)):
            out_map[int(scene_index)] = np.asarray(array_center_xy, dtype=np.float64)

    out_df = pd.DataFrame(rows).sort_values("scene_index").reset_index(drop=True)
    return out_df, out_map


def get_scene_array_center_or_fail(
    scene_index: int,
    array_center_map: dict[int, np.ndarray],
    context: str,
) -> np.ndarray:
    array_center_xy = array_center_map.get(int(scene_index), None)
    if array_center_xy is None:
        raise RuntimeError(
            f"Missing array-center coordinates for scene {int(scene_index):02d} in {context}"
        )
    array_center_xy = np.asarray(array_center_xy, dtype=np.float64)
    if array_center_xy.shape != (2,) or not np.all(np.isfinite(array_center_xy)):
        raise RuntimeError(
            f"Invalid array-center coordinates for scene {int(scene_index):02d} in {context}: {array_center_xy}"
        )
    return array_center_xy


def assign_ap_from_array_center(
    e_plus: np.ndarray,
    e_minus: np.ndarray,
    array_center_xy: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Anterior/posterior are defined by the cyst-array geometry:
    - anterior: axis endpoint closer to array center
    - posterior: axis endpoint farther from array center
    """
    if array_center_xy is not None and np.all(np.isfinite(array_center_xy)):
        d_plus = float(np.linalg.norm(e_plus - array_center_xy))
        d_minus = float(np.linalg.norm(e_minus - array_center_xy))
        if d_plus <= d_minus:
            anterior_xy = e_plus
            posterior_xy = e_minus
        else:
            anterior_xy = e_minus
            posterior_xy = e_plus
        return anterior_xy, posterior_xy, True

    # Fallback orientation when scene center is unavailable.
    return e_minus, e_plus, False


def geometry_from_mask_and_array_center(
    mask: np.ndarray,
    array_center_xy: np.ndarray | None,
) -> dict:
    centroid_xy, axis_u, boundary_xy = major_axis_from_mask(mask)
    e_plus, e_minus = choose_axis_endpoints(boundary_xy, centroid_xy, axis_u)

    anterior_xy, posterior_xy, ap_defined = assign_ap_from_array_center(
        e_plus=e_plus,
        e_minus=e_minus,
        array_center_xy=array_center_xy,
    )

    midline_u = unit_vector(posterior_xy - anterior_xy)
    tangent_u = np.array([-midline_u[1], midline_u[0]], dtype=np.float64)

    return {
        "centroid_xy": centroid_xy,
        "boundary_xy": boundary_xy,
        "anterior_xy": anterior_xy,
        "posterior_xy": posterior_xy,
        "midline_u": midline_u,
        "tangent_u": tangent_u,
        "ap_defined_from_array_center": ap_defined,
    }


def split_mask_into_two_lobes(
    mask: np.ndarray,
    min_peak_distance_px: float = 70.0,
    min_fraction: float = 0.22,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    """
    Attempt a 2-lobe split for merged cyst masks.

    Strategy:
    1) Distance-transform watershed using two separated maxima.
    2) Fallback to 2-cluster split in (x,y) space for elongated merged objects.
    """
    mask = mask.astype(bool)
    total = int(mask.sum())
    if total < 500:
        return None

    # ---- strategy 1: watershed from two distance peaks ----
    dist = ndi.distance_transform_edt(mask)
    peaks_yx = peak_local_max(
        dist,
        labels=mask,
        min_distance=max(6, int(round(min_peak_distance_px))),
        num_peaks=6,
        exclude_border=False,
    )

    if len(peaks_yx) >= 2:
        vals = dist[peaks_yx[:, 0], peaks_yx[:, 1]]
        order = np.argsort(vals)[::-1]

        chosen = []
        for oi in order:
            p = peaks_yx[int(oi)]
            if all(
                np.linalg.norm(p - q) >= float(min_peak_distance_px) for q in chosen
            ):
                chosen.append(p)
            if len(chosen) == 2:
                break

        if len(chosen) == 2:
            markers = np.zeros(mask.shape, dtype=np.int32)
            for i, (yy, xx) in enumerate(chosen, start=1):
                markers[int(yy), int(xx)] = int(i)

            ws = watershed(-dist, markers=markers, mask=mask)
            lobes = [(ws == 1), (ws == 2)]
            fracs = [float(l.sum()) / float(total) for l in lobes]

            if min(fracs) >= float(min_fraction):
                seed_xy = np.asarray(
                    [[float(x), float(y)] for y, x in chosen], dtype=np.float64
                )
                return lobes, seed_xy

    # ---- strategy 2: PCA-initialized 2-means fallback ----
    ys, xs = np.where(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    if len(pts) < 800:
        return None

    centroid = pts.mean(axis=0)
    pts0 = pts - centroid[None, :]
    cov = np.cov(pts0.T)
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, int(np.argmax(evals))]

    proj = pts0 @ u
    c1 = pts[int(np.argmin(proj))].copy()
    c2 = pts[int(np.argmax(proj))].copy()

    for _ in range(16):
        d1 = np.sum((pts - c1[None, :]) ** 2, axis=1)
        d2 = np.sum((pts - c2[None, :]) ** 2, axis=1)
        assign1 = d1 <= d2

        n1 = int(np.sum(assign1))
        n2 = int(len(assign1) - n1)
        if n1 == 0 or n2 == 0:
            return None

        nc1 = pts[assign1].mean(axis=0)
        nc2 = pts[~assign1].mean(axis=0)

        if np.allclose(nc1, c1) and np.allclose(nc2, c2):
            break
        c1, c2 = nc1, nc2

    l1 = np.zeros(mask.shape, dtype=bool)
    l2 = np.zeros(mask.shape, dtype=bool)
    l1[ys[assign1], xs[assign1]] = True
    l2[ys[~assign1], xs[~assign1]] = True

    fr1 = float(l1.sum()) / float(total)
    fr2 = float(l2.sum()) / float(total)
    if min(fr1, fr2) < float(min_fraction):
        return None

    seeds = np.asarray(
        [[float(c1[0]), float(c1[1])], [float(c2[0]), float(c2[1])]], dtype=np.float64
    )
    return [l1, l2], seeds


def _norm01_for_overlay(
    img: np.ndarray, p_lo: float = 2.0, p_hi: float = 99.5
) -> np.ndarray:
    vals = img[np.isfinite(img)]
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    den = max(1e-6, float(hi - lo))
    return np.clip((img - lo) / den, 0.0, 1.0)


def centroid_xy_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) < 10:
        raise ValueError("Mask too small for centroid estimation")
    return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)


def should_draw_two_axis_geometry(
    lobes: list[np.ndarray],
    array_center_xy: np.ndarray | None,
    outer_radius_ref_px: float | None,
    min_outer_radius_fraction: float = 0.60,
    min_radius_ratio: float = 0.55,
) -> tuple[bool, dict]:
    """
    Allow a two-axis overlay only when both split lobes behave like outer cysts.

    This suppresses false two-axis overlays for outer+center merged objects
    (for example S31/S33), while keeping true outer+outer merges
    (for example S04/S35).
    """
    if array_center_xy is None:
        return False, {"reason": "missing_array_center"}

    array_center_xy = np.asarray(array_center_xy, dtype=np.float64)
    if array_center_xy.shape != (2,) or not np.all(np.isfinite(array_center_xy)):
        return False, {"reason": "invalid_array_center"}

    if len(lobes) != 2:
        return False, {"reason": "need_exactly_two_lobes"}

    dists = []
    centroids = []
    for lmask in lobes:
        try:
            cxy = centroid_xy_from_mask(lmask)
        except Exception:
            return False, {"reason": "bad_lobe_centroid"}
        centroids.append(cxy)
        dists.append(float(np.linalg.norm(cxy - array_center_xy)))

    dists = np.asarray(dists, dtype=np.float64)
    if not np.all(np.isfinite(dists)):
        return False, {"reason": "nonfinite_lobe_distance"}

    radius_ratio = (
        float(np.min(dists) / np.max(dists)) if float(np.max(dists)) > 0 else 0.0
    )

    if (
        outer_radius_ref_px is None
        or not np.isfinite(outer_radius_ref_px)
        or float(outer_radius_ref_px) <= 0
    ):
        outer_threshold = 0.0
    else:
        outer_threshold = float(outer_radius_ref_px) * float(min_outer_radius_fraction)

    all_outer_like = bool(np.min(dists) >= outer_threshold)
    balanced = bool(radius_ratio >= float(min_radius_ratio))
    keep_two_axis = bool(all_outer_like and balanced)

    return keep_two_axis, {
        "reason": (
            "two_outer_lobes" if keep_two_axis else "outer_plus_center_like_merge"
        ),
        "lobe_centroids_xy": np.asarray(centroids, dtype=np.float64),
        "lobe_distances_px": dists,
        "outer_threshold_px": float(outer_threshold),
        "radius_ratio": float(radius_ratio),
    }


def _draw_ap_geometry(
    ax,
    g: dict,
    color,
    tangent_half_len_px: float,
    alpha: float = 0.9,
    midline_style: str = "-",
) -> None:
    c = g["centroid_xy"]
    a = g["anterior_xy"]
    p = g["posterior_xy"]
    t = g["tangent_u"]

    ax.plot(
        [a[0], p[0]],
        [a[1], p[1]],
        color=color,
        linewidth=1.0,
        linestyle=midline_style,
        alpha=alpha,
    )
    ax.plot(
        [p[0] - t[0] * tangent_half_len_px, p[0] + t[0] * tangent_half_len_px],
        [p[1] - t[1] * tangent_half_len_px, p[1] + t[1] * tangent_half_len_px],
        color=color,
        linewidth=0.8,
        linestyle="--",
        alpha=alpha,
    )

    # Point color coding requested for readability.
    ax.scatter([c[0]], [c[1]], s=12, c="cyan", marker="o", linewidths=0.0)
    ax.scatter([a[0]], [a[1]], s=13, c="lime", marker="o", linewidths=0.0)
    ax.scatter([p[0]], [p[1]], s=13, c="red", marker="o", linewidths=0.0)


def plot_all_scene_outline_before_analysis(
    bead_df: pd.DataFrame,
    project_root: Path,
    ncols: int = 6,
) -> None:
    """Simple all-scene outline montage: BF + cyst labels + bead marker."""
    scene_ids = bead_df["scene_index"].astype(int).tolist()
    n = len(scene_ids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.15 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, scene_index in zip(axes, scene_ids):
        sr = bead_df.loc[bead_df["scene_index"] == int(scene_index)].iloc[0]
        bf = load_image_2d(
            resolve_path(str(sr["brightfield_image_path"]), project_root),
            projection="first",
        )
        labels = tifffile.imread(
            resolve_path(str(sr["cyst_labels_path"]), project_root)
        ).astype(np.int32)

        ax.imshow(_norm01_for_overlay(bf), cmap="gray")
        ax.contour(
            labels > 0, levels=[0.5], colors="deepskyblue", linewidths=0.6, alpha=0.9
        )

        if np.isfinite(sr["bead_well_x_px"]) and np.isfinite(sr["bead_well_y_px"]):
            ax.scatter(
                [float(sr["bead_well_x_px"])],
                [float(sr["bead_well_y_px"])],
                c="magenta",
                s=28,
                marker="x",
                linewidths=1.2,
            )
            bead_txt = "bead=Y"
        else:
            bead_txt = "bead=N"

        ax.set_title(
            f"S{scene_index:02d} c={int(sr['n_cysts_detected'])} {bead_txt}", fontsize=8
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "All-scene outline montage: BF + cyst labels + bead marker",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.subplots_adjust(hspace=0.16, wspace=0.04)
    plt.show()


def plot_all_scenes_geometry_before_analysis(
    bead_df: pd.DataFrame,
    center_label_map: dict,
    project_root: Path,
    ncols: int = 6,
    tangent_half_len_px: float = 95.0,
) -> None:
    """
    Pre-analysis montage across all scenes showing geometry assignments.

    Important definitions used here:
    - anterior = endpoint toward array center
    - posterior = endpoint away from array center

    Center cysts (when identified) are shown without axis/AP points.
    """
    scene_ids = bead_df["scene_index"].astype(int).tolist()
    n = len(scene_ids)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.45 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, scene_index in zip(axes, scene_ids):
        sr = bead_df.loc[bead_df["scene_index"] == int(scene_index)].iloc[0]

        bf = load_image_2d(
            resolve_path(str(sr["brightfield_image_path"]), project_root),
            projection="first",
        )
        labels = tifffile.imread(
            resolve_path(str(sr["cyst_labels_path"]), project_root)
        ).astype(np.int32)

        bead_xy = None
        if np.isfinite(sr["bead_well_x_px"]) and np.isfinite(sr["bead_well_y_px"]):
            bead_xy = np.array(
                [float(sr["bead_well_x_px"]), float(sr["bead_well_y_px"])],
                dtype=np.float64,
            )

        omit_label = center_label_map.get(int(scene_index), None)

        props = measure.regionprops(labels)
        if len(props) == 0:
            ax.imshow(_norm01_for_overlay(bf), cmap="gray")
            ax.set_title(f"S{int(scene_index):02d} empty labels", fontsize=8)
            ax.axis("off")
            continue

        label_ids = np.array([int(r.label) for r in props], dtype=int)
        areas = np.array([float(r.area) for r in props], dtype=float)
        med_area = float(np.median(areas)) if len(areas) else np.nan

        # Array center comes from center-cyst centroid when available, otherwise scene mean centroid.
        centroids_xy = np.array(
            [[float(r.centroid[1]), float(r.centroid[0])] for r in props],
            dtype=np.float64,
        )
        if omit_label is not None and np.any(label_ids == int(omit_label)):
            center_prop = [r for r in props if int(r.label) == int(omit_label)][0]
            array_center_xy = np.array(
                [float(center_prop.centroid[1]), float(center_prop.centroid[0])],
                dtype=np.float64,
            )
        else:
            array_center_xy = np.array(
                [
                    float(np.mean(centroids_xy[:, 0])),
                    float(np.mean(centroids_xy[:, 1])),
                ],
                dtype=np.float64,
            )

        outer_distances_px = np.linalg.norm(
            centroids_xy - array_center_xy[None, :], axis=1
        )
        if omit_label is not None:
            outer_mask = label_ids != int(omit_label)
            if np.any(outer_mask):
                outer_distances_px = outer_distances_px[outer_mask]
        outer_radius_ref_px = (
            float(np.median(outer_distances_px)) if len(outer_distances_px) else np.nan
        )

        ax.imshow(_norm01_for_overlay(bf), cmap="gray")
        ax.scatter(
            [array_center_xy[0]],
            [array_center_xy[1]],
            c="white",
            s=16,
            marker="+",
            linewidths=0.9,
        )

        n_two_axis = 0

        for i, r in enumerate(props):
            label_id = int(r.label)
            mask = labels == label_id
            edge_color = plt.cm.tab20(i % 20)

            # Always show the cyst contour.
            try:
                boundary_xy = extract_boundary_xy(mask)
                ax.plot(
                    boundary_xy[:, 0],
                    boundary_xy[:, 1],
                    color=edge_color,
                    linewidth=0.75,
                    alpha=0.9,
                )
            except Exception:
                continue

            cxy = np.array(
                [float(r.centroid[1]), float(r.centroid[0])], dtype=np.float64
            )
            is_center = omit_label is not None and int(label_id) == int(omit_label)

            if is_center:
                # Requested behavior: no axis or A/P points for center cyst.
                ax.scatter(
                    [cxy[0]], [cxy[1]], s=12, c="cyan", marker="o", linewidths=0.0
                )
                ax.text(
                    cxy[0],
                    cxy[1],
                    "center",
                    color="white",
                    fontsize=6,
                    ha="center",
                    va="center",
                )
                continue

            area_ratio = float(r.area) / max(1.0, med_area)
            merged_candidate = bool(
                len(props) <= 6
                and np.isfinite(area_ratio)
                and area_ratio >= float(MERGED_AXIS_AREA_RATIO)
            )

            lobes_payload = None
            if merged_candidate:
                lobes_payload = split_mask_into_two_lobes(
                    mask,
                    min_peak_distance_px=MERGED_AXIS_MIN_PEAK_DISTANCE_PX,
                    min_fraction=MERGED_AXIS_MIN_FRACTION,
                )

            if lobes_payload is not None:
                lobes, seed_xy = lobes_payload
                use_two_axis, lobe_info = should_draw_two_axis_geometry(
                    lobes,
                    array_center_xy=array_center_xy,
                    outer_radius_ref_px=outer_radius_ref_px,
                    min_outer_radius_fraction=MERGED_AXIS_MIN_OUTER_RADIUS_FRACTION,
                    min_radius_ratio=MERGED_AXIS_MIN_RADIUS_RATIO,
                )
                if use_two_axis:
                    drawn = 0
                    for j, lmask in enumerate(lobes):
                        try:
                            g = geometry_from_mask_and_array_center(
                                lmask, array_center_xy=array_center_xy
                            )
                        except Exception:
                            continue
                        style = "-" if j == 0 else "-."
                        _draw_ap_geometry(
                            ax,
                            g,
                            color=edge_color,
                            tangent_half_len_px=tangent_half_len_px,
                            alpha=0.95,
                            midline_style=style,
                        )
                        drawn += 1
                    if drawn >= 2:
                        n_two_axis += 1
                        ax.scatter(
                            seed_xy[:, 0],
                            seed_xy[:, 1],
                            s=9,
                            c="yellow",
                            marker="o",
                            linewidths=0.0,
                        )
                        continue

            # Default single-axis geometry.
            try:
                g = geometry_from_mask_and_array_center(
                    mask, array_center_xy=array_center_xy
                )
                _draw_ap_geometry(
                    ax,
                    g,
                    color=edge_color,
                    tangent_half_len_px=tangent_half_len_px,
                    alpha=0.9,
                    midline_style="-",
                )
            except Exception:
                pass

        if bead_xy is not None:
            ax.scatter(
                [bead_xy[0]],
                [bead_xy[1]],
                c="magenta",
                s=28,
                marker="x",
                linewidths=1.2,
            )
            bead_txt = "bead=Y"
        else:
            bead_txt = "bead=N"

        omit_txt = (
            f"omit_label={omit_label}" if omit_label is not None else "omit_label=None"
        )
        ax.set_title(
            f"S{int(scene_index):02d} n={int(sr['n_cysts_detected'])} {bead_txt} twoAxis={n_two_axis}\n{omit_txt}",
            fontsize=7,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Pre-analysis geometry montage (all scenes): AP defined by array center; center cyst has no axis/AP points",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])
    fig.subplots_adjust(hspace=0.20, wspace=0.05)
    plt.show()


_diag_image_cache = {}


def _norm_disp(img: np.ndarray, p_lo: float = 2.0, p_hi: float = 99.8) -> np.ndarray:
    vals = img[np.isfinite(img)]
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    den = max(1e-6, float(hi - lo))
    return np.clip((img - lo) / den, 0.0, 1.0)


def _cached_load(path: Path, projection: str = "max") -> np.ndarray:
    key = (str(path), projection)
    if key not in _diag_image_cache:
        _diag_image_cache[key] = load_image_2d(path, projection=projection)
    return _diag_image_cache[key]


def _scene_row_or_fail(scene_index: int) -> pd.Series:
    row = bead_df.loc[bead_df["scene_index"] == int(scene_index)]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly 1 scene row for scene_index={scene_index}, found {len(row)}"
        )
    return row.iloc[0]


def _summary_row_any_or_fail(cyst_id: str) -> pd.Series:
    row = summary_df.loc[summary_df["cyst_id"].astype(str) == str(cyst_id)]
    if len(row) != 1:
        raise RuntimeError(
            f"Expected exactly 1 summary row for cyst_id={cyst_id}, found {len(row)}"
        )
    return row.iloc[0]


def _control_label_id_from_cyst_id(cyst_id: str) -> int:
    try:
        return int(str(cyst_id).rsplit("_", 1)[-1])
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse control label ID from cyst_id={cyst_id}"
        ) from exc


def _wrapped_abs_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.abs(
        wrap_angle_deg(
            np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        )
    )


def _sample_points_for_plot(
    x: np.ndarray, y: np.ndarray, max_points: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) <= int(max_points):
        return x, y
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(x), size=int(max_points), replace=False)
    idx.sort()
    return x[idx], y[idx]


def build_debug_payload(cyst_id: str, summary_source_df: pd.DataFrame | None = None) -> dict:
    source_df = summary_df if summary_source_df is None else summary_source_df
    row_df = source_df[
        (source_df["cyst_id"] == cyst_id) & (source_df["bead_present"].astype(bool))
    ]
    if len(row_df) != 1:
        raise RuntimeError(
            f"Expected exactly one bead-present row for {cyst_id}, found {len(row_df)}"
        )
    row = row_df.iloc[0]

    roi_path = Path(str(row["roi_file"]))
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")
    rec = json.loads(roi_path.read_text(encoding="utf-8"))

    scene_index = int(row["scene_index"])
    scene_row = _scene_row_or_fail(scene_index)

    bf_path = resolve_path(str(scene_row["brightfield_image_path"]), ROOT)
    psmad_path = resolve_path(str(rec["psmad_image_path"]), ROOT)
    dapi_path = resolve_path(str(rec["dapi_image_path"]), ROOT)

    bf = _cached_load(bf_path, projection="first")
    psmad_raw = _cached_load(psmad_path, projection=PROJECTION)
    dapi_raw = _cached_load(dapi_path, projection=PROJECTION)
    psmad_norm, dapi_norm, _ = normalize_two_channels_and_ratio(
        psmad_raw,
        dapi_raw,
        CHANNEL_BG_PARAMS,
        eps=EPS,
    )

    array_center_xy = get_scene_array_center_or_fail(
        scene_index=scene_index,
        array_center_map=roi_array_center_map,
        context="debug payload",
    )
    measurement = measure_bead_present_arrays(
        record=rec,
        roi_file=roi_path,
        psmad_raw=psmad_raw,
        dapi_raw=dapi_raw,
        psmad_norm=psmad_norm,
        dapi_norm=dapi_norm,
        array_center_xy=array_center_xy,
    )
    measurement.update(summarize_bead_present_measurement(measurement))

    payload = {
        "row": row,
        "rec": rec,
        "scene_row": scene_row,
        "bf": bf,
        "dapi": dapi_raw,
        "psmad": psmad_raw,
        "dapi_norm": dapi_norm,
        "psmad_norm": psmad_norm,
    }
    payload.update(measurement)
    return payload


def _xy_values_to_image(
    shape: tuple[int, int], xy: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Rasterize per-pixel values onto an image grid with NaN outside sampled pixels."""
    out = np.full(shape, np.nan, dtype=np.float32)
    if xy.size == 0:
        return out
    xx = np.clip(np.round(xy[:, 0]).astype(int), 0, shape[1] - 1)
    yy = np.clip(np.round(xy[:, 1]).astype(int), 0, shape[0] - 1)
    out[yy, xx] = np.asarray(values, dtype=np.float32)
    return out


def _discrete_region_boundary_mask(value_img: np.ndarray) -> np.ndarray:
    """Return a 1-pixel mask at interfaces between neighboring categorical regions."""
    valid = np.isfinite(value_img)
    if not np.any(valid):
        return np.zeros_like(valid, dtype=bool)

    filled = np.where(valid, value_img, np.nan)
    boundary = np.zeros_like(valid, dtype=bool)

    vertical_change = valid[:-1, :] & valid[1:, :] & (filled[:-1, :] != filled[1:, :])
    horizontal_change = valid[:, :-1] & valid[:, 1:] & (filled[:, :-1] != filled[:, 1:])

    boundary[:-1, :] |= vertical_change
    boundary[1:, :] |= vertical_change
    boundary[:, :-1] |= horizontal_change
    boundary[:, 1:] |= horizontal_change
    return boundary


def _mask_crop_bounds(mask: np.ndarray, pad: int = 12) -> tuple[int, int, int, int]:
    """Return a padded crop box around the nonzero mask region."""
    yy, xx = np.nonzero(mask)
    if len(xx) == 0 or len(yy) == 0:
        return 0, mask.shape[1], mask.shape[0], 0
    y0 = max(0, int(np.min(yy)) - pad)
    y1 = min(mask.shape[0], int(np.max(yy)) + pad + 1)
    x0 = max(0, int(np.min(xx)) - pad)
    x1 = min(mask.shape[1], int(np.max(xx)) + pad + 1)
    return x0, x1, y1, y0


def plot_debug_payload(
    payload: dict, scene_rows_source_df: pd.DataFrame | None = None
) -> None:
    row = payload["row"]
    bf = payload["bf"]
    dapi = payload["dapi"]
    psmad = payload["psmad"]
    ratio = payload["ratio"]
    mask = payload["mask"]

    shell_mask = payload["shell_mask"]
    boundary_xy = payload["boundary_xy"]
    shell_xy = payload["shell_xy"]
    centroid_xy = payload["centroid_xy"]
    anterior_xy = payload["anterior_xy"]
    posterior_xy = payload["posterior_xy"]
    nearest_boundary_xy = payload["nearest_boundary_xy"]
    bead_xy = payload["bead_xy"]
    array_center_xy = payload["array_center_xy"]
    tangent_u = payload["tangent_u"]

    local_distance_shell_um = payload["local_distance_shell_um"]
    theta_rel_bead_shell_deg = payload["theta_rel_bead_shell_deg"]
    ratio_norm_shell = payload["ratio_norm_shell"]
    ratio_shell = payload["ratio_shell"]

    local_distance_distance_um = payload["local_distance_distance_um"]
    ratio_distance = payload["ratio_distance"]
    distance_trimmed_xy = payload["distance_trimmed_xy"]
    distance_support_df = payload["distance_support_df"]
    theta_rel_trace_df = payload["theta_rel_trace_df"]

    theta_peak_rel_bead_deg = float(row["theta_peak_rel_bead_deg"])
    theta_moment_rel_bead_deg = float(row["theta_moment_rel_bead_deg"])

    shell_overlay = np.ma.masked_where(~shell_mask, shell_mask.astype(float))
    ratio_masked = np.where(mask, ratio, np.nan)
    shell_norm_img = _xy_values_to_image(ratio.shape, shell_xy, ratio_norm_shell)
    theta_rel_shell_img = _xy_values_to_image(ratio.shape, shell_xy, theta_rel_bead_shell_deg)
    distance_keep_mask = xy_bool_to_mask(ratio.shape, shell_xy, payload["keep_distance_pixel"])
    distance_trim_mask = xy_bool_to_mask(ratio.shape, shell_xy, ~payload["keep_distance_pixel"])
    keep_overlay = np.ma.masked_where(~distance_keep_mask, distance_keep_mask.astype(float))
    trim_overlay = np.ma.masked_where(~distance_trim_mask, distance_trim_mask.astype(float))

    vec_scale = 160.0
    line_mid = np.vstack([anterior_xy, posterior_xy])
    line_tan = np.vstack(
        [
            posterior_xy - tangent_u * vec_scale,
            posterior_xy + tangent_u * vec_scale,
        ]
    )

    fig, axes = plt.subplots(4, 3, figsize=(17, 18))
    axes = axes.ravel()

    source_df = summary_df if scene_rows_source_df is None else scene_rows_source_df
    scene_rows = (
        source_df[
            (source_df["scene_index"] == int(row["scene_index"]))
            & (source_df["bead_present"].astype(bool))
        ]
        .copy()
        .sort_values("cyst_id")
    )

    def _cyst_number_label(cid: str) -> str:
        tail = str(cid).rsplit("_", 1)[-1]
        try:
            return f"c{int(tail)}"
        except Exception:
            return str(cid)

    # Row 1: raw images only.
    axes[0].imshow(_norm_disp(bf), cmap="gray")
    axes[0].plot(boundary_xy[:, 0], boundary_xy[:, 1], color="cyan", linewidth=1.0)
    axes[0].scatter([bead_xy[0]], [bead_xy[1]], c="magenta", s=70, marker="x")
    for _, rr in scene_rows.iterrows():
        cx = float(rr["centroid_x_px"])
        cy = float(rr["centroid_y_px"])
        cid = str(rr["cyst_id"])
        label = _cyst_number_label(cid)
        is_selected = cid == str(row["cyst_id"])
        axes[0].scatter(
            [cx],
            [cy],
            s=34 if is_selected else 20,
            c="cyan" if is_selected else "white",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
        )
        axes[0].text(
            cx + 18.0,
            cy - 18.0,
            label,
            color="cyan" if is_selected else "white",
            fontsize=8,
            fontweight="bold" if is_selected else "normal",
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=(0, 0, 0, 0.55),
                edgecolor="none",
            ),
            zorder=7,
        )
    axes[0].set_title("Brightfield raw + ROI contour + bead + scene cyst labels")

    axes[1].imshow(_norm_disp(dapi), cmap="gray")
    axes[1].plot(boundary_xy[:, 0], boundary_xy[:, 1], color="cyan", linewidth=0.8)
    axes[1].set_title("DAPI raw + ROI contour")

    axes[2].imshow(_norm_disp(psmad), cmap="magma")
    axes[2].plot(
        boundary_xy[:, 0], boundary_xy[:, 1], color="white", linewidth=0.8, alpha=0.6
    )
    axes[2].set_title("pSMAD raw + ROI contour")

    axes[3].imshow(_norm_disp(ratio_masked), cmap="viridis")
    axes[3].plot(
        boundary_xy[:, 0], boundary_xy[:, 1], color="white", linewidth=0.8, alpha=0.6
    )
    axes[3].set_title("pSMAD/DAPI within cyst mask only")

    # Row 2: same images with the exact shell pixels overlaid.
    axes[4].imshow(_norm_disp(bf), cmap="gray")
    axes[4].plot(
        boundary_xy[:, 0], boundary_xy[:, 1], color="white", linewidth=0.8, alpha=0.6
    )
    axes[4].imshow(shell_overlay, cmap="winter", alpha=0.28, interpolation="nearest")
    axes[4].plot(
        line_mid[:, 0],
        line_mid[:, 1],
        color="deepskyblue",
        linewidth=2.0,
        label="AP midline",
    )
    axes[4].plot(
        line_tan[:, 0],
        line_tan[:, 1],
        color="orange",
        linewidth=1.2,
        label="tangent @ posterior",
    )
    axes[4].scatter(
        [centroid_xy[0]], [centroid_xy[1]], c="cyan", s=45, label="centroid"
    )
    axes[4].scatter(
        [array_center_xy[0]],
        [array_center_xy[1]],
        c="white",
        s=70,
        marker="+",
        label="array center",
    )
    axes[4].scatter(
        [anterior_xy[0]], [anterior_xy[1]], c="lime", s=55, label="anterior"
    )
    axes[4].scatter(
        [posterior_xy[0]], [posterior_xy[1]], c="red", s=55, label="posterior"
    )
    axes[4].scatter(
        [nearest_boundary_xy[0]],
        [nearest_boundary_xy[1]],
        c="yellow",
        s=60,
        marker="o",
        label="nearest surface point",
    )
    axes[4].scatter(
        [bead_xy[0]], [bead_xy[1]], c="magenta", s=65, marker="x", label="bead"
    )
    axes[4].set_title("Brightfield + exact shell pixels + geometry")
    axes[4].legend(fontsize=8, loc="best")

    axes[5].imshow(_norm_disp(dapi), cmap="gray")
    axes[5].plot(boundary_xy[:, 0], boundary_xy[:, 1], color="cyan", linewidth=0.8)
    axes[5].imshow(shell_overlay, cmap="winter", alpha=0.28, interpolation="nearest")
    axes[5].set_title("DAPI + exact shell pixels")

    axes[6].imshow(_norm_disp(psmad), cmap="magma")
    axes[6].plot(
        boundary_xy[:, 0], boundary_xy[:, 1], color="white", linewidth=0.8, alpha=0.6
    )
    axes[6].imshow(shell_overlay, cmap="winter", alpha=0.28, interpolation="nearest")
    axes[6].set_title("pSMAD + exact shell pixels")

    axes[7].imshow(_norm_disp(ratio_masked), cmap="viridis")
    axes[7].plot(boundary_xy[:, 0], boundary_xy[:, 1], color="white", linewidth=0.8, alpha=0.6)
    theta_overlay = np.ma.masked_invalid(theta_rel_shell_img)
    theta_norm = mcolors.Normalize(vmin=-180.0, vmax=180.0)
    theta_cmap = plt.get_cmap("twilight_shifted")
    axes[7].imshow(theta_overlay, cmap=theta_cmap, norm=theta_norm, alpha=0.92, interpolation="nearest")
    axes[7].scatter([bead_xy[0]], [bead_xy[1]], c="magenta", s=45, marker="x")
    axes[7].set_title("Shell pixels colored by angle rel. to bead")
    cax = inset_axes(
        axes[7],
        width="4%",
        height="75%",
        loc="lower left",
        bbox_to_anchor=(1.03, 0.12, 1, 1),
        bbox_transform=axes[7].transAxes,
        borderpad=0,
    )
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=theta_norm, cmap=theta_cmap),
        cax=cax,
        ticks=[-180, -90, 0, 90, 180],
    )
    cb.ax.tick_params(labelsize=7)
    cb.set_label("deg", fontsize=8)

    # Row 3: distance-support trimming diagnostics.
    axes[8].imshow(_norm_disp(bf), cmap="gray")
    axes[8].plot(
        boundary_xy[:, 0], boundary_xy[:, 1], color="white", linewidth=0.6, alpha=0.4
    )
    axes[8].imshow(keep_overlay, cmap="Greens", alpha=0.45, interpolation="nearest")
    axes[8].imshow(trim_overlay, cmap="Reds", alpha=0.55, interpolation="nearest")
    axes[8].scatter([bead_xy[0]], [bead_xy[1]], c="magenta", s=55, marker="x")
    axes[8].set_title("Distance-analysis shell pixels: kept (green), low-count edges trimmed (red)")

    support_df = distance_support_df.loc[distance_support_df["n_pixels"] > 0].copy()
    axes[9].plot(
        support_df["distance_local_um"],
        support_df["n_pixels"],
        color="0.65",
        linewidth=1.3,
        label="per-bin pixel count",
    )
    axes[9].plot(
        support_df["distance_local_um"],
        support_df["count_support_n_pixels"],
        color="black",
        linewidth=2.0,
        label=f"{int(row['distance_support_window_bins'])}-bin aggregated count support",
    )
    axes[9].axhline(
        float(row["distance_support_min_pixels"]),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="minimum count threshold",
    )
    for _, tr in support_df.loc[~support_df["keep_bin"]].iterrows():
        x0 = float(tr["distance_local_um"] - 0.5 * DISTANCE_TRACE_BIN_UM)
        x1 = float(tr["distance_local_um"] + 0.5 * DISTANCE_TRACE_BIN_UM)
        axes[9].axvspan(x0, x1, color="red", alpha=0.12)
    axes[9].set_xlabel("Local bead-to-pixel distance (um)")
    axes[9].set_ylabel("Pixel count per 10 um bin")
    axes[9].set_title("Distance-bin pixel-count support and edge trimming")
    axes[9].legend(fontsize=8, loc="best")

    axes[10].scatter(
        local_distance_shell_um,
        ratio_shell,
        s=5,
        alpha=0.04,
        color="0.8",
        label="all shell pixels",
    )
    axes[10].scatter(
        local_distance_distance_um,
        ratio_distance,
        s=6,
        alpha=0.12,
        color="gray",
        label="kept pixels",
    )
    dist_b = binned_mean_sem(
        local_distance_distance_um, ratio_distance, DISTANCE_TRACE_BIN_UM
    )
    if len(dist_b) > 0:
        axes[10].plot(
            dist_b["x_center"],
            dist_b["mean"],
            color="tab:blue",
            linewidth=2.0,
            label="binned mean (kept bins)",
        )
        axes[10].fill_between(
            dist_b["x_center"],
            dist_b["mean"] - dist_b["sem"],
            dist_b["mean"] + dist_b["sem"],
            color="tab:blue",
            alpha=0.22,
            label="±SEM",
        )
    for _, tr in support_df.loc[~support_df["keep_bin"]].iterrows():
        x0 = float(tr["distance_local_um"] - 0.5 * DISTANCE_TRACE_BIN_UM)
        x1 = float(tr["distance_local_um"] + 0.5 * DISTANCE_TRACE_BIN_UM)
        axes[10].axvspan(x0, x1, color="red", alpha=0.12)
    axes[10].axvline(
        float(row["distance_d_um"]),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="nearest surface d",
    )
    axes[10].axhline(
        float(row["ratio_peak"]),
        color="purple",
        linestyle=":",
        linewidth=1.5,
        label="peak-like summary",
    )
    axes[10].set_xlabel("Local bead-to-pixel distance after edge trimming (um)")
    axes[10].set_ylabel("Background-normalized pSMAD/DAPI")
    axes[10].set_title("Distance analysis after edge-only low-count trimming")
    axes[10].legend(fontsize=8, loc="best")

    axes[11].scatter(
        theta_rel_bead_shell_deg,
        ratio_norm_shell,
        s=6,
        alpha=0.10,
        color="gray",
        label="raw full-shell pixels",
    )
    if len(theta_rel_trace_df) > 0:
        axes[11].plot(
            theta_rel_trace_df["theta_rel_bead_deg"],
            theta_rel_trace_df["ratio_norm_mean"],
            color="tab:green",
            linewidth=2.0,
            label="binned mean",
        )
    axes[11].axvline(
        0.0, color="red", linestyle="--", linewidth=1.5, label="bead direction"
    )
    if np.isfinite(theta_peak_rel_bead_deg):
        axes[11].axvline(
            theta_peak_rel_bead_deg,
            color="purple",
            linestyle=":",
            linewidth=1.8,
            label="peak direction",
        )
    if np.isfinite(theta_moment_rel_bead_deg):
        axes[11].axvline(
            theta_moment_rel_bead_deg,
            color="orange",
            linestyle="-.",
            linewidth=1.8,
            label="first moment",
        )
    axes[11].set_xlabel("Angle relative to bead direction (deg)")
    axes[11].set_ylabel("Cyst-normalized pSMAD/DAPI")
    axes[11].set_title("Orientation analysis from full shell")
    axes[11].legend(fontsize=8, loc="best")

    for ax in axes:
        if ax in [axes[9], axes[10], axes[11]]:
            continue
        ax.axis("off")

    fig.suptitle(
        (
            f"Diagnostics for {row['cyst_id']} | scene={row['scene_index']} | "
            f"distance uses full shell with edge-only raw-DAPI trimming; orientation uses full shell | "
            f"shell_depth_px={SHELL_DEPTH_PX:.0f} | d={row['distance_d_um']:.1f} um | "
            f"theta_b={row['theta_b_deg']:.1f} deg | theta_peak={row['theta_peak_deg']:.1f} deg | "
            f"theta_moment={row['theta_moment_deg']:.1f} deg"
        ),
        fontsize=11,
        y=0.995,
    )
    # This diagnostic figure uses a dense fixed panel layout; explicit spacing
    # avoids repeated tight_layout warnings in the executed notebook.
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.92, hspace=0.22, wspace=0.12)
    plt.show()


def plot_geometry_montage(
    cyst_ids: list[str],
    ncols: int = 4,
    summary_source_df: pd.DataFrame | None = None,
) -> None:
    cyst_ids = [str(c) for c in cyst_ids]
    n = len(cyst_ids)
    if n == 0:
        print("No cyst IDs provided for montage.")
        return

    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(8.2 * ncols, 4.2 * nrows))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.18, hspace=0.30)

    for idx, cid in enumerate(cyst_ids):
        row_i = idx // ncols
        col_i = idx % ncols
        sub = outer[row_i, col_i].subgridspec(1, 2, wspace=0.05)
        ax_dapi = fig.add_subplot(sub[0, 0])
        ax_psmad = fig.add_subplot(sub[0, 1])
        try:
            p = build_debug_payload(cid, summary_source_df=summary_source_df)
            dapi = p["dapi"]
            psmad = p["psmad"]
            boundary_xy = p["boundary_xy"]
            bead_xy = p["bead_xy"]
            mask = p["mask"]
            row = p["row"]

            yy, xx = np.nonzero(mask)
            pad = 60
            y0 = max(0, int(np.min(yy)) - pad)
            y1 = min(mask.shape[0], int(np.max(yy)) + pad + 1)
            x0 = max(0, int(np.min(xx)) - pad)
            x1 = min(mask.shape[1], int(np.max(xx)) + pad + 1)

            for ax, img, title, cmap, edge_color in [
                (ax_dapi, dapi, "DAPI", "gray", "cyan"),
                (ax_psmad, psmad, "pSMAD", "magma", "white"),
            ]:
                ax.imshow(_norm_disp(img), cmap=cmap)
                ax.plot(
                    boundary_xy[:, 0],
                    boundary_xy[:, 1],
                    color=edge_color,
                    linewidth=0.8,
                    alpha=0.8,
                )
                ax.scatter(
                    [bead_xy[0]],
                    [bead_xy[1]],
                    c="magenta",
                    s=40,
                    marker="x",
                    linewidths=1.4,
                )
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)
                ax.set_title(title, fontsize=8)
                ax.axis("off")

            ax_dapi.text(
                0.00,
                1.06,
                (
                    f"{cid}\n"
                    f"d={row['distance_d_um']:.0f} um | peak={row['ratio_peak']:.3f}\n"
                    f"tb={row['theta_b_deg']:.0f} | tp={row['theta_peak_deg']:.0f} | tm={row['theta_moment_deg']:.0f}"
                ),
                transform=ax_dapi.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
            )
        except Exception as exc:
            for ax in [ax_dapi, ax_psmad]:
                ax.text(
                    0.05, 0.5, f"{cid}\nERROR:\n{exc}", transform=ax.transAxes, fontsize=7
                )
                ax.axis("off")

    total_slots = nrows * ncols
    for idx in range(n, total_slots):
        row_i = idx // ncols
        col_i = idx % ncols
        sub = outer[row_i, col_i].subgridspec(1, 2, wspace=0.05)
        fig.add_subplot(sub[0, 0]).axis("off")
        fig.add_subplot(sub[0, 1]).axis("off")

    fig.suptitle(
        "Channel diagnostics montage (DAPI and pSMAD with bead marker)",
        fontsize=11,
        y=0.995,
    )
    # Nested subgrids in this montage trigger tight_layout warnings; use
    # explicit margins instead so the executed notebook stays clean.
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93)
    plt.show()


def plot_distance_trace_outlier_context(
    cyst_id: str,
    distance_trace_df: pd.DataFrame,
    min_n_cysts_per_bin: int = 5,
    score_row: pd.Series | dict | None = None,
) -> None:
    """Show one cyst's Distance Plot 1 trace in the context of the aggregate trace."""
    df = distance_trace_df.copy()
    df = df[np.isfinite(df["distance_local_um"]) & np.isfinite(df["ratio_mean"])].copy()
    if len(df) == 0:
        print("Distance trace table is empty.")
        return

    cyst_df = df[df["cyst_id"].astype(str) == str(cyst_id)].copy()
    if len(cyst_df) == 0:
        print(f"No distance trace rows found for {cyst_id}.")
        return
    cyst_df = cyst_df.sort_values("distance_local_um").reset_index(drop=True)
    if "scene_index" in cyst_df.columns and len(cyst_df):
        scene_index = int(cyst_df["scene_index"].iloc[0])
    else:
        row = _summary_row_any_or_fail(cyst_id)
        scene_index = int(row["scene_index"])

    agg = aggregate_trace_across_cysts(
        df,
        x_col="distance_local_um",
        y_col="ratio_mean",
    )
    if len(agg) == 0:
        print("Aggregate trace could not be computed.")
        return

    cyst_df = cyst_df.merge(
        agg[["distance_local_um", "mean", "sd", "n_cysts"]],
        on="distance_local_um",
        how="left",
    )
    cyst_df["used_for_score"] = cyst_df["n_cysts"] >= int(min_n_cysts_per_bin)

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.6))

    scene_df = df[df["scene_index"] == scene_index].copy()
    scene_ids = [str(cid) for cid in scene_df["cyst_id"].drop_duplicates().tolist()]
    same_scene_other_ids = [cid for cid in scene_ids if cid != str(cyst_id)]

    def _cyst_number_label(cid: str) -> str:
        tail = str(cid).rsplit("_", 1)[-1]
        try:
            return f"c{int(tail)}"
        except Exception:
            return str(cid)

    scene_other_set = set(same_scene_other_ids)
    gray_label_used = False
    for other_id, sub in df.groupby("cyst_id", sort=False):
        sub = sub.sort_values("distance_local_um")
        other_id = str(other_id)
        if other_id == str(cyst_id) or other_id in scene_other_set:
            continue
        ax.plot(
            sub["distance_local_um"],
            sub["ratio_mean"],
            color="0.75",
            alpha=0.08,
            linewidth=0.9,
            zorder=1,
            label="Other scenes" if not gray_label_used else None,
        )
        gray_label_used = True

    ax.plot(
        agg["distance_local_um"],
        agg["mean"],
        color="tab:blue",
        linewidth=2.4,
        label="Across-cyst mean",
        zorder=3,
    )
    ax.fill_between(
        agg["distance_local_um"],
        agg["mean"] - agg["sd"],
        agg["mean"] + agg["sd"],
        color="tab:blue",
        alpha=0.20,
        label="±SD across cysts",
        zorder=2,
    )

    scene_palette = plt.get_cmap("tab10")
    peer_endpoints = []
    for idx, peer_id in enumerate(same_scene_other_ids):
        sub = scene_df[scene_df["cyst_id"].astype(str) == peer_id].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("distance_local_um")
        color = scene_palette(idx % 10)
        ax.plot(
            sub["distance_local_um"],
            sub["ratio_mean"],
            color=color,
            linewidth=1.7,
            alpha=0.95,
            zorder=4,
            label="Other cysts in same scene" if idx == 0 else None,
        )
        x_end = float(sub["distance_local_um"].iloc[-1])
        y_end = float(sub["ratio_mean"].iloc[-1])
        ax.scatter([x_end], [y_end], s=24, color=color, zorder=5)
        peer_endpoints.append(
            {
                "peer_id": peer_id,
                "label": _cyst_number_label(peer_id),
                "x_end": x_end,
                "y_end": y_end,
                "color": color,
            }
        )

    ax.plot(
        cyst_df["distance_local_um"],
        cyst_df["ratio_mean"],
        color="darkorange",
        linewidth=2.4,
        label=f"Selected outlier ({_cyst_number_label(cyst_id)})",
        zorder=5,
    )

    used = cyst_df["used_for_score"].astype(bool).to_numpy()
    ax.scatter(
        cyst_df.loc[used, "distance_local_um"],
        cyst_df.loc[used, "ratio_mean"],
        s=26,
        color="darkorange",
        edgecolor="black",
        linewidth=0.35,
        label="Bins used for outlier score",
        zorder=6,
    )
    if np.any(~used):
        ax.scatter(
            cyst_df.loc[~used, "distance_local_um"],
            cyst_df.loc[~used, "ratio_mean"],
            s=26,
            facecolors="none",
            edgecolors="darkorange",
            linewidth=1.0,
            label="Bins not used for score",
            zorder=6,
        )

    x_end_out = float(cyst_df["distance_local_um"].iloc[-1])
    y_end_out = float(cyst_df["ratio_mean"].iloc[-1])
    ax.scatter([x_end_out], [y_end_out], s=28, color="darkorange", zorder=7)
    peer_endpoints.append(
        {
            "peer_id": str(cyst_id),
            "label": _cyst_number_label(cyst_id),
            "x_end": x_end_out,
            "y_end": y_end_out,
            "color": "darkorange",
            "weight": "bold",
        }
    )

    if peer_endpoints:
        peer_endpoints = sorted(peer_endpoints, key=lambda d: d["y_end"])
        y_vals = np.array([d["y_end"] for d in peer_endpoints], dtype=float)
        y_min = float(np.nanmin(np.r_[agg["mean"] - agg["sd"], cyst_df["ratio_mean"]]))
        y_max = float(np.nanmax(np.r_[agg["mean"] + agg["sd"], cyst_df["ratio_mean"]]))
        y_span = max(y_max - y_min, 1e-6)
        min_sep = max(0.012 * y_span, 0.004)
        label_y = y_vals.copy()
        for i in range(1, len(label_y)):
            if label_y[i] - label_y[i - 1] < min_sep:
                label_y[i] = label_y[i - 1] + min_sep
        overflow = label_y[-1] - y_max
        if overflow > 0:
            label_y -= overflow
        underflow = y_min - label_y[0]
        if underflow > 0:
            label_y += underflow
        x_min = float(df["distance_local_um"].min())
        x_max = float(df["distance_local_um"].max())
        x_pad = max(0.03 * (x_max - x_min), 20.0)
        ax.set_xlim(x_min, x_max + x_pad)
        for d, y_lab in zip(peer_endpoints, label_y):
            ax.text(
                float(d["x_end"]) + 0.35 * x_pad,
                float(y_lab),
                d["label"],
                color=d["color"],
                fontsize=8,
                fontweight=d.get("weight", "normal"),
                va="center",
                ha="left",
                clip_on=False,
            )

    title = f"Distance Plot 1 context for {cyst_id}"
    if score_row is not None:
        if isinstance(score_row, pd.Series):
            score_row = score_row.to_dict()
        mean_abs_z = score_row.get("mean_trace_abs_z", np.nan)
        rms_z = score_row.get("rms_trace_z", np.nan)
        n_bins = score_row.get("n_trace_bins_compared", np.nan)
        title += (
            f"\nmean abs. scaled deviation={mean_abs_z:.2f} | "
            f"RMS={rms_z:.2f} | bins compared={int(n_bins) if np.isfinite(n_bins) else 'NA'}"
        )
    ax.set_title(title)
    ax.set_xlabel("Local bead-to-pixel distance in full shell (um)")
    ax.set_ylabel("Background-normalized pSMAD/DAPI")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    plt.show()


def build_control_channel_payload(cyst_id: str) -> dict:
    """Build a lightweight debug payload for a no-bead control cyst."""
    row = _summary_row_any_or_fail(cyst_id)
    if bool(row["bead_present"]):
        raise RuntimeError(f"{cyst_id} is bead-present, not a no-bead control")

    scene_index = int(row["scene_index"])
    scene_row = _scene_row_or_fail(scene_index)
    label_id = _control_label_id_from_cyst_id(cyst_id)

    dapi_path = resolve_path(str(row["dapi_image_path"]), ROOT)
    psmad_path = resolve_path(str(row["psmad_image_path"]), ROOT)
    labels_path = resolve_path(str(scene_row["cyst_labels_path"]), ROOT)

    dapi_raw = _cached_load(dapi_path, projection=PROJECTION)
    psmad_raw = _cached_load(psmad_path, projection=PROJECTION)
    _, _, ratio_norm = normalize_two_channels_and_ratio(
        psmad_raw,
        dapi_raw,
        CHANNEL_BG_PARAMS,
        eps=EPS,
    )
    labels = tifffile.imread(labels_path).astype(np.int32)

    mask = labels == int(label_id)
    if int(mask.sum()) == 0:
        raise RuntimeError(
            f"Control label {label_id} not found in scene {scene_index} for {cyst_id}"
        )

    boundary_xy = extract_boundary_xy(mask)
    yy, xx = np.nonzero(mask)
    pad = 60
    y0 = max(0, int(np.min(yy)) - pad)
    y1 = min(mask.shape[0], int(np.max(yy)) + pad + 1)
    x0 = max(0, int(np.min(xx)) - pad)
    x1 = min(mask.shape[1], int(np.max(xx)) + pad + 1)

    return {
        "row": row,
        "mask": mask,
        "boundary_xy": boundary_xy,
        "dapi_raw": dapi_raw,
        "psmad_raw": psmad_raw,
        "ratio_norm": ratio_norm,
        "crop": (x0, x1, y0, y1),
    }


def plot_control_channel_montage(cyst_ids: list[str], ncols: int = 4) -> None:
    """Compact DAPI/pSMAD montage for no-bead control cysts."""
    cyst_ids = [str(c) for c in cyst_ids]
    n = len(cyst_ids)
    if n == 0:
        print("No control cyst IDs provided for montage.")
        return

    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(8.2 * ncols, 4.2 * nrows))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.18, hspace=0.30)

    for idx, cid in enumerate(cyst_ids):
        row_i = idx // ncols
        col_i = idx % ncols
        sub = outer[row_i, col_i].subgridspec(1, 2, wspace=0.05)
        ax_dapi = fig.add_subplot(sub[0, 0])
        ax_psmad = fig.add_subplot(sub[0, 1])
        try:
            p = build_control_channel_payload(cid)
            row = p["row"]
            x0, x1, y0, y1 = p["crop"]

            for ax, img, title, cmap, edge_color in [
                (ax_dapi, p["dapi_raw"], "DAPI", "gray", "cyan"),
                (ax_psmad, p["psmad_raw"], "pSMAD", "magma", "white"),
            ]:
                ax.imshow(_norm_disp(img), cmap=cmap)
                ax.plot(
                    p["boundary_xy"][:, 0],
                    p["boundary_xy"][:, 1],
                    color=edge_color,
                    linewidth=0.8,
                    alpha=0.8,
                )
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)
                ax.set_title(title, fontsize=8)
                ax.axis("off")

            ax_dapi.text(
                0.00,
                1.06,
                (
                    f"{cid}\n"
                    f"peak={row['ratio_peak']:.3f} | mean={row['ratio_mean_shell']:.3f}\n"
                    f"raw max={row['ratio_raw_max']:.3f}"
                ),
                transform=ax_dapi.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
            )
        except Exception as exc:
            for ax in [ax_dapi, ax_psmad]:
                ax.text(
                    0.05, 0.5, f"{cid}\nERROR:\n{exc}", transform=ax.transAxes, fontsize=7
                )
                ax.axis("off")

    total_slots = nrows * ncols
    for idx in range(n, total_slots):
        row_i = idx // ncols
        col_i = idx % ncols
        sub = outer[row_i, col_i].subgridspec(1, 2, wspace=0.05)
        fig.add_subplot(sub[0, 0]).axis("off")
        fig.add_subplot(sub[0, 1]).axis("off")

    fig.suptitle(
        "No-bead control channel montage (DAPI and pSMAD)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    plt.show()


def plot_control_channel_panel(cyst_id: str) -> None:
    """Deeper channel-only panel for a no-bead control outlier."""
    p = build_control_channel_payload(cyst_id)
    row = p["row"]
    x0, x1, y0, y1 = p["crop"]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, img, title, cmap, edge_color in [
        (axes[0], p["dapi_raw"], "DAPI", "gray", "cyan"),
        (axes[1], p["psmad_raw"], "pSMAD", "magma", "white"),
        (
            axes[2],
            np.where(p["mask"], p["ratio_norm"], np.nan),
            "pSMAD/DAPI within mask",
            "viridis",
            "white",
        ),
    ]:
        ax.imshow(_norm_disp(img), cmap=cmap)
        ax.plot(
            p["boundary_xy"][:, 0],
            p["boundary_xy"][:, 1],
            color=edge_color,
            linewidth=0.9,
            alpha=0.85,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        (
            f"No-bead control outlier | {cyst_id} | scene={int(row['scene_index'])} | "
            f"peak={row['ratio_peak']:.3f} | mean={row['ratio_mean_shell']:.3f} | "
            f"raw max={row['ratio_raw_max']:.3f}"
        ),
        fontsize=10,
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    plt.show()


def plot_scene_qc(scene_index: int) -> None:
    """Scene-level channel and labeling diagnostics for one scene."""
    sr = _scene_row_or_fail(scene_index)

    bf = _cached_load(
        resolve_path(str(sr["brightfield_image_path"]), ROOT), projection="first"
    )
    dapi = _cached_load(
        resolve_path(str(sr["dapi_image_path"]), ROOT), projection=PROJECTION
    )
    psmad = _cached_load(
        resolve_path(str(sr["psmad_image_path"]), ROOT), projection=PROJECTION
    )
    bead = _cached_load(
        resolve_path(str(sr["bead_image_path"]), ROOT), projection=PROJECTION
    )
    labels = tifffile.imread(resolve_path(str(sr["cyst_labels_path"]), ROOT)).astype(
        np.int32
    )

    ratio = psmad_over_dapi(psmad, dapi, eps=EPS)
    ratio_display = np.where(labels > 0, ratio, np.nan)
    rows_scene = summary_df[summary_df["scene_index"] == int(scene_index)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    a = axes.ravel()

    a[0].imshow(_norm_disp(bf), cmap="gray")
    a[0].set_title("Brightfield")

    a[1].imshow(_norm_disp(dapi), cmap="gray")
    a[1].set_title("DAPI")

    a[2].imshow(_norm_disp(psmad), cmap="magma")
    a[2].set_title("pSMAD")

    a[3].imshow(_norm_disp(bead), cmap="inferno")
    a[3].set_title("Bead fluorescence (647)")

    a[4].imshow(_norm_disp(ratio_display), cmap="viridis")
    a[4].set_title("Ratio image within cyst labels only")

    a[5].imshow(_norm_disp(bf), cmap="gray")
    a[5].contour(
        labels > 0, levels=[0.5], colors="deepskyblue", linewidths=0.9, alpha=0.9
    )

    for _, rr in rows_scene.iterrows():
        if np.isfinite(rr.get("centroid_x_px", np.nan)) and np.isfinite(
            rr.get("centroid_y_px", np.nan)
        ):
            a[5].text(
                float(rr["centroid_x_px"]),
                float(rr["centroid_y_px"]),
                str(rr["cyst_id"]).split("_")[-1],
                color="yellow",
                fontsize=6,
                ha="center",
                va="center",
            )

    if np.isfinite(sr["bead_well_x_px"]) and np.isfinite(sr["bead_well_y_px"]):
        a[5].scatter(
            [float(sr["bead_well_x_px"])],
            [float(sr["bead_well_y_px"])],
            c="magenta",
            s=90,
            marker="x",
            linewidths=2,
        )

    a[5].set_title("BF + cyst labels + bead-well marker")

    for ax in a:
        ax.axis("off")

    title = (
        f"Scene {int(scene_index):02d} | status={sr['status']} | n_cysts_detected={int(sr['n_cysts_detected'])} | "
        f"bead={'yes' if np.isfinite(sr['bead_well_x_px']) else 'no'}"
    )
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_all_scene_overlays(ncols: int = 6) -> None:
    """Compact montage over all scenes for quick global sanity-check."""
    scene_ids = bead_df["scene_index"].astype(int).tolist()
    n = len(scene_ids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.15 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, scene_index in zip(axes, scene_ids):
        sr = _scene_row_or_fail(scene_index)
        bf = _cached_load(
            resolve_path(str(sr["brightfield_image_path"]), ROOT), projection="first"
        )
        labels = tifffile.imread(
            resolve_path(str(sr["cyst_labels_path"]), ROOT)
        ).astype(np.int32)

        ax.imshow(_norm_disp(bf), cmap="gray")
        ax.contour(
            labels > 0, levels=[0.5], colors="deepskyblue", linewidths=0.6, alpha=0.9
        )

        if np.isfinite(sr["bead_well_x_px"]) and np.isfinite(sr["bead_well_y_px"]):
            ax.scatter(
                [float(sr["bead_well_x_px"])],
                [float(sr["bead_well_y_px"])],
                c="magenta",
                s=28,
                marker="x",
                linewidths=1.2,
            )
            bead_txt = "bead=Y"
        else:
            bead_txt = "bead=N"

        ax.set_title(
            f"S{scene_index:02d} c={int(sr['n_cysts_detected'])} {bead_txt}", fontsize=8
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "All-scene overlay: BF + cyst labels + bead marker", fontsize=11, y=0.995
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.subplots_adjust(hspace=0.16, wspace=0.04)
    plt.show()


def top_fit_outliers(bead_rows: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Rank bead-present cysts by distance residual and orientation mismatch."""
    df = bead_rows.copy()

    # Distance residual from the peak-like amplitude summary.
    x_d = df["distance_d_um"].to_numpy(dtype=float)
    y_d = df["ratio_peak"].to_numpy(dtype=float)
    p_d = np.polyfit(x_d, y_d, deg=1)
    yhat_d = np.polyval(p_d, x_d)
    df["resid_distance"] = y_d - yhat_d

    # Orientation errors for both direction summaries.
    df["theta_peak_error_deg"] = _wrapped_abs_diff_deg(
        df["theta_peak_deg"], df["theta_b_deg"]
    )
    df["theta_moment_error_deg"] = _wrapped_abs_diff_deg(
        df["theta_moment_deg"], df["theta_b_deg"]
    )

    dist_scale = float(np.nanstd(df["resid_distance"], ddof=1)) if len(df) > 1 else 1.0
    if not np.isfinite(dist_scale) or dist_scale <= 0:
        dist_scale = 1.0

    df["abs_resid_combo"] = (
        np.abs(df["resid_distance"]) / dist_scale
        + df["theta_peak_error_deg"] / 180.0
        + df["theta_moment_error_deg"] / 180.0
    )
    return df.sort_values("abs_resid_combo", ascending=False).head(int(n)).copy()


def rank_distance_trace_outliers(
    distance_trace_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    n: int = 6,
    min_n_cysts_per_bin: int = 5,
    sd_floor_quantile: float = 20.0,
) -> pd.DataFrame:
    """Rank bead-present cysts by deviation from the Distance Plot 1 mean +/- SD trace."""
    trace_df = distance_trace_df.copy()
    if len(trace_df) == 0:
        cols = [
            "cyst_id",
            "mean_trace_abs_z",
            "rms_trace_z",
            "max_trace_abs_z",
            "n_trace_bins_compared",
            "mean_trace_signed_dev",
            "trace_sd_floor",
            "trace_min_n_cysts_per_bin",
        ]
        return pd.DataFrame(columns=cols)

    trace_df = trace_df[
        np.isfinite(trace_df["distance_local_um"]) & np.isfinite(trace_df["ratio_mean"])
    ].copy()
    if len(trace_df) == 0:
        return rank_distance_trace_outliers(
            pd.DataFrame(),
            summary_df,
            n=n,
            min_n_cysts_per_bin=min_n_cysts_per_bin,
            sd_floor_quantile=sd_floor_quantile,
        )

    agg = aggregate_trace_across_cysts(
        trace_df,
        x_col="distance_local_um",
        y_col="ratio_mean",
    ).rename(
        columns={
            "mean": "trace_mean",
            "sd": "trace_sd",
            "sem": "trace_sem",
            "n_cysts": "trace_n_cysts",
        }
    )
    if len(agg) == 0:
        return rank_distance_trace_outliers(
            pd.DataFrame(),
            summary_df,
            n=n,
            min_n_cysts_per_bin=min_n_cysts_per_bin,
            sd_floor_quantile=sd_floor_quantile,
        )

    finite_sd = agg["trace_sd"].to_numpy(dtype=float)
    finite_sd = finite_sd[np.isfinite(finite_sd) & (finite_sd > 0)]
    if finite_sd.size == 0:
        sd_floor = 1.0
    else:
        q = float(sd_floor_quantile)
        if not np.isfinite(q):
            q = 20.0
        q = min(max(q, 0.0), 100.0)
        sd_floor = float(np.nanpercentile(finite_sd, q))
        if not np.isfinite(sd_floor) or sd_floor <= 0:
            sd_floor = float(np.nanmedian(finite_sd))
        if not np.isfinite(sd_floor) or sd_floor <= 0:
            sd_floor = 1.0

    agg["trace_sd_eff"] = np.maximum(agg["trace_sd"].fillna(sd_floor), sd_floor)
    merged = trace_df.merge(
        agg[
            [
                "distance_local_um",
                "trace_mean",
                "trace_sd",
                "trace_sem",
                "trace_n_cysts",
                "trace_sd_eff",
            ]
        ],
        on="distance_local_um",
        how="left",
    )
    merged = merged[merged["trace_n_cysts"] >= int(min_n_cysts_per_bin)].copy()
    if len(merged) == 0:
        cols = [
            "cyst_id",
            "mean_trace_abs_z",
            "rms_trace_z",
            "max_trace_abs_z",
            "n_trace_bins_compared",
            "mean_trace_signed_dev",
            "trace_sd_floor",
            "trace_min_n_cysts_per_bin",
        ]
        return pd.DataFrame(columns=cols)

    merged["trace_signed_dev"] = merged["ratio_mean"] - merged["trace_mean"]
    merged["trace_abs_z"] = np.abs(merged["trace_signed_dev"] / merged["trace_sd_eff"])
    merged["trace_sq_z"] = np.square(merged["trace_signed_dev"] / merged["trace_sd_eff"])

    rows = []
    for cyst_id, sub in merged.groupby("cyst_id", sort=False):
        sq = sub["trace_sq_z"].to_numpy(dtype=float)
        rows.append(
            {
                "cyst_id": str(cyst_id),
                "mean_trace_abs_z": float(np.nanmean(sub["trace_abs_z"])),
                "rms_trace_z": float(np.sqrt(np.nanmean(sq))) if sq.size else np.nan,
                "max_trace_abs_z": float(np.nanmax(sub["trace_abs_z"])),
                "n_trace_bins_compared": int(len(sub)),
                "mean_trace_signed_dev": float(np.nanmean(sub["trace_signed_dev"])),
                "trace_sd_floor": float(sd_floor),
                "trace_min_n_cysts_per_bin": int(min_n_cysts_per_bin),
            }
        )

    out = pd.DataFrame(rows)
    summary_cols = ["cyst_id"]
    for col in [
        "scene_index",
        "scene",
        "distance_d_um",
        "ratio_peak",
        "theta_b_deg",
        "review_action",
    ]:
        if col in summary_df.columns:
            summary_cols.append(col)
    out = out.merge(summary_df[summary_cols].drop_duplicates("cyst_id"), on="cyst_id", how="left")
    out = out.sort_values(
        ["mean_trace_abs_z", "rms_trace_z", "max_trace_abs_z", "n_trace_bins_compared"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return out.head(int(n)).copy()


def rank_bead_distant_outliers(
    bead_rows: pd.DataFrame, distance_quantile: float = 0.75
) -> pd.DataFrame:
    """Rank bead-distant cysts by positive distance residual to screen for ectopic pSMAD."""
    df = bead_rows.copy()
    if len(df) == 0:
        out = df.head(0).copy()
        out["distance_far_cut_um"] = np.nan
        out["pred_ratio_peak"] = np.nan
        out["resid_distance"] = np.nan
        out["theta_peak_error_deg"] = np.nan
        out["theta_moment_error_deg"] = np.nan
        return out

    distance_quantile = float(distance_quantile)
    if not np.isfinite(distance_quantile):
        distance_quantile = 0.75
    distance_quantile = min(max(distance_quantile, 0.0), 1.0)

    x_d = df["distance_d_um"].to_numpy(dtype=float)
    y_d = df["ratio_peak"].to_numpy(dtype=float)
    p_d = np.polyfit(x_d, y_d, deg=1)
    yhat_d = np.polyval(p_d, x_d)
    df["pred_ratio_peak"] = yhat_d
    df["resid_distance"] = y_d - yhat_d
    df["theta_peak_error_deg"] = _wrapped_abs_diff_deg(
        df["theta_peak_deg"], df["theta_b_deg"]
    )
    df["theta_moment_error_deg"] = _wrapped_abs_diff_deg(
        df["theta_moment_deg"], df["theta_b_deg"]
    )

    far_cut = float(df["distance_d_um"].quantile(distance_quantile))
    far_df = df[df["distance_d_um"] >= far_cut].copy()
    far_df["distance_far_cut_um"] = far_cut
    far_df["distance_far_quantile"] = distance_quantile
    return far_df.sort_values(
        ["resid_distance", "ratio_peak", "distance_d_um"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def rank_no_bead_control_outliers(control_rows: pd.DataFrame) -> pd.DataFrame:
    """Rank no-bead control cysts by elevated response summaries."""
    df = control_rows.copy()
    if len(df) == 0:
        return df
    return df.sort_values(
        ["ratio_peak", "ratio_mean_shell", "ratio_raw_max"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _xy_bool_to_mask(shape: tuple[int, int], xy: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Rasterize a boolean keep-vector on XY pixel coordinates back to an image mask."""
    out = np.zeros(shape, dtype=bool)
    if xy.size == 0:
        return out
    keep = np.asarray(keep, dtype=bool)
    xx = np.clip(np.round(xy[keep, 0]).astype(int), 0, shape[1] - 1)
    yy = np.clip(np.round(xy[keep, 1]).astype(int), 0, shape[0] - 1)
    out[yy, xx] = True
    return out


def _outside_ring_mask(mask: np.ndarray, ring_width_px: int) -> np.ndarray:
    """Local outside-of-cyst ring used only for raw DAPI background comparison."""
    ring_width_px = max(1, int(round(ring_width_px)))
    outside = ~mask.astype(bool)
    dist_out = ndi.distance_transform_edt(outside)
    return outside & (dist_out <= ring_width_px)


def _binned_profile_same_edges(x: np.ndarray, y: np.ndarray, bin_width: float) -> dict:
    """Build the exact binned mean line used for a per-cyst distance diagnostic plot."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) == 0:
        return {
            'edges': np.array([], dtype=np.float64),
            'centers': np.array([], dtype=np.float64),
            'means': np.array([], dtype=np.float64),
            'counts': np.array([], dtype=int),
            'bin_idx': np.array([], dtype=int),
            'keep_mask': keep,
        }

    lo = float(np.floor(np.nanmin(x) / bin_width) * bin_width)
    hi = float(np.ceil(np.nanmax(x) / bin_width) * bin_width + bin_width)
    edges = np.arange(lo, hi + 0.5 * bin_width, bin_width, dtype=np.float64)
    n_bins = max(0, len(edges) - 1)
    centers = edges[:-1] + 0.5 * bin_width
    bin_idx = np.digitize(x, edges) - 1

    means = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        vals = y[bin_idx == i]
        if len(vals):
            means[i] = float(np.nanmean(vals))
            counts[i] = int(len(vals))

    return {
        'edges': edges,
        'centers': centers,
        'means': means,
        'counts': counts,
        'bin_idx': bin_idx,
        'keep_mask': keep,
    }


def build_problem_pixel_payload(payload: dict) -> dict:
    """
    Identify the exact shell pixels that create the far-cyst edge uplift in the binned
    full-shell distance plot.

    This is the diagnostic needed to distinguish:
    1. loose masks that include background-like pixels, versus
    2. true cyst pixels whose DAPI denominator still distorts the ratio.
    """
    mask = payload['mask']
    shell_xy = payload['shell_xy']
    x = payload['local_distance_shell_um']
    ratio = payload['ratio_shell']
    dapi = payload['dapi_shell']
    psmad = payload['psmad_shell']

    ratio_prof = _binned_profile_same_edges(x, ratio, DISTANCE_TRACE_BIN_UM)
    dapi_prof = _binned_profile_same_edges(x, dapi, DISTANCE_TRACE_BIN_UM)
    psmad_prof = _binned_profile_same_edges(x, psmad, DISTANCE_TRACE_BIN_UM)

    valid_bins = np.where(np.isfinite(ratio_prof['means']))[0]
    if len(valid_bins) < 12:
        raise RuntimeError('Not enough valid bins for problem-pixel diagnostic')

    edge_n = max(3, int(round(PROBLEM_EDGE_BIN_FRACTION * len(valid_bins))))
    mid_n = max(5, int(round(PROBLEM_MID_BIN_FRACTION * len(valid_bins))))
    edge_n = min(edge_n, max(1, len(valid_bins) // 4))
    mid_n = min(mid_n, max(3, len(valid_bins) - 2 * edge_n))

    left_bins = valid_bins[:edge_n]
    right_bins = valid_bins[-edge_n:]
    mid_start = (len(valid_bins) - mid_n) // 2
    mid_bins = valid_bins[mid_start : mid_start + mid_n]
    mid_ratio = float(np.nanmean(ratio_prof['means'][mid_bins]))

    left_problem_bins = left_bins[ratio_prof['means'][left_bins] > mid_ratio]
    right_problem_bins = right_bins[ratio_prof['means'][right_bins] > mid_ratio]

    keep_mask = ratio_prof['keep_mask']
    shell_xy_valid = shell_xy[keep_mask]
    left_keep = np.isin(ratio_prof['bin_idx'], left_problem_bins)
    right_keep = np.isin(ratio_prof['bin_idx'], right_problem_bins)
    mid_keep = np.isin(ratio_prof['bin_idx'], mid_bins)

    left_mask = _xy_bool_to_mask(mask.shape, shell_xy_valid, left_keep)
    right_mask = _xy_bool_to_mask(mask.shape, shell_xy_valid, right_keep)
    mid_mask = _xy_bool_to_mask(mask.shape, shell_xy_valid, mid_keep)
    bg_ring_mask = _outside_ring_mask(mask, PROBLEM_BACKGROUND_RING_PX)

    dapi_left = dapi_prof_vals_left = dapi[keep_mask][left_keep]
    dapi_right = dapi[keep_mask][right_keep]
    dapi_mid = dapi[keep_mask][mid_keep]
    dapi_bg = payload['dapi'][bg_ring_mask]

    psmad_left = psmad[keep_mask][left_keep]
    psmad_right = psmad[keep_mask][right_keep]
    psmad_mid = psmad[keep_mask][mid_keep]

    ratio_left = ratio[keep_mask][left_keep]
    ratio_right = ratio[keep_mask][right_keep]
    ratio_mid = ratio[keep_mask][mid_keep]
    mid_ratio_raw = float(np.nanmean(ratio_mid))
    left_mean_raw = float(np.nanmean(ratio_left)) if len(ratio_left) else np.nan
    right_mean_raw = float(np.nanmean(ratio_right)) if len(ratio_right) else np.nan
    left_uplift = (
        max(left_mean_raw - mid_ratio_raw, 0.0)
        if np.isfinite(left_mean_raw) and np.isfinite(mid_ratio_raw)
        else 0.0
    )
    right_uplift = (
        max(right_mean_raw - mid_ratio_raw, 0.0)
        if np.isfinite(right_mean_raw) and np.isfinite(mid_ratio_raw)
        else 0.0
    )

    return {
        'ratio_profile': ratio_prof,
        'dapi_profile': dapi_prof,
        'psmad_profile': psmad_prof,
        'left_problem_bins': left_problem_bins,
        'right_problem_bins': right_problem_bins,
        'mid_bins': mid_bins,
        'left_mask': left_mask,
        'right_mask': right_mask,
        'mid_mask': mid_mask,
        'background_ring_mask': bg_ring_mask,
        'dapi_left': dapi_left,
        'dapi_right': dapi_right,
        'dapi_mid': dapi_mid,
        'dapi_bg': dapi_bg,
        'psmad_left': psmad_left,
        'psmad_right': psmad_right,
        'psmad_mid': psmad_mid,
        'ratio_left': ratio_left,
        'ratio_right': ratio_right,
        'ratio_mid': ratio_mid,
        'mid_ratio': mid_ratio,
        'distance_support_df': payload['distance_support_df'].copy(),
        'distance_support_min_pixels': float(payload['row']['distance_support_min_pixels']),
        'distance_support_window_bins': int(payload['row']['distance_support_window_bins']),
        'edge_uplift': float(left_uplift + right_uplift),
        'left_minus_mid': float(left_mean_raw - mid_ratio_raw) if np.isfinite(left_mean_raw) and np.isfinite(mid_ratio_raw) else np.nan,
        'right_minus_mid': float(right_mean_raw - mid_ratio_raw) if np.isfinite(right_mean_raw) and np.isfinite(mid_ratio_raw) else np.nan,
    }


def compute_non_edge_support_floor(
    distance_trace_df: pd.DataFrame,
    edge_bin_fraction: float = 0.10,
) -> tuple[float, pd.DataFrame]:
    """
    Compute the strict global count-support floor that protects every occupied non-edge bin.

    Non-edge bins are defined as all occupied distance bins excluding the low/high edge blocks.
    """
    bead_df_local = distance_trace_df.loc[
        distance_trace_df["bead_present"].astype(bool)
    ].copy()
    rows = []
    for cyst_id, g in bead_df_local.groupby("cyst_id"):
        g = g.sort_values("distance_local_um").reset_index(drop=True)
        occupied = np.flatnonzero(g["n_pixels"].to_numpy(dtype=float) > 0)
        if len(occupied) < 3:
            continue
        edge_n = max(3, int(round(float(edge_bin_fraction) * len(occupied))))
        edge_n = min(edge_n, max(1, len(occupied) // 4))
        if len(occupied) - 2 * edge_n <= 0:
            continue

        interior = occupied[edge_n : len(occupied) - edge_n]
        for order_rank, idx in enumerate(interior):
            r = g.iloc[int(idx)]
            rows.append(
                {
                    "cyst_id": str(cyst_id),
                    "scene_index": int(r["scene_index"]),
                    "scene": str(r["scene"]),
                    "distance_local_um": float(r["distance_local_um"]),
                    "n_pixels": int(r["n_pixels"]),
                    "count_support_n_pixels": float(r["count_support_n_pixels"]),
                    "ratio_mean": float(r["ratio_mean"]),
                    "n_occupied_bins": int(len(occupied)),
                    "edge_n": int(edge_n),
                    "interior_rank": int(order_rank),
                }
            )

    details_df = pd.DataFrame(rows).sort_values(
        ["count_support_n_pixels", "n_pixels", "scene_index", "cyst_id", "distance_local_um"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    if len(details_df) == 0:
        return np.nan, details_df
    return float(details_df["count_support_n_pixels"].min()), details_df


def build_support_floor_example_payload(
    payload: dict,
    distance_local_um: float,
    support_floor: float,
    edge_bin_fraction: float = 0.10,
) -> dict:
    """Build a diagnostic payload for one exact interior bin that limits the strict support floor."""
    mask = payload["mask"]
    shell_xy = payload["shell_xy"]
    local_distance_shell_um = payload["local_distance_shell_um"]
    support_df = payload["distance_support_df"].loc[
        payload["distance_support_df"]["n_pixels"] > 0
    ].copy().sort_values("distance_local_um").reset_index(drop=True)
    if len(support_df) == 0:
        raise RuntimeError("No occupied support bins available for support-floor example")

    target_idx = int(
        np.argmin(
            np.abs(
                support_df["distance_local_um"].to_numpy(dtype=np.float64)
                - float(distance_local_um)
            )
        )
    )
    target_row = support_df.iloc[target_idx]
    target_center = float(target_row["distance_local_um"])
    half_bin = float(DISTANCE_TRACE_BIN_UM) / 2.0

    lo = target_center - half_bin
    hi = target_center + half_bin
    target_keep = (local_distance_shell_um >= lo) & (local_distance_shell_um < hi)
    target_mask = xy_bool_to_mask(mask.shape, shell_xy, target_keep)

    occupied = np.flatnonzero(support_df["n_pixels"].to_numpy(dtype=float) > 0)
    edge_n = max(3, int(round(float(edge_bin_fraction) * len(occupied))))
    edge_n = min(edge_n, max(1, len(occupied) // 4))
    left_edge_idx = occupied[:edge_n]
    right_edge_idx = occupied[-edge_n:]

    return {
        "support_df": support_df,
        "target_center_um": target_center,
        "target_row": target_row,
        "target_mask": target_mask,
        "target_n_pixels": int(target_row["n_pixels"]),
        "target_support_n_pixels": float(target_row["count_support_n_pixels"]),
        "left_edge_centers_um": support_df.iloc[left_edge_idx]["distance_local_um"].to_numpy(dtype=np.float64),
        "right_edge_centers_um": support_df.iloc[right_edge_idx]["distance_local_um"].to_numpy(dtype=np.float64),
        "edge_n": int(edge_n),
        "support_floor": float(support_floor),
    }


def plot_support_floor_example_payload(payload: dict, example: dict) -> None:
    """Visualize one low-support interior bin that determines the strict global floor."""
    row = payload["row"]
    bf = payload["bf"]
    dapi = payload["dapi"]
    boundary_xy = payload["boundary_xy"]
    target_mask = example["target_mask"]
    support_df = example["support_df"]
    target_center = float(example["target_center_um"])
    target_n = int(example["target_n_pixels"])
    target_support = float(example["target_support_n_pixels"])
    support_floor = float(example["support_floor"])

    target_overlay = np.ma.masked_where(~target_mask, target_mask.astype(float))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, img, title, cmap in [
        (axes[0], bf, "Brightfield + exact limiting bin", "gray"),
        (axes[1], dapi, "DAPI raw + exact limiting bin", "gray"),
    ]:
        ax.imshow(_norm_disp(img), cmap=cmap)
        ax.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="white", linewidth=0.8, alpha=0.7)
        ax.imshow(target_overlay, cmap="autumn", alpha=0.55, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    ax = axes[2]
    ax.plot(
        support_df["distance_local_um"],
        support_df["n_pixels"],
        color="0.7",
        linewidth=1.6,
        label="per-bin pixel count",
    )
    ax.plot(
        support_df["distance_local_um"],
        support_df["count_support_n_pixels"],
        color="black",
        linewidth=2.1,
        label=f"{int(DISTANCE_SUPPORT_WINDOW_BINS)}-bin aggregated count support",
    )
    half_bin = float(DISTANCE_TRACE_BIN_UM) / 2.0
    for c in example["left_edge_centers_um"]:
        ax.axvspan(c - half_bin, c + half_bin, color="0.9", alpha=0.8)
    for c in example["right_edge_centers_um"]:
        ax.axvspan(c - half_bin, c + half_bin, color="0.9", alpha=0.8)
    ax.axvline(target_center, color="magenta", linestyle="--", linewidth=1.8, label="exact limiting bin")
    ax.scatter(
        [target_center],
        [target_n],
        color="magenta",
        s=60,
        zorder=4,
    )
    ax.scatter(
        [target_center],
        [target_support],
        color="magenta",
        s=60,
        zorder=4,
    )
    ax.axhline(
        support_floor,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"strict no-interior-loss floor = {support_floor:.1f}",
    )
    ax.set_xlabel("Local bead-to-pixel distance on full shell (um)")
    ax.set_ylabel("Pixel count per 10 um bin")
    ax.set_title("Count-support trace with limiting interior bin")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Support-floor calibration | {row['cyst_id']} | scene={int(row['scene_index'])} | "
        f"target bin={target_center:.1f} um | n={target_n} | agg support={target_support:.1f}",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def build_distance_bin_geometry_payload(payload: dict) -> dict:
    """Build shell-distance bin geometry overlays for one cyst."""
    mask = payload["mask"]
    shell_xy = payload["shell_xy"]
    local_distance_shell_um = payload["local_distance_shell_um"]
    support_df = payload["distance_support_df"].loc[
        payload["distance_support_df"]["n_pixels"] > 0
    ].copy().sort_values("distance_local_um").reset_index(drop=True)
    if len(support_df) == 0:
        raise RuntimeError("No occupied distance bins available for geometry snapshot")

    dist_max = max(
        DISTANCE_TRACE_MAX_UM,
        float(np.nanmax(local_distance_shell_um) + DISTANCE_TRACE_BIN_UM),
    )
    dist_edges = np.arange(
        DISTANCE_TRACE_MIN_UM,
        dist_max + DISTANCE_TRACE_BIN_UM,
        DISTANCE_TRACE_BIN_UM,
    )
    pixel_bin_idx = np.digitize(local_distance_shell_um, dist_edges, right=False) - 1
    pixel_bin_idx = np.clip(pixel_bin_idx, 0, len(dist_edges) - 2)
    pixel_bin_centers = 0.5 * (dist_edges[pixel_bin_idx] + dist_edges[pixel_bin_idx + 1])
    pixel_bin_img = _xy_values_to_image(mask.shape, shell_xy, pixel_bin_centers)

    occupied_centers = support_df["distance_local_um"].to_numpy(dtype=np.float64)
    bin_lo = occupied_centers - 0.5 * float(DISTANCE_TRACE_BIN_UM)
    bin_hi = occupied_centers + 0.5 * float(DISTANCE_TRACE_BIN_UM)
    bin_labels = [f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(bin_lo, bin_hi)]
    bin_center_to_rank = {float(c): i for i, c in enumerate(occupied_centers.tolist())}
    support_rank = np.array(
        [bin_center_to_rank[float(c)] for c in occupied_centers], dtype=np.int32
    )
    pixel_bin_rank = np.array(
        [bin_center_to_rank.get(float(c), -1) for c in pixel_bin_centers], dtype=np.int32
    )
    pixel_rank_img = _xy_values_to_image(mask.shape, shell_xy, pixel_bin_rank)

    return {
        "support_df": support_df,
        "dist_edges": dist_edges,
        "pixel_bin_centers": pixel_bin_centers,
        "pixel_bin_img": pixel_bin_img,
        "pixel_rank_img": pixel_rank_img,
        "occupied_centers": occupied_centers,
        "bin_lo": bin_lo,
        "bin_hi": bin_hi,
        "bin_labels": bin_labels,
        "support_rank": support_rank,
    }


def plot_distance_bin_geometry_payload(payload: dict, geometry: dict) -> None:
    """Show how exact 10 um shell-distance bins tile the cyst geometry."""
    row = payload["row"]
    bf = payload["bf"]
    mask = payload["mask"]
    boundary_xy = payload["boundary_xy"]
    bead_xy = payload["bead_xy"]
    centroid_xy = payload["centroid_xy"]

    support_df = geometry["support_df"]
    occupied_centers = geometry["occupied_centers"]
    pixel_bin_img = geometry["pixel_bin_img"]
    pixel_rank_img = geometry["pixel_rank_img"]
    bin_lo = geometry["bin_lo"]
    bin_hi = geometry["bin_hi"]
    bin_labels = geometry["bin_labels"]
    support_rank = geometry["support_rank"]

    rank_overlay = np.ma.masked_invalid(pixel_rank_img)
    boundary_overlay = np.ma.masked_where(
        ~_discrete_region_boundary_mask(pixel_rank_img),
        np.ones_like(pixel_rank_img, dtype=np.float32),
    )
    cyst_overlay = np.ma.masked_where(~mask, mask.astype(float))
    x0, x1, y1, y0 = _mask_crop_bounds(mask, pad=10)

    cmap = plt.get_cmap("turbo", max(int(len(occupied_centers)), 2))
    rank_norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(occupied_centers) + 0.5, 1.0),
        cmap.N,
    )

    fig = plt.figure(figsize=(21.8, 5.8))
    gs = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.0, 1.05, 0.07, 0.18, 1.45],
        wspace=0.28,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    spacer_ax = fig.add_subplot(gs[0, 3])
    ax2 = fig.add_subplot(gs[0, 4])
    spacer_ax.axis("off")

    ax0.imshow(_norm_disp(bf), cmap="gray")
    ax0.imshow(rank_overlay, cmap=cmap, norm=rank_norm, alpha=0.90, interpolation="nearest")
    ax0.imshow(boundary_overlay, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.95, interpolation="nearest")
    ax0.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="white", linewidth=0.8, alpha=0.7)
    ax0.plot(
        [centroid_xy[0], bead_xy[0]],
        [centroid_xy[1], bead_xy[1]],
        color="red",
        linewidth=1.2,
        alpha=0.9,
    )
    ax0.scatter([centroid_xy[0]], [centroid_xy[1]], c="white", s=18)
    ax0.scatter([bead_xy[0]], [bead_xy[1]], c="magenta", s=55, marker="x")
    ax0.set_title("Brightfield + shell pixels colored by discrete occupied 10 um bins")
    ax0.axis("off")

    ax1.imshow(cyst_overlay, cmap="gray", alpha=0.25, interpolation="nearest")
    ax1.imshow(rank_overlay, cmap=cmap, norm=rank_norm, alpha=0.95, interpolation="nearest")
    ax1.imshow(boundary_overlay, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.95, interpolation="nearest")
    ax1.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="black", linewidth=0.8, alpha=0.8)
    ax1.plot(
        [centroid_xy[0], bead_xy[0]],
        [centroid_xy[1], bead_xy[1]],
        color="red",
        linewidth=1.2,
        alpha=0.9,
    )
    ax1.scatter([centroid_xy[0]], [centroid_xy[1]], c="black", s=18)
    ax1.scatter([bead_xy[0]], [bead_xy[1]], c="magenta", s=55, marker="x")
    ax1.set_title("Same shell on plain cyst mask (one categorical color per occupied bin)")
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y1, y0)
    ax1.axis("off")

    bar_colors = cmap(rank_norm(support_rank))
    widths = np.maximum(bin_hi - bin_lo, 0.85 * float(DISTANCE_TRACE_BIN_UM))
    ax2.bar(
        support_df["distance_local_um"],
        support_df["n_pixels"],
        width=widths,
        color=bar_colors,
        edgecolor="none",
        align="center",
    )
    ax2.plot(
        support_df["distance_local_um"],
        support_df["n_pixels"],
        color="black",
        linewidth=1.0,
        alpha=0.55,
    )
    ax2.set_xlabel("Local bead-to-pixel distance on full shell (um)")
    ax2.set_ylabel("Shell pixels in exact 10 um bin")
    ax2.yaxis.labelpad = 20
    ax2.set_title("Same bins shown as the shell-pixel count histogram")

    mappable = plt.cm.ScalarMappable(norm=rank_norm, cmap=cmap)
    mappable.set_array([])
    tick_idx = np.linspace(0, len(occupied_centers) - 1, min(6, len(occupied_centers)), dtype=int)
    tick_idx = np.unique(tick_idx)
    cbar = fig.colorbar(mappable, cax=cax, ticks=tick_idx)
    cbar.ax.set_yticklabels([bin_labels[i] for i in tick_idx])
    cbar.set_label("Occupied 10 um distance-bin range (um)")

    fig.suptitle(
        f"Distance-bin geometry snapshot | {row['cyst_id']} | scene={int(row['scene_index'])} | "
        f"shell-only distance histogram over {len(occupied_centers)} occupied bins",
        y=0.98,
    )
    fig.subplots_adjust(top=0.87, left=0.03, right=0.985, bottom=0.12)
    plt.show()


def plot_problem_pixel_payload(payload: dict, problem: dict) -> None:
    """Show where the edge-uplift pixels live and compare them against raw DAPI background."""
    row = payload['row']
    bf = payload['bf']
    dapi = payload['dapi']
    psmad = payload['psmad']
    ratio = payload['ratio']
    mask = payload['mask']
    boundary_xy = payload['boundary_xy']

    left_mask = problem['left_mask']
    right_mask = problem['right_mask']
    mid_mask = problem['mid_mask']
    bg_ring_mask = problem['background_ring_mask']
    support_df = problem['distance_support_df'].loc[problem['distance_support_df']['n_pixels'] > 0].copy()
    count_floor = float(problem['distance_support_min_pixels'])
    win_bins = int(problem['distance_support_window_bins'])

    ratio_masked = np.where(mask, ratio, np.nan)

    fig, axes = plt.subplots(2, 4, figsize=(21, 10.5))
    axes = axes.ravel()

    def _draw_problem_contours(ax):
        ax.plot(boundary_xy[:, 0], boundary_xy[:, 1], color='white', linewidth=0.8, alpha=0.7)
        if np.any(bg_ring_mask):
            ax.contour(
                bg_ring_mask.astype(float),
                levels=[0.5],
                colors=['tab:blue'],
                linewidths=1.0,
                linestyles='--',
            )
        if np.any(mid_mask):
            ax.contour(
                mid_mask.astype(float),
                levels=[0.5],
                colors=['limegreen'],
                linewidths=1.2,
            )
        if np.any(left_mask):
            ax.contour(
                left_mask.astype(float),
                levels=[0.5],
                colors=['orange'],
                linewidths=1.5,
            )
        if np.any(right_mask):
            ax.contour(
                right_mask.astype(float),
                levels=[0.5],
                colors=['red'],
                linewidths=1.5,
            )

    for ax, img, title, cmap in [
        (axes[0], bf, 'Brightfield + candidate problem pixels', 'gray'),
        (axes[1], dapi, 'DAPI raw + candidate problem pixels', 'gray'),
        (axes[2], psmad, 'pSMAD raw + candidate problem pixels', 'magma'),
        (axes[3], ratio_masked, 'pSMAD/DAPI within cyst mask only', 'viridis'),
    ]:
        ax.imshow(_norm_disp(img), cmap=cmap)
        _draw_problem_contours(ax)
        ax.set_title(title)
        ax.axis('off')

    ax = axes[4]
    prof = problem['dapi_profile']
    centers = prof['centers']
    means = prof['means']
    ax.plot(centers, means, color='tab:blue', linewidth=2.2)
    if len(problem['left_problem_bins']):
        ax.scatter(centers[problem['left_problem_bins']], means[problem['left_problem_bins']], color='orange', s=45, zorder=3, label='left problem bins')
    if len(problem['right_problem_bins']):
        ax.scatter(centers[problem['right_problem_bins']], means[problem['right_problem_bins']], color='red', s=45, zorder=3, label='right problem bins')
    ax.scatter(centers[problem['mid_bins']], means[problem['mid_bins']], color='green', s=30, zorder=3, label='mid bins')
    if len(support_df):
        half_bin = float(DISTANCE_TRACE_BIN_UM) / 2.0
        for _, tr in support_df.loc[~support_df['keep_bin']].iterrows():
            ax.axvspan(
                float(tr['distance_local_um']) - half_bin,
                float(tr['distance_local_um']) + half_bin,
                color='red',
                alpha=0.10,
            )
    ax.set_xlabel('Local bead-to-pixel distance on full shell (um)')
    ax.set_ylabel('Raw DAPI intensity (a.u.)')
    ax.set_title('DAPI vs local distance')
    ax.legend(loc='best', fontsize=8)

    ax = axes[5]
    if len(support_df):
        ax.plot(
            support_df['distance_local_um'],
            support_df['n_pixels'],
            color='0.7',
            linewidth=1.8,
            label='per-bin pixel count',
        )
        ax.plot(
            support_df['distance_local_um'],
            support_df['count_support_n_pixels'],
            color='black',
            linewidth=2.2,
            label=f'{win_bins}-bin aggregated count support',
        )
        half_bin = float(DISTANCE_TRACE_BIN_UM) / 2.0
        for _, tr in support_df.loc[~support_df['keep_bin']].iterrows():
            ax.axvspan(
                float(tr['distance_local_um']) - half_bin,
                float(tr['distance_local_um']) + half_bin,
                color='red',
                alpha=0.12,
            )
        support_centers = support_df['distance_local_um'].to_numpy(dtype=np.float64)
        support_counts = support_df['n_pixels'].to_numpy(dtype=np.float64)
        left_bins = np.asarray(problem['left_problem_bins'], dtype=int)
        right_bins = np.asarray(problem['right_problem_bins'], dtype=int)
        mid_bins = np.asarray(problem['mid_bins'], dtype=int)
        if len(left_bins):
            valid = left_bins[left_bins < len(support_df)]
            ax.scatter(
                support_centers[valid],
                support_counts[valid],
                color='orange',
                s=45,
                zorder=3,
                label='left problem bins',
            )
        if len(right_bins):
            valid = right_bins[right_bins < len(support_df)]
            ax.scatter(
                support_centers[valid],
                support_counts[valid],
                color='red',
                s=45,
                zorder=3,
                label='right problem bins',
            )
        valid = mid_bins[mid_bins < len(support_df)]
        ax.scatter(
            support_centers[valid],
            support_counts[valid],
            color='green',
            s=30,
            zorder=3,
            label='mid bins',
        )
    ax.axhline(count_floor, color='red', linestyle='--', linewidth=1.8, label=f'active min count = {count_floor:.0f}')
    ax.set_xlabel('Local bead-to-pixel distance on full shell (um)')
    ax.set_ylabel('Pixel count per 10 um bin')
    ax.set_title('Distance-bin pixel-count support + active edge-trim rule')
    ax.legend(loc='best', fontsize=8)

    for ax, prof, title, color in [
        (axes[6], problem['psmad_profile'], 'pSMAD vs local distance', 'tab:orange'),
        (axes[7], problem['ratio_profile'], 'Ratio vs local distance', 'tab:green'),
    ]:
        centers = prof['centers']
        means = prof['means']
        ax.plot(centers, means, color=color, linewidth=2.2)
        if len(problem['left_problem_bins']):
            ax.scatter(centers[problem['left_problem_bins']], means[problem['left_problem_bins']], color='orange', s=45, zorder=3, label='left problem bins')
        if len(problem['right_problem_bins']):
            ax.scatter(centers[problem['right_problem_bins']], means[problem['right_problem_bins']], color='red', s=45, zorder=3, label='right problem bins')
        ax.scatter(centers[problem['mid_bins']], means[problem['mid_bins']], color='green', s=30, zorder=3, label='mid bins')
        if len(support_df):
            half_bin = float(DISTANCE_TRACE_BIN_UM) / 2.0
            for _, tr in support_df.loc[~support_df['keep_bin']].iterrows():
                ax.axvspan(
                    float(tr['distance_local_um']) - half_bin,
                    float(tr['distance_local_um']) + half_bin,
                    color='red',
                    alpha=0.10,
                )
        ax.set_xlabel('Local bead-to-pixel distance on full shell (um)')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)

    fig.suptitle(
        f"Problem-pixel diagnostic | {row['cyst_id']} | scene={int(row['scene_index'])} | "
        f"d={float(row['distance_d_um']):.1f} um | edge uplift={problem['edge_uplift']:.3f} | "
        f"edge trim uses {win_bins}-bin count support and minimum count {count_floor:.0f}",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def build_orientation_bin_geometry_payload(payload: dict) -> dict:
    """Build shell-orientation bin geometry overlays for one cyst."""
    mask = payload["mask"]
    shell_xy = payload["shell_xy"]
    theta_rel_bead_shell_deg = payload["theta_rel_bead_shell_deg"]

    theta_edges = np.arange(
        -180.0,
        180.0 + ORIENTATION_TRACE_BIN_DEG,
        ORIENTATION_TRACE_BIN_DEG,
    )
    pixel_bin_idx = np.digitize(theta_rel_bead_shell_deg, theta_edges, right=False) - 1
    pixel_bin_idx = np.clip(pixel_bin_idx, 0, len(theta_edges) - 2)
    pixel_bin_centers = 0.5 * (theta_edges[pixel_bin_idx] + theta_edges[pixel_bin_idx + 1])
    pixel_bin_img = _xy_values_to_image(mask.shape, shell_xy, pixel_bin_centers)

    n_bins = len(theta_edges) - 1
    counts = np.bincount(pixel_bin_idx, minlength=n_bins)
    occupied_idx = np.flatnonzero(counts > 0)
    if len(occupied_idx) == 0:
        raise RuntimeError("No occupied orientation bins available for geometry snapshot")

    occupied_centers = 0.5 * (theta_edges[occupied_idx] + theta_edges[occupied_idx + 1])
    bin_lo = theta_edges[occupied_idx]
    bin_hi = theta_edges[occupied_idx + 1]
    bin_labels = [f"{lo:.0f} to {hi:.0f}" for lo, hi in zip(bin_lo, bin_hi)]

    bin_idx_to_rank = {int(idx): rank for rank, idx in enumerate(occupied_idx.tolist())}
    support_rank = np.array([bin_idx_to_rank[int(i)] for i in occupied_idx], dtype=np.int32)
    pixel_rank = np.array([bin_idx_to_rank[int(i)] for i in pixel_bin_idx], dtype=np.int32)
    pixel_rank_img = _xy_values_to_image(mask.shape, shell_xy, pixel_rank)

    support_df = pd.DataFrame(
        {
            "theta_rel_bead_deg": occupied_centers,
            "n_pixels": counts[occupied_idx].astype(int),
            "bin_lo_deg": bin_lo,
            "bin_hi_deg": bin_hi,
            "bin_idx": occupied_idx.astype(int),
            "bin_rank": support_rank,
            "bin_label": bin_labels,
        }
    )

    return {
        "theta_edges": theta_edges,
        "support_df": support_df,
        "pixel_bin_centers": pixel_bin_centers,
        "pixel_bin_img": pixel_bin_img,
        "pixel_rank_img": pixel_rank_img,
        "occupied_centers": occupied_centers,
        "bin_lo": bin_lo,
        "bin_hi": bin_hi,
        "bin_labels": bin_labels,
        "support_rank": support_rank,
    }


def plot_orientation_bin_geometry_payload(payload: dict, geometry: dict) -> None:
    """Show how exact 5 degree shell-orientation bins tile the cyst geometry."""
    row = payload["row"]
    bf = payload["bf"]
    mask = payload["mask"]
    boundary_xy = payload["boundary_xy"]
    bead_xy = payload["bead_xy"]
    centroid_xy = payload["centroid_xy"]

    support_df = geometry["support_df"]
    occupied_centers = geometry["occupied_centers"]
    pixel_rank_img = geometry["pixel_rank_img"]
    bin_lo = geometry["bin_lo"]
    bin_hi = geometry["bin_hi"]
    bin_labels = geometry["bin_labels"]
    support_rank = geometry["support_rank"]

    rank_overlay = np.ma.masked_invalid(pixel_rank_img)
    boundary_overlay = np.ma.masked_where(
        ~_discrete_region_boundary_mask(pixel_rank_img),
        np.ones_like(pixel_rank_img, dtype=np.float32),
    )
    cyst_overlay = np.ma.masked_where(~mask, mask.astype(float))
    x0, x1, y1, y0 = _mask_crop_bounds(mask, pad=10)

    cmap = plt.get_cmap("twilight_shifted", max(int(len(occupied_centers)), 2))
    rank_norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(occupied_centers) + 0.5, 1.0),
        cmap.N,
    )

    fig = plt.figure(figsize=(21.8, 5.8))
    gs = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.0, 1.05, 0.07, 0.18, 1.45],
        wspace=0.28,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    spacer_ax = fig.add_subplot(gs[0, 3])
    ax2 = fig.add_subplot(gs[0, 4])
    spacer_ax.axis("off")

    ax0.imshow(_norm_disp(bf), cmap="gray")
    ax0.imshow(rank_overlay, cmap=cmap, norm=rank_norm, alpha=0.90, interpolation="nearest")
    ax0.imshow(boundary_overlay, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.95, interpolation="nearest")
    ax0.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="white", linewidth=0.8, alpha=0.7)
    ax0.plot(
        [centroid_xy[0], bead_xy[0]],
        [centroid_xy[1], bead_xy[1]],
        color="red",
        linewidth=1.2,
        alpha=0.9,
    )
    ax0.scatter([centroid_xy[0]], [centroid_xy[1]], c="white", s=18)
    ax0.scatter([bead_xy[0]], [bead_xy[1]], c="magenta", s=55, marker="x")
    ax0.set_title("Brightfield + full shell colored by discrete 5 deg orientation bins")
    ax0.axis("off")

    ax1.imshow(cyst_overlay, cmap="gray", alpha=0.25, interpolation="nearest")
    ax1.imshow(rank_overlay, cmap=cmap, norm=rank_norm, alpha=0.95, interpolation="nearest")
    ax1.imshow(boundary_overlay, cmap="gray", vmin=0.0, vmax=1.0, alpha=0.95, interpolation="nearest")
    ax1.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="black", linewidth=0.8, alpha=0.8)
    ax1.plot(
        [centroid_xy[0], bead_xy[0]],
        [centroid_xy[1], bead_xy[1]],
        color="red",
        linewidth=1.2,
        alpha=0.9,
    )
    ax1.scatter([centroid_xy[0]], [centroid_xy[1]], c="black", s=18)
    ax1.scatter([bead_xy[0]], [bead_xy[1]], c="magenta", s=55, marker="x")
    ax1.set_title("Same full shell on cyst mask (0 deg points toward bead)")
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y1, y0)
    ax1.axis("off")

    bar_colors = cmap(rank_norm(support_rank))
    widths = np.maximum(bin_hi - bin_lo, 0.85 * float(ORIENTATION_TRACE_BIN_DEG))
    ax2.bar(
        support_df["theta_rel_bead_deg"],
        support_df["n_pixels"],
        width=widths,
        color=bar_colors,
        edgecolor="none",
        align="center",
    )
    ax2.plot(
        support_df["theta_rel_bead_deg"],
        support_df["n_pixels"],
        color="black",
        linewidth=1.0,
        alpha=0.55,
    )
    ax2.axvline(0.0, color="red", linestyle="--", linewidth=1.2, alpha=0.9)
    ax2.set_xlabel("Angle relative to bead direction on full shell (deg)")
    ax2.set_ylabel("Shell pixels in exact 5 deg bin")
    ax2.yaxis.labelpad = 20
    ax2.set_title("Same bins shown as the shell-pixel count histogram")

    mappable = plt.cm.ScalarMappable(norm=rank_norm, cmap=cmap)
    mappable.set_array([])
    tick_idx = np.linspace(0, len(occupied_centers) - 1, min(7, len(occupied_centers)), dtype=int)
    tick_idx = np.unique(tick_idx)
    cbar = fig.colorbar(mappable, cax=cax, ticks=tick_idx)
    cbar.ax.set_yticklabels([bin_labels[i] for i in tick_idx])
    cbar.set_label("Occupied 5 deg orientation-bin range (deg)")

    fig.suptitle(
        f"Orientation-bin geometry snapshot | {row['cyst_id']} | scene={int(row['scene_index'])} | "
        f"full-shell orientation histogram over {len(occupied_centers)} occupied bins",
        y=0.98,
    )
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.10, top=0.90, wspace=0.30)
    plt.show()


_manual_roi_summary_cache = None


def _manual_roi_summary_df() -> pd.DataFrame:
    """Load the canonical ROI summary used to reconstruct one representative scene."""
    global _manual_roi_summary_cache
    if _manual_roi_summary_cache is None:
        if ROOT is None:
            raise RuntimeError("ROOT must be set before loading the ROI summary.")
        roi_summary_tsv = resolve_path(
            "results/tables/manual_roi_summary_36locations.tsv",
            ROOT,
        )
        _manual_roi_summary_cache = pd.read_csv(roi_summary_tsv, sep="\t")
    return _manual_roi_summary_cache


def _cyst_number_label(cyst_id: str) -> str:
    suffix = str(cyst_id).split("_")[-1]
    digits = suffix.split("cyst_")[-1] if "cyst_" in suffix else suffix
    try:
        return f"c{int(digits)}"
    except Exception:
        return str(cyst_id)


def _scene_polygon_rows(scene_index: int) -> list[dict]:
    roi_df = _manual_roi_summary_df()
    scene_rows = (
        roi_df.loc[roi_df["scene_index"] == int(scene_index)]
        .copy()
        .sort_values("cyst_id")
        .reset_index(drop=True)
    )
    out = []
    for _, rr in scene_rows.iterrows():
        rec = json.loads(
            resolve_path(str(rr["roi_json_path"]), ROOT).read_text(encoding="utf-8")
        )
        out.append(
            {
                "cyst_id": str(rec["cyst_id"]),
                "polygon_xy": np.asarray(rec["polygon_xy_px"], dtype=np.float64),
                "centroid_xy": np.asarray(rec["roi_centroid_xy_px"], dtype=np.float64),
            }
        )
    return out


def _crop_bounds_from_mask_and_points(
    mask: np.ndarray,
    points_xy: list[np.ndarray] | tuple[np.ndarray, ...],
    pad: int = 18,
) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0 or len(yy) == 0:
        return 0, mask.shape[1], mask.shape[0], 0

    xs = [float(np.min(xx)), float(np.max(xx))]
    ys = [float(np.min(yy)), float(np.max(yy))]
    for pts in points_xy:
        arr = np.asarray(pts, dtype=np.float64)
        if arr.size == 0:
            continue
        arr = np.atleast_2d(arr)
        if arr.shape[1] != 2:
            continue
        xs.extend(arr[:, 0].tolist())
        ys.extend(arr[:, 1].tolist())

    x0 = max(0, int(np.floor(min(xs))) - pad)
    x1 = min(mask.shape[1], int(np.ceil(max(xs))) + pad + 1)
    y0 = max(0, int(np.floor(min(ys))) - pad)
    y1 = min(mask.shape[0], int(np.ceil(max(ys))) + pad + 1)
    return x0, x1, y1, y0


def _plot_angle_arc(
    ax,
    center_xy: np.ndarray,
    start_vec_xy: np.ndarray,
    end_vec_xy: np.ndarray,
    radius_px: float,
    color: str,
    label: str | None = None,
    lw: float = 1.8,
) -> None:
    start_vec_xy = np.asarray(start_vec_xy, dtype=np.float64)
    end_vec_xy = np.asarray(end_vec_xy, dtype=np.float64)
    if not np.all(np.isfinite(start_vec_xy)) or not np.all(np.isfinite(end_vec_xy)):
        return
    a0 = float(np.arctan2(start_vec_xy[1], start_vec_xy[0]))
    a1 = float(np.arctan2(end_vec_xy[1], end_vec_xy[0]))
    delta = float(np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)))
    angles = np.linspace(a0, a0 + delta, 80)
    center_xy = np.asarray(center_xy, dtype=np.float64)
    arc_x = center_xy[0] + float(radius_px) * np.cos(angles)
    arc_y = center_xy[1] + float(radius_px) * np.sin(angles)
    ax.plot(arc_x, arc_y, color=color, linewidth=lw, alpha=0.95)
    if label:
        mid = a0 + 0.5 * delta
        tx = center_xy[0] + 1.18 * float(radius_px) * np.cos(mid)
        ty = center_xy[1] + 1.18 * float(radius_px) * np.sin(mid)
        ax.text(
            tx,
            ty,
            label,
            color=color,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=(1, 1, 1, 0.72),
                edgecolor="none",
            ),
        )


def plot_representative_coordinate_diagrams(cyst_id: str, show: bool = True):
    """Show one representative scene/cyst as a schematic for the distance and orientation coordinates."""
    payload = build_debug_payload(cyst_id)
    distance_geometry = build_distance_bin_geometry_payload(payload)
    orientation_geometry = build_orientation_bin_geometry_payload(payload)

    row = payload["row"]
    bf = payload["bf"]
    mask = payload["mask"]
    centroid_xy = payload["centroid_xy"]
    bead_xy = payload["bead_xy"]
    nearest_boundary_xy = payload["nearest_boundary_xy"]
    boundary_xy = payload["boundary_xy"]
    anterior_xy = payload["anterior_xy"]
    posterior_xy = payload["posterior_xy"]
    midline_u = payload["midline_u"]
    theta_b_deg = float(row["theta_b_deg"])
    scene_index = int(row["scene_index"])

    scene_polygons = _scene_polygon_rows(scene_index)
    x0, x1, y1, y0 = _crop_bounds_from_mask_and_points(
        mask,
        [bead_xy, nearest_boundary_xy, centroid_xy, anterior_xy, posterior_xy],
        pad=42,
    )

    dist_rank_img = distance_geometry["pixel_rank_img"]
    dist_rank_overlay = np.ma.masked_invalid(dist_rank_img)
    dist_boundary_overlay = np.ma.masked_where(
        ~_discrete_region_boundary_mask(dist_rank_img),
        np.ones_like(dist_rank_img, dtype=np.float32),
    )
    dist_centers = distance_geometry["occupied_centers"]
    dist_cmap = plt.get_cmap("turbo", max(int(len(dist_centers)), 2))
    dist_norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(dist_centers) + 0.5, 1.0),
        dist_cmap.N,
    )

    theta_rank_img = orientation_geometry["pixel_rank_img"]
    theta_rank_overlay = np.ma.masked_invalid(theta_rank_img)
    theta_boundary_overlay = np.ma.masked_where(
        ~_discrete_region_boundary_mask(theta_rank_img),
        np.ones_like(theta_rank_img, dtype=np.float32),
    )
    theta_centers = orientation_geometry["occupied_centers"]
    theta_cmap = plt.get_cmap("twilight_shifted", max(int(len(theta_centers)), 2))
    theta_norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(theta_centers) + 0.5, 1.0),
        theta_cmap.N,
    )

    fig = plt.figure(figsize=(15.8, 11.6))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.12, 1.0, 0.045],
        height_ratios=[1.0, 1.0],
        wspace=0.10,
        hspace=0.18,
    )
    ax_dist_scene = fig.add_subplot(gs[0, 0])
    ax_dist_zoom = fig.add_subplot(gs[0, 1])
    cax_dist = fig.add_subplot(gs[0, 2])
    ax_theta_scene = fig.add_subplot(gs[1, 0])
    ax_theta_zoom = fig.add_subplot(gs[1, 1])
    cax_theta = fig.add_subplot(gs[1, 2])

    def _draw_scene_overview(ax):
        ax.imshow(_norm_disp(bf), cmap="gray")
        for rr in scene_polygons:
            poly = rr["polygon_xy"]
            is_selected = rr["cyst_id"] == str(cyst_id)
            ax.plot(
                poly[:, 0],
                poly[:, 1],
                color="cyan" if is_selected else "white",
                linewidth=2.0 if is_selected else 1.0,
                alpha=0.95 if is_selected else 0.65,
            )
            cx, cy = rr["centroid_xy"]
            ax.text(
                float(cx),
                float(cy),
                _cyst_number_label(rr["cyst_id"]),
                color="yellow" if is_selected else "white",
                fontsize=12 if is_selected else 10,
                fontweight="bold" if is_selected else "normal",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.16",
                    facecolor=(0, 0, 0, 0.48),
                    edgecolor="none",
                ),
            )
        ax.scatter(
            [bead_xy[0]],
            [bead_xy[1]],
            c="magenta",
            s=200,
            marker="x",
            linewidths=2.6,
            zorder=8,
        )
        ax.axis("off")

    _draw_scene_overview(ax_dist_scene)
    ax_dist_scene.annotate(
        "",
        xy=(float(bead_xy[0]), float(bead_xy[1])),
        xytext=(float(nearest_boundary_xy[0]), float(nearest_boundary_xy[1])),
        arrowprops=dict(arrowstyle="<->", color="red", linewidth=2.0),
    )
    ax_dist_scene.scatter(
        [nearest_boundary_xy[0]], [nearest_boundary_xy[1]], c="yellow", s=65, zorder=9
    )
    d_mid = 0.5 * (np.asarray(bead_xy) + np.asarray(nearest_boundary_xy))
    ax_dist_scene.text(
        float(d_mid[0]) - 18.0,
        float(d_mid[1]) - 22.0,
        "d",
        color="red",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.16",
            facecolor=(1, 1, 1, 0.72),
            edgecolor="none",
        ),
    )
    ax_dist_scene.set_title(
        "Distance coordinate d on the representative array",
        fontsize=13,
        pad=10,
    )

    ax_dist_zoom.imshow(_norm_disp(bf), cmap="gray")
    ax_dist_zoom.imshow(
        dist_rank_overlay,
        cmap=dist_cmap,
        norm=dist_norm,
        alpha=0.92,
        interpolation="nearest",
    )
    ax_dist_zoom.imshow(
        dist_boundary_overlay,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        alpha=0.90,
        interpolation="nearest",
    )
    ax_dist_zoom.plot(
        boundary_xy[:, 0],
        boundary_xy[:, 1],
        color="white",
        linewidth=1.0,
        alpha=0.9,
    )
    ax_dist_zoom.annotate(
        "",
        xy=(float(bead_xy[0]), float(bead_xy[1])),
        xytext=(float(nearest_boundary_xy[0]), float(nearest_boundary_xy[1])),
        arrowprops=dict(arrowstyle="<->", color="red", linewidth=2.0),
    )
    ax_dist_zoom.scatter(
        [bead_xy[0]], [bead_xy[1]], c="magenta", s=120, marker="x", linewidths=2.0, zorder=8
    )
    ax_dist_zoom.scatter(
        [nearest_boundary_xy[0]],
        [nearest_boundary_xy[1]],
        c="yellow",
        s=52,
        zorder=9,
    )
    ax_dist_zoom.set_xlim(x0, x1)
    ax_dist_zoom.set_ylim(y1, y0)
    ax_dist_zoom.axis("off")
    ax_dist_zoom.set_title(
        "Same shell pixels assigned to exact 10 um distance bins",
        fontsize=13,
        pad=10,
    )
    ax_dist_zoom.text(
        0.02,
        0.02,
        "Each shell pixel contributes to one occupied local-distance bin.",
        transform=ax_dist_zoom.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor=(0, 0, 0, 0.52),
            edgecolor="none",
        ),
    )
    dist_mappable = plt.cm.ScalarMappable(norm=dist_norm, cmap=dist_cmap)
    dist_mappable.set_array([])
    dist_tick_idx = np.linspace(
        0, len(dist_centers) - 1, min(6, len(dist_centers)), dtype=int
    )
    dist_tick_idx = np.unique(dist_tick_idx)
    dist_cbar = fig.colorbar(dist_mappable, cax=cax_dist, ticks=dist_tick_idx)
    dist_cbar.ax.set_yticklabels(
        [
            f"{distance_geometry['bin_lo'][i]:.0f}-{distance_geometry['bin_hi'][i]:.0f}"
            for i in dist_tick_idx
        ]
    )
    dist_cbar.set_label("Occupied 10 um distance bins (um)")

    _draw_scene_overview(ax_theta_scene)
    ax_theta_scene.plot(
        [float(anterior_xy[0]), float(posterior_xy[0])],
        [float(anterior_xy[1]), float(posterior_xy[1])],
        color="white",
        linewidth=2.0,
        alpha=0.95,
    )
    ax_theta_scene.scatter([anterior_xy[0]], [anterior_xy[1]], c="lime", s=55, zorder=9)
    ax_theta_scene.scatter([posterior_xy[0]], [posterior_xy[1]], c="red", s=55, zorder=9)
    ax_theta_scene.scatter([centroid_xy[0]], [centroid_xy[1]], c="white", s=35, zorder=9)
    ax_theta_scene.plot(
        [float(centroid_xy[0]), float(bead_xy[0])],
        [float(centroid_xy[1]), float(bead_xy[1])],
        color="red",
        linewidth=2.0,
        alpha=0.95,
    )
    bead_vec = np.asarray(bead_xy, dtype=np.float64) - np.asarray(
        centroid_xy, dtype=np.float64
    )
    post_vec = np.asarray(midline_u, dtype=np.float64)
    if np.linalg.norm(bead_vec) > 0:
        bead_vec = bead_vec / np.linalg.norm(bead_vec)
    if np.linalg.norm(post_vec) > 0:
        post_vec = post_vec / np.linalg.norm(post_vec)
    _plot_angle_arc(
        ax_theta_scene,
        center_xy=np.asarray(centroid_xy, dtype=np.float64),
        start_vec_xy=post_vec,
        end_vec_xy=bead_vec,
        radius_px=84.0,
        color="red",
        label=r"$\theta_b$",
        lw=1.9,
    )
    ax_theta_scene.text(
        float(anterior_xy[0]) - 28.0,
        float(anterior_xy[1]) - 18.0,
        "A",
        color="lime",
        fontsize=12,
        fontweight="bold",
    )
    ax_theta_scene.text(
        float(posterior_xy[0]) + 12.0,
        float(posterior_xy[1]) + 10.0,
        "P",
        color="red",
        fontsize=12,
        fontweight="bold",
    )
    ax_theta_scene.set_title(
        "Bead direction " + r"$\theta_b$" + " relative to the cyst AP axis",
        fontsize=13,
        pad=10,
    )

    ax_theta_zoom.imshow(_norm_disp(bf), cmap="gray")
    ax_theta_zoom.imshow(
        theta_rank_overlay,
        cmap=theta_cmap,
        norm=theta_norm,
        alpha=0.92,
        interpolation="nearest",
    )
    ax_theta_zoom.imshow(
        theta_boundary_overlay,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        alpha=0.88,
        interpolation="nearest",
    )
    ax_theta_zoom.plot(
        boundary_xy[:, 0],
        boundary_xy[:, 1],
        color="white",
        linewidth=1.0,
        alpha=0.9,
    )
    ax_theta_zoom.plot(
        [float(centroid_xy[0]), float(bead_xy[0])],
        [float(centroid_xy[1]), float(bead_xy[1])],
        color="red",
        linewidth=2.0,
        alpha=0.95,
    )
    _plot_angle_arc(
        ax_theta_zoom,
        center_xy=np.asarray(centroid_xy, dtype=np.float64),
        start_vec_xy=bead_vec,
        end_vec_xy=np.array([np.cos(np.deg2rad(55.0)), np.sin(np.deg2rad(55.0))]),
        radius_px=58.0,
        color="white",
        label=r"$\theta$",
        lw=1.6,
    )
    ax_theta_zoom.scatter(
        [bead_xy[0]], [bead_xy[1]], c="magenta", s=120, marker="x", linewidths=2.0, zorder=8
    )
    ax_theta_zoom.scatter([centroid_xy[0]], [centroid_xy[1]], c="white", s=34, zorder=9)
    ax_theta_zoom.set_xlim(x0, x1)
    ax_theta_zoom.set_ylim(y1, y0)
    ax_theta_zoom.axis("off")
    ax_theta_zoom.set_title(
        "Same shell pixels assigned to exact 5 deg angle bins",
        fontsize=13,
        pad=10,
    )
    ax_theta_zoom.text(
        0.02,
        0.02,
        "Here 0° points toward the bead; shell pixels enter exact 5° bins.",
        transform=ax_theta_zoom.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor=(0, 0, 0, 0.52),
            edgecolor="none",
        ),
    )
    theta_mappable = plt.cm.ScalarMappable(norm=theta_norm, cmap=theta_cmap)
    theta_mappable.set_array([])
    theta_tick_idx = np.linspace(
        0, len(theta_centers) - 1, min(7, len(theta_centers)), dtype=int
    )
    theta_tick_idx = np.unique(theta_tick_idx)
    theta_cbar = fig.colorbar(theta_mappable, cax=cax_theta, ticks=theta_tick_idx)
    theta_cbar.ax.set_yticklabels(
        [
            f"{orientation_geometry['bin_lo'][i]:.0f} to {orientation_geometry['bin_hi'][i]:.0f}"
            for i in theta_tick_idx
        ]
    )
    theta_cbar.set_label("Occupied 5 deg angle bins relative to bead (deg)")

    fig.suptitle(
        "Representative coordinate diagrams | "
        f"{cyst_id} | scene={scene_index} | "
        f"d={float(row['distance_d_um']):.1f} um | "
        + r"$\theta_b$"
        + f"={theta_b_deg:.1f}°",
        y=0.98,
        fontsize=15,
    )
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.05, top=0.93)
    if show:
        plt.show()
    return fig


def _representative_coordinate_context(cyst_id: str):
    payload = build_debug_payload(cyst_id)
    distance_geometry = build_distance_bin_geometry_payload(payload)
    orientation_geometry = build_orientation_bin_geometry_payload(payload)

    row = payload["row"]
    bf = payload["bf"]
    mask = payload["mask"]
    centroid_xy = payload["centroid_xy"]
    bead_xy = payload["bead_xy"]
    nearest_boundary_xy = payload["nearest_boundary_xy"]
    boundary_xy = payload["boundary_xy"]
    anterior_xy = payload["anterior_xy"]
    posterior_xy = payload["posterior_xy"]
    midline_u = payload["midline_u"]
    theta_b_deg = float(row["theta_b_deg"])
    scene_index = int(row["scene_index"])

    scene_polygons = _scene_polygon_rows(scene_index)
    x0, x1, y1, y0 = _crop_bounds_from_mask_and_points(
        mask,
        [bead_xy, nearest_boundary_xy, centroid_xy, anterior_xy, posterior_xy],
        pad=42,
    )

    dist_rank_img = distance_geometry["pixel_rank_img"]
    theta_rank_img = orientation_geometry["pixel_rank_img"]

    return {
        "cyst_id": str(cyst_id),
        "payload": payload,
        "row": row,
        "bf": bf,
        "mask": mask,
        "centroid_xy": centroid_xy,
        "bead_xy": bead_xy,
        "nearest_boundary_xy": nearest_boundary_xy,
        "boundary_xy": boundary_xy,
        "anterior_xy": anterior_xy,
        "posterior_xy": posterior_xy,
        "midline_u": midline_u,
        "theta_b_deg": theta_b_deg,
        "scene_index": scene_index,
        "scene_polygons": scene_polygons,
        "crop_bounds": (x0, x1, y1, y0),
        "distance_geometry": distance_geometry,
        "orientation_geometry": orientation_geometry,
        "dist_rank_overlay": np.ma.masked_invalid(dist_rank_img),
        "dist_boundary_overlay": np.ma.masked_where(
            ~_discrete_region_boundary_mask(dist_rank_img),
            np.ones_like(dist_rank_img, dtype=np.float32),
        ),
        "theta_rank_overlay": np.ma.masked_invalid(theta_rank_img),
        "theta_boundary_overlay": np.ma.masked_where(
            ~_discrete_region_boundary_mask(theta_rank_img),
            np.ones_like(theta_rank_img, dtype=np.float32),
        ),
    }


def _draw_representative_scene_overview(ax, ctx):
    bf = ctx["bf"]
    bead_xy = ctx["bead_xy"]
    cyst_id = ctx["cyst_id"]
    scene_polygons = ctx["scene_polygons"]

    ax.imshow(_norm_disp(bf), cmap="gray")
    for rr in scene_polygons:
        poly = rr["polygon_xy"]
        is_selected = rr["cyst_id"] == cyst_id
        ax.plot(
            poly[:, 0],
            poly[:, 1],
            color="cyan" if is_selected else "white",
            linewidth=2.0 if is_selected else 1.0,
            alpha=0.95 if is_selected else 0.65,
        )
        cx, cy = rr["centroid_xy"]
        ax.text(
            float(cx),
            float(cy),
            _cyst_number_label(rr["cyst_id"]),
            color="yellow" if is_selected else "white",
            fontsize=12 if is_selected else 10,
            fontweight="bold" if is_selected else "normal",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor=(0, 0, 0, 0.48),
                edgecolor="none",
            ),
        )
    ax.scatter(
        [bead_xy[0]],
        [bead_xy[1]],
        c="magenta",
        s=200,
        marker="x",
        linewidths=2.6,
        zorder=8,
    )
    ax.axis("off")


def _nice_distance_colorbar_ticks(centers_um: np.ndarray) -> np.ndarray:
    centers_um = np.asarray(centers_um, dtype=np.float64)
    if centers_um.size == 0:
        return np.array([], dtype=np.float64)
    start = int(np.ceil(np.nanmin(centers_um) / 100.0) * 100.0)
    stop = int(np.floor(np.nanmax(centers_um) / 100.0) * 100.0)
    ticks = np.arange(start, stop + 1, 100, dtype=np.float64)
    if ticks.size < 3:
        start = int(np.floor(np.nanmin(centers_um) / 100.0) * 100.0)
        stop = int(np.ceil(np.nanmax(centers_um) / 100.0) * 100.0)
        ticks = np.arange(start, stop + 1, 100, dtype=np.float64)
    return ticks


def _nice_angle_colorbar_ticks(centers_deg: np.ndarray) -> np.ndarray:
    centers_deg = np.asarray(centers_deg, dtype=np.float64)
    if centers_deg.size == 0:
        return np.array([], dtype=np.float64)
    ticks = np.arange(-180.0, 181.0, 60.0, dtype=np.float64)
    keep = (ticks >= np.nanmin(centers_deg)) & (ticks <= np.nanmax(centers_deg))
    ticks = ticks[keep]
    if ticks.size < 3:
        ticks = np.arange(-180.0, 181.0, 90.0, dtype=np.float64)
        keep = (ticks >= np.nanmin(centers_deg)) & (ticks <= np.nanmax(centers_deg))
        ticks = ticks[keep]
    return ticks


def plot_representative_distance_coordinate_diagram(
    cyst_id: str, show: bool = True
):
    """Show the distance-coordinate schematic as a standalone figure."""
    ctx = _representative_coordinate_context(cyst_id)

    row = ctx["row"]
    bf = ctx["bf"]
    bead_xy = ctx["bead_xy"]
    nearest_boundary_xy = ctx["nearest_boundary_xy"]
    boundary_xy = ctx["boundary_xy"]
    x0, x1, y1, y0 = ctx["crop_bounds"]
    distance_geometry = ctx["distance_geometry"]
    dist_value_overlay = np.ma.masked_invalid(distance_geometry["pixel_bin_img"])
    dist_boundary_overlay = ctx["dist_boundary_overlay"]
    dist_centers = distance_geometry["occupied_centers"]
    dist_cmap = plt.get_cmap("turbo")
    dist_norm = mcolors.Normalize(
        vmin=float(np.nanmin(dist_centers)),
        vmax=float(np.nanmax(dist_centers)),
    )

    fig = plt.figure(figsize=(14.0, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 0.05], wspace=0.12)
    ax_scene = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    _draw_representative_scene_overview(ax_scene, ctx)
    ax_scene.annotate(
        "",
        xy=(float(bead_xy[0]), float(bead_xy[1])),
        xytext=(float(nearest_boundary_xy[0]), float(nearest_boundary_xy[1])),
        arrowprops=dict(arrowstyle="<->", color="red", linewidth=2.0),
    )
    ax_scene.scatter(
        [nearest_boundary_xy[0]], [nearest_boundary_xy[1]], c="yellow", s=65, zorder=9
    )
    d_mid = 0.5 * (np.asarray(bead_xy) + np.asarray(nearest_boundary_xy))
    ax_scene.text(
        float(d_mid[0]) - 18.0,
        float(d_mid[1]) - 22.0,
        "d",
        color="red",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.16",
            facecolor=(1, 1, 1, 0.72),
            edgecolor="none",
        ),
    )
    ax_scene.set_title("Distance coordinate on the representative array", fontsize=13, pad=10)

    ax_zoom.imshow(_norm_disp(bf), cmap="gray")
    ax_zoom.imshow(
        dist_value_overlay,
        cmap=dist_cmap,
        norm=dist_norm,
        alpha=0.92,
        interpolation="nearest",
    )
    ax_zoom.imshow(
        dist_boundary_overlay,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        alpha=0.90,
        interpolation="nearest",
    )
    ax_zoom.plot(
        boundary_xy[:, 0],
        boundary_xy[:, 1],
        color="white",
        linewidth=1.0,
        alpha=0.9,
    )
    ax_zoom.annotate(
        "",
        xy=(float(bead_xy[0]), float(bead_xy[1])),
        xytext=(float(nearest_boundary_xy[0]), float(nearest_boundary_xy[1])),
        arrowprops=dict(arrowstyle="<->", color="red", linewidth=2.0),
    )
    ax_zoom.scatter(
        [bead_xy[0]], [bead_xy[1]], c="magenta", s=120, marker="x", linewidths=2.0, zorder=8
    )
    ax_zoom.scatter(
        [nearest_boundary_xy[0]],
        [nearest_boundary_xy[1]],
        c="yellow",
        s=52,
        zorder=9,
    )
    ax_zoom.set_xlim(x0, x1)
    ax_zoom.set_ylim(y1, y0)
    ax_zoom.axis("off")
    ax_zoom.set_title(
        "Shell pixels assigned to exact 10 um distance bins",
        fontsize=13,
        pad=10,
    )
    ax_zoom.text(
        0.02,
        0.02,
        "Each shell pixel contributes to one occupied local-distance bin.",
        transform=ax_zoom.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor=(0, 0, 0, 0.52),
            edgecolor="none",
        ),
    )
    dist_mappable = plt.cm.ScalarMappable(norm=dist_norm, cmap=dist_cmap)
    dist_mappable.set_array([])
    dist_ticks = _nice_distance_colorbar_ticks(dist_centers)
    dist_cbar = fig.colorbar(dist_mappable, cax=cax, ticks=dist_ticks)
    dist_cbar.ax.set_yticklabels([f"{tick:.0f}" for tick in dist_ticks])
    dist_cbar.set_label("Local bead distance (um)")

    fig.suptitle(
        "Distance coordinate diagram | "
        f"{cyst_id} | scene={ctx['scene_index']} | "
        f"d={float(row['distance_d_um']):.1f} um",
        y=0.98,
        fontsize=15,
    )
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.06, top=0.88)
    if show:
        plt.show()
    return fig


def plot_representative_orientation_coordinate_diagram(
    cyst_id: str, show: bool = True
):
    """Show the orientation-coordinate schematic as a standalone figure."""
    ctx = _representative_coordinate_context(cyst_id)

    row = ctx["row"]
    bf = ctx["bf"]
    centroid_xy = ctx["centroid_xy"]
    bead_xy = ctx["bead_xy"]
    boundary_xy = ctx["boundary_xy"]
    anterior_xy = ctx["anterior_xy"]
    posterior_xy = ctx["posterior_xy"]
    midline_u = ctx["midline_u"]
    x0, x1, y1, y0 = ctx["crop_bounds"]
    orientation_geometry = ctx["orientation_geometry"]
    theta_value_overlay = np.ma.masked_invalid(orientation_geometry["pixel_bin_img"])
    theta_boundary_overlay = ctx["theta_boundary_overlay"]
    theta_centers = orientation_geometry["occupied_centers"]
    theta_cmap = plt.get_cmap("twilight_shifted")
    theta_norm = mcolors.Normalize(
        vmin=float(np.nanmin(theta_centers)),
        vmax=float(np.nanmax(theta_centers)),
    )

    fig = plt.figure(figsize=(14.0, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 0.05], wspace=0.12)
    ax_scene = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    _draw_representative_scene_overview(ax_scene, ctx)
    ax_scene.plot(
        [float(anterior_xy[0]), float(posterior_xy[0])],
        [float(anterior_xy[1]), float(posterior_xy[1])],
        color="white",
        linewidth=2.0,
        alpha=0.95,
    )
    ax_scene.scatter([anterior_xy[0]], [anterior_xy[1]], c="lime", s=55, zorder=9)
    ax_scene.scatter([posterior_xy[0]], [posterior_xy[1]], c="red", s=55, zorder=9)
    ax_scene.scatter([centroid_xy[0]], [centroid_xy[1]], c="white", s=35, zorder=9)
    ax_scene.plot(
        [float(centroid_xy[0]), float(bead_xy[0])],
        [float(centroid_xy[1]), float(bead_xy[1])],
        color="red",
        linewidth=2.0,
        alpha=0.95,
    )
    bead_vec = np.asarray(bead_xy, dtype=np.float64) - np.asarray(
        centroid_xy, dtype=np.float64
    )
    post_vec = np.asarray(midline_u, dtype=np.float64)
    if np.linalg.norm(bead_vec) > 0:
        bead_vec = bead_vec / np.linalg.norm(bead_vec)
    if np.linalg.norm(post_vec) > 0:
        post_vec = post_vec / np.linalg.norm(post_vec)
    _plot_angle_arc(
        ax_scene,
        center_xy=np.asarray(centroid_xy, dtype=np.float64),
        start_vec_xy=post_vec,
        end_vec_xy=bead_vec,
        radius_px=84.0,
        color="red",
        label=r"$\theta_b$",
        lw=1.9,
    )
    ax_scene.text(
        float(anterior_xy[0]) - 28.0,
        float(anterior_xy[1]) - 18.0,
        "A",
        color="lime",
        fontsize=12,
        fontweight="bold",
    )
    ax_scene.text(
        float(posterior_xy[0]) + 12.0,
        float(posterior_xy[1]) + 10.0,
        "P",
        color="red",
        fontsize=12,
        fontweight="bold",
    )
    ax_scene.set_title(
        "Bead direction " + r"$\theta_b$" + " relative to the cyst AP axis",
        fontsize=13,
        pad=10,
    )

    ax_zoom.imshow(_norm_disp(bf), cmap="gray")
    ax_zoom.imshow(
        theta_value_overlay,
        cmap=theta_cmap,
        norm=theta_norm,
        alpha=0.92,
        interpolation="nearest",
    )
    ax_zoom.imshow(
        theta_boundary_overlay,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        alpha=0.88,
        interpolation="nearest",
    )
    ax_zoom.plot(
        boundary_xy[:, 0],
        boundary_xy[:, 1],
        color="white",
        linewidth=1.0,
        alpha=0.9,
    )
    ax_zoom.plot(
        [float(centroid_xy[0]), float(bead_xy[0])],
        [float(centroid_xy[1]), float(bead_xy[1])],
        color="red",
        linewidth=2.0,
        alpha=0.95,
    )
    _plot_angle_arc(
        ax_zoom,
        center_xy=np.asarray(centroid_xy, dtype=np.float64),
        start_vec_xy=bead_vec,
        end_vec_xy=np.array([np.cos(np.deg2rad(55.0)), np.sin(np.deg2rad(55.0))]),
        radius_px=58.0,
        color="white",
        label=r"$\theta$",
        lw=1.6,
    )
    ax_zoom.scatter(
        [bead_xy[0]], [bead_xy[1]], c="magenta", s=120, marker="x", linewidths=2.0, zorder=8
    )
    ax_zoom.scatter([centroid_xy[0]], [centroid_xy[1]], c="white", s=34, zorder=9)
    ax_zoom.set_xlim(x0, x1)
    ax_zoom.set_ylim(y1, y0)
    ax_zoom.axis("off")
    ax_zoom.set_title(
        "Shell pixels assigned to exact 5 deg angle bins",
        fontsize=13,
        pad=10,
    )
    ax_zoom.text(
        0.02,
        0.02,
        "Here 0° points toward the bead; shell pixels enter exact 5° bins.",
        transform=ax_zoom.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor=(0, 0, 0, 0.52),
            edgecolor="none",
        ),
    )
    theta_mappable = plt.cm.ScalarMappable(norm=theta_norm, cmap=theta_cmap)
    theta_mappable.set_array([])
    theta_ticks = _nice_angle_colorbar_ticks(theta_centers)
    theta_cbar = fig.colorbar(theta_mappable, cax=cax, ticks=theta_ticks)
    theta_cbar.ax.set_yticklabels([f"{tick:.0f}" for tick in theta_ticks])
    theta_cbar.set_label("Angle relative to bead (deg)")

    fig.suptitle(
        "Orientation coordinate diagram | "
        f"{cyst_id} | scene={ctx['scene_index']} | "
        + r"$\theta_b$"
        + f"={float(row['theta_b_deg']):.1f}°",
        y=0.98,
        fontsize=15,
    )
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.06, top=0.88)
    if show:
        plt.show()
    return fig


__all__ = ['set_runtime_context', 'get_runtime_context', 'clear_diag_cache', 'list_roi_files_from_summary', 'resolve_path', 'to_2d', 'load_image_2d', 'polygon_xy_to_mask', 'psmad_over_dapi', 'unique_scene_image_paths', 'pooled_histogram_all_pixels', 'estimate_background_half_gaussian', 'normalize_channel_from_background', 'normalize_two_channels_and_ratio', 'signed_angle_deg', 'unit_vector', 'extract_boundary_xy', 'sample_image_at_xy', 'major_axis_from_mask', 'choose_axis_endpoints', 'mask_pixels_xy', 'wrap_angle_deg', 'peripheral_shell_mask', 'bead_facing_crescent_mask', 'top_fraction_mean', 'trace_table_from_edges', 'xy_bool_to_mask', 'simple_rolling_mean', 'build_distance_support_trace', 'circular_smooth', 'circular_first_moment_deg', 'aggregate_trace_across_cysts', 'binned_mean_sem', 'linear_fit_r2', 'check_unique_cyst_ids_or_fail', 'measure_bead_present_arrays', 'summarize_bead_present_measurement', 'quantify_bead_present_cyst', 'quantify_bead_present_cyst_with_measurement', 'quantify_no_bead_control_scene', 'infer_center_candidate_from_features', 'build_roi_feature_table', 'build_label_feature_table', 'build_scene_center_exclusion_map', 'infer_array_center_from_features', 'build_scene_array_center_map', 'get_scene_array_center_or_fail', 'assign_ap_from_array_center', 'geometry_from_mask_and_array_center', 'split_mask_into_two_lobes', 'centroid_xy_from_mask', 'should_draw_two_axis_geometry', 'plot_all_scene_outline_before_analysis', 'plot_all_scenes_geometry_before_analysis', 'build_debug_payload', 'plot_debug_payload', 'plot_geometry_montage', 'plot_distance_trace_outlier_context', 'build_control_channel_payload', 'plot_control_channel_montage', 'plot_control_channel_panel', 'plot_scene_qc', 'plot_all_scene_overlays', 'top_fit_outliers', 'rank_distance_trace_outliers', 'rank_bead_distant_outliers', 'rank_no_bead_control_outliers', 'build_problem_pixel_payload', 'compute_non_edge_support_floor', 'build_support_floor_example_payload', 'plot_support_floor_example_payload', 'build_distance_bin_geometry_payload', 'plot_distance_bin_geometry_payload', 'build_orientation_bin_geometry_payload', 'plot_orientation_bin_geometry_payload', 'plot_problem_pixel_payload', 'plot_representative_coordinate_diagrams', 'plot_representative_distance_coordinate_diagram', 'plot_representative_orientation_coordinate_diagram']
