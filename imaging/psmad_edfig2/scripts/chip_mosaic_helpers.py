from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

# This script needs czifile's `filtered_subblock_directory`, which exposes czifile's own
# subblock objects and has no libCZI equivalent, so it cannot use src/trunk_morph_ref/czi_compat.
# czifile itself is no longer installable beside this project's pinned numpy: the 2019 release
# calls imagecodecs.jxr_decode (renamed jpegxr_decode) and the current release requires numpy 2.x.
# This mosaic step is therefore NOT runnable as shipped; its outputs are committed under derived/.
# The import is DEFERRED into _czifile() rather than taken at module scope so that importing
# this module still works: several runnable scripts import it for its other helpers, and a
# top-level import here made every one of them fail too.
def _czifile():
    """Import czifile on demand, with a usable message when it cannot be installed."""
    try:
        import czifile
    except Exception as exc:                                        # noqa: BLE001
        raise RuntimeError(
            "This mosaic step needs czifile, which cannot be installed beside this project's "
            "pinned numpy (see env/environment.yml). Its outputs are committed under derived/."
        ) from exc
    return czifile
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage import feature, filters, morphology, transform
from skimage.registration import phase_cross_correlation


_GLOBAL_DAPI_BF_SCENE_CACHE: Dict[Tuple[object, ...], Dict[str, object]] = {}


@dataclass(frozen=True)
class CziTileLatticeInfo:
    path: Path
    axes: str
    shape: Tuple[int, ...]
    scene_count: int
    z_values: List[int]
    y_starts: List[int]
    x_starts: List[int]
    tile_shape_yx: Tuple[int, int]
    nominal_dy_px: float
    nominal_dx_px: float
    scale_x_um: Optional[float]
    scale_y_um: Optional[float]


def rel_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


XML_START = b"<ImageDocument"
XML_END = b"</ImageDocument>"


def read_embedded_xml(path: Path, max_read_bytes: int = 8_000_000) -> ET.Element:
    with path.open("rb") as f:
        blob = f.read(max_read_bytes)
    start = blob.find(XML_START)
    end = blob.find(XML_END)
    if start < 0 or end < 0:
        raise ValueError(f"No CZI ImageDocument XML found near header: {path}")
    xml_bytes = blob[start : end + len(XML_END)]
    return ET.fromstring(xml_bytes)


def scale_axis_um(root: ET.Element, axis_id: str) -> Optional[float]:
    for dist in root.findall("Metadata/Scaling/Items/Distance"):
        if dist.attrib.get("Id") != axis_id:
            continue
        raw = dist.findtext("Value")
        if raw is None:
            return None
        try:
            return float(raw) * 1e6
        except ValueError:
            return None
    return None


def channel_name_from_xml(root: ET.Element) -> str:
    chans = root.findall("Metadata/DisplaySetting/Channels/Channel")
    for ch in chans:
        name = ch.findtext("ShortName") or ch.findtext("Name") or ch.findtext("DyeName")
        if name:
            return str(name)
    return ""


def inspect_single_channel_czi(path: Path, scene_index: int = 0) -> CziTileLatticeInfo:
    root = read_embedded_xml(path)
    scale_x_um = scale_axis_um(root, "X")
    scale_y_um = scale_axis_um(root, "Y")

    with _czifile().CziFile(path) as czi:
        axes = str(czi.axes)
        shape = tuple(int(v) for v in czi.shape)
        fsbd = czi.filtered_subblock_directory
        axis_idx = {ax: i for i, ax in enumerate(axes)}
        scene_count = int(shape[axis_idx["S"]]) if "S" in axis_idx else 1

        y_set: set[int] = set()
        x_set: set[int] = set()
        z_set: set[int] = set()
        tile_shape = None
        for d in fsbd:
            start = d.start
            s_idx = int(start[axis_idx["S"]]) if "S" in axis_idx else 0
            if s_idx != int(scene_index):
                continue
            z_idx = int(start[axis_idx["Z"]]) if "Z" in axis_idx else 0
            y0 = int(start[axis_idx["Y"]])
            x0 = int(start[axis_idx["X"]])
            arr = np.asarray(d.data_segment().data(raw=False, resize=True))
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise RuntimeError(f"Unexpected tile ndim={arr.ndim} in {path}")
            tile_shape = tuple(int(v) for v in arr.shape)
            y_set.add(y0)
            x_set.add(x0)
            z_set.add(z_idx)

    y_starts = sorted(y_set)
    x_starts = sorted(x_set)
    if not y_starts or not x_starts or tile_shape is None:
        raise RuntimeError(f"Failed to extract tile lattice from {path}")

    dy = float(np.median(np.diff(y_starts))) if len(y_starts) > 1 else float(tile_shape[0])
    dx = float(np.median(np.diff(x_starts))) if len(x_starts) > 1 else float(tile_shape[1])
    return CziTileLatticeInfo(
        path=Path(path),
        axes=axes,
        shape=shape,
        scene_count=scene_count,
        z_values=sorted(int(z) for z in z_set),
        y_starts=[int(v) for v in y_starts],
        x_starts=[int(v) for v in x_starts],
        tile_shape_yx=tuple(int(v) for v in tile_shape),
        nominal_dy_px=float(dy),
        nominal_dx_px=float(dx),
        scale_x_um=None if scale_x_um is None else float(scale_x_um),
        scale_y_um=None if scale_y_um is None else float(scale_y_um),
    )


def load_single_channel_tile_map(path: Path, scene_index: int = 0):
    info = inspect_single_channel_czi(path, scene_index=scene_index)
    with _czifile().CziFile(path) as czi:
        axes = str(czi.axes)
        axis_idx = {ax: i for i, ax in enumerate(axes)}
        fsbd = czi.filtered_subblock_directory
        tile_map: Dict[Tuple[int, int, int], np.ndarray] = {}
        for d in fsbd:
            start = d.start
            s_idx = int(start[axis_idx["S"]]) if "S" in axis_idx else 0
            if s_idx != int(scene_index):
                continue
            z_idx = int(start[axis_idx["Z"]]) if "Z" in axis_idx else 0
            y0 = int(start[axis_idx["Y"]])
            x0 = int(start[axis_idx["X"]])
            tile = np.asarray(d.data_segment().data(raw=False, resize=True))
            tile = np.squeeze(tile)
            if tile.ndim != 2:
                raise RuntimeError(f"Unexpected tile ndim={tile.ndim} in {path} start={start}")
            tile_map[(z_idx, y0, x0)] = tile
    return info, tile_map


def _zscore_norm(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    return (arr - float(np.nanmean(arr))) / (float(np.nanstd(arr)) + 1e-6)


def robust_rescale(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [float(p_low), float(p_high)])
    out = (arr - float(lo)) / max(float(hi - lo), 1e-6)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _prepare_alignment_overlap(
    img: np.ndarray,
    background_sigma_px: float = 24.0,
    feature_sigma_px: float = 1.0,
) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    if not np.isfinite(arr).any():
        return np.zeros_like(arr, dtype=np.float32)

    # Flatten broad BF illumination so seam matching is driven by sharper structure.
    bg = ndi.gaussian_filter(arr, sigma=float(background_sigma_px))
    flat = arr - bg
    if float(feature_sigma_px) > 0:
        flat = ndi.gaussian_filter(flat, sigma=float(feature_sigma_px))

    # Emphasize cyst edges and bead-well rims over low-frequency BF texture.
    grad = filters.sobel(flat)
    return _zscore_norm(robust_rescale(grad, p_low=2.0, p_high=99.8))


def _prepare_dapi_overlap(
    img: np.ndarray,
    smooth_sigma_px: float = 1.2,
    threshold_percentile: float = 90.0,
    min_object_px: int = 24,
    dilate_disk_px: int = 2,
    feature_sigma_px: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).any():
        shape = arr.shape if arr.ndim == 2 else (0, 0)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=bool)
    norm = robust_rescale(arr, p_low=1.0, p_high=99.8)
    smooth = ndi.gaussian_filter(norm, sigma=float(smooth_sigma_px))
    finite = smooth[np.isfinite(smooth)]
    if finite.size < 64:
        return np.zeros_like(smooth, dtype=np.float32), np.zeros_like(smooth, dtype=bool)
    try:
        thr_otsu = float(filters.threshold_otsu(finite))
    except Exception:
        thr_otsu = float('nan')
    thr_pct = float(np.percentile(finite, float(threshold_percentile)))
    thr = float(max(0.12, np.nanmax([thr_otsu, thr_pct])))
    mask = smooth > thr
    mask = morphology.remove_small_objects(mask, min_size=int(min_object_px))
    if int(dilate_disk_px) > 0 and np.any(mask):
        mask = morphology.binary_dilation(mask, morphology.disk(int(dilate_disk_px)))
    if int(np.count_nonzero(mask)) < int(min_object_px):
        return np.zeros_like(smooth, dtype=np.float32), mask.astype(bool)
    feat = ndi.gaussian_filter(mask.astype(np.float32), sigma=float(feature_sigma_px))
    return _zscore_norm(feat), mask.astype(bool)





def _component_rows_from_mask(mask: np.ndarray) -> List[Dict[str, float]]:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0 or not np.any(arr):
        return []
    labels, n_labels = ndi.label(arr)
    rows: List[Dict[str, float]] = []
    for label_idx in range(1, int(n_labels) + 1):
        ys, xs = np.where(labels == int(label_idx))
        if ys.size == 0:
            continue
        y0 = int(np.min(ys))
        y1 = int(np.max(ys)) + 1
        x0 = int(np.min(xs))
        x1 = int(np.max(xs)) + 1
        rows.append({
            'label_id': int(label_idx),
            'area_px': int(ys.size),
            'y0_px': int(y0),
            'y1_px': int(y1),
            'x0_px': int(x0),
            'x1_px': int(x1),
            'center_y_px': float(np.mean(ys)),
            'center_x_px': float(np.mean(xs)),
            'height_px': int(y1 - y0),
            'width_px': int(x1 - x0),
        })
    return rows


def _expand_interval(start: int, stop: int, min_size: int, limit: int) -> Tuple[int, int]:
    start = int(start)
    stop = int(stop)
    size = int(stop - start)
    if size >= int(min_size):
        return max(0, start), min(int(limit), stop)
    center = 0.5 * (float(start) + float(stop))
    half = 0.5 * float(min_size)
    new_start = int(np.floor(center - half))
    new_stop = int(np.ceil(center + half))
    if new_start < 0:
        new_stop += -new_start
        new_start = 0
    if new_stop > int(limit):
        new_start -= int(new_stop - int(limit))
        new_stop = int(limit)
    return max(0, int(new_start)), min(int(limit), int(new_stop))


def _propose_horizontal_cutoff_window_from_dapi(
    left_img: np.ndarray,
    right_img: np.ndarray,
    nominal_dx_px: int,
    seam_band_px: int = 28,
    min_foreground_px: int = 48,
    min_component_px: int = 32,
    max_center_delta_px: float = 72.0,
    pad_y_px: int = 48,
    pad_x_px: int = 24,
    min_window_h_px: int = 96,
    min_window_w_px: int = 72,
) -> Dict[str, object]:
    ovx = int(left_img.shape[1] - nominal_dx_px)
    if ovx < 24:
        return {'success': False, 'reject_reason': 'tiny_overlap'}
    left_overlap = np.asarray(left_img[:, nominal_dx_px:], dtype=np.float32)
    right_overlap = np.asarray(right_img[:, :ovx], dtype=np.float32)
    _, left_mask = _prepare_dapi_overlap(left_overlap)
    _, right_mask = _prepare_dapi_overlap(right_overlap)
    left_fg = int(np.count_nonzero(left_mask))
    right_fg = int(np.count_nonzero(right_mask))
    debug = {
        'success': False,
        'reject_reason': '',
        'foreground_px_a': int(left_fg),
        'foreground_px_b': int(right_fg),
        'overlap_width_px': int(ovx),
        'left_mask': left_mask,
        'right_mask': right_mask,
    }
    if left_fg < int(min_foreground_px) or right_fg < int(min_foreground_px):
        debug['reject_reason'] = 'low_foreground'
        return debug
    left_components = [
        row for row in _component_rows_from_mask(left_mask)
        if int(row['x0_px']) <= int(seam_band_px) and int(row['area_px']) >= int(min_component_px)
    ]
    right_components = [
        row for row in _component_rows_from_mask(right_mask)
        if int(row['x0_px']) <= int(seam_band_px) and int(row['area_px']) >= int(min_component_px)
    ]
    debug['left_components'] = left_components
    debug['right_components'] = right_components
    if not left_components or not right_components:
        debug['reject_reason'] = 'no_seam_touching_components'
        return debug
    best_pair = None
    best_score = float('-inf')
    for comp_a in left_components:
        for comp_b in right_components:
            center_delta = float(abs(float(comp_a['center_y_px']) - float(comp_b['center_y_px'])))
            y_overlap = float(max(0, min(int(comp_a['y1_px']), int(comp_b['y1_px'])) - max(int(comp_a['y0_px']), int(comp_b['y0_px']))))
            if center_delta > float(max_center_delta_px) and y_overlap <= 0:
                continue
            area_ratio = float(min(float(comp_a['area_px']), float(comp_b['area_px'])) / max(float(comp_a['area_px']), float(comp_b['area_px'])))
            pair_score = float(2.0 * y_overlap + 0.05 * min(float(comp_a['area_px']), float(comp_b['area_px'])) + 24.0 * area_ratio - 0.5 * center_delta)
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (comp_a, comp_b)
    if best_pair is None:
        debug['reject_reason'] = 'no_paired_components'
        return debug
    comp_a, comp_b = best_pair
    y0 = min(int(comp_a['y0_px']), int(comp_b['y0_px'])) - int(pad_y_px)
    y1 = max(int(comp_a['y1_px']), int(comp_b['y1_px'])) + int(pad_y_px)
    x0 = 0
    x1 = max(int(comp_a['x1_px']), int(comp_b['x1_px'])) + int(pad_x_px)
    y0, y1 = _expand_interval(y0, y1, int(min_window_h_px), int(left_overlap.shape[0]))
    x0, x1 = _expand_interval(x0, x1, int(min_window_w_px), int(left_overlap.shape[1]))
    debug.update({
        'success': True,
        'chosen_component_a': comp_a,
        'chosen_component_b': comp_b,
        'pair_score': float(best_score),
        'window_y0_px': int(y0),
        'window_y1_px': int(y1),
        'window_x0_px': int(x0),
        'window_x1_px': int(x1),
    })
    return debug


def _propose_vertical_cutoff_window_from_dapi(
    top_img: np.ndarray,
    bottom_img: np.ndarray,
    nominal_dy_px: int,
    seam_band_px: int = 28,
    min_foreground_px: int = 48,
    min_component_px: int = 32,
    max_center_delta_px: float = 72.0,
    pad_y_px: int = 24,
    pad_x_px: int = 48,
    min_window_h_px: int = 72,
    min_window_w_px: int = 96,
) -> Dict[str, object]:
    ovy = int(top_img.shape[0] - nominal_dy_px)
    if ovy < 24:
        return {'success': False, 'reject_reason': 'tiny_overlap'}
    top_overlap = np.asarray(top_img[nominal_dy_px:, :], dtype=np.float32)
    bottom_overlap = np.asarray(bottom_img[:ovy, :], dtype=np.float32)
    _, top_mask = _prepare_dapi_overlap(top_overlap)
    _, bottom_mask = _prepare_dapi_overlap(bottom_overlap)
    top_fg = int(np.count_nonzero(top_mask))
    bottom_fg = int(np.count_nonzero(bottom_mask))
    debug = {
        'success': False,
        'reject_reason': '',
        'foreground_px_a': int(top_fg),
        'foreground_px_b': int(bottom_fg),
        'overlap_height_px': int(ovy),
        'top_mask': top_mask,
        'bottom_mask': bottom_mask,
    }
    if top_fg < int(min_foreground_px) or bottom_fg < int(min_foreground_px):
        debug['reject_reason'] = 'low_foreground'
        return debug
    top_components = [
        row for row in _component_rows_from_mask(top_mask)
        if int(row['y0_px']) <= int(seam_band_px) and int(row['area_px']) >= int(min_component_px)
    ]
    bottom_components = [
        row for row in _component_rows_from_mask(bottom_mask)
        if int(row['y0_px']) <= int(seam_band_px) and int(row['area_px']) >= int(min_component_px)
    ]
    debug['top_components'] = top_components
    debug['bottom_components'] = bottom_components
    if not top_components or not bottom_components:
        debug['reject_reason'] = 'no_seam_touching_components'
        return debug
    best_pair = None
    best_score = float('-inf')
    for comp_a in top_components:
        for comp_b in bottom_components:
            center_delta = float(abs(float(comp_a['center_x_px']) - float(comp_b['center_x_px'])))
            x_overlap = float(max(0, min(int(comp_a['x1_px']), int(comp_b['x1_px'])) - max(int(comp_a['x0_px']), int(comp_b['x0_px']))))
            if center_delta > float(max_center_delta_px) and x_overlap <= 0:
                continue
            area_ratio = float(min(float(comp_a['area_px']), float(comp_b['area_px'])) / max(float(comp_a['area_px']), float(comp_b['area_px'])))
            pair_score = float(2.0 * x_overlap + 0.05 * min(float(comp_a['area_px']), float(comp_b['area_px'])) + 24.0 * area_ratio - 0.5 * center_delta)
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (comp_a, comp_b)
    if best_pair is None:
        debug['reject_reason'] = 'no_paired_components'
        return debug
    comp_a, comp_b = best_pair
    y0 = 0
    y1 = max(int(comp_a['y1_px']), int(comp_b['y1_px'])) + int(pad_y_px)
    x0 = min(int(comp_a['x0_px']), int(comp_b['x0_px'])) - int(pad_x_px)
    x1 = max(int(comp_a['x1_px']), int(comp_b['x1_px'])) + int(pad_x_px)
    y0, y1 = _expand_interval(y0, y1, int(min_window_h_px), int(top_overlap.shape[0]))
    x0, x1 = _expand_interval(x0, x1, int(min_window_w_px), int(top_overlap.shape[1]))
    debug.update({
        'success': True,
        'chosen_component_a': comp_a,
        'chosen_component_b': comp_b,
        'pair_score': float(best_score),
        'window_y0_px': int(y0),
        'window_y1_px': int(y1),
        'window_x0_px': int(x0),
        'window_x1_px': int(x1),
    })
    return debug


def _estimate_horizontal_delta_dapi_guided_bf(
    left_dapi: np.ndarray,
    right_dapi: np.ndarray,
    left_bf: np.ndarray,
    right_bf: np.ndarray,
    nominal_dx_px: int,
    upsample_factor: int = 20,
    max_cross_axis_shift_px: float = 12.0,
    max_primary_shift_from_nominal_px: float = 48.0,
) -> Optional[Dict[str, object]]:
    proposal = _propose_horizontal_cutoff_window_from_dapi(left_dapi, right_dapi, nominal_dx_px=int(nominal_dx_px))
    if not bool(proposal.get('success', False)):
        return None
    ovx = int(left_bf.shape[1] - nominal_dx_px)
    left_overlap = np.asarray(left_bf[:, nominal_dx_px:], dtype=np.float32)
    right_overlap = np.asarray(right_bf[:, :ovx], dtype=np.float32)
    y0 = int(proposal['window_y0_px'])
    y1 = int(proposal['window_y1_px'])
    x0 = int(proposal['window_x0_px'])
    x1 = int(proposal['window_x1_px'])
    left_patch = np.asarray(left_overlap[y0:y1, x0:x1], dtype=np.float32)
    right_patch = np.asarray(right_overlap[y0:y1, x0:x1], dtype=np.float32)
    left_feat = _prepare_alignment_overlap(left_patch)
    right_feat = _prepare_alignment_overlap(right_patch)
    signal_score = float(np.nanstd(left_feat) + np.nanstd(right_feat))
    if not np.isfinite(signal_score) or signal_score <= 1e-6:
        return None
    shift, error, _ = phase_cross_correlation(left_feat, right_feat, upsample_factor=int(upsample_factor))
    cross_axis_shift_px = float(shift[0])
    primary_shift_from_nominal_px = float(shift[1])
    if abs(cross_axis_shift_px) > float(max_cross_axis_shift_px):
        return None
    if abs(primary_shift_from_nominal_px) > float(max_primary_shift_from_nominal_px):
        return None
    error = float(error) if np.isfinite(error) else 1.0
    quality = float((signal_score + 0.02 * float(proposal.get('pair_score', 0.0))) / max(error, 1e-3) / (1.0 + (abs(cross_axis_shift_px) / 4.0) ** 2) / (1.0 + (abs(primary_shift_from_nominal_px) / 24.0) ** 2))
    return {
        'delta_y_px': float(cross_axis_shift_px),
        'delta_x_px': float(nominal_dx_px + primary_shift_from_nominal_px),
        'registration_error': float(error),
        'signal_score': float(signal_score),
        'quality': float(quality),
        'foreground_px_a': int(proposal.get('foreground_px_a', 0)),
        'foreground_px_b': int(proposal.get('foreground_px_b', 0)),
        'dapi_pair_score': float(proposal.get('pair_score', 0.0)),
        'window_y0_px': int(y0),
        'window_y1_px': int(y1),
        'window_x0_px': int(x0),
        'window_x1_px': int(x1),
    }


def _estimate_vertical_delta_dapi_guided_bf(
    top_dapi: np.ndarray,
    bottom_dapi: np.ndarray,
    top_bf: np.ndarray,
    bottom_bf: np.ndarray,
    nominal_dy_px: int,
    upsample_factor: int = 20,
    max_cross_axis_shift_px: float = 12.0,
    max_primary_shift_from_nominal_px: float = 48.0,
) -> Optional[Dict[str, object]]:
    proposal = _propose_vertical_cutoff_window_from_dapi(top_dapi, bottom_dapi, nominal_dy_px=int(nominal_dy_px))
    if not bool(proposal.get('success', False)):
        return None
    ovy = int(top_bf.shape[0] - nominal_dy_px)
    top_overlap = np.asarray(top_bf[nominal_dy_px:, :], dtype=np.float32)
    bottom_overlap = np.asarray(bottom_bf[:ovy, :], dtype=np.float32)
    y0 = int(proposal['window_y0_px'])
    y1 = int(proposal['window_y1_px'])
    x0 = int(proposal['window_x0_px'])
    x1 = int(proposal['window_x1_px'])
    top_patch = np.asarray(top_overlap[y0:y1, x0:x1], dtype=np.float32)
    bottom_patch = np.asarray(bottom_overlap[y0:y1, x0:x1], dtype=np.float32)
    top_feat = _prepare_alignment_overlap(top_patch)
    bottom_feat = _prepare_alignment_overlap(bottom_patch)
    signal_score = float(np.nanstd(top_feat) + np.nanstd(bottom_feat))
    if not np.isfinite(signal_score) or signal_score <= 1e-6:
        return None
    shift, error, _ = phase_cross_correlation(top_feat, bottom_feat, upsample_factor=int(upsample_factor))
    primary_shift_from_nominal_px = float(shift[0])
    cross_axis_shift_px = float(shift[1])
    if abs(cross_axis_shift_px) > float(max_cross_axis_shift_px):
        return None
    if abs(primary_shift_from_nominal_px) > float(max_primary_shift_from_nominal_px):
        return None
    error = float(error) if np.isfinite(error) else 1.0
    quality = float((signal_score + 0.02 * float(proposal.get('pair_score', 0.0))) / max(error, 1e-3) / (1.0 + (abs(cross_axis_shift_px) / 4.0) ** 2) / (1.0 + (abs(primary_shift_from_nominal_px) / 24.0) ** 2))
    return {
        'delta_y_px': float(nominal_dy_px + primary_shift_from_nominal_px),
        'delta_x_px': float(cross_axis_shift_px),
        'registration_error': float(error),
        'signal_score': float(signal_score),
        'quality': float(quality),
        'foreground_px_a': int(proposal.get('foreground_px_a', 0)),
        'foreground_px_b': int(proposal.get('foreground_px_b', 0)),
        'dapi_pair_score': float(proposal.get('pair_score', 0.0)),
        'window_y0_px': int(y0),
        'window_y1_px': int(y1),
        'window_x0_px': int(x0),
        'window_x1_px': int(x1),
    }
def _estimate_horizontal_delta_dapi(
    left_img: np.ndarray,
    right_img: np.ndarray,
    nominal_dx_px: int,
    upsample_factor: int = 20,
    max_cross_axis_shift_px: float = 18.0,
    max_primary_shift_from_nominal_px: float = 64.0,
    min_foreground_px: int = 48,
):
    ovx = int(left_img.shape[1] - nominal_dx_px)
    if ovx < 24:
        return None
    left_overlap = np.asarray(left_img[:, nominal_dx_px:], dtype=np.float32)
    right_overlap = np.asarray(right_img[:, :ovx], dtype=np.float32)
    left_feat, left_mask = _prepare_dapi_overlap(left_overlap)
    right_feat, right_mask = _prepare_dapi_overlap(right_overlap)
    left_fg = int(np.count_nonzero(left_mask))
    right_fg = int(np.count_nonzero(right_mask))
    if left_fg < int(min_foreground_px) or right_fg < int(min_foreground_px):
        return None
    shift, error, _ = phase_cross_correlation(
        left_feat,
        right_feat,
        reference_mask=left_mask,
        moving_mask=right_mask,
        upsample_factor=int(upsample_factor),
        overlap_ratio=0.2,
    )
    if not np.all(np.isfinite(shift)):
        return None
    cross_axis_shift_px = float(shift[0])
    primary_shift_from_nominal_px = float(shift[1])
    if abs(cross_axis_shift_px) > float(max_cross_axis_shift_px):
        return None
    if abs(primary_shift_from_nominal_px) > float(max_primary_shift_from_nominal_px):
        return None
    error = float(error) if np.isfinite(error) else 1.0
    signal_score = float(np.sqrt(float(left_fg) * float(right_fg)))
    quality = float(
        signal_score
        / max(error, 5e-2)
        / (1.0 + (abs(cross_axis_shift_px) / 5.0) ** 2)
        / (1.0 + (abs(primary_shift_from_nominal_px) / 24.0) ** 2)
    )
    return {
        'delta_y_px': float(cross_axis_shift_px),
        'delta_x_px': float(nominal_dx_px + primary_shift_from_nominal_px),
        'registration_error': error,
        'signal_score': signal_score,
        'quality': quality,
        'foreground_px_a': int(left_fg),
        'foreground_px_b': int(right_fg),
    }


def _estimate_vertical_delta_dapi(
    top_img: np.ndarray,
    bottom_img: np.ndarray,
    nominal_dy_px: int,
    upsample_factor: int = 20,
    max_cross_axis_shift_px: float = 18.0,
    max_primary_shift_from_nominal_px: float = 64.0,
    min_foreground_px: int = 48,
):
    ovy = int(top_img.shape[0] - nominal_dy_px)
    if ovy < 24:
        return None
    top_overlap = np.asarray(top_img[nominal_dy_px:, :], dtype=np.float32)
    bottom_overlap = np.asarray(bottom_img[:ovy, :], dtype=np.float32)
    top_feat, top_mask = _prepare_dapi_overlap(top_overlap)
    bottom_feat, bottom_mask = _prepare_dapi_overlap(bottom_overlap)
    top_fg = int(np.count_nonzero(top_mask))
    bottom_fg = int(np.count_nonzero(bottom_mask))
    if top_fg < int(min_foreground_px) or bottom_fg < int(min_foreground_px):
        return None
    shift, error, _ = phase_cross_correlation(
        top_feat,
        bottom_feat,
        reference_mask=top_mask,
        moving_mask=bottom_mask,
        upsample_factor=int(upsample_factor),
        overlap_ratio=0.2,
    )
    if not np.all(np.isfinite(shift)):
        return None
    primary_shift_from_nominal_px = float(shift[0])
    cross_axis_shift_px = float(shift[1])
    if abs(cross_axis_shift_px) > float(max_cross_axis_shift_px):
        return None
    if abs(primary_shift_from_nominal_px) > float(max_primary_shift_from_nominal_px):
        return None
    error = float(error) if np.isfinite(error) else 1.0
    signal_score = float(np.sqrt(float(top_fg) * float(bottom_fg)))
    quality = float(
        signal_score
        / max(error, 5e-2)
        / (1.0 + (abs(cross_axis_shift_px) / 5.0) ** 2)
        / (1.0 + (abs(primary_shift_from_nominal_px) / 24.0) ** 2)
    )
    return {
        'delta_y_px': float(nominal_dy_px + primary_shift_from_nominal_px),
        'delta_x_px': float(cross_axis_shift_px),
        'registration_error': error,
        'signal_score': signal_score,
        'quality': quality,
        'foreground_px_a': int(top_fg),
        'foreground_px_b': int(bottom_fg),
    }


def _estimate_horizontal_delta(left_img: np.ndarray, right_img: np.ndarray, nominal_dx_px: int, upsample_factor: int = 20):
    ovx = int(left_img.shape[1] - nominal_dx_px)
    if ovx < 24:
        return 0.0, float(nominal_dx_px)
    shift, _, _ = phase_cross_correlation(
        _zscore_norm(left_img[:, nominal_dx_px:]),
        _zscore_norm(right_img[:, :ovx]),
        upsample_factor=int(upsample_factor),
    )
    return float(shift[0]), float(nominal_dx_px + shift[1])


def _estimate_vertical_delta(top_img: np.ndarray, bottom_img: np.ndarray, nominal_dy_px: int, upsample_factor: int = 20):
    ovy = int(top_img.shape[0] - nominal_dy_px)
    if ovy < 24:
        return float(nominal_dy_px), 0.0
    shift, _, _ = phase_cross_correlation(
        _zscore_norm(top_img[nominal_dy_px:, :]),
        _zscore_norm(bottom_img[:ovy, :]),
        upsample_factor=int(upsample_factor),
    )
    return float(nominal_dy_px + shift[0]), float(shift[1])


def _solve_grid_positions(n_nodes: int, edges: Sequence[Tuple[int, int, float]], nominal_vals: np.ndarray, max_shift_from_nominal_px: int = 96) -> np.ndarray:
    rows = []
    rhs = []
    anchor = np.zeros(n_nodes, dtype=np.float64)
    anchor[0] = 1.0
    rows.append(anchor)
    rhs.append(0.0)
    for edge in edges:
        if len(edge) == 4:
            i, j, delta, weight = edge
        else:
            i, j, delta = edge
            weight = 1.0
        scale = float(np.sqrt(max(float(weight), 1e-6)))
        eq = np.zeros(n_nodes, dtype=np.float64)
        eq[int(j)] = 1.0 * scale
        eq[int(i)] = -1.0 * scale
        rows.append(eq)
        rhs.append(float(delta) * scale)
    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.float64)
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    clipped = np.clip(sol, nominal_vals - float(max_shift_from_nominal_px), nominal_vals + float(max_shift_from_nominal_px))
    clipped -= clipped[0]
    return np.round(clipped).astype(int)


def _extract_corner_patch(
    img: np.ndarray,
    corner: str,
    patch_size_px: int = 192,
) -> Tuple[np.ndarray, int, int]:
    arr = np.asarray(img, dtype=np.float32)
    H, W = arr.shape
    size = int(max(64, min(int(patch_size_px), H, W)))
    if corner == 'top_left':
        y0, x0 = 0, 0
    elif corner == 'top_right':
        y0, x0 = 0, W - size
    elif corner == 'bottom_left':
        y0, x0 = H - size, 0
    elif corner == 'bottom_right':
        y0, x0 = H - size, W - size
    else:
        raise ValueError(f'Unknown corner: {corner}')
    return arr[y0:y0 + size, x0:x0 + size], int(y0), int(x0)


def _detect_corner_circle_candidates(
    img: np.ndarray,
    corner: str,
    patch_size_px: int = 192,
    radius_min_px: int = 26,
    radius_max_px: int = 64,
    radius_step_px: int = 2,
    max_candidates: int = 8,
) -> List[Dict[str, float]]:
    patch, off_y, off_x = _extract_corner_patch(img, corner=corner, patch_size_px=int(patch_size_px))
    if patch.size == 0 or min(patch.shape) < 64:
        return []
    arr = robust_rescale(patch, p_low=1.0, p_high=99.8)
    flat = arr - ndi.gaussian_filter(arr, sigma=12.0)
    flat = robust_rescale(flat, p_low=2.0, p_high=99.8)
    edges = feature.canny(flat, sigma=1.6, low_threshold=0.08, high_threshold=0.22)
    if int(np.count_nonzero(edges)) < 24:
        edges = feature.canny(arr, sigma=1.6, low_threshold=0.08, high_threshold=0.22)
    if int(np.count_nonzero(edges)) < 24:
        return []
    radii = np.arange(int(radius_min_px), int(radius_max_px) + 1, int(max(1, radius_step_px)))
    if radii.size == 0:
        return []
    hough = transform.hough_circle(edges, radii)
    accums, cx, cy, rad = transform.hough_circle_peaks(
        hough,
        radii,
        total_num_peaks=int(max_candidates),
        normalize=True,
    )
    rows: List[Dict[str, float]] = []
    for acc, xx, yy, rr in zip(accums, cx, cy, rad):
        rows.append({
            'score': float(acc),
            'center_local_y_px': float(off_y + yy),
            'center_local_x_px': float(off_x + xx),
            'radius_px': float(rr),
        })
    return rows


def _detect_strip_circle_candidates(
    img: np.ndarray,
    radius_min_px: int = 20,
    radius_max_px: int = 64,
    radius_step_px: int = 2,
    max_candidates: int = 12,
) -> List[Dict[str, float]]:
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0 or min(arr.shape) < 64:
        return []
    norm = robust_rescale(arr, p_low=1.0, p_high=99.8)
    flat = norm - ndi.gaussian_filter(norm, sigma=12.0)
    flat = robust_rescale(flat, p_low=2.0, p_high=99.8)
    edges = feature.canny(flat, sigma=1.6, low_threshold=0.08, high_threshold=0.22)
    if int(np.count_nonzero(edges)) < 24:
        edges = feature.canny(norm, sigma=1.6, low_threshold=0.08, high_threshold=0.22)
    if int(np.count_nonzero(edges)) < 24:
        return []
    radii = np.arange(int(radius_min_px), int(radius_max_px) + 1, int(max(1, radius_step_px)))
    if radii.size == 0:
        return []
    hough = transform.hough_circle(edges, radii)
    accums, cx, cy, rad = transform.hough_circle_peaks(
        hough,
        radii,
        total_num_peaks=int(max_candidates),
        normalize=True,
    )
    rows: List[Dict[str, float]] = []
    for acc, xx, yy, rr in zip(accums, cx, cy, rad):
        rows.append({
            'score': float(acc),
            'center_local_y_px': float(yy),
            'center_local_x_px': float(xx),
            'radius_px': float(rr),
        })
    return rows


def _estimate_horizontal_delta_bf_circle(
    left_img: np.ndarray,
    right_img: np.ndarray,
    nominal_dx_px: int,
    max_cross_axis_shift_px: float = 24.0,
    max_primary_shift_from_nominal_px: float = 48.0,
    max_center_distance_px: float = 48.0,
    max_radius_diff_px: float = 14.0,
):
    ovx = int(left_img.shape[1] - nominal_dx_px)
    if ovx < 24:
        return None
    left_overlap = np.asarray(left_img[:, nominal_dx_px:], dtype=np.float32)
    right_overlap = np.asarray(right_img[:, :ovx], dtype=np.float32)
    left_candidates = _detect_strip_circle_candidates(left_overlap)
    right_candidates = _detect_strip_circle_candidates(right_overlap)
    if not left_candidates or not right_candidates:
        return None
    best = None
    best_score = float('-inf')
    for left_cand in left_candidates:
        for right_cand in right_candidates:
            delta_y_px = float(left_cand['center_local_y_px'] - right_cand['center_local_y_px'])
            primary_shift_from_nominal_px = float(left_cand['center_local_x_px'] - right_cand['center_local_x_px'])
            if abs(delta_y_px) > float(max_cross_axis_shift_px):
                continue
            if abs(primary_shift_from_nominal_px) > float(max_primary_shift_from_nominal_px):
                continue
            center_distance = float(np.hypot(delta_y_px, primary_shift_from_nominal_px))
            if center_distance > float(max_center_distance_px):
                continue
            radius_diff = float(abs(float(left_cand['radius_px']) - float(right_cand['radius_px'])))
            if radius_diff > float(max_radius_diff_px):
                continue
            score_mean = float((float(left_cand['score']) + float(right_cand['score'])) / 2.0)
            quality = float(
                score_mean
                * 100.0
                / (1.0 + (center_distance / 8.0) ** 2)
                / (1.0 + (radius_diff / 4.0) ** 2)
            )
            if quality > best_score:
                best_score = quality
                best = {
                    'delta_y_px': float(delta_y_px),
                    'delta_x_px': float(nominal_dx_px + primary_shift_from_nominal_px),
                    'registration_error': float(center_distance),
                    'signal_score': float(score_mean),
                    'quality': float(quality),
                    'center_distance_px': float(center_distance),
                    'radius_diff_px': float(radius_diff),
                }
    return best


def _estimate_vertical_delta_bf_circle(
    top_img: np.ndarray,
    bottom_img: np.ndarray,
    nominal_dy_px: int,
    max_cross_axis_shift_px: float = 24.0,
    max_primary_shift_from_nominal_px: float = 48.0,
    max_center_distance_px: float = 48.0,
    max_radius_diff_px: float = 14.0,
):
    ovy = int(top_img.shape[0] - nominal_dy_px)
    if ovy < 24:
        return None
    top_overlap = np.asarray(top_img[nominal_dy_px:, :], dtype=np.float32)
    bottom_overlap = np.asarray(bottom_img[:ovy, :], dtype=np.float32)
    top_candidates = _detect_strip_circle_candidates(top_overlap)
    bottom_candidates = _detect_strip_circle_candidates(bottom_overlap)
    if not top_candidates or not bottom_candidates:
        return None
    best = None
    best_score = float('-inf')
    for top_cand in top_candidates:
        for bottom_cand in bottom_candidates:
            primary_shift_from_nominal_px = float(top_cand['center_local_y_px'] - bottom_cand['center_local_y_px'])
            delta_x_px = float(top_cand['center_local_x_px'] - bottom_cand['center_local_x_px'])
            if abs(delta_x_px) > float(max_cross_axis_shift_px):
                continue
            if abs(primary_shift_from_nominal_px) > float(max_primary_shift_from_nominal_px):
                continue
            center_distance = float(np.hypot(primary_shift_from_nominal_px, delta_x_px))
            if center_distance > float(max_center_distance_px):
                continue
            radius_diff = float(abs(float(top_cand['radius_px']) - float(bottom_cand['radius_px'])))
            if radius_diff > float(max_radius_diff_px):
                continue
            score_mean = float((float(top_cand['score']) + float(bottom_cand['score'])) / 2.0)
            quality = float(
                score_mean
                * 100.0
                / (1.0 + (center_distance / 8.0) ** 2)
                / (1.0 + (radius_diff / 4.0) ** 2)
            )
            if quality > best_score:
                best_score = quality
                best = {
                    'delta_y_px': float(nominal_dy_px + primary_shift_from_nominal_px),
                    'delta_x_px': float(delta_x_px),
                    'registration_error': float(center_distance),
                    'signal_score': float(score_mean),
                    'quality': float(quality),
                    'center_distance_px': float(center_distance),
                    'radius_diff_px': float(radius_diff),
                }
    return best


def _estimate_corner_anchor_constraints(
    anchor_tile_map: Dict[Tuple[int, int, int], np.ndarray],
    anchor_z_values: Sequence[int],
    full_y_starts: Sequence[int],
    full_x_starts: Sequence[int],
    row_indices: Sequence[int],
    col_indices: Sequence[int],
    r_local: int,
    c_local: int,
    patch_size_px: int = 192,
    radius_min_px: int = 26,
    radius_max_px: int = 64,
    max_role_candidates: int = 8,
    max_cluster_spread_px: float = 48.0,
    max_radius_spread_px: float = 16.0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    role_specs = [
        ('top_left', (int(r_local), int(c_local)), 'bottom_right'),
        ('top_right', (int(r_local), int(c_local + 1)), 'bottom_left'),
        ('bottom_left', (int(r_local + 1), int(c_local)), 'top_right'),
        ('bottom_right', (int(r_local + 1), int(c_local + 1)), 'top_left'),
    ]
    role_candidates: Dict[str, List[Dict[str, object]]] = {}
    per_role_max_score: Dict[str, float] = {}
    for role_name, (rr_local, cc_local), corner_name in role_specs:
        rr_global = int(row_indices[rr_local])
        cc_global = int(col_indices[cc_local])
        y0 = int(full_y_starts[rr_global])
        x0 = int(full_x_starts[cc_global])
        nominal_y = float(y0 - full_y_starts[0])
        nominal_x = float(x0 - full_x_starts[0])
        candidates: List[Dict[str, object]] = []
        for z_idx in anchor_z_values:
            key = (int(z_idx), int(y0), int(x0))
            if key not in anchor_tile_map:
                continue
            for cand in _detect_corner_circle_candidates(
                anchor_tile_map[key],
                corner=corner_name,
                patch_size_px=int(patch_size_px),
                radius_min_px=int(radius_min_px),
                radius_max_px=int(radius_max_px),
                max_candidates=int(max_role_candidates),
            ):
                candidates.append({
                    'role_name': str(role_name),
                    'corner_name': str(corner_name),
                    'local_tile_row': int(rr_local),
                    'local_tile_col': int(cc_local),
                    'global_tile_row': int(rr_global),
                    'global_tile_col': int(cc_global),
                    'z_index': int(z_idx),
                    'score': float(cand['score']),
                    'center_local_y_px': float(cand['center_local_y_px']),
                    'center_local_x_px': float(cand['center_local_x_px']),
                    'radius_px': float(cand['radius_px']),
                    'global_center_y_px': float(nominal_y + float(cand['center_local_y_px'])),
                    'global_center_x_px': float(nominal_x + float(cand['center_local_x_px'])),
                })
        if not candidates:
            return [], {
                'corner_found': False,
                'reason': f'missing_{role_name}',
            }
        max_score = max(float(c['score']) for c in candidates)
        per_role_max_score[role_name] = max(float(max_score), 1e-6)
        candidates.sort(key=lambda row: row['score'], reverse=True)
        role_candidates[role_name] = candidates[: int(max_role_candidates)]

    best_combo: Optional[Tuple[Dict[str, object], ...]] = None
    best_cluster_score = float('-inf')
    best_cluster_meta: Dict[str, object] = {}
    role_order = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    for combo in product(*[role_candidates[role_name] for role_name in role_order]):
        gy = np.asarray([float(c['global_center_y_px']) for c in combo], dtype=np.float64)
        gx = np.asarray([float(c['global_center_x_px']) for c in combo], dtype=np.float64)
        rr = np.asarray([float(c['radius_px']) for c in combo], dtype=np.float64)
        cy = float(np.mean(gy))
        cx = float(np.mean(gx))
        spread = float(np.max(np.hypot(gy - cy, gx - cx)))
        if spread > float(max_cluster_spread_px):
            continue
        radius_spread = float(np.max(rr) - np.min(rr))
        if radius_spread > float(max_radius_spread_px):
            continue
        score_norm = np.asarray(
            [float(c['score']) / float(per_role_max_score[str(c['role_name'])]) for c in combo],
            dtype=np.float64,
        )
        cluster_score = float(np.mean(score_norm) - 0.01 * spread - 0.01 * radius_spread)
        if cluster_score > best_cluster_score:
            best_combo = combo
            best_cluster_score = cluster_score
            best_cluster_meta = {
                'anchor_center_y_px': float(cy),
                'anchor_center_x_px': float(cx),
                'anchor_spread_px': float(spread),
                'anchor_radius_median_px': float(np.median(rr)),
                'anchor_radius_spread_px': float(radius_spread),
                'anchor_score_mean_norm': float(np.mean(score_norm)),
            }

    if best_combo is None:
        return [], {
            'corner_found': False,
            'reason': 'no_cluster',
        }

    chosen = {str(c['role_name']): dict(c) for c in best_combo}
    anchor_weight = float(np.clip(3.0 + 2.0 * best_cluster_meta['anchor_score_mean_norm'] - 0.03 * best_cluster_meta['anchor_spread_px'], 2.0, 5.0))

    def _constraint(role_a: str, role_b: str, edge_axis: str, edge_name: str) -> Dict[str, object]:
        a = chosen[role_a]
        b = chosen[role_b]
        return {
            'edge_type': str(edge_name),
            'edge_axis': str(edge_axis),
            'local_r0': int(a['local_tile_row']),
            'local_c0': int(a['local_tile_col']),
            'local_r1': int(b['local_tile_row']),
            'local_c1': int(b['local_tile_col']),
            'global_r0': int(a['global_tile_row']),
            'global_c0': int(a['global_tile_col']),
            'global_r1': int(b['global_tile_row']),
            'global_c1': int(b['global_tile_col']),
            'delta_y_px': float(float(a['center_local_y_px']) - float(b['center_local_y_px'])),
            'delta_x_px': float(float(a['center_local_x_px']) - float(b['center_local_x_px'])),
            'weight': float(anchor_weight),
            'anchor_center_y_px': float(best_cluster_meta['anchor_center_y_px']),
            'anchor_center_x_px': float(best_cluster_meta['anchor_center_x_px']),
            'anchor_spread_px': float(best_cluster_meta['anchor_spread_px']),
            'anchor_radius_median_px': float(best_cluster_meta['anchor_radius_median_px']),
            'anchor_score_mean_norm': float(best_cluster_meta['anchor_score_mean_norm']),
            'z_index_a': int(a['z_index']),
            'z_index_b': int(b['z_index']),
            'role_a': str(role_a),
            'role_b': str(role_b),
        }

    constraints = [
        _constraint('top_left', 'top_right', edge_axis='horizontal', edge_name='corner_anchor_horizontal_top'),
        _constraint('bottom_left', 'bottom_right', edge_axis='horizontal', edge_name='corner_anchor_horizontal_bottom'),
        _constraint('top_left', 'bottom_left', edge_axis='vertical', edge_name='corner_anchor_vertical_left'),
        _constraint('top_right', 'bottom_right', edge_axis='vertical', edge_name='corner_anchor_vertical_right'),
    ]
    return constraints, {
        'corner_found': True,
        'anchor_weight': float(anchor_weight),
        **best_cluster_meta,
    }


def estimate_grid_registered_positions(
    bf_tile_map: Dict[Tuple[int, int, int], np.ndarray],
    y_starts: Sequence[int],
    x_starts: Sequence[int],
    z_ref: int,
    upsample_factor: int = 20,
    max_shift_from_nominal_px: int = 96,
):
    y_starts = [int(v) for v in y_starts]
    x_starts = [int(v) for v in x_starts]
    n_rows = len(y_starts)
    n_cols = len(x_starts)
    if n_rows < 2 or n_cols < 2:
        raise RuntimeError("Need at least a 2x2 tile grid for registration.")

    node_index = {(r, c): r * n_cols + c for r in range(n_rows) for c in range(n_cols)}
    nominal_y = np.array([y_starts[r] - y_starts[0] for r in range(n_rows) for _ in range(n_cols)], dtype=np.float64)
    nominal_x = np.array([x_starts[c] - x_starts[0] for _ in range(n_rows) for c in range(n_cols)], dtype=np.float64)

    y_edges = []
    x_edges = []
    edge_rows = []

    for r in range(n_rows):
        for c in range(n_cols - 1):
            a = bf_tile_map[(int(z_ref), int(y_starts[r]), int(x_starts[c]))]
            b = bf_tile_map[(int(z_ref), int(y_starts[r]), int(x_starts[c + 1]))]
            nominal_dx = int(x_starts[c + 1] - x_starts[c])
            dy, dx = _estimate_horizontal_delta(a, b, nominal_dx_px=nominal_dx, upsample_factor=int(upsample_factor))
            i = node_index[(r, c)]
            j = node_index[(r, c + 1)]
            y_edges.append((i, j, dy))
            x_edges.append((i, j, dx))
            edge_rows.append({
                'edge_type': 'horizontal',
                'r0': int(r), 'c0': int(c), 'r1': int(r), 'c1': int(c + 1),
                'delta_y_px': float(dy), 'delta_x_px': float(dx), 'nominal_dx_px': int(nominal_dx), 'nominal_dy_px': 0,
            })

    for r in range(n_rows - 1):
        for c in range(n_cols):
            a = bf_tile_map[(int(z_ref), int(y_starts[r]), int(x_starts[c]))]
            b = bf_tile_map[(int(z_ref), int(y_starts[r + 1]), int(x_starts[c]))]
            nominal_dy = int(y_starts[r + 1] - y_starts[r])
            dy, dx = _estimate_vertical_delta(a, b, nominal_dy_px=nominal_dy, upsample_factor=int(upsample_factor))
            i = node_index[(r, c)]
            j = node_index[(r + 1, c)]
            y_edges.append((i, j, dy))
            x_edges.append((i, j, dx))
            edge_rows.append({
                'edge_type': 'vertical',
                'r0': int(r), 'c0': int(c), 'r1': int(r + 1), 'c1': int(c),
                'delta_y_px': float(dy), 'delta_x_px': float(dx), 'nominal_dx_px': 0, 'nominal_dy_px': int(nominal_dy),
            })

    solved_y = _solve_grid_positions(len(node_index), y_edges, nominal_vals=nominal_y, max_shift_from_nominal_px=int(max_shift_from_nominal_px))
    solved_x = _solve_grid_positions(len(node_index), x_edges, nominal_vals=nominal_x, max_shift_from_nominal_px=int(max_shift_from_nominal_px))

    placements = {}
    placement_rows = []
    for (r, c), idx in node_index.items():
        yy = int(solved_y[idx])
        xx = int(solved_x[idx])
        placements[(int(r), int(c))] = (yy, xx)
        placement_rows.append({
            'tile_row': int(r),
            'tile_col': int(c),
            'raw_y_start_px': int(y_starts[r]),
            'raw_x_start_px': int(x_starts[c]),
            'nominal_y_px': int(nominal_y[idx]),
            'nominal_x_px': int(nominal_x[idx]),
            'registered_y_px': int(yy),
            'registered_x_px': int(xx),
            'delta_from_nominal_y_px': int(yy - int(nominal_y[idx])),
            'delta_from_nominal_x_px': int(xx - int(nominal_x[idx])),
        })
    placement_df = pd.DataFrame(placement_rows).sort_values(['tile_row', 'tile_col']).reset_index(drop=True)
    edge_df = pd.DataFrame(edge_rows)
    return placements, placement_df, edge_df


def compose_hard_mosaic(tiles: Dict[Tuple[int, int], np.ndarray], placements: Dict[Tuple[int, int], Tuple[int, int]], draw_order: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    if not tiles:
        raise ValueError('No tiles provided for mosaic composition.')
    tile_h, tile_w = next(iter(tiles.values())).shape
    keys = list(draw_order) if draw_order is not None else sorted(placements.keys())
    ys = [placements[k][0] for k in placements]
    xs = [placements[k][1] for k in placements]
    miny = int(min(ys))
    minx = int(min(xs))
    maxy = int(max(y + tile_h for y in ys))
    maxx = int(max(x + tile_w for x in xs))
    out = np.zeros((maxy - miny, maxx - minx), dtype=next(iter(tiles.values())).dtype)
    for key in keys:
        if key not in tiles or key not in placements:
            continue
        y, x = placements[key]
        yy = int(y - miny)
        xx = int(x - minx)
        out[yy:yy + tile_h, xx:xx + tile_w] = tiles[key]
    return out


def restitch_single_channel_stack(
    tile_map: Dict[Tuple[int, int, int], np.ndarray],
    y_starts: Sequence[int],
    x_starts: Sequence[int],
    z_values: Sequence[int],
    placements: Dict[Tuple[int, int], Tuple[int, int]],
) -> np.ndarray:
    y_starts = [int(v) for v in y_starts]
    x_starts = [int(v) for v in x_starts]
    out = []
    for z_idx in z_values:
        tiles = {}
        for r, y0 in enumerate(y_starts):
            for c, x0 in enumerate(x_starts):
                tiles[(int(r), int(c))] = tile_map[(int(z_idx), int(y0), int(x0))]
        out.append(compose_hard_mosaic(tiles=tiles, placements=placements))
    return np.stack(out, axis=0)


def choose_focus_z_index(stack_zyx: np.ndarray, crop_fraction: float = 0.5) -> Dict[str, float]:
    H, W = stack_zyx.shape[1:]
    ch = max(64, int(round(H * float(crop_fraction))))
    cw = max(64, int(round(W * float(crop_fraction))))
    y0 = max(0, (H - ch) // 2)
    x0 = max(0, (W - cw) // 2)
    best = None
    for z_idx in range(int(stack_zyx.shape[0])):
        crop = np.asarray(stack_zyx[z_idx, y0:y0 + ch, x0:x0 + cw], dtype=np.float32)
        crop = crop - float(crop.mean())
        grad = filters.sobel(crop)
        score = float(np.mean(grad ** 2))
        cand = {'focus_z_index': int(z_idx), 'focus_score': score}
        if best is None or score > best['focus_score']:
            best = cand
    if best is None:
        raise RuntimeError('Failed to choose focus z-index.')
    return best


def choose_bright_z_index(stack_zyx: np.ndarray, top_fraction: float = 0.0002, smooth_sigma: float = 1.0) -> Dict[str, float]:
    best = None
    for z_idx in range(int(stack_zyx.shape[0])):
        img = filters.gaussian(np.asarray(stack_zyx[z_idx], dtype=np.float32), sigma=float(smooth_sigma), preserve_range=True)
        vals = img[np.isfinite(img)].ravel()
        if vals.size == 0:
            continue
        k = int(max(32, round(vals.size * float(top_fraction))))
        k = min(k, int(vals.size))
        top = np.partition(vals, vals.size - k)[-k:]
        score = float(np.mean(top))
        cand = {'bright_z_index': int(z_idx), 'bright_score': score}
        if best is None or score > best['bright_score']:
            best = cand
    if best is None:
        raise RuntimeError('Failed to choose bright z-index.')
    return best


def estimate_global_translation(reference_img: np.ndarray, moving_img: np.ndarray, downsample: int = 8, upsample_factor: int = 10) -> Dict[str, float]:
    ref = robust_rescale(reference_img)[::int(downsample), ::int(downsample)]
    mov = robust_rescale(moving_img)[::int(downsample), ::int(downsample)]
    H = min(ref.shape[0], mov.shape[0])
    W = min(ref.shape[1], mov.shape[1])
    ref = _zscore_norm(ref[:H, :W])
    mov = _zscore_norm(mov[:H, :W])
    shift, error, _ = phase_cross_correlation(ref, mov, upsample_factor=int(upsample_factor))
    return {
        'shift_y_px': float(shift[0] * int(downsample)),
        'shift_x_px': float(shift[1] * int(downsample)),
        'registration_error': float(error),
    }


def translate_integer(img: np.ndarray, shift_y_px: int, shift_x_px: int, fill_value: float = 0.0) -> np.ndarray:
    arr = np.asarray(img)
    out = np.full(arr.shape, fill_value, dtype=arr.dtype)
    src_y0 = max(0, -int(shift_y_px))
    src_y1 = min(arr.shape[0], arr.shape[0] - int(shift_y_px))
    src_x0 = max(0, -int(shift_x_px))
    src_x1 = min(arr.shape[1], arr.shape[1] - int(shift_x_px))
    dst_y0 = max(0, int(shift_y_px))
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, int(shift_x_px))
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out




def crop_stack_with_bounds(stack_zyx: np.ndarray, center_y_px: float, center_x_px: float, size_px: int) -> Tuple[np.ndarray, Dict[str, int]]:
    arr = np.asarray(stack_zyx)
    if arr.ndim < 2:
        raise ValueError('crop_stack_with_bounds expects at least 2 dimensions.')
    ref = arr if arr.ndim == 2 else arr[0]
    half = int(size_px // 2)
    cy = int(round(center_y_px))
    cx = int(round(center_x_px))
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(ref.shape[0], y0 + int(size_px))
    x1 = min(ref.shape[1], x0 + int(size_px))
    y0 = max(0, y1 - int(size_px))
    x0 = max(0, x1 - int(size_px))
    crop = arr[..., y0:y1, x0:x1]
    return crop, {'y0': int(y0), 'x0': int(x0), 'y1': int(y1), 'x1': int(x1)}


def build_nominal_placements(info: CziTileLatticeInfo) -> Dict[Tuple[int, int], Tuple[int, int]]:
    return {
        (int(r), int(c)): (int(info.y_starts[r] - info.y_starts[0]), int(info.x_starts[c] - info.x_starts[0]))
        for r in range(len(info.y_starts))
        for c in range(len(info.x_starts))
    }


def select_local_tile_subset(
    info: CziTileLatticeInfo,
    center_y_px: float,
    center_x_px: float,
    crop_size_px: int,
    extra_margin_px: int = 128,
) -> Dict[str, object]:
    nominal = build_nominal_placements(info)
    tile_h, tile_w = [int(v) for v in info.tile_shape_yx]
    half = float(crop_size_px) / 2.0
    box_y0 = float(center_y_px) - half - float(extra_margin_px)
    box_y1 = float(center_y_px) + half + float(extra_margin_px)
    box_x0 = float(center_x_px) - half - float(extra_margin_px)
    box_x1 = float(center_x_px) + half + float(extra_margin_px)

    row_indices = []
    for r in range(len(info.y_starts)):
        y = float(nominal[(int(r), 0)][0])
        if (y + tile_h) > box_y0 and y < box_y1:
            row_indices.append(int(r))
    col_indices = []
    for c in range(len(info.x_starts)):
        x = float(nominal[(0, int(c))][1])
        if (x + tile_w) > box_x0 and x < box_x1:
            col_indices.append(int(c))
    if not row_indices or not col_indices:
        raise RuntimeError('Failed to select a local tile subset for ROI alignment.')
    return {
        'row_indices': [int(v) for v in row_indices],
        'col_indices': [int(v) for v in col_indices],
        'box_y0_px': float(box_y0),
        'box_y1_px': float(box_y1),
        'box_x0_px': float(box_x0),
        'box_x1_px': float(box_x1),
    }


def _estimate_horizontal_delta_scored(
    left_img: np.ndarray,
    right_img: np.ndarray,
    nominal_dx_px: int,
    upsample_factor: int = 20,
    max_cross_axis_shift_px: float = 12.0,
    max_primary_shift_from_nominal_px: float = 48.0,
):
    ovx = int(left_img.shape[1] - nominal_dx_px)
    if ovx < 24:
        return None
    left_overlap = np.asarray(left_img[:, nominal_dx_px:], dtype=np.float32)
    right_overlap = np.asarray(right_img[:, :ovx], dtype=np.float32)
    left_feat = _prepare_alignment_overlap(left_overlap)
    right_feat = _prepare_alignment_overlap(right_overlap)
    signal_score = float(np.nanstd(left_feat) + np.nanstd(right_feat))
    if not np.isfinite(signal_score) or signal_score <= 1e-6:
        return None
    shift, error, _ = phase_cross_correlation(
        left_feat,
        right_feat,
        upsample_factor=int(upsample_factor),
    )
    cross_axis_shift_px = float(shift[0])
    primary_shift_from_nominal_px = float(shift[1])
    if abs(cross_axis_shift_px) > float(max_cross_axis_shift_px):
        return None
    if abs(primary_shift_from_nominal_px) > float(max_primary_shift_from_nominal_px):
        return None
    error = float(error)
    quality = float(
        signal_score
        / max(error, 1e-3)
        / (1.0 + (abs(cross_axis_shift_px) / 4.0) ** 2)
        / (1.0 + (abs(primary_shift_from_nominal_px) / 24.0) ** 2)
    )
    return {
        'delta_y_px': float(cross_axis_shift_px),
        'delta_x_px': float(nominal_dx_px + primary_shift_from_nominal_px),
        'registration_error': error,
        'signal_score': float(signal_score),
        'quality': quality,
    }


def _estimate_vertical_delta_scored(
    top_img: np.ndarray,
    bottom_img: np.ndarray,
    nominal_dy_px: int,
    upsample_factor: int = 20,
    max_cross_axis_shift_px: float = 12.0,
    max_primary_shift_from_nominal_px: float = 48.0,
):
    ovy = int(top_img.shape[0] - nominal_dy_px)
    if ovy < 24:
        return None
    top_overlap = np.asarray(top_img[nominal_dy_px:, :], dtype=np.float32)
    bottom_overlap = np.asarray(bottom_img[:ovy, :], dtype=np.float32)
    top_feat = _prepare_alignment_overlap(top_overlap)
    bottom_feat = _prepare_alignment_overlap(bottom_overlap)
    signal_score = float(np.nanstd(top_feat) + np.nanstd(bottom_feat))
    if not np.isfinite(signal_score) or signal_score <= 1e-6:
        return None
    shift, error, _ = phase_cross_correlation(
        top_feat,
        bottom_feat,
        upsample_factor=int(upsample_factor),
    )
    primary_shift_from_nominal_px = float(shift[0])
    cross_axis_shift_px = float(shift[1])
    if abs(cross_axis_shift_px) > float(max_cross_axis_shift_px):
        return None
    if abs(primary_shift_from_nominal_px) > float(max_primary_shift_from_nominal_px):
        return None
    error = float(error)
    quality = float(
        signal_score
        / max(error, 1e-3)
        / (1.0 + (abs(cross_axis_shift_px) / 4.0) ** 2)
        / (1.0 + (abs(primary_shift_from_nominal_px) / 24.0) ** 2)
    )
    return {
        'delta_y_px': float(nominal_dy_px + primary_shift_from_nominal_px),
        'delta_x_px': float(cross_axis_shift_px),
        'registration_error': error,
        'signal_score': float(signal_score),
        'quality': quality,
    }


def _aggregate_delta_candidates(
    candidates: Sequence[Dict[str, object]],
    fallback_dy_px: float,
    fallback_dx_px: float,
    keep_top_n: int = 3,
) -> Tuple[float, float, Dict[str, object]]:
    if not candidates:
        return float(fallback_dy_px), float(fallback_dx_px), {
            'n_candidates': 0,
            'n_used': 0,
            'best_evidence_id': '',
            'quality_median': float('nan'),
        }
    cand_df = pd.DataFrame(candidates)
    cand_df = cand_df.loc[
        np.isfinite(cand_df['delta_y_px'])
        & np.isfinite(cand_df['delta_x_px'])
        & np.isfinite(cand_df['quality'])
    ].copy()
    if cand_df.empty:
        return float(fallback_dy_px), float(fallback_dx_px), {
            'n_candidates': 0,
            'n_used': 0,
            'best_evidence_id': '',
            'quality_median': float('nan'),
        }
    cand_df = cand_df.sort_values('quality', ascending=False).reset_index(drop=True)
    n_used = int(min(max(1, int(keep_top_n)), len(cand_df)))
    keep = cand_df.head(n_used)
    return float(np.median(keep['delta_y_px'])), float(np.median(keep['delta_x_px'])), {
        'n_candidates': int(len(cand_df)),
        'n_used': int(n_used),
        'best_evidence_id': str(cand_df.iloc[0]['evidence_id']),
        'quality_median': float(np.median(keep['quality'])),
    }





def _best_consensus_candidate_id(candidates: Sequence[Dict[str, object]], consensus_dy_px: float, consensus_dx_px: float) -> str:
    if len(candidates) == 0:
        return ''
    best_key = None
    best_row = None
    for row in candidates:
        dy = float(row['dy_px'])
        dx = float(row['dx_px'])
        err = float(row.get('error', np.nan))
        dist = float(np.hypot(dy - float(consensus_dy_px), dx - float(consensus_dx_px)))
        key = (dist, err if np.isfinite(err) else np.inf, int(row.get('z_index', -1)))
        if best_key is None or key < best_key:
            best_key = key
            best_row = row
    if best_row is None:
        return ''
    return f"brightfield_same_z:z{int(best_row['z_index'])}"


def _consensus_shift_from_same_z_candidates(candidates: Sequence[Dict[str, object]], cluster_tol_px: float = 8.0) -> Dict[str, object]:
    if len(candidates) == 0:
        raise RuntimeError('No same-z seam candidates available for consensus.')
    dy_vals = np.asarray([float(c['dy_px']) for c in candidates], dtype=np.float64)
    dx_vals = np.asarray([float(c['dx_px']) for c in candidates], dtype=np.float64)
    err_vals = np.asarray([float(c.get('error', np.nan)) for c in candidates], dtype=np.float64)

    best_idx = None
    best_inliers = None
    best_key = None
    for idx in range(len(candidates)):
        dist = np.hypot(dy_vals - dy_vals[idx], dx_vals - dx_vals[idx])
        inliers = np.flatnonzero(dist <= float(cluster_tol_px))
        support_n = int(len(inliers))
        med_dist = float(np.median(dist[inliers])) if support_n > 0 else np.inf
        finite_err = err_vals[inliers][np.isfinite(err_vals[inliers])] if support_n > 0 else np.asarray([], dtype=np.float64)
        med_err = float(np.median(finite_err)) if finite_err.size else np.inf
        key = (-support_n, med_dist, med_err, int(idx))
        if best_key is None or key < best_key:
            best_key = key
            best_idx = int(idx)
            best_inliers = inliers

    assert best_idx is not None and best_inliers is not None
    inlier_dy = dy_vals[best_inliers]
    inlier_dx = dx_vals[best_inliers]
    consensus_dy = float(np.median(inlier_dy))
    consensus_dx = float(np.median(inlier_dx))
    residuals = np.hypot(dy_vals - consensus_dy, dx_vals - consensus_dx)
    inlier_set = {int(i) for i in best_inliers.tolist()}
    return {
        'consensus_dy_px': float(consensus_dy),
        'consensus_dx_px': float(consensus_dx),
        'cluster_tol_px': float(cluster_tol_px),
        'support_n': int(len(best_inliers)),
        'n_candidates': int(len(candidates)),
        'inlier_z_indices': [int(candidates[i]['z_index']) for i in range(len(candidates)) if i in inlier_set],
        'outlier_z_indices': [int(candidates[i]['z_index']) for i in range(len(candidates)) if i not in inlier_set],
        'candidate_rows': [dict(c) for c in candidates],
        'max_candidate_residual_px': float(np.max(residuals)) if residuals.size else 0.0,
        'median_candidate_residual_px': float(np.median(residuals)) if residuals.size else 0.0,
        'best_evidence_id': _best_consensus_candidate_id(candidates, consensus_dy_px=consensus_dy, consensus_dx_px=consensus_dx),
    }


def _estimate_horizontal_delta_samez_consensus(
    left_stack: np.ndarray,
    right_stack: np.ndarray,
    shared_z_values: Sequence[int],
    nominal_dx_px: int,
    upsample_factor: int = 20,
) -> Tuple[float, float, Dict[str, object]]:
    left_arr = np.asarray(left_stack, dtype=np.float32)
    right_arr = np.asarray(right_stack, dtype=np.float32)
    if left_arr.ndim == 2:
        left_arr = left_arr[None, ...]
    if right_arr.ndim == 2:
        right_arr = right_arr[None, ...]
    n_shared = int(min(left_arr.shape[0], right_arr.shape[0], len(shared_z_values)))
    ovx = int(left_arr.shape[-1] - nominal_dx_px)
    if ovx < 24 or n_shared <= 0:
        diag = {
            'consensus_dy_px': 0.0,
            'consensus_dx_px': float(nominal_dx_px),
            'cluster_tol_px': 8.0,
            'support_n': 0,
            'n_candidates': 0,
            'inlier_z_indices': [],
            'outlier_z_indices': [],
            'candidate_rows': [],
            'max_candidate_residual_px': 0.0,
            'median_candidate_residual_px': 0.0,
            'best_evidence_id': '',
        }
        return 0.0, float(nominal_dx_px), diag
    candidates = []
    for idx in range(n_shared):
        z_idx = int(shared_z_values[idx])
        try:
            shift, error, _ = phase_cross_correlation(
                _zscore_norm(left_arr[idx][:, nominal_dx_px:]),
                _zscore_norm(right_arr[idx][:, :ovx]),
                upsample_factor=int(upsample_factor),
            )
        except Exception:
            continue
        candidates.append({'z_index': int(z_idx), 'dy_px': float(shift[0]), 'dx_px': float(nominal_dx_px + shift[1]), 'error': float(error)})
    if len(candidates) == 0:
        diag = {
            'consensus_dy_px': 0.0,
            'consensus_dx_px': float(nominal_dx_px),
            'cluster_tol_px': 8.0,
            'support_n': 0,
            'n_candidates': 0,
            'inlier_z_indices': [],
            'outlier_z_indices': [],
            'candidate_rows': [],
            'max_candidate_residual_px': 0.0,
            'median_candidate_residual_px': 0.0,
            'best_evidence_id': '',
        }
        return 0.0, float(nominal_dx_px), diag
    diag = _consensus_shift_from_same_z_candidates(candidates, cluster_tol_px=8.0)
    return float(diag['consensus_dy_px']), float(diag['consensus_dx_px']), diag


def _estimate_vertical_delta_samez_consensus(
    top_stack: np.ndarray,
    bottom_stack: np.ndarray,
    shared_z_values: Sequence[int],
    nominal_dy_px: int,
    upsample_factor: int = 20,
) -> Tuple[float, float, Dict[str, object]]:
    top_arr = np.asarray(top_stack, dtype=np.float32)
    bottom_arr = np.asarray(bottom_stack, dtype=np.float32)
    if top_arr.ndim == 2:
        top_arr = top_arr[None, ...]
    if bottom_arr.ndim == 2:
        bottom_arr = bottom_arr[None, ...]
    n_shared = int(min(top_arr.shape[0], bottom_arr.shape[0], len(shared_z_values)))
    ovy = int(top_arr.shape[-2] - nominal_dy_px)
    if ovy < 24 or n_shared <= 0:
        diag = {
            'consensus_dy_px': float(nominal_dy_px),
            'consensus_dx_px': 0.0,
            'cluster_tol_px': 8.0,
            'support_n': 0,
            'n_candidates': 0,
            'inlier_z_indices': [],
            'outlier_z_indices': [],
            'candidate_rows': [],
            'max_candidate_residual_px': 0.0,
            'median_candidate_residual_px': 0.0,
            'best_evidence_id': '',
        }
        return float(nominal_dy_px), 0.0, diag
    candidates = []
    for idx in range(n_shared):
        z_idx = int(shared_z_values[idx])
        try:
            shift, error, _ = phase_cross_correlation(
                _zscore_norm(top_arr[idx][nominal_dy_px:, :]),
                _zscore_norm(bottom_arr[idx][:ovy, :]),
                upsample_factor=int(upsample_factor),
            )
        except Exception:
            continue
        candidates.append({'z_index': int(z_idx), 'dy_px': float(nominal_dy_px + shift[0]), 'dx_px': float(shift[1]), 'error': float(error)})
    if len(candidates) == 0:
        diag = {
            'consensus_dy_px': float(nominal_dy_px),
            'consensus_dx_px': 0.0,
            'cluster_tol_px': 8.0,
            'support_n': 0,
            'n_candidates': 0,
            'inlier_z_indices': [],
            'outlier_z_indices': [],
            'candidate_rows': [],
            'max_candidate_residual_px': 0.0,
            'median_candidate_residual_px': 0.0,
            'best_evidence_id': '',
        }
        return float(nominal_dy_px), 0.0, diag
    diag = _consensus_shift_from_same_z_candidates(candidates, cluster_tol_px=8.0)
    return float(diag['consensus_dy_px']), float(diag['consensus_dx_px']), diag


def _estimate_local_registered_positions_foxf1_like(
    evidence_tile_maps: Dict[str, Dict[Tuple[int, int, int], np.ndarray]],
    evidence_z_values: Dict[str, Sequence[int]],
    full_y_starts: Sequence[int],
    full_x_starts: Sequence[int],
    row_indices: Sequence[int],
    col_indices: Sequence[int],
    upsample_factor: int = 20,
    max_shift_from_nominal_px: int = 64,
) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], pd.DataFrame, pd.DataFrame]:
    if 'brightfield' in evidence_tile_maps:
        channel_name = 'brightfield'
    else:
        channel_name = str(next(iter(evidence_tile_maps.keys())))
    tile_map = evidence_tile_maps[channel_name]
    z_values = [int(v) for v in evidence_z_values.get(channel_name, [])]

    row_indices = [int(v) for v in row_indices]
    col_indices = [int(v) for v in col_indices]
    local_n_rows = len(row_indices)
    local_n_cols = len(col_indices)
    node_index = {(r, c): (r * local_n_cols + c) for r in range(local_n_rows) for c in range(local_n_cols)}
    subset_origin_y = int(full_y_starts[row_indices[0]])
    subset_origin_x = int(full_x_starts[col_indices[0]])
    nominal_y = np.array([
        float(full_y_starts[row_indices[r]] - subset_origin_y)
        for r in range(local_n_rows)
        for _ in range(local_n_cols)
    ], dtype=np.float64)
    nominal_x = np.array([
        float(full_x_starts[col_indices[c]] - subset_origin_x)
        for _ in range(local_n_rows)
        for c in range(local_n_cols)
    ], dtype=np.float64)

    y_edges = []
    x_edges = []
    edge_rows = []
    for r_local, r_global in enumerate(row_indices):
        for c_local in range(local_n_cols - 1):
            c0 = col_indices[c_local]
            c1 = col_indices[c_local + 1]
            nominal_dx = int(full_x_starts[c1] - full_x_starts[c0])
            shared_z = [
                int(z) for z in z_values
                if (int(z), int(full_y_starts[r_global]), int(full_x_starts[c0])) in tile_map
                and (int(z), int(full_y_starts[r_global]), int(full_x_starts[c1])) in tile_map
            ]
            left_stack = np.stack([tile_map[(int(z), int(full_y_starts[r_global]), int(full_x_starts[c0]))] for z in shared_z], axis=0) if shared_z else np.zeros((0, 1, 1), dtype=np.float32)
            right_stack = np.stack([tile_map[(int(z), int(full_y_starts[r_global]), int(full_x_starts[c1]))] for z in shared_z], axis=0) if shared_z else np.zeros((0, 1, 1), dtype=np.float32)
            dy, dx, diag = _estimate_horizontal_delta_samez_consensus(left_stack, right_stack, shared_z_values=shared_z, nominal_dx_px=nominal_dx, upsample_factor=int(upsample_factor))
            edge_weight = 1.0 if int(diag['n_candidates']) > 0 else 0.25
            i = node_index[(r_local, c_local)]
            j = node_index[(r_local, c_local + 1)]
            y_edges.append((i, j, dy, edge_weight))
            x_edges.append((i, j, dx, edge_weight))
            edge_rows.append({
                'edge_type': 'horizontal',
                'local_r0': int(r_local), 'local_c0': int(c_local), 'local_r1': int(r_local), 'local_c1': int(c_local + 1),
                'global_r0': int(r_global), 'global_c0': int(c0), 'global_r1': int(r_global), 'global_c1': int(c1),
                'delta_y_px': float(dy), 'delta_x_px': float(dx),
                'nominal_dy_px': 0, 'nominal_dx_px': int(nominal_dx),
                'n_candidates': int(diag['n_candidates']), 'n_used': int(diag['support_n']),
                'best_evidence_id': str(diag['best_evidence_id']), 'quality_median': float(-diag['median_candidate_residual_px']),
                'edge_weight': float(edge_weight), 'evidence_strategy': 'brightfield_same_z_consensus',
                'inlier_z_indices': list(diag['inlier_z_indices']), 'outlier_z_indices': list(diag['outlier_z_indices']),
                'candidate_rows': list(diag['candidate_rows']),
                'max_candidate_residual_px': float(diag['max_candidate_residual_px']),
                'median_candidate_residual_px': float(diag['median_candidate_residual_px']),
            })

    for r_local in range(local_n_rows - 1):
        r0 = row_indices[r_local]
        r1 = row_indices[r_local + 1]
        for c_local, c_global in enumerate(col_indices):
            nominal_dy = int(full_y_starts[r1] - full_y_starts[r0])
            shared_z = [
                int(z) for z in z_values
                if (int(z), int(full_y_starts[r0]), int(full_x_starts[c_global])) in tile_map
                and (int(z), int(full_y_starts[r1]), int(full_x_starts[c_global])) in tile_map
            ]
            top_stack = np.stack([tile_map[(int(z), int(full_y_starts[r0]), int(full_x_starts[c_global]))] for z in shared_z], axis=0) if shared_z else np.zeros((0, 1, 1), dtype=np.float32)
            bottom_stack = np.stack([tile_map[(int(z), int(full_y_starts[r1]), int(full_x_starts[c_global]))] for z in shared_z], axis=0) if shared_z else np.zeros((0, 1, 1), dtype=np.float32)
            dy, dx, diag = _estimate_vertical_delta_samez_consensus(top_stack, bottom_stack, shared_z_values=shared_z, nominal_dy_px=nominal_dy, upsample_factor=int(upsample_factor))
            edge_weight = 1.0 if int(diag['n_candidates']) > 0 else 0.25
            i = node_index[(r_local, c_local)]
            j = node_index[(r_local + 1, c_local)]
            y_edges.append((i, j, dy, edge_weight))
            x_edges.append((i, j, dx, edge_weight))
            edge_rows.append({
                'edge_type': 'vertical',
                'local_r0': int(r_local), 'local_c0': int(c_local), 'local_r1': int(r_local + 1), 'local_c1': int(c_local),
                'global_r0': int(r0), 'global_c0': int(c_global), 'global_r1': int(r1), 'global_c1': int(c_global),
                'delta_y_px': float(dy), 'delta_x_px': float(dx),
                'nominal_dy_px': int(nominal_dy), 'nominal_dx_px': 0,
                'n_candidates': int(diag['n_candidates']), 'n_used': int(diag['support_n']),
                'best_evidence_id': str(diag['best_evidence_id']), 'quality_median': float(-diag['median_candidate_residual_px']),
                'edge_weight': float(edge_weight), 'evidence_strategy': 'brightfield_same_z_consensus',
                'inlier_z_indices': list(diag['inlier_z_indices']), 'outlier_z_indices': list(diag['outlier_z_indices']),
                'candidate_rows': list(diag['candidate_rows']),
                'max_candidate_residual_px': float(diag['max_candidate_residual_px']),
                'median_candidate_residual_px': float(diag['median_candidate_residual_px']),
            })

    solved_y = _solve_grid_positions(len(node_index), y_edges, nominal_vals=nominal_y, max_shift_from_nominal_px=int(max_shift_from_nominal_px))
    solved_x = _solve_grid_positions(len(node_index), x_edges, nominal_vals=nominal_x, max_shift_from_nominal_px=int(max_shift_from_nominal_px))
    solved_y = solved_y + int(full_y_starts[row_indices[0]] - full_y_starts[0])
    solved_x = solved_x + int(full_x_starts[col_indices[0]] - full_x_starts[0])

    placement_rows = []
    placements = {}
    for (r_local, c_local), idx in node_index.items():
        r_global = row_indices[r_local]
        c_global = col_indices[c_local]
        yy = int(solved_y[idx])
        xx = int(solved_x[idx])
        nom_y = int(full_y_starts[r_global] - full_y_starts[0])
        nom_x = int(full_x_starts[c_global] - full_x_starts[0])
        placements[(int(r_local), int(c_local))] = (yy, xx)
        placement_rows.append({
            'local_tile_row': int(r_local), 'local_tile_col': int(c_local),
            'global_tile_row': int(r_global), 'global_tile_col': int(c_global),
            'raw_y_start_px': int(full_y_starts[r_global]), 'raw_x_start_px': int(full_x_starts[c_global]),
            'nominal_y_px': int(nom_y), 'nominal_x_px': int(nom_x),
            'registered_y_px': int(yy), 'registered_x_px': int(xx),
            'delta_from_nominal_y_px': int(yy - nom_y), 'delta_from_nominal_x_px': int(xx - nom_x),
        })
    placement_df = pd.DataFrame(placement_rows).sort_values(['local_tile_row', 'local_tile_col']).reset_index(drop=True)
    edge_df = pd.DataFrame(edge_rows)
    return placements, placement_df, edge_df



def _scene_alignment_cache_key(
    dapi_tile_map: Dict[Tuple[int, int, int], np.ndarray],
    bf_tile_map: Dict[Tuple[int, int, int], np.ndarray],
    dapi_z_values: Sequence[int],
    bf_z_values: Sequence[int],
    full_y_starts: Sequence[int],
    full_x_starts: Sequence[int],
) -> Tuple[object, ...]:
    return (
        id(dapi_tile_map),
        id(bf_tile_map),
        tuple(int(v) for v in dapi_z_values),
        tuple(int(v) for v in bf_z_values),
        tuple(int(v) for v in full_y_starts),
        tuple(int(v) for v in full_x_starts),
    )


def _compute_shared_global_dapi_scale(
    dapi_tile_map: Dict[Tuple[int, int, int], np.ndarray],
    downsample: int = 8,
    p_low: float = 1.0,
    p_high: float = 99.8,
    threshold_percentile: float = 97.0,
    threshold_floor: float = 0.20,
) -> Dict[str, float]:
    samples = []
    for arr in dapi_tile_map.values():
        sample = np.asarray(arr, dtype=np.float32)[:: int(max(1, downsample)), :: int(max(1, downsample))]
        finite = sample[np.isfinite(sample)]
        if finite.size:
            samples.append(finite)
    if not samples:
        return {
            'low': 0.0,
            'high': 1.0,
            'den': 1.0,
            'threshold': 1.0,
        }
    global_sample = np.concatenate(samples)
    low = float(np.percentile(global_sample, float(p_low)))
    high = float(np.percentile(global_sample, float(p_high)))
    den = float(max(high - low, 1e-6))
    norm_sample = np.clip((global_sample - low) / den, 0.0, 1.0)
    threshold = float(max(float(threshold_floor), np.percentile(norm_sample, float(threshold_percentile))))
    return {
        'low': float(low),
        'high': float(high),
        'den': float(den),
        'threshold': float(threshold),
    }


def _shared_global_dapi_rescale(img: np.ndarray, scale: Dict[str, float]) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.zeros_like(arr, dtype=np.float32)
    low = float(scale.get('low', 0.0))
    den = float(max(scale.get('den', 1.0), 1e-6))
    return np.clip((arr - low) / den, 0.0, 1.0).astype(np.float32)


def _prepare_shared_global_dapi_mask(
    img: np.ndarray,
    scale: Dict[str, float],
    smooth_sigma_px: float = 1.0,
    min_component_area_px: int = 12,
    dilate_px: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(img, dtype=np.float32)
    smooth = ndi.gaussian_filter(_shared_global_dapi_rescale(arr, scale), sigma=float(smooth_sigma_px))
    mask = smooth >= float(scale.get('threshold', 1.0))
    mask = morphology.remove_small_objects(mask, min_size=int(min_component_area_px))
    if int(dilate_px) > 0 and np.any(mask):
        mask = ndi.binary_dilation(mask, iterations=int(dilate_px))
    return smooth.astype(np.float32), np.asarray(mask, dtype=bool)


def _component_rows_with_mask(mask: np.ndarray, min_component_area_px: int = 12) -> List[Dict[str, object]]:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0 or not np.any(arr):
        return []
    labels, n_labels = ndi.label(arr)
    rows: List[Dict[str, object]] = []
    for label_idx in range(1, int(n_labels) + 1):
        component_mask = labels == int(label_idx)
        area_px = int(np.count_nonzero(component_mask))
        if area_px < int(min_component_area_px):
            continue
        ys, xs = np.where(component_mask)
        if ys.size == 0:
            continue
        rows.append({
            'label': int(label_idx),
            'area_px': int(area_px),
            'y0_px': int(np.min(ys)),
            'x0_px': int(np.min(xs)),
            'y1_px': int(np.max(ys)) + 1,
            'x1_px': int(np.max(xs)) + 1,
            'center_y_px': float(np.mean(ys)),
            'center_x_px': float(np.mean(xs)),
            'component_mask': component_mask,
        })
    return rows


def _dilated_intersection_area(mask_a: np.ndarray, mask_b: np.ndarray, dilation_px: int = 2) -> int:
    dil_a = ndi.binary_dilation(np.asarray(mask_a, dtype=bool), iterations=int(max(0, dilation_px)))
    dil_b = ndi.binary_dilation(np.asarray(mask_b, dtype=bool), iterations=int(max(0, dilation_px)))
    return int(np.count_nonzero(dil_a & dil_b))


def _detect_horizontal_dapi_overlap_shared_scale(
    left_img: np.ndarray,
    right_img: np.ndarray,
    nominal_dx_px: int,
    scale: Dict[str, float],
    min_component_area_px: int = 12,
    pair_overlap_dilation_px: int = 2,
    min_axis_overlap_px: int = 12,
    min_intersection_area_px: int = 80,
    min_intersection_fraction: float = 0.25,
    edge_margin_px: int = 2,
    window_pad_px: int = 8,
) -> Dict[str, object]:
    ovx = int(left_img.shape[1] - nominal_dx_px)
    if ovx < 24:
        return {'success': False, 'reject_reason': 'tiny_overlap'}
    left_overlap = np.asarray(left_img[:, nominal_dx_px:], dtype=np.float32)
    right_overlap = np.asarray(right_img[:, :ovx], dtype=np.float32)
    left_feat, left_mask = _prepare_shared_global_dapi_mask(left_overlap, scale=scale, min_component_area_px=int(min_component_area_px))
    right_feat, right_mask = _prepare_shared_global_dapi_mask(right_overlap, scale=scale, min_component_area_px=int(min_component_area_px))
    left_components = _component_rows_with_mask(left_mask, min_component_area_px=int(min_component_area_px))
    right_components = _component_rows_with_mask(right_mask, min_component_area_px=int(min_component_area_px))
    best_pair = None
    best_score = float('-inf')
    for comp_a in left_components:
        for comp_b in right_components:
            axis_overlap_px = int(max(0, min(int(comp_a['y1_px']), int(comp_b['y1_px'])) - max(int(comp_a['y0_px']), int(comp_b['y0_px']))))
            inter_area_px = _dilated_intersection_area(comp_a['component_mask'], comp_b['component_mask'], dilation_px=int(pair_overlap_dilation_px))
            inter_frac = float(inter_area_px / max(1, min(int(comp_a['area_px']), int(comp_b['area_px']))))
            shared_overlap = bool(
                inter_area_px >= int(min_intersection_area_px)
                and inter_frac >= float(min_intersection_fraction)
                and axis_overlap_px >= int(min_axis_overlap_px)
            )
            boundary_spanning = bool(
                int(comp_a['x1_px']) >= int(left_overlap.shape[1] - edge_margin_px)
                and int(comp_b['x0_px']) <= int(edge_margin_px)
                and axis_overlap_px >= int(min_axis_overlap_px)
                and inter_area_px >= int(min_intersection_area_px)
            )
            if not (shared_overlap or boundary_spanning):
                continue
            score = float(inter_area_px + 0.2 * axis_overlap_px + (20.0 if boundary_spanning else 0.0))
            if score > best_score:
                best_score = score
                y0 = min(int(comp_a['y0_px']), int(comp_b['y0_px'])) - int(window_pad_px)
                y1 = max(int(comp_a['y1_px']), int(comp_b['y1_px'])) + int(window_pad_px)
                y0, y1 = _expand_interval(int(y0), int(y1), int(min_axis_overlap_px), int(left_overlap.shape[0]))
                x0 = min(int(comp_a['x0_px']), int(comp_b['x0_px'])) - int(window_pad_px)
                x1 = max(int(comp_a['x1_px']), int(comp_b['x1_px'])) + int(window_pad_px)
                x0, x1 = _expand_interval(int(x0), int(x1), 24, int(left_overlap.shape[1]))
                best_pair = {
                    'window_y0_px': int(y0),
                    'window_y1_px': int(y1),
                    'window_x0_px': int(x0),
                    'window_x1_px': int(x1),
                    'axis_overlap_px': int(axis_overlap_px),
                    'intersection_area_px': int(inter_area_px),
                    'intersection_fraction': float(inter_frac),
                    'match_type': 'boundary_spanning' if boundary_spanning else 'shared_overlap',
                    'score': float(score),
                }
    return {
        'success': bool(best_pair is not None),
        'reject_reason': '' if best_pair is not None else 'no_valid_component_pair',
        'left_overlap': left_feat,
        'right_overlap': right_feat,
        'left_mask': left_mask,
        'right_mask': right_mask,
        'left_components_n': int(len(left_components)),
        'right_components_n': int(len(right_components)),
        **({} if best_pair is None else best_pair),
    }


def _detect_vertical_dapi_overlap_shared_scale(
    top_img: np.ndarray,
    bottom_img: np.ndarray,
    nominal_dy_px: int,
    scale: Dict[str, float],
    min_component_area_px: int = 12,
    pair_overlap_dilation_px: int = 2,
    min_axis_overlap_px: int = 12,
    min_intersection_area_px: int = 80,
    min_intersection_fraction: float = 0.25,
    edge_margin_px: int = 2,
    window_pad_px: int = 8,
) -> Dict[str, object]:
    ovy = int(top_img.shape[0] - nominal_dy_px)
    if ovy < 24:
        return {'success': False, 'reject_reason': 'tiny_overlap'}
    top_overlap = np.asarray(top_img[nominal_dy_px:, :], dtype=np.float32)
    bottom_overlap = np.asarray(bottom_img[:ovy, :], dtype=np.float32)
    top_feat, top_mask = _prepare_shared_global_dapi_mask(top_overlap, scale=scale, min_component_area_px=int(min_component_area_px))
    bottom_feat, bottom_mask = _prepare_shared_global_dapi_mask(bottom_overlap, scale=scale, min_component_area_px=int(min_component_area_px))
    top_components = _component_rows_with_mask(top_mask, min_component_area_px=int(min_component_area_px))
    bottom_components = _component_rows_with_mask(bottom_mask, min_component_area_px=int(min_component_area_px))
    best_pair = None
    best_score = float('-inf')
    for comp_a in top_components:
        for comp_b in bottom_components:
            axis_overlap_px = int(max(0, min(int(comp_a['x1_px']), int(comp_b['x1_px'])) - max(int(comp_a['x0_px']), int(comp_b['x0_px']))))
            inter_area_px = _dilated_intersection_area(comp_a['component_mask'], comp_b['component_mask'], dilation_px=int(pair_overlap_dilation_px))
            inter_frac = float(inter_area_px / max(1, min(int(comp_a['area_px']), int(comp_b['area_px']))))
            shared_overlap = bool(
                inter_area_px >= int(min_intersection_area_px)
                and inter_frac >= float(min_intersection_fraction)
                and axis_overlap_px >= int(min_axis_overlap_px)
            )
            boundary_spanning = bool(
                int(comp_a['y1_px']) >= int(top_overlap.shape[0] - edge_margin_px)
                and int(comp_b['y0_px']) <= int(edge_margin_px)
                and axis_overlap_px >= int(min_axis_overlap_px)
                and inter_area_px >= int(min_intersection_area_px)
            )
            if not (shared_overlap or boundary_spanning):
                continue
            score = float(inter_area_px + 0.2 * axis_overlap_px + (20.0 if boundary_spanning else 0.0))
            if score > best_score:
                best_score = score
                y0 = min(int(comp_a['y0_px']), int(comp_b['y0_px'])) - int(window_pad_px)
                y1 = max(int(comp_a['y1_px']), int(comp_b['y1_px'])) + int(window_pad_px)
                y0, y1 = _expand_interval(int(y0), int(y1), 24, int(top_overlap.shape[0]))
                x0 = min(int(comp_a['x0_px']), int(comp_b['x0_px'])) - int(window_pad_px)
                x1 = max(int(comp_a['x1_px']), int(comp_b['x1_px'])) + int(window_pad_px)
                x0, x1 = _expand_interval(int(x0), int(x1), int(min_axis_overlap_px), int(top_overlap.shape[1]))
                best_pair = {
                    'window_y0_px': int(y0),
                    'window_y1_px': int(y1),
                    'window_x0_px': int(x0),
                    'window_x1_px': int(x1),
                    'axis_overlap_px': int(axis_overlap_px),
                    'intersection_area_px': int(inter_area_px),
                    'intersection_fraction': float(inter_frac),
                    'match_type': 'boundary_spanning' if boundary_spanning else 'shared_overlap',
                    'score': float(score),
                }
    return {
        'success': bool(best_pair is not None),
        'reject_reason': '' if best_pair is not None else 'no_valid_component_pair',
        'top_overlap': top_feat,
        'bottom_overlap': bottom_feat,
        'top_mask': top_mask,
        'bottom_mask': bottom_mask,
        'top_components_n': int(len(top_components)),
        'bottom_components_n': int(len(bottom_components)),
        **({} if best_pair is None else best_pair),
    }


def _measure_horizontal_shift_samez_full_bf(
    left_stack: np.ndarray,
    right_stack: np.ndarray,
    shared_z_values: Sequence[int],
    nominal_dx_px: int,
    upsample_factor: int = 20,
    max_cross_axis_shift_px: float = 64.0,
    max_along_axis_offset_px: float = 96.0,
) -> Optional[Dict[str, object]]:
    left_arr = np.asarray(left_stack, dtype=np.float32)
    right_arr = np.asarray(right_stack, dtype=np.float32)
    if left_arr.ndim == 2:
        left_arr = left_arr[None, ...]
    if right_arr.ndim == 2:
        right_arr = right_arr[None, ...]
    n_shared = int(min(left_arr.shape[0], right_arr.shape[0], len(shared_z_values)))
    ovx = int(left_arr.shape[-1] - nominal_dx_px)
    if ovx < 24 or n_shared <= 0:
        return None
    candidates = []
    for idx in range(n_shared):
        z_idx = int(shared_z_values[idx])
        left_overlap = _prepare_alignment_overlap(np.asarray(left_arr[idx][:, nominal_dx_px:], dtype=np.float32))
        right_overlap = _prepare_alignment_overlap(np.asarray(right_arr[idx][:, :ovx], dtype=np.float32))
        if min(left_overlap.shape) < 8 or min(right_overlap.shape) < 8:
            continue
        try:
            shift, error, _ = phase_cross_correlation(left_overlap, right_overlap, upsample_factor=int(upsample_factor))
        except Exception:
            continue
        dy = float(shift[0])
        dx = float(nominal_dx_px + shift[1])
        if abs(dy) > float(max_cross_axis_shift_px):
            continue
        if abs(dx - float(nominal_dx_px)) > float(max_along_axis_offset_px):
            continue
        candidates.append({
            'z_index': int(z_idx),
            'dy_px': float(dy),
            'dx_px': float(dx),
            'error': float(error),
        })
    if not candidates:
        return None
    return _consensus_shift_from_same_z_candidates(candidates, cluster_tol_px=8.0)


def _measure_vertical_shift_samez_full_bf(
    top_stack: np.ndarray,
    bottom_stack: np.ndarray,
    shared_z_values: Sequence[int],
    nominal_dy_px: int,
    upsample_factor: int = 20,
    max_cross_axis_shift_px: float = 64.0,
    max_along_axis_offset_px: float = 96.0,
) -> Optional[Dict[str, object]]:
    top_arr = np.asarray(top_stack, dtype=np.float32)
    bottom_arr = np.asarray(bottom_stack, dtype=np.float32)
    if top_arr.ndim == 2:
        top_arr = top_arr[None, ...]
    if bottom_arr.ndim == 2:
        bottom_arr = bottom_arr[None, ...]
    n_shared = int(min(top_arr.shape[0], bottom_arr.shape[0], len(shared_z_values)))
    ovy = int(top_arr.shape[-2] - nominal_dy_px)
    if ovy < 24 or n_shared <= 0:
        return None
    candidates = []
    for idx in range(n_shared):
        z_idx = int(shared_z_values[idx])
        top_overlap = _prepare_alignment_overlap(np.asarray(top_arr[idx][nominal_dy_px:, :], dtype=np.float32))
        bottom_overlap = _prepare_alignment_overlap(np.asarray(bottom_arr[idx][:ovy, :], dtype=np.float32))
        if min(top_overlap.shape) < 8 or min(bottom_overlap.shape) < 8:
            continue
        try:
            shift, error, _ = phase_cross_correlation(top_overlap, bottom_overlap, upsample_factor=int(upsample_factor))
        except Exception:
            continue
        dy = float(nominal_dy_px + shift[0])
        dx = float(shift[1])
        if abs(dx) > float(max_cross_axis_shift_px):
            continue
        if abs(dy - float(nominal_dy_px)) > float(max_along_axis_offset_px):
            continue
        candidates.append({
            'z_index': int(z_idx),
            'dy_px': float(dy),
            'dx_px': float(dx),
            'error': float(error),
        })
    if not candidates:
        return None
    return _consensus_shift_from_same_z_candidates(candidates, cluster_tol_px=8.0)


def _calibrate_scene_dapi_bf_global_fill(
    evidence_tile_maps: Dict[str, Dict[Tuple[int, int, int], np.ndarray]],
    evidence_z_values: Dict[str, Sequence[int]],
    full_y_starts: Sequence[int],
    full_x_starts: Sequence[int],
    upsample_factor: int = 20,
) -> Dict[str, object]:
    dapi_tile_map = evidence_tile_maps.get('dapi')
    bf_tile_map = evidence_tile_maps.get('brightfield')
    dapi_z_values = [int(v) for v in evidence_z_values.get('dapi', [])]
    bf_z_values = [int(v) for v in evidence_z_values.get('brightfield', [])]
    shared_bf_z_values = [int(v) for v in bf_z_values if int(v) in set(dapi_z_values)]
    nominal_horizontal = {
        'dy_px': 0.0,
        'dx_px': float(np.median(np.diff(np.asarray(full_x_starts, dtype=np.float64)))) if len(full_x_starts) > 1 else 0.0,
    }
    nominal_vertical = {
        'dy_px': float(np.median(np.diff(np.asarray(full_y_starts, dtype=np.float64)))) if len(full_y_starts) > 1 else 0.0,
        'dx_px': 0.0,
    }
    if dapi_tile_map is None or bf_tile_map is None or not dapi_z_values or not shared_bf_z_values:
        return {
            'edge_map': {},
            'global_horizontal': nominal_horizontal,
            'global_vertical': nominal_vertical,
            'shared_bf_z_values': shared_bf_z_values,
            'n_measured_edges': 0,
        }
    cache_key = _scene_alignment_cache_key(dapi_tile_map, bf_tile_map, dapi_z_values, bf_z_values, full_y_starts, full_x_starts)
    if cache_key in _GLOBAL_DAPI_BF_SCENE_CACHE:
        return _GLOBAL_DAPI_BF_SCENE_CACHE[cache_key]

    scale = _compute_shared_global_dapi_scale(dapi_tile_map)
    n_rows = len(full_y_starts)
    n_cols = len(full_x_starts)
    edge_rows = []

    for r in range(n_rows):
        for c in range(n_cols - 1):
            nominal_dx_px = int(full_x_starts[c + 1] - full_x_starts[c])
            best = None
            support_count = 0
            for z_idx in dapi_z_values:
                key_left = (int(z_idx), int(full_y_starts[r]), int(full_x_starts[c]))
                key_right = (int(z_idx), int(full_y_starts[r]), int(full_x_starts[c + 1]))
                if key_left not in dapi_tile_map or key_right not in dapi_tile_map:
                    continue
                proposal = _detect_horizontal_dapi_overlap_shared_scale(
                    np.asarray(dapi_tile_map[key_left], dtype=np.float32),
                    np.asarray(dapi_tile_map[key_right], dtype=np.float32),
                    nominal_dx_px=int(nominal_dx_px),
                    scale=scale,
                )
                if not bool(proposal.get('success', False)):
                    continue
                support_count += 1
                score = float(proposal.get('score', 0.0))
                if best is None or score > float(best['score']):
                    best = {
                        'z_index': int(z_idx),
                        'window_y0_px': int(proposal['window_y0_px']),
                        'window_y1_px': int(proposal['window_y1_px']),
                        'window_x0_px': int(proposal['window_x0_px']),
                        'window_x1_px': int(proposal['window_x1_px']),
                        'axis_overlap_px': int(proposal['axis_overlap_px']),
                        'intersection_area_px': int(proposal['intersection_area_px']),
                        'intersection_fraction': float(proposal['intersection_fraction']),
                        'match_type': str(proposal['match_type']),
                        'score': float(score),
                    }
            diag = None
            if support_count > 0:
                left_stack = []
                right_stack = []
                usable_z = []
                for z_idx in shared_bf_z_values:
                    key_left = (int(z_idx), int(full_y_starts[r]), int(full_x_starts[c]))
                    key_right = (int(z_idx), int(full_y_starts[r]), int(full_x_starts[c + 1]))
                    if key_left not in bf_tile_map or key_right not in bf_tile_map:
                        continue
                    left_stack.append(np.asarray(bf_tile_map[key_left], dtype=np.float32))
                    right_stack.append(np.asarray(bf_tile_map[key_right], dtype=np.float32))
                    usable_z.append(int(z_idx))
                if usable_z:
                    diag = _measure_horizontal_shift_samez_full_bf(
                        left_stack=np.stack(left_stack, axis=0),
                        right_stack=np.stack(right_stack, axis=0),
                        shared_z_values=usable_z,
                        nominal_dx_px=int(nominal_dx_px),
                        upsample_factor=int(upsample_factor),
                    )
            if diag is not None:
                edge_rows.append({
                    'edge_type': 'horizontal',
                    'global_r0': int(r), 'global_c0': int(c), 'global_r1': int(r), 'global_c1': int(c + 1),
                    'delta_y_px': float(diag['consensus_dy_px']),
                    'delta_x_px': float(diag['consensus_dx_px']),
                    'support_n': int(diag['support_n']),
                    'n_candidates': int(diag['n_candidates']),
                    'median_candidate_residual_px': float(diag['median_candidate_residual_px']),
                    'max_candidate_residual_px': float(diag['max_candidate_residual_px']),
                    'best_evidence_id': str(diag['best_evidence_id']),
                    'dapi_support_z_count': int(support_count),
                    'best_dapi_z_index': -1 if best is None else int(best['z_index']),
                    'match_type': '' if best is None else str(best['match_type']),
                    'axis_overlap_px': 0 if best is None else int(best['axis_overlap_px']),
                    'intersection_area_px': 0 if best is None else int(best['intersection_area_px']),
                    'intersection_fraction': 0.0 if best is None else float(best['intersection_fraction']),
                    'window_y0_px': -1 if best is None else int(best['window_y0_px']),
                    'window_y1_px': -1 if best is None else int(best['window_y1_px']),
                    'window_x0_px': -1 if best is None else int(best['window_x0_px']),
                    'window_x1_px': -1 if best is None else int(best['window_x1_px']),
                })

    for r in range(n_rows - 1):
        for c in range(n_cols):
            nominal_dy_px = int(full_y_starts[r + 1] - full_y_starts[r])
            best = None
            support_count = 0
            for z_idx in dapi_z_values:
                key_top = (int(z_idx), int(full_y_starts[r]), int(full_x_starts[c]))
                key_bottom = (int(z_idx), int(full_y_starts[r + 1]), int(full_x_starts[c]))
                if key_top not in dapi_tile_map or key_bottom not in dapi_tile_map:
                    continue
                proposal = _detect_vertical_dapi_overlap_shared_scale(
                    np.asarray(dapi_tile_map[key_top], dtype=np.float32),
                    np.asarray(dapi_tile_map[key_bottom], dtype=np.float32),
                    nominal_dy_px=int(nominal_dy_px),
                    scale=scale,
                )
                if not bool(proposal.get('success', False)):
                    continue
                support_count += 1
                score = float(proposal.get('score', 0.0))
                if best is None or score > float(best['score']):
                    best = {
                        'z_index': int(z_idx),
                        'window_y0_px': int(proposal['window_y0_px']),
                        'window_y1_px': int(proposal['window_y1_px']),
                        'window_x0_px': int(proposal['window_x0_px']),
                        'window_x1_px': int(proposal['window_x1_px']),
                        'axis_overlap_px': int(proposal['axis_overlap_px']),
                        'intersection_area_px': int(proposal['intersection_area_px']),
                        'intersection_fraction': float(proposal['intersection_fraction']),
                        'match_type': str(proposal['match_type']),
                        'score': float(score),
                    }
            diag = None
            if support_count > 0:
                top_stack = []
                bottom_stack = []
                usable_z = []
                for z_idx in shared_bf_z_values:
                    key_top = (int(z_idx), int(full_y_starts[r]), int(full_x_starts[c]))
                    key_bottom = (int(z_idx), int(full_y_starts[r + 1]), int(full_x_starts[c]))
                    if key_top not in bf_tile_map or key_bottom not in bf_tile_map:
                        continue
                    top_stack.append(np.asarray(bf_tile_map[key_top], dtype=np.float32))
                    bottom_stack.append(np.asarray(bf_tile_map[key_bottom], dtype=np.float32))
                    usable_z.append(int(z_idx))
                if usable_z:
                    diag = _measure_vertical_shift_samez_full_bf(
                        top_stack=np.stack(top_stack, axis=0),
                        bottom_stack=np.stack(bottom_stack, axis=0),
                        shared_z_values=usable_z,
                        nominal_dy_px=int(nominal_dy_px),
                        upsample_factor=int(upsample_factor),
                    )
            if diag is not None:
                edge_rows.append({
                    'edge_type': 'vertical',
                    'global_r0': int(r), 'global_c0': int(c), 'global_r1': int(r + 1), 'global_c1': int(c),
                    'delta_y_px': float(diag['consensus_dy_px']),
                    'delta_x_px': float(diag['consensus_dx_px']),
                    'support_n': int(diag['support_n']),
                    'n_candidates': int(diag['n_candidates']),
                    'median_candidate_residual_px': float(diag['median_candidate_residual_px']),
                    'max_candidate_residual_px': float(diag['max_candidate_residual_px']),
                    'best_evidence_id': str(diag['best_evidence_id']),
                    'dapi_support_z_count': int(support_count),
                    'best_dapi_z_index': -1 if best is None else int(best['z_index']),
                    'match_type': '' if best is None else str(best['match_type']),
                    'axis_overlap_px': 0 if best is None else int(best['axis_overlap_px']),
                    'intersection_area_px': 0 if best is None else int(best['intersection_area_px']),
                    'intersection_fraction': 0.0 if best is None else float(best['intersection_fraction']),
                    'window_y0_px': -1 if best is None else int(best['window_y0_px']),
                    'window_y1_px': -1 if best is None else int(best['window_y1_px']),
                    'window_x0_px': -1 if best is None else int(best['window_x0_px']),
                    'window_x1_px': -1 if best is None else int(best['window_x1_px']),
                })

    edge_map = {
        (str(row['edge_type']), int(row['global_r0']), int(row['global_c0'])): dict(row)
        for row in edge_rows
    }
    edge_df = pd.DataFrame(edge_rows)
    if not edge_df.empty and np.any(edge_df['edge_type'].astype(str) == 'horizontal'):
        horizontal_df = edge_df[edge_df['edge_type'].astype(str) == 'horizontal']
        global_horizontal = {
            'dy_px': float(horizontal_df['delta_y_px'].median()),
            'dx_px': float(horizontal_df['delta_x_px'].median()),
        }
    else:
        global_horizontal = nominal_horizontal
    if not edge_df.empty and np.any(edge_df['edge_type'].astype(str) == 'vertical'):
        vertical_df = edge_df[edge_df['edge_type'].astype(str) == 'vertical']
        global_vertical = {
            'dy_px': float(vertical_df['delta_y_px'].median()),
            'dx_px': float(vertical_df['delta_x_px'].median()),
        }
    else:
        global_vertical = nominal_vertical
    result = {
        'edge_map': edge_map,
        'global_horizontal': global_horizontal,
        'global_vertical': global_vertical,
        'shared_bf_z_values': [int(v) for v in shared_bf_z_values],
        'n_measured_edges': int(len(edge_rows)),
        'dapi_scale': dict(scale),
    }
    _GLOBAL_DAPI_BF_SCENE_CACHE[cache_key] = result
    return result




def _solve_grid_deviations(
    n_nodes: int,
    edges: Sequence[Tuple[int, int, float, float, float]],
    max_shift_from_nominal_px: int = 96,
    prior_weight: float = 0.03,
) -> np.ndarray:
    rows = []
    rhs = []
    anchor = np.zeros(n_nodes, dtype=np.float64)
    anchor[0] = 1.0
    rows.append(anchor)
    rhs.append(0.0)
    prior_scale = float(np.sqrt(max(float(prior_weight), 1e-9)))
    for idx in range(n_nodes):
        eq = np.zeros(n_nodes, dtype=np.float64)
        eq[idx] = prior_scale
        rows.append(eq)
        rhs.append(0.0)
    for i, j, measured_delta, nominal_delta, weight in edges:
        scale = float(np.sqrt(max(float(weight), 1e-6)))
        eq = np.zeros(n_nodes, dtype=np.float64)
        eq[int(j)] = 1.0 * scale
        eq[int(i)] = -1.0 * scale
        rows.append(eq)
        rhs.append(float(measured_delta - nominal_delta) * scale)
    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=np.float64)
    dev, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    dev = np.clip(dev, -float(max_shift_from_nominal_px), float(max_shift_from_nominal_px))
    dev -= dev[0]
    return np.round(dev).astype(int)


def _estimate_local_registered_positions_dapi_bf_global_fill(
    evidence_tile_maps: Dict[str, Dict[Tuple[int, int, int], np.ndarray]],
    evidence_z_values: Dict[str, Sequence[int]],
    full_y_starts: Sequence[int],
    full_x_starts: Sequence[int],
    row_indices: Sequence[int],
    col_indices: Sequence[int],
    upsample_factor: int = 20,
    max_shift_from_nominal_px: int = 96,
) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], pd.DataFrame, pd.DataFrame]:
    row_indices = [int(v) for v in row_indices]
    col_indices = [int(v) for v in col_indices]
    local_n_rows = len(row_indices)
    local_n_cols = len(col_indices)
    node_index = {(r, c): (r * local_n_cols + c) for r in range(local_n_rows) for c in range(local_n_cols)}

    nominal_y = np.array([
        float(full_y_starts[row_indices[r]] - full_y_starts[0])
        for r in range(local_n_rows)
        for _ in range(local_n_cols)
    ], dtype=np.float64)
    nominal_x = np.array([
        float(full_x_starts[col_indices[c]] - full_x_starts[0])
        for _ in range(local_n_rows)
        for c in range(local_n_cols)
    ], dtype=np.float64)

    calibration = _calibrate_scene_dapi_bf_global_fill(
        evidence_tile_maps=evidence_tile_maps,
        evidence_z_values=evidence_z_values,
        full_y_starts=full_y_starts,
        full_x_starts=full_x_starts,
        upsample_factor=int(upsample_factor),
    )
    edge_map = calibration.get('edge_map', {})
    global_horizontal = calibration.get('global_horizontal', {'dy_px': 0.0, 'dx_px': 0.0})
    global_vertical = calibration.get('global_vertical', {'dy_px': 0.0, 'dx_px': 0.0})

    h_nominal_dx = float(full_x_starts[col_indices[1]] - full_x_starts[col_indices[0]]) if local_n_cols > 1 else 0.0
    v_nominal_dy = float(full_y_starts[row_indices[1]] - full_y_starts[row_indices[0]]) if local_n_rows > 1 else 0.0
    global_h_dy = float(global_horizontal.get('dy_px', 0.0))
    global_h_dx = float(global_horizontal.get('dx_px', h_nominal_dx))
    global_v_dy = float(global_vertical.get('dy_px', v_nominal_dy))
    global_v_dx = float(global_vertical.get('dx_px', 0.0))
    block_anchor_y = float(nominal_y[0])
    block_anchor_x = float(nominal_x[0])
    lattice_y = np.array([
        block_anchor_y + float(r) * global_v_dy + float(c) * global_h_dy
        for r in range(local_n_rows)
        for c in range(local_n_cols)
    ], dtype=np.float64)
    lattice_x = np.array([
        block_anchor_x + float(r) * global_v_dx + float(c) * global_h_dx
        for r in range(local_n_rows)
        for c in range(local_n_cols)
    ], dtype=np.float64)

    edge_rows = []

    for r_local, r_global in enumerate(row_indices):
        for c_local in range(local_n_cols - 1):
            c0 = int(col_indices[c_local])
            c1 = int(col_indices[c_local + 1])
            nominal_dx = int(full_x_starts[c1] - full_x_starts[c0])
            seam_key = ('horizontal', int(r_global), int(c0))
            measured = edge_map.get(seam_key)
            if measured is not None:
                dy = float(measured['delta_y_px'])
                dx = float(measured['delta_x_px'])
                evidence_strategy = 'dapi_guided_cyst_bf_full_overlap'
                edge_weight = float(max(8.0, 4.0 * int(measured.get('support_n', 1))))
                best_evidence_id = str(measured.get('best_evidence_id', ''))
                n_candidates = int(measured.get('n_candidates', 0))
                n_used = int(measured.get('support_n', 0))
                quality_median = float(1.0 / (1.0 + float(measured.get('median_candidate_residual_px', np.nan)))) if np.isfinite(float(measured.get('median_candidate_residual_px', np.nan))) else float('nan')
            else:
                dy = float(global_horizontal.get('dy_px', 0.0))
                dx = float(global_horizontal.get('dx_px', nominal_dx))
                evidence_strategy = 'global_orientation_fill'
                edge_weight = 1.0
                best_evidence_id = 'global_horizontal_median'
                n_candidates = 0
                n_used = 0
                quality_median = float('nan')
            i = node_index[(r_local, c_local)]
            j = node_index[(r_local, c_local + 1)]
            edge_rows.append({
                'edge_type': 'horizontal',
                'local_r0': int(r_local), 'local_c0': int(c_local), 'local_r1': int(r_local), 'local_c1': int(c_local + 1),
                'global_r0': int(r_global), 'global_c0': int(c0), 'global_r1': int(r_global), 'global_c1': int(c1),
                'node_i': int(i), 'node_j': int(j),
                'delta_y_px': float(dy), 'delta_x_px': float(dx),
                'nominal_dy_px': 0, 'nominal_dx_px': int(nominal_dx),
                'n_candidates': int(n_candidates), 'n_used': int(n_used),
                'best_evidence_id': str(best_evidence_id), 'quality_median': float(quality_median),
                'edge_weight': float(edge_weight),
                'evidence_strategy': str(evidence_strategy),
            })

    for r_local in range(local_n_rows - 1):
        r0 = int(row_indices[r_local])
        r1 = int(row_indices[r_local + 1])
        for c_local, c_global in enumerate(col_indices):
            c_global = int(c_global)
            nominal_dy = int(full_y_starts[r1] - full_y_starts[r0])
            seam_key = ('vertical', int(r0), int(c_global))
            measured = edge_map.get(seam_key)
            if measured is not None:
                dy = float(measured['delta_y_px'])
                dx = float(measured['delta_x_px'])
                evidence_strategy = 'dapi_guided_cyst_bf_full_overlap'
                edge_weight = float(max(8.0, 4.0 * int(measured.get('support_n', 1))))
                best_evidence_id = str(measured.get('best_evidence_id', ''))
                n_candidates = int(measured.get('n_candidates', 0))
                n_used = int(measured.get('support_n', 0))
                quality_median = float(1.0 / (1.0 + float(measured.get('median_candidate_residual_px', np.nan)))) if np.isfinite(float(measured.get('median_candidate_residual_px', np.nan))) else float('nan')
            else:
                dy = float(global_vertical.get('dy_px', nominal_dy))
                dx = float(global_vertical.get('dx_px', 0.0))
                evidence_strategy = 'global_orientation_fill'
                edge_weight = 1.0
                best_evidence_id = 'global_vertical_median'
                n_candidates = 0
                n_used = 0
                quality_median = float('nan')
            i = node_index[(r_local, c_local)]
            j = node_index[(r_local + 1, c_local)]
            edge_rows.append({
                'edge_type': 'vertical',
                'local_r0': int(r_local), 'local_c0': int(c_local), 'local_r1': int(r_local + 1), 'local_c1': int(c_local),
                'global_r0': int(r0), 'global_c0': int(c_global), 'global_r1': int(r1), 'global_c1': int(c_global),
                'node_i': int(i), 'node_j': int(j),
                'delta_y_px': float(dy), 'delta_x_px': float(dx),
                'nominal_dy_px': int(nominal_dy), 'nominal_dx_px': 0,
                'n_candidates': int(n_candidates), 'n_used': int(n_used),
                'best_evidence_id': str(best_evidence_id), 'quality_median': float(quality_median),
                'edge_weight': float(edge_weight),
                'evidence_strategy': str(evidence_strategy),
            })

    n_nodes = len(node_index)
    measured_rows = [row for row in edge_rows if str(row['evidence_strategy']) == 'dapi_guided_cyst_bf_full_overlap']
    measured_adj = {idx: [] for idx in range(n_nodes)}
    for row in measured_rows:
        i = int(row['node_i'])
        j = int(row['node_j'])
        measured_adj[i].append(j)
        measured_adj[j].append(i)

    component_nodes: List[List[int]] = []
    component_id_by_node: Dict[int, int] = {}
    seen_nodes = set()
    for node_idx in range(n_nodes):
        if node_idx in seen_nodes:
            continue
        stack = [int(node_idx)]
        comp = []
        seen_nodes.add(int(node_idx))
        while stack:
            cur = int(stack.pop())
            comp.append(cur)
            for nb in measured_adj.get(cur, []):
                nb = int(nb)
                if nb not in seen_nodes:
                    seen_nodes.add(nb)
                    stack.append(nb)
        comp = sorted(set(comp))
        comp_id = int(len(component_nodes))
        component_nodes.append(comp)
        for node in comp:
            component_id_by_node[int(node)] = comp_id

    n_components = len(component_nodes)
    comp_offset_y = np.zeros(n_nodes, dtype=np.float64)
    comp_offset_x = np.zeros(n_nodes, dtype=np.float64)
    comp_anchor_y = np.zeros(n_components, dtype=np.float64)
    comp_anchor_x = np.zeros(n_components, dtype=np.float64)
    comp_node_count = np.zeros(n_components, dtype=int)

    for comp_id, comp in enumerate(component_nodes):
        comp = [int(v) for v in comp]
        comp_node_count[int(comp_id)] = int(len(comp))
        local_map = {int(node): int(k) for k, node in enumerate(comp)}
        if len(comp) == 1:
            node = int(comp[0])
            comp_anchor_y[int(comp_id)] = float(lattice_y[node])
            comp_anchor_x[int(comp_id)] = float(lattice_x[node])
            continue
        y_edges_comp = []
        x_edges_comp = []
        for row in measured_rows:
            i = int(row['node_i'])
            j = int(row['node_j'])
            if i not in local_map or j not in local_map:
                continue
            li = int(local_map[i])
            lj = int(local_map[j])
            y_edges_comp.append((li, lj, float(row['delta_y_px']), float(row['nominal_dy_px']), 1.0))
            x_edges_comp.append((li, lj, float(row['delta_x_px']), float(row['nominal_dx_px']), 1.0))
        comp_dev_y = _solve_grid_deviations(
            len(comp),
            y_edges_comp,
            max_shift_from_nominal_px=int(max_shift_from_nominal_px),
            prior_weight=1e-6,
        ).astype(np.float64)
        comp_dev_x = _solve_grid_deviations(
            len(comp),
            x_edges_comp,
            max_shift_from_nominal_px=int(max_shift_from_nominal_px),
            prior_weight=1e-6,
        ).astype(np.float64)
        comp_solved_y = nominal_y[np.asarray(comp, dtype=int)] + comp_dev_y
        comp_solved_x = nominal_x[np.asarray(comp, dtype=int)] + comp_dev_x
        anchor_y = float(comp_solved_y[0])
        anchor_x = float(comp_solved_x[0])
        comp_indices = np.asarray(comp, dtype=int)
        for local_idx, node in enumerate(comp):
            comp_offset_y[int(node)] = float(comp_solved_y[int(local_idx)] - anchor_y)
            comp_offset_x[int(node)] = float(comp_solved_x[int(local_idx)] - anchor_x)
        comp_anchor_y[int(comp_id)] = float(np.mean(lattice_y[comp_indices] - comp_offset_y[comp_indices]))
        comp_anchor_x[int(comp_id)] = float(np.mean(lattice_x[comp_indices] - comp_offset_x[comp_indices]))

    for row in edge_rows:
        node_i = int(row['node_i'])
        node_j = int(row['node_j'])
        comp_i = int(component_id_by_node[node_i])
        comp_j = int(component_id_by_node[node_j])
        row['component_i'] = int(comp_i)
        row['component_j'] = int(comp_j)
        row['component_size_i'] = int(comp_node_count[comp_i])
        row['component_size_j'] = int(comp_node_count[comp_j])
        if comp_i == comp_j:
            row['component_relation'] = 'measured_component_internal'
        else:
            row['component_relation'] = 'component_bridge'

    solved_y = np.zeros(n_nodes, dtype=np.float64)
    solved_x = np.zeros(n_nodes, dtype=np.float64)
    for node_idx in range(n_nodes):
        comp_id = int(component_id_by_node[int(node_idx)])
        solved_y[int(node_idx)] = float(comp_anchor_y[comp_id] + comp_offset_y[int(node_idx)])
        solved_x[int(node_idx)] = float(comp_anchor_x[comp_id] + comp_offset_x[int(node_idx)])

    placements = {}
    placement_rows = []
    for (r_local, c_local), idx in node_index.items():
        r_global = int(row_indices[r_local])
        c_global = int(col_indices[c_local])
        yy = int(np.round(solved_y[idx]))
        xx = int(np.round(solved_x[idx]))
        comp_id = int(component_id_by_node[int(idx)])
        nom_y = int(full_y_starts[r_global] - full_y_starts[0])
        nom_x = int(full_x_starts[c_global] - full_x_starts[0])
        placements[(int(r_local), int(c_local))] = (yy, xx)
        placement_rows.append({
            'local_tile_row': int(r_local),
            'local_tile_col': int(c_local),
            'global_tile_row': int(r_global),
            'global_tile_col': int(c_global),
            'component_id': int(comp_id),
            'component_node_count': int(comp_node_count[comp_id]),
            'raw_y_start_px': int(full_y_starts[r_global]),
            'raw_x_start_px': int(full_x_starts[c_global]),
            'nominal_y_px': int(nom_y),
            'nominal_x_px': int(nom_x),
            'registered_y_px': int(yy),
            'registered_x_px': int(xx),
            'delta_from_nominal_y_px': int(yy - nom_y),
            'delta_from_nominal_x_px': int(xx - nom_x),
        })

    placement_df = pd.DataFrame(placement_rows).sort_values(['local_tile_row', 'local_tile_col']).reset_index(drop=True)
    edge_df = pd.DataFrame(edge_rows)
    return placements, placement_df, edge_df

def estimate_local_registered_positions_multi_channel(
    evidence_tile_maps: Dict[str, Dict[Tuple[int, int, int], np.ndarray]],
    evidence_z_values: Dict[str, Sequence[int]],
    full_y_starts: Sequence[int],
    full_x_starts: Sequence[int],
    row_indices: Sequence[int],
    col_indices: Sequence[int],
    upsample_factor: int = 20,
    max_shift_from_nominal_px: int = 64,
    keep_top_n: int = 3,
    align_method: str = 'experimental',
    use_corner_anchor_constraints: bool = False,
    corner_anchor_patch_size_px: int = 192,
    corner_anchor_radius_min_px: int = 26,
    corner_anchor_radius_max_px: int = 64,
    corner_anchor_max_role_candidates: int = 8,
    corner_anchor_max_cluster_spread_px: float = 48.0,
    corner_anchor_max_blocks_per_roi: int = 2,
):
    row_indices = [int(v) for v in row_indices]
    col_indices = [int(v) for v in col_indices]
    local_n_rows = len(row_indices)
    local_n_cols = len(col_indices)
    node_index = {(r, c): (r * local_n_cols + c) for r in range(local_n_rows) for c in range(local_n_cols)}

    if str(align_method) == 'foxf1_like':
        return _estimate_local_registered_positions_foxf1_like(
            evidence_tile_maps=evidence_tile_maps,
            evidence_z_values=evidence_z_values,
            full_y_starts=full_y_starts,
            full_x_starts=full_x_starts,
            row_indices=row_indices,
            col_indices=col_indices,
            upsample_factor=int(upsample_factor),
            max_shift_from_nominal_px=int(max_shift_from_nominal_px),
        )

    if str(align_method) == 'dapi_bf_global_fill':
        return _estimate_local_registered_positions_dapi_bf_global_fill(
            evidence_tile_maps=evidence_tile_maps,
            evidence_z_values=evidence_z_values,
            full_y_starts=full_y_starts,
            full_x_starts=full_x_starts,
            row_indices=row_indices,
            col_indices=col_indices,
            upsample_factor=int(upsample_factor),
            max_shift_from_nominal_px=int(max_shift_from_nominal_px),
        )

    nominal_y = np.array([
        float(full_y_starts[row_indices[r]] - full_y_starts[0])
        for r in range(local_n_rows)
        for _ in range(local_n_cols)
    ], dtype=np.float64)
    nominal_x = np.array([
        float(full_x_starts[col_indices[c]] - full_x_starts[0])
        for _ in range(local_n_rows)
        for c in range(local_n_cols)
    ], dtype=np.float64)

    y_edges = []
    x_edges = []
    edge_rows = []
    horizontal_edge_meta: Dict[Tuple[int, int], Dict[str, float]] = {}
    vertical_edge_meta: Dict[Tuple[int, int], Dict[str, float]] = {}

    for r_local, r_global in enumerate(row_indices):
        for c_local in range(local_n_cols - 1):
            c0 = col_indices[c_local]
            c1 = col_indices[c_local + 1]
            nominal_dx = int(full_x_starts[c1] - full_x_starts[c0])


            dapi_candidates = []
            dapi_tile_map = evidence_tile_maps.get('dapi')
            bf_tile_map = evidence_tile_maps.get('brightfield')
            if dapi_tile_map is not None and bf_tile_map is not None:
                dapi_z_values = [int(v) for v in evidence_z_values.get('dapi', [])]
                bf_z_values = {int(v) for v in evidence_z_values.get('brightfield', [])}
                for z_idx in dapi_z_values:
                    if int(z_idx) not in bf_z_values:
                        continue
                    key_a = (int(z_idx), int(full_y_starts[r_global]), int(full_x_starts[c0]))
                    key_b = (int(z_idx), int(full_y_starts[r_global]), int(full_x_starts[c1]))
                    if key_a not in dapi_tile_map or key_b not in dapi_tile_map or key_a not in bf_tile_map or key_b not in bf_tile_map:
                        continue
                    est = _estimate_horizontal_delta_dapi_guided_bf(
                        dapi_tile_map[key_a],
                        dapi_tile_map[key_b],
                        bf_tile_map[key_a],
                        bf_tile_map[key_b],
                        nominal_dx_px=nominal_dx,
                        upsample_factor=int(upsample_factor),
                    )
                    if est is None:
                        continue
                    est['channel'] = 'brightfield'
                    est['z_index_a'] = int(z_idx)
                    est['z_index_b'] = int(z_idx)
                    est['evidence_id'] = f"dapi_guided_bf:z{int(z_idx)}"
                    est['evidence_strategy'] = 'dapi_guided_cyst_bf'
                    dapi_candidates.append(est)
            if dapi_candidates:
                candidates = dapi_candidates
                evidence_strategy = 'dapi_guided_cyst_bf'
                edge_weight = 1.5
            else:
                bf_candidates = []
                bf_tile_map = evidence_tile_maps.get('brightfield')
                if bf_tile_map is not None:
                    bf_z_values = [int(v) for v in evidence_z_values.get('brightfield', [])]
                    for z_idx in bf_z_values:
                        key_a = (int(z_idx), int(full_y_starts[r_global]), int(full_x_starts[c0]))
                        key_b = (int(z_idx), int(full_y_starts[r_global]), int(full_x_starts[c1]))
                        if key_a not in bf_tile_map or key_b not in bf_tile_map:
                            continue
                        seam_est = _estimate_horizontal_delta_scored(
                            bf_tile_map[key_a],
                            bf_tile_map[key_b],
                            nominal_dx_px=nominal_dx,
                            upsample_factor=int(upsample_factor),
                        )
                        if seam_est is not None:
                            seam_est['channel'] = 'brightfield'
                            seam_est['z_index_a'] = int(z_idx)
                            seam_est['z_index_b'] = int(z_idx)
                            seam_est['evidence_id'] = f"brightfield_seam:z{int(z_idx)}"
                            seam_est['evidence_strategy'] = 'brightfield_structure'
                            bf_candidates.append(seam_est)
                        circle_est = _estimate_horizontal_delta_bf_circle(
                            bf_tile_map[key_a],
                            bf_tile_map[key_b],
                            nominal_dx_px=nominal_dx,
                        )
                        if circle_est is not None:
                            circle_est['channel'] = 'brightfield'
                            circle_est['z_index_a'] = int(z_idx)
                            circle_est['z_index_b'] = int(z_idx)
                            circle_est['evidence_id'] = f"brightfield_arc:z{int(z_idx)}"
                            circle_est['evidence_strategy'] = 'brightfield_well_arc'
                            bf_candidates.append(circle_est)
                candidates = bf_candidates
                if bf_candidates:
                    evidence_strategy = str(pd.DataFrame(bf_candidates).sort_values('quality', ascending=False).iloc[0]['evidence_strategy'])
                else:
                    evidence_strategy = 'nominal_fallback'
                edge_weight = 1.0 if bf_candidates else 0.25
            dy, dx, agg = _aggregate_delta_candidates(candidates, fallback_dy_px=0.0, fallback_dx_px=float(nominal_dx), keep_top_n=int(keep_top_n))
            i = node_index[(r_local, c_local)]
            j = node_index[(r_local, c_local + 1)]
            y_edges.append((i, j, dy, edge_weight))
            x_edges.append((i, j, dx, edge_weight))
            edge_rows.append({
                'edge_type': 'horizontal',
                'local_r0': int(r_local), 'local_c0': int(c_local), 'local_r1': int(r_local), 'local_c1': int(c_local + 1),
                'global_r0': int(r_global), 'global_c0': int(c0), 'global_r1': int(r_global), 'global_c1': int(c1),
                'delta_y_px': float(dy), 'delta_x_px': float(dx),
                'nominal_dy_px': 0, 'nominal_dx_px': int(nominal_dx),
                'n_candidates': int(agg['n_candidates']), 'n_used': int(agg['n_used']),
                'best_evidence_id': str(agg['best_evidence_id']), 'quality_median': float(agg['quality_median']),
                'edge_weight': float(edge_weight),
                'evidence_strategy': str(evidence_strategy),
            })
            horizontal_edge_meta[(int(r_local), int(c_local))] = {
                'n_candidates': int(agg['n_candidates']),
                'quality_median': float(agg['quality_median']) if np.isfinite(float(agg['quality_median'])) else float('nan'),
                'edge_weight': float(edge_weight),
            }

    for r_local in range(local_n_rows - 1):
        r0 = row_indices[r_local]
        r1 = row_indices[r_local + 1]
        for c_local, c_global in enumerate(col_indices):
            nominal_dy = int(full_y_starts[r1] - full_y_starts[r0])


            dapi_candidates = []
            dapi_tile_map = evidence_tile_maps.get('dapi')
            bf_tile_map = evidence_tile_maps.get('brightfield')
            if dapi_tile_map is not None and bf_tile_map is not None:
                dapi_z_values = [int(v) for v in evidence_z_values.get('dapi', [])]
                bf_z_values = {int(v) for v in evidence_z_values.get('brightfield', [])}
                for z_idx in dapi_z_values:
                    if int(z_idx) not in bf_z_values:
                        continue
                    key_a = (int(z_idx), int(full_y_starts[r0]), int(full_x_starts[c_global]))
                    key_b = (int(z_idx), int(full_y_starts[r1]), int(full_x_starts[c_global]))
                    if key_a not in dapi_tile_map or key_b not in dapi_tile_map or key_a not in bf_tile_map or key_b not in bf_tile_map:
                        continue
                    est = _estimate_vertical_delta_dapi_guided_bf(
                        dapi_tile_map[key_a],
                        dapi_tile_map[key_b],
                        bf_tile_map[key_a],
                        bf_tile_map[key_b],
                        nominal_dy_px=nominal_dy,
                        upsample_factor=int(upsample_factor),
                    )
                    if est is None:
                        continue
                    est['channel'] = 'brightfield'
                    est['z_index_a'] = int(z_idx)
                    est['z_index_b'] = int(z_idx)
                    est['evidence_id'] = f"dapi_guided_bf:z{int(z_idx)}"
                    est['evidence_strategy'] = 'dapi_guided_cyst_bf'
                    dapi_candidates.append(est)
            if dapi_candidates:
                candidates = dapi_candidates
                evidence_strategy = 'dapi_guided_cyst_bf'
                edge_weight = 1.5
            else:
                bf_candidates = []
                bf_tile_map = evidence_tile_maps.get('brightfield')
                if bf_tile_map is not None:
                    bf_z_values = [int(v) for v in evidence_z_values.get('brightfield', [])]
                    for z_idx in bf_z_values:
                        key_a = (int(z_idx), int(full_y_starts[r0]), int(full_x_starts[c_global]))
                        key_b = (int(z_idx), int(full_y_starts[r1]), int(full_x_starts[c_global]))
                        if key_a not in bf_tile_map or key_b not in bf_tile_map:
                            continue
                        seam_est = _estimate_vertical_delta_scored(
                            bf_tile_map[key_a],
                            bf_tile_map[key_b],
                            nominal_dy_px=nominal_dy,
                            upsample_factor=int(upsample_factor),
                        )
                        if seam_est is not None:
                            seam_est['channel'] = 'brightfield'
                            seam_est['z_index_a'] = int(z_idx)
                            seam_est['z_index_b'] = int(z_idx)
                            seam_est['evidence_id'] = f"brightfield_seam:z{int(z_idx)}"
                            seam_est['evidence_strategy'] = 'brightfield_structure'
                            bf_candidates.append(seam_est)
                        circle_est = _estimate_vertical_delta_bf_circle(
                            bf_tile_map[key_a],
                            bf_tile_map[key_b],
                            nominal_dy_px=nominal_dy,
                        )
                        if circle_est is not None:
                            circle_est['channel'] = 'brightfield'
                            circle_est['z_index_a'] = int(z_idx)
                            circle_est['z_index_b'] = int(z_idx)
                            circle_est['evidence_id'] = f"brightfield_arc:z{int(z_idx)}"
                            circle_est['evidence_strategy'] = 'brightfield_well_arc'
                            bf_candidates.append(circle_est)
                candidates = bf_candidates
                if bf_candidates:
                    evidence_strategy = str(pd.DataFrame(bf_candidates).sort_values('quality', ascending=False).iloc[0]['evidence_strategy'])
                else:
                    evidence_strategy = 'nominal_fallback'
                edge_weight = 1.0 if bf_candidates else 0.25
            dy, dx, agg = _aggregate_delta_candidates(candidates, fallback_dy_px=float(nominal_dy), fallback_dx_px=0.0, keep_top_n=int(keep_top_n))
            i = node_index[(r_local, c_local)]
            j = node_index[(r_local + 1, c_local)]
            y_edges.append((i, j, dy, edge_weight))
            x_edges.append((i, j, dx, edge_weight))
            edge_rows.append({
                'edge_type': 'vertical',
                'local_r0': int(r_local), 'local_c0': int(c_local), 'local_r1': int(r_local + 1), 'local_c1': int(c_local),
                'global_r0': int(r0), 'global_c0': int(c_global), 'global_r1': int(r1), 'global_c1': int(c_global),
                'delta_y_px': float(dy), 'delta_x_px': float(dx),
                'nominal_dy_px': int(nominal_dy), 'nominal_dx_px': 0,
                'n_candidates': int(agg['n_candidates']), 'n_used': int(agg['n_used']),
                'best_evidence_id': str(agg['best_evidence_id']), 'quality_median': float(agg['quality_median']),
                'edge_weight': float(edge_weight),
                'evidence_strategy': str(evidence_strategy),
            })
            vertical_edge_meta[(int(r_local), int(c_local))] = {
                'n_candidates': int(agg['n_candidates']),
                'quality_median': float(agg['quality_median']) if np.isfinite(float(agg['quality_median'])) else float('nan'),
                'edge_weight': float(edge_weight),
            }

    if bool(use_corner_anchor_constraints) and evidence_tile_maps:
        anchor_channel = str(next(iter(evidence_tile_maps.keys())))
        anchor_tile_map = evidence_tile_maps.get(anchor_channel, {})
        anchor_z_values = evidence_z_values.get(anchor_channel, [])
        block_priority_rows = []
        for r_local in range(local_n_rows - 1):
            for c_local in range(local_n_cols - 1):
                neighbor_meta = [
                    horizontal_edge_meta.get((int(r_local), int(c_local)), {}),
                    horizontal_edge_meta.get((int(r_local + 1), int(c_local)), {}),
                    vertical_edge_meta.get((int(r_local), int(c_local)), {}),
                    vertical_edge_meta.get((int(r_local), int(c_local + 1)), {}),
                ]
                missing_count = int(sum(int(meta.get('n_candidates', 0)) == 0 for meta in neighbor_meta))
                low_weight_count = int(sum(float(meta.get('edge_weight', 0.0)) < 1.0 for meta in neighbor_meta))
                if missing_count <= 0 and low_weight_count <= 0:
                    continue
                block_priority_rows.append({
                    'r_local': int(r_local),
                    'c_local': int(c_local),
                    'missing_count': int(missing_count),
                    'low_weight_count': int(low_weight_count),
                })
        block_priority_rows = sorted(
            block_priority_rows,
            key=lambda row: (-int(row['missing_count']), -int(row['low_weight_count']), int(row['r_local']), int(row['c_local'])),
        )[: int(max(1, corner_anchor_max_blocks_per_roi))]
        for block_row in block_priority_rows:
            r_local = int(block_row['r_local'])
            c_local = int(block_row['c_local'])
            constraints, anchor_meta = _estimate_corner_anchor_constraints(
                anchor_tile_map=anchor_tile_map,
                anchor_z_values=anchor_z_values,
                full_y_starts=full_y_starts,
                full_x_starts=full_x_starts,
                row_indices=row_indices,
                col_indices=col_indices,
                r_local=int(r_local),
                c_local=int(c_local),
                patch_size_px=int(corner_anchor_patch_size_px),
                radius_min_px=int(corner_anchor_radius_min_px),
                radius_max_px=int(corner_anchor_radius_max_px),
                max_role_candidates=int(corner_anchor_max_role_candidates),
                max_cluster_spread_px=float(corner_anchor_max_cluster_spread_px),
            )
            if not constraints:
                continue
            for row in constraints:
                i = node_index[(int(row['local_r0']), int(row['local_c0']))]
                j = node_index[(int(row['local_r1']), int(row['local_c1']))]
                weight = float(row['weight'])
                y_edges.append((i, j, float(row['delta_y_px']), weight))
                x_edges.append((i, j, float(row['delta_x_px']), weight))
                edge_rows.append({
                    'edge_type': str(row['edge_type']),
                    'local_r0': int(row['local_r0']), 'local_c0': int(row['local_c0']),
                    'local_r1': int(row['local_r1']), 'local_c1': int(row['local_c1']),
                    'global_r0': int(row['global_r0']), 'global_c0': int(row['global_c0']),
                    'global_r1': int(row['global_r1']), 'global_c1': int(row['global_c1']),
                    'delta_y_px': float(row['delta_y_px']), 'delta_x_px': float(row['delta_x_px']),
                    'nominal_dy_px': float('nan'), 'nominal_dx_px': float('nan'),
                    'n_candidates': 4, 'n_used': 4,
                    'best_evidence_id': str(anchor_channel),
                    'quality_median': float(anchor_meta.get('anchor_score_mean_norm', np.nan)),
                    'edge_weight': float(weight),
                    'anchor_center_y_px': float(anchor_meta.get('anchor_center_y_px', np.nan)),
                    'anchor_center_x_px': float(anchor_meta.get('anchor_center_x_px', np.nan)),
                    'anchor_spread_px': float(anchor_meta.get('anchor_spread_px', np.nan)),
                    'anchor_radius_median_px': float(anchor_meta.get('anchor_radius_median_px', np.nan)),
                    'anchor_score_mean_norm': float(anchor_meta.get('anchor_score_mean_norm', np.nan)),
                })

    solved_y = _solve_grid_positions(len(node_index), y_edges, nominal_vals=nominal_y, max_shift_from_nominal_px=int(max_shift_from_nominal_px))
    solved_x = _solve_grid_positions(len(node_index), x_edges, nominal_vals=nominal_x, max_shift_from_nominal_px=int(max_shift_from_nominal_px))
    solved_y = solved_y + int(full_y_starts[row_indices[0]] - full_y_starts[0])
    solved_x = solved_x + int(full_x_starts[col_indices[0]] - full_x_starts[0])

    placements = {}
    placement_rows = []
    for (r_local, c_local), idx in node_index.items():
        r_global = row_indices[r_local]
        c_global = col_indices[c_local]
        yy = int(solved_y[idx])
        xx = int(solved_x[idx])
        nom_y = int(full_y_starts[r_global] - full_y_starts[0])
        nom_x = int(full_x_starts[c_global] - full_x_starts[0])
        placements[(int(r_local), int(c_local))] = (yy, xx)
        placement_rows.append({
            'local_tile_row': int(r_local),
            'local_tile_col': int(c_local),
            'global_tile_row': int(r_global),
            'global_tile_col': int(c_global),
            'raw_y_start_px': int(full_y_starts[r_global]),
            'raw_x_start_px': int(full_x_starts[c_global]),
            'nominal_y_px': int(nom_y),
            'nominal_x_px': int(nom_x),
            'registered_y_px': int(yy),
            'registered_x_px': int(xx),
            'delta_from_nominal_y_px': int(yy - nom_y),
            'delta_from_nominal_x_px': int(xx - nom_x),
        })

    placement_df = pd.DataFrame(placement_rows).sort_values(['local_tile_row', 'local_tile_col']).reset_index(drop=True)
    edge_df = pd.DataFrame(edge_rows)
    return placements, placement_df, edge_df


def extract_locally_aligned_roi_bundle(
    channel_infos: Dict[str, CziTileLatticeInfo],
    channel_tile_maps: Dict[str, Dict[Tuple[int, int, int], np.ndarray]],
    seed_center_y_px: float,
    seed_center_x_px: float,
    crop_size_px: int,
    evidence_channels: Sequence[str] = ('brightfield',),
    align_margin_px: int = 128,
    align_upsample_factor: int = 20,
    align_max_shift_from_nominal_px: int = 64,
    align_keep_top_n: int = 3,
    align_method: str = 'experimental',
    refine_reference_channel: str = 'dapi',
    refine_percentile: float = 99.0,
    refine_smooth_sigma: float = 3.0,
    refine_central_fraction: float = 0.85,
    refine_max_shift_fraction: float = 0.18,
) -> Dict[str, object]:
    base_info = channel_infos[str(refine_reference_channel)]
    subset = select_local_tile_subset(
        info=base_info,
        center_y_px=float(seed_center_y_px),
        center_x_px=float(seed_center_x_px),
        crop_size_px=int(crop_size_px),
        extra_margin_px=int(align_margin_px),
    )
    row_indices = [int(v) for v in subset['row_indices']]
    col_indices = [int(v) for v in subset['col_indices']]
    sub_y_starts = [int(base_info.y_starts[r]) for r in row_indices]
    sub_x_starts = [int(base_info.x_starts[c]) for c in col_indices]
    evidence_tile_maps = {str(ch): channel_tile_maps[str(ch)] for ch in evidence_channels}
    evidence_z_values = {str(ch): channel_infos[str(ch)].z_values for ch in evidence_channels}
    placements, placement_df, edge_df = estimate_local_registered_positions_multi_channel(
        evidence_tile_maps=evidence_tile_maps,
        evidence_z_values=evidence_z_values,
        full_y_starts=base_info.y_starts,
        full_x_starts=base_info.x_starts,
        row_indices=row_indices,
        col_indices=col_indices,
        upsample_factor=int(align_upsample_factor),
        max_shift_from_nominal_px=int(align_max_shift_from_nominal_px),
        keep_top_n=int(align_keep_top_n),
        align_method=str(align_method),
    )

    local_origin_y_px = int(min(y for y, _ in placements.values()))
    local_origin_x_px = int(min(x for _, x in placements.values()))
    seed_local_y_px = float(seed_center_y_px - local_origin_y_px)
    seed_local_x_px = float(seed_center_x_px - local_origin_x_px)

    seed_stack_crops = {}
    seed_max_crops = {}
    stack_crops = {}
    max_crops = {}
    local_stacks = {}
    for channel, tile_map in channel_tile_maps.items():
        sub_tile_map = {
            (int(z), int(y0), int(x0)): img
            for (z, y0, x0), img in tile_map.items()
            if int(y0) in set(sub_y_starts) and int(x0) in set(sub_x_starts)
        }
        stack = restitch_single_channel_stack(
            tile_map=sub_tile_map,
            y_starts=sub_y_starts,
            x_starts=sub_x_starts,
            z_values=channel_infos[str(channel)].z_values,
            placements=placements,
        ).astype(np.float32)
        local_stacks[str(channel)] = stack

    ref_img = np.max(local_stacks[str(refine_reference_channel)], axis=0).astype(np.float32)
    ref_crop, _ = crop_square_with_bounds(ref_img, seed_local_y_px, seed_local_x_px, crop_size_px)
    est = estimate_local_content_center(
        ref_crop,
        percentile=float(refine_percentile),
        smooth_sigma=float(refine_smooth_sigma),
        central_fraction=float(refine_central_fraction),
    )
    max_shift_px = float(crop_size_px) * float(refine_max_shift_fraction)
    local_shift_y_px = 0.0
    local_shift_x_px = 0.0
    local_refined = False
    if bool(est['success']):
        local_shift_y_px = float(np.clip(est['center_y_px'] - (ref_crop.shape[0] / 2.0), -max_shift_px, max_shift_px))
        local_shift_x_px = float(np.clip(est['center_x_px'] - (ref_crop.shape[1] / 2.0), -max_shift_px, max_shift_px))
        local_refined = bool((abs(local_shift_y_px) > 1e-6) or (abs(local_shift_x_px) > 1e-6))

    refined_local_y_px = float(seed_local_y_px + local_shift_y_px)
    refined_local_x_px = float(seed_local_x_px + local_shift_x_px)
    refined_global_y_px = float(local_origin_y_px + refined_local_y_px)
    refined_global_x_px = float(local_origin_x_px + refined_local_x_px)

    seed_crop_bounds = None
    crop_bounds = None
    for channel, stack in local_stacks.items():
        seed_stack_crop, seed_bounds = crop_stack_with_bounds(stack, seed_local_y_px, seed_local_x_px, crop_size_px)
        seed_stack_crops[str(channel)] = seed_stack_crop.astype(np.float32)
        seed_max_crops[str(channel)] = np.max(seed_stack_crop, axis=0).astype(np.float32)
        if seed_crop_bounds is None:
            seed_crop_bounds = seed_bounds
        stack_crop, bounds = crop_stack_with_bounds(stack, refined_local_y_px, refined_local_x_px, crop_size_px)
        stack_crops[str(channel)] = stack_crop.astype(np.float32)
        max_crops[str(channel)] = np.max(stack_crop, axis=0).astype(np.float32)
        if crop_bounds is None:
            crop_bounds = bounds

    placement_abs = np.hypot(
        placement_df['delta_from_nominal_y_px'].to_numpy(dtype=np.float64),
        placement_df['delta_from_nominal_x_px'].to_numpy(dtype=np.float64),
    )
    summary = {
        'seed_center_y_px': float(seed_center_y_px),
        'seed_center_x_px': float(seed_center_x_px),
        'seed_local_y_px': float(seed_local_y_px),
        'seed_local_x_px': float(seed_local_x_px),
        'refined_local_y_px': float(refined_local_y_px),
        'refined_local_x_px': float(refined_local_x_px),
        'center_y_px': float(refined_global_y_px),
        'center_x_px': float(refined_global_x_px),
        'local_shift_y_px': float(local_shift_y_px),
        'local_shift_x_px': float(local_shift_x_px),
        'local_refined': bool(local_refined),
        'local_refine_attempted': True,
        'local_weight_sum': float(est['weight_sum']),
        'local_threshold': float(est['threshold']),
        'local_origin_y_px': int(local_origin_y_px),
        'local_origin_x_px': int(local_origin_x_px),
        'roi_tile_row_count': int(len(row_indices)),
        'roi_tile_col_count': int(len(col_indices)),
        'roi_tile_count': int(len(row_indices) * len(col_indices)),
        'roi_local_align_applied': bool(np.any(placement_abs > 1e-6)),
        'roi_local_align_max_abs_tile_shift_px': float(np.max(placement_abs)) if placement_abs.size else 0.0,
        'roi_local_align_median_abs_tile_shift_px': float(np.median(placement_abs)) if placement_abs.size else 0.0,
        'roi_local_align_edge_count': int(len(edge_df)),
        'roi_local_align_candidate_count': int(edge_df['n_candidates'].sum()) if len(edge_df) else 0,
        'roi_local_align_corner_anchor_edge_count': int(np.sum(edge_df['edge_type'].astype(str).str.startswith('corner_anchor'))) if len(edge_df) else 0,
    }
    return {
        'seed_stack_crops': seed_stack_crops,
        'seed_max_crops': seed_max_crops,
        'stack_crops': stack_crops,
        'max_crops': max_crops,
        'placement_df': placement_df,
        'edge_df': edge_df,
        'subset': subset,
        'seed_crop_bounds_local': seed_crop_bounds,
        'crop_bounds_local': crop_bounds,
        'summary': summary,
    }

def _principal_axes_from_points(points_yx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_yx, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis0 = vt[0]
    axis1 = vt[1]
    # prefer axis_x to be the one more aligned with image x (col)
    if abs(axis1[1]) > abs(axis0[1]):
        axis_x = axis1
        axis_y = axis0
    else:
        axis_x = axis0
        axis_y = axis1
    # enforce consistent directions
    if axis_x[1] < 0:
        axis_x = -axis_x
    if axis_y[0] < 0:
        axis_y = -axis_y
    return center, axis_y, axis_x


def detect_array_grid_centers(
    dapi_img: np.ndarray,
    expected_rows: int = 6,
    expected_cols: int = 7,
    downsample: int = 8,
    blur_sigma_ds: float = 24.0,
    min_distance_ds: int = 120,
    threshold_rel: float = 0.15,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    arr = robust_rescale(dapi_img)
    ds = arr[::int(downsample), ::int(downsample)]
    smooth = filters.gaussian(ds, sigma=float(blur_sigma_ds), preserve_range=True)
    expected_n = int(expected_rows * expected_cols)
    peaks = feature.peak_local_max(
        smooth,
        min_distance=int(min_distance_ds),
        threshold_rel=float(threshold_rel),
        exclude_border=False,
        num_peaks=int(expected_n),
    )

    if peaks.shape[0] == expected_n:
        points_yx_ds = peaks.astype(np.float64)
        center_ds, axis_y, axis_x = _principal_axes_from_points(points_yx_ds)
        centered = points_yx_ds - center_ds
        row_coords = centered @ axis_y
        col_coords = centered @ axis_x

        row_sorted = np.sort(row_coords)
        col_sorted = np.sort(col_coords)
        row_groups = np.array_split(row_sorted, int(expected_rows))
        col_groups = np.array_split(col_sorted, int(expected_cols))
        row_centers = np.array([g.mean() for g in row_groups], dtype=np.float64)
        col_centers = np.array([g.mean() for g in col_groups], dtype=np.float64)

        rows = []
        for r, rv in enumerate(row_centers):
            for c, cv in enumerate(col_centers):
                pt_ds = center_ds + rv * axis_y + cv * axis_x
                rows.append({
                    'grid_row': int(r),
                    'grid_col': int(c),
                    'center_y_px': float(pt_ds[0] * int(downsample)),
                    'center_x_px': float(pt_ds[1] * int(downsample)),
                    'inferred_center': False,
                })
        centers_df = pd.DataFrame(rows).sort_values(['grid_row', 'grid_col']).reset_index(drop=True)
        spacing_y = float(np.median(np.diff(row_centers)) * int(downsample)) if len(row_centers) > 1 else np.nan
        spacing_x = float(np.median(np.diff(col_centers)) * int(downsample)) if len(col_centers) > 1 else np.nan
        qc = {
            'detection_mode': 'direct_2d_peaks',
            'peak_points_ds_yx': points_yx_ds,
            'smooth_ds': smooth,
            'downsample': int(downsample),
            'axis_y': axis_y,
            'axis_x': axis_x,
            'center_ds_yx': center_ds,
            'row_centers_ds': row_centers,
            'col_centers_ds': col_centers,
            'row_spacing_px': spacing_y,
            'col_spacing_px': spacing_x,
        }
        return centers_df, qc

    row_profile = smooth.sum(axis=1)
    row_peaks, _ = find_peaks(row_profile, distance=max(8, int(min_distance_ds)))
    if row_peaks.size != int(expected_rows):
        raise RuntimeError(
            f'Expected {expected_rows} row peaks in fallback grid detector, found {int(row_peaks.size)}.'
        )

    row_band_halfwidth = max(40, int(round(min_distance_ds * 2.0 / 3.0)))
    row_peak_lists: List[np.ndarray] = []
    for row_peak in row_peaks:
        y0 = max(0, int(row_peak) - row_band_halfwidth)
        y1 = min(smooth.shape[0], int(row_peak) + row_band_halfwidth + 1)
        col_profile = smooth[y0:y1].sum(axis=0)
        col_peaks, _ = find_peaks(col_profile, distance=max(8, int(min_distance_ds)))
        if col_peaks.size not in (int(expected_cols), int(expected_cols - 1)):
            raise RuntimeError(
                f'Row-aware fallback expected {expected_cols} or {expected_cols - 1} column peaks per row, '
                f'but row peak {int(row_peak)} had {int(col_peaks.size)}.'
            )
        row_peak_lists.append(col_peaks.astype(np.float64))

    complete_rows = [cols for cols in row_peak_lists if cols.size == int(expected_cols)]
    if not complete_rows:
        raise RuntimeError(
            f'Expected at least one complete row with {expected_cols} column peaks; none were found.'
        )
    reference_cols = np.median(np.stack(complete_rows, axis=0), axis=0)

    rows = []
    inferred_missing_cols: Dict[int, int] = {}
    predicted_cols_all = []
    for grid_row, (row_peak, observed_cols) in enumerate(zip(row_peaks.astype(np.float64), row_peak_lists)):
        if observed_cols.size == int(expected_cols):
            design = np.c_[np.ones_like(reference_cols), reference_cols]
            intercept, scale = np.linalg.lstsq(design, observed_cols, rcond=None)[0]
            predicted_cols = intercept + scale * reference_cols
            missing_col = None
        else:
            best = None
            for missing_idx in range(int(expected_cols)):
                kept_reference = np.delete(reference_cols, missing_idx)
                design = np.c_[np.ones_like(kept_reference), kept_reference]
                intercept, scale = np.linalg.lstsq(design, observed_cols, rcond=None)[0]
                predicted_kept = intercept + scale * kept_reference
                mse = float(np.mean((observed_cols - predicted_kept) ** 2))
                if best is None or mse < best[0]:
                    best = (mse, missing_idx, intercept, scale)
            _, missing_col, intercept, scale = best
            predicted_cols = intercept + scale * reference_cols
            inferred_missing_cols[int(grid_row)] = int(missing_col)

        predicted_cols_all.append(predicted_cols.astype(np.float64))
        for grid_col, pred_col in enumerate(predicted_cols.astype(np.float64)):
            rows.append({
                'grid_row': int(grid_row),
                'grid_col': int(grid_col),
                'center_y_px': float(row_peak * int(downsample)),
                'center_x_px': float(pred_col * int(downsample)),
                'inferred_center': bool(observed_cols.size != int(expected_cols) and missing_col is not None and int(grid_col) == int(missing_col)),
            })

    centers_df = pd.DataFrame(rows).sort_values(['grid_row', 'grid_col']).reset_index(drop=True)
    row_centers_ds = row_peaks.astype(np.float64)
    predicted_cols_arr = np.stack(predicted_cols_all, axis=0)
    col_centers_ds = np.median(predicted_cols_arr, axis=0)
    spacing_y = float(np.median(np.diff(row_centers_ds)) * int(downsample)) if row_centers_ds.size > 1 else np.nan
    spacing_x = float(np.median(np.diff(col_centers_ds)) * int(downsample)) if col_centers_ds.size > 1 else np.nan
    qc = {
        'detection_mode': 'rowwise_profile_fallback',
        'peak_points_ds_yx': peaks.astype(np.float64),
        'smooth_ds': smooth,
        'downsample': int(downsample),
        'axis_y': np.array([1.0, 0.0], dtype=np.float64),
        'axis_x': np.array([0.0, 1.0], dtype=np.float64),
        'center_ds_yx': np.array([row_centers_ds.mean(), col_centers_ds.mean()], dtype=np.float64),
        'row_centers_ds': row_centers_ds,
        'col_centers_ds': col_centers_ds,
        'row_spacing_px': spacing_y,
        'col_spacing_px': spacing_x,
        'row_peak_lists_ds': [cols.tolist() for cols in row_peak_lists],
        'reference_cols_ds': reference_cols,
        'predicted_cols_by_row_ds': predicted_cols_arr,
        'inferred_missing_cols': inferred_missing_cols,
    }
    if len(rows) != expected_n:
        raise RuntimeError(f'Expected {expected_n} inferred grid centers, found {len(rows)}.')
    return centers_df, qc

def estimate_local_content_center(
    img: np.ndarray,
    percentile: float = 99.0,
    smooth_sigma: float = 3.0,
    central_fraction: float = 0.85,
) -> Dict[str, float | bool]:
    arr = robust_rescale(img)
    if float(smooth_sigma) > 0:
        arr = filters.gaussian(arr, sigma=float(smooth_sigma), preserve_range=True)
    H, W = arr.shape
    yy, xx = np.indices(arr.shape)

    mask = np.zeros_like(arr, dtype=bool)
    ch = max(32, int(round(H * float(central_fraction))))
    cw = max(32, int(round(W * float(central_fraction))))
    y0 = max(0, (H - ch) // 2)
    x0 = max(0, (W - cw) // 2)
    mask[y0:y0 + ch, x0:x0 + cw] = True

    vals = arr[mask & np.isfinite(arr)]
    if vals.size == 0:
        return {
            'success': False,
            'center_y_px': float(H / 2.0),
            'center_x_px': float(W / 2.0),
            'threshold': float('nan'),
            'weight_sum': 0.0,
        }

    thr = float(np.percentile(vals, float(percentile)))
    weights = np.clip(arr - thr, 0.0, None)
    weights = np.where(mask, weights, 0.0)
    weight_sum = float(weights.sum())

    if weight_sum <= 0:
        return {
            'success': False,
            'center_y_px': float(H / 2.0),
            'center_x_px': float(W / 2.0),
            'threshold': thr,
            'weight_sum': 0.0,
        }

    center_y = float((yy * weights).sum() / weight_sum)
    center_x = float((xx * weights).sum() / weight_sum)
    return {
        'success': True,
        'center_y_px': center_y,
        'center_x_px': center_x,
        'threshold': thr,
        'weight_sum': weight_sum,
    }


def refine_grid_centers_by_local_content(
    centers_df: pd.DataFrame,
    reference_img: np.ndarray,
    crop_size_px: int,
    percentile: float = 99.0,
    smooth_sigma: float = 3.0,
    central_fraction: float = 0.85,
    max_shift_fraction: float = 0.18,
) -> pd.DataFrame:
    image_h, image_w = np.asarray(reference_img).shape
    half = float(crop_size_px) / 2.0
    max_shift_px = float(crop_size_px) * float(max_shift_fraction)
    rows = []
    for _, row in centers_df.iterrows():
        seed_y = float(row.center_y_px)
        seed_x = float(row.center_x_px)
        can_refine = (
            seed_y >= half and seed_y < (image_h - half) and
            seed_x >= half and seed_x < (image_w - half) and
            not bool(row.get('inferred_center', False))
        )
        local_shift_y = 0.0
        local_shift_x = 0.0
        local_success = False
        local_weight_sum = 0.0
        local_threshold = float('nan')
        if can_refine:
            crop, _ = crop_square_with_bounds(reference_img, seed_y, seed_x, crop_size_px)
            est = estimate_local_content_center(
                crop,
                percentile=float(percentile),
                smooth_sigma=float(smooth_sigma),
                central_fraction=float(central_fraction),
            )
            local_success = bool(est['success'])
            local_weight_sum = float(est['weight_sum'])
            local_threshold = float(est['threshold'])
            if local_success:
                local_shift_y = float(np.clip(est['center_y_px'] - (crop.shape[0] / 2.0), -max_shift_px, max_shift_px))
                local_shift_x = float(np.clip(est['center_x_px'] - (crop.shape[1] / 2.0), -max_shift_px, max_shift_px))

        refined = dict(row)
        refined['seed_center_y_px'] = seed_y
        refined['seed_center_x_px'] = seed_x
        refined['center_y_px'] = float(seed_y + local_shift_y)
        refined['center_x_px'] = float(seed_x + local_shift_x)
        refined['local_shift_y_px'] = float(local_shift_y)
        refined['local_shift_x_px'] = float(local_shift_x)
        refined['local_refined'] = bool(local_success and ((abs(local_shift_y) > 1e-6) or (abs(local_shift_x) > 1e-6)))
        refined['local_refine_attempted'] = bool(can_refine)
        refined['local_weight_sum'] = float(local_weight_sum)
        refined['local_threshold'] = float(local_threshold)
        rows.append(refined)
    return pd.DataFrame(rows).sort_values(['grid_row', 'grid_col']).reset_index(drop=True)


def detect_array_grid_centers_split_blocks(
    dapi_img: np.ndarray,
    top_rows: int = 3,
    bottom_rows: int = 3,
    top_cols: int = 7,
    bottom_cols: int = 6,
    downsample: int = 8,
    blur_sigma_ds: float = 24.0,
    min_distance_ds: int = 120,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    arr = robust_rescale(dapi_img)
    ds = arr[::int(downsample), ::int(downsample)]
    smooth = filters.gaussian(ds, sigma=float(blur_sigma_ds), preserve_range=True)

    expected_rows = int(top_rows + bottom_rows)
    row_profile = smooth.sum(axis=1)
    row_peaks, _ = find_peaks(row_profile, distance=max(8, int(min_distance_ds)))
    if row_peaks.size < expected_rows:
        raise RuntimeError(
            f'Expected at least {expected_rows} row peaks in split-block detector, found {int(row_peaks.size)}.'
        )
    if row_peaks.size > expected_rows:
        strengths = row_profile[row_peaks]
        keep_idx = np.argsort(strengths)[-expected_rows:]
        row_peaks = np.sort(row_peaks[keep_idx])

    row_band_halfwidth = max(40, int(round(min_distance_ds * 2.0 / 3.0)))
    row_peak_lists: List[np.ndarray] = []
    expected_cols_by_row: List[int] = []
    for row_idx, row_peak in enumerate(row_peaks):
        expected_cols = int(top_cols if row_idx < int(top_rows) else bottom_cols)
        expected_cols_by_row.append(expected_cols)
        y0 = max(0, int(row_peak) - row_band_halfwidth)
        y1 = min(smooth.shape[0], int(row_peak) + row_band_halfwidth + 1)
        col_profile = smooth[y0:y1].sum(axis=0)
        col_peaks, _ = find_peaks(col_profile, distance=max(8, int(min_distance_ds)))
        if col_peaks.size < max(1, expected_cols - 1):
            raise RuntimeError(
                f'Split-block detector expected at least {expected_cols - 1} column peaks for row {row_idx}, found {int(col_peaks.size)}.'
            )
        if col_peaks.size > expected_cols:
            strengths = col_profile[col_peaks]
            keep_idx = np.argsort(strengths)[-expected_cols:]
            col_peaks = np.sort(col_peaks[keep_idx])
        row_peak_lists.append(col_peaks.astype(np.float64))

    block_specs = [
        ('top', list(range(int(top_rows))), int(top_cols)),
        ('bottom', list(range(int(top_rows), expected_rows)), int(bottom_cols)),
    ]

    rows = []
    inferred_missing_cols: Dict[int, int] = {}
    reference_cols_by_block: Dict[str, np.ndarray] = {}
    predicted_cols_by_row: Dict[int, np.ndarray] = {}

    for block_name, row_indices, expected_cols in block_specs:
        complete_rows = [row_peak_lists[i] for i in row_indices if row_peak_lists[i].size == expected_cols]
        if not complete_rows:
            raise RuntimeError(
                f'Split-block detector expected at least one complete {block_name} row with {expected_cols} columns; none were found.'
            )
        reference_cols = np.median(np.stack(complete_rows, axis=0), axis=0)
        reference_cols_by_block[block_name] = reference_cols

        for grid_row in row_indices:
            observed_cols = row_peak_lists[grid_row]
            if observed_cols.size == expected_cols:
                design = np.c_[np.ones_like(reference_cols), reference_cols]
                intercept, scale = np.linalg.lstsq(design, observed_cols, rcond=None)[0]
                predicted_cols = intercept + scale * reference_cols
                missing_col = None
            elif observed_cols.size == expected_cols - 1:
                best = None
                for missing_idx in range(expected_cols):
                    kept_reference = np.delete(reference_cols, missing_idx)
                    design = np.c_[np.ones_like(kept_reference), kept_reference]
                    intercept, scale = np.linalg.lstsq(design, observed_cols, rcond=None)[0]
                    predicted_kept = intercept + scale * kept_reference
                    mse = float(np.mean((observed_cols - predicted_kept) ** 2))
                    if best is None or mse < best[0]:
                        best = (mse, missing_idx, intercept, scale)
                _, missing_col, intercept, scale = best
                predicted_cols = intercept + scale * reference_cols
                inferred_missing_cols[int(grid_row)] = int(missing_col)
            else:
                raise RuntimeError(
                    f'Split-block detector expected {expected_cols} or {expected_cols - 1} columns for row {grid_row}, found {int(observed_cols.size)}.'
                )

            predicted_cols_by_row[int(grid_row)] = predicted_cols.astype(np.float64)
            for grid_col, pred_col in enumerate(predicted_cols.astype(np.float64)):
                rows.append({
                    'grid_row': int(grid_row),
                    'grid_col': int(grid_col),
                    'grid_block': str(block_name),
                    'expected_cols_in_block': int(expected_cols),
                    'center_y_px': float(row_peaks[grid_row] * int(downsample)),
                    'center_x_px': float(pred_col * int(downsample)),
                    'inferred_center': bool(observed_cols.size != expected_cols and missing_col is not None and int(grid_col) == int(missing_col)),
                })

    centers_df = pd.DataFrame(rows).sort_values(['grid_row', 'grid_col']).reset_index(drop=True)
    row_spacing_px = float(np.median(np.diff(row_peaks.astype(np.float64))) * int(downsample)) if row_peaks.size > 1 else np.nan
    top_col_spacing_px = float(np.median(np.diff(reference_cols_by_block['top'])) * int(downsample)) if reference_cols_by_block['top'].size > 1 else np.nan
    bottom_col_spacing_px = float(np.median(np.diff(reference_cols_by_block['bottom'])) * int(downsample)) if reference_cols_by_block['bottom'].size > 1 else np.nan
    finite_spacings = [v for v in [top_col_spacing_px, bottom_col_spacing_px] if np.isfinite(v)]
    min_col_spacing_px = float(min(finite_spacings)) if finite_spacings else np.nan

    qc = {
        'detection_mode': 'split_row_blocks',
        'smooth_ds': smooth,
        'downsample': int(downsample),
        'row_centers_ds': row_peaks.astype(np.float64),
        'row_peak_lists_ds': [cols.tolist() for cols in row_peak_lists],
        'expected_cols_by_row': expected_cols_by_row,
        'reference_cols_top_ds': reference_cols_by_block['top'],
        'reference_cols_bottom_ds': reference_cols_by_block['bottom'],
        'predicted_cols_by_row_ds': {int(k): v.tolist() for k, v in predicted_cols_by_row.items()},
        'row_spacing_px': row_spacing_px,
        'top_col_spacing_px': top_col_spacing_px,
        'bottom_col_spacing_px': bottom_col_spacing_px,
        'col_spacing_px': min_col_spacing_px,
        'inferred_missing_cols': inferred_missing_cols,
        'expected_array_count': int(top_rows * top_cols + bottom_rows * bottom_cols),
    }
    return centers_df, qc


def derive_crop_size_px(row_spacing_px: float, col_spacing_px: float, crop_fraction: float = 0.82) -> int:
    base = float(min(row_spacing_px, col_spacing_px))
    if not np.isfinite(base) or base <= 0:
        raise RuntimeError(f'Invalid grid spacing row={row_spacing_px} col={col_spacing_px}')
    size = int(round(base * float(crop_fraction)))
    size = max(256, size)
    if size % 2 == 1:
        size += 1
    return int(size)


def crop_square_with_bounds(img: np.ndarray, center_y_px: float, center_x_px: float, size_px: int) -> Tuple[np.ndarray, Dict[str, int]]:
    arr = np.asarray(img)
    half = int(size_px // 2)
    cy = int(round(center_y_px))
    cx = int(round(center_x_px))
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(arr.shape[0], y0 + int(size_px))
    x1 = min(arr.shape[1], x0 + int(size_px))
    y0 = max(0, y1 - int(size_px))
    x0 = max(0, x1 - int(size_px))
    crop = arr[y0:y1, x0:x1]
    return crop, {'y0': int(y0), 'x0': int(x0), 'y1': int(y1), 'x1': int(x1)}


def write_tiff(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(arr), bigtiff=True)
