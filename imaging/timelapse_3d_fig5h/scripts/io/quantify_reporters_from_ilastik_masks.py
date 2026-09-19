"""NOT runnable as shipped. This stage reads the raw microscopy images, which are not deposited —
they are available from the lead contact on request, as the paper's Data Availability Statement
states. Its outputs are committed under the lane's `derived/` directory, which is what every
figure script actually reads. See `data/DOWNLOAD.md`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import morphology


ROOT = Path(__file__).resolve().parents[2]
POSITION_MANIFEST_PATH = ROOT / "results/manifests/acquisition_position_manifest.tsv"
MASK_METRICS_PATH = ROOT / "results/ilastik/qc/full_dataset_v1_mask_metrics.tsv"
ACQUISITION_QC_SUMMARY_PATH = ROOT / "results/qc/acquisition_qc_summary.json"
ILLUMINATION_FIELD_PATH = ROOT / "results/qc/04_masked_illumination_fields.npz"
BACKGROUND_EXCLUSION_WINDOW_PATH = ROOT / "results/tables/04b_background_exclusion_windows.tsv"
FRAME_METRICS_OUTPUT_PATH = ROOT / "results/tables/05_reporter_metrics_by_frame.tsv"
THRESHOLD_OUTPUT_PATH = ROOT / "results/tables/05_reporter_thresholds.tsv"
GLOBAL_THRESHOLD_OUTPUT_PATH = ROOT / "results/tables/05_global_reporter_thresholds.tsv"
POSITION_SUMMARY_OUTPUT_PATH = ROOT / "results/tables/05_position_reporter_summary.tsv"
ONSET_OUTPUT_PATH = ROOT / "results/tables/05_preliminary_onset_summary.tsv"
PARAMETERS_OUTPUT_PATH = ROOT / "results/tables/05_reporter_quantification_parameters.json"
POSITION_RE = re.compile(r"Pos(?P<position_index>\d+)$")
REPORTER_CHANNELS = {1: "RFP", 2: "YFP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify FOXF1 (RFP) and BMP4 (YFP) reporter signals using ilastik-derived organoid masks."
    )
    parser.add_argument("--position-manifest-path", type=Path, default=POSITION_MANIFEST_PATH)
    parser.add_argument("--mask-metrics-path", type=Path, default=MASK_METRICS_PATH)
    parser.add_argument("--acquisition-qc-summary-path", type=Path, default=ACQUISITION_QC_SUMMARY_PATH)
    parser.add_argument("--illumination-field-path", type=Path, default=ILLUMINATION_FIELD_PATH)
    parser.add_argument("--background-exclusion-window-path", type=Path, default=BACKGROUND_EXCLUSION_WINDOW_PATH)
    parser.add_argument("--frame-metrics-output-path", type=Path, default=FRAME_METRICS_OUTPUT_PATH)
    parser.add_argument("--threshold-output-path", type=Path, default=THRESHOLD_OUTPUT_PATH)
    parser.add_argument("--global-threshold-output-path", type=Path, default=GLOBAL_THRESHOLD_OUTPUT_PATH)
    parser.add_argument("--position-summary-output-path", type=Path, default=POSITION_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--onset-output-path", type=Path, default=ONSET_OUTPUT_PATH)
    parser.add_argument("--parameters-output-path", type=Path, default=PARAMETERS_OUTPUT_PATH)
    parser.add_argument("--background-ring-inner", type=int, default=4)
    parser.add_argument("--background-ring-outer", type=int, default=12)
    parser.add_argument("--min-ring-pixels", type=int, default=100)
    parser.add_argument(
        "--background-estimator",
        type=str,
        choices=["annulus", "whole_off_cyst"],
        default="whole_off_cyst",
    )
    parser.add_argument(
        "--threshold-scope",
        type=str,
        choices=["global_pooled_early_organoid_pixels", "per_position_early_organoid_pixels"],
        default="global_pooled_early_organoid_pixels",
    )
    parser.add_argument("--baseline-frame-count", type=int, default=4)
    parser.add_argument("--lower-tail-fraction", type=float, default=1.00)
    parser.add_argument("--threshold-z", type=float, default=4.0)
    parser.add_argument("--baseline-peak-hist-bins", type=int, default=256)
    parser.add_argument("--baseline-peak-smooth-sigma-bins", type=float, default=2.0)
    parser.add_argument("--positive-closing-radius", type=int, default=1)
    parser.add_argument("--positive-min-object-size", type=int, default=16)
    parser.add_argument("--positive-hole-area", type=int, default=16)
    parser.add_argument("--summary-window-frames", type=int, default=6)
    parser.add_argument("--onset-min-positive-fraction", type=float, default=0.01)
    parser.add_argument("--onset-min-positive-pixels", type=int, default=50)
    parser.add_argument("--onset-persistence-frames", type=int, default=3)
    return parser.parse_args()


def position_index_from_label(position_label: str) -> int:
    match = POSITION_RE.fullmatch(position_label)
    if not match:
        raise ValueError(f"Unexpected position label: {position_label}")
    return int(match.group("position_index"))


def raw_frame_path(dataset_dir: Path, position_label: str, channel_index: int, time_index: int) -> Path:
    position_index = position_index_from_label(position_label)
    return (
        dataset_dir
        / position_label
        / f"img_channel{channel_index:03d}_position{position_index:03d}_time{time_index:09d}_z000.tif"
    )


def gaussian_kernel1d_bins(sigma_bins: float) -> np.ndarray:
    sigma_bins = max(float(sigma_bins), 0.0)
    if sigma_bins <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def smooth_histogram_counts(counts: np.ndarray, sigma_bins: float) -> np.ndarray:
    kernel = gaussian_kernel1d_bins(sigma_bins)
    if kernel.size == 1:
        return counts.astype(float)
    pad = kernel.size // 2
    padded = np.pad(counts.astype(float), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smoothed_histogram_peak(values: np.ndarray, n_bins: int, smooth_sigma_bins: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    value_min = float(np.min(arr))
    value_max = float(np.max(arr))
    if not np.isfinite(value_min) or not np.isfinite(value_max):
        return float("nan")
    if value_max <= value_min:
        return value_min
    bins = max(32, int(n_bins))
    edges = np.linspace(value_min, value_max, bins + 1, dtype=float)
    counts, _ = np.histogram(arr, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth_counts = smooth_histogram_counts(counts, smooth_sigma_bins)
    mode_idx = int(np.nanargmax(smooth_counts))
    return float(centers[mode_idx])


def half_gaussian_threshold(
    values: np.ndarray,
    z: float,
    peak_hist_bins: int,
    peak_smooth_sigma_bins: float,
) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    value_min = float(np.min(arr))
    value_max = float(np.max(arr))
    if not np.isfinite(value_min) or not np.isfinite(value_max):
        return float("nan"), float("nan"), float("nan"), float("nan")
    if value_max <= value_min:
        location = value_min
        scale = 1.0
        threshold = location + z * scale
        return float(threshold), float(location), float(scale), float(location)
    bins = max(32, int(peak_hist_bins))
    edges = np.linspace(value_min, value_max, bins + 1, dtype=float)
    counts, _ = np.histogram(arr, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth_counts = smooth_histogram_counts(counts, peak_smooth_sigma_bins)
    mode_idx = int(np.nanargmax(smooth_counts))
    location = float(centers[mode_idx])
    left = centers <= location
    left_counts = counts[left].astype(float)
    left_centers = centers[left]
    if np.sum(left_counts) <= 0:
        scale = 1.0
    else:
        scale = float(
            np.sqrt(np.sum(left_counts * (left_centers - location) ** 2) / np.sum(left_counts))
        )
        scale = max(scale, 1.0)
    threshold = location + z * scale
    return float(threshold), location, float(scale), float(location)


def annulus_mask(
    organoid_mask: np.ndarray,
    inner: int,
    outer: int,
    min_ring_pixels: int,
) -> np.ndarray:
    inner_mask = morphology.binary_dilation(organoid_mask, morphology.disk(inner))
    outer_mask = morphology.binary_dilation(organoid_mask, morphology.disk(outer))
    ring = outer_mask & ~inner_mask
    if int(ring.sum()) < min_ring_pixels:
        ring = ~outer_mask
    if int(ring.sum()) < min_ring_pixels:
        ring = ~organoid_mask
    return ring


def background_reference_mask(
    organoid_mask: np.ndarray,
    background_estimator: str,
    background_ring_inner: int,
    background_ring_outer: int,
    min_ring_pixels: int,
) -> np.ndarray:
    if background_estimator == "whole_off_cyst":
        return ~organoid_mask
    return annulus_mask(
        organoid_mask=organoid_mask,
        inner=background_ring_inner,
        outer=background_ring_outer,
        min_ring_pixels=min_ring_pixels,
    )


def corrected_signal(
    signal: np.ndarray,
    organoid_mask: np.ndarray,
    background_estimator: str,
    background_ring_inner: int,
    background_ring_outer: int,
    min_ring_pixels: int,
) -> tuple[np.ndarray, float]:
    background_mask = background_reference_mask(
        organoid_mask=organoid_mask,
        background_estimator=background_estimator,
        background_ring_inner=background_ring_inner,
        background_ring_outer=background_ring_outer,
        min_ring_pixels=min_ring_pixels,
    )
    if np.any(background_mask):
        background_value = float(np.median(signal[background_mask]))
    else:
        background_value = float(np.median(signal[~organoid_mask])) if np.any(~organoid_mask) else 0.0
    corrected = signal.astype(float) - background_value
    return corrected, background_value


def apply_illumination_field(signal: np.ndarray, field: np.ndarray | None) -> np.ndarray:
    if field is None:
        return signal.astype(np.float32)
    return (signal.astype(np.float32) / np.clip(field.astype(np.float32), 1e-6, None)).astype(np.float32)


def positive_mask_from_corrected(
    corrected: np.ndarray,
    organoid_mask: np.ndarray,
    threshold_value: float,
    closing_radius: int,
    min_object_size: int,
    hole_area: int,
) -> np.ndarray:
    positive = organoid_mask & (corrected > float(threshold_value))
    if closing_radius > 0:
        positive = morphology.binary_closing(positive, morphology.disk(closing_radius))
    if min_object_size > 1:
        positive = morphology.remove_small_objects(positive, min_size=min_object_size)
    if hole_area > 1:
        positive = morphology.remove_small_holes(positive, area_threshold=hole_area)
    return positive & organoid_mask


def first_persistent_onset(
    time_indices: list[int],
    on_flags: list[bool],
    persistence_frames: int,
) -> float:
    if persistence_frames <= 0:
        return float("nan")
    for start_idx in range(0, len(time_indices) - persistence_frames + 1):
        window_times = time_indices[start_idx : start_idx + persistence_frames]
        window_flags = on_flags[start_idx : start_idx + persistence_frames]
        if not all(window_flags):
            continue
        expected_times = list(range(window_times[0], window_times[0] + persistence_frames))
        if window_times == expected_times:
            return float(window_times[0])
    return float("nan")


def summarize_position_reporter(
    group: pd.DataFrame,
    summary_window_frames: int,
) -> dict[str, object]:
    group = group.sort_values("time_index").reset_index(drop=True)
    valid_whole = group.loc[group["whole_organoid_metrics_allowed"]].copy()
    valid_signal = group.loc[group["positive_signal_only_metrics_allowed"]].copy()

    early_whole = valid_whole.head(summary_window_frames)
    late_whole = valid_whole.tail(summary_window_frames)
    early_signal = valid_signal.head(summary_window_frames)
    late_signal = valid_signal.tail(summary_window_frames)

    def safe_median(frame: pd.DataFrame, column: str) -> float:
        if frame.empty:
            return float("nan")
        return float(np.nanmedian(frame[column].to_numpy(dtype=float)))

    def safe_max(frame: pd.DataFrame, column: str) -> float:
        if frame.empty:
            return float("nan")
        return float(np.nanmax(frame[column].to_numpy(dtype=float)))

    def safe_argmax_time(frame: pd.DataFrame, column: str) -> float:
        if frame.empty or frame[column].isna().all():
            return float("nan")
        idx = frame[column].astype(float).idxmax()
        return float(frame.loc[idx, "time_index"])

    summary = {
        "position_label": str(group["position_label"].iloc[0]),
        "reporter": str(group["reporter"].iloc[0]),
        "frame_count_nonexcluded": int(valid_signal.shape[0]),
        "whole_organoid_frame_count": int(valid_whole.shape[0]),
        "signal_only_frame_count": int(valid_signal.shape[0]),
        "early_positive_fraction_median": safe_median(early_whole, "positive_fraction"),
        "late_positive_fraction_median": safe_median(late_whole, "positive_fraction"),
        "positive_fraction_late_minus_early": (
            safe_median(late_whole, "positive_fraction") - safe_median(early_whole, "positive_fraction")
        ),
        "early_positive_mean_intensity_median": safe_median(early_signal, "positive_mean_intensity"),
        "late_positive_mean_intensity_median": safe_median(late_signal, "positive_mean_intensity"),
        "positive_mean_intensity_late_minus_early": (
            safe_median(late_signal, "positive_mean_intensity")
            - safe_median(early_signal, "positive_mean_intensity")
        ),
        "early_positive_integrated_intensity_median": safe_median(early_signal, "positive_integrated_intensity"),
        "late_positive_integrated_intensity_median": safe_median(late_signal, "positive_integrated_intensity"),
        "positive_integrated_intensity_late_minus_early": (
            safe_median(late_signal, "positive_integrated_intensity")
            - safe_median(early_signal, "positive_integrated_intensity")
        ),
        "max_positive_fraction": safe_max(valid_whole, "positive_fraction"),
        "time_of_max_positive_fraction": safe_argmax_time(valid_whole, "positive_fraction"),
        "max_positive_mean_intensity": safe_max(valid_signal, "positive_mean_intensity"),
        "time_of_max_positive_mean_intensity": safe_argmax_time(valid_signal, "positive_mean_intensity"),
        "max_positive_integrated_intensity": safe_max(valid_signal, "positive_integrated_intensity"),
        "time_of_max_positive_integrated_intensity": safe_argmax_time(valid_signal, "positive_integrated_intensity"),
    }
    return summary


def combine_exclusion_reason(
    phase_excluded: bool,
    phase_reason: str,
    background_excluded: bool,
    background_cut_start_time_index: float,
) -> str:
    reasons: list[str] = []
    phase_reason = str(phase_reason).strip()
    if phase_excluded and phase_reason:
        reasons.append(phase_reason)
    if background_excluded and np.isfinite(background_cut_start_time_index):
        reasons.append(f"reporter_background_qc_cut_from_t{int(background_cut_start_time_index)}")
    return "; ".join(dict.fromkeys(reasons))


def collect_early_corrected_pixels(
    *,
    dataset_dir: Path,
    position_label: str,
    subset: pd.DataFrame,
    channel_index: int,
    reporter: str,
    illumination_field: np.ndarray | None,
    background_estimator: str,
    background_ring_inner: int,
    background_ring_outer: int,
    min_ring_pixels: int,
    baseline_frame_count: int,
) -> tuple[np.ndarray, list[float], int]:
    baseline_subset = subset.loc[~subset["exclude_from_analysis"]].head(baseline_frame_count).copy()
    if baseline_subset.empty:
        return np.array([], dtype=float), [], 0

    pooled_values = []
    background_values: list[float] = []
    for _, row in baseline_subset.iterrows():
        mask = tiff.imread(Path(str(row["mask_path"]))).astype(bool)
        if not mask.any():
            continue
        signal_raw = tiff.imread(
            raw_frame_path(dataset_dir, position_label, channel_index, int(row["time_index"]))
        ).astype(float)
        signal = apply_illumination_field(signal_raw, illumination_field).astype(float)
        corrected, background_value = corrected_signal(
            signal=signal,
            organoid_mask=mask,
            background_estimator=background_estimator,
            background_ring_inner=background_ring_inner,
            background_ring_outer=background_ring_outer,
            min_ring_pixels=min_ring_pixels,
        )
        pooled_values.append(corrected[mask])
        background_values.append(background_value)

    pooled = np.concatenate(pooled_values) if pooled_values else np.array([], dtype=float)
    return pooled, background_values, int(baseline_subset.shape[0])


def main() -> None:
    args = parse_args()

    position_manifest = pd.read_csv(args.position_manifest_path, sep="\t")
    dataset_dir = ROOT / str(position_manifest["dataset_dir"].iloc[0])
    interval_minutes = float(position_manifest["interval_ms"].dropna().iloc[0]) / 60000.0

    mask_metrics = pd.read_csv(args.mask_metrics_path, sep="\t").sort_values(
        ["position_label", "time_index"]
    ).reset_index(drop=True)
    mask_metrics["phase_qc_exclude_from_analysis"] = mask_metrics["exclude_from_analysis"].astype(bool)
    mask_metrics["phase_qc_exclusion_reason"] = mask_metrics["exclusion_reason"].fillna("").astype(str)
    mask_metrics["phase_qc_whole_organoid_metrics_allowed"] = mask_metrics["whole_organoid_metrics_allowed"].astype(bool)
    mask_metrics["phase_qc_positive_signal_only_metrics_allowed"] = mask_metrics["positive_signal_only_metrics_allowed"].astype(bool)
    mask_metrics["phase_qc_analysis_metric_scope"] = mask_metrics["analysis_metric_scope"].fillna("").astype(str)

    if args.background_exclusion_window_path and args.background_exclusion_window_path.exists():
        background_exclusion_windows = pd.read_csv(args.background_exclusion_window_path, sep="\t")
    else:
        background_exclusion_windows = pd.DataFrame(
            columns=[
                "position_label",
                "background_qc_cut_start_time_index",
                "background_qc_cut_start_time_hours",
                "background_qc_cut_end_time_index",
                "background_qc_cut_end_time_hours",
                "background_qc_cut_to_end",
                "background_qc_flagged_reporters",
                "background_qc_primary_rules",
            ]
        )
    background_exclusion_merge_columns = [
        "position_label",
        "background_qc_cut_start_time_index",
        "background_qc_cut_start_time_hours",
        "background_qc_cut_end_time_index",
        "background_qc_cut_end_time_hours",
        "background_qc_cut_to_end",
        "background_qc_flagged_reporters",
        "background_qc_primary_rules",
    ]
    mask_metrics = mask_metrics.merge(
        background_exclusion_windows[background_exclusion_merge_columns],
        on="position_label",
        how="left",
    )
    mask_metrics["background_qc_exclude_from_analysis"] = (
        mask_metrics["background_qc_cut_start_time_index"].notna()
        & (
            mask_metrics["time_index"].astype(float)
            >= mask_metrics["background_qc_cut_start_time_index"].astype(float)
        )
    )
    mask_metrics["exclude_from_analysis"] = (
        mask_metrics["phase_qc_exclude_from_analysis"] | mask_metrics["background_qc_exclude_from_analysis"]
    )
    mask_metrics["exclusion_reason"] = [
        combine_exclusion_reason(
            phase_excluded=bool(phase_excluded),
            phase_reason=str(phase_reason),
            background_excluded=bool(background_excluded),
            background_cut_start_time_index=float(background_cut_start_time_index)
            if pd.notna(background_cut_start_time_index)
            else float("nan"),
        )
        for phase_excluded, phase_reason, background_excluded, background_cut_start_time_index in zip(
            mask_metrics["phase_qc_exclude_from_analysis"],
            mask_metrics["phase_qc_exclusion_reason"],
            mask_metrics["background_qc_exclude_from_analysis"],
            mask_metrics["background_qc_cut_start_time_index"],
        )
    ]
    mask_metrics["whole_organoid_metrics_allowed"] = (
        mask_metrics["phase_qc_whole_organoid_metrics_allowed"] & (~mask_metrics["background_qc_exclude_from_analysis"])
    )
    mask_metrics["positive_signal_only_metrics_allowed"] = (
        mask_metrics["phase_qc_positive_signal_only_metrics_allowed"] & (~mask_metrics["background_qc_exclude_from_analysis"])
    )
    mask_metrics["analysis_metric_scope"] = np.where(
        mask_metrics["background_qc_exclude_from_analysis"],
        np.where(
            mask_metrics["phase_qc_exclude_from_analysis"],
            "excluded_phase_qc_and_reporter_background_qc",
            "excluded_reporter_background_qc",
        ),
        mask_metrics["phase_qc_analysis_metric_scope"],
    )

    illumination_fields: dict[str, np.ndarray | None] = {"RFP": None, "YFP": None}
    illumination_metadata: dict[str, object] = {"applied": False, "path": None}
    if args.illumination_field_path and args.illumination_field_path.exists():
        payload = np.load(args.illumination_field_path)
        illumination_fields = {
            "RFP": payload["rfp_field"].astype(np.float32),
            "YFP": payload["yfp_field"].astype(np.float32),
        }
        illumination_metadata = {
            "applied": True,
            "path": str(args.illumination_field_path),
            "rfp_field_range": [
                float(np.nanmin(illumination_fields["RFP"])),
                float(np.nanmax(illumination_fields["RFP"])),
            ],
            "yfp_field_range": [
                float(np.nanmin(illumination_fields["YFP"])),
                float(np.nanmax(illumination_fields["YFP"])),
            ],
        }

    for path in [
        args.frame_metrics_output_path,
        args.threshold_output_path,
        args.global_threshold_output_path,
        args.position_summary_output_path,
        args.onset_output_path,
        args.parameters_output_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    threshold_rows: list[dict[str, object]] = []
    global_threshold_rows: list[dict[str, object]] = []
    thresholds: dict[tuple[str, str], dict[str, float]] = {}
    reporter_global_thresholds: dict[str, dict[str, float]] = {}
    early_pool_by_position_reporter: dict[tuple[str, str], dict[str, object]] = {}

    for position_label, subset in mask_metrics.groupby("position_label", sort=True):
        subset = subset.sort_values("time_index").reset_index(drop=True)
        for channel_index, reporter in REPORTER_CHANNELS.items():
            pooled, background_values, baseline_frame_count_used = collect_early_corrected_pixels(
                dataset_dir=dataset_dir,
                position_label=position_label,
                subset=subset,
                channel_index=channel_index,
                reporter=reporter,
                illumination_field=illumination_fields[reporter],
                background_estimator=args.background_estimator,
                background_ring_inner=args.background_ring_inner,
                background_ring_outer=args.background_ring_outer,
                min_ring_pixels=args.min_ring_pixels,
                baseline_frame_count=args.baseline_frame_count,
            )
            early_pool_by_position_reporter[(position_label, reporter)] = {
                "pooled": pooled,
                "background_values": background_values,
                "baseline_frame_count": baseline_frame_count_used,
            }

    for reporter in REPORTER_CHANNELS.values():
        pooled_arrays = []
        pooled_background_values = []
        pooled_position_count = 0
        pooled_frame_count = 0
        for (position_label, reporter_key), payload in early_pool_by_position_reporter.items():
            if reporter_key != reporter:
                continue
            pooled = np.asarray(payload["pooled"], dtype=float)
            if pooled.size == 0:
                continue
            pooled_arrays.append(pooled)
            pooled_background_values.extend(payload["background_values"])
            pooled_position_count += 1
            pooled_frame_count += int(payload["baseline_frame_count"])

        pooled_all = np.concatenate(pooled_arrays) if pooled_arrays else np.array([], dtype=float)
        threshold_value, baseline_location, baseline_scale, global_cutoff = half_gaussian_threshold(
            values=pooled_all,
            z=args.threshold_z,
            peak_hist_bins=args.baseline_peak_hist_bins,
            peak_smooth_sigma_bins=args.baseline_peak_smooth_sigma_bins,
        )
        reporter_global_thresholds[reporter] = {
            "threshold_value": float(threshold_value),
            "baseline_location": float(baseline_location),
            "baseline_scale": float(baseline_scale),
            "global_cutoff": float(global_cutoff),
            "null_partition_value": float(global_cutoff),
            "threshold_scope": str(args.threshold_scope),
            "baseline_location_method": "smoothed_histogram_peak_of_full_pooled_histogram",
            "baseline_scale_method": "left_half_weighted_second_moment",
            "baseline_frame_count_per_position": int(args.baseline_frame_count),
            "pooled_position_count": int(pooled_position_count),
            "pooled_frame_count": int(pooled_frame_count),
            "pooled_pixel_count": int(pooled_all.size),
            "pooled_background_median": float(np.median(pooled_background_values)) if pooled_background_values else float("nan"),
        }
        global_threshold_rows.append(
            {
                "reporter": reporter,
                "threshold_scope": str(args.threshold_scope),
                "threshold_value": float(threshold_value),
                "baseline_location": float(baseline_location),
                "baseline_scale": float(baseline_scale),
                "global_cutoff": float(global_cutoff),
                "null_partition_value": float(global_cutoff),
                "baseline_location_method": "smoothed_histogram_peak_of_full_pooled_histogram",
                "baseline_scale_method": "left_half_weighted_second_moment",
                "baseline_frame_count_per_position": int(args.baseline_frame_count),
                "pooled_position_count": int(pooled_position_count),
                "pooled_frame_count": int(pooled_frame_count),
                "pooled_pixel_count": int(pooled_all.size),
                "pooled_background_median": float(np.median(pooled_background_values)) if pooled_background_values else float("nan"),
                "threshold_z": float(args.threshold_z),
                "lower_tail_fraction": float(args.lower_tail_fraction),
                "baseline_peak_hist_bins": int(args.baseline_peak_hist_bins),
                "baseline_peak_smooth_sigma_bins": float(args.baseline_peak_smooth_sigma_bins),
            }
        )

    for (position_label, reporter), payload in early_pool_by_position_reporter.items():
        pooled = np.asarray(payload["pooled"], dtype=float)
        background_values = payload["background_values"]
        global_info = reporter_global_thresholds[reporter]
        thresholds[(position_label, reporter)] = {
            "threshold_value": float(global_info["threshold_value"]),
            "baseline_location": float(global_info["baseline_location"]),
            "baseline_scale": float(global_info["baseline_scale"]),
            "global_cutoff": float(global_info["global_cutoff"]),
            "null_partition_value": float(global_info["null_partition_value"]),
            "threshold_scope": str(global_info["threshold_scope"]),
            "baseline_location_method": str(global_info["baseline_location_method"]),
            "baseline_scale_method": str(global_info["baseline_scale_method"]),
            "baseline_frame_count": int(payload["baseline_frame_count"]),
            "baseline_pixel_count": int(pooled.size),
            "baseline_background_median": float(np.median(background_values)) if background_values else float("nan"),
            "pooled_position_count": int(global_info["pooled_position_count"]),
            "pooled_frame_count": int(global_info["pooled_frame_count"]),
            "pooled_pixel_count": int(global_info["pooled_pixel_count"]),
            "pooled_background_median": float(global_info["pooled_background_median"]),
        }
        threshold_rows.append(
            {
                "position_label": position_label,
                "reporter": reporter,
                "threshold_scope": str(global_info["threshold_scope"]),
                "threshold_value": float(global_info["threshold_value"]),
                "baseline_location": float(global_info["baseline_location"]),
                "baseline_scale": float(global_info["baseline_scale"]),
                "global_cutoff": float(global_info["global_cutoff"]),
                "null_partition_value": float(global_info["null_partition_value"]),
                "baseline_location_method": str(global_info["baseline_location_method"]),
                "baseline_scale_method": str(global_info["baseline_scale_method"]),
                "baseline_frame_count": int(payload["baseline_frame_count"]),
                "baseline_pixel_count": int(pooled.size),
                "baseline_background_median": float(np.median(background_values)) if background_values else float("nan"),
                "pooled_position_count": int(global_info["pooled_position_count"]),
                "pooled_frame_count": int(global_info["pooled_frame_count"]),
                "pooled_pixel_count": int(global_info["pooled_pixel_count"]),
                "pooled_background_median": float(global_info["pooled_background_median"]),
                "threshold_z": float(args.threshold_z),
                "lower_tail_fraction": float(args.lower_tail_fraction),
                "baseline_peak_hist_bins": int(args.baseline_peak_hist_bins),
                "baseline_peak_smooth_sigma_bins": float(args.baseline_peak_smooth_sigma_bins),
            }
        )

    threshold_df = pd.DataFrame(threshold_rows).sort_values(["position_label", "reporter"]).reset_index(drop=True)
    threshold_df.to_csv(args.threshold_output_path, sep="\t", index=False)
    global_threshold_df = pd.DataFrame(global_threshold_rows).sort_values(["reporter"]).reset_index(drop=True)
    global_threshold_df.to_csv(args.global_threshold_output_path, sep="\t", index=False)

    frame_rows: list[dict[str, object]] = []
    for position_label, subset in mask_metrics.groupby("position_label", sort=True):
        subset = subset.sort_values("time_index").reset_index(drop=True)
        for _, row in subset.iterrows():
            time_index = int(row["time_index"])
            time_hours = time_index * interval_minutes / 60.0
            mask = tiff.imread(Path(str(row["mask_path"]))).astype(bool)
            raw_area_px = int(mask.sum())

            for channel_index, reporter in REPORTER_CHANNELS.items():
                threshold_info = thresholds[(position_label, reporter)]
                record: dict[str, object] = {
                    "position_label": position_label,
                    "position_index": position_index_from_label(position_label),
                    "time_index": time_index,
                    "time_hours": float(time_hours),
                    "reporter": reporter,
                    "exclude_from_analysis": bool(row["exclude_from_analysis"]),
                    "exclusion_reason": str(row["exclusion_reason"]),
                    "phase_qc_exclude_from_analysis": bool(row["phase_qc_exclude_from_analysis"]),
                    "phase_qc_exclusion_reason": str(row["phase_qc_exclusion_reason"]),
                    "background_qc_exclude_from_analysis": bool(row["background_qc_exclude_from_analysis"]),
                    "background_qc_cut_start_time_index": (
                        float(row["background_qc_cut_start_time_index"])
                        if pd.notna(row["background_qc_cut_start_time_index"])
                        else float("nan")
                    ),
                    "background_qc_cut_start_time_hours": (
                        float(row["background_qc_cut_start_time_hours"])
                        if pd.notna(row["background_qc_cut_start_time_hours"])
                        else float("nan")
                    ),
                    "background_qc_cut_end_time_index": (
                        float(row["background_qc_cut_end_time_index"])
                        if pd.notna(row["background_qc_cut_end_time_index"])
                        else float("nan")
                    ),
                    "background_qc_cut_end_time_hours": (
                        float(row["background_qc_cut_end_time_hours"])
                        if pd.notna(row["background_qc_cut_end_time_hours"])
                        else float("nan")
                    ),
                    "background_qc_cut_to_end": bool(row["background_qc_cut_to_end"]) if pd.notna(row["background_qc_cut_to_end"]) else False,
                    "background_qc_flagged_reporters": str(row["background_qc_flagged_reporters"]) if pd.notna(row["background_qc_flagged_reporters"]) else "",
                    "background_qc_primary_rules": str(row["background_qc_primary_rules"]) if pd.notna(row["background_qc_primary_rules"]) else "",
                    "whole_organoid_metrics_allowed": bool(row["whole_organoid_metrics_allowed"]),
                    "positive_signal_only_metrics_allowed": bool(row["positive_signal_only_metrics_allowed"]),
                    "analysis_metric_scope": str(row["analysis_metric_scope"]),
                    "touches_border": bool(row["touches_border"]),
                    "mask_path": str(row["mask_path"]),
                    "illumination_correction_applied": bool(illumination_metadata["applied"]),
                    "background_estimator": str(args.background_estimator),
                    "threshold_value": threshold_info["threshold_value"],
                    "baseline_location": threshold_info["baseline_location"],
                    "baseline_scale": threshold_info["baseline_scale"],
                    "global_cutoff": threshold_info["global_cutoff"],
                    "null_partition_value": threshold_info["null_partition_value"],
                    "threshold_scope": threshold_info["threshold_scope"],
                    "baseline_location_method": threshold_info["baseline_location_method"],
                    "baseline_scale_method": threshold_info["baseline_scale_method"],
                    "baseline_frame_count": threshold_info["baseline_frame_count"],
                    "baseline_pixel_count": threshold_info["baseline_pixel_count"],
                    "organoid_area_px_raw": raw_area_px,
                    "organoid_area_px": float(raw_area_px) if bool(row["whole_organoid_metrics_allowed"]) else float("nan"),
                }

                if bool(row["exclude_from_analysis"]) or raw_area_px == 0:
                    record.update(
                        {
                            "illumination_field_median_in_mask": float("nan"),
                            "background_value": float("nan"),
                            "organoid_mean_intensity_raw": float("nan"),
                            "organoid_mean_intensity": float("nan"),
                            "positive_pixel_count": float("nan"),
                            "positive_fraction_raw": float("nan"),
                            "positive_fraction": float("nan"),
                            "positive_mean_intensity": float("nan"),
                            "positive_integrated_intensity": float("nan"),
                        }
                    )
                    frame_rows.append(record)
                    continue

                signal_raw = tiff.imread(raw_frame_path(dataset_dir, position_label, channel_index, time_index)).astype(float)
                field = illumination_fields[reporter]
                signal = apply_illumination_field(signal_raw, field).astype(float)
                corrected, background_value = corrected_signal(
                    signal=signal,
                    organoid_mask=mask,
                    background_estimator=args.background_estimator,
                    background_ring_inner=args.background_ring_inner,
                    background_ring_outer=args.background_ring_outer,
                    min_ring_pixels=args.min_ring_pixels,
                )
                organoid_values = corrected[mask]
                positive_mask = positive_mask_from_corrected(
                    corrected=corrected,
                    organoid_mask=mask,
                    threshold_value=float(threshold_info["threshold_value"]),
                    closing_radius=args.positive_closing_radius,
                    min_object_size=args.positive_min_object_size,
                    hole_area=args.positive_hole_area,
                )
                positive_values = corrected[positive_mask]
                positive_pixel_count = int(positive_mask.sum())
                positive_fraction_raw = float(positive_pixel_count / raw_area_px) if raw_area_px > 0 else float("nan")
                org_mean_raw = float(np.mean(organoid_values)) if organoid_values.size else float("nan")
                field_median_in_mask = (
                    float(np.median(field[mask])) if field is not None and mask.any() else float("nan")
                )

                record.update(
                    {
                        "illumination_field_median_in_mask": field_median_in_mask,
                        "background_value": float(background_value),
                        "organoid_mean_intensity_raw": org_mean_raw,
                        "organoid_mean_intensity": (
                            org_mean_raw if bool(row["whole_organoid_metrics_allowed"]) else float("nan")
                        ),
                        "positive_pixel_count": positive_pixel_count,
                        "positive_fraction_raw": positive_fraction_raw,
                        "positive_fraction": (
                            positive_fraction_raw if bool(row["whole_organoid_metrics_allowed"]) else float("nan")
                        ),
                        "positive_mean_intensity": float(np.mean(positive_values)) if positive_values.size else float("nan"),
                        "positive_integrated_intensity": float(np.sum(positive_values)) if positive_values.size else 0.0,
                    }
                )
                frame_rows.append(record)

    frame_metrics_df = pd.DataFrame(frame_rows).sort_values(
        ["position_index", "reporter", "time_index"]
    ).reset_index(drop=True)
    frame_metrics_df.to_csv(args.frame_metrics_output_path, sep="\t", index=False)

    position_summary_rows = [
        summarize_position_reporter(group, summary_window_frames=args.summary_window_frames)
        for _, group in frame_metrics_df.groupby(["position_label", "reporter"], sort=True)
    ]
    position_summary_df = pd.DataFrame(position_summary_rows).sort_values(
        ["position_label", "reporter"]
    ).reset_index(drop=True)
    position_summary_df.to_csv(args.position_summary_output_path, sep="\t", index=False)

    onset_rows: list[dict[str, object]] = []
    for (position_label, reporter), group in frame_metrics_df.groupby(["position_label", "reporter"], sort=True):
        valid = group.loc[group["whole_organoid_metrics_allowed"] & (~group["exclude_from_analysis"])].copy()
        valid = valid.sort_values("time_index").reset_index(drop=True)
        on_flags = (
            (valid["positive_fraction"].fillna(0).to_numpy(dtype=float) >= args.onset_min_positive_fraction)
            & (valid["positive_pixel_count"].fillna(0).to_numpy(dtype=float) >= args.onset_min_positive_pixels)
        )
        onset_time_index = first_persistent_onset(
            time_indices=valid["time_index"].astype(int).tolist(),
            on_flags=on_flags.tolist(),
            persistence_frames=args.onset_persistence_frames,
        )
        onset_rows.append(
            {
                "position_label": position_label,
                "reporter": reporter,
                "onset_time_index": onset_time_index,
                "onset_time_hours": (
                    float(onset_time_index) * interval_minutes / 60.0 if np.isfinite(onset_time_index) else float("nan")
                ),
                "onset_rule_positive_fraction": float(args.onset_min_positive_fraction),
                "onset_rule_positive_pixels": int(args.onset_min_positive_pixels),
                "onset_rule_persistence_frames": int(args.onset_persistence_frames),
            }
        )

    onset_long_df = pd.DataFrame(onset_rows).sort_values(["position_label", "reporter"]).reset_index(drop=True)
    onset_wide_df = (
        onset_long_df.pivot(index="position_label", columns="reporter", values="onset_time_index")
        .rename(columns={"RFP": "rfp_onset_time_index", "YFP": "yfp_onset_time_index"})
        .reset_index()
    )
    onset_hours_wide = (
        onset_long_df.pivot(index="position_label", columns="reporter", values="onset_time_hours")
        .rename(columns={"RFP": "rfp_onset_time_hours", "YFP": "yfp_onset_time_hours"})
        .reset_index()
    )
    onset_summary_df = onset_wide_df.merge(onset_hours_wide, on="position_label", how="outer")
    onset_summary_df["yfp_minus_rfp_onset_frames"] = (
        onset_summary_df["yfp_onset_time_index"] - onset_summary_df["rfp_onset_time_index"]
    )
    onset_summary_df["yfp_minus_rfp_onset_hours"] = (
        onset_summary_df["yfp_onset_time_hours"] - onset_summary_df["rfp_onset_time_hours"]
    )
    onset_summary_df.to_csv(args.onset_output_path, sep="\t", index=False)

    parameters = {
        "dataset_dir": str(dataset_dir),
        "mask_metrics_path": str(args.mask_metrics_path),
        "illumination_field_path": str(args.illumination_field_path) if args.illumination_field_path else None,
        "background_exclusion_window_path": (
            str(args.background_exclusion_window_path) if args.background_exclusion_window_path else None
        ),
        "illumination_correction_applied": bool(illumination_metadata["applied"]),
        "illumination_metadata": illumination_metadata,
        "frame_metrics_output_path": str(args.frame_metrics_output_path),
        "threshold_output_path": str(args.threshold_output_path),
        "global_threshold_output_path": str(args.global_threshold_output_path),
        "position_summary_output_path": str(args.position_summary_output_path),
        "onset_output_path": str(args.onset_output_path),
        "interval_minutes": interval_minutes,
        "background_ring_inner": int(args.background_ring_inner),
        "background_ring_outer": int(args.background_ring_outer),
        "min_ring_pixels": int(args.min_ring_pixels),
        "background_estimator": str(args.background_estimator),
        "threshold_scope": str(args.threshold_scope),
        "baseline_frame_count": int(args.baseline_frame_count),
        "lower_tail_fraction": float(args.lower_tail_fraction),
        "threshold_z": float(args.threshold_z),
        "baseline_peak_hist_bins": int(args.baseline_peak_hist_bins),
        "baseline_peak_smooth_sigma_bins": float(args.baseline_peak_smooth_sigma_bins),
        "positive_closing_radius": int(args.positive_closing_radius),
        "positive_min_object_size": int(args.positive_min_object_size),
        "positive_hole_area": int(args.positive_hole_area),
        "summary_window_frames": int(args.summary_window_frames),
        "onset_min_positive_fraction": float(args.onset_min_positive_fraction),
        "onset_min_positive_pixels": int(args.onset_min_positive_pixels),
        "onset_persistence_frames": int(args.onset_persistence_frames),
        "notes": [
            "Reporter frames are first corrected by the masked illumination field, then background-corrected using the configured off-cyst background estimator.",
            "Reporter positivity thresholds are estimated separately for RFP and YFP from pooled early clean organoid pixels across positions.",
            "For each reporter, the first baseline_frame_count clean frames from each position are pooled after background subtraction to define a shared early organoid-pixel distribution.",
            "Baseline location is estimated as the smoothed histogram peak of that full pooled early distribution.",
            "Baseline sigma is estimated from the left half of the pooled histogram around that peak using the weighted second moment, following the half-Gaussian logic used in the pSMAD workflow.",
            "Reporter-positive masks are thresholded first, then cleaned with binary closing, small-object removal, and small-hole filling.",
            "Frames excluded by the phase-artifact QC are retained in the output table but all reporter metrics are set to NaN.",
            "Frames at or after the background-QC cut start from notebook 04b are also retained in the output table but excluded from downstream reporter analysis.",
            "Whole-organoid denominator metrics are set to NaN when the organoid persistently touches the image border.",
            "Positive-signal metrics remain available on border-touch frames unless the frame is artifact-excluded.",
            "This is a preliminary quantification pass intended to be rerunnable after mask updates.",
        ],
    }
    args.parameters_output_path.write_text(json.dumps(parameters, indent=2) + "\n")

    print(f"Wrote frame metrics: {args.frame_metrics_output_path}")
    print(f"Wrote thresholds: {args.threshold_output_path}")
    print(f"Wrote global thresholds: {args.global_threshold_output_path}")
    print(f"Wrote position summary: {args.position_summary_output_path}")
    print(f"Wrote onset summary: {args.onset_output_path}")
    print(f"Wrote parameters: {args.parameters_output_path}")


if __name__ == "__main__":
    main()
