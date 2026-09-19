"""NOT runnable as shipped. This stage reads Ilastik probability maps, produced by Ilastik — a
separate desktop application, not a Python package, and not something this environment can
install. Its outputs are committed under the lane's `derived/` directory.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage as ndi
from skimage import measure, morphology


ROOT = Path(__file__).resolve().parents[2]
PROBABILITY_ROOT = ROOT / "results/ilastik/pixel_probabilities/full_dataset_v1"
MASK_ROOT = ROOT / "results/ilastik/organoid_masks/full_dataset_v1"
QC_ROOT = ROOT / "results/ilastik/qc"
METRICS_PATH = QC_ROOT / "full_dataset_v1_mask_metrics.tsv"
POSITION_SUMMARY_PATH = QC_ROOT / "full_dataset_v1_mask_position_summary.tsv"
PARAMETERS_PATH = QC_ROOT / "full_dataset_v1_mask_parameters.json"
BORDER_POLICY_SUMMARY_PATH = QC_ROOT / "full_dataset_v1_analysis_scope_summary.tsv"
EXCLUSION_PATH = ROOT / "results/qc/02_phase_artifact_excluded_frames.tsv"
TRAINING_MANIFEST_PATH = ROOT / "results/ilastik/training_frames/phase_curated_v1_manifest.tsv"
TIME_RE = re.compile(r"time(?P<time_index>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ilastik organoid probability TIFFs into one cleaned binary mask per frame."
    )
    parser.add_argument("--probability-root", type=Path, default=PROBABILITY_ROOT)
    parser.add_argument("--mask-root", type=Path, default=MASK_ROOT)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--position-summary-path", type=Path, default=POSITION_SUMMARY_PATH)
    parser.add_argument("--parameters-path", type=Path, default=PARAMETERS_PATH)
    parser.add_argument("--border-policy-summary-path", type=Path, default=BORDER_POLICY_SUMMARY_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=128,
        help="Ignore tiny disconnected components when selecting the organoid component.",
    )
    parser.add_argument(
        "--exclude-lumen",
        action="store_true",
        help="Keep internal holes in the mask instead of filling them.",
    )
    return parser.parse_args()


def load_exclusion_flags(path: Path) -> dict[tuple[str, int], dict[str, str | bool]]:
    df = pd.read_csv(path, sep="\t")
    out: dict[tuple[str, int], dict[str, str | bool]] = {}
    for row in df.to_dict("records"):
        out[(row["position_label"], int(row["time_index"]))] = {
            "exclude_from_analysis": bool(row["exclude_from_analysis"]),
            "exclusion_reason": str(row["exclusion_reason"] or ""),
            "border_dark_area_frac": float(row["border_dark_area_frac"]),
            "dark_area_frac": float(row["dark_area_frac"]),
        }
    return out


def load_training_frames(path: Path) -> set[tuple[str, int]]:
    df = pd.read_csv(path, sep="\t")
    return {
        (str(row["position_label"]), int(row["time_index"]))
        for row in df.to_dict("records")
    }


def time_index_from_name(path: Path) -> int:
    match = TIME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse time index from {path}")
    return int(match.group("time_index"))


def touches_border(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    eroded = morphology.binary_erosion(mask, morphology.disk(1))
    return np.logical_xor(mask, eroded)


def temporal_comparison_metrics(
    current_mask: np.ndarray,
    previous_mask: np.ndarray,
) -> dict[str, float]:
    if not current_mask.any() or not previous_mask.any():
        return {
            "iou_prev": np.nan,
            "area_change_frac_prev": np.nan,
            "centroid_shift_px_prev": np.nan,
            "missing_prev_frac": np.nan,
            "gained_prev_frac": np.nan,
            "p95_boundary_disp_prev": np.nan,
            "p99_boundary_disp_prev": np.nan,
            "max_boundary_disp_prev": np.nan,
        }

    intersection = np.logical_and(current_mask, previous_mask).sum()
    union = np.logical_or(current_mask, previous_mask).sum()
    iou_prev = float(intersection / union) if union > 0 else np.nan

    previous_area = float(previous_mask.sum())
    current_area = float(current_mask.sum())
    area_change_frac = abs(current_area - previous_area) / previous_area if previous_area > 0 else np.nan

    current_region = measure.regionprops(measure.label(current_mask))[0]
    previous_region = measure.regionprops(measure.label(previous_mask))[0]
    centroid_shift_px = float(
        np.hypot(
            current_region.centroid[0] - previous_region.centroid[0],
            current_region.centroid[1] - previous_region.centroid[1],
        )
    )

    missing_prev = np.logical_and(previous_mask, ~current_mask)
    gained_prev = np.logical_and(current_mask, ~previous_mask)
    missing_prev_frac = float(missing_prev.sum() / previous_area) if previous_area > 0 else np.nan
    gained_prev_frac = float(gained_prev.sum() / previous_area) if previous_area > 0 else np.nan

    current_boundary = boundary_mask(current_mask)
    previous_boundary = boundary_mask(previous_mask)
    dt_prev = ndi.distance_transform_edt(~previous_boundary)
    dt_current = ndi.distance_transform_edt(~current_boundary)
    current_to_prev = dt_prev[current_boundary]
    previous_to_current = dt_current[previous_boundary]
    boundary_distances = (
        np.concatenate([current_to_prev, previous_to_current])
        if current_to_prev.size and previous_to_current.size
        else np.array([], dtype=float)
    )

    return {
        "iou_prev": iou_prev,
        "area_change_frac_prev": area_change_frac,
        "centroid_shift_px_prev": centroid_shift_px,
        "missing_prev_frac": missing_prev_frac,
        "gained_prev_frac": gained_prev_frac,
        "p95_boundary_disp_prev": (
            float(np.quantile(boundary_distances, 0.95)) if boundary_distances.size else np.nan
        ),
        "p99_boundary_disp_prev": (
            float(np.quantile(boundary_distances, 0.99)) if boundary_distances.size else np.nan
        ),
        "max_boundary_disp_prev": float(boundary_distances.max()) if boundary_distances.size else np.nan,
    }


def select_organoid_component(
    mask: np.ndarray,
    min_component_area: int,
) -> tuple[np.ndarray, int, int, int]:
    labeled = measure.label(mask)
    regions = sorted(measure.regionprops(labeled), key=lambda region: region.area, reverse=True)
    component_count = len(regions)
    if component_count == 0:
        return np.zeros_like(mask, dtype=bool), 0, 0, 0

    filtered = [region for region in regions if region.area >= min_component_area]
    selected_region = filtered[0] if filtered else regions[0]
    second_area = int(regions[1].area) if component_count > 1 else 0
    selected_mask = labeled == selected_region.label
    return selected_mask.astype(bool), component_count, int(selected_region.area), second_area


def build_mask(
    organoid_probability: np.ndarray,
    threshold: float,
    closing_radius: int,
    min_component_area: int,
    fill_lumen: bool,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    probability = organoid_probability.astype(np.float32)
    raw_mask = probability >= threshold
    cleaned = raw_mask.copy()

    if closing_radius > 0:
        cleaned = morphology.binary_closing(cleaned, morphology.disk(closing_radius))

    selected_mask, component_count, largest_component_area_px, second_component_area_px = (
        select_organoid_component(cleaned, min_component_area=min_component_area)
    )

    if fill_lumen and selected_mask.any():
        selected_mask = ndi.binary_fill_holes(selected_mask)

    if closing_radius > 0 and selected_mask.any():
        selected_mask = morphology.binary_closing(selected_mask, morphology.disk(closing_radius))
        if fill_lumen:
            selected_mask = ndi.binary_fill_holes(selected_mask)

    region_props = None
    if selected_mask.any():
        labeled = measure.label(selected_mask)
        region_props = measure.regionprops(labeled)[0]

    boundary = boundary_mask(selected_mask) if selected_mask.any() else np.zeros_like(selected_mask, dtype=bool)
    raw_area_px = int(raw_mask.sum())
    final_area_px = int(selected_mask.sum())

    metrics = {
        "raw_mask_area_px": raw_area_px,
        "raw_mask_area_frac": float(raw_area_px / raw_mask.size),
        "component_count_raw": component_count,
        "largest_component_area_px": largest_component_area_px,
        "second_component_area_px": second_component_area_px,
        "final_mask_area_px": final_area_px,
        "final_mask_area_frac": float(final_area_px / selected_mask.size),
        "touches_border": touches_border(selected_mask),
        "mean_prob_inside_mask": float(probability[selected_mask].mean()) if selected_mask.any() else np.nan,
        "median_prob_inside_mask": float(np.median(probability[selected_mask])) if selected_mask.any() else np.nan,
        "p10_prob_inside_mask": float(np.quantile(probability[selected_mask], 0.10)) if selected_mask.any() else np.nan,
        "mean_prob_outside_mask": float(probability[~selected_mask].mean()) if (~selected_mask).any() else np.nan,
        "boundary_prob_mean": float(probability[boundary].mean()) if boundary.any() else np.nan,
        "boundary_prob_p10": float(np.quantile(probability[boundary], 0.10)) if boundary.any() else np.nan,
        "selected_component_fraction_of_raw": (
            float(final_area_px / raw_area_px) if raw_area_px > 0 else np.nan
        ),
        "bbox_min_row": int(region_props.bbox[0]) if region_props is not None else -1,
        "bbox_min_col": int(region_props.bbox[1]) if region_props is not None else -1,
        "bbox_max_row": int(region_props.bbox[2]) if region_props is not None else -1,
        "bbox_max_col": int(region_props.bbox[3]) if region_props is not None else -1,
        "centroid_row": float(region_props.centroid[0]) if region_props is not None else np.nan,
        "centroid_col": float(region_props.centroid[1]) if region_props is not None else np.nan,
        "eccentricity": float(region_props.eccentricity) if region_props is not None else np.nan,
        "solidity": float(region_props.solidity) if region_props is not None else np.nan,
        "extent": float(region_props.extent) if region_props is not None else np.nan,
        "major_axis_length": float(region_props.major_axis_length) if region_props is not None else np.nan,
        "minor_axis_length": float(region_props.minor_axis_length) if region_props is not None else np.nan,
    }

    return selected_mask.astype(bool), metrics


def write_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(out_path, mask.astype(np.uint8), photometric="minisblack")


def persistent_border_touch_summary(metrics_df: pd.DataFrame) -> tuple[dict[str, int | None], dict[str, int]]:
    included = metrics_df.loc[~metrics_df["exclude_from_analysis"]].copy()
    onset_map: dict[str, int | None] = {}
    count_map: dict[str, int] = {}

    for position_label, subset in included.groupby("position_label"):
        subset = subset.sort_values("time_index").reset_index(drop=True)
        touch_flags = subset["touches_border"].astype(bool).tolist()
        time_indices = subset["time_index"].astype(int).tolist()
        persistent_onset = None
        persistent_count = 0

        for idx, flag in enumerate(touch_flags):
            if flag and all(touch_flags[idx:]):
                persistent_onset = time_indices[idx]
                persistent_count = len(touch_flags) - idx
                break

        onset_map[position_label] = persistent_onset
        count_map[position_label] = persistent_count

    return onset_map, count_map


def main() -> None:
    args = parse_args()
    probability_root: Path = args.probability_root
    mask_root: Path = args.mask_root
    metrics_path: Path = args.metrics_path
    position_summary_path: Path = args.position_summary_path
    parameters_path: Path = args.parameters_path
    border_policy_summary_path: Path = args.border_policy_summary_path
    fill_lumen = not args.exclude_lumen

    if not probability_root.exists():
        raise FileNotFoundError(f"Missing probability root: {probability_root}")

    mask_root.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    exclusion_flags = load_exclusion_flags(EXCLUSION_PATH)
    training_frames = load_training_frames(TRAINING_MANIFEST_PATH)

    rows: list[dict[str, object]] = []

    for pos_dir in sorted(probability_root.glob("Pos*")):
        probability_paths = sorted(pos_dir.glob("*.tif*"))
        previous_mask = None
        previous_time_index = None

        for probability_path in probability_paths:
            probability_image = tiff.imread(probability_path)
            organoid_probability = probability_image[..., 0]
            background_probability = probability_image[..., 1] if probability_image.shape[-1] > 1 else np.nan
            time_index = time_index_from_name(probability_path)
            frame_key = (pos_dir.name, time_index)
            exclusion_info = exclusion_flags.get(
                frame_key,
                {
                    "exclude_from_analysis": False,
                    "exclusion_reason": "",
                    "border_dark_area_frac": np.nan,
                    "dark_area_frac": np.nan,
                },
            )

            mask, metrics = build_mask(
                organoid_probability=organoid_probability,
                threshold=args.threshold,
                closing_radius=args.closing_radius,
                min_component_area=args.min_component_area,
                fill_lumen=fill_lumen,
            )

            if previous_mask is None:
                temporal_metrics = {
                    "iou_prev": np.nan,
                    "area_change_frac_prev": np.nan,
                    "centroid_shift_px_prev": np.nan,
                    "missing_prev_frac": np.nan,
                    "gained_prev_frac": np.nan,
                    "p95_boundary_disp_prev": np.nan,
                    "p99_boundary_disp_prev": np.nan,
                    "max_boundary_disp_prev": np.nan,
                }
            else:
                temporal_metrics = temporal_comparison_metrics(mask, previous_mask)

            out_name = probability_path.name.replace("_Probabilities", "_mask")
            out_path = mask_root / pos_dir.name / out_name
            write_mask(mask, out_path)

            row: dict[str, object] = {
                "position_label": pos_dir.name,
                "time_index": time_index,
                "probability_filename": probability_path.name,
                "mask_filename": out_name,
                "mask_path": str(out_path),
                "used_in_round1_training": frame_key in training_frames,
                "exclude_from_analysis": bool(exclusion_info["exclude_from_analysis"]),
                "exclusion_reason": str(exclusion_info["exclusion_reason"]),
                "phase_border_dark_area_frac": exclusion_info["border_dark_area_frac"],
                "phase_dark_area_frac": exclusion_info["dark_area_frac"],
                "organoid_probability_mean": float(np.mean(organoid_probability)),
                "organoid_probability_p95": float(np.quantile(organoid_probability, 0.95)),
                "background_probability_mean": float(np.mean(background_probability))
                if isinstance(background_probability, np.ndarray)
                else np.nan,
                "threshold": args.threshold,
                "closing_radius": args.closing_radius,
                "min_component_area": args.min_component_area,
                "fill_lumen": fill_lumen,
                "previous_time_index": previous_time_index if previous_time_index is not None else np.nan,
            }
            row.update(metrics)
            row.update(temporal_metrics)
            rows.append(row)

            previous_mask = mask
            previous_time_index = time_index

    metrics_df = pd.DataFrame(rows).sort_values(["position_label", "time_index"]).reset_index(drop=True)

    persistent_onset_map, persistent_count_map = persistent_border_touch_summary(metrics_df)
    metrics_df["persistent_border_touch_onset"] = metrics_df["position_label"].map(
        lambda label: persistent_onset_map.get(label, None)
    )
    metrics_df["persistent_border_touch_frame_count"] = metrics_df["position_label"].map(
        lambda label: persistent_count_map.get(label, 0)
    )
    metrics_df["persistent_border_touch_position"] = metrics_df["persistent_border_touch_onset"].notna()
    metrics_df["persistent_border_touch_frame"] = (
        metrics_df["persistent_border_touch_onset"].notna()
        & (metrics_df["time_index"] >= metrics_df["persistent_border_touch_onset"].fillna(np.inf))
        & (~metrics_df["exclude_from_analysis"])
    )

    metrics_df["whole_organoid_metrics_allowed"] = (
        ~metrics_df["exclude_from_analysis"]
        & ~metrics_df["persistent_border_touch_frame"]
    )
    metrics_df["positive_signal_only_metrics_allowed"] = ~metrics_df["exclude_from_analysis"]
    metrics_df["analysis_metric_scope"] = np.select(
        [
            metrics_df["exclude_from_analysis"],
            metrics_df["persistent_border_touch_frame"],
        ],
        [
            "exclude_all_metrics",
            "positive_signal_only",
        ],
        default="all_metrics",
    )

    metrics_df.to_csv(metrics_path, sep="\t", index=False)

    included = metrics_df.loc[~metrics_df["exclude_from_analysis"]].copy()
    position_summary = (
        included.groupby("position_label")
        .agg(
            frame_count=("time_index", "size"),
            training_frame_count=("used_in_round1_training", "sum"),
            mean_mask_area_frac=("final_mask_area_frac", "mean"),
            median_mask_area_frac=("final_mask_area_frac", "median"),
            max_mask_area_frac=("final_mask_area_frac", "max"),
            border_touch_frame_count=("touches_border", "sum"),
            max_component_count=("component_count_raw", "max"),
            min_iou_prev=("iou_prev", "min"),
            max_area_change_frac_prev=("area_change_frac_prev", "max"),
            max_centroid_shift_px_prev=("centroid_shift_px_prev", "max"),
            max_p95_boundary_disp_prev=("p95_boundary_disp_prev", "max"),
            max_p99_boundary_disp_prev=("p99_boundary_disp_prev", "max"),
            max_max_boundary_disp_prev=("max_boundary_disp_prev", "max"),
            max_missing_prev_frac=("missing_prev_frac", "max"),
            max_gained_prev_frac=("gained_prev_frac", "max"),
            mean_boundary_prob=("boundary_prob_mean", "mean"),
            persistent_border_touch_onset=("persistent_border_touch_onset", "first"),
            persistent_border_touch_frame_count=("persistent_border_touch_frame_count", "first"),
        )
        .reset_index()
        .sort_values("position_label")
    )
    position_summary.to_csv(position_summary_path, sep="\t", index=False)

    border_policy_summary = position_summary.loc[
        position_summary["persistent_border_touch_onset"].notna()
    ].copy()
    if not border_policy_summary.empty:
        border_policy_summary["whole_organoid_metrics_policy"] = "exclude_from_onset"
        border_policy_summary["positive_signal_metrics_policy"] = "allow_if_not_artifact_excluded"
        border_policy_summary["policy_note"] = (
            "Persistent frame-border contact: exclude denominator or full-organoid metrics from onset onward."
        )
        border_policy_summary = border_policy_summary[
            [
                "position_label",
                "persistent_border_touch_onset",
                "persistent_border_touch_frame_count",
                "whole_organoid_metrics_policy",
                "positive_signal_metrics_policy",
                "policy_note",
            ]
        ]
    border_policy_summary.to_csv(border_policy_summary_path, sep="\t", index=False)

    parameters = {
        "probability_root": str(probability_root),
        "mask_root": str(mask_root),
        "metrics_path": str(metrics_path),
        "position_summary_path": str(position_summary_path),
        "border_policy_summary_path": str(border_policy_summary_path),
        "threshold": args.threshold,
        "closing_radius": args.closing_radius,
        "min_component_area": args.min_component_area,
        "fill_lumen": fill_lumen,
        "notes": [
            "Organoid probability channel assumed to be probability TIFF channel 0.",
            "Excluded late phase-artifact frames are still processed and written, but marked exclude_from_analysis=True.",
            "One cleaned mask is written per frame; downstream QC should focus on non-excluded frames first.",
            "Persistent border-touch frames are allowed for raw positive-signal metrics but not for whole-organoid denominator metrics.",
            "Temporal QC includes local boundary-displacement and gained/missing-area metrics so conservative tip-clipping can be surfaced even when whole-mask area stays stable.",
        ],
    }
    parameters_path.write_text(json.dumps(parameters, indent=2) + "\n")

    print(f"Wrote masks to: {mask_root}")
    print(f"Wrote frame metrics: {metrics_path}")
    print(f"Wrote position summary: {position_summary_path}")
    print(f"Wrote analysis-scope summary: {border_policy_summary_path}")
    print(f"Wrote parameter record: {parameters_path}")


if __name__ == "__main__":
    main()
