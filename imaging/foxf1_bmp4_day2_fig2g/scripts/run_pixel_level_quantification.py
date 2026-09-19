#!/usr/bin/env python3
"""Run DAPI-mask, reporter-threshold, containment, and density quantification.

NOT runnable from a clone alone. It reads the raw microscopy images, which are not deposited;
they are available from the lead contact on request, as the paper's Data Availability Statement
states. Place them in this lane's `data/with DAPI/` directory. See the lane's README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

try:
    from .day2_quantification_helpers import (
        CANONICAL_CHANNEL_ORDER,
        allocate_stack_sample_counts,
        background_correct_signal,
        background_reference_mask,
        balanced_density_map_table,
        build_raw_input_manifest,
        channel_index,
        compute_axis_scale,
        estimate_background_sigma,
        half_gaussian_threshold_from_values,
        load_czi_stack,
        otsu_threshold_from_values,
        positive_signal,
        sample_masked_coordinates,
        sample_masked_values,
        segment_whole_organoid_from_dapi_with_debug,
        standardize_by_background_sigma,
    )
except ImportError:
    from day2_quantification_helpers import (
        CANONICAL_CHANNEL_ORDER,
        allocate_stack_sample_counts,
        background_correct_signal,
        background_reference_mask,
        balanced_density_map_table,
        build_raw_input_manifest,
        channel_index,
        compute_axis_scale,
        estimate_background_sigma,
        half_gaussian_threshold_from_values,
        load_czi_stack,
        otsu_threshold_from_values,
        positive_signal,
        sample_masked_coordinates,
        sample_masked_values,
        segment_whole_organoid_from_dapi_with_debug,
        standardize_by_background_sigma,
    )


REPORTER_CHANNEL_KEYS = ("foxf1", "bmp4")


RAW_MISSING = (
    "No CZI stacks in {data_dir}.\n"
    "This step reads the 14 raw stacks (1.czi to 11.czi, 13.czi, 15.czi and 16.czi), which are not in this\n"
    "repository. They are available from the lead contact on request: see Raw data in the lane's README.md.\n"
    "Fig 2g itself is redrawn without them by scripts/render_fig2g_panel.py."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Lane directory (default: the directory holding scripts/)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/with DAPI"), help="Canonical raw input directory")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("results/manifests/raw_input_manifest.tsv"),
        help="Manifest output TSV path",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("results/masks/dapi_whole_organoid_masks"),
        help="Directory for per-plane DAPI mask TIFFs",
    )
    parser.add_argument(
        "--plane-metrics-output",
        type=Path,
        default=Path("results/tables/01_mask_and_plane_metrics.tsv"),
        help="Per-plane mask metrics TSV output",
    )
    parser.add_argument(
        "--threshold-output",
        type=Path,
        default=Path("results/tables/02_reporter_thresholds.tsv"),
        help="Reporter threshold-comparison TSV output",
    )
    parser.add_argument(
        "--axis-scale-output",
        type=Path,
        default=Path("results/tables/02_reporter_axis_scales.tsv"),
        help="Axis-scaling TSV output for signed reporter density plots",
    )
    parser.add_argument(
        "--plane-summary-output",
        type=Path,
        default=Path("results/tables/02_plane_reporter_summary.tsv"),
        help="Per-plane reporter summary TSV output",
    )
    parser.add_argument(
        "--stack-summary-output",
        type=Path,
        default=Path("results/tables/02_stack_reporter_summary.tsv"),
        help="Per-stack reporter summary TSV output",
    )
    parser.add_argument(
        "--pixel-sample-output",
        type=Path,
        default=Path("results/tables/02_sampled_pixel_profiles.tsv"),
        help="Balanced sampled pixel profile TSV output",
    )
    parser.add_argument(
        "--containment-plane-output",
        type=Path,
        default=Path("results/tables/03_plane_containment_by_method.tsv"),
        help="Per-plane containment metrics TSV output",
    )
    parser.add_argument(
        "--containment-stack-output",
        type=Path,
        default=Path("results/tables/03_stack_containment_by_method.tsv"),
        help="Per-stack containment metrics TSV output",
    )
    parser.add_argument(
        "--containment-method-output",
        type=Path,
        default=Path("results/tables/03_containment_method_summary.tsv"),
        help="Per-method containment summary TSV output",
    )
    parser.add_argument(
        "--density-output",
        type=Path,
        default=Path("results/tables/04_density_maps.tsv"),
        help="Signed 2D density-map TSV output",
    )
    parser.add_argument(
        "--selected-threshold-method-key",
        type=str,
        default="snr_fixed_4sigma",
        help="Threshold method key used for the selected positive-pixel summaries",
    )
    parser.add_argument(
        "--threshold-z",
        type=float,
        default=4.0,
        help="Half-Gaussian threshold sigma multiple used for the comparison methods",
    )
    parser.add_argument(
        "--fixed-snr-threshold",
        type=float,
        default=4.0,
        help="Center fixed reporter threshold in background-sigma units; review methods span a wider family around this value",
    )
    parser.add_argument(
        "--background-estimator",
        type=str,
        default="whole_off_organoid",
        choices=["whole_off_organoid", "annulus"],
        help="Background reference mode used for reporter correction",
    )
    parser.add_argument(
        "--background-annulus-inner-px",
        type=int,
        default=6,
        help="Inner annulus dilation radius for local background estimation",
    )
    parser.add_argument(
        "--background-annulus-outer-px",
        type=int,
        default=20,
        help="Outer annulus dilation radius for local background estimation",
    )
    parser.add_argument(
        "--background-min-ring-pixels",
        type=int,
        default=1000,
        help="Minimum annulus pixel count before falling back to broader off-mask regions",
    )
    parser.add_argument(
        "--threshold-sample-per-plane",
        type=int,
        default=50000,
        help="Maximum masked pixels sampled per plane for pooled threshold estimation",
    )
    parser.add_argument(
        "--output-sample-per-stack",
        type=int,
        default=30000,
        help="Maximum masked pixels sampled per image stack for plotting/export",
    )
    parser.add_argument("--density-bins", type=int, default=96, help="2D density-map bin count per axis")
    parser.add_argument(
        "--axis-scale-quantile",
        type=float,
        default=0.995,
        help="Global corrected-signal quantile used to scale signed FOXF1/BMP4 axes",
    )
    parser.add_argument("--random-seed", type=int, default=7, help="Random seed for pixel sampling")
    return parser.parse_args()


def resolve_path(root: Path, path_like: Path) -> Path:
    return path_like.resolve() if path_like.is_absolute() else (root / path_like).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_fraction(part: int, whole: int) -> float:
    return float(part / whole) if int(whole) > 0 else np.nan


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if float(den) > 0 else np.nan


def _nice_bounds(lo: float, hi: float, step: float = 0.1) -> tuple[float, float]:
    lo_n = np.floor(float(lo) / float(step)) * float(step)
    hi_n = np.ceil(float(hi) / float(step)) * float(step)
    if not np.isfinite(lo_n):
        lo_n = 0.0
    if not np.isfinite(hi_n) or hi_n <= lo_n:
        hi_n = lo_n + float(step)
    return float(lo_n), float(hi_n)


def _density_range_from_values(
    values: np.ndarray,
    q_low: float = 0.001,
    q_high: float = 0.999,
    min_upper: float = 1.0,
    step: float = 0.1,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (-0.1, 1.0)
    lo = float(np.quantile(arr, float(q_low)))
    hi = float(np.quantile(arr, float(q_high)))
    hi = max(float(min_upper), hi)
    return _nice_bounds(lo, hi, step=float(step))


def correct_signal_with_shared_background(
    signal: np.ndarray,
    organoid_mask: np.ndarray,
    shared_background_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    signal_f = np.asarray(signal, dtype=np.float32)
    bg_mask = np.asarray(shared_background_mask, dtype=bool)
    valid_bg = bg_mask & np.isfinite(signal_f)
    if np.any(valid_bg):
        bg_value = float(np.median(signal_f[valid_bg]))
    else:
        valid_off = (~np.asarray(organoid_mask, dtype=bool)) & np.isfinite(signal_f)
        bg_value = float(np.median(signal_f[valid_off])) if np.any(valid_off) else 0.0
    corrected = signal_f - np.float32(bg_value)
    corrected = np.where(np.isfinite(corrected), corrected, np.nan).astype(np.float32)
    return corrected, float(bg_value)


def build_threshold_method_specs(
    threshold_z: float,
    fixed_snr_threshold: float,
    selected_threshold_method_key: str,
) -> list[dict[str, Any]]:
    _ = float(threshold_z)
    center_snr_threshold = float(fixed_snr_threshold)
    fixed_snr_thresholds = [
        max(0.5, center_snr_threshold - 2.0),
        max(0.5, center_snr_threshold - 1.0),
        center_snr_threshold,
        center_snr_threshold + 1.0,
    ]

    def _sigma_method_key(value: float) -> str:
        value_f = float(value)
        if abs(value_f - round(value_f)) < 1.0e-6:
            return f"snr_fixed_{int(round(value_f))}sigma"
        whole = int(np.floor(value_f))
        frac = int(round((value_f - whole) * 10))
        return f"snr_fixed_{whole}p{frac}sigma"

    specs = [
        {
            "threshold_method_key": "snr_log1p_otsu",
            "threshold_basis": "bg_sigma_units",
            "threshold_estimator": "log1p_otsu",
            "threshold_z": np.nan,
            "fixed_threshold": np.nan,
        },
        {
            "threshold_method_key": "raw_log1p_otsu",
            "threshold_basis": "corrected",
            "threshold_estimator": "log1p_otsu",
            "threshold_z": np.nan,
            "fixed_threshold": np.nan,
        },
    ]
    for fixed_threshold in fixed_snr_thresholds:
        specs.append(
            {
                "threshold_method_key": _sigma_method_key(fixed_threshold),
                "threshold_basis": "bg_sigma_units",
                "threshold_estimator": "fixed",
                "threshold_z": np.nan,
                "fixed_threshold": float(fixed_threshold),
            }
        )
    keys = [str(spec["threshold_method_key"]) for spec in specs]
    if str(selected_threshold_method_key) not in keys:
        raise KeyError(
            f"Selected threshold method {selected_threshold_method_key!r} was not found in {keys!r}"
        )
    return specs


def estimate_threshold_table(
    threshold_samples: dict[tuple[str, str], list[np.ndarray]],
    threshold_method_specs: list[dict[str, Any]],
    selected_threshold_method_key: str,
) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], float] = {}

    for spec in threshold_method_specs:
        method_key = str(spec["threshold_method_key"])
        basis = str(spec["threshold_basis"])
        estimator = str(spec["threshold_estimator"])
        for channel_key in REPORTER_CHANNEL_KEYS:
            pooled = threshold_samples.get((basis, str(channel_key)), [])
            values = (
                np.concatenate(pooled).astype(np.float32)
                if pooled
                else np.zeros((0,), dtype=np.float32)
            )
            diag: dict[str, Any]
            if estimator == "log1p_otsu":
                diag = otsu_threshold_from_values(values, transform="log1p")
                threshold_value = float(diag["threshold_value"])
                estimator_label = str(diag["method"])
            elif estimator == "half_gaussian":
                diag = half_gaussian_threshold_from_values(values, z=float(spec["threshold_z"]))
                threshold_value = float(diag["threshold_value"])
                estimator_label = "half_gaussian"
            elif estimator == "fixed":
                diag = {"sample_count": int(values.size)}
                threshold_value = float(spec["fixed_threshold"])
                estimator_label = "fixed"
            else:
                raise ValueError(f"Unsupported threshold estimator: {estimator!r}")

            lookup[(method_key, str(channel_key))] = float(threshold_value)
            rows.append(
                {
                    "threshold_method_key": method_key,
                    "channel_key": str(channel_key),
                    "threshold_basis": basis,
                    "threshold_estimator": estimator_label,
                    "threshold_value": float(threshold_value),
                    "threshold_z": float(spec.get("threshold_z", np.nan)),
                    "fixed_threshold": float(spec.get("fixed_threshold", np.nan)),
                    "sample_count": int(diag.get("sample_count", int(values.size))),
                    "baseline_location": float(diag.get("baseline_location", np.nan)),
                    "baseline_scale": float(diag.get("baseline_scale", np.nan)),
                    "threshold_transformed": float(diag.get("threshold_transformed", np.nan)),
                    "is_selected_method": bool(method_key == str(selected_threshold_method_key)),
                }
            )

    threshold_df = pd.DataFrame(rows).sort_values(
        ["threshold_method_key", "channel_key"]
    ).reset_index(drop=True)
    return threshold_df, lookup


def _selected_review_fields(row: Any, z_index: int) -> dict[str, Any]:
    return {
        "selected_review_z_1based": int(row.selected_review_z_1based)
        if pd.notna(row.selected_review_z_1based)
        else np.nan,
        "selected_review_z_0based": int(row.selected_review_z_0based)
        if pd.notna(row.selected_review_z_0based)
        else np.nan,
        "is_selected_review_z": bool(
            pd.notna(row.selected_review_z_0based)
            and int(z_index) == int(row.selected_review_z_0based)
        ),
    }


def run_quantification_pipeline(
    root: Path | str = ".",
    data_dir: Path | str = "data/with DAPI",
    manifest_output: Path | str = "results/manifests/raw_input_manifest.tsv",
    mask_dir: Path | str = "results/masks/dapi_whole_organoid_masks",
    plane_metrics_output: Path | str = "results/tables/01_mask_and_plane_metrics.tsv",
    threshold_output: Path | str = "results/tables/02_reporter_thresholds.tsv",
    axis_scale_output: Path | str = "results/tables/02_reporter_axis_scales.tsv",
    plane_summary_output: Path | str = "results/tables/02_plane_reporter_summary.tsv",
    stack_summary_output: Path | str = "results/tables/02_stack_reporter_summary.tsv",
    pixel_sample_output: Path | str = "results/tables/02_sampled_pixel_profiles.tsv",
    containment_plane_output: Path | str = "results/tables/03_plane_containment_by_method.tsv",
    containment_stack_output: Path | str = "results/tables/03_stack_containment_by_method.tsv",
    containment_method_output: Path | str = "results/tables/03_containment_method_summary.tsv",
    density_output: Path | str = "results/tables/04_density_maps.tsv",
    selected_threshold_method_key: str = "snr_fixed_4sigma",
    threshold_z: float = 4.0,
    fixed_snr_threshold: float = 4.0,
    background_estimator: str = "whole_off_organoid",
    background_annulus_inner_px: int = 6,
    background_annulus_outer_px: int = 20,
    background_min_ring_pixels: int = 1000,
    threshold_sample_per_plane: int = 50000,
    output_sample_per_stack: int = 30000,
    density_bins: int = 96,
    axis_scale_quantile: float = 0.995,
    random_seed: int = 7,
    write_outputs: bool = True,
) -> dict[str, object]:
    root = Path(root).resolve()
    data_dir = resolve_path(root, Path(data_dir))
    if not any(data_dir.glob("*.czi")):
        raise FileNotFoundError(RAW_MISSING.format(data_dir=data_dir))
    manifest_output = resolve_path(root, Path(manifest_output))
    mask_dir = resolve_path(root, Path(mask_dir))
    plane_metrics_output = resolve_path(root, Path(plane_metrics_output))
    threshold_output = resolve_path(root, Path(threshold_output))
    axis_scale_output = resolve_path(root, Path(axis_scale_output))
    plane_summary_output = resolve_path(root, Path(plane_summary_output))
    stack_summary_output = resolve_path(root, Path(stack_summary_output))
    pixel_sample_output = resolve_path(root, Path(pixel_sample_output))
    containment_plane_output = resolve_path(root, Path(containment_plane_output))
    containment_stack_output = resolve_path(root, Path(containment_stack_output))
    containment_method_output = resolve_path(root, Path(containment_method_output))
    density_output = resolve_path(root, Path(density_output))

    for path in [
        manifest_output,
        plane_metrics_output,
        threshold_output,
        axis_scale_output,
        plane_summary_output,
        stack_summary_output,
        pixel_sample_output,
        containment_plane_output,
        containment_stack_output,
        containment_method_output,
        density_output,
    ]:
        ensure_parent(path)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(random_seed))

    manifest_df = build_raw_input_manifest(data_dir=data_dir, root=root)
    if bool(write_outputs):
        manifest_df.to_csv(manifest_output, sep="\t", index=False)

    threshold_method_specs = build_threshold_method_specs(
        threshold_z=float(threshold_z),
        fixed_snr_threshold=float(fixed_snr_threshold),
        selected_threshold_method_key=str(selected_threshold_method_key),
    )
    background_kwargs = {
        "background_estimator": str(background_estimator),
        "annulus_inner_radius_px": int(background_annulus_inner_px),
        "annulus_outer_radius_px": int(background_annulus_outer_px),
        "min_ring_pixels": int(background_min_ring_pixels),
    }

    threshold_samples: dict[tuple[str, str], list[np.ndarray]] = {}
    plane_metric_rows: list[dict[str, Any]] = []

    for row in manifest_df.itertuples(index=False):
        stack = load_czi_stack(resolve_path(root, Path(str(row.file_path))))
        if int(stack.data_czyx.shape[0]) < len(CANONICAL_CHANNEL_ORDER):
            raise RuntimeError(
                f"{row.file_name} has only {stack.data_czyx.shape[0]} channels; "
                f"expected at least {len(CANONICAL_CHANNEL_ORDER)}"
            )

        dapi_idx = channel_index(stack.canonical_channel_names, "dapi")
        foxf1_idx = channel_index(stack.canonical_channel_names, "foxf1")
        bmp4_idx = channel_index(stack.canonical_channel_names, "bmp4")

        for z_index in range(int(stack.data_czyx.shape[1])):
            dapi_raw = np.asarray(stack.data_czyx[dapi_idx, z_index], dtype=np.float32)
            mask, debug = segment_whole_organoid_from_dapi_with_debug(dapi_raw)
            abs_mask_path = mask_dir / f"{int(row.file_id):02d}_z{int(z_index):02d}_mask.tif"
            rel_mask_path = abs_mask_path.resolve().relative_to(root.resolve())
            tifffile.imwrite(abs_mask_path, mask.astype(np.uint8))

            shared_bg_mask = background_reference_mask(
                organoid_mask=mask,
                background_estimator=str(background_estimator),
                annulus_inner_radius_px=int(background_annulus_inner_px),
                annulus_outer_radius_px=int(background_annulus_outer_px),
                min_ring_pixels=int(background_min_ring_pixels),
            )
            dapi_corrected, dapi_bg_value = correct_signal_with_shared_background(
                dapi_raw,
                organoid_mask=mask,
                shared_background_mask=shared_bg_mask,
            )
            dapi_bg_mask = shared_bg_mask
            dapi_bg_sigma = estimate_background_sigma(dapi_corrected, dapi_bg_mask)
            plane_row = {
                "image_id": str(row.image_id),
                "file_id": int(row.file_id),
                "position_label": str(row.position_label),
                "file_name": str(row.file_name),
                "file_path": str(row.file_path),
                "z_index": int(z_index),
                **_selected_review_fields(row=row, z_index=int(z_index)),
                "mask_path": rel_mask_path.as_posix(),
                "mask_area_px": int(np.sum(mask)),
                "mask_fraction": float(np.mean(mask)),
                "baseline_value": float(debug["baseline_value"]),
                "raw_threshold": float(debug["raw_threshold"]),
                "threshold_value": float(debug["threshold_value"]),
                "selected_variant": str(debug["selected_variant"]),
                "n_raw_components": int(debug["n_raw_components"]),
                "background_estimator": str(background_estimator),
                "background_annulus_inner_px": int(background_annulus_inner_px),
                "background_annulus_outer_px": int(background_annulus_outer_px),
                "background_min_ring_pixels": int(background_min_ring_pixels),
                "dapi_background_value": float(dapi_bg_value),
                "dapi_bg_sigma": float(dapi_bg_sigma),
            }

            if int(np.sum(mask)) > 0:
                for channel_key, channel_idx in [("foxf1", foxf1_idx), ("bmp4", bmp4_idx)]:
                    raw = np.asarray(stack.data_czyx[channel_idx, z_index], dtype=np.float32)
                    corrected, bg_value = correct_signal_with_shared_background(
                        raw,
                        organoid_mask=mask,
                        shared_background_mask=shared_bg_mask,
                    )
                    bg_mask = shared_bg_mask
                    bg_sigma = estimate_background_sigma(corrected, bg_mask)
                    signal_positive = positive_signal(corrected)
                    snr_plane = standardize_by_background_sigma(corrected, bg_sigma)

                    sampled_corrected = sample_masked_values(
                        values=signal_positive,
                        mask=mask,
                        sample_size=int(threshold_sample_per_plane),
                        rng=rng,
                    )
                    sampled_snr = sample_masked_values(
                        values=positive_signal(snr_plane),
                        mask=mask,
                        sample_size=int(threshold_sample_per_plane),
                        rng=rng,
                    )
                    threshold_samples.setdefault(("corrected", channel_key), []).append(sampled_corrected)
                    threshold_samples.setdefault(("bg_sigma_units", channel_key), []).append(sampled_snr)

                    valid_vals = corrected[mask]
                    valid_vals = valid_vals[np.isfinite(valid_vals)]
                    valid_snr = snr_plane[mask]
                    valid_snr = valid_snr[np.isfinite(valid_snr)]

                    plane_row[f"{channel_key}_background_value"] = float(bg_value)
                    plane_row[f"{channel_key}_bg_sigma"] = float(bg_sigma)
                    plane_row[f"{channel_key}_corrected_mean"] = float(np.mean(valid_vals)) if valid_vals.size else np.nan
                    plane_row[f"{channel_key}_corrected_median"] = float(np.median(valid_vals)) if valid_vals.size else np.nan
                    plane_row[f"{channel_key}_corrected_q90"] = float(np.quantile(valid_vals, 0.90)) if valid_vals.size else np.nan
                    plane_row[f"{channel_key}_corrected_q99"] = float(np.quantile(valid_vals, 0.99)) if valid_vals.size else np.nan
                    plane_row[f"{channel_key}_snr_mean"] = float(np.mean(valid_snr)) if valid_snr.size else np.nan
                    plane_row[f"{channel_key}_snr_median"] = float(np.median(valid_snr)) if valid_snr.size else np.nan
                    plane_row[f"{channel_key}_fraction_gt_background"] = float(np.mean(valid_vals > 0)) if valid_vals.size else np.nan
                plane_row["mask_status"] = "ok"
            else:
                plane_row["mask_status"] = "empty_mask"
                for channel_key in REPORTER_CHANNEL_KEYS:
                    plane_row[f"{channel_key}_background_value"] = np.nan
                    plane_row[f"{channel_key}_bg_sigma"] = np.nan
                    plane_row[f"{channel_key}_corrected_mean"] = np.nan
                    plane_row[f"{channel_key}_corrected_median"] = np.nan
                    plane_row[f"{channel_key}_corrected_q90"] = np.nan
                    plane_row[f"{channel_key}_corrected_q99"] = np.nan
                    plane_row[f"{channel_key}_snr_mean"] = np.nan
                    plane_row[f"{channel_key}_snr_median"] = np.nan
                    plane_row[f"{channel_key}_fraction_gt_background"] = np.nan

            plane_metric_rows.append(plane_row)

    plane_metrics_df = pd.DataFrame(plane_metric_rows).sort_values(
        ["file_id", "z_index"]
    ).reset_index(drop=True)
    if bool(write_outputs):
        plane_metrics_df.to_csv(plane_metrics_output, sep="\t", index=False)

    threshold_df, threshold_lookup = estimate_threshold_table(
        threshold_samples=threshold_samples,
        threshold_method_specs=threshold_method_specs,
        selected_threshold_method_key=str(selected_threshold_method_key),
    )
    if bool(write_outputs):
        threshold_df.to_csv(threshold_output, sep="\t", index=False)

    selected_threshold_lookup = {
        channel_key: float(threshold_lookup[(str(selected_threshold_method_key), channel_key)])
        for channel_key in REPORTER_CHANNEL_KEYS
    }
    selected_threshold_basis = str(
        threshold_df.loc[
            threshold_df["threshold_method_key"] == str(selected_threshold_method_key),
            "threshold_basis",
        ].iloc[0]
    )

    plane_summary_rows: list[dict[str, Any]] = []
    stack_summary_rows: list[dict[str, Any]] = []
    sampled_pixel_rows: list[dict[str, Any]] = []
    containment_plane_rows: list[dict[str, Any]] = []
    containment_stack_rows: list[dict[str, Any]] = []

    for row in manifest_df.itertuples(index=False):
        stack = load_czi_stack(resolve_path(root, Path(str(row.file_path))))
        dapi_idx = channel_index(stack.canonical_channel_names, "dapi")
        foxf1_idx = channel_index(stack.canonical_channel_names, "foxf1")
        bmp4_idx = channel_index(stack.canonical_channel_names, "bmp4")

        sub_metrics = plane_metrics_df.loc[plane_metrics_df["file_id"] == int(row.file_id)].copy()
        mask_sizes = sub_metrics["mask_area_px"].fillna(0).astype(int).tolist()
        sample_allocations = allocate_stack_sample_counts(
            mask_sizes=mask_sizes,
            target_total=int(output_sample_per_stack),
        )

        stack_mask_total = 0
        stack_sampled_total = 0
        stack_foxf1_sum = 0.0
        stack_bmp4_sum = 0.0
        stack_foxf1_snr_sum = 0.0
        stack_bmp4_snr_sum = 0.0
        stack_value_count = 0
        stack_method_totals: dict[str, dict[str, float]] = {
            str(spec["threshold_method_key"]): {
                "mask_pixels": 0,
                "foxf1_positive_pixels": 0,
                "bmp4_positive_pixels": 0,
                "double_positive_pixels": 0,
                "bmp4_positive_signal_total": 0.0,
                "bmp4_positive_signal_in_foxf1_positive": 0.0,
                "foxf1_positive_signal_total": 0.0,
                "foxf1_positive_signal_in_bmp4_positive": 0.0,
            }
            for spec in threshold_method_specs
        }

        for plane_metric, sample_count in zip(sub_metrics.itertuples(index=False), sample_allocations):
            mask_path = resolve_path(root, Path(str(plane_metric.mask_path)))
            mask = tifffile.imread(mask_path).astype(bool)
            z_index = int(plane_metric.z_index)

            plane_base = {
                "image_id": str(row.image_id),
                "file_id": int(row.file_id),
                "position_label": str(row.position_label),
                "file_name": str(row.file_name),
                "z_index": int(z_index),
                **_selected_review_fields(row=row, z_index=int(z_index)),
                "mask_pixels": int(np.sum(mask)),
                "mask_fraction": float(plane_metric.mask_fraction),
                "sampled_pixel_count": int(sample_count),
                "selected_threshold_method_key": str(selected_threshold_method_key),
                "selected_threshold_basis": selected_threshold_basis,
                "foxf1_threshold_selected": float(selected_threshold_lookup["foxf1"]),
                "bmp4_threshold_selected": float(selected_threshold_lookup["bmp4"]),
            }

            if int(np.sum(mask)) == 0:
                plane_summary_rows.append(
                    {
                        **plane_base,
                        "foxf1_corrected_mean": np.nan,
                        "foxf1_corrected_median": np.nan,
                        "foxf1_corrected_q90": np.nan,
                        "foxf1_snr_mean": np.nan,
                        "foxf1_snr_median": np.nan,
                        "bmp4_corrected_mean": np.nan,
                        "bmp4_corrected_median": np.nan,
                        "bmp4_corrected_q90": np.nan,
                        "bmp4_snr_mean": np.nan,
                        "bmp4_snr_median": np.nan,
                        "foxf1_positive_fraction_selected": np.nan,
                        "bmp4_positive_fraction_selected": np.nan,
                        "double_positive_fraction_selected": np.nan,
                        "bmp4_signal_fraction_in_foxf1_positive_selected": np.nan,
                        "foxf1_signal_fraction_in_bmp4_positive_selected": np.nan,
                    }
                )
                for spec in threshold_method_specs:
                    containment_plane_rows.append(
                        {
                            **plane_base,
                            "threshold_method_key": str(spec["threshold_method_key"]),
                            "threshold_basis": str(spec["threshold_basis"]),
                            "is_selected_method": bool(
                                str(spec["threshold_method_key"]) == str(selected_threshold_method_key)
                            ),
                            "foxf1_positive_pixels": 0,
                            "bmp4_positive_pixels": 0,
                            "double_positive_pixels": 0,
                            "foxf1_positive_fraction": np.nan,
                            "bmp4_positive_fraction": np.nan,
                            "double_positive_fraction": np.nan,
                            "bmp4_positive_in_foxf1_fraction": np.nan,
                            "foxf1_positive_in_bmp4_fraction": np.nan,
                            "bmp4_signal_fraction_in_foxf1_positive": np.nan,
                            "foxf1_signal_fraction_in_bmp4_positive": np.nan,
                        }
                    )
                continue

            foxf1_raw = np.asarray(stack.data_czyx[foxf1_idx, z_index], dtype=np.float32)
            bmp4_raw = np.asarray(stack.data_czyx[bmp4_idx, z_index], dtype=np.float32)
            shared_bg_mask = background_reference_mask(
                organoid_mask=mask,
                background_estimator=str(background_estimator),
                annulus_inner_radius_px=int(background_annulus_inner_px),
                annulus_outer_radius_px=int(background_annulus_outer_px),
                min_ring_pixels=int(background_min_ring_pixels),
            )
            foxf1_corrected, _ = correct_signal_with_shared_background(
                foxf1_raw,
                organoid_mask=mask,
                shared_background_mask=shared_bg_mask,
            )
            bmp4_corrected, _ = correct_signal_with_shared_background(
                bmp4_raw,
                organoid_mask=mask,
                shared_background_mask=shared_bg_mask,
            )
            foxf1_bg_mask = shared_bg_mask
            bmp4_bg_mask = shared_bg_mask
            foxf1_bg_sigma = estimate_background_sigma(foxf1_corrected, foxf1_bg_mask)
            bmp4_bg_sigma = estimate_background_sigma(bmp4_corrected, bmp4_bg_mask)
            foxf1_snr = standardize_by_background_sigma(foxf1_corrected, foxf1_bg_sigma)
            bmp4_snr = standardize_by_background_sigma(bmp4_corrected, bmp4_bg_sigma)
            foxf1_positive_signal = positive_signal(foxf1_corrected)
            bmp4_positive_signal = positive_signal(bmp4_corrected)

            foxf1_values = np.asarray(foxf1_corrected[mask], dtype=np.float32)
            foxf1_values = foxf1_values[np.isfinite(foxf1_values)]
            bmp4_values = np.asarray(bmp4_corrected[mask], dtype=np.float32)
            bmp4_values = bmp4_values[np.isfinite(bmp4_values)]
            foxf1_snr_values = np.asarray(foxf1_snr[mask], dtype=np.float32)
            foxf1_snr_values = foxf1_snr_values[np.isfinite(foxf1_snr_values)]
            bmp4_snr_values = np.asarray(bmp4_snr[mask], dtype=np.float32)
            bmp4_snr_values = bmp4_snr_values[np.isfinite(bmp4_snr_values)]

            selected_foxf1_basis = foxf1_corrected if selected_threshold_basis == "corrected" else foxf1_snr
            selected_bmp4_basis = bmp4_corrected if selected_threshold_basis == "corrected" else bmp4_snr
            foxf1_positive_selected = (
                mask
                & np.isfinite(selected_foxf1_basis)
                & (selected_foxf1_basis > float(selected_threshold_lookup["foxf1"]))
            )
            bmp4_positive_selected = (
                mask
                & np.isfinite(selected_bmp4_basis)
                & (selected_bmp4_basis > float(selected_threshold_lookup["bmp4"]))
            )
            double_positive_selected = foxf1_positive_selected & bmp4_positive_selected
            mask_pixels = int(np.sum(mask))
            bmp4_signal_total_plane = float(np.nansum(bmp4_positive_signal[mask]))
            foxf1_signal_total_plane = float(np.nansum(foxf1_positive_signal[mask]))

            plane_summary_rows.append(
                {
                    **plane_base,
                    "foxf1_bg_sigma": float(foxf1_bg_sigma),
                    "bmp4_bg_sigma": float(bmp4_bg_sigma),
                    "foxf1_corrected_mean": float(np.mean(foxf1_values)) if foxf1_values.size else np.nan,
                    "foxf1_corrected_median": float(np.median(foxf1_values)) if foxf1_values.size else np.nan,
                    "foxf1_corrected_q90": float(np.quantile(foxf1_values, 0.90)) if foxf1_values.size else np.nan,
                    "foxf1_snr_mean": float(np.mean(foxf1_snr_values)) if foxf1_snr_values.size else np.nan,
                    "foxf1_snr_median": float(np.median(foxf1_snr_values)) if foxf1_snr_values.size else np.nan,
                    "bmp4_corrected_mean": float(np.mean(bmp4_values)) if bmp4_values.size else np.nan,
                    "bmp4_corrected_median": float(np.median(bmp4_values)) if bmp4_values.size else np.nan,
                    "bmp4_corrected_q90": float(np.quantile(bmp4_values, 0.90)) if bmp4_values.size else np.nan,
                    "bmp4_snr_mean": float(np.mean(bmp4_snr_values)) if bmp4_snr_values.size else np.nan,
                    "bmp4_snr_median": float(np.median(bmp4_snr_values)) if bmp4_snr_values.size else np.nan,
                    "foxf1_positive_fraction_selected": float(np.mean(foxf1_positive_selected[mask])),
                    "bmp4_positive_fraction_selected": float(np.mean(bmp4_positive_selected[mask])),
                    "double_positive_fraction_selected": float(np.mean(double_positive_selected[mask])),
                    "bmp4_signal_fraction_in_foxf1_positive_selected": _safe_div(
                        float(np.nansum(bmp4_positive_signal[foxf1_positive_selected])),
                        bmp4_signal_total_plane,
                    ),
                    "foxf1_signal_fraction_in_bmp4_positive_selected": _safe_div(
                        float(np.nansum(foxf1_positive_signal[bmp4_positive_selected])),
                        foxf1_signal_total_plane,
                    ),
                }
            )

            sample_coords = sample_masked_coordinates(mask=mask, sample_size=int(sample_count), rng=rng)
            for coord in sample_coords:
                row_idx = int(coord[0])
                col_idx = int(coord[1])
                sampled_pixel_rows.append(
                    {
                        "image_id": str(row.image_id),
                        "file_id": int(row.file_id),
                        "position_label": str(row.position_label),
                        "file_name": str(row.file_name),
                        "z_index": int(z_index),
                        **_selected_review_fields(row=row, z_index=int(z_index)),
                        "row_px": row_idx,
                        "col_px": col_idx,
                        "selected_threshold_method_key": str(selected_threshold_method_key),
                        "foxf1_corrected": float(foxf1_corrected[row_idx, col_idx]),
                        "bmp4_corrected": float(bmp4_corrected[row_idx, col_idx]),
                        "foxf1_corrected_positive": float(foxf1_positive_signal[row_idx, col_idx]),
                        "bmp4_corrected_positive": float(bmp4_positive_signal[row_idx, col_idx]),
                        "foxf1_bg_sigma_plane": float(foxf1_bg_sigma),
                        "bmp4_bg_sigma_plane": float(bmp4_bg_sigma),
                        "foxf1_snr_plane": float(foxf1_snr[row_idx, col_idx]),
                        "bmp4_snr_plane": float(bmp4_snr[row_idx, col_idx]),
                        "foxf1_positive_selected": bool(foxf1_positive_selected[row_idx, col_idx]),
                        "bmp4_positive_selected": bool(bmp4_positive_selected[row_idx, col_idx]),
                        "double_positive_selected": bool(double_positive_selected[row_idx, col_idx]),
                    }
                )

            stack_mask_total += mask_pixels
            stack_sampled_total += int(sample_count)
            stack_foxf1_sum += float(np.nansum(foxf1_values))
            stack_bmp4_sum += float(np.nansum(bmp4_values))
            stack_foxf1_snr_sum += float(np.nansum(foxf1_snr_values))
            stack_bmp4_snr_sum += float(np.nansum(bmp4_snr_values))
            stack_value_count += int(mask_pixels)

            for spec in threshold_method_specs:
                method_key = str(spec["threshold_method_key"])
                threshold_basis = str(spec["threshold_basis"])
                foxf1_basis = foxf1_corrected if threshold_basis == "corrected" else foxf1_snr
                bmp4_basis = bmp4_corrected if threshold_basis == "corrected" else bmp4_snr
                foxf1_positive = (
                    mask
                    & np.isfinite(foxf1_basis)
                    & (foxf1_basis > float(threshold_lookup[(method_key, "foxf1")]))
                )
                bmp4_positive = (
                    mask
                    & np.isfinite(bmp4_basis)
                    & (bmp4_basis > float(threshold_lookup[(method_key, "bmp4")]))
                )
                double_positive = foxf1_positive & bmp4_positive
                foxf1_positive_pixels = int(np.sum(foxf1_positive))
                bmp4_positive_pixels = int(np.sum(bmp4_positive))
                double_positive_pixels = int(np.sum(double_positive))

                bmp4_signal_in_foxf1 = float(np.nansum(bmp4_positive_signal[foxf1_positive]))
                foxf1_signal_in_bmp4 = float(np.nansum(foxf1_positive_signal[bmp4_positive]))

                containment_plane_rows.append(
                    {
                        **plane_base,
                        "threshold_method_key": method_key,
                        "threshold_basis": threshold_basis,
                        "is_selected_method": bool(method_key == str(selected_threshold_method_key)),
                        "foxf1_threshold_value": float(threshold_lookup[(method_key, "foxf1")]),
                        "bmp4_threshold_value": float(threshold_lookup[(method_key, "bmp4")]),
                        "foxf1_positive_pixels": foxf1_positive_pixels,
                        "bmp4_positive_pixels": bmp4_positive_pixels,
                        "double_positive_pixels": double_positive_pixels,
                        "foxf1_positive_fraction": _safe_fraction(foxf1_positive_pixels, mask_pixels),
                        "bmp4_positive_fraction": _safe_fraction(bmp4_positive_pixels, mask_pixels),
                        "double_positive_fraction": _safe_fraction(double_positive_pixels, mask_pixels),
                        "bmp4_positive_in_foxf1_fraction": _safe_fraction(
                            double_positive_pixels,
                            bmp4_positive_pixels,
                        ),
                        "foxf1_positive_in_bmp4_fraction": _safe_fraction(
                            double_positive_pixels,
                            foxf1_positive_pixels,
                        ),
                        "bmp4_signal_fraction_in_foxf1_positive": _safe_div(
                            bmp4_signal_in_foxf1,
                            bmp4_signal_total_plane,
                        ),
                        "foxf1_signal_fraction_in_bmp4_positive": _safe_div(
                            foxf1_signal_in_bmp4,
                            foxf1_signal_total_plane,
                        ),
                    }
                )

                acc = stack_method_totals[method_key]
                acc["mask_pixels"] += mask_pixels
                acc["foxf1_positive_pixels"] += foxf1_positive_pixels
                acc["bmp4_positive_pixels"] += bmp4_positive_pixels
                acc["double_positive_pixels"] += double_positive_pixels
                acc["bmp4_positive_signal_total"] += bmp4_signal_total_plane
                acc["bmp4_positive_signal_in_foxf1_positive"] += bmp4_signal_in_foxf1
                acc["foxf1_positive_signal_total"] += foxf1_signal_total_plane
                acc["foxf1_positive_signal_in_bmp4_positive"] += foxf1_signal_in_bmp4

        selected_acc = stack_method_totals[str(selected_threshold_method_key)]
        stack_summary_rows.append(
            {
                "image_id": str(row.image_id),
                "file_id": int(row.file_id),
                "position_label": str(row.position_label),
                "file_name": str(row.file_name),
                "selected_review_z_1based": int(row.selected_review_z_1based)
                if pd.notna(row.selected_review_z_1based)
                else np.nan,
                "z_planes": int(row.size_z),
                "mask_pixels": int(stack_mask_total),
                "sampled_pixel_count": int(stack_sampled_total),
                "mean_mask_fraction": float(sub_metrics["mask_fraction"].mean()),
                "foxf1_corrected_mean": _safe_div(stack_foxf1_sum, stack_value_count),
                "bmp4_corrected_mean": _safe_div(stack_bmp4_sum, stack_value_count),
                "foxf1_snr_mean": _safe_div(stack_foxf1_snr_sum, stack_value_count),
                "bmp4_snr_mean": _safe_div(stack_bmp4_snr_sum, stack_value_count),
                "selected_threshold_method_key": str(selected_threshold_method_key),
                "selected_threshold_basis": selected_threshold_basis,
                "foxf1_positive_fraction_selected": _safe_fraction(
                    int(selected_acc["foxf1_positive_pixels"]),
                    int(selected_acc["mask_pixels"]),
                ),
                "bmp4_positive_fraction_selected": _safe_fraction(
                    int(selected_acc["bmp4_positive_pixels"]),
                    int(selected_acc["mask_pixels"]),
                ),
                "double_positive_fraction_selected": _safe_fraction(
                    int(selected_acc["double_positive_pixels"]),
                    int(selected_acc["mask_pixels"]),
                ),
                "bmp4_positive_in_foxf1_fraction_selected": _safe_fraction(
                    int(selected_acc["double_positive_pixels"]),
                    int(selected_acc["bmp4_positive_pixels"]),
                ),
                "foxf1_positive_in_bmp4_fraction_selected": _safe_fraction(
                    int(selected_acc["double_positive_pixels"]),
                    int(selected_acc["foxf1_positive_pixels"]),
                ),
                "bmp4_signal_fraction_in_foxf1_positive_selected": _safe_div(
                    float(selected_acc["bmp4_positive_signal_in_foxf1_positive"]),
                    float(selected_acc["bmp4_positive_signal_total"]),
                ),
                "foxf1_signal_fraction_in_bmp4_positive_selected": _safe_div(
                    float(selected_acc["foxf1_positive_signal_in_bmp4_positive"]),
                    float(selected_acc["foxf1_positive_signal_total"]),
                ),
            }
        )

        for spec in threshold_method_specs:
            method_key = str(spec["threshold_method_key"])
            acc = stack_method_totals[method_key]
            containment_stack_rows.append(
                {
                    "image_id": str(row.image_id),
                    "file_id": int(row.file_id),
                    "position_label": str(row.position_label),
                    "file_name": str(row.file_name),
                    "selected_review_z_1based": int(row.selected_review_z_1based)
                    if pd.notna(row.selected_review_z_1based)
                    else np.nan,
                    "z_planes": int(row.size_z),
                    "threshold_method_key": method_key,
                    "threshold_basis": str(spec["threshold_basis"]),
                    "is_selected_method": bool(method_key == str(selected_threshold_method_key)),
                    "foxf1_threshold_value": float(threshold_lookup[(method_key, "foxf1")]),
                    "bmp4_threshold_value": float(threshold_lookup[(method_key, "bmp4")]),
                    "mask_pixels": int(acc["mask_pixels"]),
                    "foxf1_positive_pixels": int(acc["foxf1_positive_pixels"]),
                    "bmp4_positive_pixels": int(acc["bmp4_positive_pixels"]),
                    "double_positive_pixels": int(acc["double_positive_pixels"]),
                    "foxf1_positive_fraction": _safe_fraction(
                        int(acc["foxf1_positive_pixels"]),
                        int(acc["mask_pixels"]),
                    ),
                    "bmp4_positive_fraction": _safe_fraction(
                        int(acc["bmp4_positive_pixels"]),
                        int(acc["mask_pixels"]),
                    ),
                    "double_positive_fraction": _safe_fraction(
                        int(acc["double_positive_pixels"]),
                        int(acc["mask_pixels"]),
                    ),
                    "bmp4_positive_in_foxf1_fraction": _safe_fraction(
                        int(acc["double_positive_pixels"]),
                        int(acc["bmp4_positive_pixels"]),
                    ),
                    "foxf1_positive_in_bmp4_fraction": _safe_fraction(
                        int(acc["double_positive_pixels"]),
                        int(acc["foxf1_positive_pixels"]),
                    ),
                    "bmp4_signal_fraction_in_foxf1_positive": _safe_div(
                        float(acc["bmp4_positive_signal_in_foxf1_positive"]),
                        float(acc["bmp4_positive_signal_total"]),
                    ),
                    "foxf1_signal_fraction_in_bmp4_positive": _safe_div(
                        float(acc["foxf1_positive_signal_in_bmp4_positive"]),
                        float(acc["foxf1_positive_signal_total"]),
                    ),
                }
            )

    plane_summary_df = pd.DataFrame(plane_summary_rows).sort_values(
        ["file_id", "z_index"]
    ).reset_index(drop=True)
    stack_summary_df = pd.DataFrame(stack_summary_rows).sort_values(
        ["file_id"]
    ).reset_index(drop=True)
    sampled_pixels_df = pd.DataFrame(sampled_pixel_rows).sort_values(
        ["file_id", "z_index", "row_px", "col_px"]
    ).reset_index(drop=True)
    containment_plane_df = pd.DataFrame(containment_plane_rows).sort_values(
        ["threshold_method_key", "file_id", "z_index"]
    ).reset_index(drop=True)
    containment_stack_df = pd.DataFrame(containment_stack_rows).sort_values(
        ["threshold_method_key", "file_id"]
    ).reset_index(drop=True)

    foxf1_axis_scale = compute_axis_scale(
        sampled_pixels_df["foxf1_corrected_positive"].to_numpy(dtype=np.float32)
        if not sampled_pixels_df.empty
        else np.zeros((0,), dtype=np.float32),
        quantile=float(axis_scale_quantile),
    )
    bmp4_axis_scale = compute_axis_scale(
        sampled_pixels_df["bmp4_corrected_positive"].to_numpy(dtype=np.float32)
        if not sampled_pixels_df.empty
        else np.zeros((0,), dtype=np.float32),
        quantile=float(axis_scale_quantile),
    )
    axis_scale_df = pd.DataFrame(
        [
            {
                "channel_key": "foxf1",
                "axis_scale_quantile": float(axis_scale_quantile),
                "axis_scale_basis": "corrected_positive_signal",
                "axis_scale_value": float(foxf1_axis_scale),
            },
            {
                "channel_key": "bmp4",
                "axis_scale_quantile": float(axis_scale_quantile),
                "axis_scale_basis": "corrected_positive_signal",
                "axis_scale_value": float(bmp4_axis_scale),
            },
        ]
    )

    if not sampled_pixels_df.empty:
        sampled_pixels_df["foxf1_signed_au"] = (
            sampled_pixels_df["foxf1_corrected"].to_numpy(dtype=np.float32) / float(foxf1_axis_scale)
        )
        sampled_pixels_df["bmp4_signed_au"] = (
            sampled_pixels_df["bmp4_corrected"].to_numpy(dtype=np.float32) / float(bmp4_axis_scale)
        )
        sampled_pixels_df["foxf1_clipped_au"] = np.clip(
            sampled_pixels_df["foxf1_corrected_positive"].to_numpy(dtype=np.float32) / float(foxf1_axis_scale),
            0.0,
            1.0,
        )
        sampled_pixels_df["bmp4_clipped_au"] = np.clip(
            sampled_pixels_df["bmp4_corrected_positive"].to_numpy(dtype=np.float32) / float(bmp4_axis_scale),
            0.0,
            1.0,
        )
    else:
        for col in ["foxf1_signed_au", "bmp4_signed_au", "foxf1_clipped_au", "bmp4_clipped_au"]:
            sampled_pixels_df[col] = pd.Series(dtype=float)

    if containment_stack_df.empty:
        containment_method_df = pd.DataFrame(
            columns=[
                "threshold_method_key",
                "threshold_basis",
                "is_selected_method",
                "foxf1_threshold_value",
                "bmp4_threshold_value",
                "n_stacks",
                "median_foxf1_positive_fraction",
                "median_bmp4_positive_fraction",
                "median_double_positive_fraction",
                "median_bmp4_positive_in_foxf1_fraction",
                "median_foxf1_positive_in_bmp4_fraction",
                "median_bmp4_signal_fraction_in_foxf1_positive",
                "median_foxf1_signal_fraction_in_bmp4_positive",
            ]
        )
    else:
        threshold_pairs = (
            containment_stack_df[
                [
                    "threshold_method_key",
                    "threshold_basis",
                    "is_selected_method",
                    "foxf1_threshold_value",
                    "bmp4_threshold_value",
                ]
            ]
            .drop_duplicates()
            .sort_values("threshold_method_key")
        )
        summary_rows: list[dict[str, Any]] = []
        for pair in threshold_pairs.itertuples(index=False):
            sub = containment_stack_df.loc[
                containment_stack_df["threshold_method_key"] == str(pair.threshold_method_key)
            ].copy()
            summary_rows.append(
                {
                    "threshold_method_key": str(pair.threshold_method_key),
                    "threshold_basis": str(pair.threshold_basis),
                    "is_selected_method": bool(pair.is_selected_method),
                    "foxf1_threshold_value": float(pair.foxf1_threshold_value),
                    "bmp4_threshold_value": float(pair.bmp4_threshold_value),
                    "n_stacks": int(sub["image_id"].nunique()),
                    "median_foxf1_positive_fraction": float(sub["foxf1_positive_fraction"].median()),
                    "median_bmp4_positive_fraction": float(sub["bmp4_positive_fraction"].median()),
                    "median_double_positive_fraction": float(sub["double_positive_fraction"].median()),
                    "median_bmp4_positive_in_foxf1_fraction": float(
                        sub["bmp4_positive_in_foxf1_fraction"].median()
                    ),
                    "median_foxf1_positive_in_bmp4_fraction": float(
                        sub["foxf1_positive_in_bmp4_fraction"].median()
                    ),
                    "median_bmp4_signal_fraction_in_foxf1_positive": float(
                        sub["bmp4_signal_fraction_in_foxf1_positive"].median()
                    ),
                    "median_foxf1_signal_fraction_in_bmp4_positive": float(
                        sub["foxf1_signal_fraction_in_bmp4_positive"].median()
                    ),
                }
            )
        containment_method_df = pd.DataFrame(summary_rows).sort_values(
            ["is_selected_method", "threshold_method_key"],
            ascending=[False, True],
        ).reset_index(drop=True)

    density_x_range = _density_range_from_values(
        sampled_pixels_df["foxf1_signed_au"].to_numpy(dtype=np.float32)
        if not sampled_pixels_df.empty
        else np.zeros((0,), dtype=np.float32)
    )
    density_y_range = _density_range_from_values(
        sampled_pixels_df["bmp4_signed_au"].to_numpy(dtype=np.float32)
        if not sampled_pixels_df.empty
        else np.zeros((0,), dtype=np.float32)
    )
    density_df = balanced_density_map_table(
        df=sampled_pixels_df,
        x_col="foxf1_signed_au",
        y_col="bmp4_signed_au",
        map_name="foxf1_vs_bmp4_signed_au",
        stack_col="image_id",
        group_specs={
            "all_masked": lambda frame: pd.Series(True, index=frame.index),
            "signal_bearing_either": lambda frame: (
                (frame["foxf1_corrected"] > 0) | (frame["bmp4_corrected"] > 0)
            ),
            "signal_bearing_both": lambda frame: (
                (frame["foxf1_corrected"] > 0) & (frame["bmp4_corrected"] > 0)
            ),
            f"{selected_threshold_method_key}_double_positive": lambda frame: frame["double_positive_selected"].astype(bool),
        },
        bins=int(density_bins),
        x_range=density_x_range,
        y_range=density_y_range,
    )

    if bool(write_outputs):
        plane_summary_df.to_csv(plane_summary_output, sep="\t", index=False)
        stack_summary_df.to_csv(stack_summary_output, sep="\t", index=False)
        axis_scale_df.to_csv(axis_scale_output, sep="\t", index=False)
        sampled_pixels_df.to_csv(pixel_sample_output, sep="\t", index=False)
        containment_plane_df.to_csv(containment_plane_output, sep="\t", index=False)
        containment_stack_df.to_csv(containment_stack_output, sep="\t", index=False)
        containment_method_df.to_csv(containment_method_output, sep="\t", index=False)
        density_df.to_csv(density_output, sep="\t", index=False)

    return {
        "root": root,
        "data_dir": data_dir,
        "manifest_output": manifest_output,
        "mask_dir": mask_dir,
        "plane_metrics_output": plane_metrics_output,
        "threshold_output": threshold_output,
        "axis_scale_output": axis_scale_output,
        "plane_summary_output": plane_summary_output,
        "stack_summary_output": stack_summary_output,
        "pixel_sample_output": pixel_sample_output,
        "containment_plane_output": containment_plane_output,
        "containment_stack_output": containment_stack_output,
        "containment_method_output": containment_method_output,
        "density_output": density_output,
        "selected_threshold_method_key": str(selected_threshold_method_key),
        "manifest_df": manifest_df,
        "plane_metrics_df": plane_metrics_df,
        "threshold_df": threshold_df,
        "axis_scale_df": axis_scale_df,
        "plane_summary_df": plane_summary_df,
        "stack_summary_df": stack_summary_df,
        "sampled_pixels_df": sampled_pixels_df,
        "containment_plane_df": containment_plane_df,
        "containment_stack_df": containment_stack_df,
        "containment_method_df": containment_method_df,
        "density_df": density_df,
        "threshold_lookup": threshold_lookup,
        "summary_text": (
            f"canonical_czi_inputs={len(manifest_df)} | "
            f"mask_planes={len(plane_metrics_df)} | "
            f"sampled_pixels={len(sampled_pixels_df)} | "
            f"selected_threshold_method={selected_threshold_method_key}"
        ),
    }


def main() -> None:
    args = parse_args()
    res = run_quantification_pipeline(
        root=args.root,
        data_dir=args.data_dir,
        manifest_output=args.manifest_output,
        mask_dir=args.mask_dir,
        plane_metrics_output=args.plane_metrics_output,
        threshold_output=args.threshold_output,
        axis_scale_output=args.axis_scale_output,
        plane_summary_output=args.plane_summary_output,
        stack_summary_output=args.stack_summary_output,
        pixel_sample_output=args.pixel_sample_output,
        containment_plane_output=args.containment_plane_output,
        containment_stack_output=args.containment_stack_output,
        containment_method_output=args.containment_method_output,
        density_output=args.density_output,
        selected_threshold_method_key=args.selected_threshold_method_key,
        threshold_z=args.threshold_z,
        fixed_snr_threshold=args.fixed_snr_threshold,
        background_estimator=args.background_estimator,
        background_annulus_inner_px=args.background_annulus_inner_px,
        background_annulus_outer_px=args.background_annulus_outer_px,
        background_min_ring_pixels=args.background_min_ring_pixels,
        threshold_sample_per_plane=args.threshold_sample_per_plane,
        output_sample_per_stack=args.output_sample_per_stack,
        density_bins=args.density_bins,
        axis_scale_quantile=args.axis_scale_quantile,
        random_seed=args.random_seed,
        write_outputs=True,
    )
    print(f"[OK] wrote manifest to {res['manifest_output']}")
    print(f"[OK] wrote plane metrics to {res['plane_metrics_output']}")
    print(f"[OK] wrote thresholds to {res['threshold_output']}")
    print(f"[OK] wrote axis scales to {res['axis_scale_output']}")
    print(f"[OK] wrote plane summary to {res['plane_summary_output']}")
    print(f"[OK] wrote stack summary to {res['stack_summary_output']}")
    print(f"[OK] wrote sampled pixels to {res['pixel_sample_output']}")
    print(f"[OK] wrote plane containment to {res['containment_plane_output']}")
    print(f"[OK] wrote stack containment to {res['containment_stack_output']}")
    print(f"[OK] wrote containment-method summary to {res['containment_method_output']}")
    print(f"[OK] wrote density maps to {res['density_output']}")


if __name__ == "__main__":
    main()
