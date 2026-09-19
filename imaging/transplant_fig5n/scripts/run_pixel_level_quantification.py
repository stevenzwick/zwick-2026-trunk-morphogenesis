#!/usr/bin/env python3
"""Run first-pass DAPI-mask and pixel-level host/donor/FOXF1 quantification.

NOT runnable as shipped. This stage reads the raw microscopy images, which are not deposited —
they are available from the lead contact on request, as the paper's Data Availability Statement
states. Its outputs are committed under the lane's `derived/` directory, which is what every
figure script actually reads. See `data/DOWNLOAD.md`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

try:
    from .transplant_quantification_helpers import (
        CANONICAL_CHANNEL_ORDER,
        background_correct_signal,
        build_raw_input_manifest,
        channel_index,
        classify_host_donor_from_binary_calls,
        compute_dapi_floor,
        density_map_table,
        donor_adjacent_foxf1_component,
        donor_graft_component,
        half_gaussian_null_from_values,
        half_gaussian_threshold_from_values,
        largest_connected_component,
        load_transplant_stack,
        otsu_threshold_from_values,
        partition_mask_by_intensity_with_seed_components,
        pixel_class_label,
        reporter_over_dapi,
        sample_masked_coordinates,
        sample_masked_values,
        segment_whole_organoid_from_dapi_with_debug,
    )
except ImportError:
    from transplant_quantification_helpers import (
        CANONICAL_CHANNEL_ORDER,
        background_correct_signal,
        build_raw_input_manifest,
        channel_index,
        classify_host_donor_from_binary_calls,
        compute_dapi_floor,
        density_map_table,
        donor_adjacent_foxf1_component,
        donor_graft_component,
        half_gaussian_null_from_values,
        half_gaussian_threshold_from_values,
        largest_connected_component,
        load_transplant_stack,
        otsu_threshold_from_values,
        partition_mask_by_intensity_with_seed_components,
        pixel_class_label,
        reporter_over_dapi,
        sample_masked_coordinates,
        sample_masked_values,
        segment_whole_organoid_from_dapi_with_debug,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Raw input directory")
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
        help="Per-plane metrics TSV output",
    )
    parser.add_argument(
        "--normalization-output",
        type=Path,
        default=Path("results/tables/01b_condition_channel_normalization.tsv"),
        help="Condition/channel normalization summary TSV output",
    )
    parser.add_argument(
        "--normalization-hist-output",
        type=Path,
        default=Path("results/tables/01b_condition_channel_normalization_histograms.tsv"),
        help="Condition/channel normalization histogram TSV output",
    )
    parser.add_argument(
        "--threshold-output",
        type=Path,
        default=Path("results/tables/02_ratio_thresholds.tsv"),
        help="Ratio-threshold summary TSV output",
    )
    parser.add_argument(
        "--class-summary-output",
        type=Path,
        default=Path("results/tables/02_pixel_class_summary.tsv"),
        help="Per-plane and per-class summary TSV output",
    )
    parser.add_argument(
        "--foxf1-identity-output",
        type=Path,
        default=Path("results/tables/02_foxf1_positive_identity_summary.tsv"),
        help="FOXF1-positive identity summary TSV output",
    )
    parser.add_argument(
        "--pixel-sample-output",
        type=Path,
        default=Path("results/tables/02_sampled_pixel_profiles.tsv"),
        help="Sampled pixel profile TSV output",
    )
    parser.add_argument(
        "--density-output",
        type=Path,
        default=Path("results/tables/02_density_maps.tsv"),
        help="2D density-map TSV output",
    )
    parser.add_argument("--threshold-z", type=float, default=4.0, help="Half-Gaussian threshold sigma multiple")
    parser.add_argument(
        "--threshold-method",
        type=str,
        choices=["log1p_otsu", "half_gaussian"],
        default="log1p_otsu",
        help="Threshold method for reporter-over-DAPI ratio classification and positivity calls",
    )
    parser.add_argument(
        "--dominance-margin-log2",
        type=float,
        default=0.75,
        help="Minimum host-vs-donor log2 margin for dominant class calls in double-positive pixels",
    )
    parser.add_argument(
        "--threshold-sample-per-plane",
        type=int,
        default=50000,
        help="Maximum masked pixels sampled per plane for pooled threshold estimation",
    )
    parser.add_argument(
        "--normalization-sample-per-plane",
        type=int,
        default=50000,
        help="Maximum off-mask raw pixels sampled per plane for condition-level normalization estimation",
    )
    parser.add_argument(
        "--output-sample-per-plane",
        type=int,
        default=20000,
        help="Maximum masked pixels sampled per plane for plotting/export",
    )
    parser.add_argument("--density-bins", type=int, default=96, help="2D density-map bin count per axis")
    parser.add_argument("--random-seed", type=int, default=7, help="Random seed for pixel sampling")
    parser.add_argument(
        "--threshold-reference-condition",
        type=str,
        default="",
        help="If set, estimate global-scope thresholds using only this condition, then apply them to all conditions",
    )
    parser.add_argument(
        "--host-threshold-scope",
        type=str,
        choices=["global", "condition", "file"],
        default="global",
        help="Grouping scope used to estimate the host threshold",
    )
    parser.add_argument(
        "--donor-threshold-scope",
        type=str,
        choices=["global", "condition", "file"],
        default="global",
        help="Grouping scope used to estimate the donor threshold",
    )
    parser.add_argument(
        "--foxf1-threshold-scope",
        type=str,
        choices=["global", "condition", "file"],
        default="global",
        help="Grouping scope used to estimate the FOXF1 threshold",
    )
    parser.add_argument(
        "--mesp2-threshold-scope",
        type=str,
        choices=["global", "condition", "file"],
        default="global",
        help="Grouping scope used to estimate the MESP2 threshold",
    )
    parser.add_argument(
        "--host-threshold-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the pooled host threshold after estimation",
    )
    parser.add_argument(
        "--donor-threshold-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the pooled donor threshold after estimation",
    )
    parser.add_argument(
        "--foxf1-threshold-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the pooled FOXF1 threshold after estimation",
    )
    parser.add_argument(
        "--foxf1-threshold-condition-scale",
        action="append",
        default=[],
        help=(
            "Optional extra FOXF1 threshold multiplier by condition, formatted as CONDITION=SCALE. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--foxf1-threshold-condition-value",
        action="append",
        default=[],
        help=(
            "Optional absolute FOXF1 threshold override by condition, formatted as CONDITION=VALUE. "
            "May be passed multiple times. When set, this overrides the estimated/scaled FOXF1 threshold "
            "for that condition."
        ),
    )
    parser.add_argument(
        "--foxf1-threshold-file-scale",
        action="append",
        default=[],
        help=(
            "Optional extra FOXF1 threshold multiplier by file_id, formatted as FILE_ID=SCALE. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--mesp2-threshold-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the pooled MESP2 threshold after estimation",
    )
    parser.add_argument(
        "--foxf1-host-seed-threshold-scale",
        type=float,
        default=1.15,
        help="Extra multiplier applied to the host threshold when defining the contiguous host seed used for FOXF1-domain lineage partitioning",
    )
    return parser.parse_args()


def resolve_path(root: Path, path_like: Path) -> Path:
    return path_like.resolve() if path_like.is_absolute() else (root / path_like).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_scale_mapping(items: list[str] | tuple[str, ...] | None) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for item in items or []:
        text = str(item).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(
                f"Scale mapping entries must be formatted as KEY=VALUE; received {text!r}"
            )
        key, value = text.split("=", 1)
        key = str(key).strip()
        if not key:
            raise ValueError(f"Scale mapping key was empty in {text!r}")
        mapping[key] = float(value)
    return mapping


def run_quantification_pipeline(
    root: Path | str = ".",
    data_dir: Path | str = "data",
    manifest_output: Path | str = "results/manifests/raw_input_manifest.tsv",
    mask_dir: Path | str = "results/masks/dapi_whole_organoid_masks",
    plane_metrics_output: Path | str = "results/tables/01_mask_and_plane_metrics.tsv",
    normalization_output: Path | str = "results/tables/01b_condition_channel_normalization.tsv",
    normalization_hist_output: Path | str = "results/tables/01b_condition_channel_normalization_histograms.tsv",
    threshold_output: Path | str = "results/tables/02_ratio_thresholds.tsv",
    class_summary_output: Path | str = "results/tables/02_pixel_class_summary.tsv",
    foxf1_identity_output: Path | str = "results/tables/02_foxf1_positive_identity_summary.tsv",
    pixel_sample_output: Path | str = "results/tables/02_sampled_pixel_profiles.tsv",
    density_output: Path | str = "results/tables/02_density_maps.tsv",
    threshold_z: float = 4.0,
    threshold_method: str = "log1p_otsu",
    dominance_margin_log2: float = 0.75,
    threshold_sample_per_plane: int = 50000,
    normalization_sample_per_plane: int = 50000,
    output_sample_per_plane: int = 20000,
    density_bins: int = 96,
    random_seed: int = 7,
    threshold_reference_condition: str | None = None,
    host_threshold_scope: str = "global",
    donor_threshold_scope: str = "global",
    foxf1_threshold_scope: str = "global",
    mesp2_threshold_scope: str = "global",
    host_threshold_scale: float = 1.0,
    donor_threshold_scale: float = 1.0,
    foxf1_threshold_scale: float = 1.0,
    mesp2_threshold_scale: float = 1.0,
    foxf1_threshold_condition_scales: dict[str, float] | None = None,
    foxf1_threshold_condition_values: dict[str, float] | None = None,
    foxf1_threshold_file_scales: dict[int, float] | None = None,
    foxf1_component_radius_px: float = 80.0,
    foxf1_host_seed_threshold_scale: float = 1.15,
    write_outputs: bool = True,
) -> dict[str, object]:
    root = Path(root).resolve()
    data_dir = resolve_path(root, Path(data_dir))
    manifest_output = resolve_path(root, Path(manifest_output))
    mask_dir = resolve_path(root, Path(mask_dir))
    plane_metrics_output = resolve_path(root, Path(plane_metrics_output))
    normalization_output = resolve_path(root, Path(normalization_output))
    normalization_hist_output = resolve_path(root, Path(normalization_hist_output))
    threshold_output = resolve_path(root, Path(threshold_output))
    class_summary_output = resolve_path(root, Path(class_summary_output))
    foxf1_identity_output = resolve_path(root, Path(foxf1_identity_output))
    pixel_sample_output = resolve_path(root, Path(pixel_sample_output))
    density_output = resolve_path(root, Path(density_output))

    ensure_parent(manifest_output)
    ensure_parent(plane_metrics_output)
    ensure_parent(normalization_output)
    ensure_parent(normalization_hist_output)
    ensure_parent(threshold_output)
    ensure_parent(class_summary_output)
    ensure_parent(foxf1_identity_output)
    ensure_parent(pixel_sample_output)
    ensure_parent(density_output)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(random_seed))
    threshold_reference_condition = (
        str(threshold_reference_condition).strip()
        if threshold_reference_condition is not None and str(threshold_reference_condition).strip() != ""
        else None
    )

    manifest_df = build_raw_input_manifest(data_dir=data_dir, root=root)
    if manifest_df.empty:
        # An empty manifest has no "condition" column, so every downstream reference to it raises
        # a bare KeyError from pandas internals rather than naming the real problem: no raw images
        # were found. Fail here, where the message can say where it looked.
        raise SystemExit(
            f"No raw transplant images found under {data_dir}\n"
            "The raw acquisitions are not in this repository; see data/DOWNLOAD.md for what to\n"
            "fetch and where to put it, then re-run with --data_dir pointing at it.")
    if threshold_reference_condition is not None:
        available_conditions = set(manifest_df["condition"].astype(str).tolist())
        if str(threshold_reference_condition) not in available_conditions:
            raise ValueError(
                f"threshold_reference_condition={threshold_reference_condition!r} was not found in manifest conditions {sorted(available_conditions)!r}"
            )
    if bool(write_outputs):
        manifest_df.to_csv(manifest_output, sep="\t", index=False)

    threshold_scope_lookup = {
        "host": str(host_threshold_scope),
        "donor": str(donor_threshold_scope),
        "foxf1": str(foxf1_threshold_scope),
        "mesp2": str(mesp2_threshold_scope),
    }
    normalization_channels = ["dapi", "mesp2", "foxf1", "donor", "host"]
    normalization_samples: dict[str, dict[str, list[np.ndarray]]] = {
        key: defaultdict(list) for key in normalization_channels
    }
    plane_records: list[dict[str, Any]] = []

    def threshold_group_key(channel_key: str, row: Any) -> str | None:
        scope = str(threshold_scope_lookup[channel_key])
        if scope == "global":
            if threshold_reference_condition is not None and str(row.condition) != str(threshold_reference_condition):
                return None
            return "global"
        if scope == "condition":
            return str(row.condition)
        if scope == "file":
            return str(int(row.file_id))
        raise ValueError(f"Unsupported threshold scope {scope!r} for channel {channel_key!r}")

    for row in manifest_df.itertuples(index=False):
        stack = load_transplant_stack(resolve_path(root, Path(str(row.file_path))))
        if int(stack.data_czyx.shape[0]) < len(CANONICAL_CHANNEL_ORDER):
            raise RuntimeError(
                f"{row.file_name} has only {stack.data_czyx.shape[0]} channels; expected at least {len(CANONICAL_CHANNEL_ORDER)}"
            )

        dapi_idx = channel_index(stack.canonical_channel_names, "dapi")
        mesp2_idx = channel_index(stack.canonical_channel_names, "mesp2")
        foxf1_idx = channel_index(stack.canonical_channel_names, "foxf1")
        donor_idx = channel_index(stack.canonical_channel_names, "donor")
        host_idx = channel_index(stack.canonical_channel_names, "host")

        for z_index in range(int(stack.data_czyx.shape[1])):
            dapi_raw = np.asarray(stack.data_czyx[dapi_idx, z_index], dtype=np.float32)
            mask, debug = segment_whole_organoid_from_dapi_with_debug(dapi_raw)
            abs_mask_path = mask_dir / f"{int(row.file_id):02d}_z{int(z_index):02d}_mask.tif"
            rel_mask_path = abs_mask_path.resolve().relative_to(root.resolve())
            tifffile.imwrite(abs_mask_path, mask.astype(np.uint8))

            plane_row = {
                "image_id": str(row.image_id),
                "file_id": int(row.file_id),
                "condition": str(row.condition),
                "position_label": str(row.position_label),
                "file_name": str(row.file_name),
                "file_path": str(row.file_path),
                "source_format": str(row.source_format),
                "z_index": int(z_index),
                "mask_path": rel_mask_path.as_posix(),
                "mask_area_px": int(np.sum(mask)),
                "mask_fraction": float(np.mean(mask)),
                "baseline_value": float(debug["baseline_value"]),
                "raw_threshold": float(debug["raw_threshold"]),
                "threshold_value": float(debug["threshold_value"]),
                "selected_variant": str(debug["selected_variant"]),
                "n_raw_components": int(debug["n_raw_components"]),
            }
            plane_records.append(plane_row)

            off_mask = ~mask
            if int(np.sum(off_mask)) == 0:
                continue

            for channel_key, channel_idx in [
                ("dapi", dapi_idx),
                ("mesp2", mesp2_idx),
                ("foxf1", foxf1_idx),
                ("donor", donor_idx),
                ("host", host_idx),
            ]:
                raw = np.asarray(stack.data_czyx[channel_idx, z_index], dtype=np.float32)
                sampled_bg = sample_masked_values(
                    values=raw,
                    mask=off_mask,
                    sample_size=int(normalization_sample_per_plane),
                    rng=rng,
                )
                if sampled_bg.size:
                    normalization_samples[channel_key][str(row.condition)].append(sampled_bg.astype(np.float32))

    plane_metrics_df = pd.DataFrame(plane_records).sort_values(["condition", "file_id", "z_index"]).reset_index(drop=True)
    normalization_rows: list[dict[str, Any]] = []
    normalization_hist_rows: list[dict[str, Any]] = []
    normalization_lookup: dict[str, dict[str, dict[str, float]]] = {}
    for condition in sorted(manifest_df["condition"].astype(str).dropna().unique().tolist()):
        normalization_lookup[str(condition)] = {}
        for channel_key in normalization_channels:
            pooled = (
                np.concatenate(normalization_samples[channel_key][str(condition)]).astype(np.float32)
                if normalization_samples[channel_key][str(condition)]
                else np.zeros((0,), dtype=np.float32)
            )
            if pooled.size == 0:
                raise ValueError(
                    f"No off-mask raw pixels were available for normalization in condition={condition!r}, channel={channel_key!r}."
                )
            diag = half_gaussian_null_from_values(pooled)
            background_mu = float(diag["null_mu"])
            background_sigma = float(max(float(diag["null_sigma"]), 1.0e-6))
            normalization_lookup[str(condition)][str(channel_key)] = {
                "mu": background_mu,
                "sigma": background_sigma,
            }
            normalization_rows.append(
                {
                    "condition": str(condition),
                    "channel_key": str(channel_key),
                    "background_mu": background_mu,
                    "background_sigma": background_sigma,
                    "sample_count": int(diag["sample_count"]),
                    "baseline_location": float(diag["baseline_location"]),
                    "baseline_scale": float(diag["baseline_scale"]),
                    "method": str(diag.get("method", "half_gaussian_left_null")),
                }
            )
            hist_edges = np.asarray(diag["hist_edges"], dtype=np.float64)
            hist_centers = np.asarray(diag["hist_centers"], dtype=np.float64)
            hist_counts = np.asarray(diag["hist_counts"], dtype=np.float64)
            smooth_counts = np.asarray(diag["smooth_counts"], dtype=np.float64)
            mode_index = int(diag.get("mode_index", int(np.nanargmax(smooth_counts))))
            peak_height = float(max(smooth_counts[mode_index], 1.0))
            null_fit_counts = peak_height * np.exp(
                -0.5 * ((hist_centers - background_mu) / max(background_sigma, 1.0e-6)) ** 2
            )
            for idx, center in enumerate(hist_centers):
                normalization_hist_rows.append(
                    {
                        "condition": str(condition),
                        "channel_key": str(channel_key),
                        "bin_index": int(idx),
                        "bin_start": float(hist_edges[idx]),
                        "bin_end": float(hist_edges[idx + 1]),
                        "bin_center": float(center),
                        "hist_count": float(hist_counts[idx]),
                        "smooth_count": float(smooth_counts[idx]),
                        "null_fit_count": float(null_fit_counts[idx]),
                        "background_mu": background_mu,
                        "background_sigma": background_sigma,
                        "mode_index": mode_index,
                        "sample_count": int(diag["sample_count"]),
                    }
                )
    normalization_df = pd.DataFrame(normalization_rows).sort_values(["condition", "channel_key"]).reset_index(drop=True)
    normalization_hist_df = pd.DataFrame(normalization_hist_rows).sort_values(
        ["condition", "channel_key", "bin_index"]
    ).reset_index(drop=True)
    if bool(write_outputs):
        normalization_df.to_csv(normalization_output, sep="\t", index=False)
        normalization_hist_df.to_csv(normalization_hist_output, sep="\t", index=False)

    threshold_samples: dict[str, dict[str, list[np.ndarray]]] = {
        key: defaultdict(list) for key in ["mesp2", "foxf1", "donor", "host"]
    }
    plane_quant_rows: list[dict[str, Any]] = []
    for row in manifest_df.itertuples(index=False):
        stack = load_transplant_stack(resolve_path(root, Path(str(row.file_path))))
        dapi_idx = channel_index(stack.canonical_channel_names, "dapi")
        mesp2_idx = channel_index(stack.canonical_channel_names, "mesp2")
        foxf1_idx = channel_index(stack.canonical_channel_names, "foxf1")
        donor_idx = channel_index(stack.canonical_channel_names, "donor")
        host_idx = channel_index(stack.canonical_channel_names, "host")

        for z_index in range(int(stack.data_czyx.shape[1])):
            mask_path = resolve_path(root, mask_dir / f"{int(row.file_id):02d}_z{int(z_index):02d}_mask.tif")
            mask = tifffile.imread(mask_path).astype(bool)
            dapi_raw = np.asarray(stack.data_czyx[dapi_idx, z_index], dtype=np.float32)
            dapi_corrected, dapi_bg_value, _ = background_correct_signal(
                signal=dapi_raw,
                organoid_mask=mask,
                background_estimator="whole_off_organoid",
            )
            dapi_floor = compute_dapi_floor(dapi_corrected, mask)

            plane_row = {
                "condition": str(row.condition),
                "file_id": int(row.file_id),
                "z_index": int(z_index),
                "dapi_background_value": float(dapi_bg_value),
                "dapi_background_sigma": np.nan,
                "dapi_floor": float(dapi_floor),
            }

            if int(np.sum(mask)) == 0:
                plane_quant_rows.append(plane_row)
                continue

            for channel_key, channel_idx in [
                ("mesp2", mesp2_idx),
                ("foxf1", foxf1_idx),
                ("donor", donor_idx),
                ("host", host_idx),
            ]:
                raw = np.asarray(stack.data_czyx[channel_idx, z_index], dtype=np.float32)
                corrected, background_value, _ = background_correct_signal(
                    signal=raw,
                    organoid_mask=mask,
                    background_estimator="whole_off_organoid",
                )
                ratio = reporter_over_dapi(corrected, dapi_corrected, mask, dapi_floor=dapi_floor)
                group_key = threshold_group_key(channel_key, row)
                if group_key is not None:
                    sampled = sample_masked_values(
                        values=ratio,
                        mask=mask,
                        sample_size=int(threshold_sample_per_plane),
                        rng=rng,
                    )
                    if sampled.size:
                        threshold_samples[channel_key][str(group_key)].append(sampled.astype(np.float32))
                plane_row[f"{channel_key}_background_value"] = float(background_value)
                plane_row[f"{channel_key}_background_sigma"] = np.nan
                plane_row[f"{channel_key}_ratio_mean"] = float(np.nanmean(ratio[mask]))
                plane_row[f"{channel_key}_ratio_median"] = float(np.nanmedian(ratio[mask]))
            plane_quant_rows.append(plane_row)

    plane_quant_df = pd.DataFrame(plane_quant_rows)
    plane_metrics_df = plane_metrics_df.merge(
        plane_quant_df,
        on=["condition", "file_id", "z_index"],
        how="left",
    )
    if bool(write_outputs):
        plane_metrics_df.to_csv(plane_metrics_output, sep="\t", index=False)

    threshold_rows: list[dict[str, Any]] = []
    threshold_scale_lookup = {
        "host": float(host_threshold_scale),
        "donor": float(donor_threshold_scale),
        "foxf1": float(foxf1_threshold_scale),
        "mesp2": float(mesp2_threshold_scale),
    }
    foxf1_threshold_condition_scales = {
        str(key): float(value)
        for key, value in dict(foxf1_threshold_condition_scales or {}).items()
    }
    foxf1_threshold_condition_values = {
        str(key): float(value)
        for key, value in dict(foxf1_threshold_condition_values or {}).items()
    }
    foxf1_threshold_file_scales = {
        int(key): float(value)
        for key, value in dict(foxf1_threshold_file_scales or {}).items()
    }
    threshold_lookup: dict[str, dict[str, float]] = {key: {} for key in ["mesp2", "foxf1", "donor", "host"]}
    for channel_key in ["mesp2", "foxf1", "donor", "host"]:
        scope = str(threshold_scope_lookup[channel_key])
        for group_key in sorted(threshold_samples[channel_key].keys(), key=str):
            pooled = (
                np.concatenate(threshold_samples[channel_key][group_key]).astype(np.float32)
                if threshold_samples[channel_key][group_key]
                else np.zeros((0,), dtype=np.float32)
            )
            if str(threshold_method) == "log1p_otsu":
                diag = otsu_threshold_from_values(pooled, transform="log1p")
                method_name = str(diag["method"])
            else:
                diag = half_gaussian_threshold_from_values(
                    pooled,
                    z=float(threshold_z),
                )
                method_name = "half_gaussian_pooled_masked_ratio_sample"
            base_threshold_value = float(diag["threshold_value"])
            threshold_scale = float(threshold_scale_lookup[channel_key])

            threshold_group_condition = ""
            threshold_group_file_id = np.nan
            if scope == "condition":
                threshold_group_condition = str(group_key)
            elif scope == "file":
                threshold_group_file_id = int(group_key)
                manifest_match = manifest_df.loc[manifest_df["file_id"] == int(group_key)]
                if not manifest_match.empty:
                    threshold_group_condition = str(manifest_match["condition"].iloc[0])

            manual_scale_factor = 1.0
            if channel_key == "foxf1":
                manual_scale_factor = float(
                    foxf1_threshold_condition_scales.get(str(threshold_group_condition), 1.0)
                )
                if np.isfinite(threshold_group_file_id):
                    manual_scale_factor *= float(
                        foxf1_threshold_file_scales.get(int(threshold_group_file_id), 1.0)
                    )
            effective_threshold_scale = float(threshold_scale * manual_scale_factor)
            scaled_threshold_value = base_threshold_value * effective_threshold_scale
            threshold_override_mode = "none"
            absolute_override_value = np.nan
            if channel_key == "foxf1" and str(threshold_group_condition) in foxf1_threshold_condition_values:
                absolute_override_value = float(foxf1_threshold_condition_values[str(threshold_group_condition)])
                scaled_threshold_value = absolute_override_value
                if np.isfinite(base_threshold_value) and float(base_threshold_value) > 0:
                    effective_threshold_scale = float(scaled_threshold_value / float(base_threshold_value))
                    if np.isfinite(threshold_scale) and float(threshold_scale) > 0:
                        manual_scale_factor = float(effective_threshold_scale / float(threshold_scale))
                    else:
                        manual_scale_factor = np.nan
                else:
                    effective_threshold_scale = np.nan
                    manual_scale_factor = np.nan
                threshold_override_mode = "condition_absolute"
            threshold_lookup[channel_key][str(group_key)] = scaled_threshold_value

            threshold_rows.append(
                {
                    "channel_key": channel_key,
                    "threshold_scope": scope,
                    "threshold_group": str(group_key),
                    "threshold_group_condition": threshold_group_condition,
                    "threshold_group_file_id": threshold_group_file_id,
                    "threshold_value": scaled_threshold_value,
                    "base_threshold_value": base_threshold_value,
                    "threshold_scale": effective_threshold_scale,
                    "base_threshold_scale": threshold_scale,
                    "manual_scale_factor": manual_scale_factor,
                    "threshold_override_mode": threshold_override_mode,
                    "absolute_override_value": absolute_override_value,
                    "baseline_location": float(diag.get("baseline_location", np.nan)),
                    "baseline_scale": float(diag.get("baseline_scale", np.nan)),
                    "threshold_z": float(diag.get("threshold_z", np.nan)),
                    "sample_count": int(diag["sample_count"]),
                    "method": method_name,
                    "threshold_reference_condition": (
                        str(threshold_reference_condition) if threshold_reference_condition is not None else "all_conditions"
                    ),
                }
            )
    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        [
            "channel_key",
            "threshold_scope",
            "threshold_group_condition",
            "threshold_group_file_id",
            "threshold_group",
        ]
    ).reset_index(drop=True)
    if bool(write_outputs):
        threshold_df.to_csv(threshold_output, sep="\t", index=False)

    def resolve_threshold(channel_key: str, row: Any) -> float:
        scope = str(threshold_scope_lookup[channel_key])
        if scope == "global":
            group_key = "global"
        elif scope == "condition":
            group_key = str(row.condition)
        elif scope == "file":
            group_key = str(int(row.file_id))
        else:
            raise ValueError(f"Unsupported threshold scope {scope!r} for channel {channel_key!r}")
        return float(threshold_lookup[channel_key][str(group_key)])

    class_rows: list[dict[str, Any]] = []
    foxf1_identity_rows: list[dict[str, Any]] = []
    sampled_pixel_rows: list[dict[str, Any]] = []

    for row in manifest_df.itertuples(index=False):
        stack = load_transplant_stack(resolve_path(root, Path(str(row.file_path))))
        dapi_idx = channel_index(stack.canonical_channel_names, "dapi")
        mesp2_idx = channel_index(stack.canonical_channel_names, "mesp2")
        foxf1_idx = channel_index(stack.canonical_channel_names, "foxf1")
        donor_idx = channel_index(stack.canonical_channel_names, "donor")
        host_idx = channel_index(stack.canonical_channel_names, "host")

        for z_index in range(int(stack.data_czyx.shape[1])):
            metric_row = plane_metrics_df.loc[
                (plane_metrics_df["file_id"] == int(row.file_id))
                & (plane_metrics_df["z_index"] == int(z_index))
            ].iloc[0]
            mask_path = resolve_path(root, Path(str(metric_row["mask_path"])))
            mask = tifffile.imread(mask_path).astype(bool)
            if int(np.sum(mask)) == 0:
                continue

            dapi_raw = np.asarray(stack.data_czyx[dapi_idx, z_index], dtype=np.float32)
            dapi_corrected, _, _ = background_correct_signal(
                signal=dapi_raw,
                organoid_mask=mask,
                background_estimator="whole_off_organoid",
            )
            dapi_floor = float(metric_row["dapi_floor"])

            ratios: dict[str, np.ndarray] = {}
            for channel_key, channel_idx in [
                ("mesp2", mesp2_idx),
                ("foxf1", foxf1_idx),
                ("donor", donor_idx),
                ("host", host_idx),
            ]:
                raw = np.asarray(stack.data_czyx[channel_idx, z_index], dtype=np.float32)
                corrected, _, _ = background_correct_signal(
                    signal=raw,
                    organoid_mask=mask,
                    background_estimator="whole_off_organoid",
                )
                ratios[channel_key] = reporter_over_dapi(corrected, dapi_corrected, mask, dapi_floor=dapi_floor)

            host_threshold_value = resolve_threshold("host", row)
            donor_threshold_value = resolve_threshold("donor", row)
            foxf1_threshold_value = resolve_threshold("foxf1", row)
            mesp2_threshold_value = resolve_threshold("mesp2", row)

            host_positive = mask & np.isfinite(ratios["host"]) & (ratios["host"] > float(host_threshold_value))
            donor_positive_raw = mask & np.isfinite(ratios["donor"]) & (ratios["donor"] > float(donor_threshold_value))
            donor_positive = donor_graft_component(
                donor_positive=donor_positive_raw,
                organoid_mask=mask,
                fill_holes=True,
            )
            classes = classify_host_donor_from_binary_calls(
                host_positive=host_positive,
                donor_positive=donor_positive,
                host_ratio=ratios["host"],
                donor_ratio=ratios["donor"],
                dominance_margin_log2=float(dominance_margin_log2),
                organoid_mask=mask,
            )
            foxf1_positive_raw = mask & np.isfinite(ratios["foxf1"]) & (ratios["foxf1"] > float(foxf1_threshold_value))
            foxf1_positive, foxf1_donor_halo = donor_adjacent_foxf1_component(
                foxf1_positive=foxf1_positive_raw,
                donor_positive=donor_positive,
                organoid_mask=mask,
                radius_px=float(foxf1_component_radius_px),
            )
            host_seed_threshold = float(host_threshold_value) * float(foxf1_host_seed_threshold_scale)
            host_positive_seed_raw = (
                mask
                & np.isfinite(ratios["host"])
                & (ratios["host"] > host_seed_threshold)
                & (~donor_positive)
            )
            host_positive_seed = largest_connected_component(host_positive_seed_raw)
            if np.any(host_positive_seed):
                host_positive_seed = (ndi.binary_fill_holes(host_positive_seed) & mask & (~donor_positive)).astype(bool)
            if not np.any(host_positive_seed):
                host_score_fallback = np.asarray(ratios["host"], dtype=np.float32) / max(float(host_threshold_value), 1e-6)
                available = mask & (~donor_positive) & np.isfinite(host_score_fallback)
                if np.any(available):
                    host_positive_seed = np.zeros_like(mask, dtype=bool)
                    host_positive_seed.flat[int(np.nanargmax(np.where(available, host_score_fallback, -np.inf)))] = True
            donor_score = np.asarray(ratios["donor"], dtype=np.float32) / max(float(donor_threshold_value), 1e-6)
            host_score = np.asarray(ratios["host"], dtype=np.float32) / max(float(host_threshold_value), 1e-6)
            donor_partition_full, host_partition_full = partition_mask_by_intensity_with_seed_components(
                mask=mask,
                primary_seed=donor_positive,
                secondary_seed=host_positive_seed,
                primary_score=donor_score,
                secondary_score=host_score,
            )
            foxf1_partition_donor = foxf1_positive & donor_partition_full
            foxf1_partition_host = foxf1_positive & host_partition_full
            total_mask_pixels = int(np.sum(mask))

            foxf1_host_strict = foxf1_positive & host_positive & (~donor_positive)
            foxf1_donor_strict = foxf1_positive & donor_positive & (~host_positive)
            foxf1_mixed = foxf1_positive & host_positive & donor_positive
            foxf1_unlabeled = foxf1_positive & (~host_positive) & (~donor_positive)
            foxf1_host_class = foxf1_positive & (classes == 1)
            foxf1_donor_class = foxf1_positive & (classes == 2)
            foxf1_host_partition = foxf1_partition_host
            foxf1_donor_partition = foxf1_partition_donor
            foxf1_positive_pixels = int(np.sum(foxf1_positive))
            foxf1_host_strict_pixels = int(np.sum(foxf1_host_strict))
            foxf1_donor_strict_pixels = int(np.sum(foxf1_donor_strict))
            foxf1_mixed_pixels = int(np.sum(foxf1_mixed))
            foxf1_unlabeled_pixels = int(np.sum(foxf1_unlabeled))
            foxf1_host_class_pixels = int(np.sum(foxf1_host_class))
            foxf1_donor_class_pixels = int(np.sum(foxf1_donor_class))
            foxf1_host_partition_pixels = int(np.sum(foxf1_host_partition))
            foxf1_donor_partition_pixels = int(np.sum(foxf1_donor_partition))

            def _frac(part: int, whole: int) -> float:
                return float(part / whole) if int(whole) > 0 else np.nan

            foxf1_values = ratios["foxf1"][foxf1_positive]
            host_values_in_foxf1 = ratios["host"][foxf1_positive]
            donor_values_in_foxf1 = ratios["donor"][foxf1_positive]
            foxf1_identity_rows.append(
                {
                    "summary_level": "plane",
                    "image_id": str(row.image_id),
                    "file_id": int(row.file_id),
                    "condition": str(row.condition),
                    "position_label": str(row.position_label),
                    "file_name": str(row.file_name),
                    "z_index": int(z_index),
                    "mask_pixels": total_mask_pixels,
                    "foxf1_positive_pixels": foxf1_positive_pixels,
                    "foxf1_positive_fraction_of_mask": _frac(foxf1_positive_pixels, total_mask_pixels),
                    "foxf1_positive_host_strict_pixels": foxf1_host_strict_pixels,
                    "foxf1_positive_donor_strict_pixels": foxf1_donor_strict_pixels,
                    "foxf1_positive_mixed_pixels": foxf1_mixed_pixels,
                    "foxf1_positive_unlabeled_pixels": foxf1_unlabeled_pixels,
                    "foxf1_positive_host_class_pixels": foxf1_host_class_pixels,
                    "foxf1_positive_donor_class_pixels": foxf1_donor_class_pixels,
                    "foxf1_positive_host_partition_pixels": foxf1_host_partition_pixels,
                    "foxf1_positive_donor_partition_pixels": foxf1_donor_partition_pixels,
                    "foxf1_positive_host_strict_fraction": _frac(foxf1_host_strict_pixels, foxf1_positive_pixels),
                    "foxf1_positive_donor_strict_fraction": _frac(foxf1_donor_strict_pixels, foxf1_positive_pixels),
                    "foxf1_positive_mixed_fraction": _frac(foxf1_mixed_pixels, foxf1_positive_pixels),
                    "foxf1_positive_unlabeled_fraction": _frac(foxf1_unlabeled_pixels, foxf1_positive_pixels),
                    "foxf1_positive_host_class_fraction": _frac(foxf1_host_class_pixels, foxf1_positive_pixels),
                    "foxf1_positive_donor_class_fraction": _frac(foxf1_donor_class_pixels, foxf1_positive_pixels),
                    "foxf1_positive_host_partition_fraction": _frac(foxf1_host_partition_pixels, foxf1_positive_pixels),
                    "foxf1_positive_donor_partition_fraction": _frac(foxf1_donor_partition_pixels, foxf1_positive_pixels),
                    "foxf1_ratio_mean_in_foxf1_positive": float(np.nanmean(foxf1_values)) if foxf1_positive_pixels else np.nan,
                    "host_ratio_mean_in_foxf1_positive": float(np.nanmean(host_values_in_foxf1)) if foxf1_positive_pixels else np.nan,
                    "donor_ratio_mean_in_foxf1_positive": float(np.nanmean(donor_values_in_foxf1)) if foxf1_positive_pixels else np.nan,
                    "foxf1_threshold": float(foxf1_threshold_value),
                    "host_threshold": float(host_threshold_value),
                    "foxf1_host_seed_threshold": float(host_seed_threshold),
                    "donor_threshold": float(donor_threshold_value),
                    "foxf1_region_mode": "donor_adjacent_component",
                    "foxf1_lineage_mode": "full_mask_intensity_seed_partition",
                    "lineage_partition_mode": "full_mask_intensity_seed_partition",
                    "foxf1_host_seed_threshold_scale": float(foxf1_host_seed_threshold_scale),
                    "foxf1_component_radius_px": float(foxf1_component_radius_px),
                    "foxf1_raw_positive_pixels": int(np.sum(foxf1_positive_raw)),
                    "donor_raw_positive_pixels": int(np.sum(donor_positive_raw)),
                    "donor_graft_pixels": int(np.sum(donor_positive)),
                    "host_lineage_pixels": int(np.sum(host_partition_full)),
                    "donor_lineage_pixels": int(np.sum(donor_partition_full)),
                    "host_seed_pixels": int(np.sum(host_positive_seed)),
                    "foxf1_donor_halo_pixels": int(np.sum(foxf1_donor_halo)),
                }
            )

            for class_code, class_label in sorted({int(v): pixel_class_label(v) for v in [0, 1, 2, 3]}.items()):
                class_mask = mask & (classes == int(class_code))
                count = int(np.sum(class_mask))
                if count == 0:
                    foxf1_positive_fraction = np.nan
                    foxf1_mean = np.nan
                    foxf1_median = np.nan
                    host_mean = np.nan
                    donor_mean = np.nan
                    mesp2_mean = np.nan
                else:
                    foxf1_positive_fraction = float(np.mean(foxf1_positive[class_mask]))
                    foxf1_mean = float(np.nanmean(ratios["foxf1"][class_mask]))
                    foxf1_median = float(np.nanmedian(ratios["foxf1"][class_mask]))
                    host_mean = float(np.nanmean(ratios["host"][class_mask]))
                    donor_mean = float(np.nanmean(ratios["donor"][class_mask]))
                    mesp2_mean = float(np.nanmean(ratios["mesp2"][class_mask]))
                class_rows.append(
                    {
                        "image_id": str(row.image_id),
                        "file_id": int(row.file_id),
                        "condition": str(row.condition),
                        "position_label": str(row.position_label),
                        "file_name": str(row.file_name),
                        "z_index": int(z_index),
                        "pixel_class_code": int(class_code),
                        "pixel_class_label": str(class_label),
                        "pixel_count": count,
                        "fraction_of_mask": float(count / total_mask_pixels) if total_mask_pixels > 0 else np.nan,
                        "foxf1_positive_fraction": foxf1_positive_fraction,
                        "foxf1_ratio_mean": foxf1_mean,
                        "foxf1_ratio_median": foxf1_median,
                        "host_ratio_mean": host_mean,
                        "donor_ratio_mean": donor_mean,
                        "mesp2_ratio_mean": mesp2_mean,
                        "foxf1_threshold": float(foxf1_threshold_value),
                        "host_threshold": float(host_threshold_value),
                        "donor_threshold": float(donor_threshold_value),
                        "mesp2_threshold": float(mesp2_threshold_value),
                    }
                )

            sample_coords = sample_masked_coordinates(
                mask=mask,
                sample_size=int(output_sample_per_plane),
                rng=rng,
            )
            for rc in sample_coords:
                row_idx = int(rc[0])
                col_idx = int(rc[1])
                class_code = int(classes[row_idx, col_idx])
                sampled_pixel_rows.append(
                    {
                        "image_id": str(row.image_id),
                        "file_id": int(row.file_id),
                        "condition": str(row.condition),
                        "position_label": str(row.position_label),
                        "file_name": str(row.file_name),
                        "z_index": int(z_index),
                        "row_px": row_idx,
                        "col_px": col_idx,
                        "pixel_class_code": class_code,
                        "pixel_class_label": pixel_class_label(class_code),
                        "mesp2_ratio": float(ratios["mesp2"][row_idx, col_idx]),
                        "foxf1_ratio": float(ratios["foxf1"][row_idx, col_idx]),
                        "donor_ratio": float(ratios["donor"][row_idx, col_idx]),
                        "host_ratio": float(ratios["host"][row_idx, col_idx]),
                        "foxf1_positive": bool(foxf1_positive[row_idx, col_idx]),
                    }
                )

    class_summary_df = pd.DataFrame(class_rows).sort_values(
        ["condition", "file_id", "z_index", "pixel_class_code"]
    ).reset_index(drop=True)
    if bool(write_outputs):
        class_summary_df.to_csv(class_summary_output, sep="\t", index=False)

    foxf1_identity_plane_df = pd.DataFrame(foxf1_identity_rows).sort_values(
        ["condition", "file_id", "z_index"]
    ).reset_index(drop=True)

    identity_metric_cols = [
        "mask_pixels",
        "foxf1_positive_pixels",
        "foxf1_positive_host_strict_pixels",
        "foxf1_positive_donor_strict_pixels",
        "foxf1_positive_mixed_pixels",
        "foxf1_positive_unlabeled_pixels",
        "foxf1_positive_host_class_pixels",
        "foxf1_positive_donor_class_pixels",
        "foxf1_positive_host_partition_pixels",
        "foxf1_positive_donor_partition_pixels",
        "foxf1_raw_positive_pixels",
        "donor_raw_positive_pixels",
        "donor_graft_pixels",
        "host_lineage_pixels",
        "donor_lineage_pixels",
        "host_seed_pixels",
        "foxf1_donor_halo_pixels",
    ]

    file_identity_rows: list[dict[str, Any]] = []
    for file_id, sub in foxf1_identity_plane_df.groupby("file_id", sort=True):
        row0 = sub.iloc[0]
        agg_counts = {col: int(sub[col].sum()) for col in identity_metric_cols}
        foxf1_total = int(agg_counts["foxf1_positive_pixels"])
        mask_total = int(agg_counts["mask_pixels"])
        file_identity_rows.append(
            {
                "summary_level": "file",
                "image_id": str(row0["image_id"]),
                "file_id": int(file_id),
                "condition": str(row0["condition"]),
                "position_label": str(row0["position_label"]),
                "file_name": str(row0["file_name"]),
                "z_index": np.nan,
                **agg_counts,
                "foxf1_positive_fraction_of_mask": float(foxf1_total / mask_total) if mask_total > 0 else np.nan,
                "foxf1_positive_host_strict_fraction": float(agg_counts["foxf1_positive_host_strict_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_donor_strict_fraction": float(agg_counts["foxf1_positive_donor_strict_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_mixed_fraction": float(agg_counts["foxf1_positive_mixed_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_unlabeled_fraction": float(agg_counts["foxf1_positive_unlabeled_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_host_class_fraction": float(agg_counts["foxf1_positive_host_class_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_donor_class_fraction": float(agg_counts["foxf1_positive_donor_class_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_host_partition_fraction": float(agg_counts["foxf1_positive_host_partition_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_donor_partition_fraction": float(agg_counts["foxf1_positive_donor_partition_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_ratio_mean_in_foxf1_positive": float(sub["foxf1_ratio_mean_in_foxf1_positive"].mean()),
                "host_ratio_mean_in_foxf1_positive": float(sub["host_ratio_mean_in_foxf1_positive"].mean()),
                "donor_ratio_mean_in_foxf1_positive": float(sub["donor_ratio_mean_in_foxf1_positive"].mean()),
                "foxf1_threshold": float(sub["foxf1_threshold"].mean()),
                "host_threshold": float(sub["host_threshold"].mean()),
                "foxf1_host_seed_threshold": float(sub["foxf1_host_seed_threshold"].mean()),
                "donor_threshold": float(sub["donor_threshold"].mean()),
            }
        )

    condition_identity_rows: list[dict[str, Any]] = []
    for condition, sub in foxf1_identity_plane_df.groupby("condition", sort=False):
        row0 = sub.iloc[0]
        agg_counts = {col: int(sub[col].sum()) for col in identity_metric_cols}
        foxf1_total = int(agg_counts["foxf1_positive_pixels"])
        mask_total = int(agg_counts["mask_pixels"])
        condition_identity_rows.append(
            {
                "summary_level": "condition",
                "image_id": "",
                "file_id": np.nan,
                "condition": str(condition),
                "position_label": "",
                "file_name": f"{condition}_all_files",
                "z_index": np.nan,
                **agg_counts,
                "foxf1_positive_fraction_of_mask": float(foxf1_total / mask_total) if mask_total > 0 else np.nan,
                "foxf1_positive_host_strict_fraction": float(agg_counts["foxf1_positive_host_strict_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_donor_strict_fraction": float(agg_counts["foxf1_positive_donor_strict_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_mixed_fraction": float(agg_counts["foxf1_positive_mixed_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_unlabeled_fraction": float(agg_counts["foxf1_positive_unlabeled_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_host_class_fraction": float(agg_counts["foxf1_positive_host_class_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_donor_class_fraction": float(agg_counts["foxf1_positive_donor_class_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_host_partition_fraction": float(agg_counts["foxf1_positive_host_partition_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_positive_donor_partition_fraction": float(agg_counts["foxf1_positive_donor_partition_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
                "foxf1_ratio_mean_in_foxf1_positive": float(sub["foxf1_ratio_mean_in_foxf1_positive"].mean()),
                "host_ratio_mean_in_foxf1_positive": float(sub["host_ratio_mean_in_foxf1_positive"].mean()),
                "donor_ratio_mean_in_foxf1_positive": float(sub["donor_ratio_mean_in_foxf1_positive"].mean()),
                "foxf1_threshold": float(sub["foxf1_threshold"].mean()),
                "host_threshold": float(sub["host_threshold"].mean()),
                "foxf1_host_seed_threshold": float(sub["foxf1_host_seed_threshold"].mean()),
                "donor_threshold": float(sub["donor_threshold"].mean()),
            }
        )

    if not foxf1_identity_plane_df.empty:
        agg_counts = {col: int(foxf1_identity_plane_df[col].sum()) for col in identity_metric_cols}
        foxf1_total = int(agg_counts["foxf1_positive_pixels"])
        mask_total = int(agg_counts["mask_pixels"])
        global_row = {
            "summary_level": "global",
            "image_id": "",
            "file_id": np.nan,
            "condition": "all_conditions",
            "position_label": "",
            "file_name": "all_files",
            "z_index": np.nan,
            **agg_counts,
            "foxf1_positive_fraction_of_mask": float(foxf1_total / mask_total) if mask_total > 0 else np.nan,
            "foxf1_positive_host_strict_fraction": float(agg_counts["foxf1_positive_host_strict_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
            "foxf1_positive_donor_strict_fraction": float(agg_counts["foxf1_positive_donor_strict_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
            "foxf1_positive_mixed_fraction": float(agg_counts["foxf1_positive_mixed_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
            "foxf1_positive_unlabeled_fraction": float(agg_counts["foxf1_positive_unlabeled_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
            "foxf1_positive_host_class_fraction": float(agg_counts["foxf1_positive_host_class_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
            "foxf1_positive_donor_class_fraction": float(agg_counts["foxf1_positive_donor_class_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
            "foxf1_positive_host_partition_fraction": float(agg_counts["foxf1_positive_host_partition_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
            "foxf1_positive_donor_partition_fraction": float(agg_counts["foxf1_positive_donor_partition_pixels"] / foxf1_total) if foxf1_total > 0 else np.nan,
            "foxf1_ratio_mean_in_foxf1_positive": float(foxf1_identity_plane_df["foxf1_ratio_mean_in_foxf1_positive"].mean()),
            "host_ratio_mean_in_foxf1_positive": float(foxf1_identity_plane_df["host_ratio_mean_in_foxf1_positive"].mean()),
            "donor_ratio_mean_in_foxf1_positive": float(foxf1_identity_plane_df["donor_ratio_mean_in_foxf1_positive"].mean()),
            "foxf1_threshold": float(foxf1_identity_plane_df["foxf1_threshold"].mean()),
            "host_threshold": float(foxf1_identity_plane_df["host_threshold"].mean()),
            "foxf1_host_seed_threshold": float(foxf1_identity_plane_df["foxf1_host_seed_threshold"].mean()),
            "donor_threshold": float(foxf1_identity_plane_df["donor_threshold"].mean()),
        }
        foxf1_identity_df = pd.concat(
            [
                foxf1_identity_plane_df,
                pd.DataFrame(file_identity_rows),
                pd.DataFrame(condition_identity_rows),
                pd.DataFrame([global_row]),
            ],
            ignore_index=True,
        )
    else:
        foxf1_identity_df = foxf1_identity_plane_df.copy()

    if bool(write_outputs):
        foxf1_identity_df.to_csv(foxf1_identity_output, sep="\t", index=False)

    sampled_pixels_df = pd.DataFrame(sampled_pixel_rows).sort_values(
        ["condition", "file_id", "z_index", "row_px", "col_px"]
    ).reset_index(drop=True)
    if bool(write_outputs):
        sampled_pixels_df.to_csv(pixel_sample_output, sep="\t", index=False)

    density_tables: list[pd.DataFrame] = []
    density_subsets = [("all_conditions", sampled_pixels_df)]
    density_subsets.extend(
        (str(condition), sub.copy())
        for condition, sub in sampled_pixels_df.groupby("condition", sort=False)
    )
    for condition_label, density_sub_df in density_subsets:
        for map_name, x_col, y_col in [
            ("host_vs_foxf1", "host_ratio", "foxf1_ratio"),
            ("donor_vs_host", "donor_ratio", "host_ratio"),
        ]:
            density_table = density_map_table(
                density_sub_df,
                x_col=x_col,
                y_col=y_col,
                map_name=map_name,
                bins=int(density_bins),
            )
            if density_table.empty:
                continue
            density_table["condition"] = str(condition_label)
            density_tables.append(density_table)
    density_df = pd.concat(density_tables, ignore_index=True) if density_tables else pd.DataFrame()
    if bool(write_outputs):
        density_df.to_csv(density_output, sep="\t", index=False)

    return {
        "root": root,
        "data_dir": data_dir,
        "manifest_output": manifest_output,
        "mask_dir": mask_dir,
        "plane_metrics_output": plane_metrics_output,
        "normalization_output": normalization_output,
        "normalization_hist_output": normalization_hist_output,
        "threshold_output": threshold_output,
        "class_summary_output": class_summary_output,
        "foxf1_identity_output": foxf1_identity_output,
        "pixel_sample_output": pixel_sample_output,
        "density_output": density_output,
        "manifest_df": manifest_df,
        "plane_metrics_df": plane_metrics_df,
        "normalization_df": normalization_df,
        "normalization_hist_df": normalization_hist_df,
        "threshold_df": threshold_df,
        "class_summary_df": class_summary_df,
        "foxf1_identity_df": foxf1_identity_df,
        "sampled_pixels_df": sampled_pixels_df,
        "density_df": density_df,
        "threshold_lookup": threshold_lookup,
        "summary_text": (
            f"mask planes={len(plane_metrics_df)} | "
            f"class rows={len(class_summary_df)} | "
            f"sampled pixels={len(sampled_pixels_df)}"
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
        normalization_output=args.normalization_output,
        normalization_hist_output=args.normalization_hist_output,
        threshold_output=args.threshold_output,
        class_summary_output=args.class_summary_output,
        foxf1_identity_output=args.foxf1_identity_output if hasattr(args, "foxf1_identity_output") else "results/tables/02_foxf1_positive_identity_summary.tsv",
        pixel_sample_output=args.pixel_sample_output,
        density_output=args.density_output,
        threshold_z=args.threshold_z,
        threshold_method=args.threshold_method,
        dominance_margin_log2=args.dominance_margin_log2,
        threshold_sample_per_plane=args.threshold_sample_per_plane,
        normalization_sample_per_plane=args.normalization_sample_per_plane,
        output_sample_per_plane=args.output_sample_per_plane,
        density_bins=args.density_bins,
        random_seed=args.random_seed,
        threshold_reference_condition=args.threshold_reference_condition,
        host_threshold_scope=args.host_threshold_scope,
        donor_threshold_scope=args.donor_threshold_scope,
        foxf1_threshold_scope=args.foxf1_threshold_scope,
        mesp2_threshold_scope=args.mesp2_threshold_scope,
        host_threshold_scale=args.host_threshold_scale,
        donor_threshold_scale=args.donor_threshold_scale,
        foxf1_threshold_scale=args.foxf1_threshold_scale,
        mesp2_threshold_scale=args.mesp2_threshold_scale,
        foxf1_threshold_condition_scales=parse_scale_mapping(args.foxf1_threshold_condition_scale),
        foxf1_threshold_condition_values=parse_scale_mapping(args.foxf1_threshold_condition_value),
        foxf1_threshold_file_scales={
            int(key): float(value)
            for key, value in parse_scale_mapping(args.foxf1_threshold_file_scale).items()
        },
        foxf1_host_seed_threshold_scale=args.foxf1_host_seed_threshold_scale,
        write_outputs=True,
    )
    print(f"[OK] wrote manifest to {res['manifest_output']}")
    print(f"[OK] wrote plane metrics to {res['plane_metrics_output']}")
    print(f"[OK] wrote normalization table to {res['normalization_output']}")
    print(f"[OK] wrote normalization histogram table to {res['normalization_hist_output']}")
    print(f"[OK] wrote thresholds to {res['threshold_output']}")
    print(f"[OK] wrote class summary to {res['class_summary_output']}")
    print(f"[OK] wrote FOXF1-positive identity summary to {res['foxf1_identity_output']}")
    print(f"[OK] wrote sampled pixels to {res['pixel_sample_output']}")
    print(f"[OK] wrote density maps to {res['density_output']}")


if __name__ == "__main__":
    main()
