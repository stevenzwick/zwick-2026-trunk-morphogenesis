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
from scipy import ndimage as ndi
from skimage import morphology


ROOT = Path(__file__).resolve().parents[2]
POSITION_MANIFEST_PATH = ROOT / "results/manifests/acquisition_position_manifest.tsv"
MASK_METRICS_PATH = ROOT / "results/ilastik/qc/full_dataset_v1_mask_metrics.tsv"
FIELD_OUTPUT_PATH = ROOT / "results/qc/04_masked_illumination_fields.npz"
POSITION_DIAGNOSTICS_PATH = ROOT / "results/tables/04_masked_illumination_position_diagnostics.tsv"
PARAMETERS_OUTPUT_PATH = ROOT / "results/qc/04_masked_illumination_parameters.json"
POSITION_RE = re.compile(r"Pos(?P<position_index>\d+)$")
REPORTER_CHANNELS = {1: "RFP", 2: "YFP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate smooth channel-specific illumination fields using only off-organoid pixels, "
            "then summarize how the correction changes local annulus background measurements."
        )
    )
    parser.add_argument("--position-manifest-path", type=Path, default=POSITION_MANIFEST_PATH)
    parser.add_argument("--mask-metrics-path", type=Path, default=MASK_METRICS_PATH)
    parser.add_argument("--field-output-path", type=Path, default=FIELD_OUTPUT_PATH)
    parser.add_argument("--position-diagnostics-path", type=Path, default=POSITION_DIAGNOSTICS_PATH)
    parser.add_argument("--parameters-output-path", type=Path, default=PARAMETERS_OUTPUT_PATH)
    parser.add_argument("--sample-every-n", type=int, default=10)
    parser.add_argument("--field-gaussian-sigma", type=float, default=25.0)
    parser.add_argument("--review-min-time", type=int, default=10)
    parser.add_argument("--background-inner", type=int, default=4)
    parser.add_argument("--background-outer", type=int, default=12)
    parser.add_argument("--min-ring-pixels", type=int, default=100)
    parser.add_argument("--min-background-pixels", type=int, default=1000)
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


def annulus_mask(organoid_mask: np.ndarray, inner: int, outer: int, min_ring_pixels: int) -> np.ndarray:
    inner_mask = morphology.binary_dilation(organoid_mask, morphology.disk(inner))
    outer_mask = morphology.binary_dilation(organoid_mask, morphology.disk(outer))
    ring = outer_mask & ~inner_mask
    if int(ring.sum()) < min_ring_pixels:
        ring = ~outer_mask
    if int(ring.sum()) < min_ring_pixels:
        ring = ~organoid_mask
    return ring


def smooth_masked_image(masked_mean: np.ndarray, observed_mask: np.ndarray, sigma: float) -> np.ndarray:
    weighted = ndi.gaussian_filter(masked_mean * observed_mask, sigma=sigma, mode="nearest")
    weight_sum = ndi.gaussian_filter(observed_mask.astype(np.float32), sigma=sigma, mode="nearest")
    return weighted / np.maximum(weight_sum, 1e-6)


def estimate_channel_field(
    sampled_metrics: pd.DataFrame,
    dataset_dir: Path,
    channel_index: int,
    sigma: float,
    min_background_pixels: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    sum_image = None
    count_image = None
    used_frames = 0

    for row in sampled_metrics.itertuples(index=False):
        mask = tiff.imread(Path(str(row.mask_path))).astype(bool)
        background_mask = ~mask
        if int(background_mask.sum()) < min_background_pixels:
            continue

        signal = tiff.imread(
            raw_frame_path(dataset_dir, str(row.position_label), channel_index, int(row.time_index))
        ).astype(np.float32)
        background_values = signal[background_mask]
        background_scale = float(np.median(background_values))
        if not np.isfinite(background_scale) or background_scale <= 0:
            continue

        normalized = signal / background_scale
        if sum_image is None:
            sum_image = np.zeros_like(normalized, dtype=np.float64)
            count_image = np.zeros_like(normalized, dtype=np.float64)

        sum_image += normalized * background_mask
        count_image += background_mask.astype(np.float64)
        used_frames += 1

    if sum_image is None or count_image is None or used_frames == 0:
        raise RuntimeError(f"No valid frames were available for channel {channel_index}")

    observed_mask = count_image > 0
    masked_mean = np.divide(sum_image, np.maximum(count_image, 1e-6))
    field = smooth_masked_image(masked_mean=masked_mean, observed_mask=observed_mask, sigma=sigma)

    finite = np.isfinite(field)
    if not np.any(finite):
        raise RuntimeError(f"Estimated illumination field is not finite for channel {channel_index}")
    field = field / np.median(field[finite])

    return field.astype(np.float32), count_image.astype(np.float32), used_frames


def main() -> None:
    args = parse_args()
    for path in [
        args.field_output_path,
        args.position_diagnostics_path,
        args.parameters_output_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    position_manifest = pd.read_csv(args.position_manifest_path, sep="\t")
    dataset_dir = ROOT / str(position_manifest["dataset_dir"].iloc[0])
    mask_metrics = pd.read_csv(args.mask_metrics_path, sep="\t").sort_values(
        ["position_label", "time_index"]
    ).reset_index(drop=True)

    sampled_metrics = mask_metrics.loc[~mask_metrics["exclude_from_analysis"]].copy()
    sampled_metrics = sampled_metrics.loc[sampled_metrics["time_index"] % args.sample_every_n == 0].copy()
    if sampled_metrics.empty:
        raise RuntimeError("No sampled clean frames available for illumination estimation.")

    field_payload: dict[str, np.ndarray] = {}
    field_summary_rows: list[dict[str, object]] = []
    used_frames_by_channel: dict[str, int] = {}

    for channel_index, reporter in REPORTER_CHANNELS.items():
        field, sample_count, used_frames = estimate_channel_field(
            sampled_metrics=sampled_metrics,
            dataset_dir=dataset_dir,
            channel_index=channel_index,
            sigma=args.field_gaussian_sigma,
            min_background_pixels=args.min_background_pixels,
        )
        field_payload[f"{reporter.lower()}_field"] = field
        field_payload[f"{reporter.lower()}_sample_count"] = sample_count
        used_frames_by_channel[reporter] = int(used_frames)
        field_summary_rows.append(
            {
                "reporter": reporter,
                "used_frames": int(used_frames),
                "field_min": float(np.nanmin(field)),
                "field_p01": float(np.nanpercentile(field, 1.0)),
                "field_median": float(np.nanmedian(field)),
                "field_p99": float(np.nanpercentile(field, 99.0)),
                "field_max": float(np.nanmax(field)),
                "center_value": float(field[field.shape[0] // 2, field.shape[1] // 2]),
                "corner_mean": float(
                    np.mean(
                        np.concatenate(
                            [
                                field[:20, :20].ravel(),
                                field[:20, -20:].ravel(),
                                field[-20:, :20].ravel(),
                                field[-20:, -20:].ravel(),
                            ]
                        )
                    )
                ),
            }
        )

    np.savez_compressed(args.field_output_path, **field_payload)

    diagnostics_rows: list[dict[str, object]] = []
    clean_metrics = mask_metrics.loc[~mask_metrics["exclude_from_analysis"]].copy()
    for position_label, subset in clean_metrics.groupby("position_label", sort=True):
        subset = subset.sort_values("time_index").reset_index(drop=True)
        preferred = subset.loc[subset["time_index"] >= args.review_min_time].copy()
        review_row = preferred.iloc[0] if not preferred.empty else subset.iloc[0]
        mask = tiff.imread(Path(str(review_row["mask_path"]))).astype(bool)
        annulus = annulus_mask(
            organoid_mask=mask,
            inner=args.background_inner,
            outer=args.background_outer,
            min_ring_pixels=args.min_ring_pixels,
        )

        yy, xx = np.nonzero(mask)
        centroid_x = float(np.mean(xx)) if len(xx) else float("nan")
        centroid_y = float(np.mean(yy)) if len(yy) else float("nan")
        raw_area_px = int(mask.sum())
        annulus_area_px = int(annulus.sum())

        for channel_index, reporter in REPORTER_CHANNELS.items():
            signal = tiff.imread(
                raw_frame_path(dataset_dir, str(position_label), channel_index, int(review_row["time_index"]))
            ).astype(np.float32)
            field = field_payload[f"{reporter.lower()}_field"]
            corrected = signal / np.clip(field, 1e-6, None)

            diagnostics_rows.append(
                {
                    "position_label": str(position_label),
                    "reporter": reporter,
                    "time_index": int(review_row["time_index"]),
                    "mask_path": str(review_row["mask_path"]),
                    "raw_area_px": raw_area_px,
                    "annulus_area_px": annulus_area_px,
                    "centroid_x_px": centroid_x,
                    "centroid_y_px": centroid_y,
                    "field_at_centroid": (
                        float(field[int(round(centroid_y)), int(round(centroid_x))])
                        if np.isfinite(centroid_x) and np.isfinite(centroid_y)
                        else float("nan")
                    ),
                    "raw_annulus_median": (
                        float(np.median(signal[annulus])) if annulus_area_px > 0 else float("nan")
                    ),
                    "corrected_annulus_median": (
                        float(np.median(corrected[annulus])) if annulus_area_px > 0 else float("nan")
                    ),
                    "raw_background_median": float(np.median(signal[~mask])) if np.any(~mask) else float("nan"),
                    "corrected_background_median": (
                        float(np.median(corrected[~mask])) if np.any(~mask) else float("nan")
                    ),
                }
            )

    diagnostics_df = pd.DataFrame(diagnostics_rows).sort_values(["reporter", "position_label"]).reset_index(drop=True)
    diagnostics_df.to_csv(args.position_diagnostics_path, sep="\t", index=False)

    parameters = {
        "dataset_dir": str(dataset_dir),
        "sample_every_n": int(args.sample_every_n),
        "field_gaussian_sigma": float(args.field_gaussian_sigma),
        "review_min_time": int(args.review_min_time),
        "background_inner": int(args.background_inner),
        "background_outer": int(args.background_outer),
        "min_ring_pixels": int(args.min_ring_pixels),
        "min_background_pixels": int(args.min_background_pixels),
        "sampled_clean_frame_count": int(sampled_metrics.shape[0]),
        "used_frames_by_channel": used_frames_by_channel,
        "field_summary": field_summary_rows,
        "notes": [
            "Each reporter channel is modeled separately.",
            "Only off-organoid pixels contribute to the illumination estimate.",
            "Each sampled frame is normalized by its off-organoid median before accumulation.",
            "The final field is a very smooth weighted average, normalized to median 1.",
        ],
    }
    args.parameters_output_path.write_text(json.dumps(parameters, indent=2))

    print(f"Wrote masked illumination fields: {args.field_output_path}")
    print(f"Wrote position diagnostics: {args.position_diagnostics_path}")
    print(f"Wrote parameters: {args.parameters_output_path}")


if __name__ == "__main__":
    main()
