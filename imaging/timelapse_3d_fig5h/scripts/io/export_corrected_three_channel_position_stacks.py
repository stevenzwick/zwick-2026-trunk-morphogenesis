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
DEFAULT_DATASET_DIR = ROOT / "data/raw/20260128_BMP4-reporter_LPM-organoids/d2-d5"
DEFAULT_MASK_ROOT = ROOT / "results/ilastik/organoid_masks/full_dataset_v1"
DEFAULT_PARAMETERS_PATH = ROOT / "results/tables/05_reporter_quantification_parameters.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/corrected_three_channel_position_stacks"
DEFAULT_MANIFEST_PATH = ROOT / "results/tables/corrected_three_channel_position_stacks.tsv"

POSITION_RE = re.compile(r"Pos(?P<position_index>\d+)$")
CHANNEL_RE = re.compile(r"img_channel(?P<channel_index>\d{3})_")
TIME_RE = re.compile(r"time(?P<time_index>\d+)_z000\.tif$")
CHANNEL_ORDER = (0, 1, 2)
REPORTER_CHANNELS = {1: "RFP", 2: "YFP"}
CHANNEL_NAMES = ("phase_raw", "RFP_corrected", "YFP_corrected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one 3-channel timelapse stack per position with raw phase and corrected reporter channels."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--parameters-path", type=Path, default=DEFAULT_PARAMETERS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--positions", nargs="*", default=None, help="Optional subset of position labels, e.g. Pos1 Pos2")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing stack files.")
    parser.add_argument(
        "--compression",
        default="zlib",
        choices=("zlib", "none"),
        help="Compression to use for exported stacks.",
    )
    return parser.parse_args()


def position_index_from_label(position_label: str) -> int:
    match = POSITION_RE.fullmatch(position_label)
    if not match:
        raise ValueError(f"Unexpected position label: {position_label}")
    return int(match.group("position_index"))


def raw_frame_path(position_dir: Path, position_label: str, channel_index: int, time_index: int) -> Path:
    position_index = position_index_from_label(position_label)
    return (
        position_dir
        / f"img_channel{channel_index:03d}_position{position_index:03d}_time{time_index:09d}_z000.tif"
    )


def mask_path(mask_root: Path, position_label: str, time_index: int) -> Path:
    position_index = position_index_from_label(position_label)
    return (
        mask_root
        / position_label
        / f"img_channel000_position{position_index:03d}_time{time_index:09d}_z000_mask.tiff"
    )


def discover_position_dirs(dataset_dir: Path, requested_positions: list[str] | None) -> list[Path]:
    position_dirs = sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir() and POSITION_RE.fullmatch(path.name)],
        key=lambda path: position_index_from_label(path.name),
    )
    if requested_positions is None:
        return position_dirs
    requested = set(requested_positions)
    selected = [path for path in position_dirs if path.name in requested]
    missing = sorted(requested.difference({path.name for path in selected}), key=position_index_from_label)
    if missing:
        raise FileNotFoundError(f"Requested positions not found: {', '.join(missing)}")
    return selected


def available_timepoints_by_channel(position_dir: Path) -> dict[int, set[int]]:
    out = {channel_index: set() for channel_index in CHANNEL_ORDER}
    for path in position_dir.glob("img_channel*_z000.tif"):
        channel_match = CHANNEL_RE.search(path.name)
        time_match = TIME_RE.search(path.name)
        if not channel_match or not time_match:
            continue
        channel_index = int(channel_match.group("channel_index"))
        time_index = int(time_match.group("time_index"))
        if channel_index in out:
            out[channel_index].add(time_index)
    return out


def common_timepoints(position_dir: Path, mask_root: Path) -> list[int]:
    per_channel = available_timepoints_by_channel(position_dir)
    if not all(per_channel[channel_index] for channel_index in CHANNEL_ORDER):
        missing = [str(channel_index) for channel_index in CHANNEL_ORDER if not per_channel[channel_index]]
        raise RuntimeError(f"{position_dir.name}: missing all frames for channel(s) {', '.join(missing)}")
    shared = sorted(set.intersection(*(per_channel[channel_index] for channel_index in CHANNEL_ORDER)))
    position_label = position_dir.name
    return [time_index for time_index in shared if mask_path(mask_root, position_label, time_index).exists()]


def apply_illumination_field(signal: np.ndarray, field: np.ndarray | None) -> np.ndarray:
    if field is None:
        return signal.astype(np.float32)
    return (signal.astype(np.float32) / np.clip(field.astype(np.float32), 1e-6, None)).astype(np.float32)


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
    corrected = signal.astype(np.float32) - np.float32(background_value)
    return corrected.astype(np.float32), background_value


def load_illumination_fields(parameters: dict[str, object]) -> dict[str, np.ndarray | None]:
    fields = {"RFP": None, "YFP": None}
    if not bool(parameters.get("illumination_correction_applied")):
        return fields
    illumination_field_path = parameters.get("illumination_field_path")
    if not illumination_field_path:
        return fields
    payload = np.load(Path(str(illumination_field_path)))
    fields["RFP"] = payload["rfp_field"].astype(np.float32)
    fields["YFP"] = payload["yfp_field"].astype(np.float32)
    return fields


def export_position_stack(
    position_dir: Path,
    mask_root: Path,
    output_dir: Path,
    compression: str,
    overwrite: bool,
    parameters: dict[str, object],
    illumination_fields: dict[str, np.ndarray | None],
) -> dict[str, object]:
    position_label = position_dir.name
    time_indices = common_timepoints(position_dir, mask_root)
    if not time_indices:
        raise RuntimeError(f"{position_label}: no common timepoints with masks across channels 0/1/2")

    stack_path = output_dir / f"{position_label}_corrected_three_channel_stack.tif"
    if stack_path.exists() and not overwrite:
        with tiff.TiffFile(stack_path) as tif:
            series = tif.series[0]
            shape = tuple(int(v) for v in series.shape)
            axes = str(series.axes)
            dtype = str(series.dtype)
        return {
            "position_label": position_label,
            "frame_count": len(time_indices),
            "first_time_index": int(time_indices[0]),
            "last_time_index": int(time_indices[-1]),
            "axes": axes,
            "dtype": dtype,
            "shape": "x".join(str(v) for v in shape),
            "stack_path": str(stack_path),
            "compression": compression,
            "channel_names": ",".join(CHANNEL_NAMES),
            "format": "imagej_hyperstack_tif",
            "illumination_correction_applied": bool(parameters.get("illumination_correction_applied")),
            "background_estimator": str(parameters.get("background_estimator", "whole_off_cyst")),
            "status": "existing",
        }

    frames = []
    for time_index in time_indices:
        phase = tiff.imread(raw_frame_path(position_dir, position_label, 0, time_index)).astype(np.float32)
        organoid_mask = tiff.imread(mask_path(mask_root, position_label, time_index)).astype(bool)
        corrected_channels = []
        for channel_index, reporter in REPORTER_CHANNELS.items():
            signal_raw = tiff.imread(raw_frame_path(position_dir, position_label, channel_index, time_index))
            signal_illum = apply_illumination_field(signal_raw, illumination_fields[reporter])
            corrected, _ = corrected_signal(
                signal=signal_illum,
                organoid_mask=organoid_mask,
                background_estimator=str(parameters.get("background_estimator", "whole_off_cyst")),
                background_ring_inner=int(parameters.get("background_ring_inner", 4)),
                background_ring_outer=int(parameters.get("background_ring_outer", 12)),
                min_ring_pixels=int(parameters.get("min_ring_pixels", 100)),
            )
            corrected_channels.append(corrected.astype(np.float32))
        frame = np.stack([phase, *corrected_channels], axis=0).astype(np.float32)
        frames.append(frame)

    stack = np.stack(frames, axis=0).astype(np.float32)
    compression_arg = None if compression == "none" else compression
    tiff.imwrite(
        stack_path,
        stack,
        imagej=True,
        compression=compression_arg,
        metadata={
            "axes": "TCYX",
            "channels": int(stack.shape[1]),
            "frames": int(stack.shape[0]),
            "hyperstack": True,
            "mode": "grayscale",
            "channel_names": list(CHANNEL_NAMES),
            "name": f"{position_label} corrected three-channel timelapse",
        },
    )

    return {
        "position_label": position_label,
        "frame_count": int(stack.shape[0]),
        "first_time_index": int(time_indices[0]),
        "last_time_index": int(time_indices[-1]),
        "axes": "TCYX",
        "dtype": str(stack.dtype),
        "shape": "x".join(str(v) for v in stack.shape),
        "stack_path": str(stack_path),
        "compression": compression,
        "channel_names": ",".join(CHANNEL_NAMES),
        "format": "imagej_hyperstack_tif",
        "illumination_correction_applied": bool(parameters.get("illumination_correction_applied")),
        "background_estimator": str(parameters.get("background_estimator", "whole_off_cyst")),
        "status": "written",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    parameters = json.loads(args.parameters_path.read_text())
    illumination_fields = load_illumination_fields(parameters)
    position_dirs = discover_position_dirs(args.dataset_dir, args.positions)

    manifest_rows = []
    for position_number, position_dir in enumerate(position_dirs, start=1):
        print(f"[{position_number}/{len(position_dirs)}] Exporting {position_dir.name}")
        row = export_position_stack(
            position_dir=position_dir,
            mask_root=args.mask_root,
            output_dir=args.output_dir,
            compression=args.compression,
            overwrite=args.overwrite,
            parameters=parameters,
            illumination_fields=illumination_fields,
        )
        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows).sort_values("position_label").reset_index(drop=True)
    manifest_df.to_csv(args.manifest_path, sep="\t", index=False)

    print(f"Wrote {len(manifest_df)} corrected three-channel stacks to: {args.output_dir}")
    print(f"Wrote manifest to: {args.manifest_path}")


if __name__ == "__main__":
    main()
