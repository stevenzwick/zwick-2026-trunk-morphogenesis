"""NOT runnable as shipped. This stage reads the raw microscopy images, which are not deposited —
they are available from the lead contact on request, as the paper's Data Availability Statement
states. Its outputs are committed under the lane's `derived/` directory, which is what every
figure script actually reads. See `data/DOWNLOAD.md`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = ROOT / "data/raw/20260128_BMP4-reporter_LPM-organoids/d2-d5"
DEFAULT_OUTPUT_DIR = ROOT / "data/raw_three_channel_position_stacks"
DEFAULT_MANIFEST_PATH = ROOT / "results/tables/raw_three_channel_position_stacks.tsv"

POSITION_RE = re.compile(r"Pos(?P<position_index>\d+)$")
CHANNEL_RE = re.compile(r"img_channel(?P<channel_index>\d{3})_")
TIME_RE = re.compile(r"time(?P<time_index>\d+)_z000\.tif$")
CHANNEL_ORDER = (0, 1, 2)
CHANNEL_NAMES = ("phase", "RFP", "YFP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one raw 3-channel timelapse stack per position for Fiji inspection."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
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


def raw_frame_path(position_dir: Path, position_label: str, channel_index: int, time_index: int) -> Path:
    position_index = position_index_from_label(position_label)
    return (
        position_dir
        / f"img_channel{channel_index:03d}_position{position_index:03d}_time{time_index:09d}_z000.tif"
    )


def common_timepoints(position_dir: Path) -> list[int]:
    per_channel = available_timepoints_by_channel(position_dir)
    if not all(per_channel[channel_index] for channel_index in CHANNEL_ORDER):
        missing = [str(channel_index) for channel_index in CHANNEL_ORDER if not per_channel[channel_index]]
        raise RuntimeError(f"{position_dir.name}: missing all frames for channel(s) {', '.join(missing)}")
    return sorted(set.intersection(*(per_channel[channel_index] for channel_index in CHANNEL_ORDER)))


def load_position_stack(position_dir: Path, time_indices: list[int]) -> np.ndarray:
    position_label = position_dir.name
    frames = []
    for time_index in time_indices:
        channels = [
            tiff.imread(raw_frame_path(position_dir, position_label, channel_index, time_index))
            for channel_index in CHANNEL_ORDER
        ]
        frame = np.stack(channels, axis=0)
        frames.append(frame)
    return np.stack(frames, axis=0)


def export_position_stack(
    position_dir: Path,
    output_dir: Path,
    compression: str,
    overwrite: bool,
) -> dict[str, object]:
    position_label = position_dir.name
    time_indices = common_timepoints(position_dir)
    if not time_indices:
        raise RuntimeError(f"{position_label}: no common timepoints across channels 0/1/2")

    stack_path = output_dir / f"{position_label}_raw_three_channel_stack.tif"
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
            "status": "existing",
        }

    stack = load_position_stack(position_dir, time_indices)
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
            "name": f"{position_label} raw three-channel timelapse",
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
        "status": "written",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    position_dirs = discover_position_dirs(args.dataset_dir, args.positions)
    manifest_rows = []
    for position_number, position_dir in enumerate(position_dirs, start=1):
        print(f"[{position_number}/{len(position_dirs)}] Exporting {position_dir.name}")
        row = export_position_stack(
            position_dir=position_dir,
            output_dir=args.output_dir,
            compression=args.compression,
            overwrite=args.overwrite,
        )
        manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows).sort_values("position_label").reset_index(drop=True)
    manifest_df.to_csv(args.manifest_path, sep="\t", index=False)

    print(f"Wrote {len(manifest_df)} raw three-channel stacks to: {args.output_dir}")
    print(f"Wrote manifest to: {args.manifest_path}")


if __name__ == "__main__":
    main()
