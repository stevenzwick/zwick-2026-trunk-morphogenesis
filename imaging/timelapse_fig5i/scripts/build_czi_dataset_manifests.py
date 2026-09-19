#!/usr/bin/env python3
"""Build per-file manifests for a CZI-based timelapse dataset.

NOT runnable as shipped. It reads the raw .czi acquisitions, which are not deposited — they are
available from the lead contact on request, per the paper's Data Availability Statement.

Note it applies only to the `czi_files` dataset (20260314). The dataset behind the published
Fig 5i panel, 20260213, is `position_tiffs`, and this tool raises NotImplementedError on it by
design. The panel is drawn from the committed table in ../derived/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
# czifile cannot be installed beside this project's pinned numpy, and mis-frames odd-sized
# acquisitions. czi_compat presents czifile's interface over libCZI. See src/trunk_morph_ref/czi_compat.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from src.trunk_morph_ref import czi_compat as czifile

from czi_dataset_utils import (
    extract_channels,
    extract_objective_name,
    extract_scale_um,
    get_scene_records,
    normalize_scene_label,
)
from dataset_config import ensure_dataset_scaffold, load_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset-local manifests for CZI-based datasets.")
    parser.add_argument(
        "--dataset-id",
        # 20260314 is the ONLY czi_files dataset; this tool raises NotImplementedError
        # on the position_tiffs ones, the published 20260213 among them.
        default="20260314",
        help="Dataset ID from configs/datasets (default: 20260314, the only czi_files dataset)",
    )
    return parser.parse_args()


def build_czi_file_manifest(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset_config(dataset_id)
    ensure_dataset_scaffold(dataset)
    if dataset.raw_kind != "czi_files":
        raise NotImplementedError(
            f"build_czi_dataset_manifests.py supports only czi_files datasets. "
            f"{dataset.dataset_id} is raw_kind={dataset.raw_kind}."
        )

    metadata_dir = dataset.dataset_results_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in sorted(dataset.raw_root.glob("*.czi")):
        with czifile.CziFile(path) as czi:
            axes = str(czi.axes)
            shape = tuple(int(v) for v in czi.shape)
            metadata = czi.metadata(raw=False)
        axis_idx = {ax: i for i, ax in enumerate(axes)}
        scale = extract_scale_um(metadata)
        channels = extract_channels(metadata, c_count=shape[axis_idx["C"]] if "C" in axis_idx else None)
        image = (
            metadata.get("ImageDocument", {})
            .get("Metadata", {})
            .get("Information", {})
            .get("Image", {})
        )
        rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "dataset_display_name": dataset.display_name,
                "file_name": path.name,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "axes": axes,
                "shape": ";".join(str(v) for v in shape),
                "size_s": int(shape[axis_idx["S"]]) if "S" in axis_idx else 1,
                "size_t": int(shape[axis_idx["T"]]) if "T" in axis_idx else 1,
                "size_c": int(shape[axis_idx["C"]]) if "C" in axis_idx else 1,
                "size_y": int(shape[axis_idx["Y"]]) if "Y" in axis_idx else None,
                "size_x": int(shape[axis_idx["X"]]) if "X" in axis_idx else None,
                "pixel_type": str(image.get("PixelType", "")),
                "acquisition_datetime": str(image.get("AcquisitionDateAndTime", "")),
                "objective_name": extract_objective_name(metadata),
                "scale_x_um": scale["X"],
                "scale_y_um": scale["Y"],
                "scale_z_um": scale["Z"],
                "channel_names": ";".join(channels),
                "is_primary_timelapse": bool(dataset.primary_timelapse_czi and path.resolve() == dataset.primary_timelapse_czi.resolve()),
            }
        )

    if not rows:
        # An empty frame has no "file_name" column, so sorting it raises a bare KeyError that
        # says nothing about the real problem: the raw acquisitions were not found.
        raise SystemExit(
            f"No .czi files found under {dataset.raw_root}\n"
            "The raw timelapse acquisitions are not in this repository; see data/DOWNLOAD.md.")
    files_df = pd.DataFrame(rows).sort_values("file_name").reset_index(drop=True)

    if dataset.primary_timelapse_czi is None:
        scene_df = pd.DataFrame()
    else:
        scene_rows = get_scene_records(dataset.primary_timelapse_czi, scene_name_parser=dataset.scene_name_parser)
        for row in scene_rows:
            row["dataset_id"] = dataset.dataset_id
            row["dataset_display_name"] = dataset.display_name
            row["primary_timelapse_file"] = str(dataset.primary_timelapse_czi)
            row["scene_label"] = normalize_scene_label(
                scene_index=row["scene_index"],
                scene_name_raw=row["scene_name_raw"],
                dataset_id=dataset.dataset_id,
            )
        scene_df = pd.DataFrame(scene_rows).sort_values("scene_index").reset_index(drop=True)
        if dataset.scene_block_conditions:
            scene_df["media_condition_scene_name"] = scene_df["media_condition"]
            scene_df["condition_block_index"] = pd.NA
            scene_df["intended_media_condition"] = pd.NA
            scene_df["condition_block_note"] = ""
            scene_df["condition_block_source"] = ""
            for spec in dataset.scene_block_conditions:
                start_scene = int(spec["start_scene"])
                end_scene = int(spec["end_scene"])
                mask = scene_df["scene_index"].between(start_scene, end_scene)
                scene_df.loc[mask, "condition_block_index"] = int(spec.get("block_index", start_scene // 4 + 1))
                scene_df.loc[mask, "intended_media_condition"] = str(spec["intended_media_condition"])
                scene_df.loc[mask, "condition_block_note"] = str(spec.get("block_note", "") or "")
                scene_df.loc[mask, "condition_block_source"] = f"{dataset.dataset_id} config scene_block_conditions"

    file_manifest_path = metadata_dir / "czi_file_manifest.tsv"
    files_df.to_csv(file_manifest_path, sep="\t", index=False)

    if not scene_df.empty:
        scene_manifest_path = metadata_dir / "primary_scene_manifest.tsv"
        scene_df.to_csv(scene_manifest_path, sep="\t", index=False)
        if dataset.scene_block_conditions:
            block_rows = []
            for spec in dataset.scene_block_conditions:
                block_rows.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "block_index": int(spec.get("block_index", int(spec["start_scene"]) // 4 + 1)),
                        "start_scene": int(spec["start_scene"]),
                        "end_scene": int(spec["end_scene"]),
                        "n_scenes": int(spec["end_scene"]) - int(spec["start_scene"]) + 1,
                        "intended_media_condition": str(spec["intended_media_condition"]),
                        "block_note": str(spec.get("block_note", "") or ""),
                        "source": f"{dataset.dataset_id} config scene_block_conditions",
                    }
                )
            pd.DataFrame(block_rows).to_csv(metadata_dir / "scene_condition_blocks.tsv", sep="\t", index=False)

    summary_lines = [
        f"# Dataset {dataset.dataset_id} CZI metadata",
        "",
        f"- dataset_id: `{dataset.dataset_id}`",
        f"- display_name: `{dataset.display_name}`",
        f"- raw_root: `{dataset.raw_root}`",
        f"- file manifest: `{metadata_dir / 'czi_file_manifest.tsv'}`",
    ]
    if dataset.primary_timelapse_czi is not None:
        summary_lines.extend(
            [
                f"- primary timelapse CZI: `{dataset.primary_timelapse_czi}`",
                f"- primary scene manifest: `{metadata_dir / 'primary_scene_manifest.tsv'}`",
            ]
        )
    if dataset.scene_block_conditions:
        summary_lines.append(f"- scene condition blocks: `{metadata_dir / 'scene_condition_blocks.tsv'}`")
    (metadata_dir / "README.md").write_text("\n".join(summary_lines) + "\n")

    return files_df, scene_df


def main() -> None:
    args = parse_args()
    files_df, scene_df = build_czi_file_manifest(args.dataset_id)
    print("Wrote file manifest rows:", len(files_df))
    if not scene_df.empty:
        print("Wrote scene manifest rows:", len(scene_df))


if __name__ == "__main__":
    main()
