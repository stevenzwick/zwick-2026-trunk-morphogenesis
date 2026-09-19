"""Rebuild FOXF1 bead manifest tables from raw workspace data.

NOT runnable as shipped. This stage reads the raw microscopy images, which are not deposited —
they are available from the lead contact on request, as the paper's Data Availability Statement
states. Its outputs are committed under the lane's `derived/` directory, which is what every
figure script actually reads. See `data/DOWNLOAD.md`.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# czifile cannot be installed beside this project's pinned numpy, and mis-frames odd-sized
# acquisitions. czi_compat presents czifile's interface over libCZI. See src/trunk_morph_ref/czi_compat.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from src.trunk_morph_ref import czi_compat as czifile
import numpy as np
import pandas as pd


SELECTION_RULE = "small_primary_large_spatial_when_available"

FILESYSTEM_COLUMNS = [
    "path",
    "kind",
    "size_bytes",
    "mtime_iso",
    "atime_iso",
    "mode",
]

CZI_METADATA_COLUMNS = [
    "path",
    "size_bytes",
    "mtime_iso",
    "SizeX",
    "SizeY",
    "SizeZ",
    "SizeC",
    "SizeT",
    "SizeS",
    "PixelType",
    "AcquisitionDateAndTime",
    "ObjectiveName",
    "ScaleX_um",
    "ScaleY_um",
    "ScaleZ_um",
    "ActiveChannels",
    "AllChannels",
]

FILE_MANIFEST_COLUMNS = [
    "cohort_id",
    "dataset_date",
    "condition",
    "canonical_position",
    "variant",
    "file_path",
    "exists",
    "include_primary_analysis",
    "include_spatial_metadata",
    "external_manifest_included",
    "external_manifest_note",
    "size_bytes",
    "acquisition_time_utc",
    "center_position",
    "size_c",
    "active_channels",
    "selection_rule",
]

POSITION_MANIFEST_COLUMNS = [
    "cohort_id",
    "dataset_date",
    "condition",
    "canonical_position",
    "has_small",
    "has_large",
    "primary_analysis_variant",
    "primary_analysis_file",
    "spatial_metadata_variant",
    "spatial_metadata_file",
    "pair_status",
    "selection_rule",
]

CROSSWALK_COLUMNS = [
    "canonical_position",
    "pre_live_small_file",
    "pre_live_large_file",
    "post_fix_small_file",
    "post_fix_large_file",
    "pre_primary_analysis_file",
    "post_primary_analysis_file",
    "pre_spatial_metadata_file",
    "post_spatial_metadata_file",
    "pair_status",
]

EPHEMERAL_DIR_NAMES = {
    ".cache",
    ".ipynb_checkpoints",
    ".ipython",
    ".jupyter",
    ".jupyter_config",
    ".mplconfig",
    ".venv",
    "__pycache__",
}


@dataclass(frozen=True)
class CohortSpec:
    cohort_id: str
    dataset_date: str
    condition: str
    relative_dir: str
    external_manifest_path: Optional[str] = None


COHORTS: Tuple[CohortSpec, ...] = (
    CohortSpec(
        cohort_id="2026-01-14_day2_live",
        dataset_date="2026-01-14",
        condition="live",
        relative_dir="data/2026-01-14_PDMS/day2/single",
    ),
    CohortSpec(
        cohort_id="2026-01-22_day2_live",
        dataset_date="2026-01-22",
        condition="live",
        relative_dir="data/2026-01-22_PDMS/day2/single",
        external_manifest_path="data/2026-01-22_PDMS/day2/analysis_manifest_day2_single.tsv",
    ),
    CohortSpec(
        cohort_id="2026-01-22_day2_fix",
        dataset_date="2026-01-22",
        condition="fixed",
        relative_dir="data/2026-01-22_PDMS/day2-fix-well2-568SOX2-647T/single",
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=".",
        help="Workspace root containing data/, notebooks/, scripts/, results/.",
    )
    p.add_argument(
        "--outdir",
        default="results/manifests",
        help="Output directory for manifest TSV/TXT files.",
    )
    return p.parse_args()


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _canonical_position_sort_key(pos: str) -> Tuple[int, int, str]:
    m = re.fullmatch(r"(\d+)-(\d+)", str(pos))
    if m:
        return (int(m.group(1)), int(m.group(2)), str(pos))
    return (10**9, 10**9, str(pos))


def _infer_variant(file_path: Union[str, Path]) -> str:
    name = Path(file_path).name
    return "large" if name.endswith("-large.czi") else "small"


def _infer_canonical_position(file_path: Union[str, Path]) -> str:
    stem = Path(file_path).stem
    if stem.endswith("-large"):
        stem = stem[: -len("-large")]
    m = re.match(r"(\d+-\d+)", stem)
    return m.group(1) if m else stem


def _iso_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _extract_all_channels(metadata: dict) -> List[str]:
    image = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]
    channels = _ensure_list(image.get("Dimensions", {}).get("Channels", {}).get("Channel"))
    names: List[str] = []
    for idx, ch in enumerate(channels):
        if not isinstance(ch, dict):
            names.append(f"Channel_{idx}")
            continue
        names.append(
            str(
                ch.get("Name")
                or ch.get("ShortName")
                or ch.get("DyeName")
                or f"Channel_{idx}"
            )
        )
    return names


def _extract_active_channels(metadata: dict, c_count: Optional[int]) -> List[str]:
    names = _extract_all_channels(metadata)
    if c_count is None or c_count <= 0:
        return names
    return names[: int(c_count)]


def _extract_scale_um(metadata: dict) -> Dict[str, float]:
    out = {"X": np.nan, "Y": np.nan, "Z": np.nan}
    dist = (
        metadata["ImageDocument"]["Metadata"]
        .get("Scaling", {})
        .get("Items", {})
        .get("Distance", [])
    )
    for item in _ensure_list(dist):
        if not isinstance(item, dict):
            continue
        axis_id = str(item.get("Id", "")).upper()
        if axis_id not in out:
            continue
        try:
            out[axis_id] = float(item.get("Value", np.nan)) * 1e6
        except Exception:
            out[axis_id] = np.nan
    return out


def _extract_objective_name(metadata: dict) -> str:
    objectives = (
        metadata["ImageDocument"]["Metadata"]
        .get("Information", {})
        .get("Instrument", {})
        .get("Objectives", {})
        .get("Objective")
    )
    objectives = _ensure_list(objectives)
    if not objectives:
        return ""
    first = objectives[0]
    if isinstance(first, dict):
        return str(first.get("Name", ""))
    return ""


def _extract_center_position(metadata: dict) -> str:
    sample_holder = (
        metadata["ImageDocument"]["Metadata"]
        .get("Experiment", {})
        .get("ExperimentBlocks", {})
        .get("AcquisitionBlock", {})
        .get("SubDimensionSetups", {})
        .get("RegionsSetup", {})
        .get("SampleHolder", {})
    )
    tile_region = _ensure_list(
        sample_holder.get("TileRegions", {}).get("TileRegion")
    )
    if not tile_region:
        return ""
    first = tile_region[0]
    if isinstance(first, dict):
        return str(first.get("CenterPosition", "") or "")
    return ""


def _axes_sizes(axes: str, shape: Sequence[int]) -> Dict[str, float]:
    sizes: Dict[str, float] = {k: np.nan for k in ["X", "Y", "Z", "C", "T", "S"]}
    for axis, dim in zip(str(axes), shape):
        if axis in sizes:
            sizes[axis] = float(dim)
    return sizes


def extract_czi_metadata_row(path: Path, root: Path) -> Dict[str, object]:
    stat_info = path.stat()
    with czifile.CziFile(path) as czi:
        axes = str(czi.axes)
        shape = tuple(int(x) for x in czi.shape)
        metadata = czi.metadata(raw=False)

    axis_sizes = _axes_sizes(axes=axes, shape=shape)
    scale_um = _extract_scale_um(metadata)
    c_count = (
        int(axis_sizes["C"])
        if np.isfinite(axis_sizes["C"])
        else None
    )
    all_channels = _extract_all_channels(metadata)
    active_channels = _extract_active_channels(metadata, c_count=c_count)
    image_meta = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]

    return {
        "path": _relpath(path, root),
        "size_bytes": int(stat_info.st_size),
        "mtime_iso": _iso_timestamp(stat_info.st_mtime),
        "SizeX": axis_sizes["X"],
        "SizeY": axis_sizes["Y"],
        "SizeZ": axis_sizes["Z"],
        "SizeC": axis_sizes["C"],
        "SizeT": axis_sizes["T"],
        "SizeS": axis_sizes["S"],
        "PixelType": str(image_meta.get("PixelType", "")),
        "AcquisitionDateAndTime": str(image_meta.get("AcquisitionDateAndTime", "")),
        "ObjectiveName": _extract_objective_name(metadata),
        "ScaleX_um": scale_um["X"],
        "ScaleY_um": scale_um["Y"],
        "ScaleZ_um": scale_um["Z"],
        "ActiveChannels": ";".join(active_channels),
        "AllChannels": ";".join(all_channels),
    }


def build_czi_metadata(root: Path) -> pd.DataFrame:
    rows = [
        extract_czi_metadata_row(path, root=root)
        for path in sorted(root.joinpath("data").rglob("*.czi"))
    ]
    out = pd.DataFrame(rows, columns=CZI_METADATA_COLUMNS)
    return out.sort_values("path").reset_index(drop=True)


def build_filesystem_inventory(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def walk(dir_path: Path) -> None:
        entries = sorted(os.scandir(dir_path), key=lambda e: e.name)
        for entry in entries:
            entry_path = Path(entry.path)
            rel = entry_path.relative_to(root).as_posix()
            stat_info = entry.stat(follow_symlinks=False)
            if entry.is_symlink():
                kind = "link"
            elif entry.is_dir(follow_symlinks=False):
                kind = "dir"
            else:
                kind = "file"

            rows.append(
                {
                    "path": rel,
                    "kind": kind,
                    "size_bytes": int(stat_info.st_size),
                    "mtime_iso": _iso_timestamp(stat_info.st_mtime),
                    "atime_iso": _iso_timestamp(stat_info.st_atime),
                    "mode": oct(stat.S_IMODE(stat_info.st_mode)),
                }
            )

            if kind == "dir" and entry.name not in EPHEMERAL_DIR_NAMES:
                walk(entry_path)

    walk(root)
    out = pd.DataFrame(rows, columns=FILESYSTEM_COLUMNS)
    return out.sort_values(["path", "kind"]).reset_index(drop=True)


def _normalize_external_note(role: str, note: str) -> str:
    role = "" if pd.isna(role) else str(role)
    note = "" if pd.isna(note) else str(note)
    if role == "excluded":
        return "Excluded: original 3-3 image with bead out of view."
    return note


def _cohort_candidate_rows(spec: CohortSpec, root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if spec.external_manifest_path:
        ext_path = root / spec.external_manifest_path
        ext_df = pd.read_csv(ext_path, sep="\t")
        for r in ext_df.itertuples(index=False):
            rel_path = str(r.file_path)
            rows.append(
                {
                    "cohort_id": spec.cohort_id,
                    "dataset_date": spec.dataset_date,
                    "condition": spec.condition,
                    "canonical_position": str(r.canonical_position),
                    "variant": _infer_variant(rel_path),
                    "file_path": rel_path,
                    "selected_for_analysis": int(bool(r.include_in_analysis)),
                    "external_manifest_included": float(r.include_in_analysis),
                    "external_manifest_note": _normalize_external_note(
                        getattr(r, "role", ""),
                        getattr(r, "note", ""),
                    ),
                }
            )
        return rows

    base = root / spec.relative_dir
    for path in sorted(base.glob("*.czi")):
        rows.append(
            {
                "cohort_id": spec.cohort_id,
                "dataset_date": spec.dataset_date,
                "condition": spec.condition,
                "canonical_position": _infer_canonical_position(path),
                "variant": _infer_variant(path),
                "file_path": _relpath(path, root),
                "selected_for_analysis": 1,
                "external_manifest_included": np.nan,
                "external_manifest_note": np.nan,
            }
        )
    return rows


def build_analysis_file_manifest(root: Path, czi_metadata_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for spec in COHORTS:
        rows.extend(_cohort_candidate_rows(spec=spec, root=root))

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=FILE_MANIFEST_COLUMNS)

    df["file_path"] = df["file_path"].astype(str)
    df["exists"] = df["file_path"].map(lambda p: int((root / p).exists()))

    selected = df["selected_for_analysis"].astype(int) == 1
    large_selected = (
        df.assign(_selected=selected, _large=df["variant"].eq("large"))
        .groupby(["cohort_id", "canonical_position"])["_large"]
        .transform(lambda s: int(bool((s & selected.loc[s.index]).any())))
    )
    df["include_primary_analysis"] = (
        selected & df["variant"].eq("small")
    ).astype(int)
    df["include_spatial_metadata"] = (
        selected
        & (
            df["variant"].eq("large")
            | (df["variant"].eq("small") & (~large_selected.astype(bool)))
        )
    ).astype(int)

    meta = czi_metadata_df.rename(
        columns={
            "AcquisitionDateAndTime": "acquisition_time_utc",
            "SizeC": "size_c",
            "ActiveChannels": "active_channels",
        }
    )
    meta = meta[
        [
            "path",
            "size_bytes",
            "acquisition_time_utc",
            "size_c",
            "active_channels",
        ]
    ].copy()

    center_positions = []
    for file_path in df["file_path"].astype(str):
        abs_path = root / file_path
        if not abs_path.exists():
            center_positions.append("")
            continue
        with czifile.CziFile(abs_path) as czi:
            metadata = czi.metadata(raw=False)
        center_positions.append(_extract_center_position(metadata))
    df["center_position"] = center_positions

    df = df.merge(meta, left_on="file_path", right_on="path", how="left").drop(
        columns=["path"]
    )
    df["selection_rule"] = SELECTION_RULE

    out = df[FILE_MANIFEST_COLUMNS].copy()
    out = out.sort_values(
        ["cohort_id", "canonical_position", "variant", "file_path"],
        key=lambda s: s.map(_canonical_position_sort_key) if s.name == "canonical_position" else s,
    ).reset_index(drop=True)
    return out


def build_analysis_position_manifest(file_manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = file_manifest_df.groupby(
        ["cohort_id", "dataset_date", "condition", "canonical_position"],
        dropna=False,
        sort=False,
    )
    for (cohort_id, dataset_date, condition, canonical_position), g in grouped:
        selected = g[
            (g["include_primary_analysis"].astype(int) == 1)
            | (g["include_spatial_metadata"].astype(int) == 1)
        ].copy()
        small = selected[selected["variant"] == "small"]["file_path"].tolist()
        large = selected[selected["variant"] == "large"]["file_path"].tolist()

        has_small = int(len(small) > 0)
        has_large = int(len(large) > 0)

        primary_variant = "small" if has_small else ("large" if has_large else "")
        primary_file = small[0] if has_small else (large[0] if has_large else "")
        spatial_variant = "large" if has_large else primary_variant
        spatial_file = large[0] if has_large else primary_file

        if has_small and has_large:
            pair_status = "paired"
        elif has_small:
            pair_status = "single_only"
        elif has_large:
            pair_status = "large_only"
        else:
            pair_status = "missing"

        rows.append(
            {
                "cohort_id": cohort_id,
                "dataset_date": dataset_date,
                "condition": condition,
                "canonical_position": canonical_position,
                "has_small": has_small,
                "has_large": has_large,
                "primary_analysis_variant": primary_variant,
                "primary_analysis_file": primary_file,
                "spatial_metadata_variant": spatial_variant,
                "spatial_metadata_file": spatial_file,
                "pair_status": pair_status,
                "selection_rule": SELECTION_RULE,
            }
        )

    out = pd.DataFrame(rows, columns=POSITION_MANIFEST_COLUMNS)
    out = out.sort_values(
        ["cohort_id", "canonical_position"],
        key=lambda s: s.map(_canonical_position_sort_key) if s.name == "canonical_position" else s,
    ).reset_index(drop=True)
    return out


def _pick_selected_file(
    file_manifest_df: pd.DataFrame,
    cohort_id: str,
    canonical_position: str,
    variant: str,
) -> str:
    sub = file_manifest_df[
        (file_manifest_df["cohort_id"] == cohort_id)
        & (file_manifest_df["canonical_position"] == canonical_position)
        & (file_manifest_df["variant"] == variant)
        & (
            (file_manifest_df["include_primary_analysis"].astype(int) == 1)
            | (file_manifest_df["include_spatial_metadata"].astype(int) == 1)
        )
    ]
    if sub.empty:
        return ""
    return str(sub.iloc[0]["file_path"])


def build_crosswalk(
    file_manifest_df: pd.DataFrame,
    position_manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    live_pos = position_manifest_df[position_manifest_df["cohort_id"] == "2026-01-22_day2_live"]
    fix_pos = position_manifest_df[position_manifest_df["cohort_id"] == "2026-01-22_day2_fix"]
    canonical_positions = sorted(
        set(live_pos["canonical_position"]).union(set(fix_pos["canonical_position"])),
        key=_canonical_position_sort_key,
    )

    live_by_pos = {
        str(r.canonical_position): r for r in live_pos.itertuples(index=False)
    }
    fix_by_pos = {
        str(r.canonical_position): r for r in fix_pos.itertuples(index=False)
    }

    rows: List[Dict[str, object]] = []
    for pos in canonical_positions:
        pre_small = _pick_selected_file(file_manifest_df, "2026-01-22_day2_live", pos, "small")
        pre_large = _pick_selected_file(file_manifest_df, "2026-01-22_day2_live", pos, "large")
        post_small = _pick_selected_file(file_manifest_df, "2026-01-22_day2_fix", pos, "small")
        post_large = _pick_selected_file(file_manifest_df, "2026-01-22_day2_fix", pos, "large")

        n_present = sum(bool(x) for x in [pre_small, pre_large, post_small, post_large])
        pair_status = "paired_all_4" if n_present == 4 else f"partial_{n_present}_of_4"

        live_row = live_by_pos.get(pos)
        fix_row = fix_by_pos.get(pos)
        rows.append(
            {
                "canonical_position": pos,
                "pre_live_small_file": pre_small,
                "pre_live_large_file": pre_large,
                "post_fix_small_file": post_small,
                "post_fix_large_file": post_large,
                "pre_primary_analysis_file": ""
                if live_row is None
                else str(live_row.primary_analysis_file),
                "post_primary_analysis_file": ""
                if fix_row is None
                else str(fix_row.primary_analysis_file),
                "pre_spatial_metadata_file": ""
                if live_row is None
                else str(live_row.spatial_metadata_file),
                "post_spatial_metadata_file": ""
                if fix_row is None
                else str(fix_row.spatial_metadata_file),
                "pair_status": pair_status,
            }
        )

    out = pd.DataFrame(rows, columns=CROSSWALK_COLUMNS)
    return out.sort_values(
        "canonical_position", key=lambda s: s.map(_canonical_position_sort_key)
    ).reset_index(drop=True)


def build_selection_summary(
    file_manifest_df: pd.DataFrame,
    position_manifest_df: pd.DataFrame,
    outdir: Path,
) -> str:
    lines = [
        f"created_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"selection_rule={SELECTION_RULE}",
        f"file_manifest={outdir.joinpath('analysis_file_manifest.tsv').as_posix()}",
        f"position_manifest={outdir.joinpath('analysis_position_manifest.tsv').as_posix()}",
        f"files_total={len(file_manifest_df)}",
        f"files_primary={int(file_manifest_df['include_primary_analysis'].astype(int).sum())}",
        f"files_spatial={int(file_manifest_df['include_spatial_metadata'].astype(int).sum())}",
        f"positions_total={len(position_manifest_df)}",
        "by_cohort",
    ]

    for cohort_id in sorted(position_manifest_df["cohort_id"].unique()):
        sub_files = file_manifest_df[file_manifest_df["cohort_id"] == cohort_id]
        sub_pos = position_manifest_df[position_manifest_df["cohort_id"] == cohort_id]
        lines.append(
            "  "
            + f"{cohort_id}: "
            + f"files_total={len(sub_files)}, "
            + f"files_primary={int(sub_files['include_primary_analysis'].astype(int).sum())}, "
            + f"files_spatial={int(sub_files['include_spatial_metadata'].astype(int).sum())}, "
            + f"positions={len(sub_pos)}"
        )
    return "\n".join(lines) + "\n"


def _format_signature_value(value: object) -> str:
    if pd.isna(value):
        return "None"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def build_workspace_summary(
    filesystem_df: pd.DataFrame,
    czi_metadata_df: pd.DataFrame,
) -> str:
    lines = [
        f"FILESYSTEM_ELEMENTS={len(filesystem_df)}",
        f"CZI_FILES={len(czi_metadata_df)}",
        f"TOTAL_CZI_GIB={czi_metadata_df['size_bytes'].sum() / (1024 ** 3):.3f}",
        "MISSING_COUNTS=",
        "BY_DATASET",
    ]

    dataset_names = (
        czi_metadata_df["path"]
        .astype(str)
        .str.extract(r"^data/([^/]+)", expand=False)
        .fillna("unknown")
    )
    by_dataset = (
        czi_metadata_df.assign(dataset=dataset_names)
        .groupby("dataset", as_index=False)
        .agg(files=("path", "size"), size_bytes=("size_bytes", "sum"))
        .sort_values("dataset")
    )
    for r in by_dataset.itertuples(index=False):
        lines.append(
            f"  {r.dataset}: files={int(r.files)}, GiB={float(r.size_bytes) / (1024 ** 3):.3f}"
        )

    lines.append("TOP_DIM_SIGNATURES")
    dim_groups = (
        czi_metadata_df.fillna(np.nan)
        .groupby(["SizeX", "SizeY", "SizeC", "PixelType"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    for (sx, sy, sc, pixel), n in dim_groups.items():
        lines.append(
            f"  n={int(n)} | "
            f"SizeX={_format_signature_value(sx)} "
            f"SizeY={_format_signature_value(sy)} "
            f"SizeC={_format_signature_value(sc)} "
            f"Pixel={_format_signature_value(pixel)}"
        )

    lines.append("TOP_SCALE_SIGNATURES_UM")
    scale_groups = (
        czi_metadata_df.groupby(["ScaleX_um", "ScaleY_um", "ScaleZ_um"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    for (sx, sy, sz), n in scale_groups.items():
        lines.append(
            f"  n={int(n)} | scale_um=("
            f"{_format_signature_value(sx)}, "
            f"{_format_signature_value(sy)}, "
            f"{_format_signature_value(sz)})"
        )

    lines.append("TOP_ACTIVE_CHANNEL_SETS")
    channel_groups = (
        czi_metadata_df.groupby("ActiveChannels", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    for channels, n in channel_groups.items():
        parts = [p for p in str(channels).split(";") if p]
        lines.append(f"  n={int(n)} | channels={parts}")

    lines.append("OBJECTIVES")
    objective_groups = (
        czi_metadata_df.groupby("ObjectiveName", dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    for objective, n in objective_groups.items():
        lines.append(f"  n={int(n)} | {_format_signature_value(objective)}")

    acq = pd.to_datetime(czi_metadata_df["AcquisitionDateAndTime"], errors="coerce", utc=True)
    if acq.notna().any():
        lines.append(
            f"ACQ_RANGE_UTC={acq.min().isoformat()} -> {acq.max().isoformat()}"
        )
    else:
        lines.append("ACQ_RANGE_UTC=")

    lines.append("LARGEST_FILES")
    largest = czi_metadata_df.sort_values("size_bytes", ascending=False).head(12)
    for r in largest.itertuples(index=False):
        lines.append(f"  {float(r.size_bytes) / (1024 ** 3):.3f} GiB | {r.path}")

    return "\n".join(lines) + "\n"


def write_manifest_outputs(
    outdir: Path,
    filesystem_df: pd.DataFrame,
    czi_metadata_df: pd.DataFrame,
    file_manifest_df: pd.DataFrame,
    position_manifest_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    summary_text: str,
    workspace_summary_text: str,
) -> Dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)

    filesystem_path = outdir / "filesystem_inventory.tsv"
    metadata_path = outdir / "czi_metadata.tsv"
    file_manifest_path = outdir / "analysis_file_manifest.tsv"
    position_manifest_path = outdir / "analysis_position_manifest.tsv"
    crosswalk_path = outdir / "analysis_crosswalk_2026-01-22_pre_post.tsv"
    summary_path = outdir / "analysis_selection_summary.txt"
    workspace_summary_path = outdir / "summary.txt"

    filesystem_df.to_csv(filesystem_path, sep="\t", index=False)
    czi_metadata_df.to_csv(metadata_path, sep="\t", index=False)
    file_manifest_df.to_csv(file_manifest_path, sep="\t", index=False)
    position_manifest_df.to_csv(position_manifest_path, sep="\t", index=False)
    crosswalk_df.to_csv(crosswalk_path, sep="\t", index=False)
    summary_path.write_text(summary_text)
    workspace_summary_path.write_text(workspace_summary_text)

    return {
        "filesystem_path": str(filesystem_path),
        "metadata_path": str(metadata_path),
        "file_manifest_path": str(file_manifest_path),
        "position_manifest_path": str(position_manifest_path),
        "crosswalk_path": str(crosswalk_path),
        "summary_path": str(summary_path),
        "workspace_summary_path": str(workspace_summary_path),
    }


def run_manifest_pipeline(
    root: Union[str, Path] = ".",
    outdir: Union[str, Path] = "results/manifests",
    write_outputs: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    root = Path(root).resolve()
    outdir = (root / outdir).resolve() if not Path(outdir).is_absolute() else Path(outdir)

    if verbose:
        print(f"Building filesystem inventory from {root}")
    filesystem_df = build_filesystem_inventory(root=root)

    if verbose:
        print("Parsing CZI metadata from data/")
    czi_metadata_df = build_czi_metadata(root=root)

    if verbose:
        print("Building analysis file manifest")
    file_manifest_df = build_analysis_file_manifest(root=root, czi_metadata_df=czi_metadata_df)

    if verbose:
        print("Building analysis position manifest")
    position_manifest_df = build_analysis_position_manifest(file_manifest_df=file_manifest_df)

    if verbose:
        print("Building 2026-01-22 live/fix crosswalk")
    crosswalk_df = build_crosswalk(
        file_manifest_df=file_manifest_df,
        position_manifest_df=position_manifest_df,
    )

    summary_text = build_selection_summary(
        file_manifest_df=file_manifest_df,
        position_manifest_df=position_manifest_df,
        outdir=outdir,
    )
    workspace_summary_text = build_workspace_summary(
        filesystem_df=filesystem_df,
        czi_metadata_df=czi_metadata_df,
    )

    output_paths: Dict[str, str] = {}
    if write_outputs:
        output_paths = write_manifest_outputs(
            outdir=outdir,
            filesystem_df=filesystem_df,
            czi_metadata_df=czi_metadata_df,
            file_manifest_df=file_manifest_df,
            position_manifest_df=position_manifest_df,
            crosswalk_df=crosswalk_df,
            summary_text=summary_text,
            workspace_summary_text=workspace_summary_text,
        )
        if verbose:
            for key, value in output_paths.items():
                print(f"Wrote {key}: {value}")

    return {
        "filesystem_df": filesystem_df,
        "czi_metadata_df": czi_metadata_df,
        "file_manifest_df": file_manifest_df,
        "position_manifest_df": position_manifest_df,
        "crosswalk_df": crosswalk_df,
        "summary_text": summary_text,
        "workspace_summary_text": workspace_summary_text,
        "output_paths": output_paths,
    }


def main() -> None:
    args = parse_args()
    run_manifest_pipeline(
        root=args.root,
        outdir=args.outdir,
        write_outputs=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
