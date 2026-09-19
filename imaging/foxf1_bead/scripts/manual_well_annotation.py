#!/usr/bin/env python3
"""
Manual well annotation helpers.

Includes:
  - Legacy square-ROI annotation + in-ROI center estimation.
  - Manual centroid annotation (multi-point, z-stack aware).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from IPython.display import Image as IPyImage
from IPython.display import display
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import RectangleSelector
from scipy import ndimage as ndi
from skimage import draw, feature, filters, measure, morphology, transform
from skimage.registration import phase_cross_correlation

# `scripts` is a package under the assay directory; put that directory on the path so
# this module imports from anywhere, as the READMEs show it being run.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from scripts import pipeline_common as common


ROI_COLUMNS = [
    "image_id",
    "cohort_id",
    "canonical_position",
    "file_path",
    "x0",
    "y0",
    "x1",
    "y1",
    "side_px",
    "updated_at",
]

CENTER_COLUMNS = [
    "image_id",
    "cohort_id",
    "canonical_position",
    "file_path",
    "well_center_x_px",
    "well_center_y_px",
    "well_radius_px",
    "well_edge_support",
    "bead_center_x_px",
    "bead_center_y_px",
    "bead_area_px",
    "bead_threshold",
    "roi_x0",
    "roi_y0",
    "roi_x1",
    "roi_y1",
    "qc_flag",
    "qc_notes",
    "updated_at",
]

CENTROID_COLUMNS = [
    "image_id",
    "cohort_id",
    "canonical_position",
    "file_path",
    "centroid_index",
    "centroid_x_px",
    "centroid_y_px",
    "z_index",
    "updated_at",
]

LARGE_CENTROID_COLUMNS = [
    "image_id",
    "cohort_id",
    "canonical_position",
    "small_file_path",
    "large_file_path",
    "annotation_status",
    "centroid_index",
    "centroid_x_px",
    "centroid_y_px",
    "updated_at",
]

PAIR_TRANSFORM_COLUMNS = [
    "image_id",
    "cohort_id",
    "canonical_position",
    "small_file_path",
    "large_file_path",
    "registration_status",
    "small_shape_y_px",
    "small_shape_x_px",
    "large_shape_y_px",
    "large_shape_x_px",
    "small_to_large_top_left_x_px",
    "small_to_large_top_left_y_px",
    "small_to_large_theta_deg",
    "small_to_large_best_z_index",
    "small_to_large_score",
    "small_to_large_alignment_method",
    "small_to_large_alignment_note",
    "large_focus_z_index",
    "large_focus_score",
    "restitch_nominal_dy_px",
    "restitch_nominal_dx_px",
    "restitch_tr_y_px",
    "restitch_tr_x_px",
    "restitch_bl_y_px",
    "restitch_bl_x_px",
    "restitch_br_y_px",
    "restitch_br_x_px",
    "updated_at",
]


@dataclass
class ImageRecord:
    image_id: str
    cohort_id: str
    canonical_position: str
    file_path: str


@dataclass
class LoadedImage:
    record: ImageRecord
    abs_path: Path
    bf: np.ndarray
    af647: np.ndarray
    fused: np.ndarray
    channels: List[str]


@dataclass
class LoadedImageStack:
    record: ImageRecord
    abs_path: Path
    bf_stack: np.ndarray  # (Z, Y, X)
    af647_stack: np.ndarray  # (Z, Y, X)
    fused_stack: np.ndarray  # (Z, Y, X, 3)
    channels: List[str]


@dataclass
class PairedImageRecord:
    image_id: str
    cohort_id: str
    canonical_position: str
    small_file_path: str
    large_file_path: str


@dataclass
class LoadedLargeRegisteredStack:
    pair: PairedImageRecord
    abs_large_path: Path
    bf_stack: np.ndarray  # (Z, Y, X)
    af647_stack: np.ndarray  # (Z, Y, X)
    fused_stack: np.ndarray  # (Z, Y, X, 3)
    channels: List[str]
    restitch_nominal_dy_px: int
    restitch_nominal_dx_px: int
    restitch_tr_y_px: int
    restitch_tr_x_px: int
    restitch_bl_y_px: int
    restitch_bl_x_px: int
    restitch_br_y_px: int
    restitch_br_x_px: int


def robust_rescale(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.quantile(arr[finite], [p_low / 100.0, p_high / 100.0])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def fuse_bf_647(bf: np.ndarray, af647: np.ndarray) -> np.ndarray:
    bf_n = robust_rescale(bf)
    af_n = robust_rescale(af647)
    rgb = np.zeros((bf.shape[0], bf.shape[1], 3), dtype=np.float32)
    rgb[..., 1] = bf_n
    rgb[..., 2] = bf_n
    rgb[..., 0] = np.maximum(af_n, 0.25 * bf_n)
    return np.clip(rgb, 0.0, 1.0)


def _resolve_local_path(path_like: str, root: Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (root / p)


def load_roi_table(roi_path: Path) -> pd.DataFrame:
    if not roi_path.exists():
        return pd.DataFrame(columns=ROI_COLUMNS)
    df = pd.read_csv(roi_path, sep="\t")
    for col in ROI_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[ROI_COLUMNS].copy()


def upsert_roi_row(roi_path: Path, row: Dict[str, object]) -> None:
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_roi_table(roi_path)
    row_df = pd.DataFrame([row])[ROI_COLUMNS]
    if df.empty:
        out = row_df
    else:
        key = row["file_path"]
        df = df[df["file_path"] != key].copy()
        out = pd.concat([df, row_df], ignore_index=True)
    out = out.sort_values(["cohort_id", "canonical_position"]).reset_index(drop=True)
    out.to_csv(roi_path, sep="\t", index=False)


def load_center_table(center_path: Path) -> pd.DataFrame:
    if not center_path.exists():
        return pd.DataFrame(columns=CENTER_COLUMNS)
    df = pd.read_csv(center_path, sep="\t")
    for col in CENTER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[CENTER_COLUMNS].copy()


def upsert_center_row(center_path: Path, row: Dict[str, object]) -> None:
    center_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_center_table(center_path)
    row_df = pd.DataFrame([row])[CENTER_COLUMNS]
    if df.empty:
        out = row_df
    else:
        key = row["file_path"]
        df = df[df["file_path"] != key].copy()
        out = pd.concat([df, row_df], ignore_index=True)
    out = out.sort_values(["cohort_id", "canonical_position"]).reset_index(drop=True)
    out.to_csv(center_path, sep="\t", index=False)


def load_centroid_table(centroid_path: Path) -> pd.DataFrame:
    if not centroid_path.exists():
        return pd.DataFrame(columns=CENTROID_COLUMNS)
    df = pd.read_csv(centroid_path, sep="\t")
    for col in CENTROID_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[CENTROID_COLUMNS].copy()


def replace_centroid_rows(
    centroid_path: Path,
    rec: ImageRecord,
    points_xyz: Sequence[Tuple[float, float, int]],
) -> None:
    centroid_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_centroid_table(centroid_path)
    key = rec.file_path
    if not df.empty:
        df = df[df["file_path"] != key].copy()

    now = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict[str, object]] = []
    for idx, (x, y, z_idx) in enumerate(points_xyz, start=1):
        rows.append(
            {
                "image_id": rec.image_id,
                "cohort_id": rec.cohort_id,
                "canonical_position": rec.canonical_position,
                "file_path": rec.file_path,
                "centroid_index": int(idx),
                "centroid_x_px": float(x),
                "centroid_y_px": float(y),
                "z_index": int(z_idx),
                "updated_at": now,
            }
        )
    add_df = pd.DataFrame(rows, columns=CENTROID_COLUMNS)
    if df.empty:
        out = add_df
    elif add_df.empty:
        out = df
    else:
        out = pd.concat([df, add_df], ignore_index=True)
    if out.empty:
        out = pd.DataFrame(columns=CENTROID_COLUMNS)
    else:
        out = out.sort_values(
            ["cohort_id", "canonical_position", "centroid_index"]
        ).reset_index(drop=True)
    out.to_csv(centroid_path, sep="\t", index=False)


def load_large_centroid_table(centroid_path: Path) -> pd.DataFrame:
    if not centroid_path.exists():
        return pd.DataFrame(columns=LARGE_CENTROID_COLUMNS)
    df = pd.read_csv(centroid_path, sep="\t")
    for col in LARGE_CENTROID_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[LARGE_CENTROID_COLUMNS].copy()


def replace_large_centroid_rows(
    centroid_path: Path,
    pair: PairedImageRecord,
    points_xy: Sequence[Tuple[float, float]],
    annotation_status: str,
) -> None:
    status = str(annotation_status).strip().lower()
    if status not in {"annotated", "no_bead"}:
        raise ValueError("annotation_status must be 'annotated' or 'no_bead'")
    if status == "annotated" and len(points_xy) == 0:
        raise ValueError("Annotated rows require at least one centroid.")

    centroid_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_large_centroid_table(centroid_path)
    if not df.empty:
        df = df[df["large_file_path"] != pair.large_file_path].copy()

    now = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict[str, object]] = []
    if status == "no_bead":
        rows.append(
            {
                "image_id": pair.image_id,
                "cohort_id": pair.cohort_id,
                "canonical_position": pair.canonical_position,
                "small_file_path": pair.small_file_path,
                "large_file_path": pair.large_file_path,
                "annotation_status": "no_bead",
                "centroid_index": 0,
                "centroid_x_px": np.nan,
                "centroid_y_px": np.nan,
                "updated_at": now,
            }
        )
    else:
        for idx, (x, y) in enumerate(points_xy, start=1):
            rows.append(
                {
                    "image_id": pair.image_id,
                    "cohort_id": pair.cohort_id,
                    "canonical_position": pair.canonical_position,
                    "small_file_path": pair.small_file_path,
                    "large_file_path": pair.large_file_path,
                    "annotation_status": "annotated",
                    "centroid_index": int(idx),
                    "centroid_x_px": float(x),
                    "centroid_y_px": float(y),
                    "updated_at": now,
                }
            )

    add_df = pd.DataFrame(rows, columns=LARGE_CENTROID_COLUMNS)
    if df.empty:
        out = add_df
    elif add_df.empty:
        out = df
    else:
        out = pd.concat([df, add_df], ignore_index=True)
    out = out.sort_values(
        ["cohort_id", "canonical_position", "annotation_status", "centroid_index"]
    ).reset_index(drop=True)
    out.to_csv(centroid_path, sep="\t", index=False)


def load_pair_transform_table(transform_path: Path) -> pd.DataFrame:
    if not transform_path.exists():
        return pd.DataFrame(columns=PAIR_TRANSFORM_COLUMNS)
    df = pd.read_csv(transform_path, sep="\t")
    for col in PAIR_TRANSFORM_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[PAIR_TRANSFORM_COLUMNS].copy()


def upsert_pair_transform_row(transform_path: Path, row: Dict[str, object]) -> None:
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_pair_transform_table(transform_path)
    row_df = pd.DataFrame([row])[PAIR_TRANSFORM_COLUMNS]
    if df.empty:
        out = row_df
    else:
        key = row["large_file_path"]
        df = df[df["large_file_path"] != key].copy()
        out = pd.concat([df, row_df], ignore_index=True)
    out = out.sort_values(["cohort_id", "canonical_position"]).reset_index(drop=True)
    out.to_csv(transform_path, sep="\t", index=False)


def _canonical_position_sort_key(pos: str) -> Tuple[int, int, str]:
    m = re.fullmatch(r"(\d+)-(\d+)", str(pos).strip())
    if m:
        return (int(m.group(1)), int(m.group(2)), str(pos))
    return (10**9, 10**9, str(pos))


def paired_records_from_directory(
    root: Path,
    target_subdir: str = "data/2026-01-22_PDMS/day2/single/",
    cohort_id: str = "2026-01-22_day2_live",
    limit: Optional[int] = None,
    allow_missing_large: bool = False,
) -> List[PairedImageRecord]:
    base = (Path(root) / target_subdir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Target directory not found: {base}")

    small_paths = sorted(
        [p for p in base.glob("*.czi") if not p.name.endswith("-large.czi")]
    )
    missing_large: List[str] = []
    pairs: List[PairedImageRecord] = []
    for small_abs in small_paths:
        canonical = small_abs.stem
        large_abs = small_abs.with_name(f"{canonical}-large.czi")
        if not large_abs.exists():
            missing_large.append(canonical)
            continue
        small_rel = small_abs.relative_to(root).as_posix()
        large_rel = large_abs.relative_to(root).as_posix()
        pairs.append(
            PairedImageRecord(
                image_id=f"{cohort_id}_{canonical}",
                cohort_id=str(cohort_id),
                canonical_position=str(canonical),
                small_file_path=small_rel,
                large_file_path=large_rel,
            )
        )
    if missing_large and not bool(allow_missing_large):
        sample = ", ".join(sorted(missing_large)[:8])
        raise RuntimeError(
            "Missing strict small/large filename pairs for canonical positions: "
            f"{sample}"
        )

    pairs = sorted(
        pairs,
        key=lambda r: _canonical_position_sort_key(r.canonical_position),
    )
    if limit is not None:
        pairs = pairs[: int(limit)]
    return pairs


def large_centroid_rows_for_pairs(
    pairs: Sequence[PairedImageRecord],
    centroid_path: Path,
) -> pd.DataFrame:
    df = load_large_centroid_table(centroid_path)
    if df.empty:
        return df.copy()
    wanted = {p.large_file_path for p in pairs}
    out = df[df["large_file_path"].isin(wanted)].copy()
    if out.empty:
        return out
    out["centroid_index"] = pd.to_numeric(out["centroid_index"], errors="coerce")
    out = out.sort_values(
        ["cohort_id", "canonical_position", "annotation_status", "centroid_index"]
    ).reset_index(drop=True)
    return out


def large_centroid_progress_for_pairs(
    pairs: Sequence[PairedImageRecord],
    centroid_path: Path,
) -> pd.DataFrame:
    rec_df = pd.DataFrame(
        [
            {
                "image_id": p.image_id,
                "cohort_id": p.cohort_id,
                "canonical_position": p.canonical_position,
                "small_file_path": p.small_file_path,
                "large_file_path": p.large_file_path,
            }
            for p in pairs
        ]
    )
    if rec_df.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "cohort_id",
                "canonical_position",
                "small_file_path",
                "large_file_path",
                "annotation_status",
                "n_centroids",
                "has_record",
            ]
        )

    rows = large_centroid_rows_for_pairs(pairs=pairs, centroid_path=centroid_path)
    if rows.empty:
        rec_df["annotation_status"] = "unannotated"
        rec_df["n_centroids"] = 0
        rec_df["has_record"] = False
        return rec_df

    rows["centroid_index"] = pd.to_numeric(rows["centroid_index"], errors="coerce").fillna(0)
    ag = rows.groupby("large_file_path", as_index=False).agg(
        annotation_status=("annotation_status", lambda s: "no_bead" if "no_bead" in set(s) else "annotated"),
        n_centroids=("centroid_index", lambda s: int((s > 0).sum())),
    )
    out = rec_df.merge(ag, on="large_file_path", how="left")
    out["annotation_status"] = out["annotation_status"].fillna("unannotated")
    out["n_centroids"] = out["n_centroids"].fillna(0).astype(int)
    out["has_record"] = out["annotation_status"] != "unannotated"
    out = out.sort_values(["cohort_id", "canonical_position"]).reset_index(drop=True)
    return out


def records_from_manifest(
    manifest_path: Path,
    cohort_id: Optional[str] = None,
    primary_analysis_variant: Optional[str] = "small",
    include_primary_path_substring: Optional[str] = None,
    exclude_primary_path_suffixes: Optional[Sequence[str]] = None,
    exclude_primary_path_substrings: Optional[Sequence[str]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[ImageRecord]:
    df = pd.read_csv(manifest_path, sep="\t")

    if cohort_id is not None:
        df = df[df["cohort_id"] == cohort_id].copy()
    if primary_analysis_variant is not None:
        df = df[df["primary_analysis_variant"] == primary_analysis_variant].copy()

    primary_paths = df["primary_analysis_file"].astype(str)
    if include_primary_path_substring:
        df = df[
            primary_paths.str.contains(include_primary_path_substring, regex=False, na=False)
        ].copy()
        primary_paths = df["primary_analysis_file"].astype(str)
    if exclude_primary_path_suffixes:
        for suffix in exclude_primary_path_suffixes:
            df = df[~primary_paths.str.endswith(str(suffix), na=False)].copy()
            primary_paths = df["primary_analysis_file"].astype(str)
    if exclude_primary_path_substrings:
        for token in exclude_primary_path_substrings:
            df = df[~primary_paths.str.contains(str(token), regex=False, na=False)].copy()
            primary_paths = df["primary_analysis_file"].astype(str)

    if canonical_positions:
        df = df[df["canonical_position"].isin(list(canonical_positions))].copy()

    df = df.sort_values(["cohort_id", "canonical_position"]).reset_index(drop=True)
    if limit is not None:
        df = df.head(int(limit)).copy()

    records = [
        ImageRecord(
            image_id=f"{r.cohort_id}_{r.canonical_position}",
            cohort_id=str(r.cohort_id),
            canonical_position=str(r.canonical_position),
            file_path=str(r.primary_analysis_file),
        )
        for r in df.itertuples(index=False)
    ]
    return records


def centroid_rows_for_records(
    records: Sequence[ImageRecord],
    centroid_path: Path,
) -> pd.DataFrame:
    centroids = load_centroid_table(centroid_path)
    if centroids.empty:
        return centroids.copy()
    wanted = {r.file_path for r in records}
    out = centroids[centroids["file_path"].isin(wanted)].copy()
    if out.empty:
        return out
    out["centroid_index"] = pd.to_numeric(out["centroid_index"], errors="coerce")
    out = out.sort_values(
        ["cohort_id", "canonical_position", "centroid_index"]
    ).reset_index(drop=True)
    return out


def centroid_progress_for_records(
    records: Sequence[ImageRecord],
    centroid_path: Path,
) -> pd.DataFrame:
    rec_df = pd.DataFrame(
        [
            {
                "image_id": r.image_id,
                "cohort_id": r.cohort_id,
                "canonical_position": r.canonical_position,
                "file_path": r.file_path,
            }
            for r in records
        ]
    )
    if rec_df.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "cohort_id",
                "canonical_position",
                "file_path",
                "n_centroids",
                "has_centroid",
            ]
        )

    cent = centroid_rows_for_records(records=records, centroid_path=centroid_path)
    if cent.empty:
        rec_df["n_centroids"] = 0
        rec_df["has_centroid"] = False
        return rec_df

    counts = (
        cent.groupby("file_path", as_index=False)["centroid_index"]
        .count()
        .rename(columns={"centroid_index": "n_centroids"})
    )
    out = rec_df.merge(counts, on="file_path", how="left")
    out["n_centroids"] = out["n_centroids"].fillna(0).astype(int)
    out["has_centroid"] = out["n_centroids"] > 0
    out = out.sort_values(["cohort_id", "canonical_position"]).reset_index(drop=True)
    return out


def demo_records_from_manifest(
    manifest_path: Path,
    cohort_id: str = "2026-01-22_day2_live",
    limit: int = 3,
) -> List[ImageRecord]:
    return records_from_manifest(
        manifest_path=manifest_path,
        cohort_id=cohort_id,
        primary_analysis_variant="small",
        limit=limit,
    )


def _collapse_to_czyx(arr: np.ndarray, axes: str) -> Tuple[np.ndarray, str]:
    """
    Convert any CZI stack variant to (C, Z, Y, X), max-projecting over T/other axes.
    """
    out = arr
    out_axes = axes

    if "C" not in out_axes:
        raise ValueError(f"Missing channel axis. axes={axes}")
    if "Y" not in out_axes or "X" not in out_axes:
        raise ValueError(f"Missing spatial axes. axes={axes}")

    c_idx = out_axes.index("C")
    out = np.moveaxis(out, c_idx, 0)
    out_axes = "C" + out_axes[:c_idx] + out_axes[c_idx + 1 :]

    if "Z" not in out_axes:
        out = np.expand_dims(out, axis=1)
        out_axes = "CZ" + out_axes[1:]
    else:
        z_idx = out_axes.index("Z")
        out = np.moveaxis(out, z_idx, 1)
        out_axes = "CZ" + out_axes[1:z_idx] + out_axes[z_idx + 1 :]

    # Collapse temporal and any non-C/Z/Y/X axes.
    for ax_name in list(out_axes):
        if ax_name in ("C", "Z", "Y", "X"):
            continue
        axis = out_axes.index(ax_name)
        out = out.max(axis=axis)
        out_axes = out_axes.replace(ax_name, "")

    # Reorder to C, Z, Y, X.
    perm = [out_axes.index("C"), out_axes.index("Z"), out_axes.index("Y"), out_axes.index("X")]
    out = np.transpose(out, axes=perm)
    return out.astype(np.float32), "CZYX"


def load_primary_image(record: ImageRecord, root: Path) -> LoadedImage:
    abs_path = _resolve_local_path(record.file_path, root=root)
    czi = common.read_czi(abs_path)
    bf_idx = common.find_channel_index(czi.channels, ["bright"])
    bead_idx = common.find_channel_index(czi.channels, ["647", "alexa fluor 647"])
    if bf_idx is None or bead_idx is None:
        raise RuntimeError(
            f"Missing required channels in {record.file_path}. channels={czi.channels}"
        )
    bf = czi.channel_images[bf_idx].astype(np.float32)
    af647 = czi.channel_images[bead_idx].astype(np.float32)
    fused = fuse_bf_647(bf, af647)
    return LoadedImage(
        record=record,
        abs_path=abs_path,
        bf=bf,
        af647=af647,
        fused=fused,
        channels=list(czi.channels),
    )


def load_primary_channel_image(
    record: ImageRecord,
    root: Path,
    channel_keywords: Sequence[str],
) -> np.ndarray:
    abs_path = _resolve_local_path(record.file_path, root=root)
    czi = common.read_czi(abs_path)
    ch_idx = common.find_channel_index(czi.channels, list(channel_keywords))
    if ch_idx is None:
        raise RuntimeError(
            f"Missing channel {list(channel_keywords)} in {record.file_path}. "
            f"channels={czi.channels}"
        )
    return czi.channel_images[ch_idx].astype(np.float32)


def load_primary_image_stack(record: ImageRecord, root: Path) -> LoadedImageStack:
    abs_path = _resolve_local_path(record.file_path, root=root)
    with _czifile().CziFile(abs_path) as czi:
        raw = czi.asarray()
        raw_axes = str(czi.axes)
        metadata = czi.metadata(raw=False)

    squeezed, squeezed_axes = common._squeeze_axes(raw, raw_axes)
    czyx, _ = _collapse_to_czyx(squeezed, squeezed_axes)
    channels = common._extract_channels(metadata, c_count=czyx.shape[0])

    bf_idx = common.find_channel_index(channels, ["bright"])
    bead_idx = common.find_channel_index(channels, ["647", "alexa fluor 647"])
    if bf_idx is None or bead_idx is None:
        raise RuntimeError(
            f"Missing required channels in {record.file_path}. channels={channels}"
        )

    bf_stack = czyx[bf_idx].astype(np.float32)
    af647_stack = czyx[bead_idx].astype(np.float32)
    fused_stack = np.stack(
        [fuse_bf_647(bf_stack[z], af647_stack[z]) for z in range(bf_stack.shape[0])],
        axis=0,
    )
    return LoadedImageStack(
        record=record,
        abs_path=abs_path,
        bf_stack=bf_stack,
        af647_stack=af647_stack,
        fused_stack=fused_stack,
        channels=list(channels),
    )


def _zscore_norm(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    return (arr - float(arr.mean())) / (float(arr.std()) + 1e-6)


def _compose_hard_mosaic_2x2(
    tiles: Dict[Tuple[int, int], np.ndarray],
    placements: Dict[Tuple[int, int], Tuple[int, int]],
    draw_order: Sequence[Tuple[int, int]] = ((0, 0), (0, 1), (1, 0), (1, 1)),
) -> np.ndarray:
    if len(tiles) == 0:
        raise ValueError("No tiles provided for hard mosaic composition.")
    tile_h, tile_w = next(iter(tiles.values())).shape
    ys = [placements[k][0] for k in placements.keys()]
    xs = [placements[k][1] for k in placements.keys()]

    miny = min(ys)
    minx = min(xs)
    maxy = max(y + tile_h for y in ys)
    maxx = max(x + tile_w for x in xs)
    H = int(maxy - miny)
    W = int(maxx - minx)

    out = np.zeros((H, W), dtype=np.float32)
    for key in draw_order:
        if key not in tiles or key not in placements:
            continue
        y, x = placements[key]
        yy = int(y - miny)
        xx = int(x - minx)
        out[yy : yy + tile_h, xx : xx + tile_w] = tiles[key]
    return out


def _estimate_registered_positions_2x2(
    tl_img: np.ndarray,
    tr_img: np.ndarray,
    bl_img: np.ndarray,
    br_img: np.ndarray,
    nominal_dx_px: int,
    nominal_dy_px: int,
    upsample_factor: int = 20,
    max_shift_from_nominal_px: int = 96,
) -> Dict[str, Tuple[int, int]]:
    ovx = int(tl_img.shape[1] - nominal_dx_px)
    ovy = int(tl_img.shape[0] - nominal_dy_px)
    if ovx < 24 or ovy < 24:
        return {
            "tr": (0, int(nominal_dx_px)),
            "bl": (int(nominal_dy_px), 0),
            "br": (int(nominal_dy_px), int(nominal_dx_px)),
        }

    tl = _zscore_norm(tl_img)
    tr = _zscore_norm(tr_img)
    bl = _zscore_norm(bl_img)
    br = _zscore_norm(br_img)

    def _clip_pos(pos_y: float, pos_x: float, nom_y: int, nom_x: int) -> Tuple[int, int]:
        yy = int(round(np.clip(pos_y, nom_y - max_shift_from_nominal_px, nom_y + max_shift_from_nominal_px)))
        xx = int(round(np.clip(pos_x, nom_x - max_shift_from_nominal_px, nom_x + max_shift_from_nominal_px)))
        return yy, xx

    try:
        shift_lr, _, _ = phase_cross_correlation(
            tl[:, nominal_dx_px:],
            tr[:, :ovx],
            upsample_factor=upsample_factor,
        )
        tr_pos = _clip_pos(float(shift_lr[0]), float(nominal_dx_px + shift_lr[1]), 0, nominal_dx_px)

        shift_tb, _, _ = phase_cross_correlation(
            tl[nominal_dy_px:, :],
            bl[:ovy, :],
            upsample_factor=upsample_factor,
        )
        bl_pos = _clip_pos(float(nominal_dy_px + shift_tb[0]), float(shift_tb[1]), nominal_dy_px, 0)

        shift_trbr, _, _ = phase_cross_correlation(
            tr[nominal_dy_px:, :],
            br[:ovy, :],
            upsample_factor=upsample_factor,
        )
        br_from_tr = (
            float(tr_pos[0] + nominal_dy_px + shift_trbr[0]),
            float(tr_pos[1] + shift_trbr[1]),
        )

        shift_blbr, _, _ = phase_cross_correlation(
            bl[:, nominal_dx_px:],
            br[:, :ovx],
            upsample_factor=upsample_factor,
        )
        br_from_bl = (
            float(bl_pos[0] + shift_blbr[0]),
            float(bl_pos[1] + nominal_dx_px + shift_blbr[1]),
        )

        br_pos = _clip_pos(
            pos_y=0.5 * (br_from_tr[0] + br_from_bl[0]),
            pos_x=0.5 * (br_from_tr[1] + br_from_bl[1]),
            nom_y=nominal_dy_px,
            nom_x=nominal_dx_px,
        )
    except Exception:
        return {
            "tr": (0, int(nominal_dx_px)),
            "bl": (int(nominal_dy_px), 0),
            "br": (int(nominal_dy_px), int(nominal_dx_px)),
        }

    return {
        "tr": tr_pos,
        "bl": bl_pos,
        "br": br_pos,
    }


def load_large_registered_channel_stack(
    pair: PairedImageRecord,
    root: Path,
    channel_keywords: Sequence[str],
    scene_index: int = 0,
    draw_order: Sequence[Tuple[int, int]] = ((0, 0), (0, 1), (1, 0), (1, 1)),
    upsample_factor: int = 20,
) -> np.ndarray:
    abs_large_path = _resolve_local_path(pair.large_file_path, root=root)
    with _czifile().CziFile(abs_large_path) as czi:
        axes = str(czi.axes)
        metadata = czi.metadata(raw=False)
        fsbd = czi.filtered_subblock_directory
        shape = tuple(int(v) for v in czi.shape)

        axis_idx = {ax: i for i, ax in enumerate(axes)}
        if "C" not in axis_idx or "Y" not in axis_idx or "X" not in axis_idx:
            raise RuntimeError(f"Unexpected large CZI axes {axes} for {pair.large_file_path}")
        c_count = shape[axis_idx["C"]]
        channels = common._extract_channels(metadata, c_count=int(c_count))

        bf_idx = common.find_channel_index(channels, ["bright"])
        target_idx = common.find_channel_index(channels, list(channel_keywords))
        if bf_idx is None or target_idx is None:
            raise RuntimeError(
                f"Missing required channels in {pair.large_file_path}. "
                f"target={list(channel_keywords)} channels={channels}"
            )

        tile_map: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        z_set: set[int] = set()
        y_set: set[int] = set()
        x_set: set[int] = set()
        for d in fsbd:
            start = d.start
            s_idx = int(start[axis_idx["S"]]) if "S" in axis_idx else 0
            c_idx = int(start[axis_idx["C"]])
            z_idx = int(start[axis_idx["Z"]]) if "Z" in axis_idx else 0
            y0 = int(start[axis_idx["Y"]])
            x0 = int(start[axis_idx["X"]])
            if s_idx != int(scene_index):
                continue
            if c_idx not in (int(bf_idx), int(target_idx)):
                continue
            tile = np.asarray(d.data_segment().data(raw=False, resize=True), dtype=np.float32)
            tile = np.squeeze(tile)
            if tile.ndim != 2:
                raise RuntimeError(
                    f"Unexpected tile ndim={tile.ndim} for {pair.large_file_path} start={start}"
                )
            tile_map[(c_idx, z_idx, y0, x0)] = tile
            z_set.add(z_idx)
            y_set.add(y0)
            x_set.add(x0)

    y_starts = sorted(y_set)
    x_starts = sorted(x_set)
    z_values = sorted(z_set)
    if len(y_starts) != 2 or len(x_starts) != 2:
        raise RuntimeError(
            f"Expected 2x2 tile mosaic for {pair.large_file_path}; got y={y_starts}, x={x_starts}"
        )
    if len(z_values) == 0:
        raise RuntimeError(f"No z planes found in large file {pair.large_file_path}")

    y0, y1 = y_starts
    x0, x1 = x_starts
    nominal_dy = int(y1 - y0)
    nominal_dx = int(x1 - x0)

    z_ref = z_values[len(z_values) // 2]
    key_tl = (int(bf_idx), int(z_ref), int(y0), int(x0))
    key_tr = (int(bf_idx), int(z_ref), int(y0), int(x1))
    key_bl = (int(bf_idx), int(z_ref), int(y1), int(x0))
    key_br = (int(bf_idx), int(z_ref), int(y1), int(x1))
    for key in [key_tl, key_tr, key_bl, key_br]:
        if key not in tile_map:
            raise RuntimeError(
                f"Missing expected BF tile key={key} in {pair.large_file_path}"
            )

    reg_pos = _estimate_registered_positions_2x2(
        tl_img=tile_map[key_tl],
        tr_img=tile_map[key_tr],
        bl_img=tile_map[key_bl],
        br_img=tile_map[key_br],
        nominal_dx_px=nominal_dx,
        nominal_dy_px=nominal_dy,
        upsample_factor=int(upsample_factor),
    )
    placements = {
        (0, 0): (0, 0),
        (0, 1): reg_pos["tr"],
        (1, 0): reg_pos["bl"],
        (1, 1): reg_pos["br"],
    }

    slices: List[np.ndarray] = []
    for z_idx in z_values:
        ch_tiles = {
            (0, 0): tile_map[(int(target_idx), int(z_idx), int(y0), int(x0))],
            (0, 1): tile_map[(int(target_idx), int(z_idx), int(y0), int(x1))],
            (1, 0): tile_map[(int(target_idx), int(z_idx), int(y1), int(x0))],
            (1, 1): tile_map[(int(target_idx), int(z_idx), int(y1), int(x1))],
        }
        slices.append(
            _compose_hard_mosaic_2x2(
                tiles=ch_tiles,
                placements=placements,
                draw_order=draw_order,
            )
        )
    return np.stack(slices, axis=0).astype(np.float32)


def load_large_registered_stack(
    pair: PairedImageRecord,
    root: Path,
    scene_index: int = 0,
    draw_order: Sequence[Tuple[int, int]] = ((0, 0), (0, 1), (1, 0), (1, 1)),
    upsample_factor: int = 20,
) -> LoadedLargeRegisteredStack:
    abs_large_path = _resolve_local_path(pair.large_file_path, root=root)
    with _czifile().CziFile(abs_large_path) as czi:
        axes = str(czi.axes)
        metadata = czi.metadata(raw=False)
        fsbd = czi.filtered_subblock_directory
        shape = tuple(int(v) for v in czi.shape)

        axis_idx = {ax: i for i, ax in enumerate(axes)}
        if "C" not in axis_idx or "Y" not in axis_idx or "X" not in axis_idx:
            raise RuntimeError(f"Unexpected large CZI axes {axes} for {pair.large_file_path}")
        c_count = shape[axis_idx["C"]]
        channels = common._extract_channels(metadata, c_count=int(c_count))

        bf_idx = common.find_channel_index(channels, ["bright"])
        bead_idx = common.find_channel_index(channels, ["647", "alexa fluor 647"])
        if bf_idx is None or bead_idx is None:
            raise RuntimeError(
                f"Missing required channels in {pair.large_file_path}. channels={channels}"
            )

        tile_map: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        z_set: set[int] = set()
        y_set: set[int] = set()
        x_set: set[int] = set()
        for d in fsbd:
            start = d.start
            s_idx = int(start[axis_idx["S"]]) if "S" in axis_idx else 0
            c_idx = int(start[axis_idx["C"]])
            z_idx = int(start[axis_idx["Z"]]) if "Z" in axis_idx else 0
            y0 = int(start[axis_idx["Y"]])
            x0 = int(start[axis_idx["X"]])
            if s_idx != int(scene_index):
                continue
            if c_idx not in (int(bf_idx), int(bead_idx)):
                continue
            tile = np.asarray(d.data_segment().data(raw=False, resize=True), dtype=np.float32)
            tile = np.squeeze(tile)
            if tile.ndim != 2:
                raise RuntimeError(
                    f"Unexpected tile ndim={tile.ndim} for {pair.large_file_path} start={start}"
                )
            tile_map[(c_idx, z_idx, y0, x0)] = tile
            z_set.add(z_idx)
            y_set.add(y0)
            x_set.add(x0)

    y_starts = sorted(y_set)
    x_starts = sorted(x_set)
    z_values = sorted(z_set)
    if len(y_starts) != 2 or len(x_starts) != 2:
        raise RuntimeError(
            f"Expected 2x2 tile mosaic for {pair.large_file_path}; got y={y_starts}, x={x_starts}"
        )
    if len(z_values) == 0:
        raise RuntimeError(f"No z planes found in large file {pair.large_file_path}")

    y0, y1 = y_starts
    x0, x1 = x_starts
    nominal_dy = int(y1 - y0)
    nominal_dx = int(x1 - x0)

    z_ref = z_values[len(z_values) // 2]
    key_tl = (int(bf_idx), int(z_ref), int(y0), int(x0))
    key_tr = (int(bf_idx), int(z_ref), int(y0), int(x1))
    key_bl = (int(bf_idx), int(z_ref), int(y1), int(x0))
    key_br = (int(bf_idx), int(z_ref), int(y1), int(x1))
    for key in [key_tl, key_tr, key_bl, key_br]:
        if key not in tile_map:
            raise RuntimeError(
                f"Missing expected BF tile key={key} in {pair.large_file_path}"
            )

    reg_pos = _estimate_registered_positions_2x2(
        tl_img=tile_map[key_tl],
        tr_img=tile_map[key_tr],
        bl_img=tile_map[key_bl],
        br_img=tile_map[key_br],
        nominal_dx_px=nominal_dx,
        nominal_dy_px=nominal_dy,
        upsample_factor=int(upsample_factor),
    )
    placements = {
        (0, 0): (0, 0),
        (0, 1): reg_pos["tr"],
        (1, 0): reg_pos["bl"],
        (1, 1): reg_pos["br"],
    }

    bf_slices: List[np.ndarray] = []
    bead_slices: List[np.ndarray] = []
    for z_idx in z_values:
        bf_tiles = {
            (0, 0): tile_map[(int(bf_idx), int(z_idx), int(y0), int(x0))],
            (0, 1): tile_map[(int(bf_idx), int(z_idx), int(y0), int(x1))],
            (1, 0): tile_map[(int(bf_idx), int(z_idx), int(y1), int(x0))],
            (1, 1): tile_map[(int(bf_idx), int(z_idx), int(y1), int(x1))],
        }
        bead_tiles = {
            (0, 0): tile_map[(int(bead_idx), int(z_idx), int(y0), int(x0))],
            (0, 1): tile_map[(int(bead_idx), int(z_idx), int(y0), int(x1))],
            (1, 0): tile_map[(int(bead_idx), int(z_idx), int(y1), int(x0))],
            (1, 1): tile_map[(int(bead_idx), int(z_idx), int(y1), int(x1))],
        }
        bf_slices.append(
            _compose_hard_mosaic_2x2(
                tiles=bf_tiles,
                placements=placements,
                draw_order=draw_order,
            )
        )
        bead_slices.append(
            _compose_hard_mosaic_2x2(
                tiles=bead_tiles,
                placements=placements,
                draw_order=draw_order,
            )
        )

    bf_stack = np.stack(bf_slices, axis=0).astype(np.float32)
    af647_stack = np.stack(bead_slices, axis=0).astype(np.float32)
    fused_stack = np.stack(
        [fuse_bf_647(bf_stack[z], af647_stack[z]) for z in range(bf_stack.shape[0])],
        axis=0,
    )

    return LoadedLargeRegisteredStack(
        pair=pair,
        abs_large_path=abs_large_path,
        bf_stack=bf_stack,
        af647_stack=af647_stack,
        fused_stack=fused_stack,
        channels=list(channels),
        restitch_nominal_dy_px=int(nominal_dy),
        restitch_nominal_dx_px=int(nominal_dx),
        restitch_tr_y_px=int(placements[(0, 1)][0]),
        restitch_tr_x_px=int(placements[(0, 1)][1]),
        restitch_bl_y_px=int(placements[(1, 0)][0]),
        restitch_bl_x_px=int(placements[(1, 0)][1]),
        restitch_br_y_px=int(placements[(1, 1)][0]),
        restitch_br_x_px=int(placements[(1, 1)][1]),
    )


def _rotate_image_no_resize(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(float(angle_deg)) < 1e-8:
        return np.asarray(img, dtype=np.float32)
    return transform.rotate(
        img,
        angle=float(angle_deg),
        resize=False,
        order=1,
        mode="edge",
        preserve_range=True,
    ).astype(np.float32)


def _match_template_center_window(
    large_img: np.ndarray,
    small_template: np.ndarray,
    search_radius_px: int = 420,
) -> Tuple[int, int, float]:
    H, W = large_img.shape
    h, w = small_template.shape
    if h > H or w > W:
        raise RuntimeError(
            f"Small template shape {(h, w)} larger than large image {(H, W)}."
        )

    x_nom = int((W - w) // 2)
    y_nom = int((H - h) // 2)
    rad = int(max(8, search_radius_px))
    x_start = max(0, x_nom - rad)
    y_start = max(0, y_nom - rad)
    x_stop = min(W, x_nom + rad + w)
    y_stop = min(H, y_nom + rad + h)

    large_crop = large_img[y_start:y_stop, x_start:x_stop]
    corr_map = feature.match_template(large_crop, small_template, pad_input=False)
    y_loc, x_loc = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    score = float(corr_map[y_loc, x_loc])
    return int(x_start + x_loc), int(y_start + y_loc), score


def _extract_center_crop(img: np.ndarray, crop_shape_yx: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    H, W = arr.shape
    crop_h = int(np.clip(int(crop_shape_yx[0]), 1, H))
    crop_w = int(np.clip(int(crop_shape_yx[1]), 1, W))
    y0 = max(0, (H - crop_h) // 2)
    x0 = max(0, (W - crop_w) // 2)
    return arr[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _focus_score_for_large_slice(
    large_bf: np.ndarray,
    focus_crop_shape_yx: Tuple[int, int],
) -> float:
    # Focus is defined on the central region overlapping the small image,
    # since that is where the cyst should be interpretable for this workflow.
    crop = _extract_center_crop(large_bf, crop_shape_yx=focus_crop_shape_yx)
    crop = crop - float(crop.mean())
    grad = filters.sobel(crop.astype(np.float32))
    return float(np.mean(grad ** 2))


def choose_large_focus_z_index(
    large_bf_stack: np.ndarray,
    focus_crop_shape_yx: Tuple[int, int],
) -> Dict[str, object]:
    best = None
    for z_idx in range(int(large_bf_stack.shape[0])):
        score = _focus_score_for_large_slice(
            large_bf=large_bf_stack[z_idx],
            focus_crop_shape_yx=focus_crop_shape_yx,
        )
        cand = {
            "focus_z_index": int(z_idx),
            "focus_score": float(score),
        }
        if best is None or cand["focus_score"] > best["focus_score"]:
            best = cand

    if best is None:
        raise RuntimeError("Failed to choose focused large-image z slice.")
    return best


def _bright_spot_score_for_large_slice(
    af647_img: np.ndarray,
    top_fraction: float = 0.0002,
    smooth_sigma: float = 1.0,
) -> float:
    img = filters.gaussian(
        np.asarray(af647_img, dtype=np.float32),
        sigma=float(smooth_sigma),
        preserve_range=True,
    )
    vals = img[np.isfinite(img)].ravel()
    if vals.size == 0:
        return float("-inf")
    k = int(max(32, round(vals.size * float(top_fraction))))
    k = min(k, int(vals.size))
    top = np.partition(vals, vals.size - k)[-k:]
    return float(np.mean(top))


def choose_large_bead_z_index(
    af647_stack: np.ndarray,
) -> Dict[str, object]:
    best = None
    for z_idx in range(int(af647_stack.shape[0])):
        score = _bright_spot_score_for_large_slice(af647_stack[z_idx])
        cand = {
            "bead_z_index": int(z_idx),
            "bead_score": float(score),
        }
        if best is None or cand["bead_score"] > best["bead_score"]:
            best = cand

    if best is None:
        raise RuntimeError("Failed to choose bead-friendly large-image z slice.")
    return best


def _best_small_to_large_match_for_single_z(
    small_norm: np.ndarray,
    large_norm: np.ndarray,
    z_idx: int,
    coarse_angles: np.ndarray,
    coarse_step_deg: float,
    fine_step_deg: float,
    search_radius_px: int,
) -> Dict[str, object]:
    best = None
    for angle in coarse_angles:
        rotated = _rotate_image_no_resize(small_norm, float(angle))
        x0, y0, score = _match_template_center_window(
            large_img=large_norm,
            small_template=rotated,
            search_radius_px=search_radius_px,
        )
        cand = {
            "best_z_index": int(z_idx),
            "theta_deg": float(angle),
            "top_left_x_px": float(x0),
            "top_left_y_px": float(y0),
            "score": float(score),
        }
        if best is None or cand["score"] > best["score"]:
            best = cand

    if best is None:
        raise RuntimeError(f"Failed coarse small-to-large registration for z={z_idx}.")

    if fine_step_deg > 0 and coarse_step_deg > fine_step_deg:
        center_angle = float(best["theta_deg"])
        fine_angles = np.arange(
            center_angle - float(coarse_step_deg),
            center_angle + float(coarse_step_deg) + 0.5 * float(fine_step_deg),
            float(fine_step_deg),
            dtype=np.float32,
        )
        for angle in fine_angles:
            rotated = _rotate_image_no_resize(small_norm, float(angle))
            x0, y0, score = _match_template_center_window(
                large_img=large_norm,
                small_template=rotated,
                search_radius_px=search_radius_px,
            )
            if score > best["score"]:
                best = {
                    "best_z_index": int(z_idx),
                    "theta_deg": float(angle),
                    "top_left_x_px": float(x0),
                    "top_left_y_px": float(y0),
                    "score": float(score),
                }

    return best


def register_small_to_large(
    small_bf: np.ndarray,
    large_bf_stack: np.ndarray,
    allow_rotation: bool = True,
    max_abs_angle_deg: float = 6.0,
    coarse_step_deg: float = 0.5,
    fine_step_deg: float = 0.1,
    search_radius_px: int = 420,
) -> Dict[str, object]:
    small_norm = _zscore_norm(small_bf)
    if allow_rotation:
        coarse_angles = np.arange(
            -float(max_abs_angle_deg),
            float(max_abs_angle_deg) + 0.5 * float(coarse_step_deg),
            float(coarse_step_deg),
            dtype=np.float32,
        )
    else:
        coarse_angles = np.array([0.0], dtype=np.float32)

    focus = choose_large_focus_z_index(
        large_bf_stack=large_bf_stack,
        focus_crop_shape_yx=small_bf.shape,
    )
    z_idx = int(focus["focus_z_index"])
    large_norm = _zscore_norm(large_bf_stack[z_idx])
    best = _best_small_to_large_match_for_single_z(
        small_norm=small_norm,
        large_norm=large_norm,
        z_idx=z_idx,
        coarse_angles=coarse_angles,
        coarse_step_deg=float(coarse_step_deg),
        fine_step_deg=float(fine_step_deg) if allow_rotation else 0.0,
        search_radius_px=int(search_radius_px),
    )

    if best is None:
        raise RuntimeError("Failed to estimate small-to-large registration.")

    return {
        "registration_status": "ok",
        "small_to_large_top_left_x_px": float(best["top_left_x_px"]),
        "small_to_large_top_left_y_px": float(best["top_left_y_px"]),
        "small_to_large_theta_deg": float(best["theta_deg"]),
        "small_to_large_best_z_index": int(best["best_z_index"]),
        "small_to_large_score": float(best["score"]),
        "small_to_large_alignment_method": "bf_template_match",
        "small_to_large_alignment_note": "",
        "large_focus_z_index": int(focus["focus_z_index"]),
        "large_focus_score": float(focus["focus_score"]),
    }


def _fit_rigid_transform_2d(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    src = np.asarray(src_xy, dtype=np.float64)
    dst = np.asarray(dst_xy, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"Expected matching (N,2) arrays; got src={src.shape} dst={dst.shape}")
    if int(src.shape[0]) < 1:
        raise ValueError("Need at least one point for rigid fit.")
    if int(src.shape[0]) == 1:
        dx = float(dst[0, 0] - src[0, 0])
        dy = float(dst[0, 1] - src[0, 1])
        return np.array(
            [
                [1.0, 0.0, dx],
                [0.0, 1.0, dy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    H = src0.T @ dst0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = dst_mean - (R @ src_mean)
    return np.array(
        [
            [float(R[0, 0]), float(R[0, 1]), float(t[0])],
            [float(R[1, 0]), float(R[1, 1]), float(t[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fit_small_to_large_from_anchor_points(
    small_xy: np.ndarray,
    large_xy: np.ndarray,
    small_shape_yx: Tuple[int, int],
) -> Dict[str, float]:
    mat = _fit_rigid_transform_2d(src_xy=small_xy, dst_xy=large_xy)
    theta_deg = float(np.rad2deg(np.arctan2(mat[1, 0], mat[0, 0])))
    h, w = int(small_shape_yx[0]), int(small_shape_yx[1])
    cx = 0.5 * (float(w) - 1.0)
    cy = 0.5 * (float(h) - 1.0)
    center = np.array([cx, cy], dtype=np.float64)
    R = np.asarray(mat[:2, :2], dtype=np.float64)
    t = np.asarray(mat[:2, 2], dtype=np.float64)
    top_left = t - center + (R @ center)

    pred = np.asarray(
        [
            map_small_xy_to_large(
                x_small_px=float(x),
                y_small_px=float(y),
                small_shape_yx=(h, w),
                top_left_x_px=float(top_left[0]),
                top_left_y_px=float(top_left[1]),
                theta_deg=float(theta_deg),
            )
            for x, y in np.asarray(small_xy, dtype=np.float64)
        ],
        dtype=np.float64,
    )
    obs = np.asarray(large_xy, dtype=np.float64)
    residuals = np.sqrt(((pred - obs) ** 2).sum(axis=1))
    return {
        "small_to_large_top_left_x_px": float(top_left[0]),
        "small_to_large_top_left_y_px": float(top_left[1]),
        "small_to_large_theta_deg": float(theta_deg),
        "anchor_mean_residual_px": float(np.mean(residuals)) if residuals.size else np.nan,
        "anchor_max_residual_px": float(np.max(residuals)) if residuals.size else np.nan,
    }


def select_centroid_by_spatial_rule(
    centroid_df: pd.DataFrame,
    rule: str = "top_left",
    x_col: str = "centroid_x_px",
    y_col: str = "centroid_y_px",
    index_col: str = "centroid_index",
) -> pd.Series:
    if centroid_df is None or centroid_df.empty:
        raise ValueError("Centroid table is empty; cannot select a spatial anchor.")
    df = centroid_df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[index_col] = pd.to_numeric(df[index_col], errors="coerce")
    df = df[np.isfinite(df[x_col]) & np.isfinite(df[y_col])].copy()
    if df.empty:
        raise ValueError("Centroid table has no finite x/y rows; cannot select a spatial anchor.")

    rule_norm = str(rule or "").strip().lower()
    if rule_norm == "top_left":
        order = df.assign(_rank=df[x_col] + df[y_col]).sort_values(
            by=["_rank", y_col, x_col, index_col],
            ascending=[True, True, True, True],
        )
    else:
        raise ValueError(f"Unsupported centroid spatial rule: {rule!r}")
    return order.iloc[0].copy()


def anchored_top_left_from_theta_and_anchor(
    small_anchor_xy: Tuple[float, float],
    large_anchor_xy: Tuple[float, float],
    small_shape_yx: Tuple[int, int],
    theta_deg: float,
) -> Tuple[float, float]:
    h, w = int(small_shape_yx[0]), int(small_shape_yx[1])
    cx = 0.5 * (float(w) - 1.0)
    cy = 0.5 * (float(h) - 1.0)
    ang = np.deg2rad(float(theta_deg))
    sx, sy = float(small_anchor_xy[0]), float(small_anchor_xy[1])
    lx, ly = float(large_anchor_xy[0]), float(large_anchor_xy[1])
    x0 = sx - cx
    y0 = sy - cy
    xr = np.cos(ang) * x0 - np.sin(ang) * y0 + cx
    yr = np.sin(ang) * x0 + np.cos(ang) * y0 + cy
    return float(lx - xr), float(ly - yr)


def _score_small_to_large_fixed_alignment(
    small_norm: np.ndarray,
    large_norm: np.ndarray,
    top_left_x_px: float,
    top_left_y_px: float,
    theta_deg: float,
) -> float:
    small_rot = _rotate_image_no_resize(np.asarray(small_norm, dtype=np.float32), float(theta_deg))
    crop = _extract_large_crop_with_padding(
        large_img=np.asarray(large_norm, dtype=np.float32),
        top_left_x_px=float(top_left_x_px),
        top_left_y_px=float(top_left_y_px),
        out_h=int(small_rot.shape[0]),
        out_w=int(small_rot.shape[1]),
    )
    small_vec = np.asarray(small_rot, dtype=np.float32).ravel()
    crop_vec = np.asarray(crop, dtype=np.float32).ravel()
    if small_vec.size == 0 or crop_vec.size == 0:
        return float("-inf")
    small_std = float(np.std(small_vec))
    crop_std = float(np.std(crop_vec))
    if small_std <= 1e-6 or crop_std <= 1e-6:
        return float("-inf")
    small_vec = (small_vec - float(np.mean(small_vec))) / small_std
    crop_vec = (crop_vec - float(np.mean(crop_vec))) / crop_std
    return float(np.mean(small_vec * crop_vec))


def register_small_to_large_with_fixed_anchor_rotation(
    small_bf: np.ndarray,
    large_bf_stack: np.ndarray,
    small_anchor_xy: Tuple[float, float],
    large_anchor_xy: Tuple[float, float],
    initial_theta_deg: float,
    search_half_range_deg: float = 4.0,
    coarse_step_deg: float = 0.25,
    fine_step_deg: float = 0.05,
    focus_z_index: Optional[int] = None,
) -> Dict[str, float]:
    small_norm = _zscore_norm(small_bf)
    if focus_z_index is None:
        focus = choose_large_focus_z_index(
            large_bf_stack=large_bf_stack,
            focus_crop_shape_yx=small_bf.shape,
        )
        z_idx = int(focus["focus_z_index"])
        focus_score = float(focus["focus_score"])
    else:
        z_idx = int(np.clip(int(focus_z_index), 0, large_bf_stack.shape[0] - 1))
        focus_score = float(
            _focus_score_for_large_slice(
                large_bf=np.asarray(large_bf_stack[z_idx], dtype=np.float32),
                focus_crop_shape_yx=small_bf.shape,
            )
        )
    large_norm = _zscore_norm(large_bf_stack[z_idx])

    def _search(theta_values: np.ndarray) -> Dict[str, float]:
        best = None
        for theta in np.asarray(theta_values, dtype=np.float32):
            tx, ty = anchored_top_left_from_theta_and_anchor(
                small_anchor_xy=small_anchor_xy,
                large_anchor_xy=large_anchor_xy,
                small_shape_yx=small_bf.shape,
                theta_deg=float(theta),
            )
            score = _score_small_to_large_fixed_alignment(
                small_norm=small_norm,
                large_norm=large_norm,
                top_left_x_px=float(tx),
                top_left_y_px=float(ty),
                theta_deg=float(theta),
            )
            cand = {
                "theta_deg": float(theta),
                "top_left_x_px": float(tx),
                "top_left_y_px": float(ty),
                "score": float(score),
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
        if best is None:
            raise RuntimeError("Anchored rotation search failed to produce a candidate.")
        return best

    coarse_vals = np.arange(
        float(initial_theta_deg) - float(search_half_range_deg),
        float(initial_theta_deg) + float(search_half_range_deg) + 0.5 * float(coarse_step_deg),
        float(coarse_step_deg),
        dtype=np.float32,
    )
    best = _search(coarse_vals)
    if float(fine_step_deg) > 0 and float(coarse_step_deg) > float(fine_step_deg):
        fine_vals = np.arange(
            float(best["theta_deg"]) - float(coarse_step_deg),
            float(best["theta_deg"]) + float(coarse_step_deg) + 0.5 * float(fine_step_deg),
            float(fine_step_deg),
            dtype=np.float32,
        )
        best = _search(fine_vals)

    return {
        "registration_status": "ok",
        "small_to_large_top_left_x_px": float(best["top_left_x_px"]),
        "small_to_large_top_left_y_px": float(best["top_left_y_px"]),
        "small_to_large_theta_deg": float(best["theta_deg"]),
        "small_to_large_best_z_index": int(z_idx),
        "small_to_large_score": float(best["score"]),
        "small_to_large_alignment_method": "manual_top_left_anchor_rotation",
        "small_to_large_alignment_note": "Top-left well anchored; theta searched with fixed anchor.",
        "large_focus_z_index": int(z_idx),
        "large_focus_score": float(focus_score),
    }


def map_small_xy_to_large(
    x_small_px: float,
    y_small_px: float,
    small_shape_yx: Tuple[int, int],
    top_left_x_px: float,
    top_left_y_px: float,
    theta_deg: float,
) -> Tuple[float, float]:
    h, w = small_shape_yx
    cx = 0.5 * (w - 1.0)
    cy = 0.5 * (h - 1.0)
    ang = np.deg2rad(float(theta_deg))
    x0 = float(x_small_px) - cx
    y0 = float(y_small_px) - cy
    xr = np.cos(ang) * x0 - np.sin(ang) * y0 + cx
    yr = np.sin(ang) * x0 + np.cos(ang) * y0 + cy
    return float(top_left_x_px + xr), float(top_left_y_px + yr)


def build_small_fov_polygon_in_large(transform_row: pd.Series) -> np.ndarray:
    h = int(transform_row["small_shape_y_px"])
    w = int(transform_row["small_shape_x_px"])
    tx = float(transform_row["small_to_large_top_left_x_px"])
    ty = float(transform_row["small_to_large_top_left_y_px"])
    th = float(transform_row["small_to_large_theta_deg"])
    corners = np.array(
        [
            [0.0, 0.0],
            [w - 1.0, 0.0],
            [w - 1.0, h - 1.0],
            [0.0, h - 1.0],
        ],
        dtype=np.float32,
    )
    mapped = np.array(
        [
            map_small_xy_to_large(
                x_small_px=float(x),
                y_small_px=float(y),
                small_shape_yx=(h, w),
                top_left_x_px=tx,
                top_left_y_px=ty,
                theta_deg=th,
            )
            for x, y in corners
        ],
        dtype=np.float32,
    )
    return mapped


def compute_pair_transforms(
    pairs: Sequence[PairedImageRecord],
    root: Path,
    transform_path: Path,
    force_recompute: bool = False,
    allow_rotation: bool = True,
    max_abs_angle_deg: float = 6.0,
    coarse_step_deg: float = 0.5,
    fine_step_deg: float = 0.1,
    search_radius_px: int = 420,
    fallback_registration_channel_keywords: Optional[Sequence[str]] = None,
    fallback_max_center_offset_px: Optional[float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    total = len(pairs)
    t_start_all = perf_counter()
    if verbose:
        print(
            f"Computing pair transforms for {total} pair(s). "
            f"force_recompute={bool(force_recompute)}",
            flush=True,
        )

    existing = load_pair_transform_table(transform_path)
    existing_ok = set()
    if not existing.empty:
        ok = existing[existing["registration_status"] == "ok"]
        existing_ok = set(ok["large_file_path"].astype(str))

    for idx, pair in enumerate(pairs, start=1):
        label = f"[{idx}/{total}] {pair.canonical_position}"
        pair_t0 = perf_counter()
        if (not force_recompute) and (pair.large_file_path in existing_ok):
            if verbose:
                elapsed = perf_counter() - t_start_all
                eta = (elapsed / idx) * (total - idx) if idx > 0 else 0.0
                print(
                    f"{label} skip (existing ok) | elapsed={elapsed:.1f}s eta~{eta:.1f}s",
                    flush=True,
                )
            continue
        now = datetime.now().isoformat(timespec="seconds")
        large_loaded = None
        small_loaded = None
        reg = None
        try:
            large_loaded = load_large_registered_stack(pair=pair, root=root)
            small_loaded = load_primary_image(
                ImageRecord(
                    image_id=pair.image_id,
                    cohort_id=pair.cohort_id,
                    canonical_position=pair.canonical_position,
                    file_path=pair.small_file_path,
                ),
                root=root,
            )
            reg_note = ""
            reg = register_small_to_large(
                small_bf=small_loaded.bf,
                large_bf_stack=large_loaded.bf_stack,
                allow_rotation=allow_rotation,
                max_abs_angle_deg=max_abs_angle_deg,
                coarse_step_deg=coarse_step_deg,
                fine_step_deg=fine_step_deg,
                search_radius_px=search_radius_px,
            )
            if (
                fallback_registration_channel_keywords is not None
                and fallback_max_center_offset_px is not None
                and np.isfinite(float(fallback_max_center_offset_px))
            ):
                x_nom = 0.5 * float(large_loaded.bf_stack.shape[2] - small_loaded.bf.shape[1])
                y_nom = 0.5 * float(large_loaded.bf_stack.shape[1] - small_loaded.bf.shape[0])
                dx = float(reg["small_to_large_top_left_x_px"]) - float(x_nom)
                dy = float(reg["small_to_large_top_left_y_px"]) - float(y_nom)
                center_offset = float(np.hypot(dx, dy))
                if center_offset > float(fallback_max_center_offset_px):
                    reg_fb = register_small_to_large(
                        small_bf=load_primary_channel_image(
                            record=ImageRecord(
                                image_id=pair.image_id,
                                cohort_id=pair.cohort_id,
                                canonical_position=pair.canonical_position,
                                file_path=pair.small_file_path,
                            ),
                            root=root,
                            channel_keywords=fallback_registration_channel_keywords,
                        ),
                        large_bf_stack=load_large_registered_channel_stack(
                            pair=pair,
                            root=root,
                            channel_keywords=fallback_registration_channel_keywords,
                        ),
                        allow_rotation=allow_rotation,
                        max_abs_angle_deg=max_abs_angle_deg,
                        coarse_step_deg=coarse_step_deg,
                        fine_step_deg=fine_step_deg,
                        search_radius_px=search_radius_px,
                    )
                    dx_fb = float(reg_fb["small_to_large_top_left_x_px"]) - float(x_nom)
                    dy_fb = float(reg_fb["small_to_large_top_left_y_px"]) - float(y_nom)
                    center_offset_fb = float(np.hypot(dx_fb, dy_fb))
                    if center_offset_fb < center_offset:
                        reg = reg_fb
                        reg_note = (
                            f" | fallback={list(fallback_registration_channel_keywords)} "
                            f"offset={center_offset:.1f}->{center_offset_fb:.1f}"
                        )
            row = {
                "image_id": pair.image_id,
                "cohort_id": pair.cohort_id,
                "canonical_position": pair.canonical_position,
                "small_file_path": pair.small_file_path,
                "large_file_path": pair.large_file_path,
                "registration_status": "ok",
                "small_shape_y_px": int(small_loaded.bf.shape[0]),
                "small_shape_x_px": int(small_loaded.bf.shape[1]),
                "large_shape_y_px": int(large_loaded.bf_stack.shape[1]),
                "large_shape_x_px": int(large_loaded.bf_stack.shape[2]),
                "small_to_large_top_left_x_px": reg["small_to_large_top_left_x_px"],
                "small_to_large_top_left_y_px": reg["small_to_large_top_left_y_px"],
                "small_to_large_theta_deg": reg["small_to_large_theta_deg"],
                "small_to_large_best_z_index": reg["small_to_large_best_z_index"],
                "small_to_large_score": reg["small_to_large_score"],
                "small_to_large_alignment_method": reg.get("small_to_large_alignment_method", "bf_template_match"),
                "small_to_large_alignment_note": reg.get("small_to_large_alignment_note", ""),
                "large_focus_z_index": reg["large_focus_z_index"],
                "large_focus_score": reg["large_focus_score"],
                "restitch_nominal_dy_px": int(large_loaded.restitch_nominal_dy_px),
                "restitch_nominal_dx_px": int(large_loaded.restitch_nominal_dx_px),
                "restitch_tr_y_px": int(large_loaded.restitch_tr_y_px),
                "restitch_tr_x_px": int(large_loaded.restitch_tr_x_px),
                "restitch_bl_y_px": int(large_loaded.restitch_bl_y_px),
                "restitch_bl_x_px": int(large_loaded.restitch_bl_x_px),
                "restitch_br_y_px": int(large_loaded.restitch_br_y_px),
                "restitch_br_x_px": int(large_loaded.restitch_br_x_px),
                "updated_at": now,
            }
            if verbose:
                pair_dt = perf_counter() - pair_t0
                elapsed = perf_counter() - t_start_all
                eta = (elapsed / idx) * (total - idx) if idx > 0 else 0.0
                print(
                    f"{label} ok score={float(reg['small_to_large_score']):.4f} "
                    f"theta={float(reg['small_to_large_theta_deg']):.2f}deg "
                    f"focus_z={int(reg['large_focus_z_index'])} "
                    f"{reg_note}| pair={pair_dt:.1f}s elapsed={elapsed:.1f}s eta~{eta:.1f}s",
                    flush=True,
                )
        except Exception as e:
            row = {
                "image_id": pair.image_id,
                "cohort_id": pair.cohort_id,
                "canonical_position": pair.canonical_position,
                "small_file_path": pair.small_file_path,
                "large_file_path": pair.large_file_path,
                "registration_status": "error",
                "small_shape_y_px": np.nan,
                "small_shape_x_px": np.nan,
                "large_shape_y_px": np.nan,
                "large_shape_x_px": np.nan,
                "small_to_large_top_left_x_px": np.nan,
                "small_to_large_top_left_y_px": np.nan,
                "small_to_large_theta_deg": np.nan,
                "small_to_large_best_z_index": np.nan,
                "small_to_large_score": np.nan,
                "small_to_large_alignment_method": "error",
                "small_to_large_alignment_note": "",
                "large_focus_z_index": np.nan,
                "large_focus_score": np.nan,
                "restitch_nominal_dy_px": np.nan,
                "restitch_nominal_dx_px": np.nan,
                "restitch_tr_y_px": np.nan,
                "restitch_tr_x_px": np.nan,
                "restitch_bl_y_px": np.nan,
                "restitch_bl_x_px": np.nan,
                "restitch_br_y_px": np.nan,
                "restitch_br_x_px": np.nan,
                "updated_at": now,
            }
            if verbose:
                pair_dt = perf_counter() - pair_t0
                elapsed = perf_counter() - t_start_all
                eta = (elapsed / idx) * (total - idx) if idx > 0 else 0.0
                print(
                    f"{label} ERROR {type(e).__name__}: {e} "
                    f"| pair={pair_dt:.1f}s elapsed={elapsed:.1f}s eta~{eta:.1f}s",
                    flush=True,
                )
        finally:
            large_loaded = None
            small_loaded = None
            reg = None
            gc.collect()
        upsert_pair_transform_row(transform_path, row=row)

    out = load_pair_transform_table(transform_path)
    wanted = {p.large_file_path for p in pairs}
    out = out[out["large_file_path"].isin(wanted)].copy()
    out = out.sort_values(["cohort_id", "canonical_position"]).reset_index(drop=True)
    if verbose:
        elapsed = perf_counter() - t_start_all
        ok_n = int((out["registration_status"] == "ok").sum()) if not out.empty else 0
        err_n = int((out["registration_status"] == "error").sum()) if not out.empty else 0
        print(
            f"Finished pair transforms in {elapsed:.1f}s. ok={ok_n} error={err_n}.",
            flush=True,
        )
    return out


def _extract_large_crop_with_padding(
    large_img: np.ndarray,
    top_left_x_px: float,
    top_left_y_px: float,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    src = np.asarray(large_img, dtype=np.float32)
    H, W = src.shape
    out = np.zeros((int(out_h), int(out_w)), dtype=np.float32)

    x0 = int(round(float(top_left_x_px)))
    y0 = int(round(float(top_left_y_px)))
    x1 = x0 + int(out_w)
    y1 = y0 + int(out_h)

    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(W, x1)
    sy1 = min(H, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return out

    dx0 = sx0 - x0
    dy0 = sy0 - y0
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)
    out[dy0:dy1, dx0:dx1] = src[sy0:sy1, sx0:sx1]
    return out


def render_small_large_alignment_qc(
    pair: PairedImageRecord,
    root: Path,
    transform_row: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    if str(transform_row.get("registration_status", "")) != "ok":
        raise ValueError(
            f"Expected registration_status='ok' for {pair.canonical_position}; "
            f"got {transform_row.get('registration_status')!r}"
        )

    large_loaded = load_large_registered_stack(pair=pair, root=root)
    small_loaded = load_primary_image(
        ImageRecord(
            image_id=pair.image_id,
            cohort_id=pair.cohort_id,
            canonical_position=pair.canonical_position,
            file_path=pair.small_file_path,
        ),
        root=root,
    )

    z_col = "large_focus_z_index" if "large_focus_z_index" in transform_row.index else "small_to_large_best_z_index"
    z_best = int(
        np.clip(
            int(round(float(transform_row[z_col]))),
            0,
            large_loaded.bf_stack.shape[0] - 1,
        )
    )
    tx = float(transform_row["small_to_large_top_left_x_px"])
    ty = float(transform_row["small_to_large_top_left_y_px"])
    th = float(transform_row["small_to_large_theta_deg"])
    score = float(transform_row["small_to_large_score"])
    method = str(transform_row.get("small_to_large_alignment_method", "bf_template_match") or "bf_template_match")
    method_note = str(transform_row.get("small_to_large_alignment_note", "") or "")

    large_bf = large_loaded.bf_stack[z_best]
    small_bf = small_loaded.bf
    h, w = small_bf.shape

    small_rot = _rotate_image_no_resize(small_bf, th)
    large_crop = _extract_large_crop_with_padding(
        large_img=large_bf,
        top_left_x_px=tx,
        top_left_y_px=ty,
        out_h=h,
        out_w=w,
    )

    poly = build_small_fov_polygon_in_large(transform_row)
    path = np.vstack([poly, poly[0]])

    small_n = robust_rescale(small_rot)
    crop_n = robust_rescale(large_crop)
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[..., 0] = small_n
    overlay[..., 1] = crop_n
    overlay[..., 2] = crop_n
    diff = np.abs(crop_n - small_n)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), constrained_layout=True)
    axes[0].imshow(robust_rescale(large_bf), cmap="gray")
    axes[0].plot(path[:, 0], path[:, 1], color="cyan", linestyle="--", linewidth=1.4)
    title_line2 = f"score={score:.4f} theta={th:.2f}deg"
    if method != "bf_template_match":
        title_line2 = f"{method} | theta={th:.2f}deg"
        if method_note:
            title_line2 += f"\n{method_note}"
    axes[0].set_title(
        f"{pair.canonical_position} LARGE BF focus z={z_best}\n"
        f"{title_line2}"
    )

    axes[1].imshow(small_n, cmap="gray")
    axes[1].set_title("SMALL BF (rotated)")

    axes[2].imshow(crop_n, cmap="gray")
    axes[2].set_title("LARGE BF crop")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (R=small, G/B=large)")

    im = axes[4].imshow(diff, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[4].set_title("|small - large crop|")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def _plot_labeled_anchor_points(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: Sequence[str],
    edgecolor: str,
    facecolor: str = "none",
    text_color: Optional[str] = None,
    marker: str = "o",
    size: float = 120.0,
    lw: float = 1.8,
) -> None:
    pts = np.asarray(xy, dtype=np.float64)
    if pts.size == 0:
        return
    scatter_kwargs = {
        "s": float(size),
        "linewidths": float(lw),
        "marker": marker,
    }
    if str(marker) in {"x", "+", "1", "2", "3", "4"}:
        scatter_kwargs["c"] = edgecolor
    else:
        scatter_kwargs["facecolors"] = facecolor
        scatter_kwargs["edgecolors"] = edgecolor
    ax.scatter(pts[:, 0], pts[:, 1], **scatter_kwargs)
    for (x, y), lab in zip(pts, labels):
        ax.text(
            float(x) + 12.0,
            float(y) - 10.0,
            str(lab),
            color=str(text_color or edgecolor),
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.45),
        )


def render_small_large_manual_review_qc(
    pair: PairedImageRecord,
    root: Path,
    automatic_transform_row: pd.Series,
    override_transform_row: pd.Series,
    small_anchor_xy: np.ndarray,
    large_anchor_xy: np.ndarray,
    small_anchor_labels: Sequence[str],
    large_anchor_labels: Sequence[str],
    all_small_xy: Optional[np.ndarray] = None,
    all_small_labels: Optional[Sequence[str]] = None,
    all_large_xy: Optional[np.ndarray] = None,
    all_large_labels: Optional[Sequence[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    large_loaded = load_large_registered_stack(pair=pair, root=root)
    small_loaded = load_primary_image(
        ImageRecord(
            image_id=pair.image_id,
            cohort_id=pair.cohort_id,
            canonical_position=pair.canonical_position,
            file_path=pair.small_file_path,
        ),
        root=root,
    )

    z_col = "large_focus_z_index" if "large_focus_z_index" in override_transform_row.index else "small_to_large_best_z_index"
    z_best = int(
        np.clip(
            int(round(float(override_transform_row[z_col]))),
            0,
            large_loaded.bf_stack.shape[0] - 1,
        )
    )
    large_bf = np.asarray(large_loaded.bf_stack[z_best], dtype=np.float32)
    small_bf = np.asarray(small_loaded.bf, dtype=np.float32)
    h, w = small_bf.shape

    auto_tx = float(automatic_transform_row["small_to_large_top_left_x_px"])
    auto_ty = float(automatic_transform_row["small_to_large_top_left_y_px"])
    auto_th = float(automatic_transform_row["small_to_large_theta_deg"])
    over_tx = float(override_transform_row["small_to_large_top_left_x_px"])
    over_ty = float(override_transform_row["small_to_large_top_left_y_px"])
    over_th = float(override_transform_row["small_to_large_theta_deg"])

    auto_poly = build_small_fov_polygon_in_large(pd.Series(automatic_transform_row))
    over_poly = build_small_fov_polygon_in_large(pd.Series(override_transform_row))
    auto_path = np.vstack([auto_poly, auto_poly[0]])
    over_path = np.vstack([over_poly, over_poly[0]])

    auto_anchor_xy = np.asarray(
        [
            map_small_xy_to_large(
                x_small_px=float(x),
                y_small_px=float(y),
                small_shape_yx=(h, w),
                top_left_x_px=auto_tx,
                top_left_y_px=auto_ty,
                theta_deg=auto_th,
            )
            for x, y in np.asarray(small_anchor_xy, dtype=np.float64)
        ],
        dtype=np.float64,
    )
    over_anchor_xy = np.asarray(
        [
            map_small_xy_to_large(
                x_small_px=float(x),
                y_small_px=float(y),
                small_shape_yx=(h, w),
                top_left_x_px=over_tx,
                top_left_y_px=over_ty,
                theta_deg=over_th,
            )
            for x, y in np.asarray(small_anchor_xy, dtype=np.float64)
        ],
        dtype=np.float64,
    )

    auto_crop = _extract_large_crop_with_padding(
        large_img=large_bf,
        top_left_x_px=auto_tx,
        top_left_y_px=auto_ty,
        out_h=h,
        out_w=w,
    )
    over_crop = _extract_large_crop_with_padding(
        large_img=large_bf,
        top_left_x_px=over_tx,
        top_left_y_px=over_ty,
        out_h=h,
        out_w=w,
    )
    small_rot_auto = _rotate_image_no_resize(small_bf, auto_th)
    small_rot_over = _rotate_image_no_resize(small_bf, over_th)

    def _overlay(rgb_small: np.ndarray, rgb_large: np.ndarray) -> np.ndarray:
        out = np.zeros((*rgb_small.shape, 3), dtype=np.float32)
        out[..., 0] = robust_rescale(rgb_small)
        out[..., 1] = robust_rescale(rgb_large)
        out[..., 2] = robust_rescale(rgb_large)
        return out

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

    axes[0, 0].imshow(robust_rescale(small_bf), cmap="gray")
    if all_small_xy is not None and all_small_labels is not None:
        _plot_labeled_anchor_points(
            axes[0, 0],
            xy=np.asarray(all_small_xy, dtype=np.float64),
            labels=list(all_small_labels),
            edgecolor="deepskyblue",
            facecolor="none",
            size=110.0,
        )
    _plot_labeled_anchor_points(
        axes[0, 0],
        xy=np.asarray(small_anchor_xy, dtype=np.float64),
        labels=list(small_anchor_labels),
        edgecolor="gold",
        facecolor="none",
        size=150.0,
        lw=2.0,
    )
    axes[0, 0].set_title("Small BF with candidate annotations and selected anchor")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(robust_rescale(large_bf), cmap="gray")
    if all_large_xy is not None and all_large_labels is not None:
        _plot_labeled_anchor_points(
            axes[0, 1],
            xy=np.asarray(all_large_xy, dtype=np.float64),
            labels=list(all_large_labels),
            edgecolor="deepskyblue",
            facecolor="none",
            size=110.0,
        )
    _plot_labeled_anchor_points(
        axes[0, 1],
        xy=np.asarray(large_anchor_xy, dtype=np.float64),
        labels=list(large_anchor_labels),
        edgecolor="gold",
        facecolor="none",
        size=150.0,
        lw=2.0,
    )
    axes[0, 1].set_title(f"Large BF focus z={z_best} with selected target anchor")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(robust_rescale(large_bf), cmap="gray")
    axes[0, 2].plot(auto_path[:, 0], auto_path[:, 1], color="cyan", linestyle="--", linewidth=1.6)
    _plot_labeled_anchor_points(
        axes[0, 2],
        xy=np.asarray(large_anchor_xy, dtype=np.float64),
        labels=list(large_anchor_labels),
        edgecolor="gold",
        facecolor="none",
        size=145.0,
        lw=2.0,
    )
    _plot_labeled_anchor_points(
        axes[0, 2],
        xy=np.asarray(auto_anchor_xy, dtype=np.float64),
        labels=[f"S{lab}" for lab in small_anchor_labels],
        edgecolor="tomato",
        facecolor="none",
        marker="x",
        size=150.0,
        lw=2.2,
    )
    axes[0, 2].set_title(
        f"Automatic BF match on large BF\n"
        f"top_left=({auto_tx:.1f}, {auto_ty:.1f}) theta={auto_th:.2f}deg"
    )
    axes[0, 2].axis("off")

    axes[1, 0].imshow(robust_rescale(large_bf), cmap="gray")
    axes[1, 0].plot(over_path[:, 0], over_path[:, 1], color="cyan", linestyle="--", linewidth=1.6)
    _plot_labeled_anchor_points(
        axes[1, 0],
        xy=np.asarray(large_anchor_xy, dtype=np.float64),
        labels=list(large_anchor_labels),
        edgecolor="gold",
        facecolor="none",
        size=145.0,
        lw=2.0,
    )
    _plot_labeled_anchor_points(
        axes[1, 0],
        xy=np.asarray(over_anchor_xy, dtype=np.float64),
        labels=[f"S{lab}" for lab in small_anchor_labels],
        edgecolor="lime",
        facecolor="none",
        marker="x",
        size=150.0,
        lw=2.2,
    )
    axes[1, 0].set_title(
        f"Top-left anchored rotation on large BF\n"
        f"top_left=({over_tx:.1f}, {over_ty:.1f}) theta={over_th:.2f}deg"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(_overlay(small_rot_auto, auto_crop))
    axes[1, 1].set_title("Automatic BF crop overlay (R=small, G/B=large)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(_overlay(small_rot_over, over_crop))
    axes[1, 2].set_title("Anchored-rotation crop overlay (R=small, G/B=large)")
    axes[1, 2].axis("off")

    fig.suptitle(
        f"{pair.canonical_position} | Manual anchor review for small-to-large alignment",
        fontsize=16,
        y=1.01,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def render_alignment_qc_for_pairs(
    pairs: Sequence[PairedImageRecord],
    root: Path,
    transform_path: Path,
    qc_dir: Optional[Path] = None,
    show_inline: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    transform_df = load_pair_transform_table(transform_path)
    transform_by_large = {
        str(r.large_file_path): r
        for r in transform_df.itertuples(index=False)
    }

    rows: List[Dict[str, object]] = []
    total = len(pairs)
    t_start_all = perf_counter()
    if verbose:
        print(
            f"Rendering alignment QC for {total} pair(s). show_inline={bool(show_inline)}",
            flush=True,
        )

    for idx, pair in enumerate(pairs, start=1):
        label = f"[QC {idx}/{total}] {pair.canonical_position}"
        t0 = perf_counter()
        tr = transform_by_large.get(str(pair.large_file_path))
        if tr is None:
            rows.append(
                {
                    "canonical_position": pair.canonical_position,
                    "small_file_path": pair.small_file_path,
                    "large_file_path": pair.large_file_path,
                    "qc_status": "missing_transform",
                    "saved_qc_path": np.nan,
                }
            )
            if verbose:
                elapsed = perf_counter() - t_start_all
                eta = (elapsed / idx) * (total - idx) if idx > 0 else 0.0
                print(
                    f"{label} missing transform row | elapsed={elapsed:.1f}s eta~{eta:.1f}s",
                    flush=True,
                )
            continue

        if str(tr.registration_status) != "ok":
            rows.append(
                {
                    "canonical_position": pair.canonical_position,
                    "small_file_path": pair.small_file_path,
                    "large_file_path": pair.large_file_path,
                    "qc_status": f"skip_{tr.registration_status}",
                    "saved_qc_path": np.nan,
                }
            )
            if verbose:
                elapsed = perf_counter() - t_start_all
                eta = (elapsed / idx) * (total - idx) if idx > 0 else 0.0
                print(
                    f"{label} skip status={tr.registration_status} "
                    f"| elapsed={elapsed:.1f}s eta~{eta:.1f}s",
                    flush=True,
                )
            continue

        save_path = None
        if qc_dir is not None:
            save_path = (
                Path(qc_dir)
                / f"{pair.cohort_id}_{pair.canonical_position}_small_large_alignment_qc.png"
            )

        try:
            tr_s = pd.Series(tr._asdict())
            fig = render_small_large_alignment_qc(
                pair=pair,
                root=root,
                transform_row=tr_s,
                save_path=save_path,
            )
            if show_inline:
                display(fig)
            plt.close(fig)
            rows.append(
                {
                    "canonical_position": pair.canonical_position,
                    "small_file_path": pair.small_file_path,
                    "large_file_path": pair.large_file_path,
                    "qc_status": "rendered",
                    "saved_qc_path": (
                        str(save_path.relative_to(root))
                        if save_path is not None
                        else np.nan
                    ),
                }
            )
            if verbose:
                pair_dt = perf_counter() - t0
                elapsed = perf_counter() - t_start_all
                eta = (elapsed / idx) * (total - idx) if idx > 0 else 0.0
                print(
                    f"{label} rendered | pair={pair_dt:.1f}s elapsed={elapsed:.1f}s eta~{eta:.1f}s",
                    flush=True,
                )
        except Exception as e:
            rows.append(
                {
                    "canonical_position": pair.canonical_position,
                    "small_file_path": pair.small_file_path,
                    "large_file_path": pair.large_file_path,
                    "qc_status": f"error_{type(e).__name__}",
                    "saved_qc_path": np.nan,
                }
            )
            if verbose:
                pair_dt = perf_counter() - t0
                elapsed = perf_counter() - t_start_all
                eta = (elapsed / idx) * (total - idx) if idx > 0 else 0.0
                print(
                    f"{label} QC ERROR {type(e).__name__}: {e} "
                    f"| pair={pair_dt:.1f}s elapsed={elapsed:.1f}s eta~{eta:.1f}s",
                    flush=True,
                )

    out = pd.DataFrame(rows)
    if verbose:
        elapsed = perf_counter() - t_start_all
        ok_n = int((out["qc_status"] == "rendered").sum()) if not out.empty else 0
        print(f"Finished QC rendering in {elapsed:.1f}s. rendered={ok_n}/{total}.", flush=True)
    return out


def _circle_perimeter_support(
    edges: np.ndarray, cx: int, cy: int, radius: int
) -> float:
    rr, cc = draw.circle_perimeter(cy, cx, radius, shape=edges.shape)
    if rr.size == 0:
        return 0.0
    return float(edges[rr, cc].mean())


def analyze_well_and_bead_in_roi(
    bf: np.ndarray,
    af647: np.ndarray,
    roi_xyxy: Tuple[float, float, float, float],
) -> Dict[str, object]:
    x0, y0, x1, y1 = [float(v) for v in roi_xyxy]
    xi0 = int(np.floor(min(x0, x1)))
    yi0 = int(np.floor(min(y0, y1)))
    xi1 = int(np.ceil(max(x0, x1)))
    yi1 = int(np.ceil(max(y0, y1)))

    h, w = bf.shape
    xi0 = int(np.clip(xi0, 0, w - 2))
    yi0 = int(np.clip(yi0, 0, h - 2))
    xi1 = int(np.clip(xi1, xi0 + 1, w - 1))
    yi1 = int(np.clip(yi1, yi0 + 1, h - 1))

    bf_roi = bf[yi0 : yi1 + 1, xi0 : xi1 + 1].astype(np.float32)
    af_roi = af647[yi0 : yi1 + 1, xi0 : xi1 + 1].astype(np.float32)
    side_px = int(min(bf_roi.shape))

    # Bead segmentation in AF647 channel.
    af_smooth = filters.gaussian(af_roi, sigma=1.5, preserve_range=True)
    af_vals = af_smooth[np.isfinite(af_smooth)]
    bead_mask = np.zeros_like(af_roi, dtype=bool)
    bead_threshold = np.nan
    bead_center = (np.nan, np.nan)
    bead_area = 0
    bead_found = False
    if af_vals.size > 50:
        med = float(np.median(af_vals))
        mad = float(np.median(np.abs(af_vals - med))) + 1e-9
        robust_sigma = 1.4826 * mad
        thr_robust = med + 3.5 * robust_sigma
        thr_q = float(np.quantile(af_vals, 0.995))
        bead_threshold = float(max(thr_robust, thr_q))
        raw = af_smooth >= bead_threshold
        clean = morphology.remove_small_objects(raw, min_size=max(20, int(raw.size * 0.0015)))
        clean = morphology.binary_closing(clean, footprint=morphology.disk(2))
        clean = ndi.binary_fill_holes(clean)
        labels = measure.label(clean)
        props = measure.regionprops(labels, intensity_image=af_roi)
        if props:
            def _score(r: measure._regionprops.RegionProperties) -> float:
                return float(r.intensity_mean) * np.sqrt(float(r.area))

            r_best = max(props, key=_score)
            bead_mask = labels == r_best.label
            bead_area = int(r_best.area)
            cy, cx = r_best.weighted_centroid
            bead_center = (float(xi0 + cx), float(yi0 + cy))
            bead_found = True

    # Well edge and center from BF channel.
    bf_norm = robust_rescale(bf_roi, p_low=1.0, p_high=99.0)
    edges = feature.canny(bf_norm, sigma=2.0, low_threshold=0.05, high_threshold=0.18)

    r_min = max(8, int(round(side_px * 0.25)))
    r_max = max(r_min + 2, int(round(side_px * 0.55)))
    radii = np.arange(r_min, r_max + 1, 2, dtype=np.int32)

    well_center = (np.nan, np.nan)
    well_radius = np.nan
    well_edge_support = 0.0
    qc_notes: List[str] = []
    qc_flag = "ok"

    if radii.size == 0:
        qc_flag = "needs_review"
        qc_notes.append("radius_range_invalid")
        best_circle = None
    else:
        hspaces = transform.hough_circle(edges, radii)
        accums, cx_all, cy_all, rad_all = transform.hough_circle_peaks(
            hspaces,
            radii,
            total_num_peaks=8,
        )
        if len(accums) == 0:
            qc_flag = "needs_review"
            qc_notes.append("well_hough_no_peak")
            best_circle = None
        else:
            candidates = []
            for acc, cx, cy, rad in zip(accums, cx_all, cy_all, rad_all):
                support = _circle_perimeter_support(edges, int(cx), int(cy), int(rad))
                score = float(acc) * (0.2 + 0.8 * support)
                candidates.append((score, float(acc), int(cx), int(cy), int(rad), support))

            if bead_found:
                bead_cx_local = bead_center[0] - xi0
                bead_cy_local = bead_center[1] - yi0
                constrained = []
                for (score, acc, cx, cy, rad, support) in candidates:
                    d = float(np.hypot(cx - bead_cx_local, cy - bead_cy_local))
                    if d <= 0.7 * rad:
                        constrained.append((score, acc, cx, cy, rad, support, d))
                if constrained:
                    constrained = sorted(constrained, key=lambda t: (t[0], -t[6]), reverse=True)
                    _, acc, cx, cy, rad, support, _ = constrained[0]
                    best_circle = (acc, cx, cy, rad, support)
                else:
                    candidates = sorted(candidates, key=lambda t: t[0], reverse=True)
                    _, acc, cx, cy, rad, support = candidates[0]
                    best_circle = (acc, cx, cy, rad, support)
                    qc_flag = "needs_review"
                    qc_notes.append("bead_outside_well_fit")
            else:
                candidates = sorted(candidates, key=lambda t: t[0], reverse=True)
                _, acc, cx, cy, rad, support = candidates[0]
                best_circle = (acc, cx, cy, rad, support)
                qc_flag = "needs_review"
                qc_notes.append("bead_not_found")

    if best_circle is not None:
        _, cx, cy, rad, support = best_circle
        well_center = (float(xi0 + cx), float(yi0 + cy))
        well_radius = float(rad)
        well_edge_support = float(support)
        if well_edge_support < 0.12:
            qc_flag = "needs_review"
            qc_notes.append("low_edge_support")
        if bead_found:
            d = float(np.hypot(bead_center[0] - well_center[0], bead_center[1] - well_center[1]))
            if d > 0.75 * well_radius:
                qc_flag = "needs_review"
                qc_notes.append("bead_far_from_well_center")

    return {
        "roi_bounds": (xi0, yi0, xi1, yi1),
        "bf_roi": bf_roi,
        "af_roi": af_roi,
        "bf_norm": bf_norm,
        "edges": edges,
        "bead_mask": bead_mask,
        "bead_threshold": float(bead_threshold) if np.isfinite(bead_threshold) else np.nan,
        "bead_center_x_px": float(bead_center[0]),
        "bead_center_y_px": float(bead_center[1]),
        "bead_area_px": int(bead_area),
        "well_center_x_px": float(well_center[0]),
        "well_center_y_px": float(well_center[1]),
        "well_radius_px": float(well_radius) if np.isfinite(well_radius) else np.nan,
        "well_edge_support": float(well_edge_support),
        "qc_flag": qc_flag,
        "qc_notes": ";".join(qc_notes) if qc_notes else "",
    }


def build_center_row(
    rec: ImageRecord,
    roi_row: pd.Series,
    analysis: Dict[str, object],
) -> Dict[str, object]:
    xi0, yi0, xi1, yi1 = analysis["roi_bounds"]
    return {
        "image_id": rec.image_id,
        "cohort_id": rec.cohort_id,
        "canonical_position": rec.canonical_position,
        "file_path": rec.file_path,
        "well_center_x_px": analysis["well_center_x_px"],
        "well_center_y_px": analysis["well_center_y_px"],
        "well_radius_px": analysis["well_radius_px"],
        "well_edge_support": analysis["well_edge_support"],
        "bead_center_x_px": analysis["bead_center_x_px"],
        "bead_center_y_px": analysis["bead_center_y_px"],
        "bead_area_px": analysis["bead_area_px"],
        "bead_threshold": analysis["bead_threshold"],
        "roi_x0": float(min(roi_row["x0"], roi_row["x1"])),
        "roi_y0": float(min(roi_row["y0"], roi_row["y1"])),
        "roi_x1": float(max(roi_row["x0"], roi_row["x1"])),
        "roi_y1": float(max(roi_row["y0"], roi_row["y1"])),
        "qc_flag": analysis["qc_flag"],
        "qc_notes": analysis["qc_notes"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def render_qc_panels(
    loaded: LoadedImage,
    analysis: Dict[str, object],
    roi_row: pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    xi0, yi0, xi1, yi1 = analysis["roi_bounds"]
    bf = robust_rescale(loaded.bf)
    af = robust_rescale(loaded.af647)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)

    # Raw/lightly processed channels.
    axes[0, 0].imshow(bf, cmap="gray")
    axes[0, 0].set_title("Brightfield")
    axes[0, 1].imshow(af, cmap="magma")
    axes[0, 1].set_title("Alexa Fluor 647")
    axes[0, 2].imshow(loaded.fused)
    axes[0, 2].set_title("BF + 647")

    # Final overlay on BF.
    axes[0, 3].imshow(bf, cmap="gray")
    rect = Rectangle(
        (xi0, yi0),
        xi1 - xi0,
        yi1 - yi0,
        fill=False,
        edgecolor="yellow",
        linewidth=1.5,
    )
    axes[0, 3].add_patch(rect)

    bead_mask = analysis["bead_mask"]
    if bead_mask.any():
        full_mask = np.zeros_like(bf, dtype=bool)
        full_mask[yi0 : yi1 + 1, xi0 : xi1 + 1] = bead_mask
        axes[0, 3].imshow(
            np.ma.masked_where(~full_mask, full_mask),
            cmap="winter",
            alpha=0.35,
            interpolation="none",
        )

    if np.isfinite(analysis["well_center_x_px"]) and np.isfinite(analysis["well_radius_px"]):
        circ = Circle(
            (analysis["well_center_x_px"], analysis["well_center_y_px"]),
            radius=analysis["well_radius_px"],
            fill=False,
            edgecolor="cyan",
            linewidth=1.8,
        )
        axes[0, 3].add_patch(circ)
        axes[0, 3].scatter(
            [analysis["well_center_x_px"]],
            [analysis["well_center_y_px"]],
            c="red",
            s=35,
            marker="x",
            label="well center",
        )
    if np.isfinite(analysis["bead_center_x_px"]):
        axes[0, 3].scatter(
            [analysis["bead_center_x_px"]],
            [analysis["bead_center_y_px"]],
            c="lime",
            s=25,
            marker="o",
            label="bead center",
        )
    axes[0, 3].set_title("Final Overlay on BF")
    axes[0, 3].legend(loc="lower right", fontsize=7, frameon=False)

    # In-ROI classification details.
    axes[1, 0].imshow(analysis["bf_roi"], cmap="gray")
    axes[1, 0].set_title("ROI BF")
    axes[1, 1].imshow(analysis["af_roi"], cmap="magma")
    axes[1, 1].set_title("ROI 647")

    roi_647_rgb = np.dstack(
        [
            robust_rescale(analysis["af_roi"]),
            robust_rescale(analysis["af_roi"]),
            robust_rescale(analysis["af_roi"]),
        ]
    )
    mask = analysis["bead_mask"]
    roi_647_rgb[mask] = [0.0, 1.0, 1.0]
    axes[1, 2].imshow(roi_647_rgb)
    axes[1, 2].set_title("Bead Pixel Classification (ROI)")

    edge_rgb = np.dstack([analysis["bf_norm"]] * 3)
    edge_rgb[analysis["edges"]] = [1.0, 0.0, 0.0]
    axes[1, 3].imshow(edge_rgb)
    axes[1, 3].set_title(
        f"ROI BF Edges\nqc={analysis['qc_flag']} support={analysis['well_edge_support']:.3f}"
    )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{loaded.record.image_id} ({loaded.record.canonical_position})",
        fontsize=12,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


class ManualWellCentroidSession:
    """
    Interactive session for manual centroid annotation (multi-point, z-stack aware).

    Controls:
      - Left-click any panel (BF, 647, BF+647) to add centroid at current z slice.
      - Save Centers + Next: save all current centroids and advance.
      - Pass: keep existing saved centroids unchanged and advance.
      - Previous: move back.
      - Erase All: clear all centroids for current image (can be undone).
      - Undo: restore state before latest add/erase action.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        root: Path,
        centroid_path: Path,
        allow_add: bool = True,
    ):
        self.records = list(records)
        if len(self.records) == 0:
            raise ValueError("No image records provided for annotation.")
        self.root = Path(root)
        self.centroid_path = Path(centroid_path)
        self.allow_add = bool(allow_add)
        self.centroid_df = load_centroid_table(self.centroid_path)

        self.index = 0
        self.current: Optional[LoadedImageStack] = None
        self.current_points_xyz: List[Tuple[float, float, int]] = []
        self.undo_stack: List[List[Tuple[float, float, int]]] = []
        self._suspend_z_callback = False

        self._bf_show_stack: Optional[np.ndarray] = None
        self._af_show_stack: Optional[np.ndarray] = None
        self.point_artists: List[Line2D] = []
        self.point_texts: List[plt.Text] = []

        # Build figure with interactive display temporarily disabled so it is shown
        # exactly once via `render()` (avoids duplicate canvas rows in widget mode).
        with plt.ioff():
            self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        self._click_cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self.btn_prev = widgets.Button(description="Previous")
        self.btn_pass = widgets.Button(description="Pass")
        self.btn_save_next = widgets.Button(
            description="Save Centers + Next", button_style="success"
        )
        self.btn_erase_all = widgets.Button(description="Erase All")
        self.btn_undo = widgets.Button(description="Undo")
        self.z_slider = widgets.IntSlider(
            description="Z slice",
            min=0,
            max=0,
            value=0,
            step=1,
            continuous_update=False,
        )
        self.progress = widgets.HTML()
        self.point_summary = widgets.HTML()
        self.status = widgets.HTML()

        self.btn_prev.on_click(self._on_prev)
        self.btn_pass.on_click(self._on_pass)
        self.btn_save_next.on_click(self._on_save_next)
        self.btn_erase_all.on_click(self._on_erase_all)
        self.btn_undo.on_click(self._on_undo)
        self.z_slider.observe(self._on_z_change, names="value")

        self._load_current_image()

    def _set_status(self, text: str) -> None:
        self.status.value = f"<pre style='margin:0'>{text}</pre>"

    def _update_progress(self) -> None:
        self.progress.value = (
            f"<b>Image {self.index + 1} / {len(self.records)}</b> "
            f"| {self.records[self.index].image_id}"
        )

    def _update_point_summary(self) -> None:
        n = len(self.current_points_xyz)
        if self.current is None:
            z_total = 0
        else:
            z_total = int(self.current.bf_stack.shape[0])
        z_idx = int(self.z_slider.value)
        self.point_summary.value = (
            f"<b>Centroids:</b> {n} | <b>Displayed Z:</b> {z_idx + 1}/{max(z_total, 1)}"
        )

    def _existing_points_for_current(self) -> List[Tuple[float, float, int]]:
        key = self.records[self.index].file_path
        sub = self.centroid_df[self.centroid_df["file_path"] == key].copy()
        if sub.empty:
            return []
        sub["centroid_index"] = pd.to_numeric(sub["centroid_index"], errors="coerce")
        sub["centroid_x_px"] = pd.to_numeric(sub["centroid_x_px"], errors="coerce")
        sub["centroid_y_px"] = pd.to_numeric(sub["centroid_y_px"], errors="coerce")
        sub["z_index"] = pd.to_numeric(sub["z_index"], errors="coerce")
        sub = sub.sort_values("centroid_index")
        out: List[Tuple[float, float, int]] = []
        for r in sub.itertuples(index=False):
            if not np.isfinite(r.centroid_x_px) or not np.isfinite(r.centroid_y_px):
                continue
            z_idx = int(r.z_index) if np.isfinite(r.z_index) else 0
            out.append((float(r.centroid_x_px), float(r.centroid_y_px), int(z_idx)))
        return out

    def _default_z_index(
        self, points_xyz: Sequence[Tuple[float, float, int]], z_count: int
    ) -> int:
        if z_count <= 1:
            return 0
        if points_xyz:
            return int(np.clip(points_xyz[0][2], 0, z_count - 1))
        return int(z_count // 2)

    def _load_current_image(self) -> None:
        rec = self.records[self.index]
        self.current = load_primary_image_stack(rec, root=self.root)
        self._update_progress()

        self._bf_show_stack = np.stack(
            [robust_rescale(self.current.bf_stack[z]) for z in range(self.current.bf_stack.shape[0])],
            axis=0,
        )
        self._af_show_stack = np.stack(
            [robust_rescale(self.current.af647_stack[z]) for z in range(self.current.af647_stack.shape[0])],
            axis=0,
        )

        self.current_points_xyz = self._existing_points_for_current()
        self.undo_stack = []

        z_count = int(self.current.bf_stack.shape[0])
        default_z = self._default_z_index(self.current_points_xyz, z_count=z_count)
        self._suspend_z_callback = True
        self.z_slider.min = 0
        self.z_slider.max = max(0, z_count - 1)
        self.z_slider.value = int(np.clip(default_z, 0, self.z_slider.max))
        self._suspend_z_callback = False

        if self.current_points_xyz:
            if self.allow_add:
                self._set_status(
                    "Existing centroids loaded. Click to add more, then Save Centers + Next."
                )
            else:
                self._set_status(
                    "Review mode: existing centroids loaded. Use Pass/Previous to inspect."
                )
        else:
            if self.allow_add:
                self._set_status(
                    "Click any panel to add centroid(s), then Save Centers + Next."
                )
            else:
                self._set_status(
                    "Review mode: no saved centroids for this image."
                )

        self._render_current_slice()

    def _remove_point_artists(self) -> None:
        for artist in self.point_artists:
            try:
                artist.remove()
            except Exception:
                pass
        for text in self.point_texts:
            try:
                text.remove()
            except Exception:
                pass
        self.point_artists = []
        self.point_texts = []

    def _render_current_slice(self) -> None:
        if self.current is None or self._bf_show_stack is None or self._af_show_stack is None:
            return
        z_idx = int(self.z_slider.value)

        titles = [
            f"Brightfield (Z {z_idx + 1})",
            f"Alexa Fluor 647 (Z {z_idx + 1})",
            f"BF + 647 (Z {z_idx + 1})",
        ]
        imgs = [
            self._bf_show_stack[z_idx],
            self._af_show_stack[z_idx],
            self.current.fused_stack[z_idx],
        ]
        cmaps = ["gray", "magma", None]
        for ax, im, title, cmap in zip(self.axes, imgs, titles, cmaps):
            ax.clear()
            if cmap is None:
                ax.imshow(im)
            else:
                ax.imshow(im, cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self._draw_points()
        self._update_point_summary()
        self.fig.canvas.draw_idle()

    def _draw_points(self) -> None:
        self._remove_point_artists()
        if self.current is None:
            return

        z_disp = int(self.z_slider.value)
        for idx, (x, y, z_idx) in enumerate(self.current_points_xyz, start=1):
            in_plane = z_idx == z_disp
            color = "lime" if in_plane else "gold"
            marker = "o" if in_plane else "x"
            alpha = 0.95 if in_plane else 0.55
            for ax in self.axes:
                artist = ax.plot(
                    [x],
                    [y],
                    marker=marker,
                    linestyle="None",
                    markersize=8,
                    markeredgewidth=1.5,
                    color=color,
                    alpha=alpha,
                )[0]
                label = ax.text(
                    x + 4.0,
                    y - 4.0,
                    str(idx),
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    alpha=alpha,
                )
                self.point_artists.append(artist)
                self.point_texts.append(label)

    def _push_undo(self) -> None:
        self.undo_stack.append(list(self.current_points_xyz))
        if len(self.undo_stack) > 100:
            self.undo_stack = self.undo_stack[-100:]

    def _clamp_point(self, x: float, y: float, z_idx: int) -> Tuple[float, float, int]:
        if self.current is None:
            return x, y, z_idx
        z_count, h, w = self.current.bf_stack.shape
        xc = float(np.clip(x, 0, w - 1))
        yc = float(np.clip(y, 0, h - 1))
        zc = int(np.clip(z_idx, 0, z_count - 1))
        return xc, yc, zc

    def _on_click(self, event) -> None:
        if self.current is None:
            return
        if not self.allow_add:
            self._set_status("Review mode: editing is disabled.")
            return
        if event.inaxes not in list(self.axes):
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        z_idx = int(self.z_slider.value)
        x, y, z_idx = self._clamp_point(x, y, z_idx)

        self._push_undo()
        self.current_points_xyz.append((x, y, z_idx))
        self._draw_points()
        self._update_point_summary()
        self._set_status(
            f"Added centroid #{len(self.current_points_xyz)} at x={x:.1f}, y={y:.1f}, z={z_idx}."
        )
        self.fig.canvas.draw_idle()

    def _on_z_change(self, change) -> None:
        if self._suspend_z_callback:
            return
        if change.get("name") != "value":
            return
        self._render_current_slice()

    def _on_prev(self, _btn) -> None:
        if self.index == 0:
            self._set_status("Already at first image.")
            return
        self.index -= 1
        self._load_current_image()

    def _on_pass(self, _btn) -> None:
        if self.index >= len(self.records) - 1:
            self._set_status("Pass applied. Reached last image.")
            return
        self.index += 1
        self._load_current_image()

    def _on_erase_all(self, _btn) -> None:
        if not self.allow_add:
            self._set_status("Review mode: editing is disabled.")
            return
        if not self.current_points_xyz:
            self._set_status("No centroids to erase.")
            return
        self._push_undo()
        self.current_points_xyz = []
        self._draw_points()
        self._update_point_summary()
        self._set_status("Erased all centroids for this image. Use Undo to restore.")
        self.fig.canvas.draw_idle()

    def _on_undo(self, _btn) -> None:
        if not self.undo_stack:
            self._set_status("Nothing to undo.")
            return
        self.current_points_xyz = self.undo_stack.pop()
        self._draw_points()
        self._update_point_summary()
        self._set_status("Undo applied.")
        self.fig.canvas.draw_idle()

    def _on_save_next(self, _btn) -> None:
        if not self.allow_add:
            self._set_status("Review mode: use Pass/Previous to navigate.")
            return
        if self.current is None:
            self._set_status("No image loaded.")
            return
        replace_centroid_rows(
            centroid_path=self.centroid_path,
            rec=self.current.record,
            points_xyz=self.current_points_xyz,
        )
        self.centroid_df = load_centroid_table(self.centroid_path)
        self.undo_stack = []

        if self.index >= len(self.records) - 1:
            self._set_status(
                f"Saved {len(self.current_points_xyz)} centroid(s). Reached last image."
            )
            return
        self.index += 1
        self._load_current_image()

    def render(self) -> None:
        if self.allow_add:
            controls = widgets.HBox(
                [
                    self.btn_prev,
                    self.btn_pass,
                    self.btn_save_next,
                    self.btn_erase_all,
                    self.btn_undo,
                ]
            )
        else:
            controls = widgets.HBox(
                [
                    self.btn_prev,
                    self.btn_pass,
                ]
            )
        display(widgets.VBox([self.progress, self.z_slider, controls, self.point_summary, self.status]))
        display(self.fig.canvas)


class ManualLargeWellCentroidSession:
    """
    Interactive session for manual centroid annotation on registered LARGE images.

    Controls:
      - Left-click any panel (BF, 647, BF+647) to add centroid.
      - Save Centers + Next: save centroid rows and advance.
      - No Bead + Next: save explicit `no_bead` record and advance.
      - Clear No Bead: clear local `no_bead` flag (for correction before save).
      - Pass: keep existing saved entry unchanged and advance.
      - Previous: move back.
      - Erase All: clear local centroid edits (can be undone).
      - Undo: restore state before latest add/erase/no-bead action.
      - Z slider: browse z-planes (centroids are x,y only and render on every z-plane).
    """

    def __init__(
        self,
        pairs: Sequence[PairedImageRecord],
        root: Path,
        centroid_path: Path,
        transform_path: Optional[Path] = None,
        allow_add: bool = True,
    ):
        self.pairs = list(pairs)
        if len(self.pairs) == 0:
            raise ValueError("No paired records provided for large-image annotation.")
        self.root = Path(root)
        self.centroid_path = Path(centroid_path)
        self.transform_path = Path(transform_path) if transform_path is not None else None
        self.allow_add = bool(allow_add)

        self.centroid_df = load_large_centroid_table(self.centroid_path)
        self.transform_df = (
            load_pair_transform_table(self.transform_path)
            if self.transform_path is not None
            else pd.DataFrame(columns=PAIR_TRANSFORM_COLUMNS)
        )

        self.index = 0
        self.current: Optional[LoadedLargeRegisteredStack] = None
        self.current_points_xy: List[Tuple[float, float]] = []
        self.current_annotation_status: Optional[str] = None  # "annotated", "no_bead", None
        self.undo_stack: List[Tuple[List[Tuple[float, float]], Optional[str]]] = []
        self._suspend_z_callback = False
        self.current_focus_z_index: int = 0
        self.current_bead_z_index: int = 0

        self._bf_show_stack: Optional[np.ndarray] = None
        self._af_show_stack: Optional[np.ndarray] = None
        self._small_fov_polygon_xy: Optional[np.ndarray] = None
        self.point_artists: List[Line2D] = []
        self.point_texts: List[plt.Text] = []
        self.fov_artists: List[Line2D] = []

        # Build figure with interactive display temporarily disabled so it is shown
        # exactly once via `render()` (avoids duplicate canvas rows in widget mode).
        with plt.ioff():
            self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        self._click_cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self.btn_prev = widgets.Button(description="Previous")
        self.btn_pass = widgets.Button(description="Pass")
        self.btn_save_next = widgets.Button(
            description="Save Centers + Next",
            button_style="success",
        )
        self.btn_no_bead = widgets.Button(description="No Bead + Next", button_style="warning")
        self.btn_clear_no_bead = widgets.Button(description="Clear No Bead")
        self.btn_erase_all = widgets.Button(description="Erase All")
        self.btn_undo = widgets.Button(description="Undo")
        self.btn_bead_z = widgets.Button(description="Bead Z")
        self.btn_focus_z = widgets.Button(description="Cyst Focus Z")
        self.z_slider = widgets.IntSlider(
            description="Z slice",
            min=0,
            max=0,
            value=0,
            step=1,
            continuous_update=False,
        )
        self.progress = widgets.HTML()
        self.point_summary = widgets.HTML()
        self.status = widgets.HTML()

        self.btn_prev.on_click(self._on_prev)
        self.btn_pass.on_click(self._on_pass)
        self.btn_save_next.on_click(self._on_save_next)
        self.btn_no_bead.on_click(self._on_no_bead_next)
        self.btn_clear_no_bead.on_click(self._on_clear_no_bead)
        self.btn_erase_all.on_click(self._on_erase_all)
        self.btn_undo.on_click(self._on_undo)
        self.btn_bead_z.on_click(self._on_jump_bead_z)
        self.btn_focus_z.on_click(self._on_jump_focus_z)
        self.z_slider.observe(self._on_z_change, names="value")

        self._load_current_image()

    def _set_status(self, text: str) -> None:
        self.status.value = f"<pre style='margin:0'>{text}</pre>"

    def _update_progress(self) -> None:
        pair = self.pairs[self.index]
        self.progress.value = (
            f"<b>Image {self.index + 1} / {len(self.pairs)}</b> "
            f"| {pair.image_id} | {pair.large_file_path}"
        )

    def _update_point_summary(self) -> None:
        n = len(self.current_points_xy)
        z_total = 0 if self.current is None else int(self.current.bf_stack.shape[0])
        z_idx = int(self.z_slider.value)
        if self.current_annotation_status == "no_bead":
            status_txt = "no_bead"
        elif n > 0:
            status_txt = "annotated (unsaved/local)"
        else:
            status_txt = "unannotated"
        self.point_summary.value = (
            f"<b>Centroids:</b> {n} | <b>Status:</b> {status_txt} | "
            f"<b>Displayed Z:</b> {z_idx + 1}/{max(z_total, 1)} | "
            f"<b>Bead Z:</b> {self.current_bead_z_index + 1}/{max(z_total, 1)} | "
            f"<b>Cyst Focus Z:</b> {self.current_focus_z_index + 1}/{max(z_total, 1)}"
        )

    def _transform_row_for_current(self) -> Optional[pd.Series]:
        if self.transform_df.empty:
            return None
        key = self.pairs[self.index].large_file_path
        sub = self.transform_df[self.transform_df["large_file_path"] == key].copy()
        if sub.empty:
            return None
        sub = sub[sub["registration_status"] == "ok"].copy()
        if sub.empty:
            return None
        return sub.iloc[0]

    def _existing_state_for_current(self) -> Tuple[List[Tuple[float, float]], Optional[str]]:
        key = self.pairs[self.index].large_file_path
        sub = self.centroid_df[self.centroid_df["large_file_path"] == key].copy()
        if sub.empty:
            return [], None
        statuses = set(sub["annotation_status"].astype(str).str.lower())
        if "no_bead" in statuses:
            return [], "no_bead"

        sub["centroid_index"] = pd.to_numeric(sub["centroid_index"], errors="coerce")
        sub["centroid_x_px"] = pd.to_numeric(sub["centroid_x_px"], errors="coerce")
        sub["centroid_y_px"] = pd.to_numeric(sub["centroid_y_px"], errors="coerce")
        sub = sub.sort_values("centroid_index")
        out: List[Tuple[float, float]] = []
        for r in sub.itertuples(index=False):
            if not np.isfinite(r.centroid_x_px) or not np.isfinite(r.centroid_y_px):
                continue
            out.append((float(r.centroid_x_px), float(r.centroid_y_px)))
        if len(out) > 0:
            return out, "annotated"
        return [], None

    def _default_z_index(self, z_count: int) -> int:
        if z_count <= 1:
            return 0
        if np.isfinite(self.current_bead_z_index):
            return int(np.clip(int(self.current_bead_z_index), 0, z_count - 1))
        if np.isfinite(self.current_focus_z_index):
            return int(np.clip(int(self.current_focus_z_index), 0, z_count - 1))
        return int(z_count // 2)

    def _focus_z_index_from_transform(self, z_count: int) -> int:
        if z_count <= 1:
            return 0
        tr = self._transform_row_for_current()
        if tr is not None:
            z_key = "large_focus_z_index" if "large_focus_z_index" in tr.index else "small_to_large_best_z_index"
            z_raw = pd.to_numeric(pd.Series([tr[z_key]]), errors="coerce").iloc[0]
            if np.isfinite(z_raw):
                return int(np.clip(int(z_raw), 0, z_count - 1))
        return int(z_count // 2)

    def _load_current_image(self) -> None:
        pair = self.pairs[self.index]
        self.current = load_large_registered_stack(pair=pair, root=self.root)
        self._update_progress()

        self._bf_show_stack = np.stack(
            [robust_rescale(self.current.bf_stack[z]) for z in range(self.current.bf_stack.shape[0])],
            axis=0,
        )
        self._af_show_stack = np.stack(
            [robust_rescale(self.current.af647_stack[z]) for z in range(self.current.af647_stack.shape[0])],
            axis=0,
        )

        self.current_points_xy, self.current_annotation_status = self._existing_state_for_current()
        self.undo_stack = []

        tr = self._transform_row_for_current()
        self._small_fov_polygon_xy = (
            build_small_fov_polygon_in_large(tr) if tr is not None else None
        )

        z_count = int(self.current.bf_stack.shape[0])
        self.current_focus_z_index = self._focus_z_index_from_transform(z_count=z_count)
        bead_pick = choose_large_bead_z_index(self.current.af647_stack)
        self.current_bead_z_index = int(np.clip(int(bead_pick["bead_z_index"]), 0, max(0, z_count - 1)))
        default_z = self._default_z_index(z_count=z_count)
        self._suspend_z_callback = True
        self.z_slider.min = 0
        self.z_slider.max = max(0, z_count - 1)
        self.z_slider.value = int(np.clip(default_z, 0, self.z_slider.max))
        self._suspend_z_callback = False

        if self.current_annotation_status == "no_bead":
            if self.allow_add:
                self._set_status(
                    "Existing status is no_bead. Defaulting to bead-friendly z. "
                    "Use Bead Z / Cyst Focus Z to jump between suggested planes."
                )
            else:
                self._set_status("Review mode: existing status is no_bead.")
        elif len(self.current_points_xy) > 0:
            if self.allow_add:
                self._set_status(
                    "Existing centroids loaded. Defaulting to bead-friendly z. "
                    "Use Bead Z / Cyst Focus Z to jump between suggested planes."
                )
            else:
                self._set_status("Review mode: existing centroids loaded.")
        else:
            if self.allow_add:
                self._set_status(
                    "Defaulting to bead-friendly z from Alexa Fluor 647. "
                    "Use Bead Z / Cyst Focus Z to jump between suggested planes, "
                    "then click any panel to add centroid(s)."
                )
            else:
                self._set_status("Review mode: no saved entry for this image.")

        self._render_current_slice()

    def _remove_point_artists(self) -> None:
        for artist in self.point_artists:
            try:
                artist.remove()
            except Exception:
                pass
        for text in self.point_texts:
            try:
                text.remove()
            except Exception:
                pass
        for artist in self.fov_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.point_artists = []
        self.point_texts = []
        self.fov_artists = []

    def _draw_small_fov_overlay(self) -> None:
        if self._small_fov_polygon_xy is None:
            return
        poly = np.asarray(self._small_fov_polygon_xy, dtype=np.float32)
        if poly.shape != (4, 2):
            return
        path = np.vstack([poly, poly[0]])
        for ax in self.axes:
            artist = ax.plot(
                path[:, 0],
                path[:, 1],
                linestyle="--",
                linewidth=1.2,
                color="cyan",
                alpha=0.9,
            )[0]
            ax.text(
                float(poly[0, 0]) + 6.0,
                float(poly[0, 1]) + 18.0,
                "small FOV",
                color="cyan",
                fontsize=8,
                fontweight="bold",
            )
            self.fov_artists.append(artist)

    def _render_current_slice(self) -> None:
        if self.current is None or self._bf_show_stack is None or self._af_show_stack is None:
            return
        z_idx = int(self.z_slider.value)

        titles = [
            f"LARGE Brightfield (Z {z_idx + 1})",
            f"LARGE Alexa Fluor 647 (Z {z_idx + 1})",
            f"LARGE BF + 647 (Z {z_idx + 1})",
        ]
        imgs = [
            self._bf_show_stack[z_idx],
            self._af_show_stack[z_idx],
            self.current.fused_stack[z_idx],
        ]
        cmaps = ["gray", "magma", None]
        for ax, im, title, cmap in zip(self.axes, imgs, titles, cmaps):
            ax.clear()
            if cmap is None:
                ax.imshow(im)
            else:
                ax.imshow(im, cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self._draw_points()
        self._update_point_summary()
        self.fig.canvas.draw_idle()

    def _draw_points(self) -> None:
        self._remove_point_artists()
        if self.current is None:
            return

        self._draw_small_fov_overlay()
        if self.current_annotation_status == "no_bead":
            for ax in self.axes:
                t = ax.text(
                    0.02,
                    0.98,
                    "NO BEAD",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                    color="yellow",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="yellow"),
                )
                self.point_texts.append(t)

        for idx, (x, y) in enumerate(self.current_points_xy, start=1):
            for ax in self.axes:
                artist = ax.plot(
                    [x],
                    [y],
                    marker="o",
                    linestyle="None",
                    markersize=8,
                    markeredgewidth=1.5,
                    color="lime",
                    alpha=0.95,
                )[0]
                label = ax.text(
                    x + 4.0,
                    y - 4.0,
                    str(idx),
                    color="lime",
                    fontsize=8,
                    fontweight="bold",
                    alpha=0.95,
                )
                self.point_artists.append(artist)
                self.point_texts.append(label)

    def _push_undo(self) -> None:
        self.undo_stack.append((list(self.current_points_xy), self.current_annotation_status))
        if len(self.undo_stack) > 100:
            self.undo_stack = self.undo_stack[-100:]

    def _clamp_point(self, x: float, y: float) -> Tuple[float, float]:
        if self.current is None:
            return x, y
        _, h, w = self.current.bf_stack.shape
        xc = float(np.clip(x, 0, w - 1))
        yc = float(np.clip(y, 0, h - 1))
        return xc, yc

    def _advance_or_end(self, end_msg: str) -> None:
        if self.index >= len(self.pairs) - 1:
            self._set_status(end_msg)
            return
        self.index += 1
        self._load_current_image()

    def _on_click(self, event) -> None:
        if self.current is None:
            return
        if not self.allow_add:
            self._set_status("Review mode: editing is disabled.")
            return
        if event.inaxes not in list(self.axes):
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        x, y = self._clamp_point(x, y)

        self._push_undo()
        self.current_points_xy.append((x, y))
        self.current_annotation_status = "annotated"
        self._draw_points()
        self._update_point_summary()
        self._set_status(
            f"Added centroid #{len(self.current_points_xy)} at x={x:.1f}, y={y:.1f}."
        )
        self.fig.canvas.draw_idle()

    def _on_z_change(self, change) -> None:
        if self._suspend_z_callback:
            return
        if change.get("name") != "value":
            return
        self._render_current_slice()

    def _jump_to_z(self, z_idx: int, label: str) -> None:
        if self.current is None:
            self._set_status("No image loaded.")
            return
        z_count = int(self.current.bf_stack.shape[0])
        z_idx = int(np.clip(int(z_idx), 0, max(0, z_count - 1)))
        self._suspend_z_callback = True
        self.z_slider.value = z_idx
        self._suspend_z_callback = False
        self._render_current_slice()
        self._set_status(f"Jumped to {label} z = {z_idx + 1}/{max(z_count, 1)}.")

    def _on_jump_bead_z(self, _btn) -> None:
        self._jump_to_z(self.current_bead_z_index, label="bead-friendly")

    def _on_jump_focus_z(self, _btn) -> None:
        self._jump_to_z(self.current_focus_z_index, label="cyst-focus")

    def _on_prev(self, _btn) -> None:
        if self.index == 0:
            self._set_status("Already at first image.")
            return
        self.index -= 1
        self._load_current_image()

    def _on_pass(self, _btn) -> None:
        self._advance_or_end(end_msg="Pass applied. Reached last image.")

    def _on_erase_all(self, _btn) -> None:
        if not self.allow_add:
            self._set_status("Review mode: editing is disabled.")
            return
        if not self.current_points_xy:
            self._set_status("No centroids to erase.")
            return
        self._push_undo()
        self.current_points_xy = []
        if self.current_annotation_status == "annotated":
            self.current_annotation_status = None
        self._draw_points()
        self._update_point_summary()
        self._set_status("Erased all centroids for this image. Use Undo to restore.")
        self.fig.canvas.draw_idle()

    def _on_undo(self, _btn) -> None:
        if not self.undo_stack:
            self._set_status("Nothing to undo.")
            return
        points_xy, status = self.undo_stack.pop()
        self.current_points_xy = list(points_xy)
        self.current_annotation_status = status
        self._draw_points()
        self._update_point_summary()
        self._set_status("Undo applied.")
        self.fig.canvas.draw_idle()

    def _save_current_rows(self, status: str) -> None:
        if self.current is None:
            raise RuntimeError("No image loaded.")
        if status == "annotated":
            if len(self.current_points_xy) == 0:
                raise ValueError("No centroids set. Add centroids or mark No Bead.")
            replace_large_centroid_rows(
                centroid_path=self.centroid_path,
                pair=self.current.pair,
                points_xy=self.current_points_xy,
                annotation_status="annotated",
            )
        elif status == "no_bead":
            replace_large_centroid_rows(
                centroid_path=self.centroid_path,
                pair=self.current.pair,
                points_xy=[],
                annotation_status="no_bead",
            )
        else:
            raise ValueError(f"Unexpected save status: {status}")

        self.centroid_df = load_large_centroid_table(self.centroid_path)
        self.undo_stack = []

    def _on_save_next(self, _btn) -> None:
        if not self.allow_add:
            self._set_status("Review mode: use Pass/Previous to navigate.")
            return
        if self.current is None:
            self._set_status("No image loaded.")
            return

        status = "annotated" if len(self.current_points_xy) > 0 else self.current_annotation_status
        if status != "annotated":
            self._set_status("No centroids set. Add centroid(s) or choose No Bead + Next.")
            return

        try:
            self._save_current_rows(status="annotated")
        except Exception as e:
            self._set_status(f"Failed to save centroids: {e}")
            return

        self._advance_or_end(
            end_msg=f"Saved {len(self.current_points_xy)} centroid(s). Reached last image."
        )

    def _on_no_bead_next(self, _btn) -> None:
        if not self.allow_add:
            self._set_status("Review mode: use Pass/Previous to navigate.")
            return
        if self.current is None:
            self._set_status("No image loaded.")
            return
        self._push_undo()
        self.current_points_xy = []
        self.current_annotation_status = "no_bead"
        self._draw_points()
        self._update_point_summary()
        self.fig.canvas.draw_idle()
        try:
            self._save_current_rows(status="no_bead")
        except Exception as e:
            self._set_status(f"Failed to save no_bead status: {e}")
            return
        self._advance_or_end(end_msg="Saved no_bead. Reached last image.")

    def _on_clear_no_bead(self, _btn) -> None:
        if not self.allow_add:
            self._set_status("Review mode: editing is disabled.")
            return
        if self.current_annotation_status != "no_bead":
            self._set_status("Current local status is not no_bead.")
            return
        self._push_undo()
        self.current_annotation_status = None
        self._draw_points()
        self._update_point_summary()
        self._set_status("Cleared local no_bead flag. Add centroid(s) and save if needed.")
        self.fig.canvas.draw_idle()

    def render(self) -> None:
        z_controls = widgets.HBox([self.z_slider, self.btn_bead_z, self.btn_focus_z])
        if self.allow_add:
            controls = widgets.HBox(
                [
                    self.btn_prev,
                    self.btn_pass,
                    self.btn_save_next,
                    self.btn_no_bead,
                    self.btn_clear_no_bead,
                    self.btn_erase_all,
                    self.btn_undo,
                ]
            )
        else:
            controls = widgets.HBox([self.btn_prev, self.btn_pass])
        display(widgets.VBox([self.progress, z_controls, controls, self.point_summary, self.status]))
        display(self.fig.canvas)


class ManualWellRoiSession:
    """
    Interactive session for manual square ROI annotation.

    Controls:
      - Draw square ROI on any panel (BF, 647, BF+647).
      - Save + Next: persist latest ROI, then advance.
      - Pass: keep existing ROI unchanged and advance.
      - Previous: move back.
      - Clear ROI: remove current unsaved ROI preview.
    """

    def __init__(self, records: Sequence[ImageRecord], root: Path, roi_path: Path):
        self.records = list(records)
        if len(self.records) == 0:
            raise ValueError("No image records provided for annotation.")
        self.root = Path(root)
        self.roi_path = Path(roi_path)
        self.roi_df = load_roi_table(self.roi_path)

        self.index = 0
        self.current: Optional[LoadedImage] = None
        self.current_roi_xyxy: Optional[Tuple[float, float, float, float]] = None

        # Build figure with interactive display temporarily disabled so it is shown
        # exactly once via `render()` (avoids duplicate canvas rows in widget mode).
        with plt.ioff():
            self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        self.selectors: List[RectangleSelector] = []
        self.roi_patches: List[Rectangle] = []
        self._init_selectors()

        self.btn_prev = widgets.Button(description="Previous")
        self.btn_pass = widgets.Button(description="Pass")
        self.btn_save_next = widgets.Button(description="Save ROI + Next", button_style="success")
        self.btn_clear = widgets.Button(description="Clear ROI")
        self.status = widgets.HTML()
        self.progress = widgets.HTML()

        self.btn_prev.on_click(self._on_prev)
        self.btn_pass.on_click(self._on_pass)
        self.btn_save_next.on_click(self._on_save_next)
        self.btn_clear.on_click(self._on_clear)

        self._load_current_image()

    def _init_selectors(self) -> None:
        for ax in self.axes:
            selector = RectangleSelector(
                ax=ax,
                onselect=self._on_select,
                useblit=False,
                button=[1],
                minspanx=4,
                minspany=4,
                spancoords="pixels",
                interactive=False,
            )
            self.selectors.append(selector)

    def _set_status(self, text: str) -> None:
        self.status.value = f"<pre style='margin:0'>{text}</pre>"

    def _update_progress(self) -> None:
        self.progress.value = (
            f"<b>Image {self.index + 1} / {len(self.records)}</b> "
            f"| {self.records[self.index].image_id}"
        )

    def _existing_roi_for_current(self) -> Optional[pd.Series]:
        key = self.records[self.index].file_path
        sub = self.roi_df[self.roi_df["file_path"] == key]
        if sub.empty:
            return None
        return sub.iloc[0]

    def _load_current_image(self) -> None:
        rec = self.records[self.index]
        self.current = load_primary_image(rec, root=self.root)
        self._update_progress()

        bf_show = robust_rescale(self.current.bf)
        af_show = robust_rescale(self.current.af647)
        fused = self.current.fused

        titles = ["Brightfield", "Alexa Fluor 647", "BF + 647"]
        imgs = [bf_show, af_show, fused]
        cmaps = ["gray", "magma", None]
        for ax, im, title, cmap in zip(self.axes, imgs, titles, cmaps):
            ax.clear()
            if cmap is None:
                ax.imshow(im)
            else:
                ax.imshow(im, cmap=cmap)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self.current_roi_xyxy = None
        existing = self._existing_roi_for_current()
        if existing is not None and np.all(np.isfinite([existing["x0"], existing["y0"], existing["x1"], existing["y1"]])):
            self.current_roi_xyxy = (
                float(existing["x0"]),
                float(existing["y0"]),
                float(existing["x1"]),
                float(existing["y1"]),
            )
            self._draw_roi_patches()
            self._set_status(
                "Existing ROI loaded. Use Pass to keep it, or redraw and Save ROI + Next."
            )
        else:
            self._remove_roi_patches()
            self._set_status("Draw a square ROI on any panel, then click Save ROI + Next.")

        self.fig.canvas.draw_idle()

    def _remove_roi_patches(self) -> None:
        for p in self.roi_patches:
            try:
                p.remove()
            except Exception:
                pass
        self.roi_patches = []

    def _draw_roi_patches(self) -> None:
        self._remove_roi_patches()
        if self.current_roi_xyxy is None:
            self.fig.canvas.draw_idle()
            return
        x0, y0, x1, y1 = self.current_roi_xyxy
        xi0, yi0 = min(x0, x1), min(y0, y1)
        side = max(abs(x1 - x0), abs(y1 - y0))
        for ax in self.axes:
            patch = Rectangle(
                (xi0, yi0),
                side,
                side,
                fill=False,
                edgecolor="yellow",
                linewidth=1.5,
            )
            ax.add_patch(patch)
            self.roi_patches.append(patch)
        self.fig.canvas.draw_idle()

    def _square_from_drag(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> Optional[Tuple[float, float, float, float]]:
        if self.current is None:
            return None
        h, w = self.current.bf.shape
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None

        sx = 1.0 if dx >= 0 else -1.0
        sy = 1.0 if dy >= 0 else -1.0
        if abs(dx) < 1e-6:
            sx = 1.0
        if abs(dy) < 1e-6:
            sy = 1.0

        target_side = max(abs(dx), abs(dy))
        max_side_x = (w - 1 - x0) if sx > 0 else x0
        max_side_y = (h - 1 - y0) if sy > 0 else y0
        side = max(2.0, min(target_side, max_side_x, max_side_y))

        x1_sq = x0 + sx * side
        y1_sq = y0 + sy * side
        x0_c = float(np.clip(x0, 0, w - 1))
        y0_c = float(np.clip(y0, 0, h - 1))
        x1_c = float(np.clip(x1_sq, 0, w - 1))
        y1_c = float(np.clip(y1_sq, 0, h - 1))
        return (x0_c, y0_c, x1_c, y1_c)

    def _on_select(self, eclick, erelease) -> None:
        if (
            eclick.xdata is None
            or eclick.ydata is None
            or erelease.xdata is None
            or erelease.ydata is None
        ):
            return
        square = self._square_from_drag(
            float(eclick.xdata),
            float(eclick.ydata),
            float(erelease.xdata),
            float(erelease.ydata),
        )
        if square is None:
            return
        self.current_roi_xyxy = square
        self._draw_roi_patches()
        self._set_status("ROI drawn. Click Save ROI + Next to persist, or redraw.")

    def _on_prev(self, _btn) -> None:
        if self.index == 0:
            self._set_status("Already at first image.")
            return
        self.index -= 1
        self._load_current_image()

    def _on_pass(self, _btn) -> None:
        if self.index >= len(self.records) - 1:
            self._set_status("Pass applied. Reached last image.")
            return
        self.index += 1
        self._load_current_image()

    def _on_clear(self, _btn) -> None:
        self.current_roi_xyxy = None
        self._remove_roi_patches()
        self._set_status("ROI cleared. Draw a new square ROI.")

    def _on_save_next(self, _btn) -> None:
        if self.current is None or self.current_roi_xyxy is None:
            self._set_status("No ROI to save. Draw ROI first.")
            return
        x0, y0, x1, y1 = self.current_roi_xyxy
        side = float(max(abs(x1 - x0), abs(y1 - y0)))
        rec = self.current.record
        row = {
            "image_id": rec.image_id,
            "cohort_id": rec.cohort_id,
            "canonical_position": rec.canonical_position,
            "file_path": rec.file_path,
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "side_px": side,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        upsert_roi_row(self.roi_path, row)
        self.roi_df = load_roi_table(self.roi_path)

        if self.index >= len(self.records) - 1:
            self._set_status("Saved ROI. Reached last image.")
            return
        self.index += 1
        self._load_current_image()

    def render(self) -> None:
        control_row = widgets.HBox(
            [self.btn_prev, self.btn_pass, self.btn_save_next, self.btn_clear]
        )
        display(widgets.VBox([self.progress, control_row, self.status]))
        display(self.fig.canvas)


def run_roi_analysis_and_render(
    records: Sequence[ImageRecord],
    root: Path,
    roi_path: Path,
    center_path: Path,
    overlay_dir: Path,
    show_debug_inline: bool = True,
) -> pd.DataFrame:
    roi_df = load_roi_table(roi_path)
    out_rows: List[Dict[str, object]] = []
    for rec in records:
        sub = roi_df[roi_df["file_path"] == rec.file_path]
        if sub.empty:
            continue
        roi_row = sub.iloc[0]
        loaded = load_primary_image(rec, root=root)
        analysis = analyze_well_and_bead_in_roi(
            bf=loaded.bf,
            af647=loaded.af647,
            roi_xyxy=(roi_row["x0"], roi_row["y0"], roi_row["x1"], roi_row["y1"]),
        )
        row = build_center_row(rec, roi_row, analysis)
        upsert_center_row(center_path, row)
        out_rows.append(row)

        overlay_path = overlay_dir / f"{rec.image_id}_well_qc.png"
        fig = render_qc_panels(
            loaded=loaded,
            analysis=analysis,
            roi_row=roi_row,
            save_path=overlay_path,
        )
        if show_debug_inline:
            # Always show an inline artifact even when the active backend is widget-based.
            if overlay_path.exists():
                display(IPyImage(filename=str(overlay_path)))
            else:
                display(fig)
        plt.close(fig)

    if not out_rows:
        return pd.DataFrame(columns=CENTER_COLUMNS)
    return pd.DataFrame(out_rows)[CENTER_COLUMNS].copy()
