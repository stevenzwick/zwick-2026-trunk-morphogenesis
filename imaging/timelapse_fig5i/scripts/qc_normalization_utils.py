"""Utilities for QC and normalization of 2D timelapse TIFF datasets.

Designed for Micro-Manager style exports with filenames like:
img_channel000_position009_time000000123_z000.tif
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - fallback path
    cv2 = None


TIFF_RE = re.compile(
    r"img_channel(?P<channel>\d+)_position(?P<position>\d+)_time(?P<time>\d+)_z(?P<z>\d+)\.tif$"
)
POS_RE = re.compile(r"Pos(?P<row>\d+)_(?P<col>\d+)$")


def list_position_dirs(data_root: Path | str) -> List[Path]:
    """Return sorted position directories under the acquisition folder."""
    root = Path(data_root)
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("1-Pos")]
    return sorted(dirs, key=lambda p: p.name)


def parse_position_label(label: str) -> Tuple[int, int]:
    """Parse row/col indices from labels like 1-Pos009_012."""
    m = POS_RE.search(label)
    if m is None:
        raise ValueError(f"Could not parse row/col from position label: {label}")
    return int(m.group("row")), int(m.group("col"))


def parse_tiff_name(path: Path | str) -> Dict[str, int]:
    """Parse channel/position/time/z from a TIFF filename."""
    p = Path(path)
    m = TIFF_RE.match(p.name)
    if m is None:
        raise ValueError(f"Filename does not match expected pattern: {p.name}")
    return {k: int(v) for k, v in m.groupdict().items()}


def load_experiment_metadata(position_dir: Path | str) -> dict:
    """Load Micro-Manager metadata.txt from one position folder."""
    position_dir = Path(position_dir)
    metadata_path = position_dir / "metadata.txt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.txt not found at: {metadata_path}")
    return json.loads(metadata_path.read_text())


def stage_positions_table(metadata_obj: Mapping) -> pd.DataFrame:
    """Extract stage position table from metadata summary."""
    stage_positions = metadata_obj["Summary"]["StagePositions"]
    rows: List[Dict[str, object]] = []
    for site in stage_positions:
        label = site["Label"]
        row_idx, col_idx = parse_position_label(label)
        x_um = np.nan
        y_um = np.nan
        z_um = np.nan
        for dev in site["DevicePositions"]:
            if dev["Device"] == "XYStage":
                x_um, y_um = dev["Position_um"]
            elif dev["Device"] == "ZeissFocusAxis":
                z_um = dev["Position_um"][0]
        rows.append(
            {
                "position_label": label,
                "row": row_idx,
                "col": col_idx,
                "x_um": float(x_um),
                "y_um": float(y_um),
                "z_um": float(z_um),
                "grid_row_field": site.get("GridRow", np.nan),
                "grid_col_field": site.get("GridCol", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(["col", "row"]).reset_index(drop=True)


def build_file_index(
    data_root: Path | str,
    include_channels: Optional[Sequence[int]] = None,
    time_step: int = 1,
    max_time: Optional[int] = None,
) -> pd.DataFrame:
    """Build a tidy index of TIFF files across all position folders."""
    records: List[Dict[str, object]] = []
    include_set = set(include_channels) if include_channels is not None else None
    for pos_dir in list_position_dirs(data_root):
        pos_label = pos_dir.name
        row_idx, col_idx = parse_position_label(pos_label)
        for tif_path in sorted(pos_dir.glob("img_channel*_position*_time*_z*.tif")):
            parsed = parse_tiff_name(tif_path)
            ch = parsed["channel"]
            t = parsed["time"]
            if include_set is not None and ch not in include_set:
                continue
            if time_step > 1 and t % time_step != 0:
                continue
            if max_time is not None and t > max_time:
                continue
            records.append(
                {
                    "position_label": pos_label,
                    "row": row_idx,
                    "col": col_idx,
                    "file_position_index": parsed["position"],
                    "channel": ch,
                    "time": t,
                    "z": parsed["z"],
                    "path": str(tif_path),
                }
            )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    return df.sort_values(["row", "col", "channel", "time"]).reset_index(drop=True)


def completeness_report(
    index_df: pd.DataFrame,
    expected_channels: int = 3,
    expected_timepoints: int = 300,
    expected_z: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Summarize completeness and duplicate/missing events."""
    if index_df.empty:
        raise ValueError("index_df is empty")

    expected_files_per_position = expected_channels * expected_timepoints * expected_z
    per_pos = (
        index_df.groupby(["position_label", "row", "col"], as_index=False)
        .size()
        .rename(columns={"size": "observed_files"})
    )
    per_pos["expected_files"] = expected_files_per_position
    per_pos["missing_files"] = per_pos["expected_files"] - per_pos["observed_files"]

    per_pos_ch = (
        index_df.groupby(["position_label", "channel"])["time"]
        .nunique()
        .rename("n_unique_times")
        .reset_index()
    )
    per_pos_ch["expected_times"] = expected_timepoints
    per_pos_ch["missing_times"] = per_pos_ch["expected_times"] - per_pos_ch["n_unique_times"]

    duplicates = (
        index_df.groupby(["position_label", "channel", "time", "z"])
        .size()
        .rename("n_files")
        .reset_index()
    )
    duplicates = duplicates[duplicates["n_files"] > 1].sort_values(
        ["position_label", "channel", "time", "z"]
    )

    return {
        "per_position": per_pos.sort_values(["row", "col"]).reset_index(drop=True),
        "per_position_channel": per_pos_ch.sort_values(
            ["position_label", "channel"]
        ).reset_index(drop=True),
        "duplicates": duplicates.reset_index(drop=True),
    }


def read_tiff(path: Path | str) -> np.ndarray:
    """Read a TIFF image into a NumPy array."""
    return tifffile.imread(str(path))


def _laplacian_variance(image: np.ndarray) -> float:
    """Focus proxy: variance of Laplacian."""
    if cv2 is not None:
        lap = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)
        return float(lap.var())
    img = image.astype(np.float32)
    gx, gy = np.gradient(img)
    return float((gx**2 + gy**2).mean())


def frame_metrics(image: np.ndarray, compute_focus: bool = False) -> Dict[str, float]:
    """Compute per-frame QC metrics."""
    img = image.astype(np.float32, copy=False)
    i_min = float(np.min(img))
    i_max = float(np.max(img))
    p01 = float(np.percentile(img, 1))
    p99 = float(np.percentile(img, 99))
    metrics = {
        "min": i_min,
        "max": i_max,
        "mean": float(np.mean(img)),
        "median": float(np.median(img)),
        "std": float(np.std(img)),
        "p01": p01,
        "p99": p99,
        "sat_zero_frac": float(np.mean(img <= 0)),
        "sat_max_frac": float(np.mean(img >= 65535)),
    }
    metrics["lap_var"] = _laplacian_variance(img) if compute_focus else np.nan
    return metrics


def _sample_group_rows(group: pd.DataFrame, max_rows: Optional[int]) -> pd.DataFrame:
    if max_rows is None or len(group) <= max_rows:
        return group
    take_idx = np.linspace(0, len(group) - 1, max_rows, dtype=int)
    return group.iloc[take_idx]


def compute_frame_metrics(
    index_df: pd.DataFrame,
    channels: Sequence[int] = (0, 1, 2),
    sample_every_n: int = 10,
    max_frames_per_group: Optional[int] = None,
    focus_channels: Sequence[int] = (0,),
    progress_every: int = 500,
) -> pd.DataFrame:
    """Compute sampled per-frame QC metrics from indexed TIFF files."""
    if index_df.empty:
        raise ValueError("index_df is empty")

    channels_set = set(channels)
    focus_set = set(focus_channels)
    df = index_df[index_df["channel"].isin(channels_set)].copy()
    if sample_every_n > 1:
        df = df[df["time"] % sample_every_n == 0]
    df = (
        df.groupby(["position_label", "channel"], group_keys=False)
        .apply(lambda g: _sample_group_rows(g.sort_values("time"), max_frames_per_group))
        .reset_index(drop=True)
    )

    out: List[Dict[str, object]] = []
    n_total = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        img = read_tiff(row.path)
        m = frame_metrics(img, compute_focus=row.channel in focus_set)
        m.update(
            {
                "position_label": row.position_label,
                "row": row.row,
                "col": row.col,
                "file_position_index": row.file_position_index,
                "channel": row.channel,
                "time": row.time,
                "z": row.z,
                "path": row.path,
            }
        )
        out.append(m)
        if progress_every > 0 and (i % progress_every == 0 or i == n_total):
            print(f"Computed frame metrics for {i}/{n_total} sampled frames")

    return pd.DataFrame(out).sort_values(["row", "col", "channel", "time"]).reset_index(
        drop=True
    )


def estimate_flatfield(
    index_df: pd.DataFrame,
    channel: int,
    sample_n: int = 200,
    random_state: int = 0,
    background_percentile: float = 1.0,
) -> np.ndarray:
    """Estimate illumination field by median-combining sampled frames."""
    df_ch = index_df[index_df["channel"] == channel].copy()
    if df_ch.empty:
        raise ValueError(f"No rows found for channel {channel}")
    n = min(sample_n, len(df_ch))
    sampled = df_ch.sample(n=n, random_state=random_state)

    stack = []
    for row in sampled.itertuples(index=False):
        img = read_tiff(row.path).astype(np.float32)
        bg = np.percentile(img, background_percentile)
        img = np.clip(img - bg, 0.0, None)
        stack.append(img)
    med = np.median(np.stack(stack, axis=0), axis=0)
    med = np.clip(med, np.percentile(med, 1), None)
    med = med / np.median(med)
    return med.astype(np.float32)


def compute_position_timeseries(
    index_df: pd.DataFrame,
    channels: Sequence[int] = (1, 2),
    sample_every_n: int = 1,
    max_frames_per_group: Optional[int] = None,
    subtract_background: bool = True,
    background_percentile: float = 1.0,
    flatfields: Optional[Mapping[int, np.ndarray]] = None,
    subtract_postflat_background: bool = False,
    postflat_background_percentile: float = 20.0,
    progress_every: int = 1000,
) -> pd.DataFrame:
    """Compute per-frame scalar features with explicit preprocessing steps.

    Per frame, the pipeline is:
    1) Read raw image
    2) Optionally subtract an additive background estimate from a percentile
    3) Optionally apply multiplicative flat-field correction
    4) Optionally subtract a second, post-flat-field additive background estimate
    5) Extract scalar intensity features

    Notes:
    - `subtract_background=True` with `background_percentile=1.0` means
      background is estimated as p1(raw) per frame.
    - Flat-field correction is applied after background subtraction.
    - `subtract_postflat_background=True` means background is re-estimated on the
      flat-field corrected frame and subtracted per frame.
    """
    if index_df.empty:
        raise ValueError("index_df is empty")
    if background_percentile < 0 or background_percentile > 100:
        raise ValueError("background_percentile must be in [0, 100]")
    if postflat_background_percentile < 0 or postflat_background_percentile > 100:
        raise ValueError("postflat_background_percentile must be in [0, 100]")

    channels_set = set(channels)
    df = index_df[index_df["channel"].isin(channels_set)].copy()
    if sample_every_n > 1:
        df = df[df["time"] % sample_every_n == 0]
    df = (
        df.groupby(["position_label", "channel"], group_keys=False)
        .apply(lambda g: _sample_group_rows(g.sort_values("time"), max_frames_per_group))
        .reset_index(drop=True)
    )

    out: List[Dict[str, object]] = []
    n_total = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        raw = read_tiff(row.path).astype(np.float32)
        if subtract_background:
            bg = float(np.percentile(raw, background_percentile))
            bgsub = np.clip(raw - bg, 0.0, None)
        else:
            bg = 0.0
            bgsub = raw

        ff_corrected = bgsub
        if flatfields is not None and row.channel in flatfields:
            ff = flatfields[row.channel].astype(np.float32)
            ff_corrected = ff_corrected / np.clip(ff, 1e-6, None)

        if subtract_postflat_background:
            ff_bg = float(np.percentile(ff_corrected, postflat_background_percentile))
            ff_bgsub = np.clip(ff_corrected - ff_bg, 0.0, None)
        else:
            ff_bg = 0.0
            ff_bgsub = ff_corrected

        out.append(
            {
                "position_label": row.position_label,
                "row": row.row,
                "col": row.col,
                "file_position_index": row.file_position_index,
                "channel": row.channel,
                "time": row.time,
                "z": row.z,
                "path": row.path,
                "bg_value": bg,
                "bg_percentile": float(background_percentile) if subtract_background else np.nan,
                "background_subtracted": bool(subtract_background),
                "ff_bg_value": ff_bg,
                "ff_bg_percentile": (
                    float(postflat_background_percentile)
                    if subtract_postflat_background
                    else np.nan
                ),
                "postflat_background_subtracted": bool(subtract_postflat_background),
                "raw_mean": float(raw.mean()),
                "raw_p10": float(np.percentile(raw, 10)),
                "raw_p50": float(np.percentile(raw, 50)),
                "raw_p90": float(np.percentile(raw, 90)),
                "raw_p99": float(np.percentile(raw, 99)),
                "ff_mean": float(ff_corrected.mean()),
                "ff_p10": float(np.percentile(ff_corrected, 10)),
                "ff_p50": float(np.percentile(ff_corrected, 50)),
                "ff_p90": float(np.percentile(ff_corrected, 90)),
                "ff_p99": float(np.percentile(ff_corrected, 99)),
                "ff_bgsub_mean": float(ff_bgsub.mean()),
                "ff_bgsub_p10": float(np.percentile(ff_bgsub, 10)),
                "ff_bgsub_p50": float(np.percentile(ff_bgsub, 50)),
                "ff_bgsub_p90": float(np.percentile(ff_bgsub, 90)),
                "ff_bgsub_p99": float(np.percentile(ff_bgsub, 99)),
                "ff_contrast_p99_p50": float(np.percentile(ff_corrected, 99))
                - float(np.percentile(ff_corrected, 50)),
                "ff_bgsub_contrast_p99_p50": float(np.percentile(ff_bgsub, 99))
                - float(np.percentile(ff_bgsub, 50)),
                "sat_zero_frac": float(np.mean(raw <= 0)),
                "sat_max_frac": float(np.mean(raw >= 65535)),
            }
        )
        if progress_every > 0 and (i % progress_every == 0 or i == n_total):
            print(f"Computed timeseries metrics for {i}/{n_total} sampled frames")

    return pd.DataFrame(out).sort_values(["row", "col", "channel", "time"]).reset_index(
        drop=True
    )


def normalize_timeseries(
    ts_df: pd.DataFrame,
    value_col: str = "ff_p90",
    group_cols: Sequence[str] = ("position_label", "channel"),
    baseline_frames: int = 30,
    smooth_window: int = 21,
) -> pd.DataFrame:
    """Add baseline, dF/F0, detrended, and robust-z columns to time series."""
    if ts_df.empty:
        raise ValueError("ts_df is empty")
    if value_col not in ts_df.columns:
        # Compatibility for older tables that used corr_* naming.
        legacy_col = (
            f"corr_{value_col[3:]}" if value_col.startswith("ff_") else None
        )
        if legacy_col and legacy_col in ts_df.columns:
            value_col = legacy_col
        else:
            raise ValueError(f"{value_col} not found in ts_df")

    out = ts_df.copy().sort_values([*group_cols, "time"]).reset_index(drop=True)
    eps = 1e-6

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        x = g[value_col].astype(float)
        n_base = min(baseline_frames, len(g))
        baseline = float(np.median(x.iloc[:n_base]))
        trend = (
            x.rolling(window=smooth_window, center=True, min_periods=max(3, smooth_window // 4))
            .median()
            .interpolate(limit_direction="both")
        )
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        robust_sigma = 1.4826 * mad + eps

        g = g.copy()
        g["baseline"] = baseline
        g["dff"] = (x - baseline) / (baseline + eps)
        g["trend"] = trend
        g["detrended_ratio"] = x / (trend + eps)
        g["robust_z"] = (x - median) / robust_sigma
        return g

    return (
        out.groupby(list(group_cols), group_keys=False)
        .apply(_per_group)
        .reset_index(drop=True)
    )


def infer_channel_names(metadata_obj: Mapping) -> Dict[int, str]:
    """Extract channel names from metadata summary."""
    ch_names = metadata_obj["Summary"].get("ChNames", [])
    return {i: str(name) for i, name in enumerate(ch_names)}
