from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


REPORTER_COLORS = {"RFP": "#d62728", "YFP": "#d8a106"}
FIT_COLORS = {"mean": "#111111", "linear": "#4c78a8", "logistic": "#d62728"}
METRIC_LABELS = {
    "organoid_mean_intensity_z": "Whole-cyst mean intensity (sigma above pooled null)",
    "brightest_decile_mean_intensity_z": "Brightest 10% mean intensity (sigma above pooled null)",
    "positive_fraction_sigma2": "Positive fraction (2 sigma mask)",
    "positive_fraction_sigma3": "Positive fraction (3 sigma mask)",
    "positive_fraction_sigma4": "Positive fraction (4 sigma mask)",
    "positive_fraction_sigma5": "Positive fraction (5 sigma mask)",
    "positive_mean_intensity_sigma2_z": "Positive-region mean intensity (2 sigma mask, sigma above pooled null)",
    "positive_mean_intensity_sigma3_z": "Positive-region mean intensity (3 sigma mask, sigma above pooled null)",
    "positive_mean_intensity_sigma4_z": "Positive-region mean intensity (4 sigma mask, sigma above pooled null)",
    "positive_mean_intensity_sigma5_z": "Positive-region mean intensity (5 sigma mask, sigma above pooled null)",
}

COMMON_HALFMAX_METRICS = [
    "organoid_mean_intensity_z",
    "positive_fraction_sigma2",
    "positive_fraction_sigma3",
    "positive_fraction_sigma4",
    "positive_fraction_sigma5",
    "positive_mean_intensity_sigma2_z",
    "positive_mean_intensity_sigma3_z",
    "positive_mean_intensity_sigma4_z",
    "positive_mean_intensity_sigma5_z",
]


def find_project_root(start: Path | None = None) -> Path:
    cwd = (start or Path.cwd()).resolve()
    root_candidates = [cwd] + list(cwd.parents[:3])
    for candidate in root_candidates:
        if (candidate / "data").exists() and (candidate / "results").exists():
            return candidate
    raise RuntimeError("Could not locate project root from current working directory.")


def load_basic_inputs(root: Path | None = None) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    root = find_project_root(root)
    population_metrics_path = root / "results/tables/05_population_trace_metrics_by_frame.tsv"
    global_thresholds_path = root / "results/tables/05_global_reporter_thresholds.tsv"

    population_metrics = pd.read_csv(population_metrics_path, sep="\t", low_memory=False)
    global_thresholds = pd.read_csv(global_thresholds_path, sep="\t")

    numeric_cols = [col for col in population_metrics.columns if col not in {"position_label", "reporter"}]
    for col in numeric_cols:
        population_metrics[col] = pd.to_numeric(population_metrics[col], errors="coerce")

    return root, population_metrics, global_thresholds


def gaussian_smooth(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    sigma = float(sigma_frames)
    if not np.isfinite(sigma) or sigma <= 0:
        return arr.copy()

    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)

    finite_mask = np.isfinite(arr)
    if finite_mask.sum() == 0:
        return np.full_like(arr, np.nan, dtype=float)

    valid_index = np.flatnonzero(finite_mask)
    valid_values = arr[finite_mask]
    filled = np.interp(np.arange(arr.size), valid_index, valid_values)
    padded = np.pad(filled, pad_width=radius, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[~finite_mask] = np.nan
    return smoothed


def rolling_mean_smooth(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    return (
        pd.Series(arr)
        .rolling(window=max(1, int(window)), center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def local_gradient(values: np.ndarray, time_hours: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    t = np.asarray(time_hours, dtype=float)
    if arr.size < 2 or t.size < 2:
        return np.full_like(arr, np.nan, dtype=float)
    if not np.isfinite(arr).any() or not np.isfinite(t).all():
        return np.full_like(arr, np.nan, dtype=float)
    return np.gradient(arr, t)


def focus_ylim_from_arrays(
    arrays: list[np.ndarray],
    include_zero: bool = False,
    padding_fraction: float = 0.10,
) -> tuple[float, float]:
    values = []
    for array in arrays:
        arr = np.asarray(array, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.append(finite.reshape(-1))
    if not values:
        return (0.0, 1.0)

    pooled = np.concatenate(values)
    lower = float(np.min(pooled))
    upper = float(np.max(pooled))
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        span = max(abs(upper), 1.0)
    pad = max(1e-6, padding_fraction * span)
    return (lower - pad, upper + pad)


def normalize_trace(values: np.ndarray, early_n: int = 8, smooth_sigma: float = 4.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    smoothed = gaussian_smooth(arr, smooth_sigma)
    finite = smoothed[np.isfinite(smoothed)]
    if finite.size < max(early_n, 4):
        return np.full_like(smoothed, np.nan, dtype=float)
    early = float(np.nanmedian(smoothed[:early_n]))
    peak = float(np.nanmax(smoothed))
    amplitude = peak - early
    if not np.isfinite(amplitude) or amplitude <= 0:
        return np.full_like(smoothed, np.nan, dtype=float)
    return (smoothed - early) / amplitude


def first_crossing_time(time_hours: np.ndarray, values: np.ndarray, target_value: float) -> float:
    t = np.asarray(time_hours, dtype=float)
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]
    if t.size < 2:
        return float("nan")
    above = y >= target_value
    if not np.any(above):
        return float("nan")
    idx = int(np.argmax(above))
    if idx == 0:
        return float(t[0])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    t0, t1 = float(t[idx - 1]), float(t[idx])
    if math.isclose(y1, y0):
        return float(t1)
    frac = (target_value - y0) / (y1 - y0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(t0 + frac * (t1 - t0))


def aggregate_mean_trace(population_metrics: pd.DataFrame, metric_name: str, reporter: str) -> pd.DataFrame:
    return (
        population_metrics.loc[population_metrics["reporter"] == reporter]
        .groupby("time_hours", as_index=False)
        .agg(
            mean=(metric_name, "mean"),
            median=(metric_name, "median"),
            count=(metric_name, "count"),
            std=(metric_name, "std"),
        )
        .sort_values("time_hours")
        .reset_index(drop=True)
    )


def aggregate_aligned_mean(aligned_df: pd.DataFrame, reporter: str) -> pd.DataFrame:
    return (
        aligned_df.loc[aligned_df["reporter"] == reporter]
        .groupby("relative_hours", as_index=False)["normalized_value"]
        .mean()
        .sort_values("relative_hours")
        .reset_index(drop=True)
    )


def halfmax_time_for_position(
    population_metrics: pd.DataFrame,
    metric_name: str,
    position_label: str,
    reporter: str,
    smooth_sigma: float = 4.0,
) -> float:
    subset = population_metrics.loc[
        (population_metrics["position_label"] == position_label)
        & (population_metrics["reporter"] == reporter)
    ].sort_values("time_hours")
    values = subset[metric_name].to_numpy(dtype=float)
    time_hours = subset["time_hours"].to_numpy(dtype=float)
    if values.size < 12:
        return float("nan")
    normalized = normalize_trace(values, early_n=8, smooth_sigma=smooth_sigma)
    return first_crossing_time(time_hours, normalized, 0.5)


def halfmax_table(
    population_metrics: pd.DataFrame,
    metric_name: str,
    smooth_sigma: float = 4.0,
) -> pd.DataFrame:
    rows = []
    for (position_label, reporter), _ in population_metrics.groupby(["position_label", "reporter"], sort=True):
        rows.append(
            {
                "position_label": position_label,
                "reporter": reporter,
                "metric_name": metric_name,
                "halfmax_time_hours": halfmax_time_for_position(
                    population_metrics,
                    metric_name,
                    position_label,
                    reporter,
                    smooth_sigma=smooth_sigma,
                ),
            }
        )
    return pd.DataFrame(rows)


def load_or_build_halfmax_cache(
    population_metrics: pd.DataFrame,
    cache_path: Path,
    metric_names: list[str] | None = None,
    smooth_sigma: float = 4.0,
) -> pd.DataFrame:
    metric_names = list(metric_names or COMMON_HALFMAX_METRICS)
    if cache_path.exists():
        cached = pd.read_csv(cache_path, sep="\t")
        if {"position_label", "reporter", "metric_name", "halfmax_time_hours"}.issubset(cached.columns):
            available = set(cached["metric_name"].dropna().unique())
            if set(metric_names).issubset(available):
                return cached.loc[cached["metric_name"].isin(metric_names)].copy()

    frames = [halfmax_table(population_metrics, metric_name, smooth_sigma=smooth_sigma) for metric_name in metric_names]
    halfmax_df = pd.concat(frames, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    halfmax_df.to_csv(cache_path, sep="\t", index=False)
    return halfmax_df


def build_halfmax_lookup(halfmax_df: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in halfmax_df.itertuples(index=False):
        lookup[(str(row.metric_name), str(row.position_label), str(row.reporter))] = float(row.halfmax_time_hours)
    return lookup


def halfmax_time_from_lookup(
    halfmax_lookup: dict[tuple[str, str, str], float],
    metric_name: str,
    position_label: str,
    reporter: str,
) -> float:
    return float(halfmax_lookup.get((metric_name, position_label, reporter), float("nan")))


def align_positions_to_anchor(
    population_metrics: pd.DataFrame,
    metric_name: str,
    anchor_metric_name: str,
    anchor_reporter: str = "YFP",
    anchor_smooth_sigma: float = 4.0,
    value_smooth_sigma: float = 4.0,
    relative_hours: np.ndarray | None = None,
    halfmax_lookup: dict[tuple[str, str, str], float] | None = None,
) -> pd.DataFrame:
    if relative_hours is None:
        relative_hours = np.arange(-24.0, 24.0 + 0.25, 0.25, dtype=float)

    rows = []
    for position_label, _ in population_metrics.groupby("position_label", sort=True):
        if halfmax_lookup is None:
            anchor_time = halfmax_time_for_position(
                population_metrics,
                anchor_metric_name,
                position_label,
                anchor_reporter,
                smooth_sigma=anchor_smooth_sigma,
            )
        else:
            anchor_time = halfmax_time_from_lookup(halfmax_lookup, anchor_metric_name, position_label, anchor_reporter)
        if not np.isfinite(anchor_time):
            continue

        for reporter in ["RFP", "YFP"]:
            subset = population_metrics.loc[
                (population_metrics["position_label"] == position_label)
                & (population_metrics["reporter"] == reporter)
            ].sort_values("time_hours")
            time_hours = subset["time_hours"].to_numpy(dtype=float)
            values = subset[metric_name].to_numpy(dtype=float)
            if values.size < 12:
                continue
            normalized = normalize_trace(values, early_n=8, smooth_sigma=value_smooth_sigma)
            finite = np.isfinite(normalized) & np.isfinite(time_hours)
            if finite.sum() < 12:
                continue
            rel_time = time_hours[finite] - anchor_time
            norm_values = normalized[finite]
            if relative_hours.min() < rel_time.min() or relative_hours.max() > rel_time.max():
                continue
            interpolated = np.interp(relative_hours, rel_time, norm_values)
            for rel_h, interp_val in zip(relative_hours, interpolated):
                rows.append(
                    {
                        "position_label": position_label,
                        "reporter": reporter,
                        "relative_hours": float(rel_h),
                        "normalized_value": float(interp_val),
                        "metric_name": metric_name,
                        "anchor_metric_name": anchor_metric_name,
                        "anchor_reporter": anchor_reporter,
                    }
                )
    return pd.DataFrame(rows)


def load_or_build_aligned_cache(
    population_metrics: pd.DataFrame,
    cache_path: Path,
    metric_name: str,
    anchor_metric_name: str,
    anchor_reporter: str = "YFP",
    anchor_smooth_sigma: float = 4.0,
    value_smooth_sigma: float = 4.0,
    relative_hours: np.ndarray | None = None,
    halfmax_lookup: dict[tuple[str, str, str], float] | None = None,
) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path, sep="\t")
        required_cols = {
            "position_label",
            "reporter",
            "relative_hours",
            "normalized_value",
            "metric_name",
            "anchor_metric_name",
            "anchor_reporter",
        }
        if required_cols.issubset(cached.columns):
            if (
                set(cached["metric_name"].dropna().unique()) == {metric_name}
                and set(cached["anchor_metric_name"].dropna().unique()) == {anchor_metric_name}
                and set(cached["anchor_reporter"].dropna().unique()) == {anchor_reporter}
            ):
                return cached.copy()

    aligned_df = align_positions_to_anchor(
        population_metrics=population_metrics,
        metric_name=metric_name,
        anchor_metric_name=anchor_metric_name,
        anchor_reporter=anchor_reporter,
        anchor_smooth_sigma=anchor_smooth_sigma,
        value_smooth_sigma=value_smooth_sigma,
        relative_hours=relative_hours,
        halfmax_lookup=halfmax_lookup,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(cache_path, sep="\t", index=False)
    return aligned_df


def align_positions_to_anchor_by_reporter_metrics(
    population_metrics: pd.DataFrame,
    metric_name_by_reporter: dict[str, str],
    anchor_metric_name: str,
    anchor_reporter: str = "YFP",
    anchor_smooth_sigma: float = 4.0,
    value_smooth_sigma: float = 4.0,
    relative_hours: np.ndarray | None = None,
    halfmax_lookup: dict[tuple[str, str, str], float] | None = None,
) -> pd.DataFrame:
    if relative_hours is None:
        relative_hours = np.arange(-24.0, 24.0 + 0.25, 0.25, dtype=float)

    rows = []
    for position_label, _ in population_metrics.groupby("position_label", sort=True):
        if halfmax_lookup is None:
            anchor_time = halfmax_time_for_position(
                population_metrics,
                anchor_metric_name,
                position_label,
                anchor_reporter,
                smooth_sigma=anchor_smooth_sigma,
            )
        else:
            anchor_time = halfmax_time_from_lookup(halfmax_lookup, anchor_metric_name, position_label, anchor_reporter)
        if not np.isfinite(anchor_time):
            continue

        for reporter in ["RFP", "YFP"]:
            metric_name = metric_name_by_reporter[reporter]
            subset = population_metrics.loc[
                (population_metrics["position_label"] == position_label)
                & (population_metrics["reporter"] == reporter)
            ].sort_values("time_hours")
            time_hours = subset["time_hours"].to_numpy(dtype=float)
            values = subset[metric_name].to_numpy(dtype=float)
            if values.size < 12:
                continue
            normalized = normalize_trace(values, early_n=8, smooth_sigma=value_smooth_sigma)
            finite = np.isfinite(normalized) & np.isfinite(time_hours)
            if finite.sum() < 12:
                continue
            rel_time = time_hours[finite] - anchor_time
            norm_values = normalized[finite]
            if relative_hours.min() < rel_time.min() or relative_hours.max() > rel_time.max():
                continue
            interpolated = np.interp(relative_hours, rel_time, norm_values)
            for rel_h, interp_val in zip(relative_hours, interpolated):
                rows.append(
                    {
                        "position_label": position_label,
                        "reporter": reporter,
                        "relative_hours": float(rel_h),
                        "normalized_value": float(interp_val),
                        "value_metric_name": metric_name,
                        "anchor_metric_name": anchor_metric_name,
                        "anchor_reporter": anchor_reporter,
                    }
                )
    return pd.DataFrame(rows)


def load_or_build_aligned_pair_cache(
    population_metrics: pd.DataFrame,
    cache_path: Path,
    metric_name_by_reporter: dict[str, str],
    anchor_metric_name: str,
    anchor_reporter: str = "YFP",
    anchor_smooth_sigma: float = 4.0,
    value_smooth_sigma: float = 4.0,
    relative_hours: np.ndarray | None = None,
    halfmax_lookup: dict[tuple[str, str, str], float] | None = None,
) -> pd.DataFrame:
    expected_metric_names = {reporter: str(metric_name_by_reporter[reporter]) for reporter in ["RFP", "YFP"]}
    if cache_path.exists():
        cached = pd.read_csv(cache_path, sep="\t")
        required_cols = {
            "position_label",
            "reporter",
            "relative_hours",
            "normalized_value",
            "value_metric_name",
            "anchor_metric_name",
            "anchor_reporter",
        }
        if required_cols.issubset(cached.columns):
            cached_metric_lookup = (
                cached[["reporter", "value_metric_name"]]
                .dropna()
                .drop_duplicates()
                .set_index("reporter")["value_metric_name"]
                .to_dict()
            )
            if (
                cached_metric_lookup == expected_metric_names
                and set(cached["anchor_metric_name"].dropna().unique()) == {anchor_metric_name}
                and set(cached["anchor_reporter"].dropna().unique()) == {anchor_reporter}
            ):
                return cached.copy()

    aligned_df = align_positions_to_anchor_by_reporter_metrics(
        population_metrics=population_metrics,
        metric_name_by_reporter=metric_name_by_reporter,
        anchor_metric_name=anchor_metric_name,
        anchor_reporter=anchor_reporter,
        anchor_smooth_sigma=anchor_smooth_sigma,
        value_smooth_sigma=value_smooth_sigma,
        relative_hours=relative_hours,
        halfmax_lookup=halfmax_lookup,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(cache_path, sep="\t", index=False)
    return aligned_df


def logistic4(x: np.ndarray, low: float, high: float, midpoint: float, slope: float) -> np.ndarray:
    return low + (high - low) / (1.0 + np.exp(-slope * (x - midpoint)))


def fit_linear_and_logistic(x: np.ndarray, y: np.ndarray) -> dict[str, float | np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 8:
        return {
            "x": x,
            "y": y,
            "linear_pred": np.full_like(x, np.nan),
            "logistic_pred": np.full_like(x, np.nan),
            "linear_rss": float("nan"),
            "logistic_rss": float("nan"),
            "logistic_midpoint": float("nan"),
            "logistic_slope": float("nan"),
            "logistic_rise_10_90_hours": float("nan"),
        }

    linear_coeffs = np.polyfit(x, y, deg=1)
    linear_pred = np.poly1d(linear_coeffs)(x)
    linear_rss = float(np.sum((y - linear_pred) ** 2))

    lower_guess = float(max(-0.1, np.nanmin(y)))
    upper_guess = float(min(1.1, np.nanmax(y)))
    midpoint_guess = float(np.nanmedian(x))
    slope_guess = 0.08
    lower_bound = [lower_guess - 0.5, 0.2, float(np.nanmin(x)) - 20.0, 1e-3]
    upper_bound = [upper_guess + 0.5, 1.5, float(np.nanmax(x)) + 20.0, 2.0]

    try:
        params, _ = curve_fit(
            logistic4,
            x,
            y,
            p0=[lower_guess, upper_guess, midpoint_guess, slope_guess],
            bounds=(lower_bound, upper_bound),
            maxfev=20000,
        )
        logistic_pred = logistic4(x, *params)
        logistic_rss = float(np.sum((y - logistic_pred) ** 2))
        slope = float(params[3])
        rise_10_90 = float(4.394 / slope) if slope > 0 else float("nan")
        midpoint = float(params[2])
    except Exception:
        logistic_pred = np.full_like(x, np.nan)
        logistic_rss = float("nan")
        midpoint = float("nan")
        slope = float("nan")
        rise_10_90 = float("nan")

    return {
        "x": x,
        "y": y,
        "linear_pred": linear_pred,
        "logistic_pred": logistic_pred,
        "linear_rss": linear_rss,
        "logistic_rss": logistic_rss,
        "logistic_midpoint": midpoint,
        "logistic_slope": slope,
        "logistic_rise_10_90_hours": rise_10_90,
    }
