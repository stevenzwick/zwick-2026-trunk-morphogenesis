from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


TIME_DISPLAY_OFFSET_HOURS = 48.0
DEFAULT_DISPLAY_TICKS = np.array([48.0, 60.0, 72.0, 84.0, 96.0, 108.0], dtype=float)
CROWDED_DISPLAY_TICKS = np.array([48.0, 72.0, 96.0], dtype=float)


def display_time_hours(values, offset_hours: float = TIME_DISPLAY_OFFSET_HOURS):
    if np.isscalar(values):
        return float(values) + offset_hours
    return np.asarray(values, dtype=float) + offset_hours


def display_time_hours_from_index(
    time_index,
    interval_hours: float,
    offset_hours: float = TIME_DISPLAY_OFFSET_HOURS,
):
    if np.isscalar(time_index):
        return float(time_index) * float(interval_hours) + offset_hours
    return np.asarray(time_index, dtype=float) * float(interval_hours) + offset_hours


def format_display_hours(
    value: float,
    decimals: int = 1,
    offset_hours: float = TIME_DISPLAY_OFFSET_HOURS,
) -> str:
    return f"{display_time_hours(float(value), offset_hours=offset_hours):.{decimals}f} h"


def format_display_hours_from_index(
    time_index: int | float,
    interval_hours: float,
    decimals: int = 1,
    offset_hours: float = TIME_DISPLAY_OFFSET_HOURS,
) -> str:
    return f"{display_time_hours_from_index(time_index, interval_hours, offset_hours=offset_hours):.{decimals}f} h"


def format_display_hour_range(
    start: float,
    end: float,
    decimals: int = 0,
    offset_hours: float = TIME_DISPLAY_OFFSET_HOURS,
) -> str:
    return (
        f"{display_time_hours(float(start), offset_hours=offset_hours):.{decimals}f}"
        f"-{display_time_hours(float(end), offset_hours=offset_hours):.{decimals}f} h"
    )


def _display_ticks(crowded: bool) -> np.ndarray:
    return CROWDED_DISPLAY_TICKS if crowded else DEFAULT_DISPLAY_TICKS


def visible_raw_time_ticks(
    limits,
    crowded: bool = False,
    offset_hours: float = TIME_DISPLAY_OFFSET_HOURS,
) -> np.ndarray:
    lo, hi = sorted(display_time_hours(np.asarray(limits, dtype=float), offset_hours=offset_hours))
    visible = _display_ticks(crowded)[(_display_ticks(crowded) >= lo - 1e-9) & (_display_ticks(crowded) <= hi + 1e-9)]
    return visible - offset_hours


def set_display_time_axis(
    ax,
    axis: str = "x",
    crowded: bool = False,
    offset_hours: float = TIME_DISPLAY_OFFSET_HOURS,
) -> None:
    formatter = FuncFormatter(lambda value, _: f"{display_time_hours(value, offset_hours=offset_hours):g}")
    if axis in {"x", "both"}:
        ax.xaxis.set_major_formatter(formatter)
        ticks = visible_raw_time_ticks(ax.get_xlim(), crowded=crowded, offset_hours=offset_hours)
        if len(ticks):
            ax.set_xticks(ticks)
    if axis in {"y", "both"}:
        ax.yaxis.set_major_formatter(formatter)
        ticks = visible_raw_time_ticks(ax.get_ylim(), crowded=crowded, offset_hours=offset_hours)
        if len(ticks):
            ax.set_yticks(ticks)


def set_display_time_colorbar(
    cbar,
    crowded: bool = True,
    offset_hours: float = TIME_DISPLAY_OFFSET_HOURS,
) -> None:
    cbar.formatter = FuncFormatter(lambda value, _: f"{display_time_hours(value, offset_hours=offset_hours):g}")
    lo, hi = sorted(display_time_hours(np.asarray(cbar.ax.get_ylim(), dtype=float), offset_hours=offset_hours))
    display_ticks = _display_ticks(crowded)
    visible = display_ticks[(display_ticks >= lo - 1e-9) & (display_ticks <= hi + 1e-9)]
    if len(visible):
        cbar.set_ticks(visible - offset_hours)
    cbar.update_ticks()


def offset_time_hours_df(
    df,
    should_offset_column=None,
    offset_hours: float = TIME_DISPLAY_OFFSET_HOURS,
):
    output = df.copy()
    predicate = should_offset_column or (lambda column_name: str(column_name).endswith("time_hours"))
    for column in output.columns:
        if predicate(column):
            output[column] = pd.to_numeric(output[column], errors="coerce") + offset_hours
    return output


def display_time_df(
    df,
    should_offset_column=None,
    offset_hours: float = TIME_DISPLAY_OFFSET_HOURS,
):
    return offset_time_hours_df(
        df,
        should_offset_column=should_offset_column,
        offset_hours=offset_hours,
    )
