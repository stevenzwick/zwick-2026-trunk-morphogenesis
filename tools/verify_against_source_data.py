"""Check a regenerated table against the paper's published Source Data.

The Source Data workbooks are the numeric substrate of the figures, so they are the natural target
for verifying a reproduction. This helper handles the three things that make a naive comparison
report false mismatches.

1. A sheet is not always one table. Four sheets carry two stacked blocks, each with its own title
   row and header. `load_blocks` parses them apart.
2. Row order differs between a sheet and the table that produced it. **Merge on keys; never compare
   positionally.**
3. Precision is not uniform. Some sheets carry full float, others are rounded to 4 or even 2
   decimals. Derive the tolerance from the block's own precision — never apply a global epsilon.

Usage:
    from verify_against_source_data import load_blocks, compare
    blocks = load_blocks("Source Data Fig.1.xlsx", "Fig.1e")
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

SOURCE_DATA_DIR = os.environ.get("SOURCE_DATA_DIR", "source_data")


def load_blocks(filename: str, sheet: str, directory: str | None = None):
    """Return [(title, DataFrame), ...] — one entry per data block in the sheet.

    A title row is text in column A with every other cell empty. The header is the next row with
    at least two non-null cells. Data runs to the next title row or the end of the sheet.
    """
    path = os.path.join(directory or SOURCE_DATA_DIR, filename)
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    titles = [
        i for i in range(len(raw))
        if isinstance(raw.iloc[i, 0], str) and pd.notna(raw.iloc[i, 0]) and raw.iloc[i, 1:].isna().all()
    ]
    blocks = []
    for n, start in enumerate(titles):
        end = titles[n + 1] if n + 1 < len(titles) else len(raw)
        header = next((j for j in range(start + 1, end) if raw.iloc[j].notna().sum() >= 2), None)
        if header is None:
            continue
        data = raw.iloc[header + 1:end].copy()
        data.columns = [str(c) for c in raw.iloc[header]]
        blocks.append((str(raw.iloc[start, 0]), data.dropna(how="all").reset_index(drop=True)))
    return blocks


#: Floor on any tolerance, as a fraction of the column's magnitude. A value that survives a
#: float64 round-trip through a spreadsheet and back is not bit-identical; agreement to ~1 part in
#: 1e12 is the practical limit, and a tolerance tighter than that reports noise as failure.
RELATIVE_FLOOR = 1e-12


def published_decimals(series: pd.Series, max_decimals: int = 12) -> int | None:
    """How many decimal places the published values actually carry.

    Inferred numerically, not from text: reading a sheet through pandas yields float64, and
    str(7.8441) can render with spurious trailing digits, which makes a text-based count wildly
    wrong. Instead, find the smallest d for which rounding to d decimals changes nothing.
    Returns None when the values are full precision (no d reproduces them).
    """
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return None
    # Not exact equality. A sheet value written as 7.8441 is frequently stored as float32 and
    # widens to 7.844099998474121 on read, so `round(v, 4) == v` is False by up to one float32
    # ulp. The slack must therefore be a float32 ulp at this magnitude — a tighter slack reads a
    # 4-decimal column as 8-decimal and drives the tolerance far below the real rounding step.
    scale = max(float(np.abs(values).max()), 1.0)
    slack = float(np.spacing(np.float32(scale)))
    for d in range(max_decimals + 1):
        if np.abs(np.round(values, d) - values).max() <= slack:
            return d
    return None


def tolerance_from_precision(series: pd.Series) -> float:
    """Half a unit in the last published decimal place, floored at float round-trip precision.

    A full-precision column carries 15+ significant digits, which would imply a tolerance near
    1e-17 — tighter than float64 can represent for values above 1. The floor keeps the tolerance
    meaningful for rounded columns while not flagging representation noise in exact ones.
    """
    decimals = published_decimals(series)
    decimal_tol = 0.5 * 10 ** -decimals if decimals is not None else 0.0
    magnitude = pd.to_numeric(series, errors="coerce").abs().max()
    floor = RELATIVE_FLOOR * float(magnitude) if pd.notna(magnitude) else 0.0
    return max(decimal_tol, floor, 1e-15)


def float32_allowance(magnitude: float, depth: int = 1) -> float:
    """Extra error budget when the stored values are float32.

    The published sheets round a value the object stores as float32. float32 cannot represent that
    value exactly, so |stored - round(published)| can exceed the decimal half-ulp by about one
    float32 ulp at that magnitude. Aggregates (a per-cluster mean over `depth` cells) accumulate
    roughly sqrt(depth) of those.

    This is a representation allowance, not a fudge factor: it is derived from the storage dtype and
    the aggregation depth, and it stays far below the precision the sheet actually publishes.
    """
    return float(np.spacing(np.float32(abs(magnitude) or 1.0)) * np.sqrt(max(depth, 1)))


def compare(published: pd.Series, regenerated: pd.Series, tol: float | None = None) -> dict:
    """Compare two aligned numeric columns. Align them by key BEFORE calling this."""
    if tol is None:
        tol = tolerance_from_precision(published)
    a = pd.to_numeric(published, errors="coerce").to_numpy(float)
    b = pd.to_numeric(regenerated, errors="coerce").to_numpy(float)
    diff = np.abs(a - b)
    diff[np.isnan(a) & np.isnan(b)] = 0.0
    # A value that is blank on one side and present on the other, or that `to_numeric` could not
    # read, is a mismatch. Left as NaN it would be skipped by `nanmax` and counted as zero by
    # `nansum`, so the comparison would pass on a row it never actually looked at.
    diff[np.isnan(a) ^ np.isnan(b)] = np.inf
    return {
        "n": int(len(a)),
        "max_abs_diff": float(np.nanmax(diff)) if len(diff) else 0.0,
        "n_over_tolerance": int(np.nansum(diff > tol)),
        "tolerance": tol,
        "pass": bool(np.nanmax(diff) <= tol) if len(diff) else True,
    }
