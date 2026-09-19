#!/usr/bin/env python3
"""Recompute the ED Fig 3j statistic from the published Source Data.

The notebook `bulkseq/notebooks/ed3j_bmpr1a_knockdown.ipynb` runs a two-sided Welch's t-test. This
recomputes that statistic from the published Source Data sheet alone.

    SOURCE_DATA_DIR=/path/to/source_data python bulkseq/check_ed3j.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
from scipy import stats

SHEET = "Extended Data Fig.3j"
BOOK = "Source Data Extended Data Fig.3.xlsx"


def main() -> int:
    directory = os.environ.get("SOURCE_DATA_DIR")
    if not directory:
        print("Set SOURCE_DATA_DIR to the folder holding the published workbooks.", file=sys.stderr)
        return 2
    import openpyxl
    rows = [list(r) for r in openpyxl.load_workbook(Path(directory) / BOOK, read_only=True)[SHEET]
            .iter_rows(values_only=True)]
    head = next(i for i, r in enumerate(rows) if r[0] == "gene_id")
    groups: dict[str, list[float]] = {}
    for row in rows[head + 1:]:
        if row[0]:
            groups.setdefault(str(row[2]).rsplit("_", 1)[0], []).append(float(row[3]))

    kd = np.array(groups["BMPR1Akd"])
    control = np.array(groups["SCRAMBLEkd"])
    t, p = stats.ttest_ind(kd, control, equal_var=False)
    v1, v2, n1, n2 = kd.var(ddof=1), control.var(ddof=1), len(kd), len(control)
    df = (v1/n1 + v2/n2) ** 2 / ((v1/n1) ** 2 / (n1-1) + (v2/n2) ** 2 / (n2-1))
    se = np.sqrt(v1/n1 + v2/n2)
    diff = kd.mean() - control.mean()
    half = stats.t.ppf(0.975, df) * se

    print(f"BMPR1A knockdown vs scrambled control, two-sided Welch's t-test (n = {n1} vs {n2})")
    print(f"  t          {t:.4f}")
    print(f"  d.f.       {df:.4f}")
    print(f"  P          {p:.4f}")
    print(f"  difference {diff:.2f} CPM   95% CI ({diff-half:.2f}, {diff+half:.2f})")
    ok = abs(t + 6.999) < 5e-3 and abs(p - 0.0102) < 5e-4
    print("\nreproduces the reported values." if ok else "\nDOES NOT reproduce.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
