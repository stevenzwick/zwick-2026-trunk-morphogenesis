"""Verify every quantified imaging panel against the paper's published Source Data.

This is the repository's own regression test. It reads each panel's derivative table, applies the
recorded subset rule, joins to the published sheet on keys, and reports the largest absolute
difference against a tolerance derived from that sheet's own decimal precision.

    SOURCE_DATA_DIR=/path/to/source_data python figures/verify_imaging_panels.py

Source Data workbooks are published with the paper; point SOURCE_DATA_DIR at them.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "figures"))
import verify_against_source_data as V          # noqa: E402
from panel_specs import SPECS, PanelSpec, plotted_rows   # noqa: E402

V.SOURCE_DATA_DIR = os.environ.get("SOURCE_DATA_DIR", str(REPO / "source_data"))

SHEET_TO_WORKBOOK = {
    "Fig.1e": "Source Data Fig.1.xlsx", "Fig.1f": "Source Data Fig.1.xlsx",
    "Fig.2c": "Source Data Fig.2.xlsx", "Fig.3f": "Source Data Fig.3.xlsx",
    "Fig.5h": "Source Data Fig.5.xlsx",
    "Fig.5i": "Source Data Fig.5.xlsx", "Fig.5n": "Source Data Fig.5.xlsx",
    "Extended Data Fig.2j": "Source Data Extended Data Fig.2.xlsx",
    "Extended Data Fig.2k": "Source Data Extended Data Fig.2.xlsx",
    "Extended Data Fig.3d": "Source Data Extended Data Fig.3.xlsx",
    "Extended Data Fig.3k": "Source Data Extended Data Fig.3.xlsx",
    "Extended Data Fig.10f": "Source Data Extended Data Fig.10.xlsx",
}

#: Fig 3c is checked separately. Its sheet publishes three unlabelled lists of per-morph values —
#: one per bead configuration, with no organoid identifier to join on — laid out as a two-level
#: header the block reader does not split. With no key, the only meaningful check is that the
#: sorted values agree, which is what `check_fig3c` does.
FIG3C_CONDITIONS = {"Unilateral bead": (0, 4, 23), "Medial bead": (4, 8, 23),
                    "Bilateral beads": (0, 4, None)}
FIG3C_MARKERS = ["FOXF1", "PAX8", "MESP2"]


def load_table(spec: PanelSpec) -> pd.DataFrame:
    """The rows the panel is drawn from — the same call the renderer makes."""
    return plotted_rows(spec)


def check(spec: PanelSpec) -> dict:
    published = V.load_blocks(SHEET_TO_WORKBOOK[spec.source_data], spec.source_data)[spec.block][1]
    published.columns = [str(c).strip() for c in published.columns]
    table = load_table(spec)

    left = pd.DataFrame(index=published.index)
    right = table.copy()
    right_keys = []
    for sd_key, tbl_key in spec.keys.items():
        col = published[sd_key]
        if sd_key in spec.key_values:
            col = col.astype(str).map(spec.key_values[sd_key])
            if col.isna().any():
                raise KeyError(f"{spec.panel}: unmapped values in key '{sd_key}'")
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().all():
            # Round BOTH sides to the published key's own precision. A sheet often stores a
            # rounded axis value while the table keeps full float; rounding only one side, or
            # both to an arbitrary depth, drops every row.
            decimals = (col.astype(str).str.extract(r"\.(\d+)$", expand=False)
                        .dropna().str.len())
            nd = min(int(decimals.max()) if len(decimals) else spec.round_keys, spec.round_keys)
            left[tbl_key] = num.round(nd)
            right[tbl_key] = pd.to_numeric(right[tbl_key], errors="coerce").round(nd)
        else:
            left[tbl_key] = col.astype(str)
            right[tbl_key] = right[tbl_key].astype(str)
        right_keys.append(tbl_key)

    merged = left.merge(right, on=right_keys, how="left", indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    results, worst, tol_used = [], 0.0, None
    for sd_col, tbl_col in spec.columns.items():
        tol = V.tolerance_from_precision(published[sd_col])
        r = V.compare(published[sd_col], merged[tbl_col], tol)
        results.append((sd_col, r))
        worst = max(worst, r["max_abs_diff"])
        # Report the STRICTEST tolerance applied, not the loosest: every column is checked against
        # its own, and showing the loosest (usually an integer count column, 0.5) would overstate
        # how lax the check is.
        tol_used = tol if tol_used is None else min(tol_used, tol)
    # The left join proves only that every PUBLISHED row exists in our table. Rows the table
    # has and the sheet does not are invisible to it — and the panel is drawn from ALL of them.
    # Requiring the two row sets to match in both directions is what makes "verified" mean
    # "we plot what the paper plots" rather than "the paper's rows are somewhere in our table".
    extra = len(table) - len(published)
    ok = (len(published) > 0 and unmatched == 0 and extra == 0
          and all(r["pass"] for _, r in results))
    return {"panel": spec.panel, "n": len(published), "unmatched": unmatched, "extra": extra,
            "max_abs_diff": worst, "tolerance": tol_used, "pass": ok, "columns": results}


def check_fig3c() -> dict:
    """Compare the plotted Fig 3c values against the published sheet, per condition and marker."""
    import openpyxl
    book = openpyxl.load_workbook(Path(V.SOURCE_DATA_DIR) / "Source Data Fig.3.xlsx",
                                  read_only=True, data_only=True)
    rows = list(book["Fig.3c"].iter_rows(values_only=True))
    header = next(i for i, r in enumerate(rows)
                  if any(isinstance(v, str) and v.startswith("FOXF1") for v in r))
    stacked = next(i for i, r in enumerate(rows[header + 1:], header + 1)
                   if any(isinstance(v, str) and v.startswith("FOXF1") for v in r))

    table = pd.read_csv(REPO / "imaging/bilaterality_fig3c/derived"
                               "/fig3c_bilaterality_index_plotted.csv")
    published_n = regenerated_n = 0
    worst, tol_used = 0.0, None
    for condition, (first, last, end) in FIG3C_CONDITIONS.items():
        start = header + 1 if end else stacked + 1
        stop = end if end else len(rows)
        names = rows[header if end else stacked]
        for column in range(first, last):
            if not isinstance(names[column], str):
                continue
            marker = names[column].split("_")[0]
            published = pd.Series([r[column] for r in rows[start:stop]
                                   if column < len(r) and isinstance(r[column], (int, float))])
            mine = table[(table["condition"] == condition) & (table["marker"] == marker)]
            published_n += len(published)
            regenerated_n += len(mine)
            tol = V.tolerance_from_precision(published)
            result = V.compare(published.sort_values(ignore_index=True),
                               mine["bilaterality_index"].sort_values(ignore_index=True), tol)
            worst = max(worst, result["max_abs_diff"])
            tol_used = tol if tol_used is None else min(tol_used, tol)
    return {"panel": "Fig 3c", "n": published_n, "unmatched": abs(published_n - regenerated_n),
            "max_abs_diff": worst, "tolerance": tol_used,
            "pass": published_n == regenerated_n and worst <= tol_used}


def main() -> int:
    if not Path(V.SOURCE_DATA_DIR).is_dir():
        print(f"Source Data directory not found: {V.SOURCE_DATA_DIR}\n"
              "Set SOURCE_DATA_DIR to the folder holding the published workbooks.", file=sys.stderr)
        return 2
    print(f"{'panel':28s} {'n':>6s} {'unmatched':>10s} {'extra':>7s} "
          f"{'max|diff|':>11s} {'tol':>9s}  result")
    failures = 0
    for spec in SPECS:
        try:
            r = check(spec)
        except Exception as exc:                                    # noqa: BLE001
            print(f"{spec.panel:28s} {'ERROR':>6s}  {exc}")
            failures += 1
            continue
        failures += 0 if r["pass"] else 1
        print(f"{r['panel']:28s} {r['n']:6d} {r['unmatched']:10d} {r['extra']:7d} "
              f"{r['max_abs_diff']:11.3e} {r['tolerance']:9.0e}  "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    checks = len(SPECS) + 1
    try:
        r = check_fig3c()
        failures += 0 if r["pass"] else 1
        print(f"{r['panel']:28s} {r['n']:6d} {r['unmatched']:10d} {'-':>7s} "
              f"{r['max_abs_diff']:11.3e} {r['tolerance']:9.0e}  "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    except Exception as exc:                                        # noqa: BLE001
        print(f"{'Fig 3c':28s} {'ERROR':>6s}  {exc}")
        failures += 1
    print(f"\n{checks - failures}/{checks} panels reproduce from the shipped derivative layer.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
