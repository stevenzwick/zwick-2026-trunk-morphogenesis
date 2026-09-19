#!/usr/bin/env python3
"""Recompute the Fig 5n group comparison, and show why its p-value needs a seed.

The Methods report a two-sided Dunnett test against the control. Its statistic is deterministic;
`scipy.stats.dunnett` evaluates the multivariate-t by Monte Carlo, so the p-value is not. Run with
--spread to see the spread over repeated identical calls.

    python imaging/transplant_fig5n/check_fig5n_stats.py [--spread] [--runs N]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import dunnett

REPO = Path(__file__).resolve().parent.parent.parent
TABLE = REPO / "imaging/transplant_fig5n/derived/02_foxf1_positive_identity_summary.tsv"
#: Fig 5n plots the STRICT host assignment; see README on why the other families differ.
METRIC = "foxf1_positive_host_strict_fraction"
SEED = 1337


def host_percent() -> dict[str, np.ndarray]:
    table = pd.read_csv(TABLE, sep="\t")
    per_morph = table[table["summary_level"] == "file"]      # 9 of 87 rows
    return {name: group[METRIC].to_numpy(float) * 100.0
            for name, group in per_morph.groupby("condition")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spread", action="store_true",
                        help="repeat the unseeded test to show the p-value moving")
    parser.add_argument("--runs", type=int, default=300)
    args = parser.parse_args()

    pct = host_percent()
    print(f"Host % of FOXF1+ (strict), n = {len(pct['ctrl'])} morphs per condition")
    for condition in ("ctrl", "BMPR1A_KD", "LDN"):
        print(f"  {condition:10s} {np.round(pct[condition], 2)}")

    result = dunnett(pct["BMPR1A_KD"], pct["LDN"], control=pct["ctrl"],
                     alternative="two-sided", random_state=SEED)
    print(f"\ntwo-sided Dunnett vs ctrl (random_state={SEED})")
    for i, condition in enumerate(("BMPR1A_KD", "LDN")):
        print(f"  {condition:10s} statistic {result.statistic[i]:9.4f}   P {result.pvalue[i]:.3e}")
    print("\nThe statistic is exact. The p-value is seeded here precisely because it is not.")

    if args.spread:
        draws = np.array([dunnett(pct["BMPR1A_KD"], pct["LDN"], control=pct["ctrl"],
                                  alternative="two-sided").pvalue for _ in range(args.runs)])
        print(f"\n{args.runs} UNSEEDED runs on identical input:")
        for i, condition in enumerate(("BMPR1A_KD", "LDN")):
            column = draws[:, i]
            print(f"  {condition:10s} median {np.median(column):.2e}  "
                  f"range {column.min():.2e} - {column.max():.2e}  "
                  f"P>1e-4 in {100*(column > 1e-4).mean():.0f}% of runs")
        print("\nQuote a threshold, not digits: P < 0.001 held in every run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
