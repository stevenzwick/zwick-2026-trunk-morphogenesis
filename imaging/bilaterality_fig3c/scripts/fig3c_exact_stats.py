#!/usr/bin/env python
"""Fig 3c: the exact experiment-stratified permutation test, and effect sizes with 95% CIs.

These are the values quoted in the Methods. The test compares **unilateral bead** against the pooled
symmetric-input conditions (**medial + bilateral**), stratified by experiment, with Holm correction
across the three markers.

Exact, not sampled. Every within-experiment relabelling is enumerated: C(28,13) x C(15,6) for FOXF1
and MESP2, C(26,13) x C(15,6) for PAX8, which is 1.9e11 and 5.2e10 assignments. That is tractable
because the statistic depends on the labels only through the sum of the unilateral values, so the
sums of all k-subsets are built once per stratum (meet in the middle) and the two strata are then
combined by sorting one side and binary-searching it. `plot_bilaterality_with_stats.py` samples the
same null with 1e6 draws instead; see README, "Exact test and Monte Carlo".

Effect size is the test statistic itself: the experiment-weighted difference in mean bilaterality
index, unilateral minus medial/bilateral, weighted by n_experiment / N. The CI is obtained by
inverting the test -- all shifts d0 for which H0: delta = d0, tested by subtracting d0 from the
unilateral values, has two-sided P >= 0.05. Per marker, not adjusted for multiplicity.

Input is the published values, the 127 points actually plotted in Fig 3c, each assigned to its
experiment. The published export carries no experiment label, so each value is matched to the
reconstruction built from the per-experiment feature tables, by optimal assignment within group and
marker. That matching is what supplies the strata, and nothing else. The printed worst-case residual
shows how close the two sets of numbers are; see README, "Two tables, and which one is the panel".

Runs from the repository with no configuration. Prints, and writes nothing.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
with contextlib.redirect_stdout(io.StringIO()):
    import plot_bilaterality_with_stats as P  # imported for its loaders; main() is not run

PUBLISHED = ROOT.parent / "derived" / "fig3c_bilaterality_index_plotted.csv"
MARKERS = ["FOXF1", "PAX8", "MESP2"]
GROUPS = ["unilateral bead", "medial bead", "bilateral bead"]
#: The published CSV names the conditions in title case; the reconstruction uses lower case.
CONDITION_NAMES = {
    "Unilateral bead": "unilateral bead",
    "Medial bead": "medial bead",
    "Bilateral beads": "bilateral bead",
}
ALPHA = 0.05
#: Bisection steps for each CI bound. 18 halvings of a 1.5-wide bracket land well inside the third
#: decimal the Methods quotes.
CI_STEPS = 18


def build_published_with_experiments() -> tuple[pd.DataFrame, float]:
    """The published values, each given the experiment of the reconstructed value it matches."""
    recon = P.build_plot_dataframe(P.build_reconstructed_rows(P.load_rows()))
    pub = pd.read_csv(PUBLISHED)
    pub["Group"] = pub["condition"].map(CONDITION_NAMES)
    pub["Channel"] = pub["marker"]
    pub["Score"] = pub["bilaterality_index"]

    rows, worst = [], 0.0
    for (group, marker), sub in pub.groupby(["Group", "Channel"]):
        rec = recon[(recon.Group == group) & (recon.Channel == marker)]
        if len(rec) != len(sub):
            raise SystemExit(
                f"{group}/{marker}: {len(sub)} published values but {len(rec)} reconstructed")
        cost = np.abs(sub.Score.to_numpy()[:, None] - rec.Score.to_numpy()[None, :])
        r, c = linear_sum_assignment(cost)
        worst = max(worst, cost[r, c].max())
        for i, j in zip(r, c):
            rows.append({"Group": group, "Channel": marker,
                         "Score": sub.Score.iloc[i], "Experiment": rec.Experiment.iloc[j]})
    return pd.DataFrame(rows), worst


def subset_sums(values: np.ndarray, k: int) -> np.ndarray:
    """Every sum of k of these values, by splitting them in half and combining the two sides."""
    n = len(values)
    half = n // 2
    left, right = values[:half], values[half:]

    def sums(arr, j):
        if j == 0:
            return np.zeros(1)
        return np.fromiter(
            (arr[list(c)].sum() for c in itertools.combinations(range(len(arr)), j)), float)

    out = []
    for j in range(max(0, k - (n - half)), min(k, half) + 1):
        out.append((sums(left, j)[:, None] + sums(right, k - j)[None, :]).ravel())
    combined = np.concatenate(out)
    assert len(combined) == math.comb(n, k)
    return combined


def exact_stratified(df: pd.DataFrame, marker: str):
    """Observed statistic, and how many of all relabellings are at least as extreme."""
    sub = df[(df.Channel == marker) & df.Group.isin(GROUPS)]
    total_n = len(sub)
    observed = 0.0
    strata = []
    for _, per_experiment in sub.groupby("Experiment"):
        values = per_experiment.Score.to_numpy(float)
        is_unilateral = (per_experiment.Group == "unilateral bead").to_numpy()
        n, k, total = len(values), int(is_unilateral.sum()), values.sum()
        diff = lambda a: a / k - (total - a) / (n - k)  # noqa: E731
        observed += (n / total_n) * diff(values[is_unilateral].sum())
        strata.append(((n / total_n) * diff(subset_sums(values, k)), n, k))

    (x, n1, k1), (y, n2, k2) = sorted(strata, key=lambda s: -len(s[0]))
    x = np.sort(x)
    a = abs(observed)
    # The statistic is a sum of floats, so an exactly-as-extreme relabelling can miss by rounding.
    eps = 1e-9 * max(1.0, a)
    upper = len(x) - np.searchsorted(x, a - eps - y, side="left")
    lower = np.searchsorted(x, -a + eps - y, side="right")
    count = int(upper.sum() + lower.sum())
    return observed, count, len(x) * len(y), (n1, k1, n2, k2)


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    out = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        out[i] = running
    return out


def p_at_shift(df: pd.DataFrame, marker: str, shift: float) -> float:
    """Two-sided exact P for H0: the difference equals `shift`."""
    shifted = df.copy()
    unilateral = shifted.Group == "unilateral bead"
    shifted.loc[unilateral, "Score"] = shifted.loc[unilateral, "Score"] - shift
    _, count, total, _ = exact_stratified(shifted, marker)
    return count / total


def ci_bound(df: pd.DataFrame, marker: str, inside: float, outside: float) -> float:
    """Bisect between a shift the test accepts and one it rejects."""
    for _ in range(CI_STEPS):
        mid = (inside + outside) / 2
        if p_at_shift(df, marker, mid) >= ALPHA:
            inside = mid
        else:
            outside = mid
    return (inside + outside) / 2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # The P values take seconds; the CIs take minutes, because each bound is found by bisection and
    # every step is another full enumeration. The release test wants only the P values.
    parser.add_argument("--no-ci", action="store_true",
                        help="skip the effect-size confidence intervals")
    args = parser.parse_args(argv)

    pubdf, worst = build_published_with_experiments()
    print(f"published values matched to the reconstruction: worst |diff| = {worst:.4f}", flush=True)
    print(pubdf.groupby(["Channel", "Experiment", "Group"]).size().unstack().to_string(), flush=True)

    print("\nexact experiment-stratified permutation test, unilateral vs medial + bilateral", flush=True)
    results = []
    for marker in MARKERS:
        observed, count, total, sizes = exact_stratified(pubdf, marker)
        results.append((marker, observed, count, total, count / total))
        print(f"  {marker:5s} strata (n,k) = {sizes}  weighted difference {observed:+.4f}  "
              f"at least as extreme: {count} of {total:.3e}  exact P = {count / total:.3e}",
              flush=True)

    adjusted = holm(np.array([r[4] for r in results]))
    print()
    for (marker, *_), p_holm in zip(results, adjusted):
        print(f"  {marker:5s} Holm-adjusted P = {p_holm:.3e}", flush=True)

    if args.no_ci:
        print("\neffect-size confidence intervals skipped (--no-ci)", flush=True)
        return

    print("\neffect size, and 95% CI by inverting the exact test", flush=True)
    for marker in MARKERS:
        observed, *_ = exact_stratified(pubdf, marker)
        low = ci_bound(pubdf, marker, observed, observed - 1.5)
        high = ci_bound(pubdf, marker, observed, observed + 1.5)
        print(f"  {marker:5s} difference {observed:+.3f}  95% CI ({low:+.3f}, {high:+.3f})",
              flush=True)


if __name__ == "__main__":
    main()
