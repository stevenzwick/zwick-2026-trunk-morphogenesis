# Bilaterality index — Fig 3c

Left–right symmetry of FOXF1, PAX8 and MESP2 in day-5 trunk morphs, across bead configurations:
a single medial bead, a single unilateral bead, or two bilateral beads.

`index = 1 - |L - R| / (L + R)`. Significance by a two-sided, experiment-stratified permutation test
with Holm correction across the three markers, comparing **unilateral** against the pooled
symmetric-input conditions (**medial + bilateral**).

| File | Role |
|---|---|
| `plot_bilaterality_with_stats.py` | **Canonical panel** — pooled comparison and the main figure exports; runs from `derived/` |
| `fig3c_exact_stats.py` | **Canonical statistics** — the exact permutation test and effect-size CIs quoted in the Methods |
| `plot_bilaterality_grouped_by_marker.py` | Alternate layout, same statistics |
| `plot_different_group_3channel.ipynb` | The collaborator's original notebook |
| `clean_replot/` | A clean re-render of the same figure (Arial, tidied labels; identical data, layout, colours and stars) |
| `derived/fig3c_bilaterality_index_plotted.csv` | **The published values** — the 127 points in Fig 3c |
| `derived/05b_day5_feature_handoff_table__*.tsv` | The two per-experiment feature tables the analysis reads |

## Two tables, and which one is the panel

Two sets of numbers for the same organoids live here, and the difference matters.

`derived/fig3c_bilaterality_index_plotted.csv` holds **the values actually plotted**, recovered from
the literal lists in `notebooks/plot_different_group_3channel.ipynb`.
`figures/verify_imaging_panels.py` compares them with Source Data Fig.3c.

`derived/05b_day5_feature_handoff_table__*.tsv` are the organoid-level feature tables — the same
experiments and the same organoids, re-exported later. Group sizes match exactly (19 unilateral ·
17 medial · 7 bilateral, including the medial group's PAX8 n = 15) and individual values agree to
within 3e-02, but they are not value-for-value the published export. They are the right input for
re-running the pipeline and the wrong one for reproducing the panel.

The organoid identities linking the two were inferred by matching values, and are hard-coded in
`plot_bilaterality_with_stats.py` as `PLOT_IDS`. The collaborator's original filtered export is not
available; the notebook's literals stand in for it.

`clean_replot/` came from a Supplementary Data section that was cut. It was written as SD4 §4
and then dropped because it duplicated main-text Fig 3c; SD4's later sections renumbered. The code
is kept here because the re-render is the tidiest version of the panel, not because it ships as a
supplementary section.

## Two experiments, merged

`derived/05b_day5_feature_handoff_table__2026-01-02.tsv` and `…__2026-02-27.tsv` are **different
experiments**, not versions of one table. Both contribute points, and the permutation test is
stratified by experiment.

## Which values the published statistics were computed on

The statistics quoted in the Supplementary Data 4 caption were computed on the **feature tables**,
not on the plotted values. Both are here — but see the caveat below: **the two columns differ by
about one Monte Carlo standard deviation, so the difference between them is not meaningful.**

| marker | on the feature tables (as published) | on the plotted values |
|---|---|---|
| FOXF1 | 3.000e-06 | 3.000e-06 |
| PAX8 | 2.700e-04 | 2.56e-04 |
| MESP2 | 0.7947 | 0.8046 |

No annotation changes — `****`, `***`, n.s. either way. Three things follow for anyone quoting a
number.

**FOXF1 is at the permutation floor.** 1,000,000 permutations with Holm across three markers cannot
produce a value below 3/(10⁶+1) = 3.000e-06, so the honest statement is *p* < 3×10⁻⁶ rather than an
equality.

⚠️ **Neither PAX8 column is precise to the digits shown, and they are not independent of each
other.** Both were computed with the same seed, so they share one permutation stream and their
errors are correlated. Measured across ten seeds, PAX8's raw *p* has a standard deviation of about
3e-05 at 200,000 permutations — roughly 1.4e-05 at 1,000,000 — and the gap between the two columns
is 1.4e-05. That is one standard deviation. Treat them as one estimate with uncertainty in the last
two digits, not as two findings.

**So the exact test is the one to quote, and it is in `fig3c_exact_stats.py`.** The full enumeration
is ~1.9e11 relabellings for FOXF1 and MESP2 and ~5.2e10 for PAX8, reachable by meet-in-the-middle
subset sums rather than sampling, which removes the seed dependence and the floor together. Those
are the values the Methods quotes.

## Exact test and Monte Carlo

Two scripts compute the same comparison two ways, and the Methods quotes the exact one.

| | `fig3c_exact_stats.py` | `plot_bilaterality_with_stats.py` |
|---|---|---|
| Null | every within-experiment relabelling | 1e6 sampled, seed 20260412 |
| Input | the published values, matched to experiments | the reconstruction from the feature tables |
| FOXF1 Holm *P* | 5.123e-10 | 3.000e-06, the floor |
| PAX8 Holm *P* | 2.919e-04 | 2.700e-04 |
| MESP2 Holm *P* | 8.045e-01 | 7.947e-01 |

The stars are the same either way. The exact script also gives each marker's effect size — the
statistic itself, the experiment-weighted difference in mean bilaterality index — with a 95% CI
obtained by inverting the test:

| marker | difference | 95% CI |
|---|---|---|
| FOXF1 | −0.578 | −0.690 to −0.465 |
| PAX8 | −0.353 | −0.516 to −0.189 |
| MESP2 | −0.015 | −0.130 to +0.101 |

The two scripts read different tables, for the reason in "Two tables, and which one is the panel"
above: the exact test reports on the values that are actually plotted, and the published export
carries no experiment label, so each value is matched to its reconstructed counterpart to recover
the strata. The worst residual in that matching is 0.0303, and the script prints it. Running the
exact test on the reconstruction instead moves FOXF1 to 4.643e-10, PAX8 to 3.063e-04 and MESP2 to
7.947e-01 — the same conclusions.

```bash
python imaging/bilaterality_fig3c/scripts/plot_bilaterality_with_stats.py   # the panel
python imaging/bilaterality_fig3c/scripts/fig3c_exact_stats.py             # the quoted statistics
```

The exact script takes a few minutes: each CI bound is found by bisection, and every step is another
full enumeration.
