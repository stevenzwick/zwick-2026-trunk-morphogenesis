# LPM graft — host vs donor FOXF1⁺ pixels (Fig 5n)

Donor LPM is grafted onto a host morph; the question is how much of the resulting FOXF1⁺ signal
belongs to the **host** rather than the graft. Feeds **Fig 5n** (host fraction by condition).

`derived/02_foxf1_positive_identity_summary.tsv` is the table Fig 5n reads. It is long-form with
four nested summary levels in one file — `plane`, `file`, `condition`, `global` (87 rows). **Fig 5n
uses `summary_level == "file"`**, one row per morph, 9 of the 87. Reading the file without that
filter silently mixes per-plane and per-condition rows into the same plot.

## Two ways to assign a FOXF1⁺ pixel, and Fig 5n uses one of them

The table carries three different host/donor assignments, and they are not interchangeable:

| Column family | Rule |
|---|---|
| `*_strict_*` | Pixel carries unambiguous host or donor reporter signal |
| `*_class_*` | Pixel assigned by classifier, mixed pixels resolved |
| `*_partition_*` | Whole-morph lineage partition, every pixel assigned |

**Fig 5n plots `foxf1_positive_host_strict_fraction`** — the Source Data column is
*"Host fraction of FOXF1+ (strict)"*. The co-graft comparison figures in `../cograft_fig5/` use the
`partition` family instead, because their donor has no reporter and must be scored by graft
exclusion. Mixing the two produces a plot that looks reasonable and is wrong.

⚠️ **The shipped notebook ends on `partition`, and says so.** `notebooks/02_host_donor_pixel_quantification.ipynb`
draws its final bar plot from `foxf1_positive_host_partition_fraction`, and the markdown above that
cell calls it "the direct readout we are now using for the bar plot", describing the strict
fractions as "older". That is where the notebook ended up; it is **not** what the paper reports.
The notebook carries a note at that cell saying so, and the code cells are left exactly as they
ran, so the notebook remains an accurate record rather than a rewritten one.

The two measures differ by a mean of 0.6 percentage points per morph (maximum 1.6), and the
effect, its direction and its significance are identical either way. They differ only in the
treatment of ambiguous pixels, and that share is not negligible — in the LDN condition 10.2% of
FOXF1+ pixels are mixed and 21.1% unlabeled. The published panel takes the conservative option:
a pixel counts only where the reporters identify it unambiguously.

## ⚠️ The Dunnett p-value is not reproducible run to run

The Methods report a **two-sided Dunnett multiple-comparison test** against the control. The test
statistic is deterministic:

| Comparison | statistic |
|---|---|
| BMPR1A-KD vs ctrl | −10.9618 |
| LDN vs ctrl | −11.8362 |

The **p-value is not**. `scipy.stats.dunnett` evaluates the multivariate-t by Monte Carlo, and
without a seed it returns a different p on every call for identical input. Over 300 runs here:

| Comparison | median P | range | P > 1e-4 |
|---|---|---|---|
| BMPR1A-KD | 4.3e-05 | 6.8e-06 – 3.3e-04 | **15% of runs** |
| LDN | 2.7e-05 | 2.0e-06 – 3.6e-04 | **11% of runs** |

The true value sits near the median, comfortably below 1e-4, so `****` is the correct annotation.
But roughly one run in seven lands above 1e-4 and would print `***` instead, and no two runs agree
on the digits. **Quote a threshold, not an exact p, or pass `random_state=` to fix it.** No p-value
above 1e-3 occurred in 300 runs, so `P < 0.001` is safe unconditionally.

```bash
python imaging/transplant_fig5n/check_fig5n_stats.py            # statistic + seeded p
python imaging/transplant_fig5n/check_fig5n_stats.py --spread   # demonstrates the variability
```

## Raw data

The `.czi` source images are not in this repository — raw imaging is available from the lead contact
per the paper's Data Availability Statement. `derived/` ships the per-plane and per-morph
measurement tables the panels are drawn from.
