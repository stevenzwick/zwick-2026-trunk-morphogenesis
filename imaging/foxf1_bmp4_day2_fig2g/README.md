# Day 2 FOXF1 and BMP4 reporters: pixel density (Fig 2g)

Day 2 cysts from the double reporter line (BMP4-2A-Achilles, imaged in the YFP channel;
FOXF1-2A-mScarlet3, imaged in the RFP channel), acquired by epifluorescence as z-stacks with DAPI
and brightfield. Fig 2g plots FOXF1 against BMP4 reporter intensity for the pixels inside the cysts,
with the percentage of those pixels in each quadrant and the Pearson R² over all of them.

## Redrawing Fig 2g

```bash
python imaging/foxf1_bmp4_day2_fig2g/scripts/render_fig2g_panel.py
```

This needs only this repository. The script reads the three tables in `derived/`, draws the panel to
`figures/output/fig2g_foxf1_bmp4_pixel_density.{png,pdf}`, and prints n, the Pearson R² and the
quadrant percentages.

| File in `derived/` | What it holds |
|---|---|
| `fig2g_histogram_counts.tsv` | pixel count in each of the panel's 96 × 96 bins, before normalisation (9,216 rows) |
| `fig2g_bin_edges.tsv` | the bin edges on each axis, in display units |
| `fig2g_summary.tsv` | n, the pooled pixel count, the two cutoffs, the display settings, the pixel count in each quadrant, and the sums over the pixels (Σx, Σy, Σx², Σy², Σxy) from which R² is computed |

The last cell of notebook `03` writes these tables. `derived/PROVENANCE.tsv` gives each file's size
and SHA-256.

## How the plotted values are computed

Notebook `03` computes everything the panel shows, from the raw stacks and the DAPI masks that
notebook `01` writes:

1. **Masks.** Each z plane of each stack has a whole-cyst mask made from its DAPI channel (notebook
   `01`).
2. **Background.** For each stack, the pixels outside the mask are pooled over all its z planes, and
   their median is subtracted from that stack's FOXF1 and BMP4 intensities.
3. **Cutoffs.** The background-subtracted pixels outside the masks are pooled over all stacks, and
   the 99.5th percentile of each channel is that channel's positivity cutoff. The cutoffs are the
   dashed lines, and they define the quadrants.
4. **Pixels.** Every pixel inside the mask, from every z plane of every stack, is pooled, with no
   subsampling or per-stack weighting. R² is the squared Pearson correlation over these pixels, and
   each quadrant's percentage is its share of them.
5. **Display.** FOXF1 is divided by 1500 and BMP4 by 200. The axes end at 1600 and 350 in raw
   units, or at the last bin if that is lower. The pooled pixels are binned 96 × 96 over their
   0.1–99.9th percentile range, rounded outward; counts are divided by the largest bin, clipped at
   the 99.5th percentile of the non-empty bins and rescaled to 0–1, which is the colour bar.
   Contours are drawn at 0.15, 0.35, 0.55 and 0.75 on that grid smoothed with a Gaussian of
   σ = 1.3 bins.

The second-to-last cell of notebook `03` draws the panel and saves it as
`results/figures/03_foxf1_bmp4_density_global_q995.{png,svg,pdf}`; the last cell writes `derived/`.
The notebook ships with its outputs, including the printed cutoffs, R² and quadrant fractions.

## Files

| File | What it does |
|---|---|
| `notebooks/00_manifest_qc.ipynb` | Lists the raw stacks with their channel names, pixel size and review plane, writes `results/manifests/raw_input_manifest.tsv`, and previews each stack's review plane |
| `notebooks/01_dapi_whole_organoid_mask_review.ipynb` | Calls `scripts/run_pixel_level_quantification.py` to write the DAPI masks for every z plane (`results/masks/`) and the per-plane metrics (`results/tables/01_mask_and_plane_metrics.tsv`), then shows mask QC |
| `notebooks/02_reporter_pixel_quantification.ipynb` | Quality control of the cutoffs: intensity histograms and thresholded review planes, saved under `results/qc/02_threshold_review/`. Notebook `03` computes the same cutoffs itself |
| `notebooks/03_global_q995_density_review.ipynb` | Computes the values described above, draws Fig 2g and writes `derived/` |
| `scripts/render_fig2g_panel.py` | Redraws Fig 2g from `derived/` alone |
| `scripts/day2_quantification_helpers.py` | Image loading, DAPI segmentation, background masks and the other shared helpers |
| `scripts/run_pixel_level_quantification.py` | The quantification stage that notebook `01` calls |
| `scripts/build_analysis_manifest.py` | The manifest builder that notebook `00` calls |
| `scripts/execute_pipeline_notebooks.py` | Runs the four notebooks in order and saves executed copies under `results/executed_notebooks/` |

Besides the masks and per-plane metrics, the call in notebook `01` also writes threshold,
sampled-pixel, containment and density-map tables (`02_` to `04_` under `results/tables/`). They
use a per-plane background and a signal-to-noise cutoff, and neither notebook `02` nor notebook `03`
reads them.

Each notebook's first cell lists what changed from the original analysis.

## Raw data

The raw images are not in this repository. They are available from the lead contact on request, as
the paper's Data Availability Statement states. The notebooks expect the 14 CZI stacks (`1.czi` to
`11.czi`, `13.czi`, `15.czi` and `16.czi`) and their paired review TIFs in `data/with DAPI/` inside
this directory. Channel order in the CZIs is brightfield, DAPI, mCherry (FOXF1 reporter) and TagYFP
(BMP4 reporter). A paired TIF's name marks the stack's review plane, the one shown in the QC
figures: `5(z3).tif` marks z plane 3 of `5.czi`. The TIFs' pixels are not read.

## Re-running from the raw images

From the repository root:

```bash
cd imaging/foxf1_bmp4_day2_fig2g
python scripts/execute_pipeline_notebooks.py    # 00, 01, 02, 03 in order
```

Everything is written under `results/` in this directory, except that notebook `03` also rewrites
`derived/`. To open the notebooks in Jupyter instead, first create `results/` here
(`mkdir results`): each notebook's first cell finds this directory by looking for `scripts/` and
`results/` in its working directory or the one above.
