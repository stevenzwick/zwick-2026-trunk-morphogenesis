# FOXF1 versus BMP4 in one day 2 cyst

Pixel-density plot of FOXF1 against BMP4 signal over one z plane of the cyst shown in ED Fig 3n.
Feeds **ED Fig 3o**.

| File | What it does |
|---|---|
| `notebooks/foxf1_bmp4_correlation.ipynb` | Reads the acquisition, masks and rescales it, draws ED Fig 3o, and writes the per-pixel table to `output/derived/` |
| `scripts/render_ed3o_panel.py` | Redraws ED Fig 3o from `derived/` and prints the pixel count and R². Runs as shipped, with no images and nothing to configure |
| `derived/ed3o_per_pixel.tsv` | One row per analysed pixel: FOXF1 and BMP4, each rescaled to 0–1 (95,287 rows) |
| `derived/ed3o_summary.tsv` | The constants the notebook used — threshold, offsets, clip percentile, bin count — and the values it printed |

## How the numbers are made

1. **Plane.** One plane, the sixth of fourteen, of a five-channel acquisition at 583 × 388. The whole
   acquired field is analysed and nothing is cropped away.
2. **Mask.** A pixel enters the analysis if its DAPI intensity exceeds 1000. That is one global
   threshold over the plane, and it keeps 95,656 pixels.
3. **Background.** A constant offset is subtracted from each channel, 1724 for FOXF1 and 880 for
   BMP4, chosen from the whole-image intensity histograms that the notebook plots first. Pixels still
   above background in both channels are kept: **95,458 of the 95,656**, so 198 are dropped, 19 for
   FOXF1 and 179 for BMP4.
4. **Outlier clip.** The brightest 0.1 per cent of each channel is dropped, so that a handful of very
   bright specks cannot drive the fit. The cut-off is the 99.9th percentile of the masked,
   background-subtracted population itself rather than a number set by hand: 946 for FOXF1 and 4966
   for BMP4, which together remove 171 pixels. **95,287 are retained.**
5. **Scaling.** Each channel is rescaled to 0–1 over the retained pixels.
6. **Panel.** A 256 × 256 two-dimensional histogram, counts through `log1p` and divided by their
   maximum, drawn white to green, with four iso-density contours at 0.25 to 1 over a Gaussian-smoothed
   copy.
7. **R².** The squared Pearson coefficient from a least-squares fit of the two channels, **0.646**.

Nothing here is random, so a re-run reproduces the same numbers.

## The shipped table and the R² in the legend

`derived/ed3o_per_pixel.tsv` holds the same values the notebook exports as source data, rounded to
six decimals. R² is unchanged by the rescaling in step 5, so the legend's value can be recomputed
from this table alone: it gives 0.6462135108 against the notebook's 0.6462134690 from the unrounded
values. Both are 0.646.

## An earlier version of this analysis

The panel was first made a different way, and that version is what an earlier draft of the legend
reported. It read three separately cropped single-plane TIFFs at 558 × 362 rather than the whole
field, and in place of step 4 it dropped bright specks using bounds set by hand, 1200 for FOXF1 and
6000 for BMP4. Its mask held 94,678 pixels and retained 94,476, giving R² 0.640.

The two differ in which pixels are analysed, not in what is measured: the field is no longer cropped,
and the outlier cut-off is computed from the data instead of chosen. The current numbers are the ones
above.

## Raw data

The acquisition the notebook reads — `BMP4-FOXF1-z8.tif`, a 16-bit ZCYX stack of shape
(14, 5, 388, 583) — is not in this repository. Raw imaging is available from the lead contact on
request, as the paper's Data Availability Statement describes. Put it in `data/` beside this file, or
point `ED3O_DATA_DIR` at it, to re-run the notebook. The panel itself needs none of it.
