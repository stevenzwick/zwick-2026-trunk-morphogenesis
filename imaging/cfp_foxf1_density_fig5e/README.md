# 10% BMPR1A knockdown morphs: CFP versus FOXF1 pixel density (Fig 5e)

Day 5 morphs in which about 10% of the cells carry the BMPR1A knockdown construct, marked by CFP.
Fig 5e plots, for the pixels inside the morphs, FOXF1 intensity against CFP intensity as a density
heatmap, with the percentage of those pixels in each quadrant set by a CFP gate and a FOXF1 gate.
The three image stacks (n = 3 morphs) are pooled.

## How the plotted values are computed

The notebook computes everything the panel shows from single-plane TIFF exports of the three
confocal stacks, one file per z plane per channel. FOXF1 is read from the exports named YFP.

1. **Mask.** A pixel is kept if its DAPI intensity is above 20. The same threshold applies to every
   z plane of every stack. There is no other segmentation and no background subtraction.
2. **Dither.** The 8-bit CFP and FOXF1 values come in steps wider than one intensity unit. Before
   pooling, each stack's values get uniform random noise to spread them across the steps: ±0.6 of
   the stack's mean step for CFP and ±0.5 of it for FOXF1, clipped to 0–255. The noise is drawn from
   NumPy's global random number generator without a seed, so a re-run gives slightly different
   counts.
3. **Pooling.** The kept pixels of all z planes of the three stacks are pooled, with no subsampling
   or per-stack weighting.
4. **Histogram.** The pooled pixels are binned 256 × 256, each axis spanning its own minimum to
   maximum. Intensities stay in 8-bit units and are not rescaled.
5. **Display.** The heatmap shows log(1 + count), divided by its maximum. The colour scale runs from
   0 to the 99th percentile of that grid over all 65,536 bins, so higher bins are clipped. Contours
   are drawn at eight evenly spaced levels from 0.1 to 1 on the same grid smoothed with a Gaussian of
   σ = 2 bins along CFP and 1 bin along FOXF1.
6. **Quadrants.** The gates are CFP = 80 and FOXF1 = 65, fixed values set in the notebook, drawn as
   dashed lines. A pixel at or above a gate is positive for that channel. Each quadrant's percentage
   is its share of all pooled pixels.

The notebook's figure cell draws the panel and saves it as `output/densityheatmap_FOXF1-CFP.{pdf,png,svg}`.
The notebook ships with the outputs of the original analysis session that saved the figure.

## Files

| File | What it does |
|---|---|
| `notebooks/cfp_foxf1_pixel_density.ipynb` | Loads the stacks, masks, dithers and pools the pixels, draws Fig 5e, and writes the tables in `output/derived/` |
| `scripts/render_fig5e_panel.py` | Redraws Fig 5e from `derived/` and prints n, the pixel count, the gates and the quadrant percentages |
| `derived/fig5e_histogram_counts.tsv` | Pixel count in each non-empty bin of the 256 × 256 histogram (bins not listed are empty) |
| `derived/fig5e_bin_edges.tsv` | The 257 bin edges of each axis, in 8-bit intensity units |
| `derived/fig5e_summary.tsv` | n, the pooled pixel count, the DAPI threshold, the two gates and the pixel count in each quadrant |
| `derived/PROVENANCE.tsv` | Size and SHA-256 of each table, and the run that wrote it |

The tables in `derived/` were written by the notebook's last cell in a run of this notebook on the
three stacks. Because of the unseeded dither, a new run writes slightly different tables, so the
notebook writes to `output/derived/` and leaves `derived/` as shipped.

The notebook's first cell lists the cells kept from the original analysis notebook and every change.

## Raw data

The images are not in this repository. They are available from the lead contact on request, as the
paper's Data Availability Statement states. The notebook expects nine folders of single-plane TIFFs
in `data/mutant analysis/` inside this directory, or in the directory named by the environment
variable `FIG5E_IMAGE_ROOT`:

| Stack | DAPI | CFP | FOXF1 | z planes |
|---|---|---|---|---|
| 1 | `0303-1-DAPI-2/` | `0303-1-CFP-2/` | `0303-1-YFP-2/` | 111 |
| 2 | `0303-2-DAPI/` | `0303-2-CFP/` | `0303-2-YFP/` | 99 |
| 3 | `0315-DAPI/` | `0315-CFP/` | `0315-YFP/` | 107 |

Each folder holds one 1024 × 1024 8-bit TIFF per z plane, and files sort in z order by name.

## Running it

From the repository root, with no images needed:

```bash
python imaging/cfp_foxf1_density_fig5e/scripts/render_fig5e_panel.py
```

This writes `figures/output/fig5e_cfp_foxf1_pixel_density.{png,pdf}`.

To recompute from the images, open `notebooks/cfp_foxf1_pixel_density.ipynb` in Jupyter from anywhere
inside the repository and run all cells. The figure and the tables go to `output/` in this directory.
Then `render_fig5e_panel.py --derived imaging/cfp_foxf1_density_fig5e/output/derived` redraws the
panel from that run's tables.
