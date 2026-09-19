# FOXA2 and SOX2 versus FOXF1 in day 2 cysts

Two pixel-density plots on one panel: FOXA2 against FOXF1 on top, SOX2 against FOXF1 on the
bottom. Seven day 2 cysts from three independent differentiations, pooled. Feeds **ED Fig 3g**.

| File | What it does |
|---|---|
| `notebooks/foxa2_sox2_foxf1_correlation.ipynb` | Reads the seven cyst crop folders, masks and rescales them, draws both panels of ED Fig 3g, and writes the per-pixel table to `output/derived/` |
| `scripts/render_ed3g_panel.py` | Redraws both panels of ED Fig 3g from `derived/` and prints the pixel counts and quadrant percentages. Runs as shipped, with no images and nothing to configure |
| `derived/ed3g_per_pixel.tsv` | One row per analysed pixel: FOXA2, FOXF1 (FOXA2 panel), SOX2, FOXF1 (SOX2 panel), each rescaled to 0-1 (504,789 rows) |
| `derived/ed3g_summary.tsv` | The constants the notebook used — offsets, gates, scales, seed, bin count — and the quadrant proportions it printed for both panels |

## How the numbers are made

1. **Region.** Each of the seven cysts is delineated by hand in brightfield (`select_bead.png`,
   red for the cyst, green for the bead); the bead is excluded from the analysed region.
2. **Background.** A constant offset is subtracted from each channel, set by hand per cyst and
   per channel (`CYSTS` in the notebook). For a given cyst the FOXF1 offset is the same in both
   panels, since it is the same image. Values below zero are clamped to zero (`SUBBACKGROUND =
   "clamp"`).
3. **Pooling and scaling.** The seven cysts' pixels are concatenated and each channel is divided
   by its maximum over the pooled pixels, so every axis spans 0 to 1. Both panels pool the same
   **504,789** pixels.
4. **Dither.** A small uniform jitter (`SEED = 0`, `JITTER = 0.5`, half an 8-bit quantization
   step) is added before binning, to suppress the lattice artifact of 8-bit data. **The quadrant
   statistics below are computed from the undithered values**, not the dithered copy used for the
   density image.
5. **Panel.** A 256 x 256 two-dimensional histogram of the dithered values, counts through
   `log1p` and divided by their maximum, drawn on the notebook's own FOXA2 (white-to-magenta) and
   SOX2 (white-to-yellow) colormaps, with twelve iso-density contours from 0.05 to 1 over a
   Gaussian-smoothed copy. The fill is then divided by the 99th percentile of the normalized bin
   density and clipped at 1 (`CLIP_PCT`), so 1 on the colour bar means "at or above that
   percentile": the bin at the origin is far denser than every other bin, and scaling to the
   maximum leaves the rest of the panel almost white. The contours are drawn from the unclipped
   values, and the quadrant statistics do not use the histogram at all.
6. **Positivity gates.** FOXF1 positivity is gated at 45 raw units in both panels; FOXA2 and SOX2
   are each gated at 15 raw units. Each gate is divided by the axis's own scale to place it on the
   normalized panel (FOXA2: 15/48 = 0.3125; SOX2: 15/44 = 0.3409...; FOXF1: 45/140 = 0.3214...).

## The shipped table and the gate precision

`derived/ed3g_per_pixel.tsv` holds the same undithered values the notebook exports as source
data, rounded to six decimals, in the same column order as the Source Data workbook's `Extended
Data Fig.3g` sheet (columns A/B for the FOXA2 panel, D/E for the SOX2 panel).

The FOXA2 gate, 15/48 = 0.3125, terminates exactly at six decimals, so comparing the rounded
table against the gate reproduces the notebook's printed quadrant counts exactly: `Qlr` = 30,685,
`Qul` = 16,228, `Qur` = 1,627.

The SOX2 gate, 15/44 = 0.3409090909..., does not terminate at six decimals. 9,416 pixels sit
exactly at the rounded table's quantization level closest to that gate, 0.340909, which is
9.09 x 10⁻⁸ **below** the unrounded gate value. Comparing the rounded table against the
*unrounded* gate with a plain `>=` therefore drops that entire level from the SOX2-positive
count: `Qlr` comes out 35,511 instead of 44,923. The two differ by 9,412 rather than 9,416 because
4 pixels at that level are also FOXF1-positive and sit in `Qur` either way. Rounding the gate itself to the table's own
six-decimal precision before comparing removes the artifact and reproduces the notebook's
printed counts exactly for both panels; `scripts/render_ed3g_panel.py` does this, and
`derived/ed3g_summary.tsv` records the unrounded-gate figure (`sox2_table_unrounded_gate_Qlr`)
alongside the rounded one so the difference is visible rather than silently absorbed.

Nothing here is random except the dither used only for the density image, so a re-run reproduces
the same quadrant statistics.

## Raw data

The seven cyst crop folders the notebook reads (`data/cropped2/1.5/`, `data/cropped2/1.5/1.5.2/`,
`data/cropped2/1.3/1.3.1/` through `1.3.4/`, `data/cropped2/1.7/`), each holding
`select_bead.png`, `FOXA2.png`, `SOX2.png` and `FOXF1.png`, are not in this repository. Raw
imaging is available from the lead contact on request, as the paper's Data Availability Statement
describes. Put them under `data/cropped2/` beside this file, or point `ED3G_DATA_DIR` at their
parent directory, to re-run the notebook. The panels themselves need none of them.
