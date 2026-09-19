# Imaging analyses

One directory per assay, each named for what it measures rather than for a figure panel — several
panels share an assay, and panel letters moved between submissions.

| Directory | Assay | Feeds |
|---|---|---|
| `psmad_fig1/` | pSMAD1/5/9 gradient, dataset 1 | Fig 1e, 1f |
| `psmad_edfig2/` | pSMAD1/5/9 gradient, **dataset 2** | ED Fig 2j, 2k |
| `foxf1_bead/` | Marker intensity vs distance from bead / central cyst | Fig 2c, ED Fig 3d (left, right) |
| `foxf1_bmp4_day2_fig2g/` | FOXF1 / BMP4 reporter pixel density in day 2 cysts | Fig 2g |
| `cfp_foxf1_density_fig5e/` | CFP vs FOXF1 pixel density in day 5 morphs with 10% BMPR1A knockdown cells | Fig 5e |
| `morphometry/` | Day-5 whole-morph geometry and mediolateral profiles | Fig 3f |
| `bilaterality_fig3c/` | FOXF1 / PAX8 / MESP2 bilaterality by bead geometry | Fig 3c |
| `timelapse_fig5i/` | FOXF1-RFP / BMP4-YFP reporter timelapse, **2D** | Fig 5i, ED Fig 10f |
| `timelapse_3d_fig5h/` | Reporter timelapse, **3D organoid** — a different experiment | Fig 5h, ED Fig 10d |
| `transplant_fig5n/` | LPM graft, host vs donor FOXF1⁺ pixels — ⚠️ see its README on the Dunnett p-value | Fig 5n |
| `cograft_fig5/` | LPM + NMP co-graft quantification | SD 4 §5 |
| `bmpr1a_psmad/` | pSMAD in BMPR1A knockdown | ED Fig 3k, SD 4 §4 |

`psmad_fig1/` and `psmad_edfig2/` are two different experiments, not two versions of one.
The same is true of the morphometry cohorts. Where directory names look like versions of each other,
they are usually separate datasets.

## `derived/` — what ships instead of raw images

Raw imaging stays with the lead contact (see the paper's Data Availability Statement). Each assay
directory ships a `derived/` folder holding the measurement tables the figures are actually drawn
from, with a `PROVENANCE.tsv` recording each file's original path, size and SHA-256.

The quantified imaging panels with a published Source Data sheet are computed from these tables.
`figures/render_imaging_panels.py` draws most of them and `bmpr1a_psmad/scripts/08_ed3k_panel.py`
draws ED Fig 3k; `figures/verify_imaging_panels.py` compares those panels' plotted rows with the
published Source Data. Fig 2g is drawn from `foxf1_bmp4_day2_fig2g/derived/` by that lane's
`scripts/render_fig2g_panel.py`, and Fig 5e from `cfp_foxf1_density_fig5e/derived/` by that lane's
`scripts/render_fig5e_panel.py`. Both sheets publish the binned counts the panel draws rather than the
per-pixel values, which run to 15.4 and 64.9 million points, so neither is compared row by row here.

Five panels are drawn by their own lane's script rather than by `render_imaging_panels.py`, and are
therefore outside what `verify_imaging_panels.py` covers: ED Fig 3k, Fig 2g, Fig 5e, ED Fig 3g and
ED Fig 3o. Each of those scripts checks its own numbers as it draws — ED 3k against the published
sheet when `SOURCE_DATA_DIR` is set, and the other lanes against the values recorded in their own
`derived/` summary table. The
per-cell table behind the nine-box Supplementary Data 4 display in `bmpr1a_psmad/` is not shipped;
see `bmpr1a_psmad/README.md`. ED Fig 3d (right) and ED Fig 10d are not drawn from `derived/`:
`foxf1_bead/notebooks/09_fixed_plotting.ipynb` and
`timelapse_3d_fig5h/notebooks/05b_reporter_spatial_context.ipynb` draw them from pixel intermediates
built from the raw images, which are not shipped.
