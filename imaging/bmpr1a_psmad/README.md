# BMPR1A knockdown — pSMAD1/5/9 validation

Quantifies pSMAD1/5/9 in cells at the colony edge after BMP4 treatment, comparing BMPR1A knockdown
against scramble control. Feeds **ED Fig 3k** and **Supplementary Data 4 §4**.

Run the numbered scripts in order:

| Script | Does |
|---|---|
| `01_segment_nuclei.py` | Nuclear segmentation |
| `02_dedge_quant.py` | Per-cell distance-to-edge and intensity quantification |
| `03_figure.py` · `07_panel_figure.py` | Panels (the nine-box Supplementary Data 4 display) |
| `04_stats.py` | Statistics |
| `05_timecourse.py` | Time-course comparison |
| `06_edge_profile.py` | Intensity vs distance from edge |
| `08_ed3k_panel.py` | **ED Fig 3k** — runs as shipped, from `derived/` |

`czi_io.py` wraps CZI reading. It uses **aicspylibczi**, not `czifile`: `czifile` reads
odd-sized CZIs (1025 / 1026 px) non-deterministically, returning garbage or saturated pixels for a
file whose checksum is stable. Use the wrapper.

## Raw data

The `.czi` source images are not in this repository. Raw imaging is available from the lead contact
on request, as the paper's Data Availability Statement describes. `derived/` ships the per-field edge summary
(`edge_summary_per_field.csv`), the per-well timecourse (`timecourse_per_well.csv`), and a
**single-field** per-cell QC export (`per_cell_dedge_QC.csv`, field `10x-1` only).

It also ships **`negctrl_summary_per_field_30min.csv`** — 27 rows, three conditions x three
categories x three fields — which holds the per-well medians behind **ED Fig 3k**.

⚠️ **Scripts `03`-`07` do not draw ED Fig 3k.** They draw the **nine-box** figure (three knockdown
conditions x no-BMP4 edge / BMP4 edge / BMP4 interior), which is the Supplementary Data 4 display.
ED 3k is the **four-box** panel: SCRAMBLE KD and BMPR1A KD, each without and with BMP4, three wells
each — the twelve values in the table above. Those scripts read
`results/tables/per_cell_dedge.csv`, the full per-cell table produced by `02_dedge_quant.py` from
the raw images, which is **not included**, so the nine-box SD4 figure cannot be rebuilt here.

**`08_ed3k_panel.py` draws ED Fig 3k, and it runs as shipped** — it needs only
`derived/negctrl_summary_per_field_30min.csv`, not the raw images. It takes its twelve plotted rows
from `figures/panel_specs.py`, the same call `figures/verify_imaging_panels.py` makes, so the
renderer and the verifier cannot drift onto different row sets. Output is
`figures/output/ed3k_psmad_edge_knockdown.{pdf,png}`, drawn at the published panel's measured
geometry (axes 25.3 x 26.3 mm at 300 ppi). With `SOURCE_DATA_DIR` set it also asserts that the
twelve plotted values equal the published sheet.

*(The original working directory also held a Google Drive download helper bound to one author's
OAuth credentials. It is not published: no reader could use it, and it embedded a credential path.)*
