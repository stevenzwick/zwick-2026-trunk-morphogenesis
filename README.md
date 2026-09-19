# Trunk morphogenesis: localized signaling centers

Analysis code for "Engineering signaling centers to trigger human trunk morphogenesis in vitro"
(Nature, in press).

The repository is for reproducibility, not reuse as a package. It is organized so that a reader
asking how a particular panel was made reaches the answer in one step.

---

## Figure → code

Pipeline code is under `imaging/`, `scrnaseq/`, `bulkseq/` and `smd/`. Helpers those lanes share —
the CZI reader, the sparse-safe cluster averages, the heatmap ordering — live in
`src/trunk_morph_ref/`; `src/README.md` says what each module is for. `figures/` holds per-panel
scripts that load a canonical object or table and draw.

### Main figures

| Panel | What it shows | Code | Source Data |
|---|---|---|---|
| Fig 1e | pSMAD1/5/9 vs distance from bead | `imaging/psmad_fig1/` | `Fig.1e` |
| Fig 1f | pSMAD vs angle relative to bead | `imaging/psmad_fig1/` | `Fig.1f` |
| Fig 2c | DAPI-normalized FOXF1 / SOX2 / TBXT vs distance from bead | `imaging/foxf1_bead/` | `Fig.2c` |
| Fig 2e | Fraction of bead-adjacent cysts with FOXF1⁺ cells | manual scoring, no code | `Fig.2e` |
| Fig 2g | FOXF1 vs BMP4 reporter pixel density, day 2 cysts | `imaging/foxf1_bmp4_day2_fig2g/` | `Fig.2g` |
| Fig 3c | FOXF1 / PAX8 / MESP2 bilaterality index by bead geometry | `imaging/bilaterality_fig3c/` | `Fig.3c` |
| Fig 3f | Mediolateral marker profiles across the midline | `imaging/morphometry/` | `Fig.3f` |
| Fig 4a | Dot plot, 38 marker genes × 18 clusters | `figures/render_scrnaseq_panels.py` | `Fig.4a` |
| Fig 4b | Trunk morph UMAP, 5,306 cells | `figures/render_scrnaseq_panels.py` | `Fig.4b` |
| Fig 4c | Cluster composition, bead vs no bead | `figures/render_scrnaseq_panels.py` | `Fig.4c` |
| Fig 4d | Morph ↔ embryo expression heatmap, 235 genes | `scrnaseq/morph_embryo_correspondence/` | — |
| Fig 4e | Morph ↔ embryo cluster correlation, 18 × 18 | `scrnaseq/morph_embryo_correspondence/` | `Fig.4e` |
| Fig 4f | LPM morphogen dot plot, 14 genes × 18 clusters | `figures/render_scrnaseq_panels.py` | `Fig.4f` |
| Fig 5e | CFP vs FOXF1 pixel density, 10% BMPR1A knockdown morphs | `imaging/cfp_foxf1_density_fig5e/` | `Fig.5e` |
| Fig 5h | FOXF1 / BMP4 reporter half-max times | `imaging/timelapse_3d_fig5h/` | `Fig.5h` |
| Fig 5i | BMP4-YFP population traces by treatment | `imaging/timelapse_fig5i/` | `Fig.5i` |
| Fig 5n | Host fraction of FOXF1⁺ pixels after LPM graft | `imaging/transplant_fig5n/` | `Fig.5n` |

### Extended Data

| Panel | What it shows | Code | Source Data |
|---|---|---|---|
| ED 1, 5, 11, 12 | Representative micrographs | acquisition and assembly only | — |
| ED 2j, 2k | pSMAD distance and orientation, second dataset | `imaging/psmad_edfig2/` | `ED Fig.2j`, `2k` |
| ED 3b, 3c (right), 3j | Bulk RNA-seq expression | `bulkseq/notebooks/` | `ED Fig.3b`, `3c`, `3j` |
| ED 3d (left) | SOX2 / TBXT vs distance from central cyst | `imaging/foxf1_bead/` | `ED Fig.3d` |
| ED 3d (right) | SOX2 vs TBXT pixel density | `imaging/foxf1_bead/notebooks/09_fixed_plotting.ipynb` | `ED Fig.3d right` |
| ED 3g | FOXA2 and SOX2 vs FOXF1 pixel density, day 2 cysts | `imaging/foxa2_sox2_foxf1_day2_ed3g/scripts/render_ed3g_panel.py` | `ED Fig.3g` |
| ED 3i | Cyst FOXF1⁺ fraction, three conditions | manual scoring, no code | `ED Fig.3i` |
| ED 3k | pSMAD in BMPR1A knockdown | `imaging/bmpr1a_psmad/scripts/08_ed3k_panel.py` | `ED Fig.3k` |
| ED 3o | FOXF1 vs BMP4 pixel density, one day 2 cyst | `imaging/foxf1_bmp4_cyst_ed3o/scripts/render_ed3o_panel.py` | `ED Fig.3o` |
| ED 4b, 4c | Trunk morph length over differentiation, ± IWP3 | see `CODE_AVAILABILITY.md` | `ED Fig.4b`, `4c` |
| ED 4e | Bead distance vs LPM position | manual scoring | `ED Fig.4d` (see note below) |
| ED 6a, 6d | Key-gene and top-5 DEG heatmaps | `scrnaseq/trunk_main/` | — |
| ED 6b | Trunk morph UMAP by condition | `figures/render_scrnaseq_panels.py` | `ED Fig.6b` |
| ED 6c | The 235-gene panel used in 6a and Fig 4d/e | `scrnaseq/trunk_main/derived/integration_gene_panel.csv` | — |
| ED 7 | Hierarchical subclustering of somite, brain, LPM, notochord / floor plate and roof plate / neural crest | `scrnaseq/subclustering/` | — |
| ED 8a, 8b | Human embryo UMAP and composition, 46,055 cells | `scrnaseq/human_embryo/` | `ED Fig.8a`, `8b` |
| ED 8c | Top-10 DEG heatmap, human embryo | `scrnaseq/final_assignments/` | — |
| ED 9a | Morphogen gene expression matrix, trunk morph cells | `scrnaseq/trunk_main/notebooks/02_trunk_main.ipynb` | — |
| ED 9b | BMP receptor expression per cluster, stacked violin | `scrnaseq/trunk_main/notebooks/02_trunk_main.ipynb` | `ED Fig.9b` |
| ED 10d | BMP4 vs FOXF1 reporter pixel density, 48–116.25 h time series | `imaging/timelapse_3d_fig5h/notebooks/05b_reporter_spatial_context.ipynb` | `ED Fig.10d` |
| ED 10f | FOXF1-RFP population traces | `imaging/timelapse_fig5i/` | `ED Fig.10f` |
| ED 10g | Bulk RNA-seq NMP / LPM markers | `bulkseq/notebooks/ed10g_nmp_lpm_markers.ipynb` | `ED Fig.10g` |

The Source Data sheet named `ED Fig.4d` contains the data for panel **4e**; panel 4d is
representative images.

Panels **ED 7a, 7b, 7c and 7f** were drawn from an earlier trunk morph clustering. The notebooks in
`scrnaseq/subclustering/` run on the clustering this repository rebuilds, and give a different
version of those four panels; 7e is unaffected. What differs, and why, is in
`scrnaseq/README.md` under "ED Fig 7 comes from both runs".

Panel **ED 8c** is the same situation on the embryo side: it was drawn from the earlier human embryo
run, of 46,768 cells and 50 fine clusters, which this repository does not rebuild. That object is in
the Zenodo deposit (`data/DOWNLOAD.md`); `08b_integration_and_final_exports.ipynb` reads it from
`SCRNASEQ_INPUT_ROOT` when present, and otherwise draws its version of the heatmap from the run
rebuilt here. See "Which saved object each figure uses" in `scrnaseq/README.md`.

### Supplementary Data

| Object | Code |
|---|---|
| SD 1 — trunk morph annotations | `scrnaseq/final_assignments/` |
| SD 2 — embryo annotations | `scrnaseq/human_embryo/` |
| SD 4, 5, 6 — analysis-display documents | assembled outside this repository; the figure code is under `imaging/`, `scrnaseq/` and `supplementary/sections/` (see `supplementary/README.md`) |

---

## Environment

The analyses ran in the author's conda `base` environment. There was no lockfile, so that
interpreter is the lock; `env/environment.yml` pins it as read from that interpreter, and
`env/requirements-freeze.txt` records the versions.

Four packages the code imports are **not** pinned there, because they are not installed in that
interpreter and a guessed version would be worse than none: `ray` (the SMD step ran on a cluster
under Python 3.10 — see `smd/README.md`) and `cellpose` and `torch` (nuclear segmentation) are left
out of the environment, and `opencv` (the timelapse QC helpers) is installed without a version.
Ilastik 1.4.0, used for pixel classification in the timelapse lanes, is a separate application
rather than a Python package, so it is named here instead of being pinned. The freeze file names
each of these explicitly rather than omitting them silently.

The work predates NumPy 2.x, and scanpy and anndata behave differently across that boundary, so the
pins matter.

```bash
conda env create -f env/environment.yml
conda activate trunk-morph
```

## Data

Raw data and the large AnnData objects are not committed. `data/DOWNLOAD.md` lists the GEO
accessions, the files in the Zenodo deposit, and what this repository ships instead: the derivative
measurement tables the quantitative imaging panels are drawn from, and the SMD outputs the
single-cell notebooks select genes from (`scrnaseq/chain_inputs/`). Two imaging panels are the
exception: ED Fig 3d (right) and ED Fig 10d are drawn by notebooks from pixel intermediates built from
the raw images, which are not shipped (`imaging/foxf1_bead/notebooks/09_fixed_plotting.ipynb` reads a
pooled pixel array, `imaging/timelapse_3d_fig5h/notebooks/05b_reporter_spatial_context.ipynb` a sampled
pixel table).

## Verifying against the paper

The published Source Data workbooks hold the plotted values of the quantified panels, so they
serve as a regression test:

```bash
SOURCE_DATA_DIR=/path/to/source_data python figures/verify_imaging_panels.py
SOURCE_DATA_DIR=/path/to/source_data OBJECT_ROOT=/path/to/analysis python figures/verify_scrnaseq_panels.py
```

Each reports, per panel, the number of published rows it could not match and the largest absolute
difference against a tolerance derived from that sheet's own precision. Fig 1e and ED Fig 2j each
publish a second, bead-proximal block on the same sheet, and each block is checked separately.

Six of the ten single-cell panels can be checked from this repository alone: Fig 4d, Fig 4e and
ED Fig 8b need no saved object, and Fig 4b, Fig 4c and ED Fig 6b read the 4.5 MB SMD object shipped
at `scrnaseq/trunk_main/objects/`. The remaining four read objects too large to commit (GitHub's
limit is 100 MB per file): Fig 4a, Fig 4f and ED Fig 9b read the 121 MB trunk morph object from the
Zenodo deposit, and ED Fig 8a reads the 859 MB human embryo object, which is not deposited and is
rebuilt by `scrnaseq/human_embryo/notebooks/07_human_embryo.ipynb`. Set `OBJECT_ROOT` to a
directory holding them to include those panels; the single-cell notebook chain (below) writes both
objects under its results directory in that layout, so `OBJECT_ROOT` can point there. Without it
they are skipped rather than failed, and the skip line says where to get each one. See
`data/DOWNLOAD.md`.

The bulk RNA-seq panels (ED Fig 3b, 3c, 3j, 10g) have their own runner and comparison script; see
`bulkseq/README.md`.

The Source Data sheets store values rounded to a few decimals and saved as float32, so a published
7.8441 reads back as 7.844099998474121. Tolerances are derived per column from the true decimal
precision plus a float32 allowance, rather than from a fixed epsilon.

`figures/panel_specs.py` records how each panel is drawn: its source table, the subset rule that
selects the plotted rows, and the join keys. The subset rules are part of the figure — omitting one
changes the panel.

## The notebooks

Each stage ships as the notebook that ran, with outputs intact including plots. Much of the analysis
is visual QC — segmentation review, mask checks, bead annotation — so the rendered figures carry a
large part of what a stage means. Printed output is kept because several notebooks report their own
gates and inclusion counts, which is sometimes the only record of a subsetting rule.

The imaging pipeline notebooks were ported with no code cell edited. The porting tool asserts it:
each ported cell must match the original once path strings are normalised away, or the port fails.
One imaging notebook also gained a cell: the last cell of
`imaging/foxf1_bmp4_day2_fig2g/notebooks/03_global_q995_density_review.ipynb` writes the tables Fig 2g
is redrawn from, and that notebook's first cell says so.
The Fig 5e notebook, `imaging/cfp_foxf1_density_fig5e/notebooks/cfp_foxf1_pixel_density.ipynb`, is
trimmed instead: its original was exploratory, so it keeps only the cells on the path to the figure,
reads its images and writes its figure through paths set in the repository, and gains a last cell
that writes the tables the panel is redrawn from. Its first cell lists the kept cells and every change.
Three groups of notebooks were adapted instead, and each lists its changes from the author's original
in its first cell: the single-cell notebooks under `scrnaseq/`, changed so they run from this
repository (where they read and write, the clipboard exports removed, and the `04`–`06` subcluster
labels computed by one Leiden call instead of read from a separate analysis's saved files); the bulk
RNA-seq notebooks under `bulkseq/notebooks/`; and the two GEO conversion notebooks described below.

### The notebooks' own figure references are not reliable

Because their analysis cells and markdown were kept as they ran, section headings inside the
notebooks refer to the figure layout as it stood when that analysis was written, and the layout changed before publication. For
example `02_trunk_main` has headings for a "Fig. 4g" — the published Figure 4 ends at **f** — and
`08b` labels as "Fig. 4e" the heatmap that became panel **4d**.

**Use the figure → code index at the top of this file, not the notebook headings.** The same applies
to saved output filenames: several encode superseded panel letters.

### The single-cell notebooks start from GEO

`scrnaseq/preprocessing/notebooks/01_preprocessing.ipynb` does not read the GEO files directly. Two
conversion notebooks build its inputs from them:
`scrnaseq/preprocessing/notebooks/00a_geo_trunk_morph.ipynb` (GSE306808) and
`scrnaseq/preprocessing/notebooks/00b_geo_human_embryo.ipynb` (GSE155121). One command runs the
single-cell chain in order, from the conversion to the final assignments:

```bash
python scrnaseq/run_chain.py                          # 00a, 00b, 01_preprocessing … 08b, in order
python scrnaseq/run_chain.py --from 01_preprocessing  # resume at a later notebook
python scrnaseq/preprocessing/run_geo_conversion.py   # only 00a and 00b
```

Each notebook runs in its own process without a Jupyter server, and the runner stops at the first
failing cell. `07b`, `07c` and the morph–embryo comparison notebooks `01` and `12` ran on the earlier
human embryo clustering object (see `data/DOWNLOAD.md`); by default the runner skips them and Fig 4d
and 4e are drawn from their saved tables, but when that optional Zenodo object is present under
`SCRNASEQ_INPUT_ROOT` the runner runs them instead (see `scrnaseq/README.md`). The notebooks read the files no notebook creates, such as the SMD z-scores, from
`scrnaseq/chain_inputs/`. Three environment variables set the other locations: `GEO_DATA_DIR` (the GEO
downloads, default `data/`), `SCRNASEQ_INPUT_ROOT` (the conversion outputs and the Zenodo object
`06_rp_nc_subclustering` reads, default `data/scrnaseq_inputs/`) and `SCRNASEQ_RESULTS_ROOT` (everything
the notebooks write, default `scrnaseq/output/`). Which files to download and where each one goes are
in `data/DOWNLOAD.md`; the stages are described in `scrnaseq/README.md`.

## Reproducing a figure

```bash
bash run_all.sh                          # render, then verify; runs with no data download

python figures/render_imaging_panels.py  # 10 imaging panels, needs only this repository
python imaging/bmpr1a_psmad/scripts/08_ed3k_panel.py   # ED Fig 3k, needs only this repository
python imaging/foxf1_bmp4_day2_fig2g/scripts/render_fig2g_panel.py   # Fig 2g, needs only this repository
python imaging/cfp_foxf1_density_fig5e/scripts/render_fig5e_panel.py   # Fig 5e, needs only this repository
python imaging/foxa2_sox2_foxf1_day2_ed3g/scripts/render_ed3g_panel.py   # ED Fig 3g, needs only this repository
python imaging/foxf1_bmp4_cyst_ed3o/scripts/render_ed3o_panel.py   # ED Fig 3o, needs only this repository
OBJECT_ROOT=/path/to/analysis python figures/render_scrnaseq_panels.py   # Fig 4b, 4c, 4e and ED Fig 9b's values; add SOURCE_DATA_DIR for Fig 4a and 4f
python bulkseq/run_bulk_notebooks.py     # ED Fig 3b, 3c, 3j, 10g; needs the GSE347564 CPM table
```

`run_all.sh` does not re-run the pipeline stages or the bulk RNA-seq notebooks. The stages are the
notebooks under `imaging/` and `scrnaseq/`: the imaging stages need the raw images, and the single-cell
stages start from the GEO data and run with `scrnaseq/run_chain.py`; SMD is not re-run. The imaging and single-cell
figure scripts above read the stages' saved outputs instead: the tables and the SMD object
committed here, and the trunk morph object in the Zenodo deposit.

Rendered panels land in `figures/output/`. They are the pre-Illustrator plots: published panels were
composed in Illustrator, including fonts, arrows, callouts, and deletion of non-highlighted gene
labels on the heatmaps. Expect a match in data and visual encoding — values, colour map,
normalisation, axes — not pixel for pixel.

## Citing

See `CITATION.cff`. The archived release on Zenodo has the DOI 10.5281/zenodo.22756591; cite that
rather than a commit hash.
