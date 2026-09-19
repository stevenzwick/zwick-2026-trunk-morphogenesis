# Data: what ships here, what to fetch, and how the two relate

Raw data and the large AnnData objects are not committed to this repository. What it does carry is
a **derivative layer**: the tables under each lane's `derived/` directory (62 files in the imaging
and single-cell lanes, about 29 MB), from which all but two of the quantified imaging panels are
drawn **without the raw microscopy data**. This file explains what those tables are, what each was
derived from, and what you need to fetch separately.

## The short version

| | |
|---|---|
| **Ships here** | per-image, per-cyst, per-bin and per-cluster measurement tables (`*/derived/`); the 4.5 MB SMD trunk morph object; the SMD z-score files the single-cell notebooks read (`scrnaseq/chain_inputs/`) |
| **Fetch from GEO** | the sequencing data (`GSE306808`, `GSE155121`, `GSE347564`) |
| **Fetch from Zenodo** | two trunk morph AnnData objects too large for GitHub (doi:10.5281/zenodo.22756591) |
| **Rebuilt, not downloaded** | the human embryo object, built from `GSE155121` by the single-cell notebooks |
| **On request** | raw microscopy images — multi-terabyte confocal and CZI timelapse |

Every quantified imaging panel except ED Fig 3d (right) and ED Fig 10d is drawn from the tables in
this repository. Those two are drawn by notebooks from pixel intermediates built from the raw images,
which are not shipped: `imaging/foxf1_bead/notebooks/09_fixed_plotting.ipynb` reads a pooled pixel
array and `imaging/timelapse_3d_fig5h/notebooks/05b_reporter_spatial_context.ipynb` a sampled pixel
table. Nothing in `figures/render_imaging_panels.py` or `figures/verify_imaging_panels.py` opens an
image.

## Why the derivative layer, and not the images

The raw data are multi-terabyte multi-channel confocal stacks and CZI timelapses. Depositing them in
full is impractical, and a reader does not need them: the expensive, judgement-laden steps — mask
generation, background handling, manual bead and well annotation, axis calls — are exactly the steps
that produce these tables. Shipping the tables means the published numbers can be checked and the
panels redrawn on a laptop, in seconds, offline.

Raw imaging is available from the lead contact on request, as the paper's Data Availability
Statement states.

## How a derived table is made

Every imaging lane follows the same shape, and each stage ships as the notebook that ran it:

```
raw .czi  →  conversion + manifest  →  ROI / annotation  →  quantification  →  derived table
              (00_, 01_)               (02_)                (03_)              (*/derived/)
```

The per-lane READMEs give the specifics. Two general points:

- **Manual calls are data, not code.** Bead positions, well outlines and axis endpoints were placed
  by eye and saved as JSON/TSV annotations, which the quantification notebooks read. Where small
  enough, those annotation files ship too, so a manual step is reproducible rather than merely
  described.
- **Each `derived/` directory carries a `PROVENANCE.tsv`** listing, for every shipped table, its
  path in the analysis tree it came from, its size, and its **SHA-256**. That is the audit trail
  from a file here back to the file that produced it.

## What each table is, and which panel it feeds

### pSMAD spatial quantification — `imaging/psmad_fig1/`, `imaging/psmad_edfig2/`

Shell pixels around each cyst, binned two ways relative to the bead. `manual_gradient_per_cyst.csv`
(210 / 223 cysts) is one row per cyst with its bead distance and response summaries;
`…_distance_trace_long.csv` (4,717 / 4,726 rows) bins those pixels by distance in 10 µm steps;
`…_orientation_trace_long.csv` (14,256 / 14,472 rows) bins the full shell by bead-relative angle in
5° steps. → **Fig 1e, 1f; ED Fig 2j, 2k.**

### Day-2 immunostain profiles — `imaging/foxf1_bead/`

DAPI-normalised marker intensity against distance, averaged across images with equal bin support.
157 rows for the bead panel, 87 for the central-cyst panel. → **Fig 2c; ED Fig 3d.**

### Day-2 reporter pixel density — `imaging/foxf1_bmp4_day2_fig2g/`

Every in-cyst pixel of 14 day-2 image stacks, pooled and binned as the panel draws them: the pixel
counts in its 96 × 96 bins (9,216 rows), the bin edges, and a summary with n, the pooled pixel count,
the two positivity cutoffs, the pixel count in each quadrant and the sums from which the Pearson R² is
computed. `scripts/render_fig2g_panel.py` in that lane draws the panel from them. → **Fig 2g.**

### Bilaterality — `imaging/bilaterality_fig3c/`

Two per-experiment day-5 feature tables (39 + 28 morphs) and the 127 plotted bilaterality-index
values recovered from the figure notebook. → **Fig 3c.**

### Mediolateral morphology — `imaging/morphometry/`

Positive-area fraction against distance from the midline: 4,824 per-file rows and the 72-row
across-morph summary. → **Fig 3f.**

### Reporter timelapses — `imaging/timelapse_3d_fig5h/`, `imaging/timelapse_fig5i/`

Half-max times per morph per reporter (1,170 rows), and condition-level population means with s.e.m.
over time (2,100 rows). → **Fig 5h; Fig 5i; ED Fig 10f.**

### Knockdown morph pixel density — `imaging/cfp_foxf1_density_fig5e/`

Every pixel above the DAPI threshold in three day-5 morph image stacks with 10% BMPR1A knockdown cells,
pooled and binned as the panel draws them: the pixel counts in the non-empty bins of its 256 × 256
histogram, the bin edges, and a summary with n, the pooled pixel count, the DAPI threshold, the two
gates and the pixel count in each quadrant. `scripts/render_fig5e_panel.py` in that lane draws the panel
from them. → **Fig 5e.**

### One-cyst FOXF1 versus BMP4 — `imaging/foxf1_bmp4_cyst_ed3o/`

Every pixel of one z plane of one day 2 cyst that passes the DAPI threshold and the per-channel
background and saturation bounds, with both channels rescaled to 0–1: 94,476 rows, plus a summary of
the constants used and the values the notebook printed. `scripts/render_ed3o_panel.py` in that lane
draws the panel and recomputes its R² from them. → **ED Fig 3o.**

### FOXA2 and SOX2 versus FOXF1, day 2 cysts — `imaging/foxa2_sox2_foxf1_day2_ed3g/`

Every in-region pixel of seven day 2 cysts from three independent differentiations, pooled and
rescaled as both panels draw them: one table holding the FOXA2 and SOX2 coordinates alongside
their paired FOXF1 coordinate (504,789 rows), plus a summary with the per-cyst offsets, the
positivity gates and scales, and the quadrant proportions the notebook printed for both panels.
`scripts/render_ed3g_panel.py` in that lane draws both panels and recomputes their quadrant
statistics from them. → **ED Fig 3g.**

### Transplants and co-grafts — `imaging/transplant_fig5n/`, `imaging/cograft_fig5/`

FOXF1⁺ pixel identity summarised at four nested levels in one file (87 rows; the panel uses the
per-file level), and per-organoid / per-z co-graft metrics. → **Fig 5n.**

### BMPR1A validation — `imaging/bmpr1a_psmad/`

Per-well timecourse (74 rows), a per-field edge summary (9 rows), and
`negctrl_summary_per_field_30min.csv` (27 rows) — the per-well medians behind **ED Fig 3k**, which
plots twelve of those rows.

⚠️ `per_cell_dedge_QC.csv` is a **single-field** QC export. The full per-cell table that scripts
`03`-`07` read is not shipped, so the **nine-box Supplementary Data 4 figure** those scripts draw
cannot be rebuilt here. ED 3k is a different, four-box panel; its data ships and
`08_ed3k_panel.py` redraws it from `derived/` alone, with no raw images and nothing to configure.

### Single-cell — `scrnaseq/trunk_main/`, `scrnaseq/morph_embryo_correspondence/`

The gene sets and orderings the figures depend on — SMD gene scores (8,098), the final gene
selection (8,103), the 235-gene integration panel, the 44 labelled genes, the 18-cluster display
order — plus the correspondence outputs: the 18 × 18 cluster correlation matrix, merged-embryo
membership and group summaries, and the 10,612-cell heatmap ordering.

The single-cell panels also read saved AnnData objects. **One of them ships here** —
`scrnaseq/trunk_main/objects/adata_morph_SMD_.h5ad` (4.5 MB: 5,306 cells × 199 genes, carrying
`leiden_morph`, the condition labels and the UMAP). It is what **Fig 4b, Fig 4c and ED Fig 6b**
need, so those can be drawn from a clone with nothing configured. Together with the three panels
that need no object at all (Fig 4d, 4e, ED 8b), **six of the ten single-cell panels can be checked
from this repository alone.**

The remaining four read objects too large to commit — GitHub's per-file limit is 100 MB. The figure
scripts look for each one under `OBJECT_ROOT`, at the path shown:

| object | size | panels | path under `OBJECT_ROOT` | where to get it |
|---|---|---|---|---|
| `adata_morph_with_clusters.h5ad` | 121 MB | Fig 4a, 4f, ED 9b | `trunk_main_dev/results/intermediates/02_trunk_main/` | the Zenodo deposit (below) |
| `adata_embryo_with_clusters.h5ad` | 859 MB | ED Fig 8a | `trunk_main_dev/results/intermediates/07_human_embryo/` | not downloaded: `scrnaseq/human_embryo/notebooks/07_human_embryo.ipynb` rebuilds it from **GSE155121** (Zeng et al.). It is a clustering of that public dataset, so it is not deposited; its cluster labels, barcodes and UMAP coordinates are in the ED Fig 8a Source Data sheet |

Both objects are also written by the single-cell notebook chain (`scrnaseq/run_chain.py`), at these
paths under its results directory `SCRNASEQ_RESULTS_ROOT` (default `scrnaseq/output/`), so
`OBJECT_ROOT` can point there. Without `OBJECT_ROOT`, the verifiers report which objects are missing
and skip those panels rather than failing.

### SMD outputs — `scrnaseq/chain_inputs/`

Rebuilding the single-cell objects from `GSE306808` does not re-run SMD, which ran on a cluster node
(`smd/README.md`). The notebooks select genes from the z-score files SMD saved, and those files are
committed under `scrnaseq/chain_inputs/`, with the other inputs the notebooks read but no notebook
creates. Its `README.md` lists each file and the notebook that reads it, and its `PROVENANCE.tsv`
gives each file's original path, size and SHA-256.

## Zenodo deposit

The archived release is **doi:10.5281/zenodo.22756591**. Besides a copy of this repository it holds
three AnnData objects that exceed GitHub's per-file limit: two trunk morph objects, both named
`adata_morph_with_clusters.h5ad` in the analysis tree the notebooks ran in, so tell them apart by
size, and one human embryo object:

| object | size | read by |
|---|---|---|
| `trunk_main_dev/results/intermediates/02_trunk_main/adata_morph_with_clusters.h5ad` | 127,256,198 bytes (121 MiB) | Fig 4a, 4f and ED Fig 9b, through `figures/render_scrnaseq_panels.py` and `figures/verify_scrnaseq_panels.py` |
| `legacy/results/intermediates/02_trunk_main/adata_morph_with_clusters.h5ad` | 101,137,514 bytes (96 MiB) | `scrnaseq/subclustering/notebooks/06_rp_nc_subclustering.ipynb`, which uses its earlier cluster labels to map the earlier neural crest and roof plate SMD z-scores onto genes |
| `adata_embryo_with_clusters_earlier_run.h5ad` (Zenodo key; 913,789,090 bytes / 871 MiB; md5 `9a4b2926748971d5375f5a67a04084bd`) | — | `scrnaseq/final_assignments/notebooks/08b_integration_and_final_exports.ipynb` for **ED Fig 8c**, and `scrnaseq/human_embryo/notebooks/07b_human_embryo_floor_plate_subclustering.ipynb`, `07c_human_embryo_nmp_subclustering.ipynb` and `scrnaseq/morph_embryo_correspondence/notebooks/12_merged_embryo_cluster_cluster_correlation.ipynb` / `01_cluster_cluster_correlations.ipynb` for **Fig 4d, 4e** — all read the earlier human embryo run (46,768 cells, 50 `leiden_embryo` clusters) |

Where to put them: the first under `OBJECT_ROOT` at the path shown, for the figure scripts; the
second and third under `SCRNASEQ_INPUT_ROOT` (default `data/scrnaseq_inputs/`). The second goes at
the path shown, where `06_rp_nc_subclustering` reads it. The third — downloaded under its Zenodo
key name — goes at
`earlier_embryo_run/results/intermediates/07_human_embryo/adata_embryo_with_clusters.h5ad` (renamed
back to `adata_embryo_with_clusters.h5ad`, in a neutral `earlier_embryo_run/` directory rather than
naming it after any one working copy in the analysis tree). `adata_embryo_raw`, which `08b` also
reads, is shared between runs and needs no override.

Stage 12 (and its sibling `01`) also read that same directory's SMD companion,
`earlier_embryo_run/results/intermediates/07_human_embryo/adata_embryo_SMD.h5ad`, for the
`gene_set_summary.csv` side table (`smd_intersection`, `smd_union_shared`) — not for the corr_df,
membership table or heatmap that make Fig 4d and 4e, which depend only on the full with-clusters
object. It is in the deposit as `adata_embryo_SMD_earlier_run.h5ad` (23,095,256 bytes, md5
`6e2251c1af182211908dbfbb717fa9f1`; 47,256 cells x 233 genes). Without it, `07b` and `07c` still run,
since neither reads it, while `12` and `01` raise `FileNotFoundError` on it specifically. Put it at the
sibling path above if you have it.

The later-run human embryo object (ED Fig 8a, 8b; 859 MB) is not in the deposit — see the table
above; it is rebuilt from `GSE155121`.

## Sequencing data

| Dataset | Accession | Notes |
|---|---|---|
| Trunk morph scRNA-seq (this paper) | **GSE306808** | Public. Sample GSM9209686, two files: `GSM9209686_filtered_feature_bc_matrix.h5` (counts) and `GSM9209686_barcode_data.csv.gz` (guide calls) → `data/GSE306808/`. These are the two files the conversion below reads. |
| Human embryo scRNA-seq (Zeng et al.) | **GSE155121** | Public third-party dataset. Three libraries are used: W3-1, W4-1, W4-2. `GSE155121_human_data_raw.h5ad.gz` → `data/GSE155121/`, then decompress it with `gunzip` (about 10 GB). The human embryo object is rebuilt from this file, not downloaded. |
| Bulk RNA-seq (this paper) | **GSE347564** | Public. `GSE347564_bulk_RNAseq_cpm_all_samples.tsv.gz`, the counts-per-million table → `data/GSE347564/`. Used by `bulkseq/`; see `bulkseq/README.md`. |

### From the GEO files to the first single-cell notebook

`scrnaseq/preprocessing/notebooks/01_preprocessing.ipynb` does not read the GEO files directly. Two
conversion notebooks in the same folder build the four files it reads. Run them first, in this order:

1. `00a_geo_trunk_morph.ipynb`: GSE306808 → `sample_284.adata`
2. `00b_geo_human_embryo.ipynb`: GSE155121, plus the gene IDs in `sample_284.adata` →
   `human_embryo/GSE155121/GSE155121_human_data_w3-4_genematch.h5ad`, `embryo_only_genes.pkl`,
   `morph_only_genes.pkl`
3. `01_preprocessing.ipynb`, then the later stages in the order given in `scrnaseq/README.md`

```bash
python scrnaseq/run_chain.py                           # 00a, 00b, then the later stages (scrnaseq/README.md)
python scrnaseq/preprocessing/run_geo_conversion.py    # only 00a and 00b
```

The conversion reads the GEO files from `data/`, or from `GEO_DATA_DIR` if set. It writes its outputs,
about 3.1 GB, to `SCRNASEQ_INPUT_ROOT` (default `data/scrnaseq_inputs/`), and `01_preprocessing` reads
them from there. `06_rp_nc_subclustering` also reads the earlier trunk morph object from the Zenodo
deposit under `SCRNASEQ_INPUT_ROOT` (see "Zenodo deposit" above); `run_chain.py` checks for it before
starting. `08b_integration_and_final_exports` likewise checks `SCRNASEQ_INPUT_ROOT` for the earlier
human embryo object, for ED Fig 8c, falling back to the run `07_human_embryo` rebuilds when it is
absent. `07b_human_embryo_floor_plate_subclustering`, `07c_human_embryo_nmp_subclustering`,
`12_merged_embryo_cluster_cluster_correlation` and `01_cluster_cluster_correlations` read the same
earlier human embryo object for Fig 4d and 4e; `run_chain.py` skips these four by default (an
optional download) and runs them instead when the object is present — see `scrnaseq/README.md`'s
"Which saved object each figure uses". Everything the notebooks write goes to `SCRNASEQ_RESULTS_ROOT`
(default `scrnaseq/output/`).

## Where outputs land

`figures/render_imaging_panels.py`, `render_scrnaseq_panels.py`,
`imaging/bmpr1a_psmad/scripts/08_ed3k_panel.py`,
`imaging/foxf1_bmp4_day2_fig2g/scripts/render_fig2g_panel.py` and
`imaging/cfp_foxf1_density_fig5e/scripts/render_fig5e_panel.py` write to `figures/output/`, which is
gitignored. The
bulk RNA-seq runner writes to `bulkseq/output/`, also gitignored. The GEO conversion writes to
`SCRNASEQ_INPUT_ROOT` (default `data/scrnaseq_inputs/`), and the single-cell notebooks write to
`SCRNASEQ_RESULTS_ROOT` (default `scrnaseq/output/`, gitignored), in the layout described in
`scrnaseq/README.md`; that includes `07_human_embryo`'s extra copy of the embryo object,
`trunk_main_dev/w3-4_humanembryo.h5ad`, which nothing reads. Nothing writes to
`scrnaseq/chain_inputs/`. Individual imaging pipeline scripts write beside their own lane, under
`results/`, `derived/` or `output/`; several of them need raw images and will not run here at all. Apart from
the GEO conversion outputs, nothing writes to `data/`.
