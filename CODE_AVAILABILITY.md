# Code availability — scope and limits

What this repository covers, what it does not, and who to ask for the rest. Written to match the
paper's Code Availability Statement rather than to claim more than it delivers.

The archived release, with the two AnnData objects too large for GitHub, is on Zenodo:
doi:10.5281/zenodo.22756591. See `data/DOWNLOAD.md`.

## Covered

- **Image analysis** for every quantified imaging panel — segmentation, background handling,
  intensity quantification, and the per-panel aggregation that produces the plotted values.
- **Single-cell RNA-seq** — conversion of the GEO files, preprocessing, gene selection from the SMD
  z-scores, clustering, annotation, and subclustering. The SMD runner scripts are under `smd/`.
- The trunk morph ↔ human embryo correspondence analysis (Fig 4d, 4e).
- **Bulk RNA-seq** (ED Fig 3b, 3c, 3j, 10g) — the summary statistics, tests, filtering and heatmap applied
  to the provider's counts per million. See `bulkseq/README.md`.
- **Seeds and parameters** — intensity thresholds, local-threshold window sizes, Gaussian σ,
  histogram edges, and the curated gene lists are in the scripts, not left to the reader.

## Not covered, and why

Bulk RNA-seq read processing (ED Fig 3b, 3c, 3j, 10g). Alignment, counting and the computation of
counts per million were done by the sequencing provider's pipeline; the repository starts from that CPM
table (GEO GSE347564).

Trunk morph length over differentiation (ED Fig 4b, 4c). A co-author's experiment and analysis,
whose code is not in this repository. The ED Fig 4c legend states the test (two-sided Mann–Whitney
U with Bonferroni correction), and the lengths it was applied to are in the ED Fig 4c Source Data
sheet.

Manually scored panels (Fig 2e, ED Fig 3i, ED Fig 4e). These are counts and categorical calls
made by eye from images — the number of bead-adjacent cysts containing FOXF1⁺ cells, and whether a
morph's LPM is bilateral or posterior. There is no script because there was no computation; the
scoring criteria are stated in the figure legends and Methods, and the resulting counts are in the
published Source Data.

Three ED Fig 7 panels as printed (7a, 7b, 7f). The subclustering code is here and runs, but on the
trunk morph clustering this repository rebuilds. Those three panels were drawn from an earlier
clustering of the same data, and its subcluster label tables are not deposited, so rerunning gives
the current run's version of that analysis rather than the printed panel. `scrnaseq/README.md`, under
"ED Fig 7 comes from both runs", says what differs. The other ED Fig 7 panels are unaffected.

Representative micrographs (ED Fig 1, 5, 11, 12 and image panels throughout). Acquisition and
figure assembly only.

**Supplementary Data 4, 5 and 6 document assembly.** The SD documents were laid out as HTML and
printed to PDF through headless Chrome, with a separate caption-typesetting step. That assembler is
**deliberately not published**: it requires a browser, it consumes roughly 157 MB of pre-rendered
panels that are not in this repository, and its output — the assembled PDFs — ships with the paper
as Supplementary Data, so a reader already has it. The **figure** code behind those documents is
here, under `imaging/`, `scrnaseq/` and `supplementary/sections/`. Nothing in this repository
requires Chrome.

**Illustrator assembly.** Published panels were composed in Illustrator — fonts, arrows, callouts,
and the deletion of non-highlighted gene labels on heatmaps. Scripts here emit the pre-assembly
plot. Expect matching data and visual encoding, not a pixel-identical panel.

## Third-party code

Sparse manifold decomposition (SMD) is published: Melton, S. & Ramanathan, S. *Bioinformatics*
**37**, 202–212 (2021), doi:10.1093/bioinformatics/btaa690, with an upstream implementation at
<https://github.com/smelton/SMD>. The runner scripts under `smd/` are a lightly adapted
implementation of that published algorithm with parameters tuned for these datasets; they are not a
new method. The upstream repository carries no license file; the adaptation here is released under
this repository's MIT license with the agreement of the SMD authors. See `smd/README.md`.

## Reproducibility notes

- Random seeds are set explicitly, including for Leiden clustering and UMAP.
- Environment pinning is load-bearing — see `env/environment.yml`. The analyses predate NumPy
  2.x.
- Where a figure depends on a specific saved intermediate, the script names that file explicitly
  rather than globbing, because more than one version of some intermediates exists.
