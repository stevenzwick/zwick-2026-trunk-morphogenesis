# Bulk RNA-seq

Code for **Extended Data Fig. 3b, 3c (right), 3j and 10g**.

## Data

Libraries were sequenced by Plasmidsaurus (3′-end RNA-seq). Counts per million (CPM) were computed by
the provider's pipeline and are used here as delivered: nothing in this directory re-normalizes them.
The CPM table is deposited at GEO under **GSE347564**.

Download `GSE347564_bulk_RNAseq_cpm_all_samples.tsv.gz` from the series' supplementary files and place
it at `data/GSE347564/GSE347564_bulk_RNAseq_cpm_all_samples.tsv.gz`, or point `BULK_CPM_TABLE` at it.

## Panels

| Panel | Notebook | Author | GEO columns |
|---|---|---|---|
| ED 3b | `notebooks/ed3b_bmp4_timecourse.ipynb` | Tianlei He | `pSMAD_0h_rep1–3`, `pSMAD_7h_rep1–3`, `pSMAD_24h_rep1–3` |
| ED 3c (right) | `notebooks/ed3c_endoderm_markers.ipynb` | Tianlei He | `day2diff_rep1–3` (with A83-01), `day2_endoderm_rep1–2` (without) |
| ED 3j | `notebooks/ed3j_bmpr1a_knockdown.ipynb` | Tianlei He | `BMPR1Akd_rep1–3`, `SCRAMBLE_rep1–3` |
| ED 10g | `notebooks/ed10g_nmp_lpm_markers.ipynb` | Yusuf Ilker Yaman | `pSMAD_0h_rep1–3` (pluripotent), `NMP_rep1–3`, `LPM_rep1–3` |

The 0 h libraries of the BMP4 time course serve twice: as the 0 h group in ED 3b and as the pluripotent
reference in ED 10g.

## What the code does

- **ED 3b, 3c:** *BMP4* (3b) and *SOX17*, *GSC*, *CER1*, *FOXA2* (3c) CPM per replicate, drawn as bars
  (mean ± s.d.) with the replicates as points. No statistical test.
- **ED 3j:** *BMPR1A* CPM, drawn the same way, with a two-sided Welch's t-test between knockdown and
  scrambled control. The notebook applies a Holm adjustment over this single comparison, which leaves
  p unchanged.
- **ED 10g:** genes with mean CPM below 2 in all three conditions are removed, the table is restricted to
  transcription factors, and values are transformed to log2(CPM + 1). The heatmap shows 24
  transcription factors as per-gene z-scores. **Those 24 are a named list, not a statistical
  selection.** The notebook also runs a one-way ANOVA and pairwise Welch's t-tests with
  Benjamini–Hochberg correction and exports the result as `marker_gene_statistics_log2cpm.tsv`
  alongside six marker lists, but nothing downstream reads them: the panel would be identical
  without that step.

Each notebook's first cell lists how it differs from the author's original: the input is the GEO table
instead of the provider's per-order files, paths are relative to the repository, and cells that do not
contribute to the published panel were removed.

## Running

```bash
python bulkseq/run_bulk_notebooks.py                                     # runs the four notebooks
SOURCE_DATA_DIR=/path/to/source_data python bulkseq/verify_bulk_panels.py  # compares all four panels with Source Data
SOURCE_DATA_DIR=/path/to/source_data python bulkseq/check_ed3j.py          # ED 3j statistic from the Source Data sheet
```

The runner executes each notebook's code cells in order without a Jupyter server. Outputs go to
`bulkseq/output/`. The notebooks also open normally in Jupyter.

## Third-party data

`derived/protein_class_Transcription.tsv` is the transcription-factor protein class from the
[Human Protein Atlas](https://www.proteinatlas.org), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The file is dated 2024-12-06.
