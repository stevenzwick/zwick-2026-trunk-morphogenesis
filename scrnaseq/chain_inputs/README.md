# scRNA-seq chain inputs

Files that the single-cell notebooks read but that no notebook in this repository creates. Every
other file the notebooks read is written by an earlier notebook, or by the GEO conversion notebooks
in `../preprocessing/`.

Paths under this directory mirror the analysis tree the notebooks ran in. For example,
`trunk_main_dev/results/smd_runs/trunk_morph__run009/z_trunk_morph__smd_input_009.npy` here was
`<analysis-root>/trunk_main_dev/results/smd_runs/trunk_morph__run009/z_trunk_morph__smd_input_009.npy`
there. `PROVENANCE.tsv` records each file's original path, size in bytes and sha256.

The notebooks read these files from here: each notebook's `PROJECT_ROOT`, which was the analysis root,
is this directory. Nothing writes to it; the notebooks write under `SCRNASEQ_RESULTS_ROOT` (see
`../run_chain.py`).

## Why the SMD z-scores ship

SMD is not re-run. Its z-score outputs are shipped as inputs, and the notebooks select genes from
them directly. The runner that produced them is in `../../smd/`. The `ray_SMD.py` files here are
copies of that runner saved with individual runs, each holding its run's parameters. Notebooks 04,
05 and 06 read them as text to report those parameters; they are not executed.

## Contents

| Path | Read by | What it is |
|---|---|---|
| `trunk_main_dev/results/smd_runs/trunk_morph__run001`, `run002`, `run007`, `run008`, `run009` | `02_trunk_main` | Trunk morph SMD z-scores. Run 009 selects the clustering genes; the other runs are compared with it. |
| `trunk_main_dev/results/smd_runs/trunk_morph_lpm_run001`, `run002` | `03_lpm_subclustering` | LPM / IM / endothelial SMD z-scores. Run 001 selects genes; run 002 is compared with it. |
| `trunk_main_dev/results/smd_runs/trunk_morph_somite__run001`, `run002` | `04_somite_subclustering` | Somite SMD z-scores and each run's `ray_SMD.py`. Run 002 is used; run 001 is compared with it. |
| `trunk_main_dev/results/smd_runs/trunk_morph_fbmb__run001` | `05_fbmb_subclustering` | Forebrain / midbrain SMD z-scores and the run's `ray_SMD.py`. |
| `trunk_main_dev/results/smd_runs/trunk_morph_rpnc__run001` | `06_rp_nc_subclustering` | Combined roof plate / neural crest SMD z-scores and the run's `ray_SMD.py`. |
| `trunk_main_dev/results/smd_runs/human_embryo__run001`, `run002` | `07_human_embryo` | Human embryo SMD z-scores. Run 001 selects genes; run 002 is compared with it. |
| `z_morph_2.12_000.npy` | `02_trunk_main` | Earlier trunk morph SMD z-scores, compared with the current gene selection. |
| `z_morph_nc4.03_000.npy`, `z_morph_rp4.03_000.npy` | `06_rp_nc_subclustering` | Earlier neural crest and roof plate SMD z-scores, mapped onto genes for the separate neural crest and roof plate subclusterings. |
| `legacy/results/intermediates/07_human_embryo/adata_embryo_SMD.h5ad` | `07_human_embryo` | Earlier human embryo SMD object. Only its gene list is used, for comparison with the run 001 selection. |
| `parity/results/intermediates/07b_human_embryo_floor_plate_subclustering/z_embryo_ivsc_07b_001.npy` | `07b_human_embryo_floor_plate_subclustering` | SMD z-scores for the floor plate subset of the earlier human embryo clustering object (see `../README.md`). |
| `parity/results/intermediates/07c_human_embryo_nmp_subclustering/z_embryo_nmp_07c_001.npy` | `07c_human_embryo_nmp_subclustering` | SMD z-scores for the NMP subset of the earlier human embryo clustering object (see `../README.md`). |

## Not in this directory

- `legacy/results/intermediates/02_trunk_main/adata_morph_with_clusters.h5ad` (about 101 MB).
  `06_rp_nc_subclustering` reads its earlier trunk morph cluster labels to map
  `z_morph_nc4.03_000.npy` and `z_morph_rp4.03_000.npy` onto genes. It is deposited on Zenodo; put it
  at that same path under `SCRNASEQ_INPUT_ROOT` (default `data/scrnaseq_inputs/`).
- `sample_284.adata`, `GSE155121_human_data_w3-4_genematch.h5ad`, `embryo_only_genes.pkl` and
  `morph_only_genes.pkl`, which the GEO conversion notebooks in `../preprocessing/` build from public
  GEO files and write to `SCRNASEQ_INPUT_ROOT`.
