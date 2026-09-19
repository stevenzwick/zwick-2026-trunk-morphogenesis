# SMD — sparse manifold decomposition

## What this is

Feature selection by **sparse manifold decomposition**, a published method:

> Melton, S. & Ramanathan, S. Discovering a sparse set of pairwise discriminating features in a
> high-dimensional space. *Bioinformatics* **37**, 202–212 (2021).
> doi:[10.1093/bioinformatics/btaa690](https://doi.org/10.1093/bioinformatics/btaa690)

Upstream implementation: <https://github.com/smelton/SMD>

The scripts here are an implementation of that published algorithm, not a new method. They were
adapted for these datasets; parameters were tuned, the algorithm was not changed. Use the upstream
repository if you want the reference implementation.

## Why it matters to read this before judging the gene lists

SMD selects genes that **pairwise discriminate** regions of the manifold. It is deliberately *not*
a highly-variable-gene or PCA-based selection, and the resulting lists look unusual next to an HVG
list — sparse, and including genes with modest overall variance. That is the method working as
designed, not an over-filter. Substituting an all-gene matrix measurably degrades cluster
discrimination on these data.

## SMD did not run in this repository's environment

The feature-selection step was executed on an HPC cluster, not on a workstation, and not in the
`env/environment.yml` environment that covers the rest of the analysis. `runner/run_2.sbatch` is the
job script exactly as submitted:

```
#SBATCH -N 1           # one node
#SBATCH -c 112         # 112 cores
#SBATCH -t 0-12:00     # up to twelve hours
#SBATCH --mem=184G
module load python/3.10.13-fasrc01
srun -c 112 python -u ray_SMD.py
```

So the SMD step used **Python 3.10 and Ray on a 112-core cluster node**, while the downstream
clustering and figure code ran locally under Python 3.12. A reader cannot simply
`conda activate trunk-morph` and reproduce this stage: it needs Ray and a large multi-core machine.

This is why the repository also ships the SMD outputs. The z-score files each single-cell notebook
selects genes from are in `../scrnaseq/chain_inputs/`, and the trunk morph gene scores
(`../scrnaseq/trunk_main/derived/trunk_morph_smd_gene_scores.csv`) and final selection
(`trunk_morph_final_gene_selection.csv`) are in `../scrnaseq/trunk_main/derived/`. The notebooks
start from these files, so the cluster job is not re-run — which is the practical path for almost
any reader.

## Which run produced the published figures

**`trunk_morph__run009`** (15,000 trials, `n_sub` = 4250). The runner and launcher in `runner/` are
that run's copies. `scrnaseq/trunk_main/notebooks/02_trunk_main.ipynb` loads runs 001, 002, 007, 008
and 009, compares them, and selects 009; the gene scores in
`../scrnaseq/trunk_main/derived/trunk_morph_smd_gene_scores.csv` are that run's output. A checksum
comparison found a single distinct `SMD.py` across all fourteen run directories, so the algorithm
code is common to them all and only the inputs and parameters differ.

## Parameters used

Gene sets were retained at a Z-score threshold on the SMD score, then curated. The exact threshold,
the retained counts, and the curated lists are recorded in the stage notebooks under
`../scrnaseq/trunk_main/`, `../scrnaseq/subclustering/` and `../scrnaseq/human_embryo/` — read them
there rather than inferring from the Methods text.

## Attribution and licensing

The optimized runner used for this paper was provided by **William Weiter**, as credited in the
paper's Acknowledgements. The scripts here are an adapted implementation of the published algorithm
(upstream: <https://github.com/smelton/SMD>). They are released under this repository's MIT license
with the agreement of William Weiter and of Sharad Ramanathan, a co-author of both the SMD paper and
this one.
