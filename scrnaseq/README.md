# Single-cell RNA-seq

Stages run in order; each writes intermediates the next one reads.

| Stage | Produces |
|---|---|
| `preprocessing/` — `00a`, `00b` | GEO conversion: `sample_284.adata` from GSE306808, and the week 3–4 human embryo object from GSE155121 with its gene names matched to the trunk morph data. Run both with `python scrnaseq/preprocessing/run_geo_conversion.py`; which GEO files to download is in `../data/DOWNLOAD.md` |
| `preprocessing/` — `01` | Filtered trunk morph and human embryo matrices |
| `trunk_main/` | Gene selection from the SMD z-scores, Leiden clustering, annotation — 5,306 cells × 199 genes × 18 clusters |
| `subclustering/` | Somite, forebrain/midbrain, LPM/IM, roof plate / neural crest, notochord / floor plate subclusters (ED Fig 7) |
| `human_embryo/` | Embryo clustering — 46,055 cells, 48 clusters (ED Fig 8), plus the floor plate and NMP subclustering used by Fig 4d/4e |
| `final_assignments/` | Published per-cell annotations (Supplementary Data 1, 2) and the integration exports |
| `morph_embryo_correspondence/` | Merged 18-label embryo taxonomy, correlation matrix, integration heatmap (Fig 4d, 4e, drawn from the saved tables in `derived/` unless the earlier human embryo object is present — see "Which saved object each figure uses") |

Run them in that order with one command:

```bash
python scrnaseq/run_chain.py                          # the notebooks above, in order (skips 07b, 07c, 01, 12 unless the earlier embryo object is present)
python scrnaseq/run_chain.py --from 07_human_embryo    # resume at a later notebook
python scrnaseq/run_chain.py --list                   # the order
```

Each notebook runs in its own process without a Jupyter server, the way "Run All" would; the runner
stops at the first failing cell, names it, and logs each notebook's run time and peak memory to
`run_chain_log.tsv` in the results directory.

Where the notebooks read and write:

| Location | What | Default |
|---|---|---|
| `scrnaseq/chain_inputs/` | files the notebooks read that no notebook writes (below) | shipped |
| `GEO_DATA_DIR` | the GEO downloads `00a` and `00b` read | `data/` |
| `SCRNASEQ_INPUT_ROOT` | what `00a` and `00b` write, the Zenodo object `06` reads, and the earlier human embryo Zenodo object `07b`, `07c`, `12` and `01` read | `data/scrnaseq_inputs/` |
| `SCRNASEQ_RESULTS_ROOT` | everything the notebooks write | `scrnaseq/output/` |

Under `SCRNASEQ_RESULTS_ROOT` the outputs keep the layout of the analysis directory the notebooks
ran in: stage outputs in `trunk_main_dev/results/intermediates/<stage>/`, and figures in
`trunk_main_dev/results/manuscript_figures/` and `extended_data_figures/`. `07_human_embryo`
also writes a second copy of the embryo object, `w3-4_humanembryo.h5ad`, to `trunk_main_dev/`; nothing
reads it. The figure scripts read the same layout, so `OBJECT_ROOT` can be set to
`SCRNASEQ_RESULTS_ROOT`.

Each notebook's first cell lists the changes made to it for this repository.

## Inputs that no notebook creates

SMD is not re-run. The notebooks select genes from the z-score files SMD saved, and those files ship
in `chain_inputs/` together with the other files the notebooks read but no notebook in this
repository writes. `chain_inputs/README.md` lists each file and the notebook that reads it. The trunk
morph clustering genes come from run `trunk_morph__run009`, which ran on one 112-core cluster node;
the runner is in `../smd/`.

One such input is too large to commit: the earlier trunk morph object that `06_rp_nc_subclustering`
reads for its cluster labels is in the Zenodo deposit. Put it at
`legacy/results/intermediates/02_trunk_main/adata_morph_with_clusters.h5ad` under
`SCRNASEQ_INPUT_ROOT`. See `../data/DOWNLOAD.md`.

## A note on the notebooks' figure references

Section headings inside these notebooks name panels as they were numbered while the analysis was
being done, and Figure 4 was renumbered before publication. Headings referring to "Fig. 4g" describe
panels that do not exist in the published figure, and `08b`'s "Fig. 4e" heading describes what
became panel 4d. The headings were left as they were; the mapping that is current is the figure →
code index in the top-level README.

## A note on the notebook filenames

The directories are named for what they do. The notebooks inside keep the numbers they carried in
the original working pipeline, so `human_embryo/` contains `07_human_embryo.ipynb` and
`morph_embryo_correspondence/` contains a notebook numbered `12`. Those numbers are provenance, not
a position in this repository: the correspondence stage came from a separate alignment study with
its own numbering, which is also why a notebook numbered `01` sits beside it.

Directory numbering was dropped rather than renumbered, since renumbering would have put, say,
`07_human_embryo.ipynb` inside a directory called `04_` and traded one mismatch for another.

## Which saved object each figure uses

The analyses were run more than once, and the saved intermediates from those runs are not
interchangeable: they differ in cell count, gene count and cluster count. The published figures do
not all come from the same run.

| Figures | Object | Shape |
|---|---|---|
| Fig 4a, 4b, 4c, 4f · ED 6a–d, 9b | trunk morph, later run | 5,306 cells × 199 genes × 18 clusters |
| ED Fig 8a, 8b | human embryo, later run | 46,055 cells, 48 clusters |
| Fig 4d, 4e · ED Fig 8c | human embryo, earlier run, plus floor plate and NMP subclustering | 46,768 cells, 50 fine clusters |
| ED Fig 7e | trunk morph, later run, subclustered | the 5,306-cell object above |
| ED Fig 7a, 7b, 7c, 7f | trunk morph, earlier run, subclustered | 5,350 cells |

The earlier embryo run applies no mitochondrial-content filter, so it retains roughly 1,100 cells
that the later one excludes; Fig 4d and 4e use it because they require the embryo floor plate and
NMP labels, which only that path produces. `07b`, `07c` and the comparison notebooks `01` and `12` ran
on that earlier object, and the SMD z-scores shipped for `07b` and `07c` come from it, so none of the
four run on the object `07_human_embryo` builds here.

The earlier embryo object is in the Zenodo deposit (`../data/DOWNLOAD.md`): `08b_integration_and_final_exports`
reads it for ED Fig 8c, and `07b`, `07c`, `12` and `01` read it for Fig 4d and 4e, all checking for it
under `SCRNASEQ_INPUT_ROOT` (`adata_embryo_raw`, which `08b` also reads, is shared between runs and
needs no override). **Absent — the default, e.g. the release test — `run_chain.py` skips `07b`, `07c`,
`12` and `01`, and Fig 4d and 4e are drawn from their saved tables in
`morph_embryo_correspondence/derived/`; `08b` falls back to the run `07_human_embryo` rebuilds here
for its version of the ED Fig 8c heatmap. Present, `run_chain.py` runs all four instead**, and `12`
additionally needs that directory's SMD companion, `adata_embryo_SMD.h5ad`, which is in the deposit
as `adata_embryo_SMD_earlier_run.h5ad` (see `../data/DOWNLOAD.md`), for a side table only; the corr_df,
membership table and heatmap that make Fig 4d and 4e do not depend on it.

Each stage writes a provenance record naming the exact input files it read — see
`morph_embryo_correspondence/derived/stage_provenance.json` for a worked example. Those paths refer
to the authors' working tree and appear as `<analysis-root>/…`; they identify *which* saved object
was used, which is the part that matters when more than one exists.

### ED Fig 7 comes from both runs

The printed ED Fig 7 draws on both trunk morph runs. Only its notochord / floor plate panel (7e)
comes from the later object. Its somite, forebrain / midbrain, LPM / IM and roof plate / neural crest
panels (7a, 7b, 7c, 7f) come from the earlier one, subclustered as the earlier analysis did it, and
the cell numbers and subpopulation counts in their legends are that run's.

The notebooks in `subclustering/` run on the later object. Rerunning them reproduces 7e; for 7a, 7b,
7c and 7f they produce the later run's version of the same analysis, which differs from what is
printed. The somite subclustering starts from 971 cells rather than 954 — the same six subcluster
names, different membership, so the composition bars shift. Roof plate and neural crest are
subclustered together into 4 groups, where the earlier analysis subclustered them separately, into 3
and 5, and the printed panel shows all 8. The LPM / IM subclustering resolves 5 subpopulations, where
the printed panel shows the earlier run's 3: it splits off intermediate mesoderm, which the printed
legend treats as input rather than as one of the three, and a neuroectodermal contamination group.
Gene selection runs on the cells given to it, so the heatmap gene sets and row orders differ as well.

The earlier trunk morph object is in the Zenodo deposit, because `06_rp_nc_subclustering` reads its
labels (`../data/DOWNLOAD.md`). Its subcluster label tables are not in this repository, so the
printed 7a, 7b and 7f are not redrawn by anything that ships here.

## The 18-label embryo taxonomy (Fig 4d, 4e)

The 50 fine-grained embryo clusters are reduced to 18 labels matching the trunk morph annotation:
**17 clusters are dropped** and the remaining 33 merged. The dropped set — definitive endoderm,
cardiomyocytes, erythroid and haematopoietic progenitors, head mesenchyme, skeletal myocytes,
cranial placodal ectoderm — are lineages the trunk morph model does not contain.

They are removed rather than displayed as uncorresponded. Correlation alone is blind to germ
layer, so leaving them in would let an unrelated lineage appear as a match. The membership table,
the drop list and a validation record ship in `morph_embryo_correspondence/derived/`.
