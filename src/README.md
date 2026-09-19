# `src/trunk_morph_ref/` — the shared helpers

Ten modules that the pipeline scripts and notebooks import. Nothing here is specific to one panel:
these are the routines the lanes have in common, kept in one place so that the same clustering
average, the same heatmap ordering and the same CZI reader are used everywhere rather than
re-implemented per lane.

A reader following a panel back to its code will meet these as `from src.trunk_morph_ref import …`.
This file says what each one is for.

| Module | What it does | Read it if you are following |
|---|---|---|
| `czi_compat.py` | A CZI reader presenting `czifile`'s interface over **aicspylibczi**. Every imaging lane reads through it | any imaging panel; see the note below |
| `preprocessing.py` | Filtering, normalisation and gene-index handling shared by the single-cell notebooks, including the gene-ID/symbol resolution the figures depend on | Fig 4, ED Fig 6–9 |
| `aggregation.py` | Cluster-level averages over sparse matrices without densifying them | Fig 4d, 4e, ED Fig 6a |
| `correlation.py` | Cluster-by-cluster correlation, likewise sparse-safe | Fig 4e |
| `cell_ordering.py` | Intra-cluster cell ordering for heatmaps, so a heatmap's column order is reproducible | Fig 4d, ED Fig 6a, 6d, 7, 8c |
| `plotting.py` | Figure-export formatting: axis and colourbar handling, selective gene labelling | every heatmap panel |
| `paths.py` | The three roots the notebooks read and write — shipped inputs, fetched objects, results — each keeping the layout of the original analysis directory | `scrnaseq/run_chain.py` and every notebook it runs |
| `pipeline_io.py` | Reading and writing stage outputs when notebooks run head-less in the chain | the notebook chain |
| `sparse_utils.py` | Small sparse-matrix helpers used by the plotting utilities | — |
| `preflight.py` | Import-time checks for the staged notebooks | — |

## Why the CZI reader is a wrapper and not `czifile`

`czifile` reads odd-sized acquisitions (1025 or 1026 px) unreliably: for a file whose checksum never
changes it can return different pixel data between reads, or channel-dependent one-pixel offsets.
That is not hypothetical here — it reached a published number once and was corrected. `czi_compat`
keeps the familiar `czifile` call shape so the analysis code did not have to change, and reads
through `aicspylibczi` underneath. **Any new code reading a `.czi` in this repository should use
this module rather than `czifile`.**

`czifile` is therefore deliberately absent from `env/environment.yml`; a few scripts still import it
in paths that are documented as not runnable as shipped.
