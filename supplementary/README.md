# Supplementary Data 4, 5 and 6 — figure code

SD4, SD5 and SD6 are the analysis-display documents published with the paper. **This directory
holds the code that produced figures for them, not the code that assembled the documents.**

## Why the assembler is not here

The document layer — cover pages, tables of contents, caption typesetting, page numbering — was a
separate pipeline that laid each section out as HTML and printed it to PDF through **headless
Chrome**. That is a browser dependency for a scientific archive, and it reproduces something a
reader already has: the assembled PDFs ship with the paper as Supplementary Data.

It also could not run here regardless, because it consumes roughly 157 MB of pre-rendered figure
panels that are not in this repository.

So the document assembly was removed and the figure code kept. Nothing in this repository requires
Chrome.

## What is here

| script | produces |
|---|---|
| `sections/07_smd_threshold/replot_tier1_metrics.py` | SMD z-cutoff robustness metric plots |
| `sections/11_somite/replot_dotplots_paga.py` | somite-progression dot plots and PAGA graphs |
| `sections/11_somite/crop_panels.py` | trims baked matplotlib titles off staged panels |

These three do not run from this repository. Each reads saved summary tables or rendered panels
from the authors' analysis tree (written as `<analysis-root>/…`), which is not shipped, so they are
a record of how those figures were drawn. Their docstrings mention the section builders that placed
the output on the page (`build_07`, `build_11_somite.py`); those builders were part of the document
assembly and are not published.

## Where the rest of the figures come from

Most SD panels are outputs of the analysis lanes, not of anything under `supplementary/`:

- **imaging displays (SD4)** → `imaging/psmad_fig1/`, `imaging/foxf1_bead/`, `imaging/morphometry/`,
  `imaging/bmpr1a_psmad/`, `imaging/cograft_fig5/`
- **single-cell displays (SD5)** → `scrnaseq/`
- **alignment displays (SD6)** → `scrnaseq/morph_embryo_correspondence/`, `scrnaseq/human_embryo/`

The quantified Main and Extended Data panels are drawn and compared with the published Source Data
by the scripts in `figures/` and `bulkseq/`; the top-level README's figure index gives the code for
each panel.
