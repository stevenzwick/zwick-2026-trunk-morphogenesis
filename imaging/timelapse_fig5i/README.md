# 2D reporter timelapse — Fig 5i, ED Fig 10f

Per-condition FOXF1-RFP/BMP4-YFP population dynamics from the 2D timelapse (dataset `20260213`,
`20260213_2D_BMP4-reporter_LPMdiff_d0-d2`).

This is a different experiment from `timelapse_3d_fig5h/`. That directory holds the **3D**
organoid timelapse behind Fig 5h and ED Fig 10d. This one is the **2D** reporter timelapse,
behind Fig 5i and ED Fig 10f. The two are easy to conflate: similar reporters, similar names,
different data.

## Contents

`derived/condition_population_mean_sem.csv`, `derived/condition_state_count_mean_sem.csv` — the
tables the panel is drawn from. `notebooks/` and `scripts/` carry the QC and normalisation stage
and the Ilastik-based segmentation and tracking stage that precede them;
`scripts/build_czi_dataset_manifests.py`'s docstring records that the published panel comes from
the `position_tiffs` dataset `20260213`, not the `czi_files` dataset `20260314` the same script can
also build a manifest for — the two are easy to conflate for the same reason as the 3D/2D split
above.

⚠️ **The last stage is not in this repository.** The step that produces these condition-level tables
— the two-channel per-cell state call behind the `+-`, `-+` and `++` columns, and whatever
`foxf1_bgsub_mean` and `bmp4_bgsub_mean` are averaged over — is not included, and the code here does
not reproduce it. The stages that are here measure the Ilastik probability map and the RFP channel;
nothing in this repository measures both reporter channels per cell. So these two tables ship as the
record of the measured values behind the panel, not as something a script here recomputes. Anyone
needing that step should contact the lead contact, as the paper's Data Availability Statement
describes.

## Ilastik pixel classifier project

`ilastik/datasets/20260213/projects/pixelclass_v1.ilp` is the ilastik 1.4.0 Pixel Classification
project whose masks feed the published Fig 5i / ED Fig 10f panel. It is shipped because the
training labels are the most judgement-laden step in this pipeline, and a reader should not have
to take the resulting masks on trust.

What is and is not in it:
- The project carries the trained classifier and the training-frame labels (small: pixel labels
  and feature-selection metadata, not imagery — the largest embedded array is a label block under
  100 KB).
- The **training frames themselves are not shipped** (too large for this repository) and are
  available from the lead contact on request, per the paper's Data Availability Statement.
- The project's embedded input paths (`Input Data/infos/lane*/Raw Data/filePath` and similar,
  11 strings in this file) originally pointed at absolute paths under the analysis machine's home
  directory. They have been rewritten to `<analysis-root>/...`, preserving the relative structure.
  `ilastik/datasets/20260213/projects/PROVENANCE.tsv` records the original file's sha256 alongside
  the shipped, path-sanitised file's own sha256.
- **The sanitised project could not be re-opened in ilastik to confirm it still loads.** ilastik
  is not installed on the machine this repository was assembled on. The check that was possible —
  and was done — is structural: the sanitised HDF5 file has the identical set of groups and
  datasets as the original, identical dtype/shape/compression on every dataset, and byte-identical
  values everywhere except the 11 rewritten path strings.

Only `20260213` is shipped. Other ilastik projects exist in the original analysis tree for
datasets `20260325` and `20260328`, but neither could be established as the source of a published
panel (`20260325` traces to segmentation-benchmark notebooks during method development;
`20260328` is a separate FOXF1-knockdown experiment not referenced by this pipeline), so they are
left out rather than shipped on a guess.

## Two details that decide whether the panel reproduces

Only two of the summary metrics in each derived table are plotted; see `figures/panel_specs.py`
and the notebooks for the full derivation chain from the ilastik masks through tracking and
condition aggregation.
