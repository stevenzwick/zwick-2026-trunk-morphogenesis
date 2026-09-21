# 3D reporter timelapse — Fig 5h, ED Fig 10d

Half-max activation times for FOXF1-RFP and BMP4-YFP in individual trunk morphs (n = 65), and the
lag between them.

This is a different experiment from `timelapse_fig5i/`. That directory holds the **2D**
reporter timelapse behind Fig 5i and ED Fig 10f. This one is the **3D** organoid timelapse, behind
Fig 5h and ED Fig 10d. `notebooks/05b_reporter_spatial_context.ipynb` draws the ED Fig 10d pixel
density from a sampled pixel table built from the raw images, which is not shipped. The two are easy to conflate: similar reporters, similar names, different data.

## Two details that decide whether the panel reproduces

**A 48-hour clock offset.** The table stores half-max times on the acquisition clock; the figure
reports them on the differentiation clock. The offset is exactly 48 h — day 2, when differentiation
begins.

Different sigma thresholds per reporter. RFP is read at `positive_fraction_sigma4`, YFP at
`positive_fraction_sigma3`. This is deliberate, not a transcription slip.

Check both half-max columns, not the lag: swapping the pairing leaves the cohort **mean** lag
almost unchanged, so a check on that alone would pass with the wrong pairing. Per morph the two
pairings do differ, so the lag column is not invariant in the way a quick look might suggest.

Both rules live in `figures/panel_specs.py::prepare_halfmax`.

## Contents

`derived/05_cooperativity_halfmax_times.tsv` — long form, 1,170 rows (65 positions × 2 reporters ×
9 metrics). The figure uses two of those nine. `notebooks/` and `scripts/` carry the QC,
normalisation and Ilastik-based segmentation chain that produced it, with one exception below.

`derived/04b_background_stability_outliers.tsv` — the background-stability QC result, one row per
flagged position and reporter. Six positions are flagged: Pos1, Pos2, Pos3, Pos9, Pos10 and Pos17.
`notebooks/04b_reporter_background_stability_qc.ipynb` writes it and holds the criterion:
after early-offset alignment, a reporter is flagged when its background stays at or beyond 4σ
robust-z from the dataset-wide aggregate background trend for at least 8 consecutive frames
(`MAIN_ALIGNED_Z_THRESHOLD = 4.0`, `MAIN_ALIGNED_Z_RUN = 8`), plus a separate RFP late-tail rule
that fires on 8 consecutive frames at or below −2.25σ within the last 35 frames
(`RFP_LATE_LOW_TAIL_Z_THRESHOLD = -2.25`, `RFP_LATE_LOW_TAIL_RUN = 8`,
`RFP_LATE_LOW_TAIL_LAST_FRAMES = 35`). Each row carries the rule that fired, the severity score, and
the frame and hour at which the flag first and last applies.

**The two panels in this lane use this table differently, which is why their n differ.**
`notebooks/05b_reporter_spatial_context.ipynb` (ED Fig 10d) drops the six flagged positions when
it builds its pixel sample, so that panel is 59 of the 65 imaged positions.
`notebooks/05_preliminary_reporter_quantification.ipynb` (Fig 5h) loads the same table for QC
review and montage display only and never filters the half-max computation, so Fig 5h is all 65.
The shipped `derived/05_cooperativity_halfmax_times.tsv` therefore contains all 65 positions,
including the six flagged ones.

⚠️ **One hand-curated input is not in this repository.**
`results/qc/02_phase_artifact_excluded_frames.tsv` lists the frames dropped for phase artifacts.
`scripts/io/postprocess_ilastik_organoid_masks.py` and `notebooks/02_retained_frame_mask_review.ipynb`
both read it; nothing here writes it, and the file itself is not deposited. It sets
`exclude_from_analysis`, which carries through the mask post-processing and the reporter
quantification into the half-max table, so it affects every number in this lane. The shipped
`derived/` table already has those exclusions applied. Anyone needing the frame list should
contact the lead contact, as the paper's Data Availability Statement describes.

## Ilastik pixel classifier project

`ilastik/projects/pixelclass_FOXF1_BMP4_3d_timelapse.ilp` is the ilastik 1.4.0 Pixel Classification
project whose masks feed the published Fig 5h / ED Fig 10d panels
(`scripts/io/postprocess_ilastik_organoid_masks.py` and
`scripts/io/quantify_reporters_from_ilastik_masks.py` both consume its output under
`results/ilastik/.../full_dataset_v1`, and it is the only ilastik project in this lane). It is
shipped because the training labels are the most judgement-laden step in this pipeline, and a
reader should not have to take the resulting organoid masks on trust.

What is and is not in it:
- The project carries the trained classifier and the training-frame labels (small: pixel labels
  and feature-selection metadata, not imagery — the largest embedded array is a label block under
  100 KB).
- The **training frames themselves are not shipped** (too large for this repository) and are
  available from the lead contact on request, per the paper's Data Availability Statement.
- The project's embedded input paths (`Input Data/infos/lane*/Raw Data/filePath`, the prediction
  export filename template, 31 strings in this file) originally pointed at absolute paths under
  the analysis machine's home directory. They have been rewritten to `<analysis-root>/...`,
  preserving the relative structure.
  `ilastik/projects/PROVENANCE.tsv` records the original file's sha256 alongside the shipped,
  path-sanitised file's own sha256.
- **The sanitised project could not be re-opened in ilastik to confirm it still loads.** ilastik
  is not installed on the machine this repository was assembled on. The check that was possible —
  and was done — is structural: the sanitised HDF5 file has the identical set of groups and
  datasets as the original, identical dtype/shape/compression on every dataset, and byte-identical
  values everywhere except the 31 rewritten path strings.
