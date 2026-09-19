# pSMAD gradient — dataset 2 (ED Fig 2j, 2k)

A second, independent pSMAD experiment. **Not** a re-run of the Fig 1 dataset: different chip,
different acquisition, separate analysis. Run the numbered notebooks in order.

⚠️ `notebooks/04_image_preparation.ipynb` is a **scaffold copied from the Fig 1 workflow and never
adapted to this dataset** — it says so in its own first cell. It ships for completeness of the
chain, but it produced nothing used in the paper. The panels for ED Fig 2j/2k come from
`03_gradient_quantification.ipynb`.

⚠️ This directory's saved figure folders are named `Fig1e_*` and `Fig1f_*`, inherited from the
workflow it was adapted from. They feed **ED Fig 2j and 2k**. Identify panels from the top-level
README's figure index, never from these names.

`derived/` holds the per-cyst and per-bin measurement tables the published panels are drawn from.
