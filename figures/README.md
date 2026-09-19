# Figure scripts

Two kinds of file live here.

`panel_specs.py` and `scrnaseq_panels.py` are the readable record of *how each published panel is
made* — which table or object it reads, the subset rule that picks the plotted rows, the join keys,
and the normalisation. Those subset rules are part of the figure, not incidental: omit one and the
panel silently changes.

`verify_imaging_panels.py` and `verify_scrnaseq_panels.py` check each panel against the Source
Data published with the paper. They are the repository's regression test:

```bash
SOURCE_DATA_DIR=/path/to/source_data python figures/verify_imaging_panels.py
SOURCE_DATA_DIR=/path/to/source_data OBJECT_ROOT=/path/to/analysis python figures/verify_scrnaseq_panels.py
```

Each prints, per panel, the published rows it could not match and the largest difference against
the tolerance. `imaging/bmpr1a_psmad/scripts/08_ed3k_panel.py` draws ED Fig 3k and, with
`SOURCE_DATA_DIR` set, compares its plotted values with that sheet. The bulk RNA-seq panels are
compared by `bulkseq/verify_bulk_panels.py`.

## A note on tolerances

The Source Data sheets store values **rounded to a few decimals and saved as float32**, so a
published `7.8441` reads back as `7.844099998474121`. Tolerances are therefore derived per column
from the true decimal precision plus a float32 representation allowance — never from a fixed
epsilon, and never from the textual rendering of a float, which invents digits.

Getting this wrong does not produce a small error; it produces confident false failures.
