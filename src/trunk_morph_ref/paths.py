"""Path helpers for reference notebook outputs."""

from __future__ import annotations

import os
from pathlib import Path


def init_output_paths(base_dir: str | Path = ".") -> tuple[Path, Path, Path]:
    """Return and ensure standard results directories used by the notebook."""
    base = Path(base_dir)
    results_dir = base / "results"
    manuscript_fig_dir = results_dir / "manuscript_figures"
    extended_fig_dir = results_dir / "extended_data_figures"

    for output_dir in (results_dir, manuscript_fig_dir, extended_fig_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    return results_dir, manuscript_fig_dir, extended_fig_dir


# Added for this repository. The single-cell notebooks ran inside one analysis directory, reading
# their inputs and writing their outputs relative to it. Here those locations are three separate
# roots, and each keeps the layout the notebooks expect.


def chain_inputs_root(repo_root: str | Path) -> Path:
    """Files the single-cell notebooks read but no notebook writes: ``scrnaseq/chain_inputs/``.

    Committed to the repository, laid out as in the original analysis directory. Read-only: no
    notebook writes here.
    """
    return Path(repo_root) / "scrnaseq" / "chain_inputs"


def scrnaseq_input_root(repo_root: str | Path) -> Path:
    """Inputs built from GEO or fetched from Zenodo: ``$SCRNASEQ_INPUT_ROOT``.

    Holds what the GEO conversion notebooks (``00a``, ``00b``) write, the earlier trunk morph
    object from the Zenodo deposit at
    ``legacy/results/intermediates/02_trunk_main/adata_morph_with_clusters.h5ad``, and the earlier
    human embryo object at
    ``earlier_embryo_run/results/intermediates/07_human_embryo/adata_embryo_with_clusters.h5ad``.
    Defaults to ``data/scrnaseq_inputs/`` in the repository.
    """
    value = os.environ.get("SCRNASEQ_INPUT_ROOT")
    return Path(value) if value else Path(repo_root) / "data" / "scrnaseq_inputs"


def scrnaseq_results_root(repo_root: str | Path) -> Path:
    """Where the single-cell notebooks write: ``$SCRNASEQ_RESULTS_ROOT``.

    Laid out as the original analysis directory: ``trunk_main_dev/results/intermediates/<stage>/``
    for the pipeline stages, and ``experiments/rq1_5_human_embryo_alignment/results/`` for the
    morph-embryo comparison. The figure scripts read the same layout, so ``OBJECT_ROOT`` can point
    here. Defaults to ``scrnaseq/output/`` in the repository.
    """
    value = os.environ.get("SCRNASEQ_RESULTS_ROOT")
    return Path(value) if value else Path(repo_root) / "scrnaseq" / "output"
