"""Render the single-cell panels.

Fig 4e needs nothing but this repository. The rest read the saved AnnData objects:

    OBJECT_ROOT=/path/to/analysis python figures/render_scrnaseq_panels.py

Fig 4a and Fig 4f additionally need the published Source Data, which is where their gene sets are
listed:

    OBJECT_ROOT=/path/to/analysis SOURCE_DATA_DIR=/path/to/workbooks python figures/render_scrnaseq_panels.py

Panels whose inputs are absent are skipped, with a line saying which, rather than failing or
drawing something else. See scrnaseq/README.md for which object each panel uses — they are not
interchangeable.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "figures"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from _plotting import dot_panel, batlow, save                     # noqa: E402
import scrnaseq_panels as S                                       # noqa: E402
from verify_scrnaseq_panels import gene_columns, obs_column, obs_index  # noqa: E402

OBJECT_ROOT = os.environ.get("OBJECT_ROOT", "")


def fig4e_correlation():
    """Morph <-> embryo cluster correlation. Shipped data; no object needed."""
    corr = pd.read_csv(REPO / S.CORR_FIG4E, index_col=0)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    image = ax.imshow(corr.to_numpy(float), cmap=batlow(), vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(corr.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(corr.shape[0]))
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_xlabel("Human embryo cluster", fontsize=8)
    ax.set_ylabel("Trunk morph cluster", fontsize=8)
    ax.set_title("Fig 4e — morph / embryo cluster correspondence", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.55, label="Pearson r").ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, "fig4e_correlation")


def _dotplot(sheet_genes, title, cbar, name, normalise):
    expression, _, labels = gene_columns(S.object_path(S.MORPH_FULLGENE), sheet_genes)
    clusters = S.display_order(labels)
    means, fracs = [], []
    for i, _gene in enumerate(sheet_genes):
        table = S.cluster_means(expression[:, i], labels).reindex(clusters)
        column = S.min_max_normalise(table["mean"]) if normalise else table["mean"]
        means.append(column.to_numpy(float))
        fracs.append(table["fraction"].to_numpy(float))
    fig = dot_panel(np.array(means), np.array(fracs), sheet_genes, clusters, cbar, title,
                    vmin=0, vmax=(1 if normalise else 5))
    save(fig, name)


def fig4a_dotplot(genes):
    _dotplot(genes, "Fig 4a — marker expression by cluster",
             "Mean scaled expression", "fig4a_dotplot", normalise=False)


def fig4f_dotplot(genes):
    _dotplot(genes, "Fig 4f — LPM morphogens by cluster",
             "Per-gene normalized expression", "fig4f_dotplot", normalise=True)


def fig4b_umap():
    import h5py
    with h5py.File(S.object_path(S.MORPH_SMD), "r") as fh:
        umap = fh["obsm"]["X_umap"][:]
        labels = obs_column(fh, "leiden_morph")
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for cluster in S.display_order(labels):
        sel = labels == cluster
        ax.scatter(umap[sel, 0], umap[sel, 1], s=2, label=cluster, edgecolors="none", alpha=0.85)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("Fig 4b — trunk morph UMAP (5,306 cells)", fontsize=9)
    ax.legend(frameon=False, fontsize=5, markerscale=3, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig4b_umap")


def fig4c_composition():
    import h5py
    with h5py.File(S.object_path(S.MORPH_SMD), "r") as fh:
        labels, source = obs_column(fh, "leiden_morph"), obs_column(fh, "source")
    counts = pd.crosstab(pd.Series(labels), pd.Series(source)).reindex(S.display_order(labels))
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bottom = np.zeros(len(fractions))
    for column in fractions.columns:
        ax.bar(range(len(fractions)), fractions[column], bottom=bottom, label=column, width=0.8)
        bottom += fractions[column].to_numpy()
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels(fractions.index, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Fraction of cluster")
    ax.set_title("Fig 4c — cluster composition by condition", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig4c_composition")


def ed9b_receptors():
    # Not the published ED Fig 9b, which is the stacked violin plot that
    # scrnaseq/trunk_main/notebooks/02_trunk_main.ipynb draws. This dot plot aggregates the same
    # expression values per cluster; figures/verify_scrnaseq_panels.py checks them against Source
    # Data per cell, joined on barcode, which is how that sheet is laid out.
    expression, _, labels = gene_columns(S.object_path(S.MORPH_FULLGENE), S.RECEPTORS)
    clusters = S.display_order(labels)
    means, fracs = [], []
    for i, _g in enumerate(S.RECEPTORS):
        table = S.cluster_means(expression[:, i], labels).reindex(clusters)
        means.append(table["mean"].to_numpy(float))
        fracs.append(table["fraction"].to_numpy(float))
    fig = dot_panel(np.array(means), np.array(fracs), S.RECEPTORS, clusters,
                    "Mean scaled expression", "ED Fig 9b values — BMP receptor expression per cluster",
                    vmin=0, vmax=2)
    save(fig, "ed9b_receptors")


def main() -> int:
    print("Rendering single-cell panels:")
    fig4e_correlation()

    # The SMD object is shipped in this repository, so these run from a clone with nothing set.
    if S.object_path(S.MORPH_SMD):
        fig4b_umap()
        fig4c_composition()
    else:
        print("\n  SMD object not found — skipping Fig 4b and Fig 4c.")

    # The full-gene object is 121 MB, past GitHub's per-file limit; it comes from the Zenodo
    # deposit. See data/DOWNLOAD.md.
    if not S.object_path(S.MORPH_FULLGENE):
        print("\n  Full-gene object not available — skipping Fig 4a, Fig 4f and ED 9b."
              "\n  Fetch it from the Zenodo deposit and set OBJECT_ROOT (data/DOWNLOAD.md).")
        return 0
    # Fig 4a's 38 genes and Fig 4f's 14 are listed nowhere in this repository — they are read from
    # the published Source Data sheets. Drawing either from a substitute list would put a panel
    # titled "Fig 4a" in front of a reader with the wrong rows in it, so they are skipped instead.
    if os.environ.get("SOURCE_DATA_DIR"):
        import verify_scrnaseq_panels as W
        fig4a_dotplot(list(dict.fromkeys(W.sheet("Fig.4a")["Gene"].astype(str))))
        fig4f_dotplot(list(dict.fromkeys(W.sheet("Fig.4f")["Gene"].astype(str))))
    else:
        print("\n  SOURCE_DATA_DIR not set — skipping Fig 4a and Fig 4f."
              "\n  Their gene sets come from the published Source Data sheets, not from this repository.")
    ed9b_receptors()
    print("\nDone. These are pre-Illustrator plots — see figures/_plotting.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
