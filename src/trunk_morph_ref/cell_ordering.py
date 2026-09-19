"""Utilities for intra-cluster cell ordering used in heatmap figures."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import seaborn as sns

from src.trunk_morph_ref.preprocessing import resolve_symbols, var_names_to_symbols


def ordered_cells_by_cluster_clustermap(
    adata,
    cluster_key: str,
    var_names: Sequence[str],
    *,
    figsize=(5, 5),
    vmin=0,
    vmax=5,
    cmap=None,
    method="ward",
    metric="euclidean",
    gene_symbols: str = "gene_symbol",
    gene_symbol_original_col: str = "gene_symbol_original",
    strict_symbols: bool = False,
    allow_missing_symbols: bool = True,
) -> list[str]:
    """Return cell names ordered within each cluster via clustermap dendrogram order."""
    resolved_var_names = resolve_symbols(
        adata,
        var_names,
        strict=strict_symbols,
        allow_missing=allow_missing_symbols,
        gene_symbol_col=gene_symbols,
        gene_symbol_original_col=gene_symbol_original_col,
    )
    if not resolved_var_names:
        raise ValueError(
            "ordered_cells_by_cluster_clustermap resolved zero genes for var_names input."
        )
    heatmap_labels = var_names_to_symbols(adata, resolved_var_names, gene_symbol_col=gene_symbols)

    cell_order: list[str] = []
    cluster_series = adata.obs[cluster_key]
    for cluster in cluster_series.cat.categories:
        cluster_mask = cluster_series == cluster
        cluster_cell_names = list(adata.obs_names[cluster_mask])
        # Clustermap cannot build a dendrogram for clusters with fewer than 2 cells.
        if len(cluster_cell_names) < 2:
            cell_order.extend(cluster_cell_names)
            continue
        # The analysis run that produced the figures used this per-cluster fallback: a cluster whose
        # clustermap ordering fails keeps its existing cell order, and the run continues.
        try:
            cluster_intracluster = sns.clustermap(
                adata[cluster_mask, resolved_var_names].X.todense().T,
                figsize=figsize,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                row_cluster=False,
                col_cluster=True,
                yticklabels=heatmap_labels,
                method=method,
                metric=metric,
            )
        except Exception as exc:
            print(
                f"[ordered_cells_by_cluster_clustermap] fallback to existing order for cluster={cluster!r}: {exc}"
            )
            cell_order.extend(cluster_cell_names)
            continue
        cell_order.extend(
            list(
                adata.obs_names[cluster_mask][
                    cluster_intracluster.dendrogram_col.reordered_ind
                ]
            )
        )
        plt.close(cluster_intracluster.fig)

    return cell_order
