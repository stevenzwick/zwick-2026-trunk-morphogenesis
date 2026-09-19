"""Plot formatting helpers for figure export consistency."""

from __future__ import annotations

import matplotlib.pyplot as plt
import scanpy as sc

from src.trunk_morph_ref.preprocessing import resolve_symbols, var_names_to_symbols
from src.trunk_morph_ref.sparse_utils import sparse_zero_matrix_like


def hide_heatmap_axes_and_colorbar(fig_dict: dict) -> None:
    """Apply existing notebook heatmap cleanup for no-axes exports."""
    fig = plt.gcf()
    if len(fig.axes) > 0:
        colorbar_ax = fig.axes[-1]  # usually last axis is colorbar
        if colorbar_ax != fig_dict.get("groupby_ax", None):
            fig.delaxes(colorbar_ax)

    for ax in fig_dict.values():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("")
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if "groupby_ax" in fig_dict:
        fig_dict["groupby_ax"].set_visible(False)


def bold_selected_heatmap_yticklabels(ax_dict: dict, selected_genes) -> None:
    """Bold ytick labels on Scanpy heatmap axes when label text is selected."""
    selected = set(selected_genes)
    for label in ax_dict["heatmap_ax"].get_yticklabels():
        if label.get_text() in selected:
            label.set_fontweight("bold")


def keep_only_selected_heatmap_yticklabels(ax_dict: dict, selected_genes) -> None:
    """Keep only selected gene labels on Scanpy heatmap y-axis, hide others."""
    selected = set(selected_genes)
    tick_labels = [label.get_text() for label in ax_dict["heatmap_ax"].get_yticklabels()]
    new_labels = [gene if gene in selected else "" for gene in tick_labels]
    ax_dict["heatmap_ax"].set_yticklabels(new_labels)


def plot_empty_heatmap_axes(
    adata,
    *,
    var_names,
    groupby,
    gene_symbols: str = "gene_symbol",
    gene_symbol_original_col: str = "gene_symbol_original",
    strict_symbols: bool = False,
    allow_missing_symbols: bool = True,
    **heatmap_kwargs,
):
    """Plot heatmap using a zero-valued copy for axes-only exports."""
    resolved_var_names = resolve_symbols(
        adata,
        var_names,
        strict=strict_symbols,
        allow_missing=allow_missing_symbols,
        gene_symbol_col=gene_symbols,
        gene_symbol_original_col=gene_symbol_original_col,
    )
    if not resolved_var_names:
        raise ValueError("plot_empty_heatmap_axes resolved zero genes for var_names input.")
    input_tokens = [str(token) for token in var_names]
    var_name_index = set(adata.var_names.astype(str))
    all_tokens_are_gene_ids = bool(input_tokens) and all(
        token in var_name_index for token in input_tokens
    )

    if all_tokens_are_gene_ids:
        plot_var_names = resolved_var_names
    else:
        plot_var_names = var_names_to_symbols(
            adata,
            resolved_var_names,
            gene_symbol_col=gene_symbols,
        )

    adata_empty = adata.copy()
    adata_empty.X = sparse_zero_matrix_like(adata_empty.X)
    heatmap_kwargs = dict(heatmap_kwargs)
    if not all_tokens_are_gene_ids:
        heatmap_kwargs.setdefault("gene_symbols", gene_symbols)

    return sc.pl.heatmap(
        adata_empty,
        var_names=plot_var_names,
        groupby=groupby,
        **heatmap_kwargs,
    )
