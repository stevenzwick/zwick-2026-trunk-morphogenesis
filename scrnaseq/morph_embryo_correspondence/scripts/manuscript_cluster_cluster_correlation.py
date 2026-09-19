from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from src.trunk_morph_ref.aggregation import cluster_averages_sparse_safe
from src.trunk_morph_ref.preprocessing import resolve_symbols


EXPECTED_EMBRYO_BY_MORPH_CURRENT: dict[str, list[str]] = {
    "Forebrain / Midbrain": ["Week 4 Forebrain / Midbrain"],
    "Hindbrain": ["Week 4 Hindbrain"],
    "Intermediate-Ventral Spinal Cord": ["Week 4 Intermediate-Ventral Spinal Cord"],
    "Dorsal Spinal Cord": ["Week 4 Dorsal Spinal Cord"],
    "Roof Plate": ["Week 4 Roof Plate"],
    "Neural Crest": [
        "Week 3 Neural Crest - Early Migratory",
        "Week 4 Neural Crest - Derivatives",
        "Week 4 Neural Crest - Maturing Cranial",
    ],
    "Immature Neuron": ["Week 4 Neuron - CNS Excitatory"],
    "Posterior Neural Tube": ["Week 3 Posterior Neural Tube / Neuromesodermal Progenitors"],
    "Neuromesodermal Progenitors": ["Week 3 Posterior Neural Tube / Neuromesodermal Progenitors"],
    "Floor Plate": [],
    "Notochord": ["Week 3 / 4 Notochord"],
    "Presomitic Mesoderm": [
        "Week 3 Presomitic Mesoderm - Posterior",
        "Week 3 Presomitic Mesoderm - Anterior",
    ],
    "Early Somite": ["Week 3 Early Somite"],
    "Mature Somite": [
        "Week 4 Somite - Dorsal / Dermomyotome",
        "Week 4 Somite - Ventral / Sclerotome",
    ],
    "Intermediate Mesoderm": [
        "Week 4 Intermediate Mesoderm / Kidney Progenitors",
        "Week 3 Intermediate Mesoderm",
    ],
    "Lateral Plate Mesoderm": [
        "Week 3 Lateral Plate Mesoderm - Anterior",
        "Week 4 Lateral Plate Mesoderm - Splanchnic",
        "Week 4 Lateral Plate Mesoderm - Posterior",
        "Week 4 Lateral Plate Mesoderm - Gut Visceral Mesenchyme",
    ],
    "Endothelial": ["Week 3 Endothelial", "Week 4 Endothelial"],
    "Non-Neural Ectoderm": [
        "Week 4 Non-Neural Ectoderm - Surface Ectoderm",
        "Week 4 Non-Neural Ectoderm - Cranial Placodal",
        "Week 3 Non-Neural Ectoderm",
    ],
}


MORPH_DISPLAY_LABELS: dict[str, str] = {
    "Forebrain / Midbrain": "FB/MB",
    "Hindbrain": "HB",
    "Intermediate-Ventral Spinal Cord": "Int.-Vent. SC",
    "Dorsal Spinal Cord": "Dorsal SC",
    "Roof Plate": "Roof Plate",
    "Neural Crest": "Neural Crest",
    "Immature Neuron": "Imm. Neuron",
    "Posterior Neural Tube": "Posterior NT",
    "Neuromesodermal Progenitors": "NMP",
    "Floor Plate": "Floor Plate",
    "Notochord": "Notochord",
    "Presomitic Mesoderm": "PSM",
    "Early Somite": "Early Somite",
    "Mature Somite": "Mature Somite",
    "Intermediate Mesoderm": "IM",
    "Lateral Plate Mesoderm": "LPM",
    "Endothelial": "Endothelial",
    "Non-Neural Ectoderm": "NNE",
}


EMBRYO_DISPLAY_LABELS: dict[str, str] = {
    "Week 4 Forebrain - Optic Field": "W4 FB Optic",
    "Week 4 Forebrain / Midbrain": "W4 FB/MB",
    "Week 4 Forebrain - Diencephalon / Telencephalon": "W4 FB Di/Tel",
    "Week 4 Dorsal Midbrain": "W4 Dorsal MB",
    "Week 4 Hindbrain": "W4 HB",
    "Week 3 Anterior Neuroectoderm": "W3 Ant NeuroEcto",
    "Week 4 Intermediate-Ventral Spinal Cord": "W4 Int.-Vent. SC",
    "Week 4 Dorsal Spinal Cord": "W4 Dorsal SC",
    "Week 4 Roof Plate": "W4 Roof Plate",
    "Week 3 Neural Crest - Early Migratory": "W3 NC Early",
    "Week 4 Neural Crest - Derivatives": "W4 NC Deriv",
    "Week 4 Neural Crest - Maturing Cranial": "W4 NC Cranial",
    "Week 4 Neuron - CNS Excitatory": "W4 Neuron Exc",
    "Week 4 Neuron - Peripheral Sensory": "W4 Neuron Sens",
    "Week 4 Neuron - Hypothalamus / Neuroendocrine": "W4 Neuron Hyp/Ne",
    "Week 4 Neuron - CNS Inhibitory": "W4 Neuron Inhib",
    "Week 4 Neuron - Autonomic / Cholinergic": "W4 Neuron Aut/Chol",
    "Week 4 Neuron - Catecholaminergic / Glutamatergic": "W4 Neuron Cat/Glu",
    "Week 3 Posterior Neural Tube / Neuromesodermal Progenitors": "W3 Post NT/NMP",
    "Week 3 Presomitic Mesoderm - Posterior": "W3 PSM Post",
    "Week 3 / 4 Notochord": "W3/4 Noto",
    "Week 3 Presomitic Mesoderm - Anterior": "W3 PSM Ant",
    "Week 3 Early Somite": "W3 Early Som",
    "Week 4 Somite - Dorsal / Dermomyotome": "W4 Somite D",
    "Week 4 Somite - Ventral / Sclerotome": "W4 Somite V",
    "Week 4 Intermediate Mesoderm / Kidney Progenitors": "W4 IM/Kidney",
    "Week 3 Intermediate Mesoderm": "W3 IM",
    "Week 3 Lateral Plate Mesoderm - Anterior": "W3 LPM Ant",
    "Week 3 Lateral Plate Mesoderm - Craniofacial / Pharyngeal": "W3 LPM Cran/Phar",
    "Week 4 Lateral Plate Mesoderm - Splanchnic": "W4 LPM Spl",
    "Week 4 Lateral Plate Mesoderm - Craniofacial / Pharyngeal": "W4 LPM Cran/Phar",
    "Week 4 Lateral Plate Mesoderm - Posterior": "W4 LPM Post",
    "Week 4 Lateral Plate Mesoderm - Gut Visceral Mesenchyme": "W4 LPM Gut",
    "Week 3 / 4 Cardiomyocytes": "W3/4 Cardio",
    "Week 4 Skeletal Myocytes": "W4 Skeletal Myo",
    "Week 3 Endothelial": "W3 Endothelial",
    "Week 4 Endothelial": "W4 Endothelial",
    "Week 4 Head Mesenchyme - Multipotent Progenitors": "W4 Head Mes MP",
    "Week 4 Head Mesenchyme - Pharyngeal Arch Core Mesoderm": "W4 Head Mes ArchCore",
    "Week 4 Head Mesenchyme - Frontonasal Mesoderm": "W4 Head Mes Frontonasal",
    "Week 4 Head Mesenchyme - First Arch Oral / Palatal Mesenchyme": "W4 Head Mes Oral/Pal",
    "Week 4 Head Mesenchyme - Branchiomeric Muscle Progenitors": "W4 Head Mes BrMusc",
    "Week 4 Trunk Mesenchyme": "W4 Trunk Mes",
    "Week 4 Non-Neural Ectoderm - Surface Ectoderm": "W4 Surface Ecto",
    "Week 4 Non-Neural Ectoderm - Cranial Placodal": "W4 Cranial Plac",
    "Week 3 Non-Neural Ectoderm": "W3 NNE",
    "Week 4 Definitive Endoderm - Fetal Liver": "W4 DE Liver",
    "Week 4 Definitive Endoderm - Intestinal": "W4 DE Intest",
    "Week 3 / 4 Erythroid Precursors": "W3/4 Erythroid",
    "Week 3 / 4 Hematopoietic Progenitors": "W3/4 Hem Prog",
}

EMBRYO_ORDER_SWAPS: tuple[tuple[str, str], ...] = (
    ("Week 3 Presomitic Mesoderm - Posterior", "Week 3 / 4 Notochord"),
)


def cluster_averages(adata, cluster_key: str) -> pd.DataFrame:
    return cluster_averages_sparse_safe(adata, cluster_key)


def wrapped_labels(labels: Sequence[str], *, width: int) -> list[str]:
    wrapped: list[str] = []
    for label in labels:
        token = str(label).replace(" / ", " /\n").replace(" - ", " -\n")
        if "\n" not in token:
            token = textwrap.fill(token, width=width)
        wrapped.append(token)
    return wrapped


def compact_labels(labels: Sequence[str], lookup: Mapping[str, str]) -> list[str]:
    return [lookup.get(str(label), str(label)) for label in labels]


def apply_embryo_display_order(embryo_order_all: Sequence[str]) -> list[str]:
    ordered = list(embryo_order_all)
    for first, second in EMBRYO_ORDER_SWAPS:
        if first in ordered and second in ordered:
            i = ordered.index(first)
            j = ordered.index(second)
            ordered[i], ordered[j] = ordered[j], ordered[i]
    return ordered


def apply_morph_display_order(morph_order_all: Sequence[str]) -> list[str]:
    ordered = list(morph_order_all)
    anchor = "Intermediate-Ventral Spinal Cord"
    mover = "Floor Plate"
    if anchor in ordered and mover in ordered:
        ordered.remove(mover)
        ordered.insert(ordered.index(anchor), mover)
    return ordered


def heatmap_style(
    corr_df: pd.DataFrame,
    *,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
) -> dict[str, object]:
    values = corr_df.to_numpy(dtype=float).ravel()
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    vmin = float(np.nanquantile(values, lower_q))
    vmax = float(np.nanquantile(values, upper_q))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = data_min, data_max

    if vmin == vmax:
        vmin, vmax = data_min, data_max
    if vmin == vmax:
        pad = 1e-3 if vmin == 0 else abs(vmin) * 0.05
        vmin -= pad
        vmax += pad

    # Manuscript correlation heatmaps should use a symmetric zero-centered
    # diverging scale so equal positive/negative magnitudes receive equal
    # visual weight. We still use robust quantiles to avoid a single outlier
    # setting the full dynamic range.
    robust_abs = max(abs(vmin), abs(vmax))
    data_abs = max(abs(data_min), abs(data_max))
    span = robust_abs if np.isfinite(robust_abs) and robust_abs > 0 else data_abs
    if not np.isfinite(span) or span == 0:
        span = 1e-3

    return {
        "cmap": "seismic",
        "vmin": -span,
        "vmax": span,
        "center": 0,
    }


def build_gene_sets(
    *,
    adata_morph_full,
    adata_embryo_full,
    adata_morph_smd,
    adata_embryo_smd,
    integration_gene_panel: Sequence[str],
) -> dict[str, list[str]]:
    shared_full = pd.Index(adata_morph_full.var_names.astype(str)).intersection(
        adata_embryo_full.var_names.astype(str)
    )

    integration_gene_ids = resolve_symbols(
        adata_morph_full,
        integration_gene_panel,
        strict=False,
        allow_missing=True,
    )
    integration_gene_ids = [gene_id for gene_id in integration_gene_ids if gene_id in shared_full]

    morph_smd = pd.Index(adata_morph_smd.var_names.astype(str))
    embryo_smd = pd.Index(adata_embryo_smd.var_names.astype(str))

    morph_smd_shared = shared_full.intersection(morph_smd)
    embryo_smd_shared = shared_full.intersection(embryo_smd)
    smd_intersection = morph_smd_shared.intersection(embryo_smd_shared)
    smd_union_shared = morph_smd_shared.union(embryo_smd_shared)

    return {
        "integration_gene_panel": list(integration_gene_ids),
        "smd_intersection": list(smd_intersection),
        "smd_union_shared": list(smd_union_shared),
        "all_shared_genes": list(shared_full),
    }


def cross_cluster_correlation(
    avg_morph: pd.DataFrame,
    avg_embryo: pd.DataFrame,
    genes: Sequence[str],
) -> pd.DataFrame:
    gene_index = pd.Index(genes).astype(str)
    gene_index = avg_morph.columns.intersection(gene_index).intersection(avg_embryo.columns)
    if len(gene_index) == 0:
        raise ValueError("No shared genes available for correlation.")

    left = avg_morph[gene_index]
    right = avg_embryo[gene_index]
    corr_matrix = np.corrcoef(left.values, right.values)
    n_left = left.shape[0]
    return pd.DataFrame(
        corr_matrix[:n_left, n_left:],
        index=left.index,
        columns=right.index,
    )


def comparable_embryo_clusters(
    embryo_order_all: Sequence[str],
    *,
    expected_by_morph: Mapping[str, Sequence[str]] = EXPECTED_EMBRYO_BY_MORPH_CURRENT,
) -> list[str]:
    expected = {
        embryo_label
        for embryo_labels in expected_by_morph.values()
        for embryo_label in embryo_labels
    }
    return [label for label in embryo_order_all if label in expected]


def mapped_morph_clusters(
    morph_order_all: Sequence[str],
    *,
    expected_by_morph: Mapping[str, Sequence[str]] = EXPECTED_EMBRYO_BY_MORPH_CURRENT,
) -> list[str]:
    return [label for label in morph_order_all if expected_by_morph.get(label)]


def summarize_top_matches(
    corr_df: pd.DataFrame,
    *,
    expected_by_morph: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for morph_label in corr_df.index:
        ranked = corr_df.loc[morph_label].sort_values(ascending=False)
        top1_label = str(ranked.index[0])
        top1_corr = float(ranked.iloc[0])
        top2_corr = float(ranked.iloc[1]) if len(ranked) > 1 else np.nan
        expected_labels = list((expected_by_morph or {}).get(str(morph_label), []))
        expected_corrs = {
            embryo_label: float(corr_df.loc[morph_label, embryo_label])
            for embryo_label in expected_labels
            if embryo_label in corr_df.columns
        }
        expected_best = max(expected_corrs.values()) if expected_corrs else np.nan
        rows.append(
            {
                "morph_label": str(morph_label),
                "top1_embryo_label": top1_label,
                "top1_corr": top1_corr,
                "top2_corr": top2_corr,
                "top1_minus_top2": top1_corr - top2_corr if np.isfinite(top2_corr) else np.nan,
                "expected_labels": "|".join(expected_labels),
                "expected_best_corr": expected_best,
                "expected_contains_top1": top1_label in expected_labels if expected_labels else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    *,
    title: str,
    expected_by_morph: Mapping[str, Sequence[str]] | None = None,
    morph_label_lookup: Mapping[str, str] | None = None,
    embryo_label_lookup: Mapping[str, str] | None = None,
    x_label: str = "Trunk Morph Cluster",
    y_label: str = "Human Embryo Cluster",
    output_path: Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    n_rows, n_cols = corr_df.T.shape
    if figsize is None:
        matrix_width = max(4.8, 0.42 * n_cols)
        matrix_height = max(5.4, 0.30 * n_rows)
        figsize = (matrix_width + 7.6, matrix_height + 2.6)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    y_fontsize = 10 if n_rows <= 32 else 9 if n_rows <= 44 else 8
    style = heatmap_style(corr_df)
    norm = TwoSlopeNorm(vmin=style["vmin"], vcenter=0, vmax=style["vmax"])
    sns.heatmap(
        corr_df.T,
        cmap=style["cmap"],
        norm=norm,
        linewidths=0.15,
        linecolor="#f0f0f0",
        cbar_kws={"label": "Pearson r", "shrink": 0.82, "pad": 0.02},
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    morph_lookup = morph_label_lookup or MORPH_DISPLAY_LABELS
    embryo_lookup = embryo_label_lookup or EMBRYO_DISPLAY_LABELS
    ax.set_xticklabels(
        wrapped_labels(compact_labels(corr_df.index, morph_lookup), width=12),
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )
    ax.set_yticklabels(
        wrapped_labels(compact_labels(corr_df.columns, embryo_lookup), width=14),
        rotation=0,
        fontsize=y_fontsize,
    )

    for morph_ix, morph_label in enumerate(corr_df.index):
        best_embryo = corr_df.loc[morph_label].idxmax()
        embryo_ix = corr_df.columns.get_loc(best_embryo)
        ax.scatter(
            morph_ix + 0.5,
            embryo_ix + 0.5,
            s=20,
            facecolors="white",
            edgecolors="black",
            marker="o",
            linewidths=0.6,
            zorder=5,
        )

    if expected_by_morph:
        for morph_ix, morph_label in enumerate(corr_df.index):
            for embryo_label in expected_by_morph.get(str(morph_label), []):
                if embryo_label not in corr_df.columns:
                    continue
                embryo_ix = corr_df.columns.get_loc(embryo_label)
                ax.add_patch(
                    Rectangle(
                        (morph_ix, embryo_ix),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=0.8,
                    )
                )

    fig.subplots_adjust(left=0.42, bottom=0.24, right=0.93, top=0.90)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    return fig


def plot_correlation_heatmap_no_axes(
    corr_df: pd.DataFrame,
    *,
    expected_by_morph: Mapping[str, Sequence[str]] | None = None,
    output_path: Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    n_rows, n_cols = corr_df.T.shape
    if figsize is None:
        matrix_width = max(4.8, 0.42 * n_cols)
        matrix_height = max(5.4, 0.30 * n_rows)
        figsize = (matrix_width, matrix_height)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    style = heatmap_style(corr_df)
    norm = TwoSlopeNorm(vmin=style["vmin"], vcenter=0, vmax=style["vmax"])
    sns.heatmap(
        corr_df.T,
        cmap=style["cmap"],
        norm=norm,
        linewidths=0.15,
        linecolor="#f0f0f0",
        cbar=False,
        ax=ax,
    )

    for morph_ix, morph_label in enumerate(corr_df.index):
        best_embryo = corr_df.loc[morph_label].idxmax()
        embryo_ix = corr_df.columns.get_loc(best_embryo)
        ax.scatter(
            morph_ix + 0.5,
            embryo_ix + 0.5,
            s=20,
            facecolors="white",
            edgecolors="black",
            marker="o",
            linewidths=0.6,
            zorder=5,
        )

    if expected_by_morph:
        for morph_ix, morph_label in enumerate(corr_df.index):
            for embryo_label in expected_by_morph.get(str(morph_label), []):
                if embryo_label not in corr_df.columns:
                    continue
                embryo_ix = corr_df.columns.get_loc(embryo_label)
                ax.add_patch(
                    Rectangle(
                        (morph_ix, embryo_ix),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=0.8,
                    )
                )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    return fig


def plot_correlation_heatmap_label_template(
    corr_df: pd.DataFrame,
    *,
    morph_label_lookup: Mapping[str, str] | None = None,
    embryo_label_lookup: Mapping[str, str] | None = None,
    x_label: str = "Trunk morph cluster",
    y_label: str = "Human embryo cluster",
    output_path: Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    n_rows, n_cols = corr_df.T.shape
    if figsize is None:
        matrix_width = max(4.8, 0.42 * n_cols)
        matrix_height = max(5.4, 0.30 * n_rows)
        figsize = (matrix_width + 7.6, matrix_height + 2.6)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    y_fontsize = 10 if n_rows <= 32 else 9 if n_rows <= 44 else 8
    style = heatmap_style(corr_df)
    norm = TwoSlopeNorm(vmin=style["vmin"], vcenter=0, vmax=style["vmax"])
    sns.heatmap(
        corr_df.T,
        cmap=style["cmap"],
        norm=norm,
        linewidths=0.15,
        linecolor="#f0f0f0",
        cbar_kws={"label": "Pearson r", "shrink": 0.82, "pad": 0.02},
        ax=ax,
    )
    ax.set_title("")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    morph_lookup = morph_label_lookup or MORPH_DISPLAY_LABELS
    embryo_lookup = embryo_label_lookup or EMBRYO_DISPLAY_LABELS
    ax.set_xticklabels(
        wrapped_labels(compact_labels(corr_df.index, morph_lookup), width=12),
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )
    ax.set_yticklabels(
        wrapped_labels(compact_labels(corr_df.columns, embryo_lookup), width=14),
        rotation=0,
        fontsize=y_fontsize,
    )
    fig.subplots_adjust(left=0.42, bottom=0.24, right=0.93, top=0.98)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return fig


def plot_gene_set_grid(
    corr_results: Mapping[str, pd.DataFrame],
    *,
    title: str,
    expected_by_morph: Mapping[str, Sequence[str]] | None = None,
    morph_label_lookup: Mapping[str, str] | None = None,
    embryo_label_lookup: Mapping[str, str] | None = None,
    x_label: str = "Morph",
    y_label: str = "Embryo",
    output_path: Path | None = None,
) -> plt.Figure:
    names = list(corr_results.keys())
    n_panels = len(names)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12.8, 4.8 * n_rows),
        dpi=300,
        squeeze=False,
    )
    morph_lookup = morph_label_lookup or MORPH_DISPLAY_LABELS
    embryo_lookup = embryo_label_lookup or EMBRYO_DISPLAY_LABELS

    for panel_ix, (ax, name) in enumerate(zip(axes.flat, names, strict=False)):
        corr_df = corr_results[name]
        row_ix = panel_ix // n_cols
        col_ix = panel_ix % n_cols
        style = heatmap_style(corr_df)
        norm = TwoSlopeNorm(vmin=style["vmin"], vcenter=0, vmax=style["vmax"])
        sns.heatmap(
            corr_df.T,
            cmap=style["cmap"],
            norm=norm,
            cbar=False,
            linewidths=0.1,
            linecolor="#f0f0f0",
            xticklabels=compact_labels(corr_df.index, morph_lookup),
            yticklabels=compact_labels(corr_df.columns, embryo_lookup),
            ax=ax,
        )
        ax.set_title(name)
        show_x = row_ix == n_rows - 1
        show_y = col_ix == 0
        ax.set_xlabel(x_label if show_x else "")
        ax.set_ylabel(y_label if show_y else "")
        ax.tick_params(axis="x", labelbottom=show_x, length=0, labelrotation=40, labelsize=9)
        ax.tick_params(axis="y", labelleft=show_y, length=0, labelrotation=0, labelsize=9)
        if show_x:
            for label in ax.get_xticklabels():
                label.set_ha("right")
                label.set_rotation_mode("anchor")

        for morph_ix, morph_label in enumerate(corr_df.index):
            best_embryo = corr_df.loc[morph_label].idxmax()
            embryo_ix = corr_df.columns.get_loc(best_embryo)
            ax.scatter(
                morph_ix + 0.5,
                embryo_ix + 0.5,
                s=12,
                facecolors="white",
                edgecolors="black",
                marker="o",
                linewidths=0.45,
                zorder=5,
            )

        if expected_by_morph:
            for morph_ix, morph_label in enumerate(corr_df.index):
                for embryo_label in expected_by_morph.get(str(morph_label), []):
                    if embryo_label not in corr_df.columns:
                        continue
                    embryo_ix = corr_df.columns.get_loc(embryo_label)
                    ax.add_patch(
                        Rectangle(
                            (morph_ix, embryo_ix),
                            1,
                            1,
                            fill=False,
                            edgecolor="black",
                            linewidth=0.55,
                        )
                    )

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    return fig
