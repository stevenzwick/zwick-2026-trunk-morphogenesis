from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.io import mmwrite
from scipy import sparse
from scipy.optimize import nnls
from sklearn.decomposition import PCA

from src.trunk_morph_ref.preprocessing import resolve_symbols


EXPECTED_EMBRYO_BY_MORPH: dict[str, list[str]] = {
    "Forebrain / Midbrain": ["Week 4 Forebrain / Midbrain"],
    "Hindbrain": ["Week 4 Hindbrain"],
    "Ventral Spinal Cord": ["Week 4 Intermediate-Ventral Spinal Cord"],
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
    "Non-Neural Ectoderm": [
        "Week 4 Non-Neural Ectoderm - Surface Ectoderm",
        "Week 4 Non-Neural Ectoderm - Cranial Placodal",
        "Week 3 Non-Neural Ectoderm",
    ],
}


MORPH_DISPLAY_LABELS: dict[str, str] = {
    "Forebrain / Midbrain": "FB/MB",
    "Hindbrain": "HB",
    "Floor Plate": "Floor Plate",
    "Intermediate-Ventral Spinal Cord": "Int.-Vent. SC",
    "Ventral Spinal Cord": "Ventral SC",
    "Dorsal Spinal Cord": "Dorsal SC",
    "Roof Plate": "Roof Plate",
    "Neural Crest": "Neural Crest",
    "Immature Neuron": "Imm. Neuron",
    "Posterior Neural Tube": "Posterior NT",
    "Neuromesodermal Progenitors": "NMP",
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
    "Forebrain / Midbrain": "FB/MB",
    "Hindbrain": "HB",
    "Floor Plate": "Floor Plate",
    "Intermediate-Ventral Spinal Cord": "Int.-Vent. SC",
    "Dorsal Spinal Cord": "Dorsal SC",
    "Roof Plate": "Roof Plate",
    "Neural Crest": "Neural Crest",
    "Immature Neuron": "Imm. Neuron",
    "Posterior NT": "Posterior NT",
    "NMP": "NMP",
    "Notochord": "Notochord",
    "Presomitic Mesoderm": "PSM",
    "Early Somite": "Early Somite",
    "Mature Somite": "Mature Somite",
    "Intermediate Mesoderm": "IM",
    "Lateral Plate Mesoderm": "LPM",
    "Endothelial": "Endothelial",
    "Non-Neural Ectoderm": "NNE",
    "Week 4 Forebrain / Midbrain": "W4 FB/MB",
    "Week 4 Hindbrain": "W4 HB",
    "Week 4 Intermediate-Ventral Spinal Cord": "W4 Ventral SC",
    "Week 4 Dorsal Spinal Cord": "W4 Dorsal SC",
    "Week 4 Roof Plate": "W4 Roof Plate",
    "Week 3 Neural Crest - Early Migratory": "W3 NC Early",
    "Week 4 Neural Crest - Derivatives": "W4 NC Deriv",
    "Week 4 Neural Crest - Maturing Cranial": "W4 NC Cranial",
    "Week 4 Neuron - CNS Excitatory": "W4 Neuron Exc",
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
    "Week 4 Lateral Plate Mesoderm - Splanchnic": "W4 LPM Spl",
    "Week 4 Lateral Plate Mesoderm - Posterior": "W4 LPM Post",
    "Week 4 Lateral Plate Mesoderm - Gut Visceral Mesenchyme": "W4 LPM Gut",
    "Week 4 Non-Neural Ectoderm - Surface Ectoderm": "W4 NNE Surf",
    "Week 4 Non-Neural Ectoderm - Cranial Placodal": "W4 NNE Plac",
    "Week 3 Non-Neural Ectoderm": "W3 NNE",
}


MANUAL_CANONICAL_MARKERS: dict[str, list[str]] = {
    "Forebrain / Midbrain": ["OTX2", "SIX3", "LHX2"],
    "Hindbrain": ["CRABP1", "HOXA2", "EGR2"],
    "Ventral Spinal Cord": ["RNF220", "PTCH1", "RARB"],
    "Dorsal Spinal Cord": ["LMX1A", "ZIC2", "MSX1", "IRX3"],
    "Roof Plate": ["WNT3A", "OLIG3", "ZIC1"],
    "Neural Crest": ["FOXD3", "SOX10", "SNAI2", "TFAP2A", "TFAP2B"],
    "Immature Neuron": ["TUBB3", "ELAVL3", "STMN2"],
    "Posterior Neural Tube": ["CDX2", "CDX4", "HOXA9", "HOXB3", "HOXD3"],
    "Neuromesodermal Progenitors": ["TBXT", "FGF8", "WNT5A"],
    "Notochord": ["NOTO", "FOXA2", "SHH"],
    "Presomitic Mesoderm": ["MSGN1", "TBX6", "DLL3"],
    "Early Somite": ["RIPPLY1", "MEOX1", "TCF15", "FOXC2"],
    "Mature Somite": ["FOXC1", "MEOX2", "UNCX", "PAX9"],
    "Intermediate Mesoderm": ["PAX8", "ITGA8", "WT1", "OSR1"],
    "Lateral Plate Mesoderm": ["HAND1", "TMEM88", "PRRX1", "HAND2", "GATA6"],
    "Non-Neural Ectoderm": ["GRHL2", "CDH1", "EPCAM"],
}


def comparable_embryo_clusters(embryo_order: Sequence[str]) -> list[str]:
    expected = {
        embryo_label
        for labels in EXPECTED_EMBRYO_BY_MORPH.values()
        for embryo_label in labels
    }
    return [label for label in embryo_order if label in expected]


def cluster_averages(adata, cluster_key: str) -> pd.DataFrame:
    labels = adata.obs[cluster_key]
    if hasattr(labels.dtype, "categories"):
        categories = list(labels.cat.categories)
        categorical = pd.Categorical(labels.astype(str), categories=categories, ordered=True)
    else:
        categories = list(pd.Index(labels.astype(str)).unique())
        categorical = pd.Categorical(labels.astype(str), categories=categories, ordered=True)

    codes = categorical.codes
    valid_mask = codes >= 0
    if not np.all(valid_mask):
        categories = [categories[i] for i in sorted(set(codes[valid_mask]))]
        categorical = pd.Categorical(
            labels.astype(str)[valid_mask],
            categories=categories,
            ordered=True,
        )
        codes = categorical.codes

    x = adata.X[valid_mask]
    n_cells = int(valid_mask.sum())
    indicator = sparse.csr_matrix(
        (np.ones(n_cells, dtype=np.float64), (codes, np.arange(n_cells))),
        shape=(len(categories), n_cells),
    )
    sums = indicator @ x
    counts = np.asarray(indicator.sum(axis=1)).ravel()
    inv_counts = np.divide(
        1.0,
        counts,
        out=np.zeros_like(counts, dtype=np.float64),
        where=counts != 0,
    )

    if sparse.issparse(sums):
        means = sums.multiply(inv_counts[:, None]).toarray()
    else:
        means = np.asarray(sums) * inv_counts[:, None]

    return pd.DataFrame(means, index=categories, columns=adata.var_names)


def label_counts(adata, label_key: str) -> pd.Series:
    return adata.obs[label_key].astype(str).value_counts().sort_index()


def build_label_downsample_sizes(
    adata,
    label_key: str,
    *,
    cap: int,
    min_cells: int = 1,
) -> pd.Series:
    counts = label_counts(adata, label_key)
    target = counts.clip(upper=int(cap)).astype(int)
    target = target[target >= int(min_cells)]
    return target.sort_index()


def sample_cells_by_label(
    adata,
    label_key: str,
    sample_n_by_label: Mapping[str, int],
    *,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    labels = adata.obs[label_key].astype(str).to_numpy()
    selected_indices: list[np.ndarray] = []

    for label, requested_n in sample_n_by_label.items():
        label_mask = labels == str(label)
        label_indices = np.flatnonzero(label_mask)
        if len(label_indices) == 0:
            continue
        n_take = int(min(len(label_indices), int(requested_n)))
        if n_take <= 0:
            continue
        if n_take == len(label_indices):
            chosen = label_indices
        else:
            chosen = rng.choice(label_indices, size=n_take, replace=False)
        selected_indices.append(np.sort(chosen))

    if not selected_indices:
        raise ValueError("No cells selected during label-balanced downsampling.")
    return np.sort(np.concatenate(selected_indices))


def downsampled_cluster_averages(
    adata,
    cluster_key: str,
    sample_n_by_label: Mapping[str, int],
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_indices = sample_cells_by_label(
        adata,
        cluster_key,
        sample_n_by_label,
        random_state=random_state,
    )
    sampled = adata[selected_indices].copy()
    avg = cluster_averages(sampled, cluster_key)
    manifest = pd.DataFrame(
        {
            "label": pd.Index(sample_n_by_label.keys(), dtype=str),
            "target_n": [int(sample_n_by_label[k]) for k in sample_n_by_label],
        }
    )
    sampled_counts = sampled.obs[cluster_key].astype(str).value_counts()
    manifest["sampled_n"] = manifest["label"].map(sampled_counts).fillna(0).astype(int)
    manifest["random_state"] = int(random_state)
    return avg, manifest


def wrapped_labels(labels: Sequence[str], *, width: int) -> list[str]:
    wrapped = []
    for label in labels:
        token = str(label).replace(" / ", " /\n").replace(" - ", " -\n")
        if "\n" not in token:
            token = textwrap.fill(token, width=width)
        wrapped.append(token)
    return wrapped


def compact_labels(labels: Sequence[str], lookup: Mapping[str, str]) -> list[str]:
    return [lookup.get(str(label), str(label)) for label in labels]


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

    # All-positive matrices should use a sequential map; diverging maps waste dynamic range.
    if data_min >= 0:
        return {
            "cmap": "Reds",
            "vmin": max(data_min, vmin),
            "vmax": min(data_max, vmax),
            "center": None,
            "data_min": data_min,
            "data_max": data_max,
        }

    if data_max <= 0:
        return {
            "cmap": "Blues_r",
            "vmin": max(data_min, vmin),
            "vmax": min(data_max, vmax),
            "center": None,
            "data_min": data_min,
            "data_max": data_max,
        }

    return {
        "cmap": "seismic",
        "vmin": max(data_min, vmin),
        "vmax": min(data_max, vmax),
        "center": 0,
        "data_min": data_min,
        "data_max": data_max,
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
        "integration_gene_panel": integration_gene_ids,
        "all_shared_genes": list(shared_full),
        "morph_smd_genes": list(morph_smd_shared),
        "embryo_smd_genes": list(embryo_smd_shared),
        "smd_intersection": list(smd_intersection),
        "smd_union_shared": list(smd_union_shared),
    }


def build_manual_canonical_marker_panel(
    adata_morph_full,
    adata_embryo_full,
    *,
    marker_panel: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    panel = marker_panel or MANUAL_CANONICAL_MARKERS
    shared_full = pd.Index(adata_morph_full.var_names.astype(str)).intersection(
        adata_embryo_full.var_names.astype(str)
    )

    resolved_panel: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    for marker_set, genes in panel.items():
        resolved_ids = resolve_symbols(
            adata_morph_full,
            genes,
            strict=False,
            allow_missing=True,
        )
        resolved_ids = [str(gid) for gid in resolved_ids if str(gid) in shared_full]

        # Preserve order while removing duplicates after symbol resolution.
        resolved_unique = list(dict.fromkeys(resolved_ids))
        resolved_panel[str(marker_set)] = resolved_unique

        gene_to_id: dict[str, str | None] = {}
        for gene_symbol in genes:
            resolved_single = resolve_symbols(
                adata_morph_full,
                [gene_symbol],
                strict=False,
                allow_missing=True,
            )
            gene_id = str(resolved_single[0]) if resolved_single else None
            if gene_id not in shared_full:
                gene_id = None
            gene_to_id[str(gene_symbol)] = gene_id

        for gene_symbol in genes:
            gene_id = gene_to_id[str(gene_symbol)]
            rows.append(
                {
                    "marker_set": marker_set,
                    "gene_symbol": gene_symbol,
                    "gene_id": gene_id,
                    "kept": gene_id is not None,
                }
            )

    marker_table = pd.DataFrame(rows)
    return resolved_panel, marker_table


def marker_set_score_matrices(
    avg_morph: pd.DataFrame,
    avg_embryo: pd.DataFrame,
    marker_panel_gene_ids: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel_genes: list[str] = []
    for genes in marker_panel_gene_ids.values():
        panel_genes.extend([str(g) for g in genes])
    panel_gene_index = pd.Index(panel_genes).drop_duplicates()
    panel_gene_index = avg_morph.columns.intersection(panel_gene_index).intersection(avg_embryo.columns)
    if len(panel_gene_index) == 0:
        raise ValueError("No shared genes available for manual canonical marker scoring.")

    morph_aligned = avg_morph[panel_gene_index].copy()
    embryo_aligned = avg_embryo[panel_gene_index].copy()
    combined = pd.concat([morph_aligned, embryo_aligned], axis=0)
    means = combined.mean(axis=0)
    stds = combined.std(axis=0, ddof=0)
    keep = stds > 0
    if not np.any(keep):
        raise ValueError("All manual canonical marker genes have zero variance across cluster averages.")

    morph_scaled = (morph_aligned.loc[:, keep] - means[keep]) / stds[keep]
    embryo_scaled = (embryo_aligned.loc[:, keep] - means[keep]) / stds[keep]

    score_rows: list[dict[str, object]] = []
    marker_set_sizes: list[dict[str, object]] = []
    marker_set_names = list(marker_panel_gene_ids.keys())

    morph_scores = pd.DataFrame(index=morph_scaled.index)
    embryo_scores = pd.DataFrame(index=embryo_scaled.index)
    for marker_set in marker_set_names:
        marker_genes = [g for g in marker_panel_gene_ids[marker_set] if g in morph_scaled.columns]
        if not marker_genes:
            continue
        morph_scores[marker_set] = morph_scaled[marker_genes].mean(axis=1)
        embryo_scores[marker_set] = embryo_scaled[marker_genes].mean(axis=1)
        marker_set_sizes.append(
            {
                "marker_set": marker_set,
                "n_genes": len(marker_genes),
                "genes": "|".join(marker_genes),
            }
        )

    marker_set_df = pd.DataFrame(marker_set_sizes)
    return morph_scores, embryo_scores, marker_set_df


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


def map_query_cells_to_reference_clusters(
    query_adata,
    reference_avg: pd.DataFrame,
    genes: Sequence[str],
    *,
    query_label_key: str,
    expected_by_query_label: Mapping[str, Sequence[str]] | None = None,
    batch_size: int = 512,
) -> pd.DataFrame:
    gene_index = pd.Index(genes).astype(str)
    gene_index = (
        pd.Index(query_adata.var_names.astype(str))
        .intersection(gene_index)
        .intersection(reference_avg.columns.astype(str))
    )
    if len(gene_index) == 0:
        raise ValueError("No shared genes available for query-to-reference mapping.")

    reference = reference_avg.loc[:, gene_index].to_numpy(dtype=np.float64)
    reference_labels = reference_avg.index.astype(str).to_numpy()
    ref_centered = reference - reference.mean(axis=1, keepdims=True)
    ref_norm = np.linalg.norm(ref_centered, axis=1)
    ref_norm_safe = np.where(ref_norm == 0, np.nan, ref_norm)

    query_view = query_adata[:, gene_index]
    query_labels = query_adata.obs[query_label_key].astype(str).to_numpy()
    query_ids = query_adata.obs_names.astype(str).to_numpy()

    rows: list[pd.DataFrame] = []
    expected_lookup = expected_by_query_label or {}

    for start in range(0, query_adata.n_obs, batch_size):
        stop = min(start + batch_size, query_adata.n_obs)
        x_batch = query_view.X[start:stop]
        if sparse.issparse(x_batch):
            x_batch = x_batch.toarray()
        x_batch = np.asarray(x_batch, dtype=np.float64)

        batch_centered = x_batch - x_batch.mean(axis=1, keepdims=True)
        batch_norm = np.linalg.norm(batch_centered, axis=1, keepdims=True)
        batch_norm_safe = np.where(batch_norm == 0, np.nan, batch_norm)
        corr = (batch_centered @ ref_centered.T) / (batch_norm_safe * ref_norm_safe[None, :])
        corr = np.asarray(corr, dtype=np.float64)

        sort_idx = np.argsort(np.where(np.isnan(corr), -np.inf, corr), axis=1)
        top1_idx = sort_idx[:, -1]
        top2_idx = sort_idx[:, -2] if corr.shape[1] > 1 else np.full(corr.shape[0], -1)

        top1_label = reference_labels[top1_idx]
        top1_corr = corr[np.arange(corr.shape[0]), top1_idx]
        top2_label = np.where(top2_idx >= 0, reference_labels[np.maximum(top2_idx, 0)], None)
        top2_corr = np.where(top2_idx >= 0, corr[np.arange(corr.shape[0]), top2_idx], np.nan)

        batch_expected = [list(expected_lookup.get(label, [])) for label in query_labels[start:stop]]
        expected_best_corr: list[float | None] = []
        expected_contains_top1: list[bool | None] = []
        expected_labels_joined: list[str] = []
        for row_idx, expected_labels in enumerate(batch_expected):
            expected_labels_joined.append("|".join(expected_labels))
            expected_indices = [i for i, ref_label in enumerate(reference_labels) if ref_label in expected_labels]
            if expected_indices:
                expected_best_corr.append(float(np.nanmax(corr[row_idx, expected_indices])))
                expected_contains_top1.append(bool(top1_label[row_idx] in expected_labels))
            else:
                expected_best_corr.append(None)
                expected_contains_top1.append(None)

        batch_df = pd.DataFrame(
            {
                "cell_id": query_ids[start:stop],
                "morph_label": query_labels[start:stop],
                "top1_embryo_label": top1_label,
                "top1_corr": top1_corr,
                "top2_embryo_label": top2_label,
                "top2_corr": top2_corr,
                "top1_minus_top2": top1_corr - top2_corr,
                "expected_embryo_labels": expected_labels_joined,
                "expected_best_corr": expected_best_corr,
                "expected_contains_top1": expected_contains_top1,
                "n_genes": len(gene_index),
            }
        )
        rows.append(batch_df)

    return pd.concat(rows, axis=0, ignore_index=True)


def summarize_cell_reference_mapping(
    assignment_df: pd.DataFrame,
    *,
    morph_order: Sequence[str] | None = None,
    embryo_order: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if assignment_df.empty:
        raise ValueError("Assignment dataframe is empty.")

    cluster_summary = (
        assignment_df.groupby("morph_label", observed=True)
        .apply(
            lambda df: pd.Series(
                {
                    "n_cells": int(len(df)),
                    "top_assigned_embryo_label": df["top1_embryo_label"].value_counts().idxmax(),
                    "top_assigned_fraction": float(df["top1_embryo_label"].value_counts(normalize=True).iloc[0]),
                    "expected_assignment_rate": float(df["expected_contains_top1"].dropna().mean()),
                    "median_top1_corr": float(df["top1_corr"].median()),
                    "median_top1_minus_top2": float(df["top1_minus_top2"].median()),
                }
            )
        )
        .reset_index()
    )
    if morph_order is not None:
        cluster_summary["morph_label"] = pd.Categorical(
            cluster_summary["morph_label"],
            categories=list(morph_order),
            ordered=True,
        )
        cluster_summary = cluster_summary.sort_values("morph_label").reset_index(drop=True)

    assignment_fraction = pd.crosstab(
        assignment_df["morph_label"],
        assignment_df["top1_embryo_label"],
        normalize="index",
    )
    if morph_order is not None:
        assignment_fraction = assignment_fraction.reindex(list(morph_order))
    if embryo_order is not None:
        assignment_fraction = assignment_fraction.reindex(columns=list(embryo_order), fill_value=0.0)
    assignment_fraction = assignment_fraction.fillna(0.0)

    overview = pd.DataFrame(
        [
            {
                "n_cells": int(len(assignment_df)),
                "micro_expected_assignment_rate": float(assignment_df["expected_contains_top1"].dropna().mean()),
                "macro_expected_assignment_rate": float(cluster_summary["expected_assignment_rate"].dropna().mean()),
                "median_top1_corr": float(assignment_df["top1_corr"].median()),
                "median_top1_minus_top2": float(assignment_df["top1_minus_top2"].median()),
            }
        ]
    )
    return cluster_summary, assignment_fraction, overview


def collapse_assignment_matrix_for_display(
    assignment_fraction: pd.DataFrame,
    *,
    comparable_embryo_labels: Sequence[str],
    other_label: str = "Other embryo",
) -> pd.DataFrame:
    comparable_labels = [label for label in comparable_embryo_labels if label in assignment_fraction.columns]
    display_df = assignment_fraction.reindex(columns=comparable_labels, fill_value=0.0).copy()
    other_fraction = 1.0 - display_df.sum(axis=1)
    other_fraction = other_fraction.clip(lower=0.0)
    display_df[other_label] = other_fraction
    return display_df


def _dense_gene_matrix(adata, gene_index: Sequence[str]) -> np.ndarray:
    view = adata[:, list(gene_index)]
    x = view.X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def export_adata_subset_to_mtx(
    adata,
    genes: Sequence[str],
    *,
    label_key: str,
    out_dir: Path,
    prefix: str,
) -> pd.Index:
    gene_index = pd.Index(adata.var_names.astype(str)).intersection(pd.Index(genes).astype(str))
    if len(gene_index) == 0:
        raise ValueError("No shared genes available for matrix export.")

    out_dir.mkdir(parents=True, exist_ok=True)
    view = adata[:, list(gene_index)]
    x = view.X
    if sparse.issparse(x):
        x = x.tocoo()
    else:
        x = sparse.coo_matrix(np.asarray(x, dtype=np.float32))

    mmwrite(out_dir / f"{prefix}_matrix.mtx", x.T)
    pd.Series(gene_index.astype(str)).to_csv(
        out_dir / f"{prefix}_genes.csv",
        index=False,
        header=False,
    )
    pd.DataFrame(
        {
            "cell_id": view.obs_names.astype(str),
            label_key: view.obs[label_key].astype(str).to_numpy(),
        }
    ).to_csv(out_dir / f"{prefix}_meta.csv", index=False)
    return gene_index


def build_joint_integration_adata(
    morph_adata,
    embryo_adata,
    genes: Sequence[str],
    *,
    morph_label_key: str,
    embryo_label_key: str,
):
    import anndata as ad

    gene_index = pd.Index(genes).astype(str)
    gene_index = (
        pd.Index(morph_adata.var_names.astype(str))
        .intersection(gene_index)
        .intersection(embryo_adata.var_names.astype(str))
    )
    if len(gene_index) == 0:
        raise ValueError("No shared genes available for joint integration.")

    morph_view = morph_adata[:, list(gene_index)].copy()
    embryo_view = embryo_adata[:, list(gene_index)].copy()

    morph_view.obs = pd.DataFrame(
        {
            "dataset": "morph",
            "morph_label": morph_view.obs[morph_label_key].astype(str).to_numpy(),
            "embryo_label": "",
        },
        index=morph_view.obs_names.astype(str),
    )
    embryo_view.obs = pd.DataFrame(
        {
            "dataset": "embryo",
            "morph_label": "",
            "embryo_label": embryo_view.obs[embryo_label_key].astype(str).to_numpy(),
        },
        index=embryo_view.obs_names.astype(str),
    )

    joint = ad.concat(
        [morph_view, embryo_view],
        join="inner",
        merge="same",
        index_unique=None,
    )
    joint.obs["dataset"] = pd.Categorical(
        joint.obs["dataset"],
        categories=["morph", "embryo"],
        ordered=True,
    )
    return joint, gene_index


def assign_query_by_reference_embedding_knn(
    query_embedding_df: pd.DataFrame,
    reference_embedding_df: pd.DataFrame,
    *,
    query_label_col: str,
    reference_label_col: str,
    expected_by_query_label: Mapping[str, Sequence[str]] | None = None,
    embedding_cols: Sequence[str] | None = None,
    k: int = 25,
    batch_size: int = 512,
) -> pd.DataFrame:
    if embedding_cols is None:
        excluded = {"cell_id", query_label_col, reference_label_col, "dataset"}
        inferred_cols = [
            col
            for col in query_embedding_df.columns
            if col in reference_embedding_df.columns and col not in excluded
        ]
        embedding_cols = inferred_cols

    embedding_cols = list(embedding_cols)
    if len(embedding_cols) == 0:
        raise ValueError("No shared embedding columns available for kNN assignment.")

    x_ref = reference_embedding_df.loc[:, embedding_cols].to_numpy(dtype=np.float32, copy=True)
    x_query = query_embedding_df.loc[:, embedding_cols].to_numpy(dtype=np.float32, copy=True)
    if x_ref.shape[0] == 0 or x_query.shape[0] == 0:
        raise ValueError("Reference or query embedding is empty.")

    reference_labels = reference_embedding_df[reference_label_col].astype(str).to_numpy()
    label_index = pd.Index(pd.Index(reference_labels).unique())
    query_labels = query_embedding_df[query_label_col].astype(str).to_numpy()
    query_ids = query_embedding_df["cell_id"].astype(str).to_numpy()
    expected_lookup = expected_by_query_label or {}
    k_eff = int(min(k, x_ref.shape[0]))

    ref_sq_norm = np.sum(x_ref * x_ref, axis=1, dtype=np.float32)
    rows: list[pd.DataFrame] = []

    for start in range(0, x_query.shape[0], batch_size):
        stop = min(start + batch_size, x_query.shape[0])
        query_block = x_query[start:stop]
        query_sq_norm = np.sum(query_block * query_block, axis=1, dtype=np.float32)[:, None]
        dist_sq = query_sq_norm + ref_sq_norm[None, :] - (2.0 * (query_block @ x_ref.T))
        np.maximum(dist_sq, 0.0, out=dist_sq)

        idx_part = np.argpartition(dist_sq, kth=k_eff - 1, axis=1)[:, :k_eff]
        dist_sq_part = np.take_along_axis(dist_sq, idx_part, axis=1)
        sort_order = np.argsort(dist_sq_part, axis=1)
        indices = np.take_along_axis(idx_part, sort_order, axis=1)
        distances = np.sqrt(np.take_along_axis(dist_sq_part, sort_order, axis=1)).astype(
            np.float32,
            copy=False,
        )
        neighbor_labels = reference_labels[indices]
        neighbor_codes = label_index.get_indexer(neighbor_labels.ravel()).reshape(neighbor_labels.shape)

        top1_labels: list[str] = []
        top2_labels: list[str | None] = []
        top1_vote_fracs: list[float] = []
        top2_vote_fracs: list[float | None] = []
        top1_minus_top2: list[float | None] = []
        top1_mean_distances: list[float] = []
        expected_best_vote_fracs: list[float | None] = []
        expected_contains_top1: list[bool | None] = []
        expected_labels_joined: list[str] = []

        for row_idx in range(neighbor_codes.shape[0]):
            counts = np.bincount(neighbor_codes[row_idx], minlength=len(label_index)).astype(np.float64)
            vote_fracs = counts / counts.sum()
            top_rank = np.argsort(vote_fracs)[::-1]
            top1_code = int(top_rank[0])
            top2_code = int(top_rank[1]) if len(top_rank) > 1 else None
            top1_label = str(label_index[top1_code])
            top2_label = str(label_index[top2_code]) if top2_code is not None else None
            top1_frac = float(vote_fracs[top1_code])
            top2_frac = float(vote_fracs[top2_code]) if top2_code is not None else None

            top1_neighbor_mask = neighbor_codes[row_idx] == top1_code
            top1_mean_distance = float(distances[row_idx, top1_neighbor_mask].mean())

            expected_labels = list(expected_lookup.get(query_labels[start + row_idx], []))
            expected_labels_joined.append("|".join(expected_labels))
            expected_codes = [label_index.get_loc(lbl) for lbl in expected_labels if lbl in label_index]
            if expected_codes:
                expected_best_vote_fracs.append(float(vote_fracs[expected_codes].max()))
                expected_contains_top1.append(bool(top1_label in expected_labels))
            else:
                expected_best_vote_fracs.append(None)
                expected_contains_top1.append(None)

            top1_labels.append(top1_label)
            top2_labels.append(top2_label)
            top1_vote_fracs.append(top1_frac)
            top2_vote_fracs.append(top2_frac)
            top1_minus_top2.append(None if top2_frac is None else top1_frac - top2_frac)
            top1_mean_distances.append(top1_mean_distance)

        rows.append(
            pd.DataFrame(
                {
                    "cell_id": query_ids[start:stop],
                    "morph_label": query_labels[start:stop],
                    "top1_embryo_label": top1_labels,
                    "top1_vote_fraction": top1_vote_fracs,
                    "top2_embryo_label": top2_labels,
                    "top2_vote_fraction": top2_vote_fracs,
                    "top1_minus_top2": top1_minus_top2,
                    "top1_mean_distance": top1_mean_distances,
                    "expected_embryo_labels": expected_labels_joined,
                    "expected_best_vote_fraction": expected_best_vote_fracs,
                    "expected_contains_top1": expected_contains_top1,
                    "embedding_dim": len(embedding_cols),
                    "k_neighbors": k_eff,
                }
            )
        )

    return pd.concat(rows, axis=0, ignore_index=True)


def assign_query_by_graph_connectivities(
    joint_adata,
    *,
    dataset_key: str = "dataset",
    query_dataset: str = "morph",
    reference_dataset: str = "embryo",
    query_label_key: str = "morph_label",
    reference_label_key: str = "embryo_label",
    expected_by_query_label: Mapping[str, Sequence[str]] | None = None,
    fallback_embedding_key: str = "X_pca",
    fallback_k: int = 25,
) -> pd.DataFrame:
    if "connectivities" not in joint_adata.obsp:
        raise ValueError("Joint AnnData object has no connectivities graph.")

    graph = joint_adata.obsp["connectivities"]
    if not sparse.issparse(graph):
        graph = sparse.csr_matrix(np.asarray(graph, dtype=np.float32))
    else:
        graph = graph.tocsr().astype(np.float32)

    distance_graph = joint_adata.obsp.get("distances")
    if distance_graph is not None:
        if not sparse.issparse(distance_graph):
            distance_graph = sparse.csr_matrix(np.asarray(distance_graph, dtype=np.float32))
        else:
            distance_graph = distance_graph.tocsr().astype(np.float32)

    dataset = joint_adata.obs[dataset_key].astype(str).to_numpy()
    query_mask = dataset == str(query_dataset)
    reference_mask = dataset == str(reference_dataset)
    if not np.any(query_mask) or not np.any(reference_mask):
        raise ValueError("Joint AnnData must contain both query and reference datasets.")

    reference_indices = np.flatnonzero(reference_mask)
    reference_labels = joint_adata.obs.loc[reference_mask, reference_label_key].astype(str).to_numpy()
    label_index = pd.Index(pd.Index(reference_labels).unique())
    reference_lookup = pd.Series(reference_labels, index=reference_indices)
    fallback_embedding = joint_adata.obsm.get(fallback_embedding_key)
    if fallback_embedding is not None:
        fallback_embedding = np.asarray(fallback_embedding, dtype=np.float32)
        ref_fallback = fallback_embedding[reference_mask]
        ref_fallback_sq_norm = np.sum(ref_fallback * ref_fallback, axis=1, dtype=np.float32)
        fallback_k_eff = int(min(fallback_k, ref_fallback.shape[0]))
    else:
        ref_fallback = None
        ref_fallback_sq_norm = None
        fallback_k_eff = 0

    query_indices = np.flatnonzero(query_mask)
    query_ids = joint_adata.obs_names[query_mask].astype(str)
    query_labels = joint_adata.obs.loc[query_mask, query_label_key].astype(str).to_numpy()
    expected_lookup = expected_by_query_label or {}
    rows: list[dict[str, object]] = []

    for local_idx, query_idx in enumerate(query_indices):
        row = graph.getrow(query_idx)
        neighbor_global = row.indices
        neighbor_weights = row.data.astype(np.float64, copy=False)
        keep = reference_mask[neighbor_global] & (neighbor_weights > 0)

        if not np.any(keep):
            if ref_fallback is None or fallback_k_eff < 1:
                raise ValueError(
                    f"Query cell {joint_adata.obs_names[query_idx]} has no reference neighbors in the integration graph."
                )
            query_vec = fallback_embedding[query_idx]
            query_sq_norm = np.sum(query_vec * query_vec, dtype=np.float32)
            dist_sq = query_sq_norm + ref_fallback_sq_norm - (2.0 * (ref_fallback @ query_vec))
            np.maximum(dist_sq, 0.0, out=dist_sq)
            idx_part = np.argpartition(dist_sq, kth=fallback_k_eff - 1)[:fallback_k_eff]
            sort_order = np.argsort(dist_sq[idx_part])
            ref_local_indices = idx_part[sort_order]
            ref_neighbor_indices = reference_indices[ref_local_indices]
            ref_neighbor_labels = reference_labels[ref_local_indices]
            ref_neighbor_weights = np.full(len(ref_local_indices), 1.0 / len(ref_local_indices), dtype=np.float64)
            fallback_top1_mean_distance = float(np.sqrt(dist_sq[ref_local_indices]).mean())
        else:
            ref_neighbor_indices = neighbor_global[keep]
            ref_neighbor_weights = neighbor_weights[keep]
            ref_neighbor_labels = reference_lookup.loc[ref_neighbor_indices].to_numpy()
            fallback_top1_mean_distance = np.nan

        counts = (
            pd.Series(ref_neighbor_weights, index=ref_neighbor_labels)
            .groupby(level=0, sort=False)
            .sum()
            .reindex(label_index, fill_value=0.0)
        )
        vote_fracs = counts.to_numpy(dtype=np.float64)
        vote_fracs = vote_fracs / vote_fracs.sum()
        top_rank = np.argsort(vote_fracs)[::-1]
        top1_code = int(top_rank[0])
        top2_code = int(top_rank[1]) if len(top_rank) > 1 else None
        top1_label = str(label_index[top1_code])
        top2_label = str(label_index[top2_code]) if top2_code is not None else None
        top1_frac = float(vote_fracs[top1_code])
        top2_frac = float(vote_fracs[top2_code]) if top2_code is not None else None

        if distance_graph is not None:
            dist_row = distance_graph.getrow(query_idx)
            dist_map = dict(zip(dist_row.indices.tolist(), dist_row.data.tolist(), strict=False))
            top1_dists = [dist_map[idx] for idx, lbl in zip(ref_neighbor_indices, ref_neighbor_labels, strict=False) if lbl == top1_label and idx in dist_map]
            top1_mean_distance = float(np.mean(top1_dists)) if top1_dists else fallback_top1_mean_distance
        else:
            top1_mean_distance = fallback_top1_mean_distance

        expected_labels = list(expected_lookup.get(query_labels[local_idx], []))
        expected_codes = [label_index.get_loc(lbl) for lbl in expected_labels if lbl in label_index]
        expected_best_vote_fraction = float(vote_fracs[expected_codes].max()) if expected_codes else None
        expected_contains_top1 = bool(top1_label in expected_labels) if expected_labels else None

        rows.append(
            {
                "cell_id": str(query_ids[local_idx]),
                "morph_label": str(query_labels[local_idx]),
                "top1_embryo_label": top1_label,
                "top1_vote_fraction": top1_frac,
                "top2_embryo_label": top2_label,
                "top2_vote_fraction": top2_frac,
                "top1_minus_top2": None if top2_frac is None else top1_frac - top2_frac,
                "top1_mean_distance": top1_mean_distance,
                "expected_embryo_labels": "|".join(expected_labels),
                "expected_best_vote_fraction": expected_best_vote_fraction,
                "expected_contains_top1": expected_contains_top1,
                "graph_degree_to_reference": int(len(ref_neighbor_indices)),
            }
        )

    return pd.DataFrame(rows)


def run_bbknn_integration(
    morph_adata,
    embryo_adata,
    genes: Sequence[str],
    *,
    morph_label_key: str,
    embryo_label_key: str,
    n_components: int = 30,
    neighbors_within_batch: int = 3,
    random_state: int = 0,
):
    import scanpy as sc
    import bbknn

    joint, gene_index = build_joint_integration_adata(
        morph_adata,
        embryo_adata,
        genes,
        morph_label_key=morph_label_key,
        embryo_label_key=embryo_label_key,
    )

    if sparse.issparse(joint.X):
        joint.X = joint.X.astype(np.float32).toarray()
    else:
        joint.X = np.asarray(joint.X, dtype=np.float32)

    sc.pp.scale(joint, zero_center=True)
    n_components_eff = int(min(n_components, joint.n_vars, joint.n_obs - 1))
    if n_components_eff < 2:
        raise ValueError("Not enough genes/cells for BBKNN PCA.")

    sc.tl.pca(joint, n_comps=n_components_eff, svd_solver="arpack")
    bbknn.bbknn(
        joint,
        batch_key="dataset",
        use_rep="X_pca",
        neighbors_within_batch=int(neighbors_within_batch),
        n_pcs=n_components_eff,
    )
    sc.tl.umap(joint, random_state=random_state)
    return joint, gene_index


def project_query_to_reference_knn(
    query_adata,
    reference_adata,
    genes: Sequence[str],
    *,
    reference_label_key: str,
    query_label_key: str,
    expected_by_query_label: Mapping[str, Sequence[str]] | None = None,
    n_components: int = 30,
    k: int = 25,
    scale: bool = True,
    random_state: int = 0,
    batch_size: int = 512,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gene_index = pd.Index(genes).astype(str)
    gene_index = (
        pd.Index(reference_adata.var_names.astype(str))
        .intersection(gene_index)
        .intersection(query_adata.var_names.astype(str))
    )
    if len(gene_index) == 0:
        raise ValueError("No shared genes available for query-to-reference projection.")

    x_ref = _dense_gene_matrix(reference_adata, gene_index)
    ref_means = x_ref.mean(axis=0)
    ref_stds = x_ref.std(axis=0, ddof=0)
    keep = np.isfinite(ref_stds) & (ref_stds > 0)
    if scale:
        x_ref = (x_ref[:, keep] - ref_means[keep]) / ref_stds[keep]
        kept_gene_index = gene_index[keep]
    else:
        kept_gene_index = gene_index

    n_components_eff = int(min(n_components, x_ref.shape[1], x_ref.shape[0] - 1))
    if n_components_eff < 2:
        raise ValueError("Not enough genes/cells for PCA projection.")

    pca = PCA(n_components=n_components_eff, svd_solver="randomized", random_state=random_state)
    ref_scores = pca.fit_transform(x_ref)

    ref_labels_series = reference_adata.obs[reference_label_key]
    if hasattr(ref_labels_series.dtype, "categories"):
        reference_label_order = list(ref_labels_series.cat.categories)
    else:
        reference_label_order = list(pd.Index(ref_labels_series.astype(str)).unique())
    reference_label_order = [str(x) for x in reference_label_order]
    ref_labels = ref_labels_series.astype(str).to_numpy()

    k_eff = int(min(k, ref_scores.shape[0]))
    ref_scores = np.asarray(ref_scores, dtype=np.float32)
    ref_sq_norm = np.sum(ref_scores * ref_scores, axis=1, dtype=np.float32)

    label_index = pd.Index(reference_label_order)
    query_labels = query_adata.obs[query_label_key].astype(str).to_numpy()
    query_ids = query_adata.obs_names.astype(str).to_numpy()
    expected_lookup = expected_by_query_label or {}
    rows: list[pd.DataFrame] = []
    query_score_rows: list[pd.DataFrame] = []

    for start in range(0, query_adata.n_obs, batch_size):
        stop = min(start + batch_size, query_adata.n_obs)
        x_query = _dense_gene_matrix(query_adata[start:stop], kept_gene_index)
        if scale:
            x_query = (x_query - ref_means[keep]) / ref_stds[keep]

        query_scores = np.asarray(pca.transform(x_query), dtype=np.float32)
        query_sq_norm = np.sum(query_scores * query_scores, axis=1, dtype=np.float32)[:, None]
        dist_sq = query_sq_norm + ref_sq_norm[None, :] - (2.0 * (query_scores @ ref_scores.T))
        np.maximum(dist_sq, 0.0, out=dist_sq)

        idx_part = np.argpartition(dist_sq, kth=k_eff - 1, axis=1)[:, :k_eff]
        dist_sq_part = np.take_along_axis(dist_sq, idx_part, axis=1)
        sort_order = np.argsort(dist_sq_part, axis=1)
        indices = np.take_along_axis(idx_part, sort_order, axis=1)
        distances = np.sqrt(np.take_along_axis(dist_sq_part, sort_order, axis=1)).astype(np.float32, copy=False)
        neighbor_labels = ref_labels[indices]
        neighbor_codes = label_index.get_indexer(neighbor_labels.ravel()).reshape(neighbor_labels.shape)

        top1_labels: list[str] = []
        top2_labels: list[str | None] = []
        top1_vote_fracs: list[float] = []
        top2_vote_fracs: list[float | None] = []
        top1_minus_top2: list[float | None] = []
        top1_mean_distances: list[float] = []
        expected_best_vote_fracs: list[float | None] = []
        expected_contains_top1: list[bool | None] = []
        expected_labels_joined: list[str] = []

        for row_idx in range(neighbor_codes.shape[0]):
            counts = np.bincount(neighbor_codes[row_idx], minlength=len(label_index)).astype(np.float64)
            vote_fracs = counts / counts.sum()
            top_rank = np.argsort(vote_fracs)[::-1]
            top1_code = int(top_rank[0])
            top2_code = int(top_rank[1]) if len(top_rank) > 1 else None
            top1_label = str(label_index[top1_code])
            top2_label = str(label_index[top2_code]) if top2_code is not None else None
            top1_frac = float(vote_fracs[top1_code])
            top2_frac = float(vote_fracs[top2_code]) if top2_code is not None else None

            top1_neighbor_mask = neighbor_codes[row_idx] == top1_code
            top1_mean_distance = float(distances[row_idx, top1_neighbor_mask].mean())

            expected_labels = list(expected_lookup.get(query_labels[start + row_idx], []))
            expected_labels_joined.append("|".join(expected_labels))
            expected_codes = [label_index.get_loc(lbl) for lbl in expected_labels if lbl in label_index]
            if expected_codes:
                expected_best_vote_fracs.append(float(vote_fracs[expected_codes].max()))
                expected_contains_top1.append(bool(top1_label in expected_labels))
            else:
                expected_best_vote_fracs.append(None)
                expected_contains_top1.append(None)

            top1_labels.append(top1_label)
            top2_labels.append(top2_label)
            top1_vote_fracs.append(top1_frac)
            top2_vote_fracs.append(top2_frac)
            top1_minus_top2.append(None if top2_frac is None else top1_frac - top2_frac)
            top1_mean_distances.append(top1_mean_distance)

        batch_assignments = pd.DataFrame(
            {
                "cell_id": query_ids[start:stop],
                "morph_label": query_labels[start:stop],
                "top1_embryo_label": top1_labels,
                "top1_vote_fraction": top1_vote_fracs,
                "top2_embryo_label": top2_labels,
                "top2_vote_fraction": top2_vote_fracs,
                "top1_minus_top2": top1_minus_top2,
                "top1_mean_distance": top1_mean_distances,
                "expected_embryo_labels": expected_labels_joined,
                "expected_best_vote_fraction": expected_best_vote_fracs,
                "expected_contains_top1": expected_contains_top1,
                "n_genes": int(len(kept_gene_index)),
                "n_components": n_components_eff,
                "k_neighbors": k_eff,
            }
        )
        rows.append(batch_assignments)

        batch_scores = pd.DataFrame(
            query_scores,
            index=query_ids[start:stop],
            columns=[f"PC{i+1}" for i in range(query_scores.shape[1])],
        )
        batch_scores.insert(0, "morph_label", query_labels[start:stop])
        query_score_rows.append(batch_scores.reset_index(names="cell_id"))

    assignment_df = pd.concat(rows, axis=0, ignore_index=True)
    query_scores_df = pd.concat(query_score_rows, axis=0, ignore_index=True)
    reference_scores_df = pd.DataFrame(
        ref_scores,
        index=reference_adata.obs_names.astype(str),
        columns=[f"PC{i+1}" for i in range(ref_scores.shape[1])],
    )
    reference_scores_df.insert(0, "embryo_label", ref_labels)
    reference_scores_df = reference_scores_df.reset_index(names="cell_id")
    return assignment_df, query_scores_df, reference_scores_df


def project_query_with_scanpy_ingest(
    query_adata,
    reference_adata,
    genes: Sequence[str],
    *,
    reference_label_key: str,
    query_label_key: str,
    expected_by_query_label: Mapping[str, Sequence[str]] | None = None,
    n_components: int = 30,
    k: int = 25,
    scale: bool = True,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import anndata as ad
    import scanpy as sc

    gene_index = pd.Index(genes).astype(str)
    gene_index = (
        pd.Index(reference_adata.var_names.astype(str))
        .intersection(gene_index)
        .intersection(query_adata.var_names.astype(str))
    )
    if len(gene_index) == 0:
        raise ValueError("No shared genes available for Scanpy ingest.")

    x_ref = _dense_gene_matrix(reference_adata, gene_index)
    x_query = _dense_gene_matrix(query_adata, gene_index)
    ref_means = x_ref.mean(axis=0)
    ref_stds = x_ref.std(axis=0, ddof=0)
    keep = np.isfinite(ref_stds) & (ref_stds > 0)
    if not np.any(keep):
        raise ValueError("No non-constant genes available for Scanpy ingest.")

    if scale:
        x_ref = (x_ref[:, keep] - ref_means[keep]) / ref_stds[keep]
        x_query = (x_query[:, keep] - ref_means[keep]) / ref_stds[keep]
        kept_gene_index = gene_index[keep]
    else:
        x_ref = x_ref[:, keep]
        x_query = x_query[:, keep]
        kept_gene_index = gene_index[keep]

    n_components_eff = int(min(n_components, x_ref.shape[1], x_ref.shape[0] - 1))
    if n_components_eff < 2:
        raise ValueError("Not enough genes/cells for Scanpy ingest PCA.")

    k_eff = int(min(k, x_ref.shape[0]))

    ref = ad.AnnData(
        X=np.asarray(x_ref, dtype=np.float32),
        obs=reference_adata.obs[[reference_label_key]].copy(),
        var=pd.DataFrame(index=kept_gene_index.astype(str)),
    )
    query = ad.AnnData(
        X=np.asarray(x_query, dtype=np.float32),
        obs=query_adata.obs[[query_label_key]].copy(),
        var=pd.DataFrame(index=kept_gene_index.astype(str)),
    )

    sc.tl.pca(ref, n_comps=n_components_eff, svd_solver="arpack")
    sc.pp.neighbors(ref, n_neighbors=k_eff, n_pcs=n_components_eff)
    sc.tl.umap(ref, random_state=random_state)
    sc.tl.ingest(
        query,
        ref,
        obs=reference_label_key,
        embedding_method=("pca", "umap"),
        labeling_method="knn",
        inplace=True,
    )

    ref_scores = np.asarray(ref.obsm["X_pca"], dtype=np.float32)
    query_scores = np.asarray(query.obsm["X_pca"], dtype=np.float32)
    ref_sq_norm = np.sum(ref_scores * ref_scores, axis=1, dtype=np.float32)
    query_sq_norm = np.sum(query_scores * query_scores, axis=1, dtype=np.float32)[:, None]
    dist_sq = query_sq_norm + ref_sq_norm[None, :] - (2.0 * (query_scores @ ref_scores.T))
    np.maximum(dist_sq, 0.0, out=dist_sq)

    idx_part = np.argpartition(dist_sq, kth=k_eff - 1, axis=1)[:, :k_eff]
    dist_sq_part = np.take_along_axis(dist_sq, idx_part, axis=1)
    sort_order = np.argsort(dist_sq_part, axis=1)
    indices = np.take_along_axis(idx_part, sort_order, axis=1)
    distances = np.sqrt(np.take_along_axis(dist_sq_part, sort_order, axis=1)).astype(
        np.float32,
        copy=False,
    )

    ref_labels_series = ref.obs[reference_label_key]
    if hasattr(ref_labels_series.dtype, "categories"):
        reference_label_order = [str(x) for x in ref_labels_series.cat.categories]
    else:
        reference_label_order = [str(x) for x in pd.Index(ref_labels_series.astype(str)).unique()]
    label_index = pd.Index(reference_label_order)
    ref_labels = ref_labels_series.astype(str).to_numpy()
    neighbor_labels = ref_labels[indices]
    neighbor_codes = label_index.get_indexer(neighbor_labels.ravel()).reshape(neighbor_labels.shape)

    query_ids = query_adata.obs_names.astype(str).to_numpy()
    query_labels = query_adata.obs[query_label_key].astype(str).to_numpy()
    ingest_labels = query.obs[reference_label_key].astype(str).to_numpy()
    expected_lookup = expected_by_query_label or {}

    rows: list[dict[str, object]] = []
    for row_idx in range(query.n_obs):
        counts = np.bincount(neighbor_codes[row_idx], minlength=len(label_index)).astype(np.float64)
        vote_fracs = counts / counts.sum()
        vote_rank = np.argsort(vote_fracs)[::-1]
        vote_top1_code = int(vote_rank[0])
        vote_top1_label = str(label_index[vote_top1_code])

        ingest_label = ingest_labels[row_idx]
        if ingest_label in label_index:
            ingest_code = int(label_index.get_loc(ingest_label))
            ingest_vote_fraction = float(vote_fracs[ingest_code])
            other_codes = [code for code in vote_rank if code != ingest_code]
            if other_codes:
                top2_code = int(other_codes[0])
                top2_label = str(label_index[top2_code])
                top2_vote_fraction = float(vote_fracs[top2_code])
            else:
                top2_code = None
                top2_label = None
                top2_vote_fraction = None
            ingest_neighbor_mask = neighbor_codes[row_idx] == ingest_code
            top1_mean_distance = (
                float(distances[row_idx, ingest_neighbor_mask].mean())
                if np.any(ingest_neighbor_mask)
                else np.nan
            )
        else:
            ingest_code = -1
            ingest_vote_fraction = np.nan
            top2_code = vote_top1_code
            top2_label = vote_top1_label
            top2_vote_fraction = float(vote_fracs[vote_top1_code])
            top1_mean_distance = np.nan

        expected_labels = list(expected_lookup.get(query_labels[row_idx], []))
        expected_codes = [label_index.get_loc(lbl) for lbl in expected_labels if lbl in label_index]
        if expected_codes:
            expected_best_vote_fraction = float(vote_fracs[expected_codes].max())
            expected_contains_top1 = bool(ingest_label in expected_labels)
        else:
            expected_best_vote_fraction = None
            expected_contains_top1 = None

        rows.append(
            {
                "cell_id": query_ids[row_idx],
                "morph_label": query_labels[row_idx],
                "top1_embryo_label": ingest_label,
                "top1_vote_fraction": ingest_vote_fraction,
                "top2_embryo_label": top2_label,
                "top2_vote_fraction": top2_vote_fraction,
                "top1_minus_top2": (
                    None
                    if top2_vote_fraction is None or not np.isfinite(ingest_vote_fraction)
                    else float(ingest_vote_fraction - top2_vote_fraction)
                ),
                "top1_mean_distance": top1_mean_distance,
                "vote_top1_embryo_label": vote_top1_label,
                "ingest_matches_vote_top1": bool(ingest_label == vote_top1_label),
                "expected_embryo_labels": "|".join(expected_labels),
                "expected_best_vote_fraction": expected_best_vote_fraction,
                "expected_contains_top1": expected_contains_top1,
                "n_genes": int(len(kept_gene_index)),
                "n_components": n_components_eff,
                "k_neighbors": k_eff,
            }
        )

    assignment_df = pd.DataFrame(rows)
    query_scores_df = pd.DataFrame(
        query_scores,
        index=query_ids,
        columns=[f"PC{i+1}" for i in range(query_scores.shape[1])],
    )
    query_scores_df.insert(0, "morph_label", query_labels)
    query_scores_df.insert(1, "ingest_embryo_label", ingest_labels)
    query_scores_df = query_scores_df.reset_index(names="cell_id")

    reference_scores_df = pd.DataFrame(
        ref_scores,
        index=ref.obs_names.astype(str),
        columns=[f"PC{i+1}" for i in range(ref_scores.shape[1])],
    )
    reference_scores_df.insert(0, "embryo_label", ref_labels)
    reference_scores_df = reference_scores_df.reset_index(names="cell_id")

    query_umap_df = pd.DataFrame(
        np.asarray(query.obsm["X_umap"], dtype=np.float32),
        index=query_ids,
        columns=["UMAP1", "UMAP2"],
    )
    query_umap_df.insert(0, "morph_label", query_labels)
    query_umap_df.insert(1, "ingest_embryo_label", ingest_labels)
    query_umap_df = query_umap_df.reset_index(names="cell_id")

    reference_umap_df = pd.DataFrame(
        np.asarray(ref.obsm["X_umap"], dtype=np.float32),
        index=ref.obs_names.astype(str),
        columns=["UMAP1", "UMAP2"],
    )
    reference_umap_df.insert(0, "embryo_label", ref_labels)
    reference_umap_df = reference_umap_df.reset_index(names="cell_id")
    return (
        assignment_df,
        query_scores_df,
        reference_scores_df,
        query_umap_df,
        reference_umap_df,
    )


def summarize_projection_mapping(
    assignment_df: pd.DataFrame,
    *,
    morph_order: Sequence[str] | None = None,
    embryo_order: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if assignment_df.empty:
        raise ValueError("Assignment dataframe is empty.")

    cluster_summary = (
        assignment_df.groupby("morph_label", observed=True)
        .apply(
            lambda df: pd.Series(
                {
                    "n_cells": int(len(df)),
                    "top_assigned_embryo_label": df["top1_embryo_label"].value_counts().idxmax(),
                    "top_assigned_fraction": float(df["top1_embryo_label"].value_counts(normalize=True).iloc[0]),
                    "expected_assignment_rate": float(df["expected_contains_top1"].dropna().mean()),
                    "median_top1_vote_fraction": float(df["top1_vote_fraction"].median()),
                    "median_top1_minus_top2": float(df["top1_minus_top2"].median()),
                    "median_top1_mean_distance": float(df["top1_mean_distance"].median()),
                }
            )
        )
        .reset_index()
    )
    if morph_order is not None:
        cluster_summary["morph_label"] = pd.Categorical(
            cluster_summary["morph_label"],
            categories=list(morph_order),
            ordered=True,
        )
        cluster_summary = cluster_summary.sort_values("morph_label").reset_index(drop=True)

    assignment_fraction = pd.crosstab(
        assignment_df["morph_label"],
        assignment_df["top1_embryo_label"],
        normalize="index",
    )
    if morph_order is not None:
        assignment_fraction = assignment_fraction.reindex(list(morph_order))
    if embryo_order is not None:
        assignment_fraction = assignment_fraction.reindex(columns=list(embryo_order), fill_value=0.0)
    assignment_fraction = assignment_fraction.fillna(0.0)

    overview = pd.DataFrame(
        [
            {
                "n_cells": int(len(assignment_df)),
                "micro_expected_assignment_rate": float(assignment_df["expected_contains_top1"].dropna().mean()),
                "macro_expected_assignment_rate": float(cluster_summary["expected_assignment_rate"].dropna().mean()),
                "median_top1_vote_fraction": float(assignment_df["top1_vote_fraction"].median()),
                "median_top1_minus_top2": float(assignment_df["top1_minus_top2"].median()),
                "median_top1_mean_distance": float(assignment_df["top1_mean_distance"].median()),
            }
        ]
    )
    return cluster_summary, assignment_fraction, overview


def summarize_label_transfer_mapping(
    assignment_df: pd.DataFrame,
    *,
    morph_order: Sequence[str] | None = None,
    embryo_order: Sequence[str] | None = None,
    top1_score_col: str = "top1_vote_fraction",
    margin_col: str = "top1_minus_top2",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if assignment_df.empty:
        raise ValueError("Assignment dataframe is empty.")

    cluster_summary = (
        assignment_df.groupby("morph_label", observed=True)
        .apply(
            lambda df: pd.Series(
                {
                    "n_cells": int(len(df)),
                    "top_assigned_embryo_label": df["top1_embryo_label"].value_counts().idxmax(),
                    "top_assigned_fraction": float(df["top1_embryo_label"].value_counts(normalize=True).iloc[0]),
                    "expected_assignment_rate": float(df["expected_contains_top1"].dropna().mean()),
                    "median_top1_score": float(df[top1_score_col].dropna().median()),
                    "median_top1_minus_top2": float(df[margin_col].dropna().median()),
                }
            )
        )
        .reset_index()
    )
    if morph_order is not None:
        cluster_summary["morph_label"] = pd.Categorical(
            cluster_summary["morph_label"],
            categories=list(morph_order),
            ordered=True,
        )
        cluster_summary = cluster_summary.sort_values("morph_label").reset_index(drop=True)

    assignment_fraction = pd.crosstab(
        assignment_df["morph_label"],
        assignment_df["top1_embryo_label"],
        normalize="index",
    )
    if morph_order is not None:
        assignment_fraction = assignment_fraction.reindex(list(morph_order))
    if embryo_order is not None:
        assignment_fraction = assignment_fraction.reindex(columns=list(embryo_order), fill_value=0.0)
    assignment_fraction = assignment_fraction.fillna(0.0)

    overview = pd.DataFrame(
        [
            {
                "n_cells": int(len(assignment_df)),
                "micro_expected_assignment_rate": float(assignment_df["expected_contains_top1"].dropna().mean()),
                "macro_expected_assignment_rate": float(cluster_summary["expected_assignment_rate"].dropna().mean()),
                "median_top1_score": float(assignment_df[top1_score_col].dropna().median()),
                "median_top1_minus_top2": float(assignment_df[margin_col].dropna().median()),
            }
        ]
    )
    return cluster_summary, assignment_fraction, overview


def summarize_top_matches(
    corr_df: pd.DataFrame,
    *,
    expected_by_morph: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for morph_label in corr_df.index:
        ranked = corr_df.loc[morph_label].sort_values(ascending=False)
        top1_label = ranked.index[0]
        top1_corr = float(ranked.iloc[0])
        top2_label = ranked.index[1] if len(ranked) > 1 else None
        top2_corr = float(ranked.iloc[1]) if len(ranked) > 1 else None
        expected = list(expected_by_morph.get(morph_label, [])) if expected_by_morph else []
        expected_scores = {
            embryo_label: float(corr_df.loc[morph_label, embryo_label])
            for embryo_label in expected
            if embryo_label in corr_df.columns
        }
        expected_best_label = None
        expected_best_corr = None
        if expected_scores:
            expected_best_label = max(expected_scores, key=expected_scores.get)
            expected_best_corr = expected_scores[expected_best_label]

        rows.append(
            {
                "morph_label": morph_label,
                "top1_embryo_label": top1_label,
                "top1_corr": top1_corr,
                "top2_embryo_label": top2_label,
                "top2_corr": top2_corr,
                "top1_minus_top2": None if top2_corr is None else top1_corr - top2_corr,
                "expected_embryo_labels": "|".join(expected),
                "expected_best_label": expected_best_label,
                "expected_best_corr": expected_best_corr,
                "expected_contains_top1": top1_label in expected if expected else None,
            }
        )

    return pd.DataFrame(rows)


def summarize_column_best_matches(corr_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for embryo_label in corr_df.columns:
        ranked = corr_df[embryo_label].sort_values(ascending=False)
        top1_label = ranked.index[0]
        top1_corr = float(ranked.iloc[0])
        top2_label = ranked.index[1] if len(ranked) > 1 else None
        top2_corr = float(ranked.iloc[1]) if len(ranked) > 1 else None
        rows.append(
            {
                "embryo_label": embryo_label,
                "top1_morph_label": top1_label,
                "top1_corr": top1_corr,
                "top2_morph_label": top2_label,
                "top2_corr": top2_corr,
                "top1_minus_top2": None if top2_corr is None else top1_corr - top2_corr,
            }
        )
    return pd.DataFrame(rows)


def summarize_reciprocal_best_matches(
    corr_df: pd.DataFrame,
    *,
    expected_by_morph: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_top = summarize_top_matches(corr_df, expected_by_morph=expected_by_morph)
    col_top = summarize_column_best_matches(corr_df)
    col_lookup = col_top.set_index("embryo_label")

    rows: list[dict[str, object]] = []
    reciprocal_pairs: list[dict[str, object]] = []
    for _, row in row_top.iterrows():
        morph_label = str(row["morph_label"])
        embryo_label = str(row["top1_embryo_label"])
        embryo_top1_morph = None
        embryo_top1_corr = None
        embryo_top1_minus_top2 = None
        if embryo_label in col_lookup.index:
            embryo_row = col_lookup.loc[embryo_label]
            embryo_top1_morph = str(embryo_row["top1_morph_label"])
            embryo_top1_corr = float(embryo_row["top1_corr"])
            if pd.notna(embryo_row["top1_minus_top2"]):
                embryo_top1_minus_top2 = float(embryo_row["top1_minus_top2"])
        reciprocal_top1 = embryo_top1_morph == morph_label
        expected_contains_top1 = row["expected_contains_top1"]
        expected_reciprocal_top1 = (
            bool(reciprocal_top1 and expected_contains_top1)
            if pd.notna(expected_contains_top1)
            else None
        )

        row_dict = {
            "morph_label": morph_label,
            "top1_embryo_label": embryo_label,
            "top1_corr": float(row["top1_corr"]),
            "top1_minus_top2": (
                None if pd.isna(row["top1_minus_top2"]) else float(row["top1_minus_top2"])
            ),
            "expected_embryo_labels": row["expected_embryo_labels"],
            "expected_contains_top1": expected_contains_top1,
            "embryo_top1_morph_label": embryo_top1_morph,
            "embryo_top1_corr": embryo_top1_corr,
            "embryo_top1_minus_top2": embryo_top1_minus_top2,
            "reciprocal_top1": reciprocal_top1,
            "expected_reciprocal_top1": expected_reciprocal_top1,
        }
        rows.append(row_dict)

        if reciprocal_top1:
            reciprocal_pairs.append(
                {
                    "morph_label": morph_label,
                    "embryo_label": embryo_label,
                    "corr": float(row["top1_corr"]),
                    "morph_top1_minus_top2": (
                        None
                        if pd.isna(row["top1_minus_top2"])
                        else float(row["top1_minus_top2"])
                    ),
                    "embryo_top1_minus_top2": embryo_top1_minus_top2,
                    "expected_pair": expected_contains_top1,
                }
            )

    reciprocal_df = pd.DataFrame(rows)
    reciprocal_pairs_df = pd.DataFrame(reciprocal_pairs)
    return reciprocal_df, reciprocal_pairs_df


def invert_expected_mapping(
    expected_by_morph: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    embryo_to_morph: dict[str, list[str]] = {}
    for morph_label, embryo_labels in expected_by_morph.items():
        for embryo_label in embryo_labels:
            embryo_to_morph.setdefault(str(embryo_label), []).append(str(morph_label))
    return embryo_to_morph


def _aligned_cluster_matrices(
    left: pd.DataFrame,
    right: pd.DataFrame,
    genes: Sequence[str],
    *,
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_index = pd.Index(genes).astype(str)
    gene_index = left.columns.intersection(gene_index).intersection(right.columns)
    if len(gene_index) == 0:
        raise ValueError("No shared genes available for regression.")

    left_aligned = left[gene_index].copy()
    right_aligned = right[gene_index].copy()

    if not standardize:
        return left_aligned, right_aligned

    combined = pd.concat([left_aligned, right_aligned], axis=0)
    means = combined.mean(axis=0)
    stds = combined.std(axis=0, ddof=0)
    keep = stds > 0
    if not np.any(keep):
        raise ValueError("All candidate genes have zero variance across cluster averages.")

    left_scaled = (left_aligned.loc[:, keep] - means[keep]) / stds[keep]
    right_scaled = (right_aligned.loc[:, keep] - means[keep]) / stds[keep]
    return left_scaled, right_scaled


def nnls_cluster_regression(
    target_avg: pd.DataFrame,
    source_avg: pd.DataFrame,
    genes: Sequence[str],
    *,
    expected_by_target: Mapping[str, Sequence[str]] | None = None,
    target_label_name: str = "target",
    source_label_name: str = "source",
    standardize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_matrix, source_matrix = _aligned_cluster_matrices(
        target_avg,
        source_avg,
        genes,
        standardize=standardize,
    )

    design = source_matrix.to_numpy(dtype=float).T
    source_labels = pd.Index(source_matrix.index.astype(str))
    rows: list[dict[str, object]] = []
    weight_rows: list[pd.Series] = []

    for target_label in target_matrix.index.astype(str):
        target_vector = target_matrix.loc[target_label].to_numpy(dtype=float)
        coeffs, residual_l2 = nnls(design, target_vector)
        coeff_sum = float(coeffs.sum())
        weights = coeffs / coeff_sum if coeff_sum > 0 else coeffs
        reconstruction = design @ coeffs

        ss_res = float(np.square(target_vector - reconstruction).sum())
        ss_tot = float(np.square(target_vector - target_vector.mean()).sum())
        fit_r2 = np.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
        if np.std(target_vector) == 0 or np.std(reconstruction) == 0:
            fit_corr = np.nan
        else:
            fit_corr = float(np.corrcoef(target_vector, reconstruction)[0, 1])

        ranked_idx = np.argsort(weights)[::-1]
        top1_idx = int(ranked_idx[0])
        top2_idx = int(ranked_idx[1]) if len(ranked_idx) > 1 else None
        top1_label = str(source_labels[top1_idx])
        top1_weight = float(weights[top1_idx])
        top2_label = str(source_labels[top2_idx]) if top2_idx is not None else None
        top2_weight = float(weights[top2_idx]) if top2_idx is not None else None

        expected_labels = (
            [str(x) for x in expected_by_target.get(target_label, [])]
            if expected_by_target
            else []
        )
        expected_mask = source_labels.isin(expected_labels)
        expected_weight_sum = (
            float(weights[expected_mask].sum()) if expected_labels else None
        )
        expected_top_weight = (
            float(weights[expected_mask].max()) if np.any(expected_mask) else None
        )

        rows.append(
            {
                f"{target_label_name}_label": target_label,
                f"top1_{source_label_name}_label": top1_label,
                "top1_weight": top1_weight,
                f"top2_{source_label_name}_label": top2_label,
                "top2_weight": top2_weight,
                "top1_minus_top2": (
                    None if top2_weight is None else top1_weight - top2_weight
                ),
                f"expected_{source_label_name}_labels": "|".join(expected_labels),
                "expected_weight_sum": expected_weight_sum,
                "expected_top_weight": expected_top_weight,
                "expected_contains_top1": top1_label in expected_labels if expected_labels else None,
                "fit_corr": fit_corr,
                "fit_r2": fit_r2,
                "residual_l2": float(residual_l2),
                "n_genes": int(target_matrix.shape[1]),
            }
        )
        weight_rows.append(pd.Series(weights, index=source_labels, name=target_label))

    summary_df = pd.DataFrame(rows)
    weight_df = pd.DataFrame(weight_rows)
    weight_df.index.name = f"{target_label_name}_label"
    weight_df.columns.name = f"{source_label_name}_label"
    return weight_df, summary_df


def summarize_reciprocal_nnls(
    morph_from_embryo: pd.DataFrame,
    embryo_from_morph: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    embryo_lookup = embryo_from_morph.set_index("embryo_label")
    rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for _, row in morph_from_embryo.iterrows():
        morph_label = str(row["morph_label"])
        embryo_label = str(row["top1_embryo_label"])
        embryo_top1_morph = None
        embryo_top1_weight = None
        embryo_expected_weight_sum = None
        if embryo_label in embryo_lookup.index:
            embryo_row = embryo_lookup.loc[embryo_label]
            embryo_top1_morph = str(embryo_row["top1_morph_label"])
            embryo_top1_weight = float(embryo_row["top1_weight"])
            if pd.notna(embryo_row["expected_weight_sum"]):
                embryo_expected_weight_sum = float(embryo_row["expected_weight_sum"])

        reciprocal_top1 = embryo_top1_morph == morph_label
        expected_contains_top1 = row["expected_contains_top1"]
        expected_reciprocal_top1 = (
            bool(reciprocal_top1 and expected_contains_top1)
            if pd.notna(expected_contains_top1)
            else None
        )

        row_dict = {
            "morph_label": morph_label,
            "top1_embryo_label": embryo_label,
            "top1_weight": float(row["top1_weight"]),
            "top1_minus_top2": (
                None if pd.isna(row["top1_minus_top2"]) else float(row["top1_minus_top2"])
            ),
            "expected_embryo_labels": row["expected_embryo_labels"],
            "expected_weight_sum": (
                None if pd.isna(row["expected_weight_sum"]) else float(row["expected_weight_sum"])
            ),
            "expected_contains_top1": expected_contains_top1,
            "fit_corr": None if pd.isna(row["fit_corr"]) else float(row["fit_corr"]),
            "fit_r2": None if pd.isna(row["fit_r2"]) else float(row["fit_r2"]),
            "embryo_top1_morph_label": embryo_top1_morph,
            "embryo_top1_weight": embryo_top1_weight,
            "embryo_expected_weight_sum": embryo_expected_weight_sum,
            "reciprocal_top1": reciprocal_top1,
            "expected_reciprocal_top1": expected_reciprocal_top1,
        }
        rows.append(row_dict)

        if reciprocal_top1:
            pair_rows.append(
                {
                    "morph_label": morph_label,
                    "embryo_label": embryo_label,
                    "morph_top1_weight": float(row["top1_weight"]),
                    "embryo_top1_weight": embryo_top1_weight,
                    "morph_expected_weight_sum": (
                        None
                        if pd.isna(row["expected_weight_sum"])
                        else float(row["expected_weight_sum"])
                    ),
                    "embryo_expected_weight_sum": embryo_expected_weight_sum,
                    "fit_corr": None if pd.isna(row["fit_corr"]) else float(row["fit_corr"]),
                    "fit_r2": None if pd.isna(row["fit_r2"]) else float(row["fit_r2"]),
                    "expected_pair": expected_contains_top1,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    *,
    title: str,
    expected_by_morph: Mapping[str, Sequence[str]] | None = None,
    output_path: Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    n_rows, n_cols = corr_df.T.shape
    if figsize is None:
        matrix_width = max(4.8, 0.42 * n_cols)
        matrix_height = max(5.4, 0.30 * n_rows)
        figsize = (matrix_width + 7.6, matrix_height + 2.6)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    style = heatmap_style(corr_df)
    sns.heatmap(
        corr_df.T,
        cmap=style["cmap"],
        vmin=style["vmin"],
        vmax=style["vmax"],
        center=style["center"],
        linewidths=0.15,
        linecolor="#f0f0f0",
        cbar_kws={"label": "Pearson r", "shrink": 0.82, "pad": 0.02},
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Trunk Morph Cluster")
    ax.set_ylabel("Human Embryo Cluster")
    ax.set_xticklabels(
        wrapped_labels(compact_labels(corr_df.index, MORPH_DISPLAY_LABELS), width=12),
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )
    ax.set_yticklabels(
        wrapped_labels(compact_labels(corr_df.columns, EMBRYO_DISPLAY_LABELS), width=14),
        rotation=0,
        fontsize=10,
    )

    # Mark row-wise best matches with a dot.
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

    # Outline expected matches to make reviewer-facing interpretation explicit.
    if expected_by_morph:
        for morph_ix, morph_label in enumerate(corr_df.index):
            for embryo_label in expected_by_morph.get(morph_label, []):
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


def plot_assignment_heatmap(
    assignment_fraction: pd.DataFrame,
    *,
    title: str,
    expected_by_morph: Mapping[str, Sequence[str]] | None = None,
    output_path: Path | None = None,
    figsize: tuple[float, float] | None = None,
    cbar_label: str = "Assigned-cell fraction",
) -> plt.Figure:
    n_rows, n_cols = assignment_fraction.T.shape
    if figsize is None:
        matrix_width = max(4.8, 0.42 * n_cols)
        matrix_height = max(5.4, 0.30 * n_rows)
        figsize = (matrix_width + 8.6, matrix_height + 3.2)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sns.heatmap(
        assignment_fraction.T,
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        center=None,
        linewidths=0.15,
        linecolor="#f0f0f0",
        cbar_kws={"label": cbar_label, "shrink": 0.82, "pad": 0.02},
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Trunk Morph Cluster")
    ax.set_ylabel("Assigned Human Embryo Label")
    ax.set_xticklabels(
        wrapped_labels(compact_labels(assignment_fraction.index, MORPH_DISPLAY_LABELS), width=12),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=8.5,
    )
    ax.set_yticklabels(
        wrapped_labels(compact_labels(assignment_fraction.columns, EMBRYO_DISPLAY_LABELS), width=14),
        rotation=0,
        fontsize=9.5,
    )

    for morph_ix, morph_label in enumerate(assignment_fraction.index):
        best_embryo = assignment_fraction.loc[morph_label].idxmax()
        embryo_ix = assignment_fraction.columns.get_loc(best_embryo)
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
        for morph_ix, morph_label in enumerate(assignment_fraction.index):
            for embryo_label in expected_by_morph.get(morph_label, []):
                if embryo_label not in assignment_fraction.columns:
                    continue
                embryo_ix = assignment_fraction.columns.get_loc(embryo_label)
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

    fig.subplots_adjust(left=0.46, bottom=0.29, right=0.94, top=0.90)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    return fig


def plot_gene_set_grid(
    corr_results: Mapping[str, pd.DataFrame],
    *,
    title: str,
    expected_by_morph: Mapping[str, Sequence[str]] | None = None,
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

    for panel_ix, (ax, name) in enumerate(zip(axes.flat, names, strict=False)):
        corr_df = corr_results[name]
        row_ix = panel_ix // n_cols
        col_ix = panel_ix % n_cols
        xlabels = compact_labels(corr_df.index, MORPH_DISPLAY_LABELS)
        ylabels = compact_labels(corr_df.columns, EMBRYO_DISPLAY_LABELS)
        style = heatmap_style(corr_df)
        sns.heatmap(
            corr_df.T,
            cmap=style["cmap"],
            vmin=style["vmin"],
            vmax=style["vmax"],
            center=style["center"],
            cbar=False,
            linewidths=0.1,
            linecolor="#f0f0f0",
            xticklabels=xlabels,
            yticklabels=ylabels,
            ax=ax,
        )
        ax.set_title(name)
        show_x = row_ix == n_rows - 1
        show_y = col_ix == 0
        ax.set_xlabel("Morph" if show_x else "")
        ax.set_ylabel("Embryo" if show_y else "")
        ax.tick_params(
            axis="x",
            labelbottom=show_x,
            length=0,
            labelrotation=40,
            labelsize=9,
        )
        ax.tick_params(
            axis="y",
            labelleft=show_y,
            length=0,
            labelrotation=0,
            labelsize=9,
        )
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
                for embryo_label in expected_by_morph.get(morph_label, []):
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


def plot_assignment_grid(
    assignment_results: Mapping[str, pd.DataFrame],
    *,
    title: str,
    expected_by_morph: Mapping[str, Sequence[str]] | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    names = list(assignment_results.keys())
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

    heatmap_mappable = None
    for panel_ix, (ax, name) in enumerate(zip(axes.flat, names, strict=False)):
        assignment_df = assignment_results[name]
        row_ix = panel_ix // n_cols
        col_ix = panel_ix % n_cols
        xlabels = compact_labels(assignment_df.index, MORPH_DISPLAY_LABELS)
        ylabels = compact_labels(assignment_df.columns, EMBRYO_DISPLAY_LABELS)
        hm = sns.heatmap(
            assignment_df.T,
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            center=None,
            cbar=False,
            linewidths=0.1,
            linecolor="#f0f0f0",
            xticklabels=xlabels,
            yticklabels=ylabels,
            ax=ax,
        )
        if heatmap_mappable is None:
            heatmap_mappable = hm.collections[0]
        ax.set_title(name)
        show_x = row_ix == n_rows - 1
        show_y = col_ix == 0
        ax.set_xlabel("Morph" if show_x else "")
        ax.set_ylabel("Embryo" if show_y else "")
        ax.tick_params(axis="x", labelbottom=show_x, length=0, labelrotation=40, labelsize=9)
        ax.tick_params(axis="y", labelleft=show_y, length=0, labelrotation=0, labelsize=9)
        if show_x:
            for label in ax.get_xticklabels():
                label.set_ha("right")
                label.set_rotation_mode("anchor")

        for morph_ix, morph_label in enumerate(assignment_df.index):
            best_embryo = assignment_df.loc[morph_label].idxmax()
            embryo_ix = assignment_df.columns.get_loc(best_embryo)
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
            for morph_ix, morph_label in enumerate(assignment_df.index):
                for embryo_label in expected_by_morph.get(morph_label, []):
                    if embryo_label not in assignment_df.columns:
                        continue
                    embryo_ix = assignment_df.columns.get_loc(embryo_label)
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

    fig.tight_layout(rect=(0, 0, 0.94, 0.98))

    if heatmap_mappable is not None:
        cax = fig.add_axes([0.955, 0.18, 0.018, 0.62])
        cbar = fig.colorbar(
            heatmap_mappable,
            cax=cax,
        )
        cbar.set_label("Assigned-cell fraction")

    fig.suptitle(title, y=1.01)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    return fig


def plot_marker_score_heatmaps(
    morph_scores: pd.DataFrame,
    embryo_scores: pd.DataFrame,
    *,
    title: str,
    output_path: Path | None = None,
) -> plt.Figure:
    marker_sets = list(morph_scores.columns)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, max(6.2, 0.36 * len(marker_sets) + 3.8)),
        dpi=300,
        gridspec_kw={"width_ratios": [1, 1.15]},
    )

    combined_values = np.concatenate(
        [
            morph_scores.to_numpy(dtype=float).ravel(),
            embryo_scores.to_numpy(dtype=float).ravel(),
        ]
    )
    vmax = float(np.nanquantile(np.abs(combined_values), 0.98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    marker_labels = wrapped_labels(compact_labels(marker_sets, MORPH_DISPLAY_LABELS), width=14)

    sns.heatmap(
        morph_scores.T,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        center=0,
        linewidths=0.15,
        linecolor="#f0f0f0",
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title("Trunk morph", pad=8)
    axes[0].set_xlabel("Morph cluster")
    axes[0].set_ylabel("Manual marker set")
    axes[0].set_xticks(np.arange(morph_scores.shape[0]) + 0.5)
    axes[0].set_yticks(np.arange(len(marker_labels)) + 0.5)
    axes[0].set_xticklabels(
        wrapped_labels(compact_labels(morph_scores.index, MORPH_DISPLAY_LABELS), width=12),
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )
    axes[0].set_yticklabels(marker_labels, rotation=0, fontsize=10)

    # Mark the row-wise maximum cluster for each marker set.
    for marker_ix, marker_set in enumerate(morph_scores.columns):
        best_cluster = morph_scores[marker_set].idxmax()
        cluster_ix = morph_scores.index.get_loc(best_cluster)
        axes[0].scatter(
            cluster_ix + 0.5,
            marker_ix + 0.5,
            s=22,
            facecolors="white",
            edgecolors="black",
            marker="o",
            linewidths=0.6,
            zorder=5,
        )

    hm = sns.heatmap(
        embryo_scores.T,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        center=0,
        linewidths=0.15,
        linecolor="#f0f0f0",
        cbar_kws={"label": "Marker-set score", "shrink": 0.82, "pad": 0.02},
        ax=axes[1],
    )
    axes[1].set_title("Human embryo", pad=8)
    axes[1].set_xlabel("Embryo cluster")
    axes[1].set_ylabel("")
    axes[1].set_xticks(np.arange(embryo_scores.shape[0]) + 0.5)
    axes[1].set_yticks(np.arange(len(marker_labels)) + 0.5)
    axes[1].set_xticklabels(
        wrapped_labels(compact_labels(embryo_scores.index, EMBRYO_DISPLAY_LABELS), width=14),
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )
    axes[1].set_yticklabels(marker_labels, rotation=0, fontsize=10)

    for marker_ix, marker_set in enumerate(embryo_scores.columns):
        best_cluster = embryo_scores[marker_set].idxmax()
        cluster_ix = embryo_scores.index.get_loc(best_cluster)
        axes[1].scatter(
            cluster_ix + 0.5,
            marker_ix + 0.5,
            s=22,
            facecolors="white",
            edgecolors="black",
            marker="o",
            linewidths=0.6,
            zorder=5,
        )

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    return fig
