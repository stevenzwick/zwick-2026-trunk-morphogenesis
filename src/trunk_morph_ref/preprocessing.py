"""Shared preprocessing and gene-indexing helpers used across staged notebooks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

GENE_ID_FALLBACK_COLUMNS = (
    "gene_ids",
    "gene_id",
    "ensembl_id",
    "ensembl_gene_id",
)
GENE_SYMBOL_FALLBACK_COLUMNS = (
    "gene_symbol",
    "gene",
    "gene_name",
    "gene_names",
    "symbol",
    "feature_name",
    "feature",
)
NULL_LIKE_SYMBOLS = {"", "nan", "none", "<na>", "na"}


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str], exclude: set[str] | None = None):
    blocked = exclude or set()
    for column in candidates:
        if column in frame.columns and column not in blocked:
            return column
    return None


def _normalize_str_series(series: pd.Series, fallback: pd.Series | None = None) -> pd.Series:
    values = series.astype("string")
    if fallback is not None:
        values = values.fillna(fallback.astype("string"))
    values = values.fillna("")
    return values.astype(str)


def filter_genes_sparse_safe(adata, mean_cutoff, std_cutoff, print_output=True):
    """Filter genes by nonzero-count, mean, and std thresholds in adata.X."""
    counts = adata.X
    if sp.issparse(counts):
        nnz = np.asarray(counts.getnnz(axis=0)).ravel()
        means = np.asarray(counts.mean(axis=0)).ravel()
        mean_sq = np.asarray(counts.power(2).mean(axis=0)).ravel()
        stds = np.sqrt(np.clip(mean_sq - np.square(means), a_min=0.0, a_max=None))
    else:
        nnz = np.asarray(np.sum(counts > 0, axis=0)).ravel()
        means = np.asarray(np.mean(counts, axis=0)).ravel()
        stds = np.asarray(np.std(counts, axis=0)).ravel()

    keep_genes = np.logical_and(
        np.logical_and(nnz > 0, means > mean_cutoff),
        stds > std_cutoff,
    )
    adata_filtered = adata[:, keep_genes].copy()
    if print_output:
        print(str(adata_filtered.shape[1]) + " genes after filtering")
    return adata_filtered


def normalize_unit_variance_sparse_safe(adata):
    """Scale each gene to unit variance while preserving sparsity layout."""
    if adata.is_view:
        adata = adata.copy()
    counts = adata.X
    if sp.issparse(counts):
        means = np.asarray(counts.mean(axis=0)).ravel()
        mean_sq = np.asarray(counts.power(2).mean(axis=0)).ravel()
        stds = np.sqrt(np.clip(mean_sq - np.square(means), a_min=0.0, a_max=None))
        stds[stds == 0] = 1.0
        adata.X = counts.multiply(1.0 / stds).tocsr()
    else:
        stds = np.asarray(np.std(counts, axis=0)).ravel()
        stds[stds == 0] = 1.0
        adata.X = counts / stds
    return adata


def filter_genes_dense_legacy(adata, mean_cutoff, std_cutoff, print_output=True):
    """Legacy-compatible dense filter used for manuscript-parity runs."""
    counts = adata.X.todense() if sp.issparse(adata.X) else np.asarray(adata.X)
    keep_mean = np.mean(counts, axis=0).transpose() > mean_cutoff
    keep_std = np.std(counts, axis=0).transpose() > std_cutoff
    adata_filtered = adata[:, keep_mean & keep_std].copy()
    if print_output:
        print(str(adata_filtered.shape[1]) + " genes after filtering")
    return adata_filtered


def normalize_unit_variance_dense_legacy(adata):
    """Legacy-compatible dense normalization used for manuscript-parity runs."""
    counts = adata.X.todense() if sp.issparse(adata.X) else np.asarray(adata.X)
    std_counts = counts.std(axis=0)
    counts_norm = adata.X / std_counts
    adata_filtered = adata.copy()
    adata_filtered.X = sp.csr_matrix(counts_norm)
    return adata_filtered


def filter_genes(adata, mean_cutoff, std_cutoff, print_output=True):
    """Default filter path for active notebooks (legacy-dense parity)."""
    return filter_genes_dense_legacy(
        adata,
        mean_cutoff=mean_cutoff,
        std_cutoff=std_cutoff,
        print_output=print_output,
    )


def normalize_unit_variance(adata):
    """Default normalization path for active notebooks (legacy-dense parity)."""
    return normalize_unit_variance_dense_legacy(adata)


def _mean_axis0_array(matrix) -> np.ndarray:
    """Return dense 1D array of column means from dense or sparse matrix."""
    if sp.issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).ravel()
    return np.asarray(np.mean(matrix, axis=0)).ravel()


def build_legacy_smd_score_table(
    *,
    legacy_adata_path: str | Path,
    legacy_cluster_col: str,
    legacy_clusters: Sequence[str],
    zscore_path: str | Path,
    score_col: str,
    raw_adata=None,
    raw_adata_path: str | Path | None = None,
    minimum_reads: int = 5000,
    mean_cutoff: float = 0.05,
    std_cutoff: float = 0.05,
    cv_mean_cutoff: float = 1.0,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Recreate legacy pre-SMD gene ordering on a legacy cluster subset and attach z-scores.

    This is used when current cluster membership differs from legacy, causing direct
    z-score vector assignment to fail by length mismatch.
    """
    import anndata as ad  # local import to avoid hard dependency at module import time
    import scanpy as sc

    legacy_adata = ad.read_h5ad(Path(legacy_adata_path))
    if legacy_cluster_col not in legacy_adata.obs.columns:
        raise KeyError(
            f"Missing legacy cluster column `{legacy_cluster_col}` in "
            f"{legacy_adata_path}"
        )

    if raw_adata is None:
        if raw_adata_path is None:
            # Backward-compatible fallback when only a single legacy h5ad is available.
            raw_adata = legacy_adata.copy()
        else:
            raw_adata = ad.read_h5ad(Path(raw_adata_path))
    else:
        raw_adata = raw_adata.copy()

    # Transfer legacy cluster labels onto the raw-expression object by shared cell IDs.
    raw_adata.obs[legacy_cluster_col] = legacy_adata.obs[legacy_cluster_col].reindex(
        raw_adata.obs_names
    )
    raw_adata = raw_adata[raw_adata.obs[legacy_cluster_col].notna()].copy()
    raw_adata.obs[legacy_cluster_col] = raw_adata.obs[legacy_cluster_col].astype(str)

    legacy_subset = raw_adata[
        raw_adata.obs[legacy_cluster_col].isin([str(c) for c in legacy_clusters])
    ].copy()
    if legacy_subset.n_obs == 0:
        raise ValueError(
            "Legacy subset is empty for clusters: "
            + ", ".join([str(c) for c in legacy_clusters])
        )

    ensure_gene_id_index(legacy_subset)

    # Reproduce legacy preprocessing used before SMD.
    sc.pp.filter_cells(legacy_subset, min_counts=minimum_reads)
    sc.pp.downsample_counts(
        legacy_subset,
        counts_per_cell=minimum_reads,
        replace=True,
        random_state=random_state,
    )
    sc.pp.log1p(legacy_subset)
    legacy_filtered = filter_genes(
        legacy_subset,
        mean_cutoff=mean_cutoff,
        std_cutoff=std_cutoff,
        print_output=False,
    )
    legacy_filtered = normalize_unit_variance(legacy_filtered)
    legacy_cv_mask = _mean_axis0_array(legacy_filtered.X) < cv_mean_cutoff
    legacy_filtered = legacy_filtered[:, legacy_cv_mask]

    legacy_z_scores = np.load(Path(zscore_path))
    if len(legacy_z_scores) != legacy_filtered.n_vars:
        raise ValueError(
            f"Legacy z-score length mismatch for {zscore_path}: "
            f"{len(legacy_z_scores)} scores vs {legacy_filtered.n_vars} genes."
        )

    score_table = pd.DataFrame(
        {
            "gene_ids": legacy_filtered.var_names.astype(str),
            "gene_symbol": legacy_filtered.var["gene_symbol"].astype(str).to_numpy(),
            "gene_symbol_original": legacy_filtered.var["gene_symbol_original"]
            .astype(str)
            .to_numpy(),
            score_col: legacy_z_scores,
        }
    )
    score_table[f"log1p_{score_col}"] = np.log1p(score_table[score_col].to_numpy())
    return score_table


def map_legacy_smd_scores_to_adata(
    adata_filtered,
    legacy_score_table: pd.DataFrame,
    *,
    score_col: str,
) -> tuple:
    """
    Map legacy z-scores by gene_id to current filtered genes.

    Returns `(adata_mapped, diagnostics)`, where diagnostics includes counts and
    gene-id lists for unmatched directions.
    """
    assert_gene_id_index(adata_filtered)
    required = {"gene_ids", score_col}
    missing_cols = required - set(legacy_score_table.columns)
    if missing_cols:
        raise KeyError(
            f"Legacy score table missing required columns: {sorted(missing_cols)}"
        )

    legacy_scores = legacy_score_table.copy()
    legacy_scores["gene_ids"] = legacy_scores["gene_ids"].astype(str)
    legacy_scores = legacy_scores.drop_duplicates(subset=["gene_ids"], keep="first")

    legacy_score_by_gene_id = legacy_scores.set_index("gene_ids")[score_col]
    current_gene_ids = pd.Index(adata_filtered.var_names.astype(str))
    mapped_scores = legacy_score_by_gene_id.reindex(current_gene_ids)

    keep_mask = mapped_scores.notna().to_numpy()
    n_matched = int(keep_mask.sum())
    n_missing_in_current = int((~keep_mask).sum())
    if n_matched == 0:
        raise ValueError(
            f"No legacy `{score_col}` values matched current filtered genes."
        )

    missing_in_current_gene_ids = current_gene_ids[~keep_mask].tolist()
    legacy_only_gene_ids = (
        legacy_score_by_gene_id.index.difference(current_gene_ids).astype(str).tolist()
    )

    adata_mapped = adata_filtered[:, keep_mask].copy()
    mapped_scores = mapped_scores[keep_mask]
    adata_mapped.var[score_col] = mapped_scores.to_numpy()
    adata_mapped.var[f"log1p_{score_col}"] = np.log1p(
        adata_mapped.var[score_col].to_numpy()
    )

    diagnostics = {
        "n_matched": n_matched,
        "n_missing_in_current": n_missing_in_current,
        "n_legacy_only": int(len(legacy_only_gene_ids)),
        "missing_in_current_gene_ids": missing_in_current_gene_ids,
        "legacy_only_gene_ids": legacy_only_gene_ids,
    }
    return adata_mapped, diagnostics


def ensure_gene_id_index(
    adata,
    *,
    gene_id_col: str = "gene_ids",
    gene_symbol_col: str = "gene_symbol",
    gene_symbol_original_col: str = "gene_symbol_original",
) -> None:
    """
    Standardize var metadata so var_names are stable gene IDs.

    This function mutates `adata` in-place.
    """
    var = adata.var
    if gene_id_col not in var.columns:
        fallback_id_col = _first_existing_column(
            var,
            GENE_ID_FALLBACK_COLUMNS,
            exclude={gene_id_col},
        )
        if fallback_id_col is not None:
            var[gene_id_col] = _normalize_str_series(var[fallback_id_col])
        else:
            index_values = pd.Index(adata.var_names.astype(str))
            looks_like_ensembl = index_values.str.startswith("ENSG").all()
            if not looks_like_ensembl:
                raise KeyError(
                    "Could not infer gene ID column for adata.var. "
                    f"Expected one of {GENE_ID_FALLBACK_COLUMNS}."
                )
            var[gene_id_col] = index_values.astype(str)
    else:
        var[gene_id_col] = _normalize_str_series(var[gene_id_col])

    if var[gene_id_col].duplicated().any():
        duplicate_ids = var.loc[var[gene_id_col].duplicated(), gene_id_col].head(5).tolist()
        raise ValueError(
            f"{gene_id_col} contains duplicates; cannot use as stable var index. "
            f"Examples: {duplicate_ids}"
        )

    current_symbols = pd.Series(adata.var_names.astype(str), index=var.index)
    if gene_symbol_original_col in var.columns:
        var[gene_symbol_original_col] = _normalize_str_series(
            var[gene_symbol_original_col],
            fallback=current_symbols,
        )
    else:
        var[gene_symbol_original_col] = _normalize_str_series(current_symbols)

    if gene_symbol_col in var.columns:
        var[gene_symbol_col] = _normalize_str_series(
            var[gene_symbol_col],
            fallback=var[gene_symbol_original_col],
        )
    else:
        fallback_symbol_col = _first_existing_column(
            var,
            GENE_SYMBOL_FALLBACK_COLUMNS,
            exclude={gene_symbol_col, gene_symbol_original_col, gene_id_col},
        )
        if fallback_symbol_col is not None:
            var[gene_symbol_col] = _normalize_str_series(
                var[fallback_symbol_col],
                fallback=var[gene_symbol_original_col],
            )
        else:
            var[gene_symbol_col] = _normalize_str_series(var[gene_symbol_original_col])

    adata.var_names = pd.Index(var[gene_id_col].astype(str), name=gene_id_col)
    assert_gene_id_index(
        adata,
        gene_id_col=gene_id_col,
        gene_symbol_col=gene_symbol_col,
        gene_symbol_original_col=gene_symbol_original_col,
    )


def assert_gene_id_index(
    adata,
    *,
    gene_id_col: str = "gene_ids",
    gene_symbol_col: str = "gene_symbol",
    gene_symbol_original_col: str = "gene_symbol_original",
) -> None:
    """Validate expected var index invariants for stable gene-ID indexing."""
    if gene_id_col not in adata.var.columns:
        raise AssertionError(f"Missing required var column: {gene_id_col}")
    if gene_symbol_col not in adata.var.columns:
        raise AssertionError(f"Missing required var column: {gene_symbol_col}")
    if gene_symbol_original_col not in adata.var.columns:
        raise AssertionError(f"Missing required var column: {gene_symbol_original_col}")

    var_gene_ids = pd.Index(adata.var[gene_id_col].astype(str))
    index_gene_ids = pd.Index(adata.var_names.astype(str))

    if not index_gene_ids.is_unique:
        raise AssertionError("adata.var_names must be unique (expected unique gene_ids).")
    if not var_gene_ids.is_unique:
        raise AssertionError(f"adata.var[{gene_id_col}] must be unique.")
    if not index_gene_ids.equals(var_gene_ids):
        raise AssertionError(
            f"adata.var_names must exactly match adata.var[{gene_id_col}] in order and values."
        )


def _build_symbol_to_gene_ids(
    adata,
    *,
    gene_symbol_col: str,
    gene_symbol_original_col: str,
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    gene_ids = adata.var_names.astype(str)

    for column in (gene_symbol_col, gene_symbol_original_col):
        if column not in adata.var.columns:
            continue
        symbols = adata.var[column].astype(str)
        for gene_id, symbol in zip(gene_ids, symbols):
            key = symbol.strip()
            if key.lower() in NULL_LIKE_SYMBOLS:
                continue
            if gene_id not in mapping[key]:
                mapping[key].append(gene_id)
    return mapping


def resolve_symbols(
    adata,
    symbols: Sequence[str],
    *,
    strict: bool = True,
    allow_missing: bool = False,
    gene_symbol_col: str = "gene_symbol",
    gene_symbol_original_col: str = "gene_symbol_original",
) -> list[str]:
    """
    Resolve symbol or gene_id tokens to stable gene_id var_names.

    - Exact gene_id tokens are accepted directly.
    - Symbol lookup checks canonical + original symbol columns.
    - `strict=True` raises on symbol ambiguity.
    - `allow_missing=True` skips unknown symbols instead of raising.
    """
    assert_gene_id_index(
        adata,
        gene_symbol_col=gene_symbol_col,
        gene_symbol_original_col=gene_symbol_original_col,
    )

    symbol_to_gene_ids = _build_symbol_to_gene_ids(
        adata,
        gene_symbol_col=gene_symbol_col,
        gene_symbol_original_col=gene_symbol_original_col,
    )
    available_gene_ids = set(adata.var_names.astype(str))

    resolved: list[str] = []
    seen: set[str] = set()
    unresolved: list[str] = []
    ambiguous: list[tuple[str, list[str]]] = []

    for token in symbols:
        query = str(token).strip()
        if not query:
            continue

        if query in available_gene_ids:
            matches = [query]
        else:
            matches = symbol_to_gene_ids.get(query, [])

        if not matches:
            unresolved.append(query)
            continue

        if len(matches) > 1 and strict:
            ambiguous.append((query, matches))
            continue

        chosen = [matches[0]] if len(matches) > 1 else matches
        for gene_id in chosen:
            if gene_id not in seen:
                seen.add(gene_id)
                resolved.append(gene_id)

    if ambiguous:
        details = "; ".join(
            f"{symbol} -> {', '.join(ids[:3])}{'...' if len(ids) > 3 else ''}"
            for symbol, ids in ambiguous[:5]
        )
        raise ValueError(
            "Ambiguous gene symbols encountered in strict mode. "
            f"Examples: {details}"
        )

    if unresolved and not allow_missing:
        preview = ", ".join(unresolved[:10])
        suffix = "..." if len(unresolved) > 10 else ""
        raise KeyError(f"Unresolved gene symbols: {preview}{suffix}")

    return resolved


def resolve_symbol_dict(
    adata,
    symbol_groups: Mapping[str, Sequence[str]],
    *,
    strict: bool = True,
    allow_missing: bool = True,
    gene_symbol_col: str = "gene_symbol",
    gene_symbol_original_col: str = "gene_symbol_original",
) -> dict[str, list[str]]:
    """Resolve a mapping of named symbol lists to gene_id lists."""
    resolved: dict[str, list[str]] = {}
    for key, symbols in symbol_groups.items():
        gene_ids = resolve_symbols(
            adata,
            symbols,
            strict=strict,
            allow_missing=allow_missing,
            gene_symbol_col=gene_symbol_col,
            gene_symbol_original_col=gene_symbol_original_col,
        )
        if gene_ids:
            resolved[key] = gene_ids
    return resolved


def contains_symbol(
    adata,
    symbol: str,
    *,
    gene_symbol_col: str = "gene_symbol",
    gene_symbol_original_col: str = "gene_symbol_original",
) -> bool:
    """Return True if a gene symbol (or gene_id) resolves in adata."""
    return bool(
        resolve_symbols(
            adata,
            [symbol],
            strict=False,
            allow_missing=True,
            gene_symbol_col=gene_symbol_col,
            gene_symbol_original_col=gene_symbol_original_col,
        )
    )


def symbols_present(
    adata,
    symbols: Sequence[str],
    *,
    gene_symbol_col: str = "gene_symbol",
    gene_symbol_original_col: str = "gene_symbol_original",
) -> list[str]:
    """Return input symbols that resolve to at least one gene_id."""
    present: list[str] = []
    for symbol in symbols:
        if contains_symbol(
            adata,
            symbol,
            gene_symbol_col=gene_symbol_col,
            gene_symbol_original_col=gene_symbol_original_col,
        ):
            present.append(symbol)
    return present


def symbols_missing(
    adata,
    symbols: Sequence[str],
    *,
    gene_symbol_col: str = "gene_symbol",
    gene_symbol_original_col: str = "gene_symbol_original",
) -> list[str]:
    """Return input symbols that do not resolve to any gene_id."""
    missing: list[str] = []
    for symbol in symbols:
        if not contains_symbol(
            adata,
            symbol,
            gene_symbol_col=gene_symbol_col,
            gene_symbol_original_col=gene_symbol_original_col,
        ):
            missing.append(symbol)
    return missing


def var_names_to_symbols(
    adata,
    var_names: Sequence[str] | None = None,
    *,
    gene_symbol_col: str = "gene_symbol",
) -> list[str]:
    """Map gene_id var_names to display symbols, preserving order."""
    if var_names is None:
        var_names = adata.var_names.astype(str).tolist()
    else:
        var_names = [str(name) for name in var_names]

    if gene_symbol_col not in adata.var.columns:
        return var_names

    lookup = adata.var[gene_symbol_col].astype(str)
    symbols = lookup.reindex(var_names)
    fallback = pd.Series(var_names, index=var_names, dtype="string")
    symbols = _normalize_str_series(symbols, fallback=fallback)
    return symbols.tolist()


def install_scanpy_symbol_defaults(sc_module, *, symbol_col: str = "gene_symbol") -> None:
    """
    Wrap scanpy plotting helpers to default `gene_symbols=symbol_col`.

    This keeps symbol-based plotting calls working after migrating var_names to gene_ids.
    """

    def _flatten_var_names(var_names) -> list[str]:
        if var_names is None:
            return []
        if isinstance(var_names, str):
            return [var_names]
        if isinstance(var_names, Mapping):
            flattened: list[str] = []
            for values in var_names.values():
                flattened.extend(_flatten_var_names(values))
            return flattened
        flattened: list[str] = []
        for value in var_names:
            if isinstance(value, str):
                flattened.append(value)
            elif isinstance(value, Mapping):
                flattened.extend(_flatten_var_names(value))
            else:
                flattened.append(str(value))
        return flattened

    def _gene_id_to_symbol_map(adata) -> dict[str, str]:
        if adata is None or symbol_col not in adata.var.columns:
            return {}
        gene_ids = adata.var_names.astype(str)
        symbols = _normalize_str_series(
            adata.var[symbol_col],
            fallback=pd.Series(gene_ids, index=adata.var.index),
        )
        return {
            gene_id: symbol if symbol.strip().lower() not in NULL_LIKE_SYMBOLS else gene_id
            for gene_id, symbol in zip(gene_ids, symbols.astype(str))
        }

    def _set_axis_labels(ax, axis: str, labels: list[str]) -> None:
        if axis == "x":
            locs = ax.get_xticks()
            old = ax.get_xticklabels()
            ax.set_xticks(locs)
            ax.set_xticklabels(labels)
            new = ax.get_xticklabels()
        else:
            locs = ax.get_yticks()
            old = ax.get_yticklabels()
            ax.set_yticks(locs)
            ax.set_yticklabels(labels)
            new = ax.get_yticklabels()

        if not old:
            return
        template = old[0]
        for label in new:
            label.set_rotation(template.get_rotation())
            label.set_ha(template.get_ha())
            label.set_va(template.get_va())

    def _relabel_heatmap_ticks_to_symbols(result, adata) -> None:
        if not isinstance(result, Mapping):
            return
        heatmap_ax = result.get("heatmap_ax")
        if heatmap_ax is None:
            return

        lookup = _gene_id_to_symbol_map(adata)
        if not lookup:
            return

        for axis in ("x", "y"):
            if axis == "x":
                current = [tick.get_text() for tick in heatmap_ax.get_xticklabels()]
            else:
                current = [tick.get_text() for tick in heatmap_ax.get_yticklabels()]
            if not current:
                continue
            remapped = [lookup.get(token, token) for token in current]
            if remapped != current:
                _set_axis_labels(heatmap_ax, axis, remapped)

    for fn_name in ("heatmap", "dotplot", "stacked_violin", "matrixplot"):
        plot_fn = getattr(sc_module.pl, fn_name, None)
        if plot_fn is None:
            continue
        if getattr(plot_fn, "_trunk_symbol_wrapped", False):
            continue

        @wraps(plot_fn)
        def wrapped(*args, _fn=plot_fn, _fn_name=fn_name, **kwargs):
            adata = kwargs.get("adata")
            if adata is None and args:
                adata = args[0]

            var_names = kwargs.get("var_names")
            if var_names is None and len(args) > 1:
                var_names = args[1]

            tokens = _flatten_var_names(var_names)
            all_tokens_are_gene_ids = False

            if "gene_symbols" not in kwargs:
                if adata is not None and tokens:
                    var_name_index = set(adata.var_names.astype(str))
                    all_tokens_are_gene_ids = all(token in var_name_index for token in tokens)
                    if not all_tokens_are_gene_ids:
                        kwargs["gene_symbols"] = symbol_col

            result = _fn(*args, **kwargs)

            if (
                _fn_name == "heatmap"
                and adata is not None
                and tokens
                and all_tokens_are_gene_ids
            ):
                _relabel_heatmap_ticks_to_symbols(result, adata)

            return result

        wrapped._trunk_symbol_wrapped = True  # type: ignore[attr-defined]
        setattr(sc_module.pl, fn_name, wrapped)
