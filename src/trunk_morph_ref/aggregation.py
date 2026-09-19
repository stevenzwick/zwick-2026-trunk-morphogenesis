"""Aggregation helpers for cluster-level summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse


def _group_index_and_codes(labels: pd.Series) -> tuple[pd.Index, np.ndarray]:
    """Return group index and integer codes with pandas-groupby-compatible ordering."""
    if isinstance(labels.dtype, pd.CategoricalDtype):
        categories = labels.cat.categories
        index = pd.CategoricalIndex(
            categories,
            categories=categories,
            ordered=labels.cat.ordered,
            name=labels.name,
        )
        return index, labels.cat.codes.to_numpy(copy=False)

    non_na = labels[labels.notna()]
    categories = pd.Index(sorted(pd.unique(non_na)), name=labels.name)
    code_map = {label: i for i, label in enumerate(categories)}
    codes = labels.map(code_map).fillna(-1).astype(np.int64).to_numpy()
    return categories, codes


def cluster_averages_sparse_safe(adata, cluster_key: str) -> pd.DataFrame:
    """Compute mean expression per cluster using matrix aggregation.

    This preserves sparse inputs without routing through pandas sparse groupby,
    which is much slower on large cell-by-gene matrices.
    """
    labels = adata.obs[cluster_key]
    group_index, codes = _group_index_and_codes(labels)
    n_groups = len(group_index)

    if n_groups == 0:
        return pd.DataFrame(index=group_index, columns=adata.var_names, dtype=float)

    valid_mask = codes >= 0
    x = adata.X[valid_mask]
    codes_valid = codes[valid_mask]
    n_valid = int(valid_mask.sum())

    membership = csr_matrix(
        (np.ones(n_valid, dtype=np.float64), (codes_valid, np.arange(n_valid))),
        shape=(n_groups, n_valid),
    )
    sums = membership @ x
    sums = sums.toarray() if issparse(sums) else np.asarray(sums, dtype=np.float64)

    counts = np.bincount(codes_valid, minlength=n_groups).astype(np.float64)
    means = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts[:, None], out=means, where=counts[:, None] != 0)

    return pd.DataFrame(means, index=group_index, columns=adata.var_names)


def cluster_averages_dense_legacy(adata, cluster_key: str) -> pd.DataFrame:
    """Compute mean expression per cluster via legacy dense conversion."""
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    df = pd.DataFrame(x, index=adata.obs_names, columns=adata.var_names)
    df[cluster_key] = adata.obs[cluster_key].values
    return df.groupby(cluster_key, observed=False).mean()
