"""Correlation helpers that avoid densifying full cell-by-gene sparse matrices."""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse


def gene_corrcoef_sparse_safe(x):
    """Compute gene-gene correlation matrix (columns) with sparse-safe path.

    Equivalent target: np.corrcoef(x.todense().T) for sparse x.
    """
    if not issparse(x):
        return np.corrcoef(np.asarray(x).T)

    n_obs = x.shape[0]
    if n_obs < 2:
        raise ValueError("At least two observations are required for correlation")

    x = x.tocsr()
    sum_x = np.asarray(x.sum(axis=0)).ravel().astype(np.float64)
    mean_x = sum_x / n_obs

    # Dense gene-by-gene second-moment matrix only (p x p), not full dense cells-by-genes.
    xtx = (x.T @ x).toarray().astype(np.float64)
    cov = (xtx - n_obs * np.outer(mean_x, mean_x)) / (n_obs - 1)

    sumsq_x = np.asarray(x.power(2).sum(axis=0)).ravel().astype(np.float64)
    var_x = (sumsq_x - n_obs * np.square(mean_x)) / (n_obs - 1)
    var_x = np.maximum(var_x, 0.0)
    std_x = np.sqrt(var_x)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std_x, std_x)

    return corr


def gene_corrcoef_dense_legacy(x):
    """Compute gene-gene correlation matrix via legacy dense conversion."""
    if issparse(x):
        return np.corrcoef(x.todense().T)
    return np.corrcoef(np.asarray(x).T)
