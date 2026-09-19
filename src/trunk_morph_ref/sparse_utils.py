"""Sparse matrix helpers for memory-safe plotting utilities."""

from __future__ import annotations

from scipy.sparse import csr_matrix


def sparse_zero_matrix_like(matrix):
    """Return a CSR all-zero matrix with the same shape and dtype as input."""
    return csr_matrix(matrix.shape, dtype=matrix.dtype)
