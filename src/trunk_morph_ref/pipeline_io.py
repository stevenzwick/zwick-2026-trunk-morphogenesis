"""I/O helpers for staged notebook execution."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def stage_dir(results_dir: str | Path, stage_name: str) -> Path:
    """Return and create the intermediate directory for a pipeline stage."""
    root = Path(results_dir) / "intermediates" / stage_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_h5ad(adata, path: str | Path) -> None:
    """Write AnnData object to h5ad."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize_frame(df: pd.DataFrame) -> pd.DataFrame:
        fixed = df.copy()

        # Handle index-name/column-name collisions unsupported by h5ad.
        if (
            fixed.index.name is not None
            and fixed.index.name in fixed.columns
            and not pd.Series(fixed.index, index=fixed.index).equals(fixed[fixed.index.name])
        ):
            fixed.index.name = f"_{fixed.index.name}_index"

        # Coerce object columns to h5ad-safe types for intermediate serialization.
        for col in fixed.columns:
            series = fixed[col]
            if not pd.api.types.is_object_dtype(series):
                continue
            nonnull = series.dropna()
            if nonnull.empty:
                continue
            if nonnull.map(
                lambda v: isinstance(
                    v,
                    (
                        int,
                        float,
                        bool,
                        np.integer,
                        np.floating,
                        np.bool_,
                    ),
                )
            ).all():
                fixed[col] = pd.to_numeric(series, errors="coerce")
            else:
                fixed[col] = series.astype(str)

        return fixed

    def _write_once(target: Path) -> None:
        try:
            adata.write_h5ad(str(target))
            return
        except (ValueError, TypeError):
            pass

        adata_fixed = adata.copy()
        adata_fixed.obs = _sanitize_frame(adata_fixed.obs)
        adata_fixed.var = _sanitize_frame(adata_fixed.var)
        adata_fixed.write_h5ad(str(target))

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.stem}.",
        suffix=".tmp.h5ad",
        dir=str(path.parent),
    )
    os.close(tmp_fd)
    Path(tmp_name).unlink(missing_ok=True)
    try:
        _write_once(Path(tmp_name))
        Path(tmp_name).replace(path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def load_h5ad(path: str | Path):
    """Read AnnData object from h5ad."""
    return sc.read_h5ad(str(path))


def save_pickle(obj, path: str | Path) -> None:
    """Write a Python object via pickle."""
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path):
    """Read a Python object from pickle."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_npy(array, path: str | Path) -> None:
    """Write a NumPy array to .npy."""
    np.save(path, np.asarray(array))


def load_npy(path: str | Path):
    """Read a NumPy array from .npy."""
    return np.load(path, allow_pickle=False)


def save_json(payload: dict, path: str | Path) -> None:
    """Write a JSON payload with stable formatting."""
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
