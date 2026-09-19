"""Verify the single-cell panels against the paper's published Source Data.

Runs in two tiers:

* Shipped tier — Fig 4d, Fig 4e and ED Fig 8b are checkable from artifacts in this repository
  alone. Always runs.
* Object tier — the remaining panels need the saved AnnData objects. Set OBJECT_ROOT to the
  analysis root; without it these are reported as skipped, not failed.

    SOURCE_DATA_DIR=... OBJECT_ROOT=... python figures/verify_scrnaseq_panels.py

Objects are read with h5py and sliced by column; the expression matrix is never densified whole.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "figures"))
import verify_against_source_data as V             # noqa: E402
import scrnaseq_panels as S                        # noqa: E402

V.SOURCE_DATA_DIR = os.environ.get("SOURCE_DATA_DIR", str(REPO / "source_data"))
OBJECT_ROOT = os.environ.get("OBJECT_ROOT", "")

WORKBOOK = {
    "Fig.4a": "Source Data Fig.4.xlsx", "Fig.4b": "Source Data Fig.4.xlsx",
    "Fig.4c": "Source Data Fig.4.xlsx", "Fig.4e": "Source Data Fig.4.xlsx",
    "Fig.4f": "Source Data Fig.4.xlsx",
    "Extended Data Fig.6b": "Source Data Extended Data Fig.6.xlsx",
    "Extended Data Fig.8a": "Source Data Extended Data Fig.8.xlsx",
    "Extended Data Fig.8b": "Source Data Extended Data Fig.8.xlsx",
    "Extended Data Fig.9b": "Source Data Extended Data Fig.9.xlsx",
}


def sheet(name: str, block: int = 0) -> pd.DataFrame:
    df = V.load_blocks(WORKBOOK[name], name)[block][1]
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------- object access (h5py, read-only)
def _decode(values):
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values])


def obs_column(handle, name):
    import h5py
    node = handle["obs"][name]
    if isinstance(node, h5py.Group):
        return _decode(node["categories"][:])[node["codes"][:]]
    return _decode(node[:])


def obs_index(handle):
    key = handle["obs"].attrs.get("_index", "_index")
    return _decode(handle["obs"][key.decode() if isinstance(key, bytes) else key][:])


def gene_symbols(handle):
    import h5py
    node = handle["var"]["gene_symbol"]
    if isinstance(node, h5py.Group):
        return _decode(node["categories"][:])[node["codes"][:]]
    return _decode(node[:])


def gene_columns(path: Path, genes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (expression[cells, genes], barcodes, cluster labels) for named genes only."""
    import h5py
    from scipy.sparse import csr_matrix
    with h5py.File(path, "r") as fh:
        barcodes, symbols = obs_index(fh), gene_symbols(fh)
        labels = obs_column(fh, "leiden_morph")
        node = fh["X"]
        matrix = csr_matrix((node["data"][:], node["indices"][:], node["indptr"][:]),
                            shape=tuple(node.attrs["shape"])).tocsc()
    lookup: dict[str, int] = {}
    for i, sym in enumerate(symbols):
        lookup.setdefault(sym, i)
    missing = [g for g in genes if g not in lookup]
    if missing:
        raise KeyError(f"genes absent from object: {missing[:6]}")
    dense = np.asarray(matrix[:, [lookup[g] for g in genes]].todense(), dtype=float)
    return dense, barcodes, labels


def align(order: np.ndarray, barcodes: np.ndarray) -> np.ndarray:
    pos = pd.Series(np.arange(len(barcodes)), index=barcodes).reindex(order)
    if pos.isna().any():
        raise KeyError("Source Data barcodes missing from the object")
    return pos.to_numpy(int)


# ---------------------------------------------------------------------------------- shipped tier
def check_fig4e() -> dict:
    published = sheet("Fig.4e").set_index("Human embryo cluster").apply(pd.to_numeric, errors="coerce")
    stored = pd.read_csv(REPO / S.CORR_FIG4E, index_col=0)
    # The sheet is the transpose of the stored matrix: stored is morph x embryo.
    rebuilt = stored.loc[list(published.columns), list(published.index)].to_numpy(float).T
    diff = np.abs(published.to_numpy(float) - rebuilt)
    tol = V.tolerance_from_precision(published.iloc[:, 0].astype(str))
    return {"n": diff.size, "max_abs_diff": float(np.nanmax(diff)), "tolerance": tol,
            "pass": bool(np.nanmax(diff) <= tol)}


def check_ed8b() -> dict:
    cells, totals = sheet("Extended Data Fig.8a"), sheet("Extended Data Fig.8b")
    counts = pd.crosstab(cells["Cluster"], cells["Embryo"])
    worst = 0.0
    for cluster in totals["Cluster"]:
        for short in ("W3-1", "W4-1", "W4-2"):
            column = next(c for c in counts.columns if short in str(c))
            published = float(totals.loc[totals["Cluster"] == cluster,
                                         f"{short} Embryo - cell count"].iloc[0])
            worst = max(worst, abs(published - float(counts.loc[cluster, column])))
    return {"n": len(totals) * 3, "max_abs_diff": worst, "tolerance": 0.0, "pass": worst == 0}


def check_fig4d() -> dict:
    """No Source Data sheet — check internal consistency of the shipped heatmap artifacts."""
    order = pd.read_csv(REPO / S.HEATMAP_ORDER)
    counts = pd.read_csv(REPO / S.HEATMAP_COUNTS)
    regenerated = (order.groupby(["tissue", "leiden_mixed"]).size()
                   .rename("n_cells").reset_index())
    merged = counts.merge(regenerated, on=["tissue", "leiden_mixed"], how="outer",
                          suffixes=("_published", "_rebuilt"), indicator=True)
    # The outer join puts NaN on whichever side lacks a group, and `Series.max()` skips NaN, so a
    # group present in only one file would leave the difference at zero. Require every group to
    # appear on both sides, and take the maximum without skipping.
    unmatched = int((merged["_merge"] != "both").sum())
    diff = (merged["n_cells_published"] - merged["n_cells_rebuilt"]).abs()
    worst = float(diff.max(skipna=False))
    embryo_fp = counts[(counts["tissue"].str.contains("Embryo")) &
                       (counts["leiden_mixed"].isin(["Floor Plate", "Neuromesodermal Progenitors"]))]
    ok = bool(unmatched == 0 and worst == 0
              and len(embryo_fp) == 2 and (embryo_fp["n_cells"] > 0).all())
    extra = ("embryo cells present in Floor Plate and NMP" if ok
             else f"CHECK FAILED ({unmatched} groups on one side only)" if unmatched
             else "CHECK FAILED")
    return {"n": len(merged), "max_abs_diff": worst, "tolerance": 0.0, "pass": ok, "extra": extra}


# ----------------------------------------------------------------------------------- object tier
def check_fig4b() -> dict:
    import h5py
    published = sheet("Fig.4b")
    with h5py.File(S.object_path(S.MORPH_SMD), "r") as fh:
        barcodes, labels = obs_index(fh), obs_column(fh, "leiden_morph")
        umap = fh["obsm"]["X_umap"][:]
    idx = align(published["Cell barcode"].astype(str).to_numpy(), barcodes)
    agree = float((labels[idx] == published["Cluster"].astype(str).to_numpy()).mean())
    coords = np.abs(umap[idx][:, :2] -
                    published[["UMAP 1", "UMAP 2"]].apply(pd.to_numeric).to_numpy(float)).max()
    tol = (V.tolerance_from_precision(published["UMAP 1"].astype(str))
           + V.float32_allowance(np.abs(umap).max()))
    return {"n": len(published), "max_abs_diff": float(coords), "tolerance": tol,
            "pass": bool(agree == 1.0 and coords <= tol),
            "extra": f"cluster agreement {agree * 100:.2f}%"}


def check_fig4c() -> dict:
    import h5py
    published = sheet("Fig.4c")
    with h5py.File(S.object_path(S.MORPH_SMD), "r") as fh:
        labels, source = obs_column(fh, "leiden_morph"), obs_column(fh, "source")
    counts = pd.crosstab(pd.Series(labels), pd.Series(source))
    worst = 0.0
    for _, row in published.iterrows():
        for label, column in (("No Bead Morph", "No Bead - cell count"),
                              ("BMP4 Bead Morph", "BMP4 Bead - cell count")):
            worst = max(worst, abs(float(row[column]) - float(counts.loc[row["Cluster"], label])))
    return {"n": len(published) * 2, "max_abs_diff": worst, "tolerance": 0.0, "pass": worst == 0}


def _dotplot(sheet_name: str, value_column: str, normalise: bool) -> dict:
    published = sheet(sheet_name)
    genes = list(dict.fromkeys(published["Gene"].astype(str)))
    expression, barcodes, labels = gene_columns(S.object_path(S.MORPH_FULLGENE), genes)
    tables = {g: S.cluster_means(expression[:, i], labels) for i, g in enumerate(genes)}
    worst, count = 0.0, 0
    for _, row in published.iterrows():
        table = tables[str(row["Gene"])]
        cluster = str(row["Cluster"])
        value = table.loc[cluster, "mean"]
        if normalise:
            value = S.min_max_normalise(table["mean"])[cluster]
        worst = max(worst, abs(float(row[value_column]) - float(value)),
                    abs(float(row["Fraction expressing (dot size)"]) -
                        float(table.loc[cluster, "fraction"])))
        count += 2
    depth = int(pd.Series(labels).value_counts().max())     # largest cluster = aggregation depth
    tol = (V.tolerance_from_precision(published[value_column].astype(str))
           + V.float32_allowance(float(np.abs(expression).max()), depth))
    return {"n": count, "max_abs_diff": worst, "tolerance": tol, "pass": worst <= tol,
            "extra": f"float32 allowance over {depth} cells"}


def check_fig4a() -> dict:
    return _dotplot("Fig.4a", "Mean scaled expression (dot color; colorbar 0-5, saturates above)",
                    normalise=False)


def check_fig4f() -> dict:
    return _dotplot("Fig.4f", "Per-gene normalized expression (dot color, 0-1)", normalise=True)


def check_ed9b() -> dict:
    published = sheet("Extended Data Fig.9b")
    expression, barcodes, _ = gene_columns(S.object_path(S.MORPH_FULLGENE), S.RECEPTORS)
    idx = align(published["Cell barcode"].astype(str).to_numpy(), barcodes)
    worst = 0.0
    for i, gene in enumerate(S.RECEPTORS):
        column = next(c for c in published.columns if c.startswith(gene))
        worst = max(worst, float(np.abs(
            expression[idx, i] - pd.to_numeric(published[column]).to_numpy(float)).max()))
    tol = (V.tolerance_from_precision(published[published.columns[2]].astype(str))
           + V.float32_allowance(float(np.abs(expression).max())))
    return {"n": len(published) * len(S.RECEPTORS), "max_abs_diff": worst, "tolerance": tol,
            "pass": worst <= tol}


def check_ed6b() -> dict:
    a, b = sheet("Fig.4b"), sheet("Extended Data Fig.6b")
    identical = a.shape == b.shape and all(
        (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all() for c in a.columns)
    return {"n": len(b), "max_abs_diff": 0.0, "tolerance": 0.0, "pass": identical,
            "extra": "byte-identical to Fig 4b" if identical else "DIFFERS from Fig 4b"}


def check_ed8a() -> dict:
    import h5py
    published = sheet("Extended Data Fig.8a")
    with h5py.File(S.object_path(S.EMBRYO_ED8), "r") as fh:
        barcodes, labels = obs_index(fh), obs_column(fh, "leiden_embryo")
    idx = align(published["Cell barcode"].astype(str).to_numpy(), barcodes)
    agree = float((labels[idx] == published["Cluster"].astype(str).to_numpy()).mean())
    return {"n": len(published), "max_abs_diff": 1 - agree, "tolerance": 0.0,
            "pass": agree == 1.0, "extra": f"cluster agreement {agree * 100:.2f}%"}


CHECKS = {
    "Fig 4a": check_fig4a, "Fig 4b": check_fig4b, "Fig 4c": check_fig4c, "Fig 4d": check_fig4d,
    "Fig 4e": check_fig4e, "Fig 4f": check_fig4f, "ED Fig 6b": check_ed6b,
    "ED Fig 8a": check_ed8a, "ED Fig 8b": check_ed8b, "ED Fig 9b": check_ed9b,
}


def main() -> int:
    print(f"{'panel':12s} {'n':>7s} {'max|diff|':>11s} {'tol':>9s}  result   notes")
    failures = skipped = 0
    for spec in S.PANELS:
        # Skip on whether the object is actually REACHABLE, not on whether an environment
        # variable happens to be set: the SMD object ships in this repository, so the panels
        # that need only it verify from a bare clone.
        if spec.needs_object and not S.object_path(spec.needs_object):
            if spec.needs_object == S.MORPH_SMD:
                where = "SMD object missing from this clone"
            elif spec.needs_object in (S.EMBRYO_ED8, S.EMBRYO_FIG4DE):
                # A clustering of a public third-party dataset — point at the accession, not
                # at a deposit of ours. See data/DOWNLOAD.md.
                where = "rebuild from GSE155121 (Zeng et al.) and set OBJECT_ROOT"
            else:
                where = "121 MB — fetch from the Zenodo deposit and set OBJECT_ROOT"
            print(f"{spec.panel:12s} {'':>7s} {'':>11s} {'':>9s}  SKIP     {where}")
            skipped += 1
            continue
        try:
            r = CHECKS[spec.panel]()
        except Exception as exc:                                    # noqa: BLE001
            print(f"{spec.panel:12s} {'ERROR':>7s}  {exc}")
            failures += 1
            continue
        failures += 0 if r["pass"] else 1
        print(f"{spec.panel:12s} {r['n']:7d} {r['max_abs_diff']:11.3e} {r['tolerance']:9.0e}  "
              f"{'PASS' if r['pass'] else 'FAIL':7s}  {r.get('extra', '')}")
    total = len(S.PANELS) - skipped
    print(f"\n{total - failures}/{total} panels verified" + (f" ({skipped} skipped)" if skipped else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
