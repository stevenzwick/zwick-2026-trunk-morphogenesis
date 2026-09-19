"""Recipes for the single-cell panels, and the object lineage each one requires.

Lineage matters more here than anywhere else in the repository. The analysis directories hold
more than one saved state of the same objects, and the directory whose name looks canonical is not
the one the paper used:

* The trunk morph figures come from `trunk_main_dev/` — 5,306 cells × 199 genes × 18 clusters.
  The top-level `results/` tree is an earlier run (5,350 × 204 × 16) and reproduces only ~92% of
  the published cluster assignment.
* ED Fig 8 uses the `trunk_main_dev` embryo object (46,055 cells, 48 clusters).
* Fig 4d and Fig 4e use the `parity` embryo object plus the Floor Plate (07b) and NMP (07c)
  subclustering overrides. That is a deliberate difference, not an oversight — those two panels need
  embryo Floor Plate and NMP labels, which only the override path provides.

Each object was identified by reproducing published output, never by path or file date.
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

TRUNK = "trunk_main_dev/results/intermediates"
PARITY = "parity/results/intermediates"

#: Objects, relative to the analysis root (set OBJECT_ROOT to point at it).
MORPH_SMD = f"{TRUNK}/02_trunk_main/adata_morph_SMD_.h5ad"
MORPH_FULLGENE = f"{TRUNK}/02_trunk_main/adata_morph_with_clusters.h5ad"
EMBRYO_ED8 = f"{TRUNK}/07_human_embryo/adata_embryo_with_clusters.h5ad"
EMBRYO_FIG4DE = f"{PARITY}/07_human_embryo/adata_embryo_with_clusters.h5ad"

#: Objects SHIPPED in this repository, keyed by their path in the analysis tree. Only the SMD
#: object is small enough to commit (4.5 MB); it carries the clustering, the UMAP and the
#: condition labels, which is everything Fig 4b, Fig 4c and ED Fig 6b need. The others are
#: 121 MB and 859/871 MB — past GitHub's 100 MB per-file limit — and are covered instead by the
#: Zenodo deposit and by the public accession the embryo object derives from. See data/DOWNLOAD.md.
SHIPPED_OBJECTS = {MORPH_SMD: "scrnaseq/trunk_main/objects/adata_morph_SMD_.h5ad"}

REPO = Path(__file__).resolve().parent.parent


def object_path(rel: str) -> Path | None:
    """Where to read a saved object from, preferring the copy shipped here.

    Returns None when the object is neither shipped nor reachable through OBJECT_ROOT, so a
    caller can skip that panel rather than fail. Checking the shipped copy FIRST is what lets
    the three SMD-object panels run from a clone with no environment variable set at all.
    """
    shipped = REPO / SHIPPED_OBJECTS.get(rel, "")
    if rel in SHIPPED_OBJECTS and shipped.exists():
        return shipped
    root = os.environ.get("OBJECT_ROOT", "")
    if root and (Path(root) / rel).exists():
        return Path(root) / rel
    return None

#: Shipped artifacts (present in this repo; no object needed).
CORR_FIG4E = "scrnaseq/morph_embryo_correspondence/derived/corr_merged_embryo__integration_gene_panel.csv"
HEATMAP_COUNTS = "scrnaseq/morph_embryo_correspondence/derived/mixed_heatmap_group_counts.csv"
HEATMAP_ORDER = "scrnaseq/morph_embryo_correspondence/derived/mixed_heatmap_cell_order.csv"
GENE_PANEL = "scrnaseq/trunk_main/derived/integration_gene_panel.csv"
FIG4D_LABELS = "scrnaseq/trunk_main/derived/fig4d_labeled_genes.csv"
CLUSTER_ORDER = "scrnaseq/trunk_main/derived/cluster_display_order.csv"


def display_order(labels) -> list[str]:
    """The published cluster order, restricted to the clusters actually present.

    The panels run anterior to posterior, which is not alphabetical: sorting the labels puts
    Dorsal Spinal Cord first and scatters the neural tube. Any label not in the shipped order
    is appended alphabetically rather than dropped, so an object with extra clusters still draws.
    """
    published = pd.read_csv(REPO / CLUSTER_ORDER)["cluster"].astype(str).tolist()
    present = set(pd.unique(pd.Series(labels).astype(str)))
    ordered = [c for c in published if c in present]
    return ordered + sorted(present.difference(ordered))


@dataclass(frozen=True)
class SCPanel:
    panel: str
    source_data: str | None
    needs_object: str | None          # None when verifiable from shipped artifacts alone
    recipe: str
    note: str = ""


PANELS: list[SCPanel] = [
    SCPanel("Fig 4a", "Fig.4a", MORPH_FULLGENE,
            "Per-cluster mean of the unit-variance-scaled expression matrix, plus the fraction of "
            "cells with expression > 0. 38 genes x 18 clusters.",
            "The colour bar saturates at 5; that is display only — the stored values are unclipped."),
    SCPanel("Fig 4b", "Fig.4b", MORPH_SMD,
            "obsm['X_umap'] and obs['leiden_morph'] for all 5,306 cells.",
            "leiden_morph already holds the final annotation names, not integers."),
    SCPanel("Fig 4c", "Fig.4c", MORPH_SMD,
            "Cross-tabulation of obs['leiden_morph'] against obs['source'] — 18 clusters x 2 "
            "conditions, plus within-cluster fractions."),
    SCPanel("Fig 4d", None, None,
            "Heatmap of the 235-gene panel across merged morph + embryo cells, ordered by "
            "mixed_heatmap_cell_order.csv; 44 genes labelled (fig4d_labeled_genes.csv).",
            "No Source Data sheet. Composition is checkable from mixed_heatmap_group_counts.csv."),
    SCPanel("Fig 4e", "Fig.4e", None,
            "18 x 18 correlation over the integration gene panel, morph clusters against the "
            "merged 18-label embryo taxonomy.",
            "The published sheet is the TRANSPOSE of the stored matrix, and the gene set is "
            "decisive — the three sibling gene sets disagree by 0.29 to 1.05."),
    SCPanel("Fig 4f", "Fig.4f", MORPH_FULLGENE,
            "As Fig 4a for 14 morphogen genes, plus a per-gene MIN-MAX normalisation across the "
            "18 clusters.",
            "Min-max, NOT divide-by-max. 12 of 14 genes have a zero minimum so the two agree; "
            "TGFB1 and WNT5B do not, and only min-max reproduces them."),
    SCPanel("ED Fig 6b", "Extended Data Fig.6b", MORPH_SMD,
            "Identical content to Fig 4b.",
            "The two Source Data sheets are byte-identical — one script should emit both panels."),
    SCPanel("ED Fig 8a", "Extended Data Fig.8a", EMBRYO_ED8,
            "Embryo UMAP and obs['leiden_embryo'] for 46,055 cells across 48 clusters.",
            "UMAP coordinates are not stored on this object; the labels are."),
    SCPanel("ED Fig 8b", "Extended Data Fig.8b", None,
            "Cluster x embryo cell counts and within-cluster fractions.",
            "A pure aggregate of ED Fig 8a — verifiable from that sheet alone."),
    SCPanel("ED Fig 9b", "Extended Data Fig.9b", MORPH_FULLGENE,
            "Unit-variance-scaled expression of BMPR1A, BMPR1B, BMPR2 and ACVR1 per cell.",
            "The matrix is ALREADY unit-variance scaled (per-gene sigma = 0.99999). Scaling it "
            "again produces a plausible near-miss of about 8e-05."),
]

RECEPTORS = ["BMPR1A", "BMPR1B", "BMPR2", "ACVR1"]


def cluster_means(expression: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster mean expression and fraction expressing, for one gene."""
    frame = pd.DataFrame({"value": expression, "cluster": labels})
    return pd.DataFrame({
        "mean": frame.groupby("cluster")["value"].mean(),
        "fraction": frame.groupby("cluster")["value"].apply(lambda s: (s > 0).mean()),
    })


def min_max_normalise(means: pd.Series) -> pd.Series:
    """Fig 4f's colour scale: per gene, across clusters. Not divide-by-max."""
    lo, hi = means.min(), means.max()
    return (means - lo) / (hi - lo) if hi > lo else means * 0.0


def by_panel() -> dict[str, SCPanel]:
    return {p.panel: p for p in PANELS}
