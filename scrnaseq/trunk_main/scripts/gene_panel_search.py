from __future__ import annotations

import hashlib
import sys
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from src.trunk_morph_ref.aggregation import cluster_averages_sparse_safe

try:
    from .manuscript_cluster_cluster_correlation import (
        EXPECTED_EMBRYO_BY_MORPH_CURRENT,
        apply_embryo_display_order,
        cross_cluster_correlation,
    )
except ImportError:
    # It lives in a sibling lane's scripts directory, which is on no path by default.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "morph_embryo_correspondence" / "scripts"))
    from manuscript_cluster_cluster_correlation import (
        EXPECTED_EMBRYO_BY_MORPH_CURRENT,
        apply_embryo_display_order,
        cross_cluster_correlation,
    )


UNSCORED_MORPH_LABELS = {"Floor Plate"}


@dataclass(frozen=True)
class SearchConfig:
    top_morph_smd: int = 1000
    top_embryo_smd: int = 1000
    top_morph_deg_per_cluster: int = 40
    top_embryo_deg_per_cluster: int = 40
    single_add_screen_limit: int | None = None
    removal_pool_size: int = 40
    addition_pool_size: int = 120
    swap_pool_size_remove: int = 25
    swap_pool_size_add: int = 80
    beam_width: int = 20
    max_rounds: int = 5
    proposals_per_panel_add: int = 20
    proposals_per_panel_remove: int = 10
    proposals_per_panel_swap: int = 40
    max_total_evaluations: int = 12000


def gene_symbol_series(adata: ad.AnnData) -> pd.Series:
    if "gene_symbol" in adata.var.columns:
        return adata.var["gene_symbol"].astype(str)
    if "gene_name" in adata.var.columns:
        return adata.var["gene_name"].astype(str)
    if "gene_symbol_original" in adata.var.columns:
        return adata.var["gene_symbol_original"].astype(str)
    return pd.Series(adata.var_names.astype(str), index=adata.var_names.astype(str))


def cluster_averages(adata: ad.AnnData, cluster_key: str) -> pd.DataFrame:
    return cluster_averages_sparse_safe(adata, cluster_key)


def reconstruct_embryo_smd_rank_table(
    *,
    embryo_filtered_path: Path,
    embryo_z_scores_path: Path,
) -> pd.DataFrame:
    adata = ad.read_h5ad(embryo_filtered_path)
    z_scores = np.load(embryo_z_scores_path)
    if len(z_scores) != adata.n_vars:
        raise ValueError(
            f"Embryo z-score length {len(z_scores)} != filtered gene count {adata.n_vars}"
        )
    gene_symbols = gene_symbol_series(adata)
    df = pd.DataFrame(
        {
            "gene_ids": adata.var_names.astype(str),
            "gene_symbol": gene_symbols.reindex(adata.var_names).astype(str).values,
            "z_score_SMD": z_scores.astype(float),
        }
    )
    df["smd_rank"] = df["z_score_SMD"].rank(method="first", ascending=False).astype(int)
    df["selected_at_z_gt_2"] = df["z_score_SMD"] > 2
    return df.sort_values(["z_score_SMD", "gene_ids"], ascending=[False, True]).reset_index(drop=True)


def morph_smd_rank_table_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [c for c in ["gene_ids", "gene_symbol", "z_score_SMD", "selected_at_z_gt_2"] if c in df.columns]
    out = df[keep].copy()
    out["smd_rank"] = out["z_score_SMD"].rank(method="first", ascending=False).astype(int)
    return out.sort_values(["z_score_SMD", "gene_ids"], ascending=[False, True]).reset_index(drop=True)


def rank_genes_uns_to_long_df(
    adata: ad.AnnData,
    *,
    key: str,
    n_top: int = 100,
    cluster_col_name: str = "cluster",
) -> pd.DataFrame:
    names = pd.DataFrame(adata.uns[key]["names"]).iloc[:n_top]
    scores = pd.DataFrame(adata.uns[key]["scores"]).iloc[:n_top]
    gene_symbols = gene_symbol_series(adata)
    rows: list[dict[str, object]] = []
    for cluster in names.columns:
        for rank0, gene_id in enumerate(names[cluster].astype(str).tolist(), start=1):
            score = float(scores.loc[rank0 - 1, cluster])
            rows.append(
                {
                    cluster_col_name: str(cluster),
                    "gene_ids": gene_id,
                    "gene_symbol": str(gene_symbols.get(gene_id, gene_id)),
                    "logreg_score": score,
                    "logreg_rank": rank0,
                }
            )
    return pd.DataFrame(rows)


def compute_logreg_deg_table(
    adata: ad.AnnData,
    *,
    groupby: str,
    n_top: int = 100,
    key_added: str,
    max_iter: int = 1000,
) -> pd.DataFrame:
    import scanpy as sc

    local = adata.copy()
    sc.tl.rank_genes_groups(
        local,
        groupby=groupby,
        method="logreg",
        rankby_abs=False,
        max_iter=max_iter,
        multi_class="multinomial",
        key_added=key_added,
    )
    return rank_genes_uns_to_long_df(local, key=key_added, n_top=n_top)


def build_preferred_mapping() -> dict[str, list[str]]:
    return {k: list(v) for k, v in EXPECTED_EMBRYO_BY_MORPH_CURRENT.items()}


def build_family_acceptable_mapping(all_embryo_labels: Sequence[str]) -> dict[str, list[str]]:
    by_morph: dict[str, list[str]] = defaultdict(list)
    for label in all_embryo_labels:
        if "Forebrain" in label or "Midbrain" in label:
            by_morph["Forebrain / Midbrain"].append(label)
        if "Neuron -" in label:
            by_morph["Immature Neuron"].append(label)
        if label == "Week 4 Intermediate-Ventral Spinal Cord":
            by_morph["Floor Plate"].append(label)
    return {k: sorted(v) for k, v in by_morph.items()}


def build_soft_mapping() -> dict[str, list[str]]:
    return {
        "Dorsal Spinal Cord": ["Week 4 Roof Plate"],
        "Roof Plate": ["Week 4 Dorsal Spinal Cord"],
    }


def build_candidate_universe(
    *,
    shared_genes: Sequence[str],
    baseline_genes: Sequence[str],
    morph_smd_table: pd.DataFrame,
    embryo_smd_table: pd.DataFrame,
    morph_deg_table: pd.DataFrame,
    embryo_deg_table: pd.DataFrame,
    config: SearchConfig,
) -> pd.DataFrame:
    shared = set(map(str, shared_genes))
    baseline = set(map(str, baseline_genes))

    morph_smd = morph_smd_table.loc[morph_smd_table["gene_ids"].isin(shared)].copy()
    embryo_smd = embryo_smd_table.loc[embryo_smd_table["gene_ids"].isin(shared)].copy()
    morph_deg = morph_deg_table.loc[morph_deg_table["gene_ids"].isin(shared)].copy()
    embryo_deg = embryo_deg_table.loc[embryo_deg_table["gene_ids"].isin(shared)].copy()

    morph_smd_top = morph_smd.nsmallest(config.top_morph_smd, "smd_rank")
    embryo_smd_top = embryo_smd.nsmallest(config.top_embryo_smd, "smd_rank")
    morph_deg_top = morph_deg.loc[morph_deg["logreg_rank"] <= config.top_morph_deg_per_cluster]
    embryo_deg_top = embryo_deg.loc[embryo_deg["logreg_rank"] <= config.top_embryo_deg_per_cluster]

    universe_ids = (
        set(morph_smd_top["gene_ids"])
        | set(embryo_smd_top["gene_ids"])
        | set(morph_deg_top["gene_ids"])
        | set(embryo_deg_top["gene_ids"])
        | baseline
    )

    records: list[dict[str, object]] = []
    gene_symbols = {}
    for frame in [morph_smd, embryo_smd, morph_deg, embryo_deg]:
        for row in frame[["gene_ids", "gene_symbol"]].drop_duplicates().itertuples(index=False):
            gene_symbols.setdefault(str(row.gene_ids), str(row.gene_symbol))

    morph_deg_best = (
        morph_deg.sort_values(["logreg_rank", "logreg_score"], ascending=[True, False])
        .drop_duplicates("gene_ids")
        .set_index("gene_ids")
    )
    embryo_deg_best = (
        embryo_deg.sort_values(["logreg_rank", "logreg_score"], ascending=[True, False])
        .drop_duplicates("gene_ids")
        .set_index("gene_ids")
    )
    morph_smd_best = morph_smd.drop_duplicates("gene_ids").set_index("gene_ids")
    embryo_smd_best = embryo_smd.drop_duplicates("gene_ids").set_index("gene_ids")

    for gene_id in sorted(universe_ids):
        rec: dict[str, object] = {
            "gene_ids": gene_id,
            "gene_symbol": gene_symbols.get(gene_id, gene_id),
            "in_baseline_panel": gene_id in baseline,
        }
        if gene_id in morph_smd_best.index:
            rec["morph_smd_z"] = float(morph_smd_best.loc[gene_id, "z_score_SMD"])
            rec["morph_smd_rank"] = int(morph_smd_best.loc[gene_id, "smd_rank"])
            rec["in_top_morph_smd"] = gene_id in set(morph_smd_top["gene_ids"])
        else:
            rec["morph_smd_z"] = np.nan
            rec["morph_smd_rank"] = np.nan
            rec["in_top_morph_smd"] = False
        if gene_id in embryo_smd_best.index:
            rec["embryo_smd_z"] = float(embryo_smd_best.loc[gene_id, "z_score_SMD"])
            rec["embryo_smd_rank"] = int(embryo_smd_best.loc[gene_id, "smd_rank"])
            rec["in_top_embryo_smd"] = gene_id in set(embryo_smd_top["gene_ids"])
        else:
            rec["embryo_smd_z"] = np.nan
            rec["embryo_smd_rank"] = np.nan
            rec["in_top_embryo_smd"] = False
        if gene_id in morph_deg_best.index:
            rec["morph_deg_score"] = float(morph_deg_best.loc[gene_id, "logreg_score"])
            rec["morph_deg_rank"] = int(morph_deg_best.loc[gene_id, "logreg_rank"])
            rec["morph_deg_cluster"] = str(morph_deg_best.loc[gene_id, "cluster"])
            rec["in_top_morph_deg"] = gene_id in set(morph_deg_top["gene_ids"])
        else:
            rec["morph_deg_score"] = np.nan
            rec["morph_deg_rank"] = np.nan
            rec["morph_deg_cluster"] = ""
            rec["in_top_morph_deg"] = False
        if gene_id in embryo_deg_best.index:
            rec["embryo_deg_score"] = float(embryo_deg_best.loc[gene_id, "logreg_score"])
            rec["embryo_deg_rank"] = int(embryo_deg_best.loc[gene_id, "logreg_rank"])
            rec["embryo_deg_cluster"] = str(embryo_deg_best.loc[gene_id, "cluster"])
            rec["in_top_embryo_deg"] = gene_id in set(embryo_deg_top["gene_ids"])
        else:
            rec["embryo_deg_score"] = np.nan
            rec["embryo_deg_rank"] = np.nan
            rec["embryo_deg_cluster"] = ""
            rec["in_top_embryo_deg"] = False
        rec["source_count"] = int(
            rec["in_baseline_panel"]
            + rec["in_top_morph_smd"]
            + rec["in_top_embryo_smd"]
            + rec["in_top_morph_deg"]
            + rec["in_top_embryo_deg"]
        )
        records.append(rec)

    universe = pd.DataFrame(records)
    universe["priority_rank"] = universe.apply(priority_rank_tuple, axis=1).rank(method="dense").astype(int)
    return universe.sort_values(
        by=[
            "source_count",
            "in_baseline_panel",
            "morph_smd_rank",
            "embryo_smd_rank",
            "morph_deg_rank",
            "embryo_deg_rank",
            "gene_ids",
        ],
        ascending=[False, False, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def priority_rank_tuple(row: pd.Series) -> tuple:
    def norm(x):
        return int(x) if pd.notna(x) else 10**9

    return (
        -int(row.get("source_count", 0)),
        -int(bool(row.get("in_baseline_panel", False))),
        norm(row.get("morph_smd_rank", np.nan)),
        norm(row.get("embryo_smd_rank", np.nan)),
        norm(row.get("morph_deg_rank", np.nan)),
        norm(row.get("embryo_deg_rank", np.nan)),
        str(row.get("gene_ids", "")),
    )


def panel_signature(genes: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(map(str, genes))))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def panel_metrics_sort_key(metrics: Mapping[str, object]) -> tuple:
    def get_float(name: str, default: float) -> float:
        value = metrics.get(name, default)
        return float(value) if value is not None else default

    def get_int(name: str, default: int) -> int:
        value = metrics.get(name, default)
        return int(value) if value is not None else default

    return (
        get_int("bad_deviation_count", 10**9),
        get_int("soft_deviation_count", 10**9),
        get_int("family_deviation_count", 10**9),
        -get_float("median_expected_margin_bad", -np.inf),
        -get_float("median_expected_margin_all", -np.inf),
        -get_float("median_top1_minus_top2", -np.inf),
        get_int("panel_size", 10**9),
    )


def evaluate_panel(
    *,
    panel_genes: Sequence[str],
    avg_morph: pd.DataFrame,
    avg_embryo: pd.DataFrame,
    preferred_by_morph: Mapping[str, Sequence[str]],
    family_by_morph: Mapping[str, Sequence[str]],
    soft_by_morph: Mapping[str, Sequence[str]],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    genes = sorted(set(map(str, panel_genes)))
    corr_df = cross_cluster_correlation(avg_morph, avg_embryo, genes)
    rows: list[dict[str, object]] = []
    expected_margin_all: list[float] = []
    expected_margin_bad: list[float] = []
    top_gaps: list[float] = []

    for morph_label in corr_df.index:
        row = corr_df.loc[morph_label].sort_values(ascending=False)
        preferred = set(map(str, preferred_by_morph.get(str(morph_label), [])))
        family = set(map(str, family_by_morph.get(str(morph_label), [])))
        soft = set(map(str, soft_by_morph.get(str(morph_label), [])))
        top1_label = str(row.index[0])
        top1_corr = float(row.iloc[0])
        top2_corr = float(row.iloc[1]) if len(row) > 1 else np.nan
        top_gap = top1_corr - top2_corr if np.isfinite(top2_corr) else np.nan
        if np.isfinite(top_gap):
            top_gaps.append(top_gap)

        if morph_label in UNSCORED_MORPH_LABELS or not preferred:
            category = "unscored"
        elif top1_label in preferred:
            category = "preferred"
        elif top1_label in soft:
            category = "soft"
        elif top1_label in family:
            category = "family"
        else:
            category = "bad"

        expected_best = np.nan
        bad_margin = np.nan
        all_margin = np.nan
        if preferred:
            preferred_vals = [float(corr_df.loc[morph_label, e]) for e in preferred if e in corr_df.columns]
            if preferred_vals:
                expected_best = max(preferred_vals)
                nonpreferred = [e for e in corr_df.columns if e not in preferred]
                if nonpreferred:
                    all_margin = expected_best - float(corr_df.loc[morph_label, nonpreferred].max())
                bad_pool = [e for e in corr_df.columns if e not in (preferred | family | soft)]
                if bad_pool:
                    bad_margin = expected_best - float(corr_df.loc[morph_label, bad_pool].max())
        if np.isfinite(all_margin):
            expected_margin_all.append(float(all_margin))
        if np.isfinite(bad_margin):
            expected_margin_bad.append(float(bad_margin))

        rows.append(
            {
                "morph_label": str(morph_label),
                "top1_embryo_label": top1_label,
                "top1_corr": top1_corr,
                "top2_corr": top2_corr,
                "top1_minus_top2": top_gap,
                "expected_labels": "|".join(sorted(preferred)),
                "family_labels": "|".join(sorted(family)),
                "soft_labels": "|".join(sorted(soft)),
                "expected_best_corr": expected_best,
                "expected_margin_all": all_margin,
                "expected_margin_bad": bad_margin,
                "top1_category": category,
            }
        )

    top_matches = pd.DataFrame(rows)
    scored = top_matches.loc[top_matches["top1_category"] != "unscored"].copy()
    metrics = {
        "panel_signature": panel_signature(genes),
        "panel_size": len(genes),
        "preferred_top1_rate": float((scored["top1_category"] == "preferred").mean()) if len(scored) else np.nan,
        "acceptable_top1_rate": float(scored["top1_category"].isin(["preferred", "family", "soft"]).mean()) if len(scored) else np.nan,
        "bad_deviation_count": int((scored["top1_category"] == "bad").sum()),
        "soft_deviation_count": int((scored["top1_category"] == "soft").sum()),
        "family_deviation_count": int((scored["top1_category"] == "family").sum()),
        "median_expected_margin_all": float(np.nanmedian(expected_margin_all)) if expected_margin_all else np.nan,
        "median_expected_margin_bad": float(np.nanmedian(expected_margin_bad)) if expected_margin_bad else np.nan,
        "median_top1_minus_top2": float(np.nanmedian(top_gaps)) if top_gaps else np.nan,
        "mean_top1_corr": float(np.nanmean(top_matches["top1_corr"])) if len(top_matches) else np.nan,
        "floor_plate_top1_embryo": str(top_matches.loc[top_matches["morph_label"] == "Floor Plate", "top1_embryo_label"].iloc[0]) if (top_matches["morph_label"] == "Floor Plate").any() else "",
        "floor_plate_top1_corr": float(top_matches.loc[top_matches["morph_label"] == "Floor Plate", "top1_corr"].iloc[0]) if (top_matches["morph_label"] == "Floor Plate").any() else np.nan,
    }
    metrics["sort_key"] = json.dumps(panel_metrics_sort_key(metrics))
    return metrics, top_matches, corr_df


def evaluate_panels(
    panel_defs: Sequence[dict[str, object]],
    *,
    avg_morph: pd.DataFrame,
    avg_embryo: pd.DataFrame,
    preferred_by_morph: Mapping[str, Sequence[str]],
    family_by_morph: Mapping[str, Sequence[str]],
    soft_by_morph: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    rows = []
    top_matches = {}
    corr_mats = {}
    for panel_def in panel_defs:
        metrics, top_df, corr_df = evaluate_panel(
            panel_genes=panel_def["genes"],
            avg_morph=avg_morph,
            avg_embryo=avg_embryo,
            preferred_by_morph=preferred_by_morph,
            family_by_morph=family_by_morph,
            soft_by_morph=soft_by_morph,
        )
        row = dict(panel_def)
        row.update(metrics)
        rows.append(row)
        top_matches[str(row["panel_id"])] = top_df
        corr_mats[str(row["panel_id"])] = corr_df
    return pd.DataFrame(rows), top_matches, corr_mats


def choose_addition_candidates(universe: pd.DataFrame, baseline_genes: set[str], limit: int | None = None) -> list[str]:
    df = universe.loc[~universe["gene_ids"].isin(baseline_genes)].copy()
    df = df.sort_values(
        by=[
            "source_count",
            "morph_smd_rank",
            "embryo_smd_rank",
            "morph_deg_rank",
            "embryo_deg_rank",
        ],
        ascending=[False, True, True, True, True],
        na_position="last",
    )
    genes = df["gene_ids"].astype(str).tolist()
    return genes if limit is None else genes[:limit]


def deduplicate_panel_defs(panel_defs: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    seen = set()
    out = []
    for panel in panel_defs:
        sig = panel_signature(panel["genes"])
        if sig in seen:
            continue
        seen.add(sig)
        item = dict(panel)
        item["panel_signature"] = sig
        out.append(item)
    return out


def pick_top_panels(df: pd.DataFrame, width: int) -> pd.DataFrame:
    if df.empty:
        return df
    order = sorted(range(len(df)), key=lambda i: panel_metrics_sort_key(df.iloc[i].to_dict()))
    return df.iloc[order[:width]].copy().reset_index(drop=True)


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep = []
    metrics = [
        "bad_deviation_count",
        "soft_deviation_count",
        "family_deviation_count",
        "median_expected_margin_bad",
        "median_expected_margin_all",
        "median_top1_minus_top2",
    ]
    rows = df.to_dict(orient="records")
    for i, row_i in enumerate(rows):
        dominated = False
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            no_worse = (
                row_j["bad_deviation_count"] <= row_i["bad_deviation_count"]
                and row_j["soft_deviation_count"] <= row_i["soft_deviation_count"]
                and row_j["family_deviation_count"] <= row_i["family_deviation_count"]
                and row_j["median_expected_margin_bad"] >= row_i["median_expected_margin_bad"]
                and row_j["median_expected_margin_all"] >= row_i["median_expected_margin_all"]
                and row_j["median_top1_minus_top2"] >= row_i["median_top1_minus_top2"]
            )
            strictly_better = (
                row_j["bad_deviation_count"] < row_i["bad_deviation_count"]
                or row_j["soft_deviation_count"] < row_i["soft_deviation_count"]
                or row_j["family_deviation_count"] < row_i["family_deviation_count"]
                or row_j["median_expected_margin_bad"] > row_i["median_expected_margin_bad"]
                or row_j["median_expected_margin_all"] > row_i["median_expected_margin_all"]
                or row_j["median_top1_minus_top2"] > row_i["median_top1_minus_top2"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return pick_top_panels(df.iloc[keep].copy().reset_index(drop=True), width=len(keep))


def single_removal_panels(baseline_genes: Sequence[str]) -> list[dict[str, object]]:
    baseline = sorted(set(map(str, baseline_genes)))
    panels = []
    for gene in baseline:
        panels.append(
            {
                "panel_id": f"remove__{gene}",
                "parent_id": "baseline",
                "move_type": "remove",
                "removed_gene_ids": gene,
                "added_gene_ids": "",
                "genes": [g for g in baseline if g != gene],
            }
        )
    return panels


def single_addition_panels(
    baseline_genes: Sequence[str],
    addition_candidates: Sequence[str],
) -> list[dict[str, object]]:
    baseline = sorted(set(map(str, baseline_genes)))
    panels = []
    for gene in addition_candidates:
        if gene in baseline:
            continue
        panels.append(
            {
                "panel_id": f"add__{gene}",
                "parent_id": "baseline",
                "move_type": "add",
                "removed_gene_ids": "",
                "added_gene_ids": gene,
                "genes": baseline + [gene],
            }
        )
    return panels


def swap_panels(
    baseline_genes: Sequence[str],
    removal_candidates: Sequence[str],
    addition_candidates: Sequence[str],
) -> list[dict[str, object]]:
    baseline = sorted(set(map(str, baseline_genes)))
    panels = []
    for out_gene in removal_candidates:
        if out_gene not in baseline:
            continue
        for in_gene in addition_candidates:
            if in_gene in baseline or in_gene == out_gene:
                continue
            new_genes = [g for g in baseline if g != out_gene] + [in_gene]
            panels.append(
                {
                    "panel_id": f"swap__{out_gene}__{in_gene}",
                    "parent_id": "baseline",
                    "move_type": "swap",
                    "removed_gene_ids": out_gene,
                    "added_gene_ids": in_gene,
                    "genes": new_genes,
                }
            )
    return panels


def frontier_gene_frequencies(frontier: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    add_counts = defaultdict(int)
    remove_counts = defaultdict(int)
    for row in frontier.itertuples(index=False):
        for g in str(getattr(row, "added_gene_ids", "")).split("|"):
            if g:
                add_counts[g] += 1
        for g in str(getattr(row, "removed_gene_ids", "")).split("|"):
            if g:
                remove_counts[g] += 1
    return pd.Series(add_counts).sort_values(ascending=False), pd.Series(remove_counts).sort_values(ascending=False)


def expand_from_frontier(
    frontier: pd.DataFrame,
    *,
    baseline_genes: set[str],
    addition_pool: Sequence[str],
    removal_pool: Sequence[str],
    config: SearchConfig,
) -> list[dict[str, object]]:
    proposals: list[dict[str, object]] = []
    for row in frontier.itertuples(index=False):
        current_genes = set(str(g) for g in str(row.genes_json).split("|") if g)
        add_candidates = [g for g in addition_pool if g not in current_genes][: config.proposals_per_panel_add]
        remove_candidates = [g for g in removal_pool if g in current_genes][: config.proposals_per_panel_remove]
        swap_limit = config.proposals_per_panel_swap
        for gene in add_candidates:
            new_genes = sorted(current_genes | {gene})
            proposals.append(
                {
                    "panel_id": f"{row.panel_id}__add__{gene}",
                    "parent_id": row.panel_id,
                    "move_type": "add",
                    "removed_gene_ids": "",
                    "added_gene_ids": gene,
                    "genes": new_genes,
                }
            )
        for gene in remove_candidates:
            new_genes = sorted(g for g in current_genes if g != gene)
            proposals.append(
                {
                    "panel_id": f"{row.panel_id}__remove__{gene}",
                    "parent_id": row.panel_id,
                    "move_type": "remove",
                    "removed_gene_ids": gene,
                    "added_gene_ids": "",
                    "genes": new_genes,
                }
            )
        swap_count = 0
        for out_gene in remove_candidates:
            for in_gene in add_candidates:
                if in_gene == out_gene:
                    continue
                new_genes = sorted((current_genes - {out_gene}) | {in_gene})
                proposals.append(
                    {
                        "panel_id": f"{row.panel_id}__swap__{out_gene}__{in_gene}",
                        "parent_id": row.panel_id,
                        "move_type": "swap",
                        "removed_gene_ids": out_gene,
                        "added_gene_ids": in_gene,
                        "genes": new_genes,
                    }
                )
                swap_count += 1
                if swap_count >= swap_limit:
                    break
            if swap_count >= swap_limit:
                break
    return deduplicate_panel_defs(proposals)


def encode_panel_genes(genes: Sequence[str]) -> str:
    return "|".join(sorted(set(map(str, genes))))


def adaptive_beam_search(
    *,
    baseline_genes: Sequence[str],
    baseline_metrics: Mapping[str, object],
    removal_screen: pd.DataFrame,
    addition_screen: pd.DataFrame,
    swap_screen: pd.DataFrame,
    avg_morph: pd.DataFrame,
    avg_embryo: pd.DataFrame,
    preferred_by_morph: Mapping[str, Sequence[str]],
    family_by_morph: Mapping[str, Sequence[str]],
    soft_by_morph: Mapping[str, Sequence[str]],
    config: SearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_set = set(map(str, baseline_genes))

    removal_pool = removal_screen.sort_values(
        by=[
            "bad_deviation_count",
            "soft_deviation_count",
            "family_deviation_count",
            "median_expected_margin_bad",
            "median_expected_margin_all",
            "median_top1_minus_top2",
        ],
        ascending=[True, True, True, False, False, False],
    )["removed_gene_ids"].head(config.removal_pool_size).astype(str).tolist()

    addition_pool = addition_screen.sort_values(
        by=[
            "bad_deviation_count",
            "soft_deviation_count",
            "family_deviation_count",
            "median_expected_margin_bad",
            "median_expected_margin_all",
            "median_top1_minus_top2",
        ],
        ascending=[True, True, True, False, False, False],
    )["added_gene_ids"].head(config.addition_pool_size).astype(str).tolist()

    initial_frontier = pd.concat(
        [
            pick_top_panels(addition_screen, width=max(5, config.beam_width // 2)),
            pick_top_panels(removal_screen, width=max(5, config.beam_width // 2)),
            pick_top_panels(swap_screen, width=config.beam_width),
        ],
        ignore_index=True,
    )
    initial_frontier = deduplicate_panel_defs(
        [
            {
                "panel_id": row.panel_id,
                "parent_id": row.parent_id,
                "move_type": row.move_type,
                "removed_gene_ids": row.removed_gene_ids,
                "added_gene_ids": row.added_gene_ids,
                "genes": str(row.genes_json).split("|"),
            }
            for row in initial_frontier.itertuples(index=False)
        ]
    )

    evaluated_rows: list[dict[str, object]] = []
    frontier_rows: list[dict[str, object]] = []

    for panel in initial_frontier:
        panel_df, _, _ = evaluate_panels(
            [panel],
            avg_morph=avg_morph,
            avg_embryo=avg_embryo,
            preferred_by_morph=preferred_by_morph,
            family_by_morph=family_by_morph,
            soft_by_morph=soft_by_morph,
        )
        row = panel_df.iloc[0].to_dict()
        row["genes_json"] = encode_panel_genes(panel["genes"])
        row["round"] = 1
        evaluated_rows.append(row)
        frontier_rows.append(row)

    evaluated = pd.DataFrame(evaluated_rows)
    frontier = pick_top_panels(pd.DataFrame(frontier_rows), width=config.beam_width)
    frontier["genes_json"] = frontier["genes_json"].astype(str)

    best_key = min(panel_metrics_sort_key(frontier.iloc[i].to_dict()) for i in range(len(frontier))) if len(frontier) else panel_metrics_sort_key(baseline_metrics)
    stagnant_rounds = 0
    seen_signatures = set(evaluated["panel_signature"].astype(str).tolist())

    for round_idx in range(2, config.max_rounds + 1):
        if len(evaluated) >= config.max_total_evaluations:
            break
        proposals = expand_from_frontier(
            frontier,
            baseline_genes=baseline_set,
            addition_pool=addition_pool,
            removal_pool=removal_pool,
            config=config,
        )
        proposals = [p for p in proposals if panel_signature(p["genes"]) not in seen_signatures]
        if not proposals:
            break
        proposal_df, _, _ = evaluate_panels(
            proposals,
            avg_morph=avg_morph,
            avg_embryo=avg_embryo,
            preferred_by_morph=preferred_by_morph,
            family_by_morph=family_by_morph,
            soft_by_morph=soft_by_morph,
        )
        proposal_df["genes_json"] = proposal_df["genes"].apply(encode_panel_genes)
        proposal_df["round"] = round_idx
        evaluated = pd.concat([evaluated, proposal_df], ignore_index=True)
        for sig in proposal_df["panel_signature"].astype(str):
            seen_signatures.add(sig)

        frontier = pick_top_panels(evaluated, width=config.beam_width)
        current_best_key = min(panel_metrics_sort_key(frontier.iloc[i].to_dict()) for i in range(len(frontier)))
        if current_best_key < best_key:
            best_key = current_best_key
            stagnant_rounds = 0
        else:
            stagnant_rounds += 1

        add_freq, remove_freq = frontier_gene_frequencies(frontier)
        addition_pool = list(dict.fromkeys(add_freq.index.astype(str).tolist() + addition_pool))[: config.addition_pool_size]
        removal_pool = list(dict.fromkeys(remove_freq.index.astype(str).tolist() + removal_pool))[: config.removal_pool_size]

        if stagnant_rounds >= 2:
            break

    return evaluated.reset_index(drop=True), frontier.reset_index(drop=True)


def annotate_relative_to_baseline(df: pd.DataFrame, baseline_metrics: Mapping[str, object]) -> pd.DataFrame:
    out = df.copy()
    for metric in [
        "bad_deviation_count",
        "soft_deviation_count",
        "family_deviation_count",
        "preferred_top1_rate",
        "acceptable_top1_rate",
        "median_expected_margin_bad",
        "median_expected_margin_all",
        "median_top1_minus_top2",
    ]:
        out[f"delta__{metric}"] = out[metric] - float(baseline_metrics[metric])
    return out


def best_panels_by_category(evaluated: pd.DataFrame, *, width: int = 25) -> pd.DataFrame:
    return pick_top_panels(evaluated, width=width)


def build_all_embryo_order(embryo_h5ad_path: Path) -> list[str]:
    adata = ad.read_h5ad(embryo_h5ad_path)
    return apply_embryo_display_order(list(adata.obs["leiden_embryo"].cat.categories))


def build_search_context(
    *,
    morph_h5ad_path: Path,
    embryo_h5ad_path: Path,
    morph_order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    morph = ad.read_h5ad(morph_h5ad_path)
    embryo = ad.read_h5ad(embryo_h5ad_path)
    embryo_order = apply_embryo_display_order(list(embryo.obs["leiden_embryo"].cat.categories))
    avg_morph = cluster_averages(morph, "leiden_morph").loc[list(morph_order)]
    avg_embryo = cluster_averages(embryo, "leiden_embryo").loc[embryo_order]
    preferred = build_preferred_mapping()
    family = build_family_acceptable_mapping(embryo_order)
    soft = build_soft_mapping()
    return avg_morph, avg_embryo, embryo_order, preferred, family, soft
