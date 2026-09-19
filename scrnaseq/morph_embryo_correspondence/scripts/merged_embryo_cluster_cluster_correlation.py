from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from src.trunk_morph_ref.aggregation import cluster_averages_sparse_safe


MERGED_EMBRYO_GROUPS_V1: dict[str, list[str]] = {
    "Forebrain / Midbrain": [
        "Week 4 Forebrain / Midbrain",
        "Week 4 Forebrain - Optic Field",
        "Week 4 Forebrain - Diencephalon / Telencephalon",
        "Week 4 Dorsal Midbrain",
    ],
    "Hindbrain": [
        "Week 4 Hindbrain",
    ],
    "Intermediate-Ventral Spinal Cord": [
        "Week 4 Intermediate-Ventral Spinal Cord",
    ],
    "Dorsal Spinal Cord": [
        "Week 4 Dorsal Spinal Cord",
    ],
    "Roof Plate": [
        "Week 4 Roof Plate",
    ],
    "Neural Crest": [
        "Week 3 Neural Crest - Early Migratory",
        "Week 4 Neural Crest - Derivatives",
        "Week 4 Neural Crest - Maturing Cranial",
    ],
    "Immature Neuron": [
        "Week 4 Neuron - CNS Excitatory",
        "Week 4 Neuron - CNS Inhibitory",
        "Week 4 Neuron - Catecholaminergic / Glutamatergic",
        "Week 4 Neuron - Peripheral Sensory",
        "Week 4 Neuron - Autonomic / Cholinergic",
    ],
    "Posterior Neural Tube / NMP": [
        "Week 3 Posterior Neural Tube / Neuromesodermal Progenitors",
    ],
    "Presomitic Mesoderm": [
        "Week 3 Presomitic Mesoderm - Posterior",
        "Week 3 Presomitic Mesoderm - Anterior",
    ],
    "Notochord": [
        "Week 3 / 4 Notochord",
    ],
    "Early Somite": [
        "Week 3 Early Somite",
    ],
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
    "Endothelial": [
        "Week 3 Endothelial",
        "Week 4 Endothelial",
    ],
    "Non-Neural Ectoderm": [
        "Week 4 Non-Neural Ectoderm - Surface Ectoderm",
        "Week 3 Non-Neural Ectoderm",
    ],
}


DROPPED_EMBRYO_LABELS_V1: list[str] = [
    "Week 3 Anterior Neuroectoderm",
    "Week 4 Neuron - Hypothalamus / Neuroendocrine",
    "Week 3 Lateral Plate Mesoderm - Craniofacial / Pharyngeal",
    "Week 4 Lateral Plate Mesoderm - Craniofacial / Pharyngeal",
    "Week 3 / 4 Cardiomyocytes",
    "Week 4 Skeletal Myocytes",
    "Week 4 Head Mesenchyme - Multipotent Progenitors",
    "Week 4 Head Mesenchyme - Pharyngeal Arch Core Mesoderm",
    "Week 4 Head Mesenchyme - Frontonasal Mesoderm",
    "Week 4 Head Mesenchyme - First Arch Oral / Palatal Mesenchyme",
    "Week 4 Head Mesenchyme - Branchiomeric Muscle Progenitors",
    "Week 4 Trunk Mesenchyme",
    "Week 4 Non-Neural Ectoderm - Cranial Placodal",
    "Week 4 Definitive Endoderm - Fetal Liver",
    "Week 4 Definitive Endoderm - Intestinal",
    "Week 3 / 4 Erythroid Precursors",
    "Week 3 / 4 Hematopoietic Progenitors",
]


MERGED_EMBRYO_ORDER_V1: list[str] = [
    "Forebrain / Midbrain",
    "Hindbrain",
    "Intermediate-Ventral Spinal Cord",
    "Dorsal Spinal Cord",
    "Roof Plate",
    "Neural Crest",
    "Immature Neuron",
    "Posterior Neural Tube / NMP",
    "Notochord",
    "Presomitic Mesoderm",
    "Early Somite",
    "Mature Somite",
    "Intermediate Mesoderm",
    "Lateral Plate Mesoderm",
    "Endothelial",
    "Non-Neural Ectoderm",
]


EXPECTED_MERGED_EMBRYO_BY_MORPH_V1: dict[str, list[str]] = {
    "Forebrain / Midbrain": ["Forebrain / Midbrain"],
    "Hindbrain": ["Hindbrain"],
    "Floor Plate": [],
    "Intermediate-Ventral Spinal Cord": ["Intermediate-Ventral Spinal Cord"],
    "Dorsal Spinal Cord": ["Dorsal Spinal Cord"],
    "Roof Plate": ["Roof Plate"],
    "Neural Crest": ["Neural Crest"],
    "Immature Neuron": ["Immature Neuron"],
    "Posterior Neural Tube": ["Posterior Neural Tube / NMP"],
    "Neuromesodermal Progenitors": ["Posterior Neural Tube / NMP"],
    "Notochord": ["Notochord"],
    "Presomitic Mesoderm": ["Presomitic Mesoderm"],
    "Early Somite": ["Early Somite"],
    "Mature Somite": ["Mature Somite"],
    "Intermediate Mesoderm": ["Intermediate Mesoderm"],
    "Lateral Plate Mesoderm": ["Lateral Plate Mesoderm"],
    "Endothelial": ["Endothelial"],
    "Non-Neural Ectoderm": ["Non-Neural Ectoderm"],
}


MERGED_EMBRYO_DISPLAY_LABELS_V1: dict[str, str] = {
    "Forebrain / Midbrain": "FB/MB",
    "Hindbrain": "HB",
    "Intermediate-Ventral Spinal Cord": "Int.-Vent. SC",
    "Dorsal Spinal Cord": "Dorsal SC",
    "Roof Plate": "Roof Plate",
    "Neural Crest": "Neural Crest",
    "Immature Neuron": "Imm. Neuron",
    "Posterior Neural Tube / NMP": "Post NT/NMP",
    "Notochord": "Notochord",
    "Presomitic Mesoderm": "PSM",
    "Early Somite": "Early Somite",
    "Mature Somite": "Mature Somite",
    "Intermediate Mesoderm": "IM",
    "Lateral Plate Mesoderm": "LPM",
    "Endothelial": "Endothelial",
    "Non-Neural Ectoderm": "NNE",
}


def merged_embryo_membership_table() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for merged_group, members in MERGED_EMBRYO_GROUPS_V1.items():
        for embryo_label in members:
            rows.append(
                {
                    "embryo_label": embryo_label,
                    "status": "merge_keep",
                    "merged_group": merged_group,
                }
            )
    for embryo_label in DROPPED_EMBRYO_LABELS_V1:
        rows.append(
            {
                "embryo_label": embryo_label,
                "status": "drop",
                "merged_group": "",
            }
        )
    return pd.DataFrame(rows)


def embryo_to_merged_group_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for merged_group, members in MERGED_EMBRYO_GROUPS_V1.items():
        for embryo_label in members:
            if embryo_label in mapping:
                raise ValueError(f"Duplicate embryo label in merge scheme: {embryo_label}")
            mapping[embryo_label] = merged_group
    return mapping


def assign_merged_embryo_labels(embryo_labels: Sequence[str]) -> pd.Series:
    mapping = embryo_to_merged_group_map()
    values = [mapping.get(str(label), pd.NA) for label in embryo_labels]
    return pd.Series(values, index=getattr(embryo_labels, "index", None), dtype="object")


def validate_merge_scheme(all_embryo_labels: Sequence[str]) -> tuple[list[str], list[str]]:
    all_labels = list(map(str, all_embryo_labels))
    mapping = embryo_to_merged_group_map()
    assigned = set(mapping)
    dropped = set(map(str, DROPPED_EMBRYO_LABELS_V1))
    missing = [label for label in all_labels if label not in assigned and label not in dropped]
    unused = [label for label in list(assigned | dropped) if label not in all_labels]
    return missing, unused


def merged_group_counts(adata, *, embryo_key: str = "leiden_embryo", merged_key: str = "merged_embryo_v1") -> pd.DataFrame:
    obs = adata.obs[[embryo_key, merged_key]].copy()
    keep = obs[merged_key].notna()
    out = (
        obs.loc[keep]
        .groupby([merged_key, embryo_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values([merged_key, embryo_key])
        .reset_index(drop=True)
    )
    return out


def merged_cluster_averages(
    adata,
    *,
    embryo_key: str = "leiden_embryo",
    merged_key: str = "merged_embryo_v1",
    merged_order: Sequence[str] = MERGED_EMBRYO_ORDER_V1,
) -> pd.DataFrame:
    adata.obs[merged_key] = assign_merged_embryo_labels(adata.obs[embryo_key]).values
    subset = adata[adata.obs[merged_key].notna()].copy()
    averages = cluster_averages_sparse_safe(subset, merged_key)
    keep_order = [label for label in merged_order if label in averages.index]
    return averages.loc[keep_order]


def merged_group_summary_table() -> pd.DataFrame:
    rows = []
    for merged_group in MERGED_EMBRYO_ORDER_V1:
        members = MERGED_EMBRYO_GROUPS_V1[merged_group]
        rows.append(
            {
                "merged_group": merged_group,
                "n_member_clusters": len(members),
                "members": " | ".join(members),
            }
        )
    return pd.DataFrame(rows)


FLOOR_PLATE_PARENT_LABEL_07B = "Week 4 Intermediate-Ventral Spinal Cord"
FLOOR_PLATE_CLUSTER_LABEL_07B = "Floor Plate-like"
OTHER_IVSC_CLUSTER_LABEL_07B = "Other Intermediate-Ventral Spinal Cord"

FLOOR_PLATE_07B_TO_FINAL_GROUP: dict[str, str] = {
    FLOOR_PLATE_CLUSTER_LABEL_07B: "Floor Plate",
    OTHER_IVSC_CLUSTER_LABEL_07B: "Intermediate-Ventral Spinal Cord",
}


MERGED_EMBRYO_GROUPS_V2_NON_IVSC: dict[str, list[str]] = {
    "Forebrain / Midbrain": [
        "Week 4 Forebrain / Midbrain",
        "Week 4 Forebrain - Optic Field",
        "Week 4 Forebrain - Diencephalon / Telencephalon",
        "Week 4 Dorsal Midbrain",
    ],
    "Hindbrain": [
        "Week 4 Hindbrain",
    ],
    "Dorsal Spinal Cord": [
        "Week 4 Dorsal Spinal Cord",
    ],
    "Roof Plate": [
        "Week 4 Roof Plate",
    ],
    "Neural Crest": [
        "Week 3 Neural Crest - Early Migratory",
        "Week 4 Neural Crest - Derivatives",
        "Week 4 Neural Crest - Maturing Cranial",
    ],
    "Immature Neuron": [
        "Week 4 Neuron - CNS Excitatory",
        "Week 4 Neuron - CNS Inhibitory",
        "Week 4 Neuron - Catecholaminergic / Glutamatergic",
        "Week 4 Neuron - Peripheral Sensory",
        "Week 4 Neuron - Autonomic / Cholinergic",
    ],
    "Posterior Neural Tube / NMP": [
        "Week 3 Posterior Neural Tube / Neuromesodermal Progenitors",
    ],
    "Presomitic Mesoderm": [
        "Week 3 Presomitic Mesoderm - Posterior",
        "Week 3 Presomitic Mesoderm - Anterior",
    ],
    "Notochord": [
        "Week 3 / 4 Notochord",
    ],
    "Early Somite": [
        "Week 3 Early Somite",
    ],
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
    "Endothelial": [
        "Week 3 Endothelial",
        "Week 4 Endothelial",
    ],
    "Non-Neural Ectoderm": [
        "Week 4 Non-Neural Ectoderm - Surface Ectoderm",
        "Week 3 Non-Neural Ectoderm",
    ],
}


MERGED_EMBRYO_ORDER_V2: list[str] = [
    "Forebrain / Midbrain",
    "Hindbrain",
    "Floor Plate",
    "Intermediate-Ventral Spinal Cord",
    "Dorsal Spinal Cord",
    "Roof Plate",
    "Neural Crest",
    "Immature Neuron",
    "Posterior Neural Tube / NMP",
    "Notochord",
    "Presomitic Mesoderm",
    "Early Somite",
    "Mature Somite",
    "Intermediate Mesoderm",
    "Lateral Plate Mesoderm",
    "Endothelial",
    "Non-Neural Ectoderm",
]


EXPECTED_MERGED_EMBRYO_BY_MORPH_V2: dict[str, list[str]] = {
    "Forebrain / Midbrain": ["Forebrain / Midbrain"],
    "Hindbrain": ["Hindbrain"],
    "Floor Plate": ["Floor Plate"],
    "Intermediate-Ventral Spinal Cord": ["Intermediate-Ventral Spinal Cord"],
    "Dorsal Spinal Cord": ["Dorsal Spinal Cord"],
    "Roof Plate": ["Roof Plate"],
    "Neural Crest": ["Neural Crest"],
    "Immature Neuron": ["Immature Neuron"],
    "Posterior Neural Tube": ["Posterior Neural Tube / NMP"],
    "Neuromesodermal Progenitors": ["Posterior Neural Tube / NMP"],
    "Notochord": ["Notochord"],
    "Presomitic Mesoderm": ["Presomitic Mesoderm"],
    "Early Somite": ["Early Somite"],
    "Mature Somite": ["Mature Somite"],
    "Intermediate Mesoderm": ["Intermediate Mesoderm"],
    "Lateral Plate Mesoderm": ["Lateral Plate Mesoderm"],
    "Endothelial": ["Endothelial"],
    "Non-Neural Ectoderm": ["Non-Neural Ectoderm"],
}


MERGED_EMBRYO_DISPLAY_LABELS_V2: dict[str, str] = {
    "Forebrain / Midbrain": "FB/MB",
    "Hindbrain": "HB",
    "Floor Plate": "Floor Plate",
    "Intermediate-Ventral Spinal Cord": "Int.-Vent. SC",
    "Dorsal Spinal Cord": "Dorsal SC",
    "Roof Plate": "Roof Plate",
    "Neural Crest": "Neural Crest",
    "Immature Neuron": "Imm. Neuron",
    "Posterior Neural Tube / NMP": "Post NT/NMP",
    "Notochord": "Notochord",
    "Presomitic Mesoderm": "PSM",
    "Early Somite": "Early Somite",
    "Mature Somite": "Mature Somite",
    "Intermediate Mesoderm": "IM",
    "Lateral Plate Mesoderm": "LPM",
    "Endothelial": "Endothelial",
    "Non-Neural Ectoderm": "NNE",
}


def embryo_to_merged_group_map_v2_non_ivsc() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for merged_group, members in MERGED_EMBRYO_GROUPS_V2_NON_IVSC.items():
        for embryo_label in members:
            if embryo_label in mapping:
                raise ValueError(f"Duplicate embryo label in merge scheme: {embryo_label}")
            mapping[embryo_label] = merged_group
    return mapping


def load_floor_plate_reassignment_07b(csv_path) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0)
    if "leiden_embryo_ivsc" not in df.columns:
        raise ValueError("07b floor plate table is missing 'leiden_embryo_ivsc'.")
    reassigned = df["leiden_embryo_ivsc"].map(FLOOR_PLATE_07B_TO_FINAL_GROUP)
    unknown_labels = sorted(set(df["leiden_embryo_ivsc"].dropna()) - set(FLOOR_PLATE_07B_TO_FINAL_GROUP))
    if unknown_labels:
        raise ValueError(f"Unknown 07b floor plate labels: {unknown_labels}")
    return reassigned.astype("object")


def assign_merged_embryo_labels_v2(embryo_obs: pd.DataFrame, floor_plate_reassignment_07b: pd.Series) -> pd.Series:
    if "leiden_embryo" not in embryo_obs.columns:
        raise KeyError("embryo_obs must contain 'leiden_embryo'.")

    mapping = embryo_to_merged_group_map_v2_non_ivsc()
    embryo_labels = embryo_obs["leiden_embryo"].astype(str)
    values = pd.Series([mapping.get(label, pd.NA) for label in embryo_labels], index=embryo_obs.index, dtype="object")

    parent_mask = embryo_labels == FLOOR_PLATE_PARENT_LABEL_07B
    parent_index = embryo_obs.index[parent_mask]
    reassigned_parent = floor_plate_reassignment_07b.reindex(parent_index)
    if reassigned_parent.isna().any():
        missing = reassigned_parent.index[reassigned_parent.isna()].tolist()
        raise ValueError(f"Missing 07b floor plate assignments for {len(missing)} IVSC cells.")
    values.loc[parent_index] = reassigned_parent.values
    return values


def validate_merge_scheme_v2(all_embryo_labels: Sequence[str]) -> tuple[list[str], list[str]]:
    all_labels = list(map(str, all_embryo_labels))
    mapping = embryo_to_merged_group_map_v2_non_ivsc()
    assigned = set(mapping)
    dropped = set(map(str, DROPPED_EMBRYO_LABELS_V1))
    special = {FLOOR_PLATE_PARENT_LABEL_07B}
    missing = [label for label in all_labels if label not in assigned and label not in dropped and label not in special]
    used = assigned | dropped | special
    unused = [label for label in list(used) if label not in all_labels]
    return missing, unused


def merged_group_counts_v2(
    adata,
    *,
    embryo_key: str = "leiden_embryo",
    merged_key: str = "merged_embryo_v2",
) -> pd.DataFrame:
    obs = adata.obs[[embryo_key, merged_key]].copy()
    keep = obs[merged_key].notna()
    out = (
        obs.loc[keep]
        .groupby([merged_key, embryo_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values([merged_key, embryo_key])
        .reset_index(drop=True)
    )
    return out


def merged_cluster_averages_v2(
    adata,
    floor_plate_reassignment_07b: pd.Series,
    *,
    merged_key: str = "merged_embryo_v2",
    merged_order: Sequence[str] = MERGED_EMBRYO_ORDER_V2,
) -> pd.DataFrame:
    adata.obs[merged_key] = assign_merged_embryo_labels_v2(adata.obs[["leiden_embryo"]], floor_plate_reassignment_07b).values
    subset = adata[adata.obs[merged_key].notna()].copy()
    averages = cluster_averages_sparse_safe(subset, merged_key)
    keep_order = [label for label in merged_order if label in averages.index]
    return averages.loc[keep_order]


def merged_embryo_membership_table_v2() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for merged_group, members in MERGED_EMBRYO_GROUPS_V2_NON_IVSC.items():
        for embryo_label in members:
            rows.append(
                {
                    "embryo_label": embryo_label,
                    "status": "merge_keep",
                    "merged_group": merged_group,
                    "assignment_rule": "direct merge from parity 07 embryo label",
                }
            )
    rows.extend(
        [
            {
                "embryo_label": FLOOR_PLATE_PARENT_LABEL_07B,
                "status": "split_keep",
                "merged_group": "Floor Plate",
                "assignment_rule": "07b IVSC subcluster 'Floor Plate-like'",
            },
            {
                "embryo_label": FLOOR_PLATE_PARENT_LABEL_07B,
                "status": "split_keep",
                "merged_group": "Intermediate-Ventral Spinal Cord",
                "assignment_rule": "07b IVSC subcluster 'Other Intermediate-Ventral Spinal Cord'",
            },
        ]
    )
    for embryo_label in DROPPED_EMBRYO_LABELS_V1:
        rows.append(
            {
                "embryo_label": embryo_label,
                "status": "drop",
                "merged_group": "",
                "assignment_rule": "excluded from merged correlation analysis",
            }
        )
    return pd.DataFrame(rows)


def merged_group_summary_table_v2() -> pd.DataFrame:
    source_rules = {
        "Floor Plate": "07b IVSC cells labeled 'Floor Plate-like'",
        "Intermediate-Ventral Spinal Cord": "07b IVSC cells labeled 'Other Intermediate-Ventral Spinal Cord'",
    }
    rows = []
    for merged_group in MERGED_EMBRYO_ORDER_V2:
        if merged_group in source_rules:
            members = FLOOR_PLATE_PARENT_LABEL_07B
            n_members = 1
            rule = source_rules[merged_group]
        else:
            members = " | ".join(MERGED_EMBRYO_GROUPS_V2_NON_IVSC[merged_group])
            n_members = len(MERGED_EMBRYO_GROUPS_V2_NON_IVSC[merged_group])
            rule = "direct merge from parity 07 embryo labels"
        rows.append(
            {
                "merged_group": merged_group,
                "n_member_clusters": n_members,
                "members": members,
                "assignment_rule": rule,
            }
        )
    return pd.DataFrame(rows)


NMP_PARENT_LABEL_07C = "Week 3 Posterior Neural Tube / Neuromesodermal Progenitors"
NMP_CLUSTER_LABEL_07C = "NMP"


MERGED_EMBRYO_GROUPS_V3_NON_OVERRIDDEN: dict[str, list[str]] = {
    "Forebrain / Midbrain": [
        "Week 4 Forebrain / Midbrain",
        "Week 4 Forebrain - Optic Field",
        "Week 4 Forebrain - Diencephalon / Telencephalon",
        "Week 4 Dorsal Midbrain",
    ],
    "Hindbrain": [
        "Week 4 Hindbrain",
    ],
    "Dorsal Spinal Cord": [
        "Week 4 Dorsal Spinal Cord",
    ],
    "Roof Plate": [
        "Week 4 Roof Plate",
    ],
    "Neural Crest": [
        "Week 3 Neural Crest - Early Migratory",
        "Week 4 Neural Crest - Derivatives",
        "Week 4 Neural Crest - Maturing Cranial",
    ],
    "Immature Neuron": [
        "Week 4 Neuron - CNS Excitatory",
        "Week 4 Neuron - CNS Inhibitory",
        "Week 4 Neuron - Catecholaminergic / Glutamatergic",
        "Week 4 Neuron - Peripheral Sensory",
        "Week 4 Neuron - Autonomic / Cholinergic",
    ],
    "Presomitic Mesoderm": [
        "Week 3 Presomitic Mesoderm - Posterior",
        "Week 3 Presomitic Mesoderm - Anterior",
    ],
    "Notochord": [
        "Week 3 / 4 Notochord",
    ],
    "Early Somite": [
        "Week 3 Early Somite",
    ],
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
    "Endothelial": [
        "Week 3 Endothelial",
        "Week 4 Endothelial",
    ],
    "Non-Neural Ectoderm": [
        "Week 4 Non-Neural Ectoderm - Surface Ectoderm",
        "Week 3 Non-Neural Ectoderm",
    ],
}


MERGED_EMBRYO_ORDER_V3: list[str] = [
    "Forebrain / Midbrain",
    "Hindbrain",
    "Floor Plate",
    "Intermediate-Ventral Spinal Cord",
    "Dorsal Spinal Cord",
    "Roof Plate",
    "Neural Crest",
    "Immature Neuron",
    "Posterior NT",
    "NMP",
    "Notochord",
    "Presomitic Mesoderm",
    "Early Somite",
    "Mature Somite",
    "Intermediate Mesoderm",
    "Lateral Plate Mesoderm",
    "Endothelial",
    "Non-Neural Ectoderm",
]


EXPECTED_MERGED_EMBRYO_BY_MORPH_V3: dict[str, list[str]] = {
    "Forebrain / Midbrain": ["Forebrain / Midbrain"],
    "Hindbrain": ["Hindbrain"],
    "Floor Plate": ["Floor Plate"],
    "Intermediate-Ventral Spinal Cord": ["Intermediate-Ventral Spinal Cord"],
    "Dorsal Spinal Cord": ["Dorsal Spinal Cord"],
    "Roof Plate": ["Roof Plate"],
    "Neural Crest": ["Neural Crest"],
    "Immature Neuron": ["Immature Neuron"],
    "Posterior Neural Tube": ["Posterior NT"],
    "Neuromesodermal Progenitors": ["NMP"],
    "Notochord": ["Notochord"],
    "Presomitic Mesoderm": ["Presomitic Mesoderm"],
    "Early Somite": ["Early Somite"],
    "Mature Somite": ["Mature Somite"],
    "Intermediate Mesoderm": ["Intermediate Mesoderm"],
    "Lateral Plate Mesoderm": ["Lateral Plate Mesoderm"],
    "Endothelial": ["Endothelial"],
    "Non-Neural Ectoderm": ["Non-Neural Ectoderm"],
}


MERGED_EMBRYO_DISPLAY_LABELS_V3: dict[str, str] = {
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
}


def embryo_to_merged_group_map_v3_non_overridden() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for merged_group, members in MERGED_EMBRYO_GROUPS_V3_NON_OVERRIDDEN.items():
        for embryo_label in members:
            if embryo_label in mapping:
                raise ValueError(f"Duplicate embryo label in merge scheme: {embryo_label}")
            mapping[embryo_label] = merged_group
    return mapping


def load_nmp_reassignment_07c(csv_path) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0)
    if "leiden_embryo_nmp" not in df.columns:
        raise ValueError("07c NMP table is missing 'leiden_embryo_nmp'.")
    reassigned = df["leiden_embryo_nmp"].map(
        lambda label: "NMP" if str(label) == NMP_CLUSTER_LABEL_07C else "Posterior NT"
    )
    return reassigned.astype("object")


def assign_merged_embryo_labels_v3(
    embryo_obs: pd.DataFrame,
    floor_plate_reassignment_07b: pd.Series,
    nmp_reassignment_07c: pd.Series,
) -> pd.Series:
    if "leiden_embryo" not in embryo_obs.columns:
        raise KeyError("embryo_obs must contain 'leiden_embryo'.")

    mapping = embryo_to_merged_group_map_v3_non_overridden()
    embryo_labels = embryo_obs["leiden_embryo"].astype(str)
    values = pd.Series([mapping.get(label, pd.NA) for label in embryo_labels], index=embryo_obs.index, dtype="object")

    ivsc_mask = embryo_labels == FLOOR_PLATE_PARENT_LABEL_07B
    ivsc_index = embryo_obs.index[ivsc_mask]
    reassigned_ivsc = floor_plate_reassignment_07b.reindex(ivsc_index)
    if reassigned_ivsc.isna().any():
        missing = reassigned_ivsc.index[reassigned_ivsc.isna()].tolist()
        raise ValueError(f"Missing 07b floor plate assignments for {len(missing)} IVSC cells.")
    values.loc[ivsc_index] = reassigned_ivsc.values

    nmp_mask = embryo_labels == NMP_PARENT_LABEL_07C
    nmp_index = embryo_obs.index[nmp_mask]
    reassigned_nmp = nmp_reassignment_07c.reindex(nmp_index)
    if reassigned_nmp.isna().any():
        missing = reassigned_nmp.index[reassigned_nmp.isna()].tolist()
        raise ValueError(f"Missing 07c NMP assignments for {len(missing)} PNT/NMP cells.")
    values.loc[nmp_index] = reassigned_nmp.values
    return values


def validate_merge_scheme_v3(all_embryo_labels: Sequence[str]) -> tuple[list[str], list[str]]:
    all_labels = list(map(str, all_embryo_labels))
    mapping = embryo_to_merged_group_map_v3_non_overridden()
    assigned = set(mapping)
    dropped = set(map(str, DROPPED_EMBRYO_LABELS_V1))
    special = {FLOOR_PLATE_PARENT_LABEL_07B, NMP_PARENT_LABEL_07C}
    missing = [label for label in all_labels if label not in assigned and label not in dropped and label not in special]
    used = assigned | dropped | special
    unused = [label for label in list(used) if label not in all_labels]
    return missing, unused


def merged_group_counts_v3(
    adata,
    *,
    embryo_key: str = "leiden_embryo",
    merged_key: str = "merged_embryo_v3",
) -> pd.DataFrame:
    obs = adata.obs[[embryo_key, merged_key]].copy()
    keep = obs[merged_key].notna()
    out = (
        obs.loc[keep]
        .groupby([merged_key, embryo_key], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values([merged_key, embryo_key])
        .reset_index(drop=True)
    )
    return out


def merged_cluster_averages_v3(
    adata,
    floor_plate_reassignment_07b: pd.Series,
    nmp_reassignment_07c: pd.Series,
    *,
    merged_key: str = "merged_embryo_v3",
    merged_order: Sequence[str] = MERGED_EMBRYO_ORDER_V3,
) -> pd.DataFrame:
    adata.obs[merged_key] = assign_merged_embryo_labels_v3(
        adata.obs[["leiden_embryo"]],
        floor_plate_reassignment_07b,
        nmp_reassignment_07c,
    ).values
    subset = adata[adata.obs[merged_key].notna()].copy()
    averages = cluster_averages_sparse_safe(subset, merged_key)
    keep_order = [label for label in merged_order if label in averages.index]
    return averages.loc[keep_order]


def merged_embryo_membership_table_v3() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for merged_group, members in MERGED_EMBRYO_GROUPS_V3_NON_OVERRIDDEN.items():
        for embryo_label in members:
            rows.append(
                {
                    "embryo_label": embryo_label,
                    "status": "merge_keep",
                    "merged_group": merged_group,
                    "assignment_rule": "direct merge from parity 07 embryo label",
                }
            )
    rows.extend(
        [
            {
                "embryo_label": FLOOR_PLATE_PARENT_LABEL_07B,
                "status": "split_keep",
                "merged_group": "Floor Plate",
                "assignment_rule": "07b IVSC subcluster 'Floor Plate-like'",
            },
            {
                "embryo_label": FLOOR_PLATE_PARENT_LABEL_07B,
                "status": "split_keep",
                "merged_group": "Intermediate-Ventral Spinal Cord",
                "assignment_rule": "all remaining 07b IVSC cells",
            },
            {
                "embryo_label": NMP_PARENT_LABEL_07C,
                "status": "split_keep",
                "merged_group": "NMP",
                "assignment_rule": "07c PNT/NMP subcluster labeled 'NMP'",
            },
            {
                "embryo_label": NMP_PARENT_LABEL_07C,
                "status": "split_keep",
                "merged_group": "Posterior NT",
                "assignment_rule": "all remaining 07c PNT/NMP cells",
            },
        ]
    )
    for embryo_label in DROPPED_EMBRYO_LABELS_V1:
        rows.append(
            {
                "embryo_label": embryo_label,
                "status": "drop",
                "merged_group": "",
                "assignment_rule": "excluded from merged correlation analysis",
            }
        )
    return pd.DataFrame(rows)


def merged_group_summary_table_v3() -> pd.DataFrame:
    source_rules = {
        "Floor Plate": "07b IVSC cells labeled 'Floor Plate-like'",
        "Intermediate-Ventral Spinal Cord": "all remaining 07b IVSC cells",
        "Posterior NT": "all remaining 07c PNT/NMP parent cells",
        "NMP": "07c PNT/NMP cells labeled 'NMP'",
    }
    rows = []
    for merged_group in MERGED_EMBRYO_ORDER_V3:
        if merged_group in source_rules:
            if merged_group in {"Floor Plate", "Intermediate-Ventral Spinal Cord"}:
                members = FLOOR_PLATE_PARENT_LABEL_07B
            else:
                members = NMP_PARENT_LABEL_07C
            n_members = 1
            rule = source_rules[merged_group]
        else:
            members = " | ".join(MERGED_EMBRYO_GROUPS_V3_NON_OVERRIDDEN[merged_group])
            n_members = len(MERGED_EMBRYO_GROUPS_V3_NON_OVERRIDDEN[merged_group])
            rule = "direct merge from parity 07 embryo labels"
        rows.append(
            {
                "merged_group": merged_group,
                "n_member_clusters": n_members,
                "members": members,
                "assignment_rule": rule,
            }
        )
    return pd.DataFrame(rows)
