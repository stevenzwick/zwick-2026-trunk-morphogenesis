"""Pipeline contract for the current morphological quantification workspace.

This module keeps the current reference dataset, notebook chain, and canonical
output expectations in one place so we can adapt the pipeline to future datasets
without losing track of what the current dataset is supposed to produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    cohort_id: str
    raw_data_glob: str
    marker_keys: tuple[str, ...]
    excluded_file_ids: tuple[int, ...]
    expected_total_files: int
    expected_included_files: int


@dataclass(frozen=True)
class TableContract:
    path: str
    required_columns: tuple[str, ...] = ()
    exact_columns: tuple[str, ...] | None = None
    expected_rows: int | None = None


CURRENT_DATASET_PROFILE = DatasetProfile(
    name="day5 trunk morphs with FOXF1 / MESP2 / PAX8 (2026-01-02 dataset 3)",
    cohort_id="2026-01-02_day5_pax8",
    raw_data_glob="data/day5 after fix and stain for PAX8-647/*.czi",
    marker_keys=("mesp2", "foxf1", "pax8"),
    excluded_file_ids=(),
    expected_total_files=40,
    expected_included_files=40,
)


MAINLINE_NOTEBOOKS = (
    "notebooks/00_manifest_qc.ipynb",
    "notebooks/01_per_z_whole_morph_candidates.ipynb",
    "notebooks/02_whole_morph_geometry.ipynb",
    "notebooks/03_posterior_annotation.ipynb",
    "notebooks/04_marker_positive_region_review.ipynb",
    "notebooks/05_domain_quantification.ipynb",
)


AUXILIARY_NOTEBOOKS = (
    "notebooks/04a_off_cyst_background_qc.ipynb",
    "notebooks/04b_representative_positive_region_montage.ipynb",
    "notebooks/05b_day5_feature_handoff.ipynb",
    "notebooks/05c_lpm_size_across_organoids.ipynb",
    "notebooks/05d_mesp2_expression_across_organoids.ipynb",
    "notebooks/05e_pax8_expression_across_organoids.ipynb",
    "notebooks/05f_whole_morph_length_width_across_organoids.ipynb",
    "notebooks/05g_transverse_distance_profiles.ipynb",
    "notebooks/05g_transverse_distance_profiles_merged.ipynb",
)


ALL_NOTEBOOKS = MAINLINE_NOTEBOOKS + AUXILIARY_NOTEBOOKS


CANONICAL_OUTPUT_PATHS = (
    "results/manifests/analysis_manifest.tsv",
    "results/tables/04_global_marker_thresholds.tsv",
    "results/tables/05_posterior_oriented_consensus_axes.tsv",
    "results/tables/05_marker_domain_metrics_by_plane.tsv",
    "results/tables/05_marker_domain_summary_by_file.tsv",
    "results/tables/05_marker_pairwise_metrics_by_plane.tsv",
    "results/tables/05_marker_pairwise_summary_by_file.tsv",
    "results/tables/05b_day5_feature_handoff_table.tsv",
    "results/tables/05b_day5_feature_handoff_table.xlsx",
    "results/tables/05c_lpm_size_summary_by_file.tsv",
    "results/tables/05d_mesp2_expression_summary_by_file.tsv",
    "results/tables/05e_pax8_expression_summary_by_file.tsv",
    "results/tables/05f_whole_morph_geometry_summary_by_file.tsv",
    "results/tables/05g_transverse_distance_profiles_by_file.tsv",
    "results/tables/05g_transverse_distance_profiles_summary.tsv",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_ci.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_ci.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_fraction_norm.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_fraction_norm.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_fraction_norm_ci.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_fraction_norm_ci.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_fraction_minmax.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_fraction_minmax.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_fraction_minmax_ci.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_fraction_minmax_ci.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_raw.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_raw.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_raw_ci.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_raw_ci.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_norm.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_norm.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_norm_ci.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_norm_ci.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_all.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_all.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_all_ci.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_intensity_all_ci.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_soft_threshold_raw.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_soft_threshold_raw.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_soft_threshold_raw_ci.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_soft_threshold_raw_ci.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_soft_threshold_norm.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_soft_threshold_norm.svg",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_soft_threshold_norm_ci.png",
    "results/qc/transverse_distance_profiles/05g_transverse_distance_profiles_soft_threshold_norm_ci.svg",
    "results/tables/05g_transverse_distance_profiles_merged_by_file.tsv",
    "results/tables/05g_transverse_distance_profiles_merged_summary.tsv",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_norm.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_norm.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_norm_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_norm_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_minmax.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_minmax.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_minmax_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_minmax_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_raw.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_raw.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_raw_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_raw_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_norm.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_norm.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_norm_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_norm_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_all.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_all.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_all_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_all_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_raw.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_raw.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_raw_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_raw_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_norm.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_norm.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_norm_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_norm_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_mean_only.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_mean_only.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_mean_only_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_mean_only_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_norm_mean_only.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_norm_mean_only.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_norm_mean_only_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_norm_mean_only_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_minmax_mean_only.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_minmax_mean_only.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_minmax_mean_only_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_fraction_minmax_mean_only_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_raw_mean_only.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_raw_mean_only.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_raw_mean_only_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_raw_mean_only_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_norm_mean_only.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_norm_mean_only.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_norm_mean_only_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_norm_mean_only_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_all_mean_only.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_all_mean_only.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_all_mean_only_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_intensity_all_mean_only_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_raw_mean_only.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_raw_mean_only.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_raw_mean_only_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_raw_mean_only_ci.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_norm_mean_only.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_norm_mean_only.svg",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_norm_mean_only_ci.png",
    "results/qc/transverse_distance_profiles_merged/05g_transverse_distance_profiles_soft_threshold_norm_mean_only_ci.svg",
)


REFERENCE_TABLE_CONTRACTS = (
    TableContract(
        path="results/manifests/analysis_manifest.tsv",
        expected_rows=CURRENT_DATASET_PROFILE.expected_total_files,
        required_columns=(
            "image_id",
            "cohort_id",
            "file_id",
            "file_name",
            "file_path",
            "acquisition_date",
            "size_z",
            "include_in_analysis",
        ),
    ),
    TableContract(
        path="results/tables/04_global_marker_thresholds.tsv",
        required_columns=(
            "marker_key",
            "marker_display_name",
            "sigma_multiple",
            "selected_as_default",
            "threshold_value",
        ),
    ),
    TableContract(
        path="results/tables/05_marker_domain_summary_by_file.tsv",
        expected_rows=CURRENT_DATASET_PROFILE.expected_included_files * len(CURRENT_DATASET_PROFILE.marker_keys),
        required_columns=(
            "file_id",
            "marker_key",
            "marker_display_name",
            "positive_area_um2_median",
            "posterior_extent_um_median",
            "axis_centroid_um_median",
            "side_balance_index_median",
        ),
    ),
    TableContract(
        path="results/tables/05b_day5_feature_handoff_table.tsv",
        expected_rows=CURRENT_DATASET_PROFILE.expected_included_files,
        exact_columns=(
            "trunk_morph_id",
            "image_id",
            "file_name",
            "length_width_ratio_mean",
            "major_axis_length_um_mean",
            "minor_axis_length_um_mean",
            "consensus_axis_length_um",
            "consensus_axis_perp_width_um_mean",
            "consensus_axis_length_width_ratio_mean",
            "overall_shape_note",
            "foxf1_area_um2_mean",
            "foxf1_fraction_mean",
            "foxf1_posterior_edge_distance_um_mean",
            "foxf1_posterior_p10_distance_um_mean",
            "foxf1_posterior_centroid_distance_um_mean",
            "foxf1_bilaterality_index_mean",
            "foxf1_size_note",
            "foxf1_posterior_position_note",
            "foxf1_bilaterality_note",
            "mesp2_area_um2_mean",
            "mesp2_fraction_mean",
            "mesp2_bilaterality_index_mean",
            "mesp2_size_note",
            "mesp2_bilaterality_note",
            "pax8_area_um2_mean",
            "pax8_fraction_mean",
            "pax8_bilaterality_index_mean",
            "pax8_size_note",
            "pax8_bilaterality_note",
            "draft_day5_description",
            "bilateral somites (y/n)",
            "neural tube with lumen (y/n)",
        ),
    ),
    TableContract(
        path="results/tables/05c_lpm_size_summary_by_file.tsv",
        expected_rows=CURRENT_DATASET_PROFILE.expected_included_files,
        required_columns=(
            "file_id",
            "trunk_morph_id",
            "lpm_area_um2_mean_z",
            "lpm_fraction_mean_z",
            "lpm_posterior_extent_um_mean_z",
            "lpm_axis_p10_um_mean_z",
            "lpm_centroid_um_mean_z",
            "lpm_bilaterality_mean_z",
        ),
    ),
    TableContract(
        path="results/tables/05d_mesp2_expression_summary_by_file.tsv",
        expected_rows=CURRENT_DATASET_PROFILE.expected_included_files,
        required_columns=(
            "file_id",
            "trunk_morph_id",
            "mesp2_area_um2_mean_z",
            "mesp2_fraction_mean_z",
            "mesp2_bilaterality_mean_z",
        ),
    ),
    TableContract(
        path="results/tables/05e_pax8_expression_summary_by_file.tsv",
        expected_rows=CURRENT_DATASET_PROFILE.expected_included_files,
        required_columns=(
            "file_id",
            "trunk_morph_id",
            "pax8_area_um2_mean_z",
            "pax8_fraction_mean_z",
            "pax8_bilaterality_mean_z",
        ),
    ),
    TableContract(
        path="results/tables/05f_whole_morph_geometry_summary_by_file.tsv",
        expected_rows=CURRENT_DATASET_PROFILE.expected_included_files,
        required_columns=(
            "file_id",
            "trunk_morph_id",
            "length_um_mean_z",
            "width_um_mean_z",
            "length_width_ratio_mean_z",
            "consensus_axis_length_um",
            "consensus_axis_perp_width_um_mean_z",
            "consensus_axis_length_width_ratio_mean_z",
        ),
    ),
)


def executed_notebook_path(notebook_relpath: str) -> Path:
    notebook = Path(notebook_relpath)
    return PROJECT_ROOT / "results" / "executed_notebooks" / f"{notebook.stem}.executed.ipynb"
