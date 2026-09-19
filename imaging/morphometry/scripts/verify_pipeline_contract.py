"""Verify that the current reference dataset still matches the pipeline contract.

NOT runnable as shipped. This stage reads the raw microscopy images, which are not deposited —
they are available from the lead contact on request, as the paper's Data Availability Statement
states. Its outputs are committed under the lane's `derived/` directory, which is what every
figure script actually reads. See `data/DOWNLOAD.md`.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

try:
    from .pipeline_contract import (
        ALL_NOTEBOOKS,
        CANONICAL_OUTPUT_PATHS,
        CURRENT_DATASET_PROFILE,
        PROJECT_ROOT,
        REFERENCE_TABLE_CONTRACTS,
        executed_notebook_path,
    )
except ImportError:
    from pipeline_contract import (  # type: ignore
        ALL_NOTEBOOKS,
        CANONICAL_OUTPUT_PATHS,
        CURRENT_DATASET_PROFILE,
        PROJECT_ROOT,
        REFERENCE_TABLE_CONTRACTS,
        executed_notebook_path,
    )


def _check_exists(path: Path, label: str, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"Missing {label}: {path.relative_to(PROJECT_ROOT).as_posix()}")


def _check_table_contract(path: Path, failures: list[str]) -> None:
    contract = next(c for c in REFERENCE_TABLE_CONTRACTS if c.path == path.relative_to(PROJECT_ROOT).as_posix())
    df = pd.read_csv(path, sep="\t")

    if contract.expected_rows is not None and len(df) != contract.expected_rows:
        failures.append(
            f"{contract.path}: expected {contract.expected_rows} rows, found {len(df)}"
        )

    missing = [col for col in contract.required_columns if col not in df.columns]
    if missing:
        failures.append(f"{contract.path}: missing required columns {missing}")

    if contract.exact_columns is not None:
        observed = tuple(df.columns.tolist())
        if observed != contract.exact_columns:
            failures.append(
                f"{contract.path}: exact column mismatch\n"
                f"  expected: {list(contract.exact_columns)}\n"
                f"  observed: {list(observed)}"
            )

    rel = contract.path
    if rel.endswith("analysis_manifest.tsv"):
        include_mask = df["include_in_analysis"].fillna(True).astype(bool)
        included_count = int(include_mask.sum())
        if included_count != CURRENT_DATASET_PROFILE.expected_included_files:
            failures.append(
                f"{rel}: expected {CURRENT_DATASET_PROFILE.expected_included_files} included rows, found {included_count}"
            )
        excluded_ids = tuple(sorted(df.loc[~include_mask, "file_id"].astype(int).tolist()))
        if excluded_ids != CURRENT_DATASET_PROFILE.excluded_file_ids:
            failures.append(
                f"{rel}: expected excluded file IDs {CURRENT_DATASET_PROFILE.excluded_file_ids}, found {excluded_ids}"
            )
        cohort_ids = sorted(df["cohort_id"].dropna().astype(str).unique().tolist())
        if cohort_ids != [CURRENT_DATASET_PROFILE.cohort_id]:
            failures.append(
                f"{rel}: expected cohort_id {CURRENT_DATASET_PROFILE.cohort_id}, found {cohort_ids}"
            )

    if rel.endswith("04_global_marker_thresholds.tsv"):
        default_df = df.loc[df["selected_as_default"].fillna(False).astype(bool)].copy()
        default_markers = tuple(sorted(default_df["marker_key"].astype(str).unique().tolist()))
        if default_markers != tuple(sorted(CURRENT_DATASET_PROFILE.marker_keys)):
            failures.append(
                f"{rel}: expected default markers {CURRENT_DATASET_PROFILE.marker_keys}, found {default_markers}"
            )

    if rel.endswith("05_marker_domain_summary_by_file.tsv"):
        markers = tuple(sorted(df["marker_key"].astype(str).unique().tolist()))
        if markers != tuple(sorted(CURRENT_DATASET_PROFILE.marker_keys)):
            failures.append(
                f"{rel}: expected markers {CURRENT_DATASET_PROFILE.marker_keys}, found {markers}"
            )
        unique_files = int(df["file_id"].nunique())
        if unique_files != CURRENT_DATASET_PROFILE.expected_included_files:
            failures.append(
                f"{rel}: expected {CURRENT_DATASET_PROFILE.expected_included_files} unique files, found {unique_files}"
            )

    if rel.endswith("05b_day5_feature_handoff_table.tsv"):
        unique_ids = int(df["trunk_morph_id"].nunique())
        if unique_ids != CURRENT_DATASET_PROFILE.expected_included_files:
            failures.append(
                f"{rel}: expected {CURRENT_DATASET_PROFILE.expected_included_files} unique trunk morph IDs, found {unique_ids}"
            )


def main() -> int:
    failures: list[str] = []

    print("Checking notebook sources and executed snapshots...")
    for rel in ALL_NOTEBOOKS:
        _check_exists(PROJECT_ROOT / rel, "notebook source", failures)
        _check_exists(executed_notebook_path(rel), "executed notebook", failures)

    print("Checking canonical output files...")
    for rel in CANONICAL_OUTPUT_PATHS:
        _check_exists(PROJECT_ROOT / rel, "canonical output", failures)

    print("Checking reference table contracts...")
    for contract in REFERENCE_TABLE_CONTRACTS:
        path = PROJECT_ROOT / contract.path
        if path.exists():
            _check_table_contract(path, failures)

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPASS")
    print(f"Reference dataset: {CURRENT_DATASET_PROFILE.name}")
    print(f"Cohort ID: {CURRENT_DATASET_PROFILE.cohort_id}")
    print(f"Expected included files: {CURRENT_DATASET_PROFILE.expected_included_files}")
    print(f"Excluded file IDs: {CURRENT_DATASET_PROFILE.excluded_file_ids}")
    print(f"Canonical outputs checked: {len(CANONICAL_OUTPUT_PATHS)}")
    print(f"Notebook snapshots checked: {len(ALL_NOTEBOOKS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
