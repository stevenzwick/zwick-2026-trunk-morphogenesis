from __future__ import annotations

import re
from typing import Iterable, Mapping

import pandas as pd


THEME_DEFAULT_SIGMA_BY_REPORTER = {
    "RFP": 4,
    "YFP": 3,
}


def _context_prefix(context: str | None) -> str:
    return f"{context}: " if context else ""


def assert_theme_default_sigmas(
    sigma_by_reporter: Mapping[str, int | float],
    *,
    context: str | None = None,
    expected_sigma_by_reporter: Mapping[str, int] = THEME_DEFAULT_SIGMA_BY_REPORTER,
) -> None:
    normalized = {
        str(reporter): int(float(value))
        for reporter, value in sigma_by_reporter.items()
    }
    expected = {
        str(reporter): int(float(value))
        for reporter, value in expected_sigma_by_reporter.items()
    }
    if normalized != expected:
        raise AssertionError(
            f"{_context_prefix(context)}theme-default sigma mismatch: "
            f"expected {expected}, observed {normalized}"
        )


def assert_metric_name_sigma_consistency(
    metric_name_by_reporter: Mapping[str, str],
    *,
    context: str | None = None,
    expected_sigma_by_reporter: Mapping[str, int] = THEME_DEFAULT_SIGMA_BY_REPORTER,
) -> None:
    for reporter, expected_sigma in expected_sigma_by_reporter.items():
        metric_name = metric_name_by_reporter.get(reporter)
        if metric_name is None:
            raise AssertionError(
                f"{_context_prefix(context)}missing metric-name mapping for reporter {reporter}"
            )
        match = re.search(r"sigma(?P<sigma>\d+)", str(metric_name))
        if not match:
            raise AssertionError(
                f"{_context_prefix(context)}metric `{metric_name}` for reporter {reporter} "
                "does not encode a sigma threshold"
            )
        observed_sigma = int(match.group("sigma"))
        if observed_sigma != int(expected_sigma):
            raise AssertionError(
                f"{_context_prefix(context)}metric `{metric_name}` for reporter {reporter} "
                f"uses sigma {observed_sigma}, expected {int(expected_sigma)}"
            )


def assert_background_cutoff_application(
    pre_cut_df: pd.DataFrame,
    post_cut_df: pd.DataFrame,
    cut_start_lookup: Mapping[str, int | float],
    *,
    key_columns: Iterable[str] = ("position_label", "reporter", "time_index"),
    position_column: str = "position_label",
    time_column: str = "time_index",
    context: str | None = None,
) -> None:
    key_columns = list(key_columns)
    required_columns = set(key_columns) | {position_column, time_column}
    missing_pre = required_columns - set(pre_cut_df.columns)
    missing_post = required_columns - set(post_cut_df.columns)
    if missing_pre:
        raise AssertionError(f"{_context_prefix(context)}pre-cut dataframe missing columns: {sorted(missing_pre)}")
    if missing_post:
        raise AssertionError(f"{_context_prefix(context)}post-cut dataframe missing columns: {sorted(missing_post)}")

    pre_keys = pre_cut_df.loc[:, key_columns].drop_duplicates().reset_index(drop=True)
    post_keys = post_cut_df.loc[:, key_columns].drop_duplicates().reset_index(drop=True)

    cut_start_series = pre_cut_df[position_column].map(cut_start_lookup)
    expected_post = (
        pre_cut_df.loc[
            cut_start_series.isna()
            | (pd.to_numeric(pre_cut_df[time_column], errors="coerce") < pd.to_numeric(cut_start_series, errors="coerce")),
            key_columns,
        ]
        .drop_duplicates()
        .sort_values(key_columns)
        .reset_index(drop=True)
    )
    observed_post = post_keys.sort_values(key_columns).reset_index(drop=True)

    if not expected_post.equals(observed_post):
        missing_from_post = expected_post.merge(observed_post, on=key_columns, how="left", indicator=True)
        missing_from_post = missing_from_post.loc[missing_from_post["_merge"] == "left_only", key_columns]
        unexpected_in_post = observed_post.merge(expected_post, on=key_columns, how="left", indicator=True)
        unexpected_in_post = unexpected_in_post.loc[unexpected_in_post["_merge"] == "left_only", key_columns]
        raise AssertionError(
            f"{_context_prefix(context)}background-cutoff application mismatch: "
            f"expected {len(expected_post)} retained keys, observed {len(observed_post)}. "
            f"Missing examples: {missing_from_post.head(5).to_dict('records')}. "
            f"Unexpected examples: {unexpected_in_post.head(5).to_dict('records')}."
        )

    violating = post_cut_df.loc[
        post_cut_df[position_column].map(cut_start_lookup).notna()
        & (
            pd.to_numeric(post_cut_df[time_column], errors="coerce")
            >= pd.to_numeric(post_cut_df[position_column].map(cut_start_lookup), errors="coerce")
        ),
        key_columns,
    ].drop_duplicates()
    if not violating.empty:
        raise AssertionError(
            f"{_context_prefix(context)}retained rows remain at/after cut start: "
            f"{violating.head(5).to_dict('records')}"
        )

    if len(observed_post) > len(pre_keys):
        raise AssertionError(
            f"{_context_prefix(context)}post-cut key count ({len(observed_post)}) exceeds "
            f"pre-cut key count ({len(pre_keys)})"
        )


def assert_no_excluded_keys_in_analysis(
    frame_metrics: pd.DataFrame,
    analysis_df: pd.DataFrame,
    *,
    key_columns: Iterable[str],
    exclude_column: str = "exclude_from_analysis",
    context: str | None = None,
) -> None:
    key_columns = list(key_columns)
    required_frame_columns = set(key_columns) | {exclude_column}
    missing_frame = required_frame_columns - set(frame_metrics.columns)
    missing_analysis = set(key_columns) - set(analysis_df.columns)
    if missing_frame:
        raise AssertionError(
            f"{_context_prefix(context)}frame-metrics dataframe missing columns: {sorted(missing_frame)}"
        )
    if missing_analysis:
        raise AssertionError(
            f"{_context_prefix(context)}analysis dataframe missing key columns: {sorted(missing_analysis)}"
        )

    excluded_keys = (
        frame_metrics.loc[frame_metrics[exclude_column].fillna(False), key_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    analysis_keys = analysis_df.loc[:, key_columns].drop_duplicates().reset_index(drop=True)
    if excluded_keys.empty or analysis_keys.empty:
        return

    overlap = analysis_keys.merge(excluded_keys, on=key_columns, how="inner")
    if not overlap.empty:
        raise AssertionError(
            f"{_context_prefix(context)}excluded keys reappeared downstream: "
            f"{overlap.head(10).to_dict('records')}"
        )
