"""Helpers for mapping large-image manual bead centroids into small-image space."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


MAPPED_SMALL_CENTROID_COLUMNS = [
    "image_id",
    "cohort_id",
    "canonical_position",
    "small_file_path",
    "large_file_path",
    "annotation_status",
    "mapping_status",
    "centroid_index",
    "centroid_large_x_px",
    "centroid_large_y_px",
    "centroid_small_x_px",
    "centroid_small_y_px",
    "inside_small_fov",
    "small_shape_y_px",
    "small_shape_x_px",
    "small_to_large_top_left_x_px",
    "small_to_large_top_left_y_px",
    "small_to_large_theta_deg",
    "updated_at",
]


def map_large_xy_to_small(
    x_large_px: float,
    y_large_px: float,
    small_shape_yx: Tuple[int, int],
    top_left_x_px: float,
    top_left_y_px: float,
    theta_deg: float,
) -> Tuple[float, float]:
    h, w = small_shape_yx
    cx = 0.5 * (w - 1.0)
    cy = 0.5 * (h - 1.0)
    xr = float(x_large_px) - float(top_left_x_px)
    yr = float(y_large_px) - float(top_left_y_px)
    ang = np.deg2rad(float(theta_deg))
    x0r = xr - cx
    y0r = yr - cy
    x0 = np.cos(ang) * x0r + np.sin(ang) * y0r
    y0 = -np.sin(ang) * x0r + np.cos(ang) * y0r
    return float(x0 + cx), float(y0 + cy)


def load_transform_table(transform_path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(Path(transform_path), sep="\t")


def load_large_centroid_table(centroid_path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(Path(centroid_path), sep="\t")


def build_mapped_manual_centroid_table(
    large_centroid_path: Union[str, Path] = "results/annotations/well_centroids_large.tsv",
    transform_path: Union[str, Path] = "results/annotations/small_to_large_pair_transforms.tsv",
    out_path: Optional[Union[str, Path]] = "results/annotations/well_centroids_small_mapped.tsv",
    cohort_ids: Optional[Sequence[str]] = None,
    canonical_positions: Optional[Sequence[str]] = None,
    require_ok_registration: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    cent_df = load_large_centroid_table(large_centroid_path)
    tr_df = load_transform_table(transform_path)

    if cohort_ids:
        cent_df = cent_df[cent_df["cohort_id"].isin(list(cohort_ids))].copy()
        tr_df = tr_df[tr_df["cohort_id"].isin(list(cohort_ids))].copy()
    if canonical_positions:
        cent_df = cent_df[cent_df["canonical_position"].isin(list(canonical_positions))].copy()
        tr_df = tr_df[tr_df["canonical_position"].isin(list(canonical_positions))].copy()

    keep_cols = [
        "image_id",
        "cohort_id",
        "canonical_position",
        "small_file_path",
        "large_file_path",
        "registration_status",
        "small_shape_y_px",
        "small_shape_x_px",
        "small_to_large_top_left_x_px",
        "small_to_large_top_left_y_px",
        "small_to_large_theta_deg",
    ]
    merged = cent_df.merge(
        tr_df[keep_cols],
        on=[
            "image_id",
            "cohort_id",
            "canonical_position",
            "small_file_path",
            "large_file_path",
        ],
        how="left",
    )

    rows: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        reg_ok = str(row.get("registration_status", "")) == "ok"
        if row.get("annotation_status") != "annotated":
            rows.append(
                {
                    "image_id": row["image_id"],
                    "cohort_id": row["cohort_id"],
                    "canonical_position": row["canonical_position"],
                    "small_file_path": row["small_file_path"],
                    "large_file_path": row["large_file_path"],
                    "annotation_status": row["annotation_status"],
                    "mapping_status": "no_bead",
                    "centroid_index": int(row["centroid_index"]),
                    "centroid_large_x_px": np.nan,
                    "centroid_large_y_px": np.nan,
                    "centroid_small_x_px": np.nan,
                    "centroid_small_y_px": np.nan,
                    "inside_small_fov": False,
                    "small_shape_y_px": row.get("small_shape_y_px", np.nan),
                    "small_shape_x_px": row.get("small_shape_x_px", np.nan),
                    "small_to_large_top_left_x_px": row.get("small_to_large_top_left_x_px", np.nan),
                    "small_to_large_top_left_y_px": row.get("small_to_large_top_left_y_px", np.nan),
                    "small_to_large_theta_deg": row.get("small_to_large_theta_deg", np.nan),
                    "updated_at": row.get("updated_at", ""),
                }
            )
            continue

        if require_ok_registration and not reg_ok:
            rows.append(
                {
                    "image_id": row["image_id"],
                    "cohort_id": row["cohort_id"],
                    "canonical_position": row["canonical_position"],
                    "small_file_path": row["small_file_path"],
                    "large_file_path": row["large_file_path"],
                    "annotation_status": row["annotation_status"],
                    "mapping_status": "missing_registration",
                    "centroid_index": int(row["centroid_index"]),
                    "centroid_large_x_px": float(row["centroid_x_px"]),
                    "centroid_large_y_px": float(row["centroid_y_px"]),
                    "centroid_small_x_px": np.nan,
                    "centroid_small_y_px": np.nan,
                    "inside_small_fov": False,
                    "small_shape_y_px": row.get("small_shape_y_px", np.nan),
                    "small_shape_x_px": row.get("small_shape_x_px", np.nan),
                    "small_to_large_top_left_x_px": row.get("small_to_large_top_left_x_px", np.nan),
                    "small_to_large_top_left_y_px": row.get("small_to_large_top_left_y_px", np.nan),
                    "small_to_large_theta_deg": row.get("small_to_large_theta_deg", np.nan),
                    "updated_at": row.get("updated_at", ""),
                }
            )
            continue

        h = int(row["small_shape_y_px"])
        w = int(row["small_shape_x_px"])
        xs, ys = map_large_xy_to_small(
            x_large_px=float(row["centroid_x_px"]),
            y_large_px=float(row["centroid_y_px"]),
            small_shape_yx=(h, w),
            top_left_x_px=float(row["small_to_large_top_left_x_px"]),
            top_left_y_px=float(row["small_to_large_top_left_y_px"]),
            theta_deg=float(row["small_to_large_theta_deg"]),
        )
        inside = bool((0.0 <= xs < float(w)) and (0.0 <= ys < float(h)))
        rows.append(
            {
                "image_id": row["image_id"],
                "cohort_id": row["cohort_id"],
                "canonical_position": row["canonical_position"],
                "small_file_path": row["small_file_path"],
                "large_file_path": row["large_file_path"],
                "annotation_status": row["annotation_status"],
                "mapping_status": "ok",
                "centroid_index": int(row["centroid_index"]),
                "centroid_large_x_px": float(row["centroid_x_px"]),
                "centroid_large_y_px": float(row["centroid_y_px"]),
                "centroid_small_x_px": float(xs),
                "centroid_small_y_px": float(ys),
                "inside_small_fov": inside,
                "small_shape_y_px": h,
                "small_shape_x_px": w,
                "small_to_large_top_left_x_px": float(row["small_to_large_top_left_x_px"]),
                "small_to_large_top_left_y_px": float(row["small_to_large_top_left_y_px"]),
                "small_to_large_theta_deg": float(row["small_to_large_theta_deg"]),
                "updated_at": row.get("updated_at", ""),
            }
        )

    out_df = pd.DataFrame(rows, columns=MAPPED_SMALL_CENTROID_COLUMNS)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, sep="\t", index=False)
        if verbose:
            print(f"Wrote {out_path}")
    return out_df
