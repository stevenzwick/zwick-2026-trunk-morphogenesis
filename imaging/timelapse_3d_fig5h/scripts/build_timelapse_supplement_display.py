from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PRESENTATION_DIR = RESULTS_DIR / "presentation"
BUNDLE_DIR = RESULTS_DIR / "supplement_display_bundle"
NOTEBOOK_NAME = "foxf1_bmp4_3d_timelapse_supplement_display.ipynb"
HTML_NAME = "foxf1_bmp4_3d_timelapse_supplement_display.html"

ASSETS: dict[str, dict[str, str]] = {
    "mask_review": {
        "repo_rel": "figures/03/03_ilastik_mask_review_by_position.png",
        "bundle_name": "01_ilastik_mask_review_by_position.png",
    },
    "illumination_fields": {
        "repo_rel": "figures/04/04_masked_illumination_fields.png",
        "bundle_name": "02_masked_illumination_fields.png",
    },
    "background_qc_review": {
        "repo_rel": "figures/04b/04b_background_flagged_position_review.png",
        "bundle_name": "03_background_flagged_position_review.png",
    },
    "threshold_summary": {
        "repo_rel": "figures/05/05_reporter_threshold_summary.png",
        "bundle_name": "04_reporter_threshold_summary.png",
    },
    "final_frame_gallery": {
        "repo_rel": "figures/05aa/05aa_final_frame_gallery.png",
        "bundle_name": "05_final_frame_gallery.png",
    },
    "yfp_in_rfp_containment": {
        "repo_rel": "figures/05b/candidate_figures/yfp_in_rfp_containment.png",
        "bundle_name": "06_yfp_in_rfp_containment.png",
    },
    "threshold_independent_density": {
        "repo_rel": "figures/05b/candidate_figures/threshold_independent_pixel_density.png",
        "bundle_name": "07_threshold_independent_pixel_density.png",
    },
    "positive_fraction_trace_summary": {
        "repo_rel": "figures/05c/candidate_figures/positive_fraction_trace_summary.png",
        "bundle_name": "08_positive_fraction_trace_summary.png",
    },
    "positive_fraction_timing_support": {
        "repo_rel": "figures/05c/candidate_figures/positive_fraction_timing_support.png",
        "bundle_name": "09_positive_fraction_timing_support.png",
    },
    "aligned_derivative": {
        "repo_rel": "figures/05d/05d_yfp_halfmax_aligned_derivative.png",
        "bundle_name": "10_yfp_halfmax_aligned_derivative.png",
    },
}


def as_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


CELL_COUNTER = 0


def next_cell_id() -> str:
    global CELL_COUNTER
    CELL_COUNTER += 1
    return f"cell-{CELL_COUNTER:02d}"


def markdown_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": next_cell_id(),
        "metadata": {},
        "source": as_lines(text),
    }


def code_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": next_cell_id(),
        "metadata": {},
        "outputs": [],
        "source": as_lines(text),
    }


def notebook_payload() -> dict[str, object]:
    global CELL_COUNTER
    CELL_COUNTER = 0
    setup_code = f"""
from pathlib import Path

from IPython.display import Image, display


ASSETS = {json.dumps(ASSETS, indent=4)}


def find_asset_root(start: Path) -> tuple[Path, str]:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        direct = candidate / "figures"
        direct_tables = candidate / "tables"
        nested = candidate / "results"
        if direct.exists() and direct_tables.exists():
            return candidate, "project"
        if direct.exists():
            return candidate, "bundle"
        if (nested / "figures").exists() and (nested / "tables").exists():
            return nested, "project"
    raise FileNotFoundError("Could not locate local supplement figures.")


ASSET_ROOT, MODE = find_asset_root(Path.cwd())

required = [
    meta["bundle_name"] if MODE == "bundle" else meta["repo_rel"]
    for meta in ASSETS.values()
]
missing = []
for rel in required:
    path = (ASSET_ROOT / "figures" / rel) if MODE == "bundle" else (ASSET_ROOT / rel)
    if not path.exists():
        missing.append(str(path))
if missing:
    raise FileNotFoundError("Missing saved figures: " + "; ".join(missing))


def resolve_asset(key: str) -> Path:
    meta = ASSETS[key]
    if MODE == "bundle":
        return ASSET_ROOT / "figures" / meta["bundle_name"]
    return ASSET_ROOT / meta["repo_rel"]


def show_figure(key: str, width: int = 1400) -> None:
    display(Image(filename=str(resolve_asset(key)), width=width))
""".strip()

    cells = [
        markdown_cell(
            """
# FOXF1/BMP4 3D Timelapse Supplement

This notebook organizes the saved figures from the `FOXF1_BMP4_3d_timelapse` pipeline into a compact supplementary sequence. The analyzed cohort contains `65` organoids, with one tracked position per organoid and frame-window exclusions applied only where masking, background stability, or border contact would otherwise bias the quantitative summaries. The two endpoint claims emphasized here are that `BMP4-YFP` appears within an existing `FOXF1-RFP` domain and that the `BMP4-YFP` positive-fraction rise follows the `FOXF1-RFP` rise in time. A final alignment panel is included only as supportive evidence that the shared `BMP4-YFP` transition becomes easier to read after position-wise temporal registration.
            """.strip()
        ),
        code_cell(setup_code),
        markdown_cell(
            """
## 1. Mask QC defines the organoid measurement domain used throughout the timelapse analysis

These panels show the retained-mask review after the phase-to-ilastik handoff. The important pattern is that the analysis domain follows the organoid body rather than the reporter-positive subregions, while obvious clipping, off-frame growth, or unstable boundaries are handled explicitly instead of being silently pooled into the measurements. This matters because every whole-organoid and positive-domain summary later in the pipeline inherits the same mask definition. The panel is diagnostic support rather than a biological endpoint, but it establishes that the downstream FOXF1/BMP comparisons are made within a consistent organoid domain.
            """.strip()
        ),
        code_cell("show_figure('mask_review', width=1450)"),
        markdown_cell(
            """
## 2. Illumination correction and background QC stabilize reporter measurements before comparison

These figures summarize the two main preprocessing steps used before reporter interpretation. The illumination fields correct broad channel-wide structure, and the background-QC review makes explicit which positions require frame-window cuts because late background drift would otherwise distort the reporter traces. Together they show that the later FOXF1/BMP ordering is not driven by obvious whole-frame illumination or unstable background structure. These panels are preprocessing support rather than the main claim, but they justify treating the corrected traces as quantitatively comparable across time.
            """.strip()
        ),
        code_cell(
            """
show_figure('illumination_fields', width=1450)
show_figure('background_qc_review', width=1450)
            """.strip()
        ),
        markdown_cell(
            """
## 3. Shared within-cyst thresholds define the reporter-positive domains used downstream

The threshold summary condenses the reporter-positive definitions used in the manuscript-oriented notebooks. The key point is that the positive-domain comparisons are anchored to shared within-cyst baseline-plus-sigma rules rather than ad hoc per-panel tuning, with the default comparisons using `FOXF1-RFP 4σ` and `BMP4-YFP 3σ`. That consistency matters because the same thresholds feed both the spatial containment summaries and the temporal ordering summaries below. This panel supports interpretation of the domain-based metrics rather than standing as an endpoint on its own.
            """.strip()
        ),
        code_cell("show_figure('threshold_summary', width=1400)"),
        markdown_cell(
            """
## 4. Final-frame galleries provide qualitative context for the quantified endpoint

The final-frame gallery shows representative end-state cysts from the analyzed cohort and gives image-level context for the later spatial and temporal summaries. The visible pattern is that FOXF1-positive domains are already established at the positions where BMP4 reporter signal is later assessed, so the overlap analyses below are tied back to real image structure rather than abstract traces alone. This panel is shown as context rather than as a quantified test. It helps connect the statistical summaries to the range of cyst morphologies and reporter distributions present in the cohort.
            """.strip()
        ),
        code_cell("show_figure('final_frame_gallery', width=1450)"),
        markdown_cell(
            """
## 5. BMP4-YFP appears within an existing FOXF1-RFP domain

These are the primary spatial endpoint panels. The containment summary asks whether `BMP4-YFP` positive pixels fall inside an existing `FOXF1-RFP` positive context, while the threshold-independent density view checks the same relationship without depending entirely on a single YFP threshold choice. Both views support the same conclusion: BMP reporter activation is concentrated inside the FOXF1 domain rather than appearing as an independent outside territory. The density-style panel is supportive rather than primary, but it shows that the spatial nesting pattern survives a less threshold-bound representation.
            """.strip()
        ),
        code_cell(
            """
show_figure('yfp_in_rfp_containment', width=1300)
show_figure('threshold_independent_density', width=1300)
            """.strip()
        ),
        markdown_cell(
            """
## 6. Clock-time population summaries show FOXF1-RFP precedes BMP4-YFP

These panels provide the primary temporal ordering support. The clock-time positive-fraction summary shows the earlier rise of `FOXF1-RFP` relative to `BMP4-YFP`, and the timing-support panel asks whether that ordering persists when positions are compared through their own lag distribution rather than only through the pooled mean. Together they support the interpretation that BMP reporter activation follows FOXF1 activation across the cohort rather than preceding it. The per-position timing panel is especially important because it guards against the pooled mean being driven by only a small subset of organoids.
            """.strip()
        ),
        code_cell(
            """
show_figure('positive_fraction_trace_summary', width=1300)
show_figure('positive_fraction_timing_support', width=1300)
            """.strip()
        ),
        markdown_cell(
            """
## 7. Aligned BMP4-YFP dynamics provide supportive evidence for a shared delayed transition

This aligned panel is included as compact supportive evidence rather than as a separate mechanistic claim. Aligning positions to their own `BMP4-YFP` half-max sharpens the aggregate YFP progression and makes the delayed reporter transition easier to read, while the derivative view summarizes where the aligned rise is most concentrated. The useful point here is that the same delayed BMP reporter progression becomes cleaner after timing heterogeneity is reduced across positions. It is shown as a descriptive alignment view, not as evidence for a specific positive-feedback model.
            """.strip()
        ),
        code_cell("show_figure('aligned_derivative', width=1300)"),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_notebook() -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    notebook_path = PRESENTATION_DIR / NOTEBOOK_NAME
    notebook_path.write_text(json.dumps(notebook_payload(), indent=2) + "\n")
    return notebook_path


def stage_bundle() -> None:
    notebook_path = PRESENTATION_DIR / NOTEBOOK_NAME
    html_path = PRESENTATION_DIR / HTML_NAME
    if not notebook_path.exists():
        raise FileNotFoundError(f"Missing executed presentation notebook: {notebook_path}")
    if not html_path.exists():
        raise FileNotFoundError(f"Missing HTML export: {html_path}")

    figures_dir = BUNDLE_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, BUNDLE_DIR / NOTEBOOK_NAME)
    shutil.copy2(html_path, BUNDLE_DIR / HTML_NAME)
    for meta in ASSETS.values():
        source_path = RESULTS_DIR / meta["repo_rel"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing figure for bundle staging: {source_path}")
        shutil.copy2(source_path, figures_dir / meta["bundle_name"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or stage the FOXF1/BMP4 timelapse supplement display.")
    parser.add_argument(
        "--stage-bundle",
        action="store_true",
        help="Copy the executed notebook, HTML, and referenced PNGs into results/supplement_display_bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage_bundle:
        stage_bundle()
        return
    notebook_path = build_notebook()
    print(f"Wrote notebook: {notebook_path}")


if __name__ == "__main__":
    main()
