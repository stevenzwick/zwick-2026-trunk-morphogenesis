"""Declarative recipe for every quantified imaging panel.

Each spec records, for one published panel: the derivative table it is drawn from, the **subset
rule that selects the plotted rows, the join keys**, and the mapping from table columns to the
columns of the paper's Source Data sheet.

Three things here are not incidental detail — each one, omitted, produces a wrong panel or a false
mismatch:

* Subset rules. A Source Data sheet is usually a subset of the table that produced it.
* Join keys. Row order differs between sheet and table; comparing positionally is wrong.
* Float key rounding. Time and distance keys round-trip through Excel, so an exact float join
  silently drops rows.

All of it was recovered by reproducing the published numbers, not by reading the Methods.

Fig 3c is the one quantified imaging panel not specified here. Its sheet publishes unlabelled
lists of per-morph values with nothing to join on, so it is checked as a sorted comparison by
`verify_imaging_panels.check_fig3c` instead.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

#: The sheets name markers in display form; the tables use channel / measurement identifiers.
MARKER_TO_MEASUREMENT = {
    "SOX2": "fixed_sox2_bgz_over_dapi_gate",
    "TBXT": "fixed_t_bgz_over_dapi_gate",
    "FOXF1 (immunostain)": "fixed_tagyfp_bgz_over_dapi_gate",
}
MARKER_TO_MORPHOMETRY = {"MESP2": "mesp2", "PAX8": "pax8", "FOXF1": "foxf1"}

#: Parameters of the responsive-cyst distance gate used by the orientation panels.
#:
#: The gate is NOT a constant. `03_gradient_quantification.ipynb` derives it per dataset by
#: fitting a piecewise responsive-then-plateau regime to per-cyst distance vs
#: `ratio_near_q10_mean` and gating on the fitted breakpoint. The two pSMAD datasets fit
#: DIFFERENT cutoffs — 621.93 µm for Fig 1 (78 of 198 bead-present cysts) and 635.04 µm for
#: ED Fig 2 (156 of 201) — so a single frozen number reproduces one panel and silently
#: corrupts the other. See responsive_cutoff_um().
PLOT2_STANDARD_METRIC_COL = "ratio_near_q10_mean"
PLOT2_RESPONSIVE_MIN_POINTS = 20
PLOT2_RESPONSIVE_MIN_FRACTION = 0.15

PSMAD_F1 = "imaging/psmad_fig1/derived"
PSMAD_E2 = "imaging/psmad_edfig2/derived"
BEAD = "imaging/foxf1_bead/derived"
MORPH = "imaging/morphometry/derived"
TIME = "imaging/timelapse_fig5i/derived"
GRAFT = "imaging/transplant_fig5n/derived"
BMPR1A = "imaging/bmpr1a_psmad/derived"

TRACE_COLS = {"Normalized pSMAD/DAPI (per-cyst bin mean)": "ratio_mean",
              "Per-cyst bin s.d.": "ratio_std", "n pixels": "n_pixels"}
ORIENT_COLS = {"Normalized pSMAD/DAPI (per-cyst bin mean)": "ratio_norm_mean",
               "Per-cyst bin s.d.": "ratio_norm_std", "n pixels": "n_pixels"}
PROXIMAL_COLS = {"Distance to nearest bead d_min (µm)": "distance_d_um",
                 "Bead-proximal pSMAD/DAPI (nearest 10% of shell)": "ratio_near_q10_mean"}
INTENSITY_COLS = {"DAPI-normalized intensity (mean)": "mean", "s.d.": "sd", "s.e.m.": "sem"}

TIMELAPSE_KEYS = {"Treatment key": "treatment_key", "Elapsed time (h)": "elapsed_time_hours"}
PLOTTED_TREATMENTS = ["14hr_basal", "19hr_basal", "19hr_LDN"]


@dataclass(frozen=True)
class PanelSpec:
    panel: str
    table: str
    source_data: str | None
    block: int = 0
    #: Reshape the raw table into one row per published row. Used where a panel is stored
    #: long-form, or on a different time origin, and must be pivoted before comparison.
    prepare: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    subset: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    keys: dict[str, str] = field(default_factory=dict)
    key_values: dict[str, dict[str, str]] = field(default_factory=dict)
    columns: dict[str, str] = field(default_factory=dict)
    round_keys: int = 6
    note: str = ""


#: Fig 5h reports half-max times on the DIFFERENTIATION clock, while the table stores them on the
#: acquisition clock. The offset is exactly 48 h — day 2, when differentiation begins.
HALFMAX_TIME_OFFSET_H = 48.0

#: The two reporters use DIFFERENT sigma thresholds. This is not a typo: RFP is read at sigma4
#: and YFP at sigma3. Check both half-max columns, not the lag: swapping the pairing leaves the
#: cohort MEAN lag almost unchanged, so a check on that alone would pass with the wrong pairing.
HALFMAX_METRIC = {"RFP": "positive_fraction_sigma4", "YFP": "positive_fraction_sigma3"}


def prepare_halfmax(table: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-form half-max table into one row per morph, on the differentiation clock."""
    rows = {}
    for reporter, metric in HALFMAX_METRIC.items():
        sel = table[(table["metric_name"] == metric) & (table["reporter"] == reporter)]
        rows[reporter] = (sel.set_index("position_label")["halfmax_time_hours"]
                          + HALFMAX_TIME_OFFSET_H)
    out = pd.DataFrame(rows)
    out["lag_hours"] = out["YFP"] - out["RFP"]
    return out.reset_index().rename(columns={"position_label": "morph"})


def read_table(rel: str) -> pd.DataFrame:
    """Load a shipped derivative table by its repository-relative path."""
    path = REPO / rel
    return pd.read_csv(path, sep="\t" if path.suffix in {".tsv", ".txt"} else ",")


def plotted_rows(spec: "PanelSpec") -> pd.DataFrame:
    """The exact rows a panel is drawn from: its table, prepared and cut by its own subset rule.

    The renderer and the verifier both go through here. That is deliberate: when the rule that
    selects plotted rows lives anywhere else, the two can disagree, and the verifier checks a
    row set the figure was never drawn from. Fig 1f shipped that way once — rendered from all
    198 cysts while the paper published 78 — and verification passed.
    """
    df = read_table(spec.table)
    if spec.prepare:
        df = spec.prepare(df)
    return spec.subset(df).copy() if spec.subset else df


def orientation_gate(per_cyst_rel: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Subset rule for an orientation panel: keep only the responsive, bead-present cysts.

    Returns a closure so the per-cyst table is read when the rule runs, not at import.
    """
    def gate(trace: pd.DataFrame) -> pd.DataFrame:
        return apply_orientation_gate(trace, read_table(per_cyst_rel))
    return gate


#: ED Fig 3k compares two knockdowns in colony-EDGE cells only. The shipped table is wider than
#: the panel on both axes: it also carries the no-KD condition and the BMP4-INTERIOR category,
#: which belong to the nine-box Supplementary Data 4 figure drawn by `03_figure.py`. Without the
#: subset the table holds 27 rows against the sheet's 12, and the row sets disagree in the
#: direction a left join cannot see.
ED3K_KNOCKDOWNS = {"SCRAMBLE KD": "SCRAMBLE_KD", "BMPR1A KD": "BMPR1A_KD"}
ED3K_CATEGORIES = {"without BMP4": "0 ng/mL BMP4, edge cells",
                   "with BMP4": "10 ng/mL BMP4, edge cells"}
#: The sheet numbers the wells 1-3; the table names the acquisitions they came from.
ED3K_WELLS = {"1": "10x-1", "2": "10x-2", "3": "10x-3"}


def ed3k_plotted(table: pd.DataFrame) -> pd.DataFrame:
    """The four boxes of ED Fig 3k: SCRAMBLE KD and BMPR1A KD, edge cells, without / with BMP4."""
    return table[table["condition"].isin(ED3K_KNOCKDOWNS.values())
                 & table["category"].isin(ED3K_CATEGORIES.values())]


SPECS: list[PanelSpec] = [
    PanelSpec("Fig 1e", f"{PSMAD_F1}/manual_gradient_distance_trace_long.csv", "Fig.1e", 0,
              keys={"Cyst ID": "cyst_id", "Distance from bead d (µm)": "distance_local_um"},
              columns=TRACE_COLS,
              note="Per-cyst binned profile vs distance from bead. Full float precision."),
    PanelSpec("Fig 1e (bead-proximal)", f"{PSMAD_F1}/manual_gradient_per_cyst.csv", "Fig.1e", 1,
              subset=lambda d: d[d["bead_present"]],
              keys={"Cyst ID": "cyst_id"}, columns=PROXIMAL_COLS,
              note="One point per cyst; no-bead controls excluded."),
    PanelSpec("Fig 1f", f"{PSMAD_F1}/manual_gradient_orientation_trace_long.csv", "Fig.1f", 0,
              subset=orientation_gate(f"{PSMAD_F1}/manual_gradient_per_cyst.csv"),
              keys={"Cyst ID": "cyst_id",
                    "Angle relative to bead direction (deg)": "theta_rel_bead_deg"},
              columns=ORIENT_COLS,
              note=("Plotted cysts are gated: bead_present AND distance_d_um <= the FITTED "
                    "responsive cutoff, 621.93 µm for this dataset (78 of 198). The cutoff is "
                    "derived, not assumed — see responsive_cutoff_um().")),
    PanelSpec("ED Fig 2j", f"{PSMAD_E2}/manual_gradient_distance_trace_long.csv",
              "Extended Data Fig.2j", 0,
              keys={"Cyst ID": "cyst_id", "Distance from bead d (µm)": "distance_local_um"},
              columns=TRACE_COLS,
              note="A second, independent pSMAD dataset — NOT a re-run of Fig 1."),
    PanelSpec("ED Fig 2j (bead-proximal)", f"{PSMAD_E2}/manual_gradient_per_cyst.csv",
              "Extended Data Fig.2j", 1, subset=lambda d: d[d["bead_present"]],
              keys={"Cyst ID": "cyst_id"}, columns=PROXIMAL_COLS,
              note="201 of 223 cysts; the 22 dropped are no-bead controls."),
    PanelSpec("ED Fig 2k", f"{PSMAD_E2}/manual_gradient_orientation_trace_long.csv",
              "Extended Data Fig.2k", 0,
              keys={"Cyst ID": "cyst_id",
                    "Angle relative to bead direction (deg)": "theta_rel_bead_deg"},
              subset=orientation_gate(f"{PSMAD_E2}/manual_gradient_per_cyst.csv"),
              columns=ORIENT_COLS,
              note=("Plotted cysts are gated: bead_present AND distance_d_um <= the FITTED "
                    "responsive cutoff, 635.04 µm for this dataset (156 of 201). See "
                    "responsive_cutoff_um().")),
    PanelSpec("Fig 2c", f"{BEAD}/fixed_ratio_distance_trace_equal_support_across_images.tsv",
              "Fig.2c", 0,
              keys={"Marker": "measurement_name", "Distance from bead (µm)": "bin_mid_um"},
              key_values={"Marker": MARKER_TO_MEASUREMENT}, columns=INTENSITY_COLS,
              note=("The ACROSS-IMAGES table, not its pixel-level sibling. Both share an identical "
                    "bin_mid_um column, so the distance axis cannot tell them apart — the value "
                    "columns can.")),
    PanelSpec("ED Fig 3d",
              f"{BEAD}/fixed_center_ratio_distance_trace_equal_support_across_images.tsv",
              "Extended Data Fig.3d", 0,
              keys={"Marker": "measurement_name", "Distance from central cyst (µm)": "bin_mid_um"},
              key_values={"Marker": MARKER_TO_MEASUREMENT}, columns=INTENSITY_COLS,
              subset=lambda d: d[d["measurement_name"] != MARKER_TO_MEASUREMENT["FOXF1 (immunostain)"]],
              note=("SOX2 and TBXT only — the FOXF1 trace is present in the table but not "
                    "plotted, so the subset rule drops it (58 of 87 rows).")),
    PanelSpec("Fig 3f", f"{MORPH}/05g_transverse_distance_profiles_merged_summary.tsv",
              "Fig.3f", 0,
              keys={"Marker": "marker_key", "Mediolateral distance from midline (µm)": "bin_center_um"},
              key_values={"Marker": MARKER_TO_MORPHOMETRY},
              columns={"Positive-area fraction (raw mean)": "mean_fraction",
                       "Positive-area fraction (raw s.d.)": "std_fraction"},
              note=("Merge of two cohorts — baseline (28 morphs) + 2026-01-02 (39) = 67. The "
                    "2024-05-15 cohort was analysed but excluded. 'Relative normalized fraction' "
                    "is mean_fraction / per-marker peak.")),
    PanelSpec("Fig 5i", f"{TIME}/condition_population_mean_sem.csv", "Fig.5i", 0,
              subset=lambda d: d[d["treatment_key"].isin(PLOTTED_TREATMENTS)],
              keys=TIMELAPSE_KEYS,
              columns={"Absolute time (h)": "absolute_hours",
                       "BMP4-YFP signal (bg-subtracted) mean": "bmp4_bgsub_mean_mean",
                       "BMP4-YFP signal s.e.m.": "bmp4_bgsub_mean_sem"},
              note="3 of 7 treatment keys. Same source file as ED Fig 10f, different channel."),
    PanelSpec("ED Fig 10f", f"{TIME}/condition_population_mean_sem.csv",
              "Extended Data Fig.10f", 0,
              subset=lambda d: d[d["treatment_key"].isin(PLOTTED_TREATMENTS)],
              keys=TIMELAPSE_KEYS,
              columns={"Absolute time (h)": "absolute_hours",
                       "FOXF1-RFP signal (bg-subtracted) mean": "foxf1_bgsub_mean_mean",
                       "FOXF1-RFP signal s.e.m.": "foxf1_bgsub_mean_sem"},
              note="Same source file as Fig 5i; reads the FOXF1-RFP pair."),
    PanelSpec("Fig 5h", "imaging/timelapse_3d_fig5h/derived/05_cooperativity_halfmax_times.tsv",
              "Fig.5h", 0, prepare=prepare_halfmax,
              keys={"Morph": "morph"},
              columns={"FOXF1-RFP half-max time (h)": "RFP",
                       "BMP4-YFP half-max time (h)": "YFP",
                       "Lag: BMP4 - FOXF1 (h)": "lag_hours"},
              note=("From the 3D reporter timelapse — a DIFFERENT experiment from Fig 5i/ED 10f, "
                    "which come from the 2D timelapse. Stored long-form on the acquisition clock; "
                    "see prepare_halfmax() for the pivot, the 48 h offset and the per-reporter "
                    "sigma thresholds.")),
    PanelSpec("Fig 5n", f"{GRAFT}/02_foxf1_positive_identity_summary.tsv", "Fig.5n", 0,
              subset=lambda d: d[d["summary_level"] == "file"],
              keys={"Morph (image id)": "image_id"},
              columns={"FOXF1+ pixels (total)": "foxf1_positive_pixels",
                       "FOXF1+ host pixels (strict)": "foxf1_positive_host_strict_pixels",
                       "Host fraction of FOXF1+ (strict)": "foxf1_positive_host_strict_fraction"},
              note=("One row per morph: summary_level == 'file' (9 of 87; the table also holds "
                    "plane, condition and global levels). 'Host %' is the fraction x 100.")),
    PanelSpec("ED Fig 3k", f"{BMPR1A}/negctrl_summary_per_field_30min.csv",
              "Extended Data Fig.3k", 0, subset=ed3k_plotted,
              keys={"Knockdown": "condition", "BMP4": "category", "Replicate well": "field"},
              key_values={"Knockdown": ED3K_KNOCKDOWNS, "BMP4": ED3K_CATEGORIES,
                          "Replicate well": ED3K_WELLS},
              columns={"pSMAD1/5/9 in edge cells, per-well median (a.u.)": "pSMAD_med"},
              note=("12 of the table's 27 rows - edge cells, two knockdowns. The other 15 rows "
                    "belong to the nine-box Supplementary Data 4 figure, not to this panel.")),
]


def responsive_cutoff_um(per_cyst: pd.DataFrame,
                         metric_col: str = PLOT2_STANDARD_METRIC_COL,
                         min_points: int = PLOT2_RESPONSIVE_MIN_POINTS,
                         min_fraction: float = PLOT2_RESPONSIVE_MIN_FRACTION) -> float:
    """Fit the responsive-then-plateau breakpoint in distance, and return it in µm.

    A port of `_fit_plot2_responsive_regime` from
    `imaging/psmad_fig1/notebooks/03_gradient_quantification.ipynb`, reduced to the single
    quantity the figure layer needs. Sweeps every split of the distance-sorted bead-present
    cysts; left of the split fits a line, right of it a constant; keeps the split with the
    smallest total squared error, preferring a negative (responsive) slope exactly as the
    notebook does. The cutoff is the midpoint between the two cysts flanking that split.

    Deriving this rather than hard-coding it is the point: the published Fig 1f and ED Fig 2k
    gates differ (621.93 vs 635.04 µm), and freezing either one corrupts the other panel.
    """
    d = (per_cyst.loc[per_cyst["bead_present"].astype(bool), ["distance_d_um", metric_col]]
         .apply(pd.to_numeric, errors="coerce").dropna().sort_values("distance_d_um"))
    x = d["distance_d_um"].to_numpy(dtype=float)
    y = d[metric_col].to_numpy(dtype=float)
    n = len(d)
    if n < 6:
        raise ValueError(f"responsive_cutoff_um: only {n} usable bead-present cysts")
    min_n = min(max(int(min_points), int(np.ceil(float(min_fraction) * n))), max(2, n // 3))
    if n < 2 * min_n + 1:
        raise ValueError(f"responsive_cutoff_um: {n} cysts cannot support a two-regime split")

    best_any = best_neg = None
    for split in range(min_n, n - min_n + 1):
        x_left, y_left, y_right = x[:split], y[:split], y[split:]
        slope, intercept = np.polyfit(x_left, y_left, deg=1)
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            continue
        sse = float(np.sum((y_left - (slope * x_left + intercept)) ** 2)
                    + np.sum((y_right - np.mean(y_right)) ** 2))
        cand = (sse, float(0.5 * (x[split - 1] + x[split])))
        if best_any is None or sse < best_any[0]:
            best_any = cand
        if slope < 0 and (best_neg is None or sse < best_neg[0]):
            best_neg = cand
    best = best_neg if best_neg is not None else best_any
    if best is None:
        raise ValueError("responsive_cutoff_um: no valid split found")
    return best[1]


def apply_orientation_gate(trace: pd.DataFrame, per_cyst: pd.DataFrame,
                           cutoff_um: float | None = None) -> pd.DataFrame:
    """Restrict an orientation trace to the responsive, bead-present cysts that were plotted.

    `cutoff_um=None` derives the cutoff from this dataset's own per-cyst table, which is what
    the published panels did. Pass a number only to explore an alternative gate.
    """
    if cutoff_um is None:
        cutoff_um = responsive_cutoff_um(per_cyst)
    keep = per_cyst.loc[
        per_cyst["bead_present"] & (per_cyst["distance_d_um"] <= cutoff_um), "cyst_id"]
    return trace[trace["cyst_id"].isin(set(keep))]


def by_panel() -> dict[str, PanelSpec]:
    return {s.panel: s for s in SPECS}
