"""Render the quantified imaging panels from the shipped derivative layer.

Needs nothing but this repository — no raw images, no Source Data, no saved objects.

    python figures/render_imaging_panels.py

Output lands in figures/output/. See _plotting.py on how these differ from the published panels.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "figures"))
from _plotting import trace_panel, save                        # noqa: E402
from panel_specs import by_panel, plotted_rows, MARKER_TO_MEASUREMENT  # noqa: E402


def read(rel: str) -> pd.DataFrame:
    path = REPO / rel
    return pd.read_csv(path, sep="\t" if path.suffix in {".tsv", ".txt"} else ",")


def psmad_distance(panel: str, name: str):
    spec = by_panel()[panel]
    table = plotted_rows(spec)
    binned = table.groupby("distance_local_um")["ratio_mean"].agg(["mean", "std"]).reset_index()
    fig = trace_panel([("all cysts", binned["distance_local_um"], binned["mean"], binned["std"])],
                      "Distance from bead (µm)", "Normalized pSMAD/DAPI", f"{panel} — pSMAD vs distance")
    save(fig, name)


def psmad_orientation(panel: str, name: str):
    spec = by_panel()[panel]
    trace = plotted_rows(spec)
    binned = trace.groupby("theta_rel_bead_deg")["ratio_norm_mean"].agg(["mean", "std"]).reset_index()
    fig = trace_panel([("responsive cysts", binned["theta_rel_bead_deg"], binned["mean"], binned["std"])],
                      "Angle relative to bead (deg)", "Normalized pSMAD/DAPI",
                      f"{panel} — pSMAD vs bead direction")
    save(fig, name)


def marker_profiles(panel: str, name: str, xlabel: str):
    spec = by_panel()[panel]
    table = plotted_rows(spec)
    reverse = {v: k for k, v in MARKER_TO_MEASUREMENT.items()}
    groups = [(reverse.get(m, m), g["bin_mid_um"], g["mean"], g["sem"])
              for m, g in table.groupby("measurement_name")]
    fig = trace_panel(groups, xlabel, "DAPI-normalized intensity", f"{panel} — marker profiles")
    save(fig, name)


def mediolateral():
    spec = by_panel()["Fig 3f"]
    table = plotted_rows(spec)
    groups = [(m.upper(), g["bin_center_um"], g["mean_fraction"], g["std_fraction"])
              for m, g in table.groupby("marker_key")]
    fig = trace_panel(groups, "Mediolateral distance from midline (µm)", "Positive-area fraction",
                      "Fig 3f — mediolateral marker profiles")
    save(fig, "fig3f_mediolateral")


def timelapse(panel: str, mean_col: str, sem_col: str, label: str, name: str):
    spec = by_panel()[panel]
    table = plotted_rows(spec)
    groups = [(k, g["absolute_hours"], g[mean_col], g[sem_col])
              for k, g in table.groupby("treatment_key")]
    fig = trace_panel(groups, "Time (h)", f"{label} (bg-subtracted)", f"{panel} — reporter traces")
    save(fig, name)


def bilaterality():
    """Fig 3c. Drawn from the published values, not the later feature-table export —
    see imaging/bilaterality_fig3c/README.md on why the two differ."""
    import matplotlib.pyplot as plt
    table = read("imaging/bilaterality_fig3c/derived/fig3c_bilaterality_index_plotted.csv")
    conditions = ["Unilateral bead", "Medial bead", "Bilateral beads"]
    markers = ["FOXF1", "PAX8", "MESP2"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    positions, labels = [], []
    for i, marker in enumerate(markers):
        for j, condition in enumerate(conditions):
            values = table[(table["marker"] == marker) &
                           (table["condition"] == condition)]["bilaterality_index"]
            x = i * (len(conditions) + 1) + j
            ax.boxplot(values, positions=[x], widths=0.65, showfliers=False)
            ax.scatter([x] * len(values), values, s=12, zorder=3, edgecolors="none", alpha=0.7)
            positions.append(x)
            labels.append(f"{condition.split()[0]}\n(n={len(values)})")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6)
    for i, marker in enumerate(markers):
        ax.text(i * (len(conditions) + 1) + 1, 1.06, marker, ha="center", fontsize=9)
    ax.set_ylabel("Bilaterality index")
    ax.set_ylim(-0.05, 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig3c_bilaterality")


def main() -> int:
    print("Rendering imaging panels from the shipped derivative layer:")
    psmad_distance("Fig 1e", "fig1e_psmad_distance")
    psmad_orientation("Fig 1f", "fig1f_psmad_orientation")
    psmad_distance("ED Fig 2j", "ed2j_psmad_distance")
    psmad_orientation("ED Fig 2k", "ed2k_psmad_orientation")
    marker_profiles("Fig 2c", "fig2c_bead_profiles", "Distance from bead (µm)")
    marker_profiles("ED Fig 3d", "ed3d_center_profiles", "Distance from central cyst (µm)")
    mediolateral()
    bilaterality()
    timelapse("Fig 5i", "bmp4_bgsub_mean_mean", "bmp4_bgsub_mean_sem", "BMP4-YFP",
              "fig5i_bmp4_traces")
    timelapse("ED Fig 10f", "foxf1_bgsub_mean_mean", "foxf1_bgsub_mean_sem", "FOXF1-RFP",
              "ed10f_foxf1_traces")
    print("\nDone. These are pre-Illustrator plots — see figures/_plotting.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
