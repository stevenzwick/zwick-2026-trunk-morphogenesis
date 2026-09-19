#!/usr/bin/env python3
"""Runs as shipped. It reads only the two tables in this lane's `derived/` directory and needs
no images and nothing configured. It redraws the panel from the measured values, as a check that
the shipped tables carry what the published figure shows.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Both input tables ship in this repository.
BASE_DIR = Path(__file__).resolve().parent.parent / "derived"
# Write beside the repository, not into the shipped data directory.
OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "reorganized_overlays"

CONDITIONS = ["14hr_basal", "19hr_basal", "19hr_LDN"]
LABELS = {
    "14hr_basal": "14 hr basal",
    "19hr_basal": "19 hr basal",
    "19hr_LDN": "19 hr LDN",
}
COLORS = {
    "14hr_basal": "#1f77b4",
    "19hr_basal": "#d62728",
    "19hr_LDN": "#2ca02c",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BASE_DIR / "condition_population_mean_sem.csv")
    df = df[df["treatment_key"].isin(CONDITIONS)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True, sharex=True)
    specs = [
        ("foxf1_bgsub_mean_mean", "foxf1_bgsub_mean_sem", "FOXF1-RFP bg-sub", axes[0]),
        ("bmp4_bgsub_mean_mean", "bmp4_bgsub_mean_sem", "BMP4-YFP bg-sub", axes[1]),
    ]

    for mean_col, sem_col, title, ax in specs:
        for condition in CONDITIONS:
            sdf = df[df["treatment_key"] == condition].sort_values("absolute_hours")
            x = sdf["absolute_hours"].to_numpy()
            y = sdf[mean_col].to_numpy()
            sem = sdf[sem_col].to_numpy()
            color = COLORS[condition]
            ax.plot(x, y, lw=2.5, color=color, label=LABELS[condition])
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18)

        ax.set_title(title)
        ax.set_xlabel("Absolute hours")
        ax.set_ylabel("Mean +/- SEM")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    fig.suptitle("Selected Condition Overlays", fontsize=13)

    png_path = OUT_DIR / "foxf1_bmp4_selected_condition_overlays_absolute_time.png"
    svg_path = OUT_DIR / "foxf1_bmp4_selected_condition_overlays_absolute_time.svg"
    pdf_path = OUT_DIR / "foxf1_bmp4_selected_condition_overlays_absolute_time.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Selected Condition Overlays",
                "",
                "Two-panel overlay from `condition_population_mean_sem.csv`.",
                "",
                "Conditions:",
                "- 14 hr basal",
                "- 19 hr basal",
                "- 19 hr LDN",
                "",
                "Panels:",
                "- FOXF1-RFP bg-sub",
                "- BMP4-YFP bg-sub",
                "",
                f"- PNG: `{png_path.name}`",
                f"- SVG: `{svg_path.name}`",
                f"- PDF: `{pdf_path.name}`",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
