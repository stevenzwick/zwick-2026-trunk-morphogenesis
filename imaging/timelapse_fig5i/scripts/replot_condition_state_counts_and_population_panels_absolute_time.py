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
OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "repro_check"


STATE_COLORS = {
    "+-": "#1f77b4",
    "-+": "#2ca02c",
    "++": "#9467bd",
}

POP_COLORS = {
    "foxf1_bgsub_mean": "#d62728",
    "bmp4_bgsub_mean": "#d4a017",
}


def _pretty_condition(name: str) -> str:
    return name.replace("_", " ")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(BASE_DIR / "condition_state_count_mean_sem.csv")
    population = pd.read_csv(BASE_DIR / "condition_population_mean_sem.csv")

    conditions = list(counts["treatment_key"].drop_duplicates())
    n_conditions = len(conditions)

    fig, axes = plt.subplots(
        2,
        n_conditions,
        figsize=(24, 7),
        sharex=False,
        constrained_layout=True,
    )

    for idx, condition in enumerate(conditions):
        ax_counts = axes[0, idx]
        ax_pop = axes[1, idx]

        cdf = counts[counts["treatment_key"] == condition].sort_values("absolute_hours")
        pdf = population[population["treatment_key"] == condition].sort_values("absolute_hours")

        x_counts = cdf["absolute_hours"].to_numpy()
        x_pop = pdf["absolute_hours"].to_numpy()

        for state, color in STATE_COLORS.items():
            mean_col = f"{state}_mean"
            sem_col = f"{state}_sem"
            y = cdf[mean_col].to_numpy()
            sem = cdf[sem_col].to_numpy()
            ax_counts.plot(x_counts, y, color=color, lw=2.0, label=state)
            ax_counts.fill_between(x_counts, y - sem, y + sem, color=color, alpha=0.2)

        fox = pdf["foxf1_bgsub_mean_mean"].to_numpy()
        fox_sem = pdf["foxf1_bgsub_mean_sem"].to_numpy()
        bmp = pdf["bmp4_bgsub_mean_mean"].to_numpy()
        bmp_sem = pdf["bmp4_bgsub_mean_sem"].to_numpy()

        ax_pop.plot(x_pop, fox, color=POP_COLORS["foxf1_bgsub_mean"], lw=2.5, label="FOXF1-RFP bg-sub")
        ax_pop.fill_between(
            x_pop,
            fox - fox_sem,
            fox + fox_sem,
            color=POP_COLORS["foxf1_bgsub_mean"],
            alpha=0.2,
        )
        ax_pop.plot(x_pop, bmp, color=POP_COLORS["bmp4_bgsub_mean"], lw=2.5, label="BMP4-YFP bg-sub")
        ax_pop.fill_between(
            x_pop,
            bmp - bmp_sem,
            bmp + bmp_sem,
            color=POP_COLORS["bmp4_bgsub_mean"],
            alpha=0.2,
        )

        ax_counts.set_title(_pretty_condition(condition), fontsize=10)
        ax_counts.grid(alpha=0.2)
        ax_pop.grid(alpha=0.2)
        ax_pop.set_xlabel("Absolute hours")

        if idx == 0:
            ax_counts.set_ylabel("Thresholded states\n(absolute counts)")
            ax_pop.set_ylabel("Reporter dynamics\n(bg-sub mean)")

        if idx == 0:
            ax_counts.legend(loc="upper left", fontsize=8, frameon=False)
            ax_pop.legend(loc="upper left", fontsize=8, frameon=False)

    fig.suptitle(
        "Thresholded states (absolute counts) and reporter dynamics by condition (absolute time)",
        fontsize=13,
    )

    png_path = OUT_DIR / "condition_state_counts_and_population_panels_absolute_time_replotted.png"
    svg_path = OUT_DIR / "condition_state_counts_and_population_panels_absolute_time_replotted.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Repro Check",
                "",
                "This folder contains a scratch re-render of",
                "`condition_state_counts_and_population_panels_absolute_time`",
                "using only the summary CSVs in the parent directory:",
                "",
                "- `condition_state_count_mean_sem.csv`",
                "- `condition_population_mean_sem.csv`",
                "",
                "The goal is reproducibility of the plotted trends, not exact style matching.",
                "",
                f"- PNG: `{png_path.name}`",
                f"- SVG: `{svg_path.name}`",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
