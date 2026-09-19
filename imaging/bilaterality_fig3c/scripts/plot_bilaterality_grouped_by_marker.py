from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from plot_bilaterality_with_stats import (
    OUTPUT,
    PALETTE,
    ROOT,
    build_plot_dataframe,
    build_reconstructed_rows,
    holm_adjust,
    load_rows,
    p_to_stars,
    stratified_permutation_test,
)


CHANNEL_ORDER = ["MESP2", "PAX8", "FOXF1"]
GROUP_ORDER = ["unilateral bead", "medial bead", "bilateral bead"]
GROUP_HATCHES = {
    "unilateral bead": "///",
    "medial bead": "...",
    "bilateral bead": "xx",
}
GROUP_MARKERS = {
    "unilateral bead": "o",
    "medial bead": "s",
    "bilateral bead": "D",
}
RNG = np.random.default_rng(20260412)


def add_pooled_sig_bar(
    ax,
    x_uni: float,
    x_med: float,
    x_bil: float,
    y: float,
    text: str,
    h: float = 0.018,
) -> None:
    x_pool = (x_med + x_bil) / 2
    ax.plot([x_med, x_bil], [y, y], lw=1.5, c="black")
    ax.plot([x_uni, x_uni, x_pool, x_pool], [y, y + h, y + h, y], lw=1.5, c="black")
    ax.text((x_uni + x_pool) / 2, y + h + 0.004, text, ha="center", va="bottom", fontsize=11)


def make_plot(df, pooled_results: list[dict], stem: str) -> tuple[Path, Path, Path]:
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 6))

    cluster_centers = np.arange(len(CHANNEL_ORDER), dtype=float) * 1.6
    offsets = {
        "unilateral bead": -0.24,
        "medial bead": 0.0,
        "bilateral bead": 0.24,
    }
    width = 0.22

    for center, channel in zip(cluster_centers, CHANNEL_ORDER):
        color = PALETTE[channel]
        for group in GROUP_ORDER:
            values = df[(df["Channel"] == channel) & (df["Group"] == group)]["Score"].to_numpy(dtype=float)
            pos = center + offsets[group]
            bp = ax.boxplot(
                [values],
                positions=[pos],
                widths=width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.6},
                whiskerprops={"color": "black", "linewidth": 1.2},
                capprops={"color": "black", "linewidth": 1.2},
                boxprops={"edgecolor": "black", "linewidth": 1.5},
            )
            box = bp["boxes"][0]
            box.set_facecolor(color)
            box.set_alpha(0.32)
            box.set_hatch(GROUP_HATCHES[group])

            jitter = RNG.normal(0, width * 0.12, size=len(values))
            ax.scatter(
                np.full(len(values), pos, dtype=float) + jitter,
                values,
                s=26,
                c=color,
                alpha=0.55,
                marker=GROUP_MARKERS[group],
                edgecolors="black",
                linewidths=0.35,
                zorder=3,
            )

    results_by_marker = {result["marker"]: result for result in pooled_results}
    y_level = 1.03
    for center, channel in zip(cluster_centers, CHANNEL_ORDER):
        result = results_by_marker[channel]
        label = p_to_stars(result["p_holm"])
        add_pooled_sig_bar(
            ax,
            x_uni=center + offsets["unilateral bead"],
            x_med=center + offsets["medial bead"],
            x_bil=center + offsets["bilateral bead"],
            y=y_level,
            text=label,
        )

    ax.set_xticks(cluster_centers)
    ax.set_xticklabels(CHANNEL_ORDER, fontsize=12)
    for tick in ax.get_xticklabels():
        tick.set_color(PALETTE[tick.get_text()])

    ax.set_ylabel("bilaterality_index_mean")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.12)

    group_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=GROUP_HATCHES[group], label=group)
        for group in GROUP_ORDER
    ]
    ax.legend(
        handles=group_handles,
        title="Bead Group",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT / f"{stem}.pdf"
    png_path = OUTPUT / f"{stem}.png"
    svg_path = OUTPUT / f"{stem}.svg"
    fig.savefig(pdf_path, format="pdf", dpi=300)
    fig.savefig(png_path, format="png", dpi=300)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    return pdf_path, png_path, svg_path


def main() -> None:
    rows = load_rows()
    reconstructed = build_reconstructed_rows(rows)
    df = build_plot_dataframe(reconstructed)

    pooled_results = []
    for channel in ["FOXF1", "PAX8", "MESP2"]:
        pooled_results.append(
            stratified_permutation_test(
                df=df,
                marker=channel,
                group_a="unilateral bead",
                group_b_groups=["medial bead", "bilateral bead"],
            )
        )
    holm_adjust(pooled_results)

    stem = "Orgs_bilaterality_score_grouped_by_marker_with_pooled_stats_mesp2_pax8_foxf1_order"
    pdf_path, png_path, svg_path = make_plot(df, pooled_results, stem)

    print("Grouped-by-marker pooled experiment-stratified permutation tests:")
    for result in pooled_results:
        print(
            f"{result['marker']}: unilateral bead vs medial bead, bilateral bead pooled | "
            f"diff={result['observed_diff']:.6f} | p_two={result['p_two']:.8f} | "
            f"p_holm={result['p_holm']:.8f} | label={p_to_stars(result['p_holm'])}"
        )
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
