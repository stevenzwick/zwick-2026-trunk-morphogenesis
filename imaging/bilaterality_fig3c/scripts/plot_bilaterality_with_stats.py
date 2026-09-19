from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
#: Rendered panels go to the lane's output/ directory, as in every other lane, so that running this
#: script never modifies a tracked file. Until 2026-09-18 it wrote them beside itself in scripts/.
OUTPUT = ROOT.parent / "output"
#: The two per-experiment feature tables. In the working directory these sat in dated subfolders
#: (`2026-01-02/05b_day5_feature_handoff_table.tsv`); here they are flattened into `derived/` with
#: the date in the filename, so the experiment label is parsed from the name rather than the parent.
DERIVED = ROOT.parent / "derived"
EXPERIMENTS = ["2026-01-02", "2026-02-27"]
TABLES = [DERIVED / f"05b_day5_feature_handoff_table__{e}.tsv" for e in EXPERIMENTS]

# Reconstructed organoid IDs underlying the notebook figure.
PLOT_IDS = {
    "unilateral bead": {
        "2026-01-02": ["30", "3202", "24", "39", "01", "35", "14", "15", "38", "21", "09", "12", "07"],
        "2026-02-27": ["04", "05", "06", "09", "20", "22"],
    },
    "medial bead": {
        "2026-01-02": ["29", "17", "33", "28", "27", "05", "04", "02", "18", "11"],
        "2026-02-27": ["07", "08", "11", "12", "15", "17", "18"],
    },
    "bilateral bead": {
        "2026-01-02": ["06", "20", "31", "10", "34"],
        "2026-02-27": ["19", "13"],
    },
}

MARKERS = [
    ("FOXF1", "foxf1_bilaterality_index_mean"),
    ("PAX8", "pax8_bilaterality_index_mean"),
    ("MESP2", "mesp2_bilaterality_index_mean"),
]

GROUP_ORDER = ["unilateral bead", "medial bead", "bilateral bead"]
CHANNEL_ORDER = ["FOXF1", "PAX8", "MESP2"]
ALT_CHANNEL_ORDER = ["MESP2", "PAX8", "FOXF1"]
PALETTE = {"FOXF1": "blue", "PAX8": "green", "MESP2": "red"}
PRIMARY_PERMUTATIONS = 1_000_000
RNG = np.random.default_rng(20260412)
#: seaborn's stripplot jitter draws from the legacy global state, not from RNG, so seeding
#: RNG alone leaves the rendered point cloud different on every run.
np.random.seed(20260412)


def load_rows() -> list[dict]:
    rows = []
    for path in TABLES:
        experiment = path.stem.split("__")[-1]
        with path.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row = dict(row)
                row["experiment"] = experiment
                for _, column in MARKERS:
                    try:
                        row[column] = float(row[column])
                    except Exception:
                        row[column] = None
                rows.append(row)
    return rows


def build_reconstructed_rows(rows: list[dict]) -> list[dict]:
    reconstructed = []
    for group, experiments in PLOT_IDS.items():
        for experiment, trunk_ids in experiments.items():
            for trunk_id in trunk_ids:
                matches = [
                    row
                    for row in rows
                    if row["experiment"] == experiment and row["trunk_morph_id"] == trunk_id
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one row for {group} {experiment} {trunk_id}, found {len(matches)}"
                    )
                record = dict(matches[0])
                record["group"] = group
                reconstructed.append(record)
    return reconstructed


def build_plot_dataframe(reconstructed: list[dict]) -> pd.DataFrame:
    plot_rows = []
    for row in reconstructed:
        for channel, column in MARKERS:
            value = row[column]
            if value is None:
                continue
            plot_rows.append(
                {
                    "Group": row["group"],
                    "Channel": channel,
                    "Score": value,
                    "Experiment": row["experiment"],
                    "Organoid": row["trunk_morph_id"],
                }
            )
    return pd.DataFrame(plot_rows)


def stratified_permutation_test(
    df: pd.DataFrame,
    marker: str,
    group_a: str,
    group_b_groups: list[str],
    n_perm: int = PRIMARY_PERMUTATIONS,
) -> dict:
    subset = df[(df["Channel"] == marker) & (df["Group"].isin([group_a, *group_b_groups]))].copy()
    subset["label"] = (subset["Group"] == group_a).astype(int)
    values = subset["Score"].to_numpy(dtype=float)
    labels = subset["label"].to_numpy(dtype=int)
    experiments = subset["Experiment"].to_numpy(dtype=object)
    total_n = len(values)

    def statistic(vals: np.ndarray, labs: np.ndarray, exps: np.ndarray) -> tuple[float, list[tuple]]:
        total = 0.0
        details = []
        for experiment in sorted(set(exps)):
            idx = np.where(exps == experiment)[0]
            vals_exp = vals[idx]
            labs_exp = labs[idx]
            a_vals = vals_exp[labs_exp == 1]
            b_vals = vals_exp[labs_exp == 0]
            diff = a_vals.mean() - b_vals.mean()
            total += len(idx) * diff
            details.append(
                (
                    experiment,
                    len(a_vals),
                    len(b_vals),
                    float(a_vals.mean()),
                    float(b_vals.mean()),
                    float(diff),
                )
            )
        return float(total / total_n), details

    observed, details = statistic(values, labels, experiments)
    strata = []
    for experiment in sorted(set(experiments)):
        idx = np.where(experiments == experiment)[0]
        strata.append(
            (
                values[idx],
                int(labels[idx].sum()),
                len(idx),
            )
        )

    lower = 0
    upper = 0
    absolute = 0
    chunk_size = 50_000
    done = 0
    while done < n_perm:
        batch = min(chunk_size, n_perm - done)
        stats = np.zeros(batch, dtype=float)
        for vals_exp, k, n in strata:
            rand = RNG.random((batch, n))
            chosen = np.argpartition(rand, k - 1, axis=1)[:, :k]
            a_sum = vals_exp[chosen].sum(axis=1)
            total_sum = vals_exp.sum()
            b_sum = total_sum - a_sum
            diff = (a_sum / k) - (b_sum / (n - k))
            stats += (n / total_n) * diff
        lower += int((stats <= observed).sum())
        upper += int((stats >= observed).sum())
        absolute += int((np.abs(stats) >= abs(observed)).sum())
        done += batch

    return {
        "marker": marker,
        "group_a": group_a,
        "group_b_groups": list(group_b_groups),
        "observed_diff": observed,
        "details": details,
        "p_two": (absolute + 1) / (n_perm + 1),
        "p_lower": (lower + 1) / (n_perm + 1),
        "p_upper": (upper + 1) / (n_perm + 1),
    }


def holm_adjust(results: list[dict]) -> None:
    order = sorted(range(len(results)), key=lambda idx: results[idx]["p_two"])
    max_so_far = 0.0
    total = len(results)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (total - rank) * results[idx]["p_two"])
        max_so_far = max(max_so_far, adjusted)
        results[idx]["p_holm"] = max_so_far


def p_to_stars(p_value: float) -> str:
    if p_value <= 0.0001:
        return "****"
    if p_value <= 0.001:
        return "***"
    if p_value <= 0.01:
        return "**"
    if p_value <= 0.05:
        return "*"
    return "ns"


def add_sig_bar(ax, x1: float, x2: float, y: float, text: str, h: float = 0.015) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="black")
    ax.text((x1 + x2) / 2, y + h + 0.004, text, ha="center", va="bottom", fontsize=11)


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


def hue_offsets(channel_order: list[str], width: float = 0.5) -> dict[str, float]:
    if len(channel_order) == 1:
        return {channel_order[0]: 0.0}
    positions = np.linspace(-width / 3, width / 3, len(channel_order))
    return {channel: float(pos) for channel, pos in zip(channel_order, positions)}


def make_plot(
    df: pd.DataFrame,
    pooled_results: list[dict],
    channel_order: list[str],
    stem: str,
) -> tuple[Path, Path, Path]:
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 6))
    box_width = 0.5

    sns.boxplot(
        data=df,
        x="Group",
        y="Score",
        hue="Channel",
        order=GROUP_ORDER,
        hue_order=channel_order,
        palette=PALETTE,
        width=box_width,
        boxprops=dict(alpha=0.3),
        fliersize=0,
        linewidth=1.5,
        ax=ax,
    )

    sns.stripplot(
        data=df,
        x="Group",
        y="Score",
        hue="Channel",
        order=GROUP_ORDER,
        hue_order=channel_order,
        palette=PALETTE,
        dodge=True,
        jitter=0.18,
        size=4,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.45,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:3],
        labels[:3],
        title="Channel",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.set_ylabel("bilaterality_index_mean")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.22)

    offsets = hue_offsets(channel_order, width=box_width)
    y_levels = {"FOXF1": 1.03, "PAX8": 1.11, "MESP2": 1.19}

    group_centers = {group: idx for idx, group in enumerate(GROUP_ORDER)}
    for result in pooled_results:
        marker = result["marker"]
        label = p_to_stars(result["p_holm"])
        x_uni = group_centers[result["group_a"]] + offsets[marker]
        x_med = group_centers["medial bead"] + offsets[marker]
        x_bil = group_centers["bilateral bead"] + offsets[marker]
        add_pooled_sig_bar(ax, x_uni, x_med, x_bil, y_levels[marker], label)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
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
    for marker, _ in MARKERS:
        pooled_results.append(
            stratified_permutation_test(
                df=df,
                marker=marker,
                group_a="unilateral bead",
                group_b_groups=["medial bead", "bilateral bead"],
            )
        )
    holm_adjust(pooled_results)

    output_sets = [
        (
            CHANNEL_ORDER,
            "Orgs_bilaterality_score_three_channels_with_pooled_stats",
            "Saved default channel order",
        ),
        (
            ALT_CHANNEL_ORDER,
            "Orgs_bilaterality_score_three_channels_with_pooled_stats_mesp2_pax8_foxf1_order",
            "Saved alternate channel order",
        ),
    ]

    saved_paths = []
    for channel_order, stem, _label in output_sets:
        saved_paths.append((channel_order, *make_plot(df, pooled_results, channel_order, stem)))

    print("Pooled experiment-stratified permutation tests used for plot annotations:")
    for result in pooled_results:
        print(
            f"{result['marker']}: {result['group_a']} vs {', '.join(result['group_b_groups'])} pooled | "
            f"diff={result['observed_diff']:.6f} | p_two={result['p_two']:.8f} | "
            f"p_holm={result['p_holm']:.8f} | label={p_to_stars(result['p_holm'])}"
        )
    for channel_order, pdf_path, png_path, svg_path in saved_paths:
        print(f"Saved ({', '.join(channel_order)}): {pdf_path}")
        print(f"Saved ({', '.join(channel_order)}): {png_path}")
        print(f"Saved ({', '.join(channel_order)}): {svg_path}")


if __name__ == "__main__":
    main()
