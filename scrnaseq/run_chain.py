#!/usr/bin/env python3
"""Run the single-cell notebook chain without a Jupyter server.

    python scrnaseq/run_chain.py                           # the whole chain, 00a to 08b
    python scrnaseq/run_chain.py --from 01_preprocessing   # start at a later notebook
    python scrnaseq/run_chain.py --from 07_human_embryo --to 07c_human_embryo_nmp_subclustering
    python scrnaseq/run_chain.py --list                    # print the order and exit

Each notebook runs in its own process: its code cells execute in order in a fresh IPython shell, the
same way Jupyter's "Run All" would, with a non-interactive plotting backend. As in Jupyter, open
figures are closed after each cell. The run stops at the first cell that raises, names the notebook
and cell, and says how to resume. Each notebook's wall time and peak memory (the maximum resident set
size of its process) are printed and appended to run_chain_log.tsv in the results directory.

The list includes four notebooks that the runner skips by default, with a one-line message:
07b_human_embryo_floor_plate_subclustering, 07c_human_embryo_nmp_subclustering and the morph-embryo
comparison notebooks 01 and 12. They ran on the earlier human embryo clustering object (46,768 cells,
50 leiden_embryo clusters), whose SMD z-scores are the ones shipped for 07b and 07c, not on the object
07_human_embryo builds here (46,055 cells, 48 clusters). That earlier object is an optional Zenodo
download (see data/DOWNLOAD.md); when it is present under SCRNASEQ_INPUT_ROOT, these four notebooks run
instead of being skipped, drawing Fig 4d and 4e from the run this chain reruns rather than their saved
tables in scrnaseq/morph_embryo_correspondence/derived/. Absent, as for a reader who has not fetched it,
they are skipped and the chain still completes.

Where files are read and written (see data/DOWNLOAD.md):

    GEO_DATA_DIR           the GEO downloads 00a and 00b read (default data/)
    SCRNASEQ_INPUT_ROOT    what 00a and 00b write, the earlier trunk morph object from Zenodo that
                           06_rp_nc_subclustering reads, and the earlier human embryo object from
                           Zenodo that 07b, 07c, 01 and 12 read (default data/scrnaseq_inputs/)
    SCRNASEQ_RESULTS_ROOT  everything the notebooks write (default scrnaseq/output/)

Files that no notebook writes and that ship with the repository, such as the SMD z-scores, are read
from scrnaseq/chain_inputs/.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.trunk_morph_ref.paths import scrnaseq_input_root, scrnaseq_results_root  # noqa: E402

NOTEBOOKS = [
    "preprocessing/notebooks/00a_geo_trunk_morph.ipynb",
    "preprocessing/notebooks/00b_geo_human_embryo.ipynb",
    "preprocessing/notebooks/01_preprocessing.ipynb",
    "trunk_main/notebooks/02_trunk_main.ipynb",
    "subclustering/notebooks/03_lpm_subclustering.ipynb",
    "subclustering/notebooks/04_somite_subclustering.ipynb",
    "subclustering/notebooks/05_fbmb_subclustering.ipynb",
    "subclustering/notebooks/06_rp_nc_subclustering.ipynb",
    "subclustering/notebooks/06b_notochord_floor_plate_subclustering.ipynb",
    "human_embryo/notebooks/07_human_embryo.ipynb",
    "human_embryo/notebooks/07b_human_embryo_floor_plate_subclustering.ipynb",
    "human_embryo/notebooks/07c_human_embryo_nmp_subclustering.ipynb",
    "final_assignments/notebooks/08a_final_morph_cluster_assignments.ipynb",
    "final_assignments/notebooks/08b_integration_and_final_exports.ipynb",
    "morph_embryo_correspondence/notebooks/01_cluster_cluster_correlations.ipynb",
    "morph_embryo_correspondence/notebooks/12_merged_embryo_cluster_cluster_correlation.ipynb",
]

#: Files a notebook needs that neither an earlier notebook nor the repository provides.
GEO_FILES = [
    Path("GSE306808") / "GSM9209686_filtered_feature_bc_matrix.h5",
    Path("GSE306808") / "GSM9209686_barcode_data.csv.gz",
    Path("GSE155121") / "GSE155121_human_data_raw.h5ad",
]
CONVERSION_OUTPUTS = [
    Path("sample_284.adata"),
    Path("human_embryo") / "GSE155121" / "GSE155121_human_data_w3-4_genematch.h5ad",
    Path("embryo_only_genes.pkl"),
    Path("morph_only_genes.pkl"),
]
ZENODO_OBJECT = Path("legacy/results/intermediates/02_trunk_main/adata_morph_with_clusters.h5ad")
#: The earlier human embryo run's clusters object, an optional Zenodo download (see data/DOWNLOAD.md
#: and src.trunk_morph_ref.paths.scrnaseq_input_root). Its presence, not just notebook identity,
#: decides whether EARLIER_EMBRYO_OBJECT notebooks run or are skipped.
EARLIER_EMBRYO_RUN_OBJECT = Path(
    "earlier_embryo_run/results/intermediates/07_human_embryo/adata_embryo_with_clusters.h5ad"
)

#: Notebooks that ran on the earlier human embryo clustering object (see above). Skipped when
#: EARLIER_EMBRYO_RUN_OBJECT is absent from SCRNASEQ_INPUT_ROOT (the default, e.g. the release
#: test); run when it is present, reading it via each notebook's own fallback (see its first cell).
EARLIER_EMBRYO_OBJECT = {
    "07b_human_embryo_floor_plate_subclustering",
    "07c_human_embryo_nmp_subclustering",
    "01_cluster_cluster_correlations",
    "12_merged_embryo_cluster_cluster_correlation",
}


def stem(entry: str) -> str:
    return Path(entry).stem


def run_one(path: Path) -> int:
    """Execute one notebook's code cells in this process (called in a child process)."""
    from IPython.core.interactiveshell import InteractiveShell

    def close_figures(result=None):
        pyplot = sys.modules.get("matplotlib.pyplot")
        if pyplot is not None:
            pyplot.close("all")

    shell = InteractiveShell.instance()
    shell.events.register("post_run_cell", close_figures)
    cells = json.loads(path.read_text())["cells"]
    code = [(index, c) for index, c in enumerate(cells) if c["cell_type"] == "code"]
    for i, (index, cell) in enumerate(code):
        start = time.time()
        result = shell.run_cell("".join(cell["source"]), store_history=True)
        print(f"[{path.name}, code cell {i}: {time.time() - start:.1f} s]", flush=True)
        if not result.success:
            print(f"FAILED: {path.name}, code cell {i} (cell {index} of the notebook)", file=sys.stderr, flush=True)
            return 1
    return 0


def select(start: str | None, stop: str | None) -> list[str]:
    names = [stem(n) for n in NOTEBOOKS]

    def position(name: str) -> int:
        matches = [i for i, n in enumerate(names) if n == name or n.startswith(name)]
        exact = [i for i in matches if names[i] == name]
        if exact:
            return exact[0]
        if len(matches) != 1:
            raise SystemExit(f"'{name}' does not name exactly one notebook; use --list")
        return matches[0]

    first = position(start) if start else 0
    last = position(stop) if stop else len(NOTEBOOKS) - 1
    if last < first:
        raise SystemExit("--to comes before --from")
    return NOTEBOOKS[first:last + 1]


def missing_inputs(selected: list[str]) -> list[str]:
    """Check up front for inputs a reader supplies, so a long run does not stop partway for them."""
    names = [stem(n) for n in selected]
    problems = []
    geo_dir = Path(os.environ.get("GEO_DATA_DIR", REPO / "data"))
    input_root = scrnaseq_input_root(REPO)
    if "00a_geo_trunk_morph" in names:
        problems += [f"GEO file not found: {geo_dir / f}" for f in GEO_FILES if not (geo_dir / f).is_file()]
    elif "01_preprocessing" in names:
        problems += [f"GEO conversion output not found: {input_root / f} (run 00a and 00b first)"
                     for f in CONVERSION_OUTPUTS if not (input_root / f).is_file()]
    if "06_rp_nc_subclustering" in names and not (input_root / ZENODO_OBJECT).is_file():
        problems.append(f"Zenodo object not found: {input_root / ZENODO_OBJECT}")
    return problems


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        return run_one(Path(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", help="first notebook to run (name without .ipynb)")
    parser.add_argument("--to", dest="stop", help="last notebook to run (name without .ipynb)")
    parser.add_argument("--list", action="store_true", help="print the notebooks in order and exit")
    args = parser.parse_args()
    has_earlier_embryo_object = (scrnaseq_input_root(REPO) / EARLIER_EMBRYO_RUN_OBJECT).is_file()
    if args.list:
        skip_tag = "  (skipped; earlier human embryo object absent)"
        run_tag = "  (runs; earlier human embryo object present)"
        print("\n".join(
            f"scrnaseq/{n}" + (
                (skip_tag if not has_earlier_embryo_object else run_tag)
                if stem(n) in EARLIER_EMBRYO_OBJECT else ""
            )
            for n in NOTEBOOKS
        ))
        return 0

    selected = select(args.start, args.stop)
    problems = missing_inputs(selected)
    if problems:
        print("\n".join(problems) + "\nSee data/DOWNLOAD.md.", file=sys.stderr)
        return 2

    results_root = scrnaseq_results_root(REPO)
    results_root.mkdir(parents=True, exist_ok=True)
    log = results_root / "run_chain_log.tsv"
    if not log.exists():
        log.write_text("notebook\tstatus\twall_s\tpeak_rss_mb\tfinished\n")
    print(f"inputs: {scrnaseq_input_root(REPO)}\nresults: {results_root}", flush=True)

    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    ran = skipped = 0
    for entry in selected:
        name = stem(entry)
        if name in EARLIER_EMBRYO_OBJECT and not has_earlier_embryo_object:
            print(f"--- {name}: skipped; it ran on the earlier human embryo clustering object, "
                  f"not found at {scrnaseq_input_root(REPO) / EARLIER_EMBRYO_RUN_OBJECT} (see its first cell)")
            skipped += 1
            continue
        print(f"--- {name}", flush=True)
        start = time.time()
        child = subprocess.Popen([sys.executable, __file__, "--one", str(REPO / "scrnaseq" / entry)],
                                 cwd=REPO, env=env)
        _, status, usage = os.wait4(child.pid, 0)
        child.returncode = os.waitstatus_to_exitcode(status)
        wall = time.time() - start
        # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
        peak_mb = usage.ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
        status_text = "ok" if child.returncode == 0 else "FAILED"
        print(f"=== {name}: {status_text}, {wall:.0f} s, peak memory {peak_mb:,.0f} MB", flush=True)
        with log.open("a") as handle:
            handle.write(f"{name}\t{status_text}\t{wall:.1f}\t{peak_mb:.0f}\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if child.returncode:
            print(f"ran {ran}/{len(selected) - skipped} notebooks; stopped at {name}. "
                  f"Resume with: python scrnaseq/run_chain.py --from {name}", file=sys.stderr)
            return 1
        ran += 1
    print(f"ran {ran}/{len(selected) - skipped} notebooks" + (f", skipped {skipped}" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
