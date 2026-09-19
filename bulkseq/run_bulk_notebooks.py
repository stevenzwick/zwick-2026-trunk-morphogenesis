#!/usr/bin/env python3
"""Execute the bulk RNA-seq notebooks without a Jupyter server.

Each notebook runs in its own process: its code cells execute in order in a fresh IPython shell, the
same way Jupyter's "Run All" would, with a non-interactive plotting backend. Figures and tables go to
bulkseq/output/.

    BULK_CPM_TABLE=/path/to/GSE347564_bulk_RNAseq_cpm_all_samples.tsv.gz python bulkseq/run_bulk_notebooks.py

Without BULK_CPM_TABLE the notebooks look for data/GSE347564/GSE347564_bulk_RNAseq_cpm_all_samples.tsv.gz.
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "ed3b_bmp4_timecourse.ipynb",
    "ed3c_endoderm_markers.ipynb",
    "ed3j_bmpr1a_knockdown.ipynb",
    "ed10g_nmp_lpm_markers.ipynb",
]
DEFAULT_TABLE = REPO / "data" / "GSE347564" / "GSE347564_bulk_RNAseq_cpm_all_samples.tsv.gz"


def run_one(path: Path) -> int:
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    cells = [c for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(cells):
        result = shell.run_cell("".join(cell["source"]), store_history=True)
        if not result.success:
            print(f"FAILED: {path.name}, code cell {i}", file=sys.stderr)
            return 1
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        return run_one(Path(sys.argv[2]))
    table = Path(os.environ.get("BULK_CPM_TABLE", DEFAULT_TABLE))
    if not table.is_file():
        print(f"CPM table not found: {table}\nDownload it from GEO series GSE347564; see bulkseq/README.md.",
              file=sys.stderr)
        return 2
    env = {**os.environ, "MPLBACKEND": "Agg", "BULK_CPM_TABLE": str(table)}
    ran = 0
    for name in NOTEBOOKS:
        print(f"--- {name}")
        r = subprocess.run([sys.executable, __file__, "--one", str(REPO / "bulkseq" / "notebooks" / name)],
                           cwd=REPO, env=env)
        if r.returncode:
            print(f"ran {ran}/{len(NOTEBOOKS)} bulk notebooks; stopped at {name}", file=sys.stderr)
            return 1
        ran += 1
    print(f"ran {ran}/{len(NOTEBOOKS)} bulk notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
