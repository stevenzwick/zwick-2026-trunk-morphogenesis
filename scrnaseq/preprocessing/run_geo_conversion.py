#!/usr/bin/env python3
"""Build the first scRNA-seq notebook's inputs from public GEO files, without a Jupyter server.

Runs `00a_geo_trunk_morph.ipynb`, then `00b_geo_human_embryo.ipynb`. Each notebook runs in its own
process: its code cells execute in order in a fresh IPython shell, the same way Jupyter's "Run All"
would, with a non-interactive plotting backend. Each cell's run time is printed.

    python scrnaseq/preprocessing/run_geo_conversion.py

GEO files are read from data/GSE306808/ and data/GSE155121/, or from GEO_DATA_DIR if set. Outputs go to
SCRNASEQ_INPUT_ROOT (default data/scrnaseq_inputs/), which is where 01_preprocessing reads them. To run
the whole single-cell chain, including these two notebooks, use scrnaseq/run_chain.py. See
data/DOWNLOAD.md.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOKS = ["00a_geo_trunk_morph.ipynb", "00b_geo_human_embryo.ipynb"]
GEO_FILES = [
    Path("GSE306808") / "GSM9209686_filtered_feature_bc_matrix.h5",
    Path("GSE306808") / "GSM9209686_barcode_data.csv.gz",
    Path("GSE155121") / "GSE155121_human_data_raw.h5ad",
]


def run_one(path: Path) -> int:
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    cells = [c for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(cells):
        start = time.time()
        result = shell.run_cell("".join(cell["source"]), store_history=True)
        print(f"[{path.name}, code cell {i}: {time.time() - start:.1f} s]", flush=True)
        if not result.success:
            print(f"FAILED: {path.name}, code cell {i}", file=sys.stderr)
            return 1
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        return run_one(Path(sys.argv[2]))
    geo_dir = Path(os.environ.get("GEO_DATA_DIR", REPO / "data"))
    missing = [geo_dir / f for f in GEO_FILES if not (geo_dir / f).is_file()]
    for m in missing:
        hint = " (decompress the .gz with gunzip)" if m.with_name(m.name + ".gz").is_file() else ""
        print(f"GEO file not found: {m}{hint}", file=sys.stderr)
    if missing:
        print("See data/DOWNLOAD.md.", file=sys.stderr)
        return 2
    env = {**os.environ, "MPLBACKEND": "Agg", "GEO_DATA_DIR": str(geo_dir)}
    ran = 0
    for name in NOTEBOOKS:
        print(f"--- {name}", flush=True)
        r = subprocess.run([sys.executable, __file__, "--one", str(REPO / "scrnaseq" / "preprocessing" / "notebooks" / name)],
                           cwd=REPO, env=env)
        if r.returncode:
            print(f"ran {ran}/{len(NOTEBOOKS)} GEO conversion notebooks; stopped at {name}", file=sys.stderr)
            return 1
        ran += 1
    print(f"ran {ran}/{len(NOTEBOOKS)} GEO conversion notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
