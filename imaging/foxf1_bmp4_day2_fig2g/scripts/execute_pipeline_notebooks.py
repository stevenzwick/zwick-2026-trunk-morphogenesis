#!/usr/bin/env python3
"""Execute the ordered pipeline notebooks and save executed copies."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAW_MISSING = (
    "No CZI stacks in {data_dir}.\n"
    "This step reads the 14 raw stacks (1.czi to 11.czi, 13.czi, 15.czi and 16.czi), which are not in this\n"
    "repository. They are available from the lead contact on request: see Raw data in the lane's README.md.\n"
    "Fig 2g itself is redrawn without them by scripts/render_fig2g_panel.py."
)


NOTEBOOK_ORDER = [
    "00_manifest_qc.ipynb",
    "01_dapi_whole_organoid_mask_review.ipynb",
    "02_reporter_pixel_quantification.ipynb",
    "03_global_q995_density_review.ipynb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook_names",
        nargs="*",
        help="Optional subset of notebook filenames to execute in order",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Lane directory (default: the directory holding scripts/)")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("notebooks"),
        help="Directory containing source notebooks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/executed_notebooks"),
        help="Directory for executed notebook snapshots",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5400,
        help="Per-notebook execution timeout in seconds",
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        default="python3",
        help="Kernel name used for notebook execution",
    )
    return parser.parse_args()


def resolve_path(root: Path, path_like: Path) -> Path:
    return path_like.resolve() if path_like.is_absolute() else (root / path_like).resolve()


def executed_name(path: Path) -> str:
    return f"{path.stem}.executed.ipynb"


def execute_notebook(source_path: Path, output_path: Path, cwd: Path, timeout: int, kernel_name: str) -> None:
    nb = nbformat.read(source_path, as_version=4)
    client = NotebookClient(
        nb,
        kernel_name=str(kernel_name),
        timeout=int(timeout),
        resources={"metadata": {"path": str(cwd)}},
    )
    client.execute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output_path)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not any((root / "data" / "with DAPI").glob("*.czi")):
        raise SystemExit(RAW_MISSING.format(data_dir=root / "data" / "with DAPI"))
    source_dir = resolve_path(root, Path(args.source_dir))
    output_dir = resolve_path(root, Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    notebook_names = list(args.notebook_names) if args.notebook_names else list(NOTEBOOK_ORDER)

    for notebook_name in notebook_names:
        source_path = source_dir / notebook_name
        if not source_path.exists():
            raise FileNotFoundError(f"Notebook not found: {source_path}")
        output_path = output_dir / executed_name(source_path)
        print(f"[RUN] {source_path.name} -> {output_path.name}")
        execute_notebook(
            source_path=source_path,
            output_path=output_path,
            cwd=root,
            timeout=int(args.timeout),
            kernel_name=str(args.kernel_name),
        )
        print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
