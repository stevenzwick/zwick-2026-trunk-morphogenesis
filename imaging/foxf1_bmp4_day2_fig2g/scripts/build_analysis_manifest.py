#!/usr/bin/env python3
"""Build the raw-input manifest for the FOXF1/BMP4 day-2 workspace.

NOT runnable from a clone alone. It reads the raw microscopy images, which are not deposited;
they are available from the lead contact on request, as the paper's Data Availability Statement
states. Place them in this lane's `data/with DAPI/` directory. See the lane's README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .day2_quantification_helpers import build_raw_input_manifest
except ImportError:
    from day2_quantification_helpers import build_raw_input_manifest


def run_manifest_pipeline(
    root: Path | str = ".",
    data_dir: Path | str = "data/with DAPI",
    output: Path | str = "results/manifests/raw_input_manifest.tsv",
    write_output: bool = True,
) -> dict[str, object]:
    root = Path(root).resolve()
    data_dir = (root / data_dir).resolve() if not Path(data_dir).is_absolute() else Path(data_dir).resolve()
    output = (root / output).resolve() if not Path(output).is_absolute() else Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest_df = build_raw_input_manifest(data_dir=data_dir, root=root)
    if bool(write_output):
        manifest_df.to_csv(output, sep="\t", index=False)

    return {
        "root": root,
        "data_dir": data_dir,
        "output_path": output,
        "manifest_df": manifest_df,
        "summary_text": f"{len(manifest_df)} canonical CZI inputs",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Workspace root")
    parser.add_argument("--data-dir", type=Path, default=Path("data/with DAPI"), help="Canonical raw input directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/manifests/raw_input_manifest.tsv"),
        help="Manifest output TSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    res = run_manifest_pipeline(
        root=args.root,
        data_dir=args.data_dir,
        output=args.output,
        write_output=True,
    )
    print(f"[OK] wrote {len(res['manifest_df'])} manifest rows to {res['output_path']}")


if __name__ == "__main__":
    main()

