"""Execute the notebook chain into results/executed_notebooks.

NOT runnable as shipped. This stage reads the raw microscopy images, which are not deposited —
they are available from the lead contact on request, as the paper's Data Availability Statement
states. Its outputs are committed under the lane's `derived/` directory, which is what every
figure script actually reads. See `data/DOWNLOAD.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

try:
    from .pipeline_contract import AUXILIARY_NOTEBOOKS, MAINLINE_NOTEBOOKS, PROJECT_ROOT
except ImportError:
    from pipeline_contract import AUXILIARY_NOTEBOOKS, MAINLINE_NOTEBOOKS, PROJECT_ROOT  # type: ignore


def _matches(token: str, relpath: str) -> bool:
    path = Path(relpath)
    return token in {relpath, path.name, path.stem}


def _slice_notebooks(notebooks: tuple[str, ...], start_at: str | None, stop_after: str | None) -> list[str]:
    selected = list(notebooks)
    if start_at:
        try:
            start_idx = next(i for i, rel in enumerate(selected) if _matches(start_at, rel))
        except StopIteration as exc:
            raise SystemExit(f"Unknown start notebook: {start_at}") from exc
        selected = selected[start_idx:]
    if stop_after:
        try:
            stop_idx = next(i for i, rel in enumerate(selected) if _matches(stop_after, rel))
        except StopIteration as exc:
            raise SystemExit(f"Unknown stop notebook: {stop_after}") from exc
        selected = selected[: stop_idx + 1]
    return selected


def _run_notebook(relpath: str) -> None:
    notebook_path = PROJECT_ROOT / relpath
    output_dir = PROJECT_ROOT / "results" / "executed_notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)

    jupyter_cmd = shutil.which("jupyter")
    if jupyter_cmd is None:
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
        ]
    else:
        cmd = [
            jupyter_cmd,
            "nbconvert",
        ]

    cmd.extend(
        [
        "--to",
        "notebook",
        "--execute",
        str(notebook_path),
        "--output",
        f"{notebook_path.stem}.executed.ipynb",
        "--output-dir",
        str(output_dir),
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
        ]
    )
    # Run the command directly rather than through `/bin/zsh -lc`. The login shell was there to
    # pick up the author's PATH; it hard-codes an interpreter that need not exist (no zsh on many
    # Linux images) and it swallows the notebook's own error behind the shell's exit status.
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        # Surface the notebook's actual failure. `check=True` alone raises CalledProcessError
        # naming only the command line, so a reader learns that nbconvert failed but never why --
        # and the why is almost always a missing input this repository does not ship.
        detail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-25:])
        raise SystemExit(
            f"Notebook failed: {notebook_path.name}\n"
            f"{'-' * 70}\n{detail}\n{'-' * 70}\n"
            "These notebooks read the raw acquisitions, which are not in this repository.\n"
            "See data/DOWNLOAD.md.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=("mainline", "auxiliary", "all"),
        default="mainline",
        help="Which notebook group to run.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Optional notebook stem or path to start at.",
    )
    parser.add_argument(
        "--stop-after",
        default=None,
        help="Optional notebook stem or path to stop after.",
    )
    args = parser.parse_args()

    if args.group == "mainline":
        notebooks = MAINLINE_NOTEBOOKS
    elif args.group == "auxiliary":
        notebooks = AUXILIARY_NOTEBOOKS
    else:
        notebooks = MAINLINE_NOTEBOOKS + AUXILIARY_NOTEBOOKS

    selected = _slice_notebooks(notebooks, start_at=args.start_at, stop_after=args.stop_after)
    if not selected:
        raise SystemExit("No notebooks selected.")

    for idx, relpath in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] Executing {relpath}")
        _run_notebook(relpath)

    print("Notebook execution completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
