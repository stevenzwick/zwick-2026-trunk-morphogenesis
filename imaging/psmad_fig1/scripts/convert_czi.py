#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def _tool_or_raise(
    explicit_tool: str | None,
    fallback_name: str,
    install_hint: str,
    *,
    allow_missing_for_dry_run: bool,
) -> str:
    if explicit_tool:
        return explicit_tool
    found = shutil.which(fallback_name)
    if found:
        return found
    if allow_missing_for_dry_run:
        return fallback_name
    raise RuntimeError(
        f"Required tool '{fallback_name}' not found on PATH.\n"
        f"{install_hint}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert CZI files using Bio-Formats command-line tools")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing .czi files")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument(
        "--format",
        choices=["ome-tiff", "ome-zarr"],
        default="ome-tiff",
        help="Output format. ome-zarr uses bioformats2raw; ome-tiff uses bfconvert.",
    )
    p.add_argument(
        "--tool",
        default=None,
        help="Optional explicit tool path (bfconvert or bioformats2raw depending on --format)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p.add_argument(
        "--resolutions",
        type=int,
        default=4,
        help="For ome-zarr only: number of pyramid resolutions (bioformats2raw --resolutions)",
    )
    return p.parse_args()


def _build_cmd(
    tool: str,
    src: Path,
    dst: Path,
    fmt: str,
    overwrite: bool,
    resolutions: int,
) -> list[str]:
    if fmt == "ome-tiff":
        cmd = [tool]
        if overwrite:
            cmd.append("-overwrite")
        cmd.extend([str(src), str(dst)])
        return cmd

    # ome-zarr via bioformats2raw
    cmd = [tool, str(src), str(dst), "--resolutions", str(resolutions)]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    czi_files = sorted(args.input_dir.glob("*.czi"))
    if not czi_files:
        raise RuntimeError(f"No .czi files found in {args.input_dir}")

    if args.format == "ome-tiff":
        tool = _tool_or_raise(
            args.tool,
            "bfconvert",
            "Install Bio-Formats command line tools and ensure bfconvert is on PATH.",
            allow_missing_for_dry_run=args.dry_run,
        )
        out_suffix = ".ome.tif"
    else:
        tool = _tool_or_raise(
            args.tool,
            "bioformats2raw",
            "Install bioformats2raw and ensure it is on PATH.",
            allow_missing_for_dry_run=args.dry_run,
        )
        out_suffix = ".ome.zarr"

    for src in czi_files:
        dst = args.out_dir / f"{src.stem}{out_suffix}"
        if dst.exists() and not args.overwrite:
            print(f"[SKIP] {dst} exists (use --overwrite)")
            continue

        cmd = _build_cmd(tool, src=src, dst=dst, fmt=args.format, overwrite=args.overwrite, resolutions=args.resolutions)
        print("[CMD]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            print(f"[OK] {src.name} -> {dst.name}")

    print("Done.")


if __name__ == "__main__":
    main()
