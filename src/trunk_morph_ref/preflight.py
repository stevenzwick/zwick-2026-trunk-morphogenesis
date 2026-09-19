"""Storage-agnostic filesystem preflight checks for staged analysis runs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = Path(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def check_input_files(input_files: Iterable[Path]) -> list[str]:
    """Return human-readable issues for missing/unreadable/empty input files."""
    issues: list[str] = []
    for path in _unique_paths(input_files):
        if not path.exists():
            issues.append(f"missing file: {path}")
            continue
        if not path.is_file():
            issues.append(f"not a regular file: {path}")
            continue
        if path.stat().st_size <= 0:
            issues.append(f"empty file: {path}")
            continue
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except Exception as exc:  # pragma: no cover - defensive path diagnostics
            issues.append(f"unreadable file: {path} ({exc})")
    return issues


def check_output_dirs(output_dirs: Iterable[Path]) -> list[str]:
    """Return human-readable issues for non-creatable/non-writable directories."""
    issues: list[str] = []
    for path in _unique_paths(output_dirs):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive path diagnostics
            issues.append(f"cannot create directory: {path} ({exc})")
            continue

        if not path.is_dir():
            issues.append(f"not a directory: {path}")
            continue

        probe: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".preflight_write_probe.",
                suffix=".tmp",
                dir=str(path),
                delete=False,
            ) as handle:
                probe = Path(handle.name)
                handle.write(b"ok")
            probe.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - defensive path diagnostics
            issues.append(f"directory not writable: {path} ({exc})")
            if probe is not None:
                probe.unlink(missing_ok=True)
    return issues


def format_preflight_failure(stage_name: str, file_issues: list[str], dir_issues: list[str]) -> str:
    """Build a concise actionable failure report for notebook preflight."""
    lines = [f"Preflight failed for stage '{stage_name}'."]

    if file_issues:
        lines.append("Input file issues:")
        for issue in file_issues:
            lines.append(f"  - {issue}")

    if dir_issues:
        lines.append("Output directory issues:")
        for issue in dir_issues:
            lines.append(f"  - {issue}")

    lines.append("Suggested fixes:")
    lines.append("  1) Ensure required input files exist, are readable, and non-empty.")
    lines.append("  2) Ensure you have write permissions for output directories.")
    lines.append("  3) If starting from an intermediate stage, rerun upstream stages first.")
    return "\n".join(lines)

