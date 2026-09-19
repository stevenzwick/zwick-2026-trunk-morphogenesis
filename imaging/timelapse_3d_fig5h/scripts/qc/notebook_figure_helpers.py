from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_VECTOR_EXPORT_SUFFIXES = (".pdf", ".svg")


def install_png_vector_savefig_exports(
    vector_suffixes: tuple[str, ...] = DEFAULT_VECTOR_EXPORT_SUFFIXES,
) -> None:
    import matplotlib.figure

    figure_cls = matplotlib.figure.Figure
    original = getattr(figure_cls, "_notebook_original_savefig", None)
    if original is None:
        original = figure_cls.savefig
        figure_cls._notebook_original_savefig = original

    figure_cls._notebook_vector_export_suffixes = tuple(vector_suffixes)

    def savefig_with_vector_exports(self, fname, *args, **kwargs):
        result = figure_cls._notebook_original_savefig(self, fname, *args, **kwargs)
        try:
            output_path = Path(fname)
        except TypeError:
            return result
        if output_path.suffix.lower() != ".png":
            return result
        for extra_suffix in figure_cls._notebook_vector_export_suffixes:
            figure_cls._notebook_original_savefig(
                self,
                output_path.with_suffix(extra_suffix),
                *args,
                **kwargs,
            )
        return result

    figure_cls.savefig = savefig_with_vector_exports


def normalize_export_filename(
    filename: str | Path,
    strip_prefixes: tuple[str, ...] = (),
) -> str:
    output_name = Path(filename).name
    changed = True
    while changed:
        changed = False
        for prefix in strip_prefixes:
            if output_name.startswith(prefix):
                output_name = output_name[len(prefix) :]
                changed = True
    return output_name


@dataclass(frozen=True)
class NotebookFigureExportPaths:
    figure_dir: Path
    alternate_dir: Path
    candidate_dir: Path
    alternate_strip_prefixes: tuple[str, ...] = ()
    candidate_strip_prefixes: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        root: Path,
        notebook_key: str,
        alternate_strip_prefixes: tuple[str, ...] = (),
        candidate_strip_prefixes: tuple[str, ...] = (),
    ) -> "NotebookFigureExportPaths":
        figure_dir = root / "results" / "figures" / notebook_key
        alternate_dir = figure_dir / "alternate_figures"
        candidate_dir = figure_dir / "candidate_figures"
        alternate_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            figure_dir=figure_dir,
            alternate_dir=alternate_dir,
            candidate_dir=candidate_dir,
            alternate_strip_prefixes=tuple(alternate_strip_prefixes),
            candidate_strip_prefixes=tuple(candidate_strip_prefixes),
        )

    def alternate_path(self, filename: str | Path) -> Path:
        return self.alternate_dir / normalize_export_filename(
            filename,
            strip_prefixes=self.alternate_strip_prefixes,
        )

    def candidate_path(self, filename: str | Path) -> Path:
        return self.candidate_dir / normalize_export_filename(
            filename,
            strip_prefixes=self.candidate_strip_prefixes,
        )

    def figure_path(self, filename: str | Path, candidate: bool = False) -> Path:
        return self.candidate_path(filename) if candidate else self.alternate_path(filename)
