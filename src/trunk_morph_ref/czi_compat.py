"""A czifile-compatible CZI reader backed by aicspylibczi.

WHY THIS EXISTS — two independent reasons, either one sufficient.

**Correctness.** `czifile` mis-frames odd-sized acquisitions. On `w2-6.czi` (a 1024x1024
acquisition) czifile reports shape (6, 4, 1025, 1024, 1): an oversized frame in which the true
image sits at a channel-dependent offset next to a junk strip, so a DAPI mask and its marker
channels can be misaligned by up to a pixel. aicspylibczi (libCZI) returns the correct
(6, 4, 1024, 1024). On even-sized files the two agree exactly. See `czi_io.py` in the co-graft
lane, which reached the same conclusion first, and memory `czifile-odd-czi-bug`.

**Installability.** `czifile` can no longer be installed beside this project's pinned numpy at
all. The 2019 release reads `imagecodecs.jxr_decode`, renamed to `jpegxr_decode` in modern
imagecodecs, so it raises AttributeError on import; and the current release resolves numpy to
2.5.x, which breaks the pinned scipy and numba. There is no version of czifile that works here.

USAGE — a drop-in for the one pattern the shipped code uses:

    from trunk_morph_ref import czi_compat as czifile     # was: import czifile
    with czifile.CziFile(path) as czi:
        arr = czi.asarray()

`axes`, `shape`, `dtype`, `asarray()` and `metadata()` follow czifile's conventions, including
its trailing singleton sample axis, so indexing downstream is unchanged. The ONE deliberate
difference is the fix above: on an odd-sized file the array is the true frame, not the oversized
one. `filtered_subblock_directory` is NOT provided — it exposes czifile's own subblock objects,
which have no aicspylibczi equivalent; the three mosaic scripts that use it say so at their import.
"""
from __future__ import annotations

from xml.etree import ElementTree

import numpy as np
from aicspylibczi import CziFile as _LibCzi

__all__ = ["CziFile"]


def _element_to_dict(element):
    """One XML element as czifile's xml2dict renders it: nested dicts, repeats as lists.

    Attributes and child elements share the key space, children winning, which is the shape the
    metadata extractors were written against. A leaf with neither children nor attributes
    collapses to its text so `...["Channel"]["Name"]` is a string rather than a wrapper.
    """
    result: dict = {}
    for child in element:
        value = _element_to_dict(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(value)
        else:
            result[child.tag] = value
    for key, val in element.attrib.items():
        result.setdefault(key, val)
    if not result:
        return (element.text or "").strip()
    text = (element.text or "").strip()
    if text:
        result.setdefault("Value", text)
    return result


class CziFile:
    """Reads a CZI through libCZI, presenting czifile's interface."""

    def __init__(self, path):
        self._path = str(path)
        self._array: np.ndarray | None = None
        self._dims: list[tuple[str, int]] | None = None
        try:
            self._czi = _LibCzi(self._path)
        except RuntimeError as exc:
            # Some one-graft acquisitions are ImageJ TIF hyperstacks saved with a .czi
            # extension; libCZI rejects them with "Invalid FileHdr-magic". czi_io.py in the
            # co-graft lane found this first and falls back the same way.
            if "FileHdr-magic" not in str(exc):
                raise
            self._czi = None

    def __enter__(self) -> "CziFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._array, self._czi = None, None

    def _read_tif(self) -> np.ndarray:
        import tifffile
        with tifffile.TiffFile(self._path) as tf:
            arr = np.asarray(tf.asarray())
            axes = (getattr(tf.series[0], "axes", "") or "").upper()
        self._dims = [(a, n) for a, n in zip(axes, arr.shape)]
        return arr

    def _read(self) -> np.ndarray:
        if self._array is None:
            if self._czi is None:
                self._array = self._read_tif()[..., np.newaxis]
                return self._array
            img, shape = self._czi.read_image()
            self._dims = [(str(name), int(size)) for name, size in shape]
            # czifile appends a trailing sample axis ("0") of length 1 for single-sample pixels.
            # Keeping it means downstream indexing written against czifile is unchanged.
            self._array = np.asarray(img)[..., np.newaxis]
        return self._array

    def asarray(self, *_args, **_kwargs) -> np.ndarray:
        return self._read()

    @property
    def axes(self) -> str:
        self._read()
        return "".join(name for name, _ in self._dims) + "0"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._read().shape)

    @property
    def dtype(self):
        return self._read().dtype

    def metadata(self, raw: bool = True):
        """The CZI's XML metadata block, as czifile returns it.

        `raw=True` gives the XML string. `raw=False` gives a **nested dict**, which is what
        czifile returns and what every caller here expects: the extractors walk it with
        `metadata.get("ImageDocument", {}).get("Metadata", {})...`.

        Returning the ElementTree Element instead is not a near-miss, it is a silent-wrong-answer
        bug. `Element.get(key, default)` looks up an XML *attribute*, so that chain does not raise
        — it returns the default and keeps going, and every metadata extractor yields empty
        results while appearing to work. Channel names, pixel scale, objective and acquisition
        timestamp all came back blank across three lanes before this was fixed.
        """
        if self._czi is None:
            return "" if raw else {}
        meta = self._czi.meta
        if raw:
            return ElementTree.tostring(meta, encoding="unicode") if meta is not None else ""
        if meta is None:
            return {}
        return {meta.tag: _element_to_dict(meta)}
