#!/usr/bin/env python3
"""Robust CZI reader (aicspylibczi / libCZI) for the co-graft quant.

WHY NOT czifile: the R1 helper's `load_czi_stack` reads via `czifile`, which
MIS-FRAMES the odd-sized acquisitions in this set (img11 Y=1029, img13 X=1026,
img14 Y=1025, img15 1025x1025). czifile returns an OVERSIZED array (e.g. 1029
rows) in which the true 1024-px image sits at a CHANNEL-DEPENDENT offset (+ a junk
strip) — so the DAPI mask and the marker channels can be misframed/misaligned by up
to 1 px. Verified here: on EVEN files czifile == aicspylibczi exactly; on the ODD
files a channel-wise crop of czifile's array is pixel-identical to aicspylibczi
(max|Δ|=0), i.e. the pixels are correct but mis-placed in an oversized frame.

aicspylibczi (libCZI / C++) returns the correct, reproducible 1024^2 planes with all
channels aligned. Cross-machine finding — see memory `czifile-odd-czi-bug`. Pixel
sizes come from the manifest, so this reader only needs to return the pixel stack.
"""
from __future__ import annotations

import numpy as np
from aicspylibczi import CziFile


class CziStack:
    """Minimal stand-in for the R1 helper's CziStack — exposes only `.data_czyx`."""

    def __init__(self, data_czyx: np.ndarray):
        self.data_czyx = data_czyx


def _load_via_libczi(path) -> CziStack:
    czi = CziFile(str(path))
    c0, c1 = czi.get_dims_shape()[0]["C"]
    # Read ONE channel at a time straight into a preallocated float32 (C, Z, Y, X) and free each
    # source immediately. Avoids holding all uint16 channels + a stacked copy + the float32 copy
    # simultaneously (that ~4x transient on the big 0.65 um stacks blew past 2 GB). Per-channel
    # read_image is unchanged, so the odd-CZI framing fix (see module docstring) is preserved.
    arr = None
    for idx, c in enumerate(range(c0, c1)):
        img, _ = czi.read_image(C=c)
        pl = np.squeeze(np.asarray(img))                # -> (Z, Y, X)  (or (Y, X) if single-z)
        if pl.ndim == 2:
            pl = pl[None, :, :]                         # single-z -> insert the Z axis
        if arr is None:
            arr = np.empty((c1 - c0,) + pl.shape, dtype=np.float32)
        arr[idx] = pl
        del img, pl
    return CziStack(arr)


def _load_via_tifffile(path) -> CziStack:
    """Some one-graft files (LPM_transplant) are ImageJ TIF hyperstacks saved with a
    `.czi` extension — libCZI rejects them ('Invalid FileHdr-magic'). Read via tifffile
    and normalise to (C, Z, Y, X). ImageJ hyperstacks in this set are (Z, C, Y, X)."""
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        arr = np.asarray(tf.asarray())
        axes = (getattr(tf.series[0], "axes", "") or "").upper()
    if arr.ndim == 4:
        if all(a in axes for a in "CZYX"):
            arr = np.transpose(arr, [axes.index(a) for a in "CZYX"])
        else:
            arr = np.swapaxes(arr, 0, 1)                # (Z, C, Y, X) -> (C, Z, Y, X)
    elif arr.ndim == 3:
        arr = arr[:, None, :, :]                        # (C, Y, X) -> (C, Z, Y, X)
    else:
        raise ValueError(f"unexpected TIF ndim {arr.ndim} (axes={axes}) for {path}")
    return CziStack(np.ascontiguousarray(arr.astype(np.float32)))


def load_czi_stack(path) -> CziStack:
    """`.data_czyx` = (C, Z, Y, X) float32. libCZI for true CZIs; tifffile fallback for
    ImageJ-hyperstack-as-.czi files (some LPM_transplant one-graft images)."""
    try:
        return _load_via_libczi(path)
    except Exception:
        return _load_via_tifffile(path)
