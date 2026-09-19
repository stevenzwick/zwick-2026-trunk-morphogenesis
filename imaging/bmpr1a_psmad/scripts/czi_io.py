#!/usr/bin/env python
"""Robust CZI channel + pixel-size reader (aicspylibczi / libCZI).

WHY: `czifile` decodes the odd-sized (1025/1026-px) acquisitions in this dataset
NONDETERMINISTICALLY - repeated reads return different pixels, sometimes saturated
garbage - which corrupted 11/37 fields (empty colony -> every cell pinned to the
d_edge floor -> fake-low edge medians). aicspylibczi reads all fields correctly
and reproducibly. All pipeline IO goes through here.

Channels (confirmed from metadata): 0 brightfield, 1 DAPI, 2 pSMAD1/5/9 (Alexa 568),
3 transduction marker (EGFP channel).
"""
import numpy as np
from aicspylibczi import CziFile

CH_DAPI, CH_PSMAD, CH_MARKER = 1, 2, 3


def _channel(czi, c):
    img, _ = czi.read_image(C=c)
    return np.squeeze(img).astype(float)


def read_channel(path, c):
    """Single 2D float channel."""
    return _channel(CziFile(path), c)


def read_dapi_psmad_px(path):
    """(DAPI, pSMAD, pixel_size_um) - matches the pipeline's load() signature."""
    czi = CziFile(path)
    return _channel(czi, CH_DAPI), _channel(czi, CH_PSMAD), _pixel_size_um(czi)


def _pixel_size_um(czi):
    try:
        for d in czi.meta.iter("Distance"):
            if d.get("Id") == "X":
                return float(d.find("Value").text) * 1e6   # metres -> microns
    except Exception:
        pass
    return None
