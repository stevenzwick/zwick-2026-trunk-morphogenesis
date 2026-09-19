#!/usr/bin/env python
"""01 - Nuclei segmentation from DAPI (ch1) with Cellpose 'nuclei' model.

For each 10x .czi (default: all; or explicit paths passed as args) segments
nuclei from the DAPI channel and writes:
  results/masks/<cond>/<tp>/<field>_nuclei.tif   uint16 instance labels
  results/masks/qc/<cond>__<tp>__<field>.png     DAPI+outlines | pSMAD  (for visual review)

Channels (confirmed from CZI metadata): ch0 brightfield, ch1 DAPI, ch2 pSMAD1/5/9
(Alexa 568), ch3 transduction marker (EGFP channel).

Env: psmad_quant (cellpose>=3,<4). Deterministic CPU run.
Run: /opt/anaconda3/bin/conda run -n psmad_quant python scripts/01_segment_nuclei.py [files...]

NOT runnable as shipped. Nuclear segmentation needs cellpose and torch, which are deliberately
left out of `env/environment.yml` — they are heavy and needed by this one script. Uncomment them
there to re-run segmentation; its outputs are committed under `derived/`.
"""
import os, glob, argparse
import numpy as np
import tifffile
from czi_io import read_channel as load_channel, CH_DAPI, CH_PSMAD

WORK  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(WORK, "data")
MASKS = os.path.join(WORK, "results", "masks")
QC    = os.path.join(MASKS, "qc")


def meta(path):
    """.../data/<cond>/<tp>/<field>.czi -> (cond, tp, field)"""
    p = os.path.normpath(path).split(os.sep)
    return p[-3], p[-2], os.path.splitext(p[-1])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="explicit .czi paths; default = all 10x")
    ap.add_argument("--diameter", type=float, default=None, help="nucleus diameter (px); None=auto")
    args = ap.parse_args()

    import torch
    torch.manual_seed(0); np.random.seed(0)
    from cellpose import models
    from skimage.segmentation import find_boundaries
    from skimage.exposure import rescale_intensity
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = models.Cellpose(gpu=False, model_type="nuclei")
    targets = args.files if args.files else sorted(glob.glob(os.path.join(DATA, "*", "*", "10x-*.czi")))
    os.makedirs(QC, exist_ok=True)
    print(f"segmenting {len(targets)} field(s); diameter={args.diameter}")

    diams = []
    for f in targets:
        cond, tp, field = meta(f)
        dapi  = load_channel(f, CH_DAPI)
        psmad = load_channel(f, CH_PSMAD)
        masks, _flows, _styles, diam = model.eval(dapi, diameter=args.diameter, channels=[0, 0])
        d = float(np.atleast_1d(diam)[0]); diams.append(d)
        n = int(masks.max())

        outdir = os.path.join(MASKS, cond, tp); os.makedirs(outdir, exist_ok=True)
        tifffile.imwrite(os.path.join(outdir, f"{field}_nuclei.tif"), masks.astype(np.uint16))

        b  = find_boundaries(masks, mode="outer")
        d8 = rescale_intensity(dapi.astype(float),  in_range=tuple(np.percentile(dapi,  (1, 99.5))), out_range=(0, 1))
        p8 = rescale_intensity(psmad.astype(float), in_range=tuple(np.percentile(psmad, (1, 99.5))), out_range=(0, 1))
        ov = np.dstack([d8, d8, d8]); ov[b] = [1, 0, 0]
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(ov);                ax[0].set_title(f"{cond}  {tp}  {field}   DAPI + {n} nuclei")
        ax[1].imshow(p8, cmap="magma");  ax[1].set_title("pSMAD (ch2)")
        for a_ in ax: a_.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(QC, f"{cond}__{tp}__{field}.png"), dpi=110)
        plt.close(fig)
        print(f"  {cond}/{tp}/{field}: {n} nuclei, diam~{d:.1f}px")

    if diams:
        print(f"\nmedian auto-diameter: {np.median(diams):.1f}px over {len(diams)} field(s)")


if __name__ == "__main__":
    main()
