#!/usr/bin/env python
"""02 - d_edge quantification (Steven's algorithm) + per-cell pSMAD.

For each 10x field, using DAPI (ch1), pSMAD (ch2) and the Cellpose nuclei
labels from 01:

  (1) cell mask  = blurred + Otsu-thresholded DAPI   (default; robust to Cellpose
                   misses) OR (nuclei labels > 0)      (--footprint seg)
  (2) colony     = pixels within d_min of a cell = binary_dilation(cell mask, d_min)
      background = everything else
  (3) d_edge(cell) = EDT_to_nearest_background[cell centroid] - d_min
      -> outermost (edge) cells ~0; interior cells = depth from the colony edge.

d_min is specified in microns and converted with the CZI pixel size. The colony
mask is built from DAPI by default so a missed nucleus costs only one sampled
cell and cannot punch a false hole in the colony.

Writes:
  results/tables/per_cell_dedge.csv               one row per segmented nucleus
  results/dedge_qc/<cond>__<tp>__<field>.png      d_edge map + pSMAD-vs-d_edge (for visual review)

Env: psmad_quant.
Run: /opt/anaconda3/bin/conda run -n psmad_quant python scripts/02_dedge_quant.py \
        [files...] [--d_min_um 20] [--footprint dapi|seg] [--blur_sigma_um 4]
"""
import os, glob, argparse, csv
import numpy as np
import tifffile
from czi_io import read_dapi_psmad_px as load, CH_DAPI, CH_PSMAD
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops

WORK   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(WORK, "data")
MASKS  = os.path.join(WORK, "results", "masks")
QC     = os.path.join(WORK, "results", "dedge_qc")
TABLES = os.path.join(WORK, "results", "tables")


def meta(path):
    p = os.path.normpath(path).split(os.sep)
    return p[-3], p[-2], os.path.splitext(p[-1])[0]


def colony_footprint(dapi, labels, source, d_min_px, blur_px):
    if source == "seg":
        cell = labels > 0
    else:
        b = gaussian(dapi, sigma=blur_px, preserve_range=True)
        cell = b > threshold_otsu(b)
    # "within d_min of a cell" = EDT-based dilation (fast + exact at any radius)
    colony = ndi.distance_transform_edt(~cell) <= d_min_px
    return colony


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="explicit .czi; default = all 10x with masks")
    ap.add_argument("--d_min_um", type=float, default=20.0)
    ap.add_argument("--footprint", choices=["dapi", "seg"], default="dapi")
    ap.add_argument("--blur_sigma_um", type=float, default=4.0)
    ap.add_argument("--out", default=os.path.join(TABLES, "per_cell_dedge.csv"))
    args = ap.parse_args()

    os.makedirs(QC, exist_ok=True); os.makedirs(TABLES, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    targets = args.files if args.files else sorted(glob.glob(os.path.join(DATA, "*", "*", "10x-*.czi")))
    rows = []
    for f in targets:
        cond, tp, field = meta(f)
        mp = os.path.join(MASKS, cond, tp, f"{field}_nuclei.tif")
        if not os.path.exists(mp):
            print("  no mask, skip:", cond, tp, field); continue
        dapi, psmad, px = load(f)
        if px is None:
            px = 1.0; print("  WARN no pixel size, using 1.0 um/px:", cond, tp, field)
        d_min_px = args.d_min_um / px
        blur_px  = max(1.0, args.blur_sigma_um / px)
        labels = tifffile.imread(mp)

        colony = colony_footprint(dapi, labels, args.footprint, d_min_px, blur_px)
        dedge_map = ndi.distance_transform_edt(colony) - d_min_px        # pixels
        psmad_bg = float(np.median(psmad[~colony])) if (~colony).any() else 0.0
        border = bool(colony[0, :].any() or colony[-1, :].any() or colony[:, 0].any() or colony[:, -1].any())
        H, W = dapi.shape

        field_rows = []
        for p, pd in zip(regionprops(labels, intensity_image=psmad),
                         regionprops(labels, intensity_image=dapi)):
            y, x = p.centroid
            yi = min(max(int(round(y)), 0), H - 1); xi = min(max(int(round(x)), 0), W - 1)
            de_px = float(dedge_map[yi, xi])
            ps = float(p.intensity_mean) - psmad_bg
            dp = float(pd.intensity_mean)
            field_rows.append(dict(
                condition=cond, timepoint=tp, field=field, cell_id=int(p.label),
                y=round(y, 1), x=round(x, 1), area_px=int(p.area),
                dedge_px=round(de_px, 2), dedge_um=round(de_px * px, 2),
                psmad_mean=round(float(p.intensity_mean), 2), psmad_bgsub=round(ps, 2),
                dapi_mean=round(dp, 2), psmad_over_dapi=round(ps / max(dp, 1e-6), 4),
                px_um=round(px, 4), d_min_um=args.d_min_um, footprint=args.footprint,
                colony_border_touch=int(border)))
        rows.extend(field_rows)

        # ---- per-field QC (to check the edge definition) ----
        de_um = np.array([r["dedge_um"] for r in field_rows])
        ps_bg = np.array([r["psmad_bgsub"] for r in field_rows])
        ys = np.array([r["y"] for r in field_rows]); xs = np.array([r["x"] for r in field_rows])
        disp = np.where(colony, dedge_map * px, np.nan)
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        im = ax[0].imshow(disp, cmap="viridis"); ax[0].imshow(~colony, cmap="gray", alpha=0.12)
        ax[0].scatter(xs, ys, s=2, c="white", alpha=0.4)
        ax[0].set_title(f"{cond} {tp} {field}  d_edge (um), d_min={args.d_min_um}um")
        ax[0].axis("off"); fig.colorbar(im, ax=ax[0], fraction=0.046)
        ax[1].scatter(de_um, ps_bg, s=6, alpha=0.4)
        ax[1].set_xlabel("d_edge (um)"); ax[1].set_ylabel("pSMAD (bg-subtracted)")
        ax[1].set_title(f"pSMAD vs d_edge  (n={len(field_rows)} cells)"); ax[1].axhline(0, color="k", lw=0.5)
        fig.tight_layout(); fig.savefig(os.path.join(QC, f"{cond}__{tp}__{field}.png"), dpi=110); plt.close(fig)
        print(f"  {cond}/{tp}/{field}: {len(field_rows)} cells | px={px:.3f}um d_min={d_min_px:.1f}px border={border}")

    if rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} cells -> {args.out}")


if __name__ == "__main__":
    main()
