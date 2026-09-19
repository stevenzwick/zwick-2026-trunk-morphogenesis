# LPM + NMP co-grafts

Quantification of co-transplanted LPM and NMP donor populations in host trunk morphs. This lane
feeds **Supplementary Data 4 §5**, not a main or Extended Data panel, which is why it does not
appear in the figure-to-code table in the top-level `README.md`.

## What is here, and what it can do

| | |
|---|---|
| `scripts/00*` – `02*` | Staging and QC: build a manifest over the raw `.czi` acquisitions, resolve channels **by dye name**, and check that the FOXF1 reporter and the LPM bead label are distinct signals |
| `scripts/03*` – `05*` | Morphological and per-z quantification: domain masks, axis calls, per-organoid and per-z metrics |
| `scripts/06*` – `08*` | The comparison figures behind the Supplementary Data section |
| `derived/05_per_organoid_metrics_all13.tsv` | Per-organoid metrics, 16 rows across three acquisition groups (`day5_exp1` 7, `d4images_exp1` 5, `day4_exp2` 4). The `all13` in the name is from an earlier count and was not renamed when the set grew |
| `derived/05_per_z_metrics_fix.tsv` | Per-z metrics for the two `day4_exp2` organoids that needed a per-plane look, 10 rows |

## What will not run here

**None of these scripts runs as shipped, and the two tables above are not read by anything in this
repository.** The stages read the raw `.czi` acquisitions, which are not deposited and are available
from the lead contact on request, as the paper's Data Availability Statement describes. The figure
scripts then read the full results directory of the original analysis run, including hand-curated
filter files (which organoids carry two grafts, which donor population, which to exclude) that are
part of that working directory rather than of this repository.

The two `derived/` tables are shipped as a record of the measured values behind that Supplementary
Data section, not as inputs a script here consumes. The section itself ships with the paper as a
PDF, so a reader does not need to rebuild it.

⚠️ **The two tables are from different runs, and do not reconcile with each other.** The per-z
table matches `scripts/05_cograft_quant_perz.py` as shipped. The per-organoid table does not: it
predates the per-image half-Gaussian thresholds that script now uses, which is visible in its
columns — it has 32 where the script writes 37, and the five it lacks are exactly the threshold
and threshold-method columns. For the two organoids that appear in both (`img02`, `img03`),
averaging the per-z rows does not reproduce the per-organoid row. Read each table as the record of
its own run.

Channel resolution is the one part worth reading even without the images: `00_build_raw_manifest.py`
maps dyes to markers **by dye name**, because channel position is not uniform across this set — the
two bead channels are swapped between the day 4 and day 5 acquisitions, and the `d4_images/` subset
has no DAPI. Its docstring carries the full dye-to-marker legend.
