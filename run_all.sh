#!/usr/bin/env bash
# Regenerate what this repository can regenerate on its own, and say plainly what it cannot.
#
# Runs end to end with no arguments and no data download: the quantified panels are drawn from the
# derivative tables committed here. Two optional variables widen it:
#
#   OBJECT_ROOT=/path/to/analysis      the saved AnnData objects, for the single-cell panels
#   SOURCE_DATA_DIR=/path/to/workbooks the published Source Data, to verify the numbers
#
# What this script does NOT do is re-run the pipeline stages. Those are notebooks: the single-cell
# chain has its own runner and needs the GEO data, and the imaging stages need the raw images — see
# "the pipeline stages" at the end.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> figures"
python figures/render_imaging_panels.py
python figures/render_scrnaseq_panels.py

# Five panels are drawn by their own lane's script rather than by the two renderers above, each
# from that lane's derived/ tables and needing no images and nothing configured. They are listed
# under "Reproducing a figure" in README.md; run them individually if you want those panels.
#   imaging/bmpr1a_psmad/scripts/08_ed3k_panel.py                  ED Fig 3k
#   imaging/foxf1_bmp4_day2_fig2g/scripts/render_fig2g_panel.py    Fig 2g
#   imaging/cfp_foxf1_density_fig5e/scripts/render_fig5e_panel.py  Fig 5e
#   imaging/foxa2_sox2_foxf1_day2_ed3g/scripts/render_ed3g_panel.py  ED Fig 3g
#   imaging/foxf1_bmp4_cyst_ed3o/scripts/render_ed3o_panel.py      ED Fig 3o

echo
echo "==> verification"
if [[ -n "${SOURCE_DATA_DIR:-}" ]]; then
  python figures/verify_imaging_panels.py
  python figures/verify_scrnaseq_panels.py
else
  echo "    skipped — set SOURCE_DATA_DIR to check every panel against the published Source Data"
fi

cat <<'EOF'

==> the pipeline stages

Not run here. The single-cell stages are the notebooks under scrnaseq/, run in order by

    python scrnaseq/run_chain.py

from the public GEO data and the Zenodo object listed in data/DOWNLOAD.md. SMD is not re-run: the
z-scores it saved ship in scrnaseq/chain_inputs/ (smd/README.md). The imaging stages are the
notebooks under imaging/ and need the raw images. The figure scripts above read the stages' saved
outputs instead, which is why everything above runs without them.
EOF
