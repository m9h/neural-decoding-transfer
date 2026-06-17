#!/usr/bin/env bash
# Reproducible build (Python/Sweave pattern): tangle the experiment logs into figures +
# macros + tables, then weave the LaTeX. One command regenerates the manuscript from the
# raw pod pulls in ../dataeff/*.jsonl + data/facts.json -- nothing is hand-transcribed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TECTONIC=/home/mhough/miniforge3/envs/texlive/bin/tectonic
python "$HERE/generate.py"                                     # tangle
( cd "$HERE" && "$TECTONIC" --keep-logs --reruns 3 paper.tex ) # weave
echo "built paper/paper.pdf"
