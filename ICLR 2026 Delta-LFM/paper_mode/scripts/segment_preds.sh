#!/usr/bin/env bash
# Segment the exported predictions on the CPU and compute the paper's volume-based Region MAE.
# CPU only, on purpose: TensorFlow reserves a whole GPU on startup and must never share a card
# with a training job (it killed one on 2026-09-28).
set -u
export PATH="${CONDA_ENV:-$HOME/miniconda3/envs/longi}/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""
THREADS="${THREADS:-12}"
cd "/mnt/aix22308/longi_v2/d_lfm/ICLR 2026 Delta-LFM" || exit 1
for tag in "$@"; do
  echo "################ $tag ################"
  python -u paper_mode/src/region_volumes.py --tag "$tag" --cpu --threads "$THREADS"
  echo "$tag exit: $?"
done
