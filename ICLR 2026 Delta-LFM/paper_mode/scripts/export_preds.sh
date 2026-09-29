#!/usr/bin/env bash
# Decode the saved latent changes of the finished repo-version evaluations into NIfTI, so the
# volume-based Region MAE can segment them later. Export only (--skip_seg): the segmentation runs
# on the CPU afterwards, because TensorFlow grabs a whole card and must not share a GPU.
set -u
export PATH="${CONDA_ENV:-$HOME/miniconda3/envs/longi}/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${GPU:-4}"
cd "/mnt/aix22308/longi_v2/d_lfm/ICLR 2026 Delta-LFM" || exit 1
for tag in "$@"; do
  echo "################ $tag ################"
  python -u paper_mode/src/region_volumes.py --tag "$tag" --skip_seg
  echo "$tag export exit: $?"
done
