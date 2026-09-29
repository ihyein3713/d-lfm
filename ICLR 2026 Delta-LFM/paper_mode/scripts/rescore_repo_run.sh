#!/usr/bin/env bash
# Add the paper's metrics (Delta-RMAE, Region MAE) to the finished repo-version evaluation,
# by rebuilding each prediction from the saved latent change instead of re-running the model.
# Writes only into paper_mode/outputs/paper_metrics.
set -u
export PATH="${CONDA_ENV:-$HOME/miniconda3/envs/longi}/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${GPU:-4}"
cd "/mnt/aix22308/longi_v2/d_lfm/ICLR 2026 Delta-LFM" || exit 1
for tag in fm_rectified_ep7 fm_rectified_ep8 fm_rectified_ep9 \
           fm_base_rectified_ep27 fm_base_rectified_ep28 fm_base_rectified_ep29; do
  echo "################ $tag ################"
  python paper_mode/src/paper_metrics_from_dump.py --tag "$tag"
  echo "$tag exit: $?"
done
echo "################ done ################"
