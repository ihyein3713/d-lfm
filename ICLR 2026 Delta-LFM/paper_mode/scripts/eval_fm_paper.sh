#!/usr/bin/env bash
# Evaluate a paper_mode flow-matching run: the repo panel (eval_fm.py) followed by the paper
# metrics (Delta-RMAE, Region MAE) rebuilt from the saved latent changes.
#
# Usage: TAG=ae_paperA EPOCHS="197 198 199" GPU=4 bash scripts/eval_fm_paper.sh
# Three consecutive epochs, as the paper's evaluation standard asks for a three-epoch mean.
set -u
export PATH="${CONDA_ENV:-$HOME/miniconda3/envs/longi}/bin:$PATH"
GPU="${GPU:-4}"
export CUDA_VISIBLE_DEVICES="$GPU"
unset ACCELERATE_MIXED_PRECISION
TAG="${TAG:-ae_paperA}"
EPOCHS="${EPOCHS:-197 198 199}"
REPO="/mnt/aix22308/longi_v2/d_lfm/ICLR 2026 Delta-LFM"
PM="$REPO/paper_mode"
PM_DATA="$PM/data"
OUT="$PM/outputs"
LAT="$OUT/latents_$TAG"
CSV="$PM_DATA/derived/AD-Progression-multiprior-dt15.csv"
AE_CKPT="${AE_CKPT:-$(ls -1 "$OUT/$TAG"/all-ae-*-3D.pth | sed 's/.*all-ae-\([0-9]*\)-3D.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)}"
# paper eq 14-15: Euler with dt = 0.01 years, so the step count follows the interval of each case.
# paper_mode/src/eval_fm_paper.py computes it per batch (batch_size 1 here = per case).
DT_STEP="${DT_STEP:-0.01}"
COND_MODE="${COND_MODE:-adaln}"
cd "$REPO" || exit 1

for ep in $EPOCHS; do
  ck="$OUT/fm_${TAG}_rectified/fm-unet-ep-$ep.pth"
  tag="fm_${TAG}_ep$ep"
  [ -f "$ck" ] || { echo "missing $ck"; continue; }
  echo "################ $tag ################"
  python paper_mode/src/eval_fm_paper.py \
    --config config/default/Step3_3D_FM_multidt15.yaml \
    --data_dir "$PM_DATA" --dataset_csv "$CSV" --latent_path "$LAT" --temp_path "$OUT" \
    --split_col split --aekl_ckpt "$AE_CKPT" --fm_ckpt "$ck" \
    --norm_mode std --cache_dir none \
    --fm_scheme realtime --fm_scale_norm 0 --fm_res_noise 0 \
    --fm_mask_lambda 0 --fm_hist_mode none \
    --fm_det 1 --n_avg 1 --fm_antithetic 0 --fm_seed 1234 --fm_sigma 0 \
    --pm_dt_step "$DT_STEP" --pm_cond_mode "$COND_MODE" \
    --batch_size 1 --num_workers 1 --n_eval 1000000 --raw_out 1 \
    --tag "$tag" --gpu "$GPU"
  echo "$tag eval exit: $?"
  python paper_mode/src/paper_metrics_from_dump.py --tag "$tag" \
    --latents "$LAT" --ae_ckpt "$AE_CKPT"
  echo "$tag paper-metrics exit: $?"
done
echo "################ done ################"
