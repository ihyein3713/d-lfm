#!/usr/bin/env bash
# Paper-setting autoencoder (arXiv 2512.09185v4, Sec. 4.1 + Appendix A + Appendix C).
# Everything it writes stays under paper_mode/outputs; the repo tree is only read.
#
# Usage: TAG=ae_paperA PM_ARC=0.35 PM_RANK=0.2 PM_MARGIN=0.05 GPU=4 bash scripts/train_ae_paper.sh
#   SMOKE=1 runs 10 steps and exits, for checking the wiring before a long run.
#
# Paper values used here: AdamW, lr 1e-3, batch 2, 300 epochs (Appendix A), 64^3 crop and
# encoder channels [64,128,256] (Appendix C; the MAISI autoencoder already has those channels).
# Off, because the paper does not have them (decision D4/D6): patch adversarial loss,
# --ae_chgrec_w, --ae_dircosE_w, and every repo trajectory term (the paper ArcRank replaces them).
# Kept from the repo recipe, since the paper is silent and they are data-pipeline defaults:
# the shared rigid augmentation of a triplet (--aug_rigid 3 --aug_rot_deg 5).
set -u
export PATH="${CONDA_ENV:-$HOME/miniconda3/envs/longi}/bin:$PATH"
GPU="${GPU:-4}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# precision: bf16, the same choice the repo-version autoencoder run used, so the two are
# comparable. The paper does not state a precision. AE_PRECISION=fp32 switches it off.
if [ "${AE_PRECISION:-bf16}" = "bf16" ]; then export ACCELERATE_MIXED_PRECISION=bf16; else unset ACCELERATE_MIXED_PRECISION; fi

TAG="${TAG:-ae_paper}"
PM_ARC="${PM_ARC:-0.35}"
PM_RANK="${PM_RANK:-0.2}"
PM_MARGIN="${PM_MARGIN:-0.05}"
N_EPOCHS="${N_EPOCHS:-300}"
SMOKE="${SMOKE:-0}"
# RESUME_CKPT: continue from an existing autoencoder checkpoint (weights only; the optimizer state
# is not saved by the trainer). Use a NEW TAG when resuming, otherwise the fresh epoch numbering
# overwrites the checkpoints of the interrupted run.
RESUME_CKPT="${RESUME_CKPT:-}"

REPO="/mnt/aix22308/longi_v2/d_lfm/ICLR 2026 Delta-LFM"
PM="$REPO/paper_mode"
export DATA_DIR=/mnt/aix22308/data/ADNI_delta_lfm
export WORK_DIR="$PM/outputs"
mkdir -p "$WORK_DIR"
cd "$REPO" || exit 1

EXTRA=()
if [ "$SMOKE" = "1" ]; then
  EXTRA=(--n_epochs 1 --steps_per_epoch "${SMOKE_STEPS:-10}" --save_every 1 --eval_every 1 --eval_steps "${SMOKE_STEPS:-10}")
  TAG="${TAG}_smoke"
else
  # 306 train triplets / batch 2 = 153 steps per epoch, so 300 epochs is about 46k steps.
  # save_every 10 keeps 30 checkpoints instead of 300; eval_every 25 runs the linearity probe
  # a dozen times, which is how the ArcRank effect on the trajectory gets tracked.
  EXTRA=(--n_epochs "$N_EPOCHS" --steps_per_epoch 0 --save_every 10 --eval_every 25 --eval_steps 250)
fi

echo "[paper_mode] TAG=$TAG GPU=$GPU arc=$PM_ARC rank=$PM_RANK margin=$PM_MARGIN epochs=$N_EPOCHS"
python paper_mode/src/step1_paper.py \
  --config paper_mode/config/Step1_paper.yaml \
  --data_dir "$DATA_DIR" --temp_path "$WORK_DIR" --output_dir "$TAG" \
  --dataset_csv derived/AD-Progression-All.csv \
  --split_col split --eval_split val \
  --norm_mode std --crop_mode patch \
  --batch_size 2 --lr 1e-3 --num_workers "${NUM_WORKERS:-8}" --warmup_epochs 0 \
  --aug_rigid 3 --aug_rot_deg 5 \
  --no-use_adv \
  --ae_chgrec_w 0 --ae_dircosE_w 0 \
  --lin_max_batches 8 \
  ${RESUME_CKPT:+--aekl_ckpt "$RESUME_CKPT"} \
  --pm_arc "$PM_ARC" --pm_rank "$PM_RANK" --pm_margin "$PM_MARGIN" \
  --gpu "$GPU" "${EXTRA[@]}"
echo "[paper_mode] exit: $?"
