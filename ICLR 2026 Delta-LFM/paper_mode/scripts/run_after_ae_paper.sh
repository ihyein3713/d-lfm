#!/usr/bin/env bash
# Continue the paper-setting pipeline the moment a paper_mode autoencoder run finishes:
#   step 4  latents for every visit          -> outputs/latents_<TAG>
#   step 6  flow matching, ONE stage         -> outputs/fm_<TAG>_rectified
# There is no sigma probe: the paper's flow starts at the baseline latent z_i, not at Gaussian
# noise, so --fm_flowinit_sigma has nothing to scale (see below).
#
# Usage: TAG=ae_paperA GPU=4 bash scripts/run_after_ae_paper.sh
#   It waits for the step1_paper.py process of that TAG to exit, then uses its LAST checkpoint.
#   AE_CKPT=<path> skips the wait and uses that checkpoint directly.
#
# Differences from the repo chain (paper_mode/PAPER_SETTINGS.md section 4):
#   --fm_scheme realtime --fm_res_noise 0 --fm_x0_pred 0
#                                         u* = (z1-z0)/dt along the straight path z_i -> z_j,
#                                         which is what src/flow.py calls "Delta-LFM real-time
#                                         (Eq.12)". The repo recipe instead uses --fm_res_noise 1
#                                         (Delta-Res-Flow: noise -> delta), a repo addition the
#                                         paper does not have.
#   --pm_no_class 1                       continuous interval conditioning, no 15-year cap (App. B)
#   --pm_dx_noise                         Gaussian noise on the clinical status (App. F)
#   --fm_hist_mode none                   the paper conditions on the baseline only, not prev1/prev2
#   one stage, 200 epochs, lr 3e-5, batch 4                                   (App. A)
#   no energy mask, no soft cF1           those are repo additions
#   latents at the native canonical grid 128x144x128 -> 4x32x36x32 (no --res_scale)
# The card is pinned with CUDA_VISIBLE_DEVICES: step2/step3 set it only after importing
# diffusers/accelerate, too late for --gpu to take effect.
set -u
export PATH="${CONDA_ENV:-$HOME/miniconda3/envs/longi}/bin:$PATH"
TAG="${TAG:-ae_paperA}"
GPU="${GPU:-4}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset ACCELERATE_MIXED_PRECISION          # fp32, as in the repo-version flow matching

REPO="/mnt/aix22308/longi_v2/d_lfm/ICLR 2026 Delta-LFM"
PM="$REPO/paper_mode"
PM_DATA="$PM/data"                        # uncapped pairs, built by scripts/../outputs/data_prep.log
OUT="$PM/outputs"
LAT="$OUT/latents_$TAG"
LOG="$OUT/chain_$TAG.log"
CSV="$PM_DATA/derived/AD-Progression-multiprior-dt15.csv"   # name is the builder's; no dt cap inside
SPLIT_ARGS="--split_col split --eval_split val"
DX_NOISE="${DX_NOISE:-0.1667}"            # see open item M3 in PAPER_SETTINGS.md
COND_MODE="${COND_MODE:-adaln}"           # paper Appendix B: AdaLN beats additive biasing
N_EPOCHS="${N_EPOCHS:-200}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
cd "$REPO" || exit 1

# ---- wait for this TAG's autoencoder to finish ----
AE_CKPT="${AE_CKPT:-}"
if [ -z "$AE_CKPT" ]; then
  while :; do
    n=$(for p in $(pgrep -f step1_paper.py 2>/dev/null); do
          tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -c "output_dir $TAG"
        done | paste -sd+ | bc 2>/dev/null)
    [ "${n:-0}" -eq 0 ] && break
    log "waiting for the $TAG autoencoder to finish ($n processes)"; sleep 300
  done
  AE_CKPT=$(ls -1 "$OUT/$TAG"/all-ae-*-3D.pth 2>/dev/null \
            | sed 's/.*all-ae-\([0-9]*\)-3D.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
fi
[ -n "$AE_CKPT" ] && [ -f "$AE_CKPT" ] || { log "no autoencoder checkpoint for $TAG"; exit 1; }
AE_CKPT="$(readlink -f "$AE_CKPT")"       # absolute: load_if() silently keeps the pretrained MAISI
                                          # weights when a relative path does not resolve here
log "chain started: TAG=$TAG GPU=$GPU AE_CKPT=$AE_CKPT"
[ -f "$CSV" ] || { log "pair CSV missing: $CSV (run custom/run_adni.sh with DATA_DIR=$PM_DATA)"; exit 1; }

# ---- step 4: latents, native canonical grid ----
log "step 4: extracting latents into $LAT"
python step2_extract_latents.py \
  --config config/default/Step2_3D_Extract.yaml \
  --data_dir "$PM_DATA" --latent_path "$LAT" --output_dir "$LAT" \
  --temp_path "$OUT" --aekl_ckpt "$AE_CKPT" \
  --dataset_csv derived/AD-Progression-All.csv --split_col split \
  --norm_mode std --gpu "$GPU" >> "$OUT/step2_$TAG.log" 2>&1 || {
    log "step 4 FAILED, see step2_$TAG.log"; exit 1; }
n_lat=$(find "$LAT" -name '*.npz' | wc -l)
log "step 4 done: $n_lat latents (753 expected)"
if grep -q "not found, using random initialization" "$OUT/step2_$TAG.log"; then
  log "step 4 did NOT load $AE_CKPT: the latents would come from the pretrained MAISI. Stopping."
  exit 1
fi
grep -a "Successful load" "$OUT/step2_$TAG.log" | tail -1 | while read -r l; do log "loaded: $l"; done

# ---- step 6: flow matching, single stage, paper settings ----
log "step 6: flow matching, $N_EPOCHS epochs, lr 3e-5, batch 4"
python paper_mode/src/step3_paper.py \
  --config config/default/Step3_3D_FM_multidt15.yaml \
  --data_dir "$PM_DATA" --dataset_csv "$CSV" --latent_path "$LAT" --temp_path "$OUT" \
  --aekl_ckpt "$AE_CKPT" --output_dir "fm_$TAG" \
  --norm_mode std --cache_dir none \
  --fm_scheme realtime --fm_scale_norm 0 --fm_res_noise 0 --fm_x0_pred 0 \
  --fm_hist_mode none $SPLIT_ARGS \
  --fm_mask_lambda 0 --fm_rf1_w 0.0 --fm_cf1_w 0.0 --fm_dir_w 0.0 \
  --pm_no_class 1 --pm_dx_noise "$DX_NOISE" --pm_cond_mode "$COND_MODE" \
  --batch_size 4 --lr 3e-5 --n_epochs "$N_EPOCHS" \
  --gpu "$GPU" >> "$OUT/fm_$TAG.log" 2>&1
rc=$?
log "step 6 finished with exit code $rc (log: fm_$TAG.log)"
log "checkpoints: $OUT/fm_${TAG}_rectified"
log "next: evaluation with paper metrics (scripts/eval_fm_paper.sh)"
exit $rc
