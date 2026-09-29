#!/usr/bin/env python
"""Latent-trajectory metrics for saved autoencoder checkpoints, for picking one (RECIPE section 3).

The trainer only runs the linearity probe every --eval_every epochs (the recipe sets 99, so it
fires once). This runs the same probe offline on any checkpoints, on the same visit loader the
trainer uses, so reconstruction and trajectory quality can be compared epoch by epoch.

Usage (pin the card with CUDA_VISIBLE_DEVICES, --gpu alone is too late for these scripts):
    CUDA_VISIBLE_DEVICES=5 python custom/ae_traj_metrics.py \
        --config config/default/Step1_3D_AE_Contastive.yaml \
        --data_dir $DATA_DIR --dataset_csv derived/AD-Progression-All.csv \
        --split_col split --eval_split val --norm_mode std --res_scale 0.82 --crop_mode patch \
        --ckpts outputs/ae/all-ae-*.pth
"""
import os
import re
import sys
import glob
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ckpts = []
if "--ckpts" in sys.argv:                      # consume before utils.options parses argv
    i = sys.argv.index("--ckpts")
    j = i + 1
    while j < len(sys.argv) and not sys.argv[j].startswith("--"):
        _ckpts.extend(glob.glob(sys.argv[j]))
        j += 1
    del sys.argv[i:j]

import torch  # noqa: E402
from utils.options import args  # noqa: E402
from utils import import_from_dotted_path  # noqa: E402
from dataset.ad_progression_3D_triplet import get_visit_dataset  # noqa: E402
from monitor_latent_linearity import evaluate_linearity  # noqa: E402


def epoch_of(path):
    m = re.search(r"-ae-(\d+)-", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    if not _ckpts:
        sys.exit("pass checkpoints with --ckpts <glob>")
    ckpts = sorted(set(_ckpts), key=epoch_of)
    split = getattr(args, "eval_split", "test") or "test"
    warnings.filterwarnings("ignore")

    loader, vdf = get_visit_dataset(args, mode=split)
    print(f"probe loader: {len(loader.dataset)} visits from the '{split}' split", flush=True)

    model = import_from_dotted_path(args.autoencoder)(args).float().cuda().eval()
    cols = ("line_r2_mean", "mono_rho_mean", "mono_rho_shuf_mean", "mono_gap",
            "step_cos_mean", "dir_diversity", "vel_r2_mean", "n_patients")
    print("\nckpt(epoch)  " + "  ".join(f"{c.replace('_mean',''):>12}" for c in cols))
    for p in ckpts:
        state = torch.load(p, map_location="cpu", weights_only=True)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        with torch.no_grad():
            m = evaluate_linearity(model, loader, "cuda",
                                   max_batches=(getattr(args, "lin_max_batches", 0) or None),
                                   norm_mode=getattr(args, "norm_mode", "01"))
        # the checkpoint of epoch N is saved as all-ae-<N+1>-3D.pth
        print(f"{os.path.basename(p):<18s} ep{epoch_of(p) - 1:<3d}" +
              "  ".join(f"{m.get(c, float('nan')):12.4f}" for c in cols), flush=True)


if __name__ == "__main__":
    main()
