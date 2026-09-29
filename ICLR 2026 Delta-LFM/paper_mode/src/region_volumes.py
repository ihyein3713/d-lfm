#!/usr/bin/env python
"""Region MAE the way the paper measures it: volumes of clinical structures, not voxel intensities.

The paper follows BrLP and reports "region-level MAE over clinically relevant structures
(hippocampus, amygdala, lateral ventricles, CSF, thalamus)". BrLP measures the VOLUME of each
structure, which means the generated image has to be segmented; comparing intensities inside the
baseline's mask (what src/paper_metrics.py does) is a different quantity.

This script produces the volume version:
  1. rebuild every test prediction from the saved latent change, as src/paper_metrics_from_dump.py
     does (decode(z0 + delta)), and write prediction and ground truth as NIfTI;
  2. run standalone SynthSeg on both, with --vol, which writes each structure's volume in mm^3;
  3. report |V_pred - V_gt| per structure, and the mean over structures.

Both sides are segmented with the same model, so the comparison is internally consistent. Note it
is NOT the stored FreeSurfer segmentation: only SynthSeg 1.0 ships with the standalone repo
(2.0 and robust are behind an interactive download), so --v1 is used, and the ground truth is
re-segmented rather than reusing seg.nii.gz.

The affine written into the NIfTI files is the baseline scan's, so the voxel size is the real
1.5 mm; the padding offset of the canonical grid does not affect volumes.

Usage (run from the repo root; GPU needed, SynthSeg peaks near 11 GB):
    CUDA_VISIBLE_DEVICES=4 python paper_mode/src/region_volumes.py --tag fm_rectified_ep9
"""
import os
import sys
import json
import glob
import argparse
import subprocess

_PM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_PM_DIR))
for _p in (_REPO, _PM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import nibabel as nib

from paper_metrics import REGIONS
from paper_metrics_from_dump import load_ae, latent_of, up, CANON

SYNTHSEG_DIR = os.environ.get("SYNTHSEG_DIR", "/mnt/aix22308/tools/SynthSeg")
SYNTHSEG_PY = os.environ.get("SYNTHSEG_PY", "/mnt/aix22308/envs/synthseg/bin/python")
# SynthSeg's volume CSV names columns after the FreeSurfer structures; map the paper's five
# structures onto those names (left and right are summed, as the labels are in paper_metrics).
CSV_COLUMNS = {
    "hippocampus": ["left hippocampus", "right hippocampus"],
    "amygdala": ["left amygdala", "right amygdala"],
    "lateral_ventricle": ["left lateral ventricle", "right lateral ventricle"],
    "csf": ["CSF"],
    "thalamus": ["left thalamus", "right thalamus"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--eval_dir", default=os.path.join(_REPO, "ae_runs", "fm_eval"))
    p.add_argument("--latents", default="/mnt/aix22308/longi_v2/d_lfm/outputs/latents")
    p.add_argument("--ae_ckpt", default="/mnt/aix22308/longi_v2/d_lfm/outputs/ae/all-ae-27-3D.pth")
    p.add_argument("--work_dir", default=os.path.join(_PM_DIR, "..", "outputs", "region_volumes"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gt_only", action="store_true", help="write and segment the ground truth only")
    p.add_argument("--skip_seg", action="store_true", help="reuse existing segmentations/volumes")
    p.add_argument("--cpu", action="store_true", help="run SynthSeg on the CPU (no GPU contention)")
    p.add_argument("--threads", type=int, default=8, help="CPU threads for SynthSeg")
    return p.parse_args()


def write_nii(vol, affine, path):
    nib.save(nib.Nifti1Image(np.asarray(vol, dtype=np.float32), affine), path)


def export(a):
    """Decode every pair of the tag and write pred/gt NIfTI files. Returns the case ids."""
    dump = os.path.join(a.eval_dir, "delta", a.tag)
    pids = [l.strip() for l in open(os.path.join(dump, "pids.txt")) if l.strip()]
    deltas = sorted(glob.glob(os.path.join(dump, "delta_*.npy")))
    assert len(pids) == len(deltas), (len(pids), len(deltas))
    if a.limit:
        pids, deltas = pids[:a.limit], deltas[:a.limit]

    pred_dir = os.path.join(a.work_dir, a.tag, "pred")
    gt_dir = os.path.join(a.work_dir, "gt")
    for d in (pred_dir, gt_dir):
        os.makedirs(d, exist_ok=True)

    # the autoencoder is only needed when something still has to be decoded; loading it
    # unconditionally would require a GPU even on a pure re-read of an existing export
    ae = None

    def _ae():
        nonlocal ae
        if ae is None:
            ae = load_ae(a.ae_ckpt)
        return ae

    cases = []
    for i, (pid, dfile) in enumerate(zip(pids, deltas)):
        p0, p1 = pid.split("->")
        b = p0.rstrip("/").split("/")
        f = p1.rstrip("/").split("/")
        # the ground truth is one per FOLLOW-UP visit, but a prediction exists per PAIR: the same
        # follow-up is predicted from several baselines, so the prediction key carries both dates
        case_gt = f[-3] + "__" + f[-2]
        case_pred = f[-3] + "__" + b[-2] + "__" + f[-2]
        cases.append((case_pred, case_gt, pid))
        f_pred = os.path.join(pred_dir, case_pred + ".nii.gz")
        f_gt = os.path.join(gt_dir, case_gt + ".nii.gz")
        if os.path.exists(f_gt) and (a.gt_only or os.path.exists(f_pred)):
            continue
        affine = nib.load(p0).affine                  # real 1.5 mm geometry
        with torch.no_grad():
            if not os.path.exists(f_gt):
                z1 = torch.from_numpy(latent_of(p1, a.latents))[None].float()
                write_nii(up(_ae().decode(z1.cuda()).float().cpu()[0, 0].numpy()), affine, f_gt)
            if not a.gt_only and not os.path.exists(f_pred):
                z0 = torch.from_numpy(latent_of(p0, a.latents))[None].float()
                dz = torch.from_numpy(np.load(dfile))[None].float()
                write_nii(up(_ae().decode((z0 + dz).cuda()).float().cpu()[0, 0].numpy()), affine, f_pred)
        if (i + 1) % 25 == 0:
            print(f"  exported {i + 1}/{len(pids)}", flush=True)
    if ae is not None:
        del ae
        torch.cuda.empty_cache()
    return cases, pred_dir, gt_dir


def segment(in_dir, out_dir, vol_csv, cpu=False, threads=8):
    """Run standalone SynthSeg over a folder. --v1 because only that model ships with the repo.

    TF_FORCE_GPU_ALLOW_GROWTH is mandatory: TensorFlow otherwise reserves the whole card on
    startup (36 GB observed), which killed a PyTorch training job sharing the GPU. Even with it,
    only run this on a card with no other job.
    """
    if os.path.exists(vol_csv):
        print(f"  volumes already present: {vol_csv}", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    cmd = [SYNTHSEG_PY, os.path.join(SYNTHSEG_DIR, "scripts", "commands", "SynthSeg_predict.py"),
           "--i", in_dir, "--o", out_dir, "--vol", vol_csv, "--v1"]
    if cpu:
        cmd += ["--cpu", "--threads", str(threads)]
    env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="1")
    print("  " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=SYNTHSEG_DIR, env=env)
    if r.returncode != 0 or not os.path.exists(vol_csv):
        raise SystemExit(f"SynthSeg failed (exit {r.returncode}) for {in_dir}")


def read_volumes(csv_path):
    import pandas as pd
    d = pd.read_csv(csv_path)
    key = d.columns[0]
    d[key] = d[key].astype(str).str.replace(r"\.nii(\.gz)?$", "", regex=True)
    out = {}
    for _, row in d.iterrows():
        per = {}
        for name, cols in CSV_COLUMNS.items():
            have = [c for c in cols if c in d.columns]
            if not have:
                continue
            per[name] = float(sum(float(row[c]) for c in have))
        out[str(row[key])] = per
    return out


def main():
    a = parse_args()
    a.work_dir = os.path.abspath(a.work_dir)
    cases, pred_dir, gt_dir = export(a)
    gt_csv = os.path.join(a.work_dir, "gt_volumes.csv")
    pred_csv = os.path.join(a.work_dir, a.tag, "pred_volumes.csv")
    if not a.skip_seg:
        segment(gt_dir, os.path.join(a.work_dir, "gt_seg"), gt_csv, a.cpu, a.threads)
        if not a.gt_only:
            segment(pred_dir, os.path.join(a.work_dir, a.tag, "pred_seg"), pred_csv, a.cpu, a.threads)
    if a.gt_only:
        print("ground truth done:", gt_csv)
        return

    if not (os.path.exists(gt_csv) and os.path.exists(pred_csv)):
        # --skip_seg exports the volumes and stops; the segmentation runs separately on the CPU
        print("volumes not computed yet (missing %s). Export done."
              % (gt_csv if not os.path.exists(gt_csv) else pred_csv), flush=True)
        return
    gtv, prv = read_volumes(gt_csv), read_volumes(pred_csv)
    rows = []
    for case_pred, case_gt, pid in cases:
        if case_gt not in gtv or case_pred not in prv:
            continue
        r = {"case": case_pred, "pid": pid}
        for name in REGIONS:
            if name in gtv[case_gt] and name in prv[case_pred]:
                r["mae_" + name] = abs(prv[case_pred][name] - gtv[case_gt][name])
                r["rel_" + name] = r["mae_" + name] / max(gtv[case_gt][name], 1e-6)
        vals = [r[k] for k in r if k.startswith("mae_")]
        rel = [r[k] for k in r if k.startswith("rel_")]
        r["mae_mean"] = float(np.mean(vals)) if vals else float("nan")
        r["rel_mean"] = float(np.mean(rel)) if rel else float("nan")
        rows.append(r)

    keys = [k for k in rows[0] if k not in ("case", "pid")]
    summary = {k: round(float(np.nanmean([r.get(k, np.nan) for r in rows])), 4) for k in keys}
    out = {"tag": a.tag, "n_cases": len(rows), "model": "synthseg_1.0",
           "units": "mm^3 for mae_*, fraction of the ground-truth volume for rel_*",
           "mean": summary, "rows": rows}
    dst = os.path.join(a.work_dir, a.tag, "region_volume_mae.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
