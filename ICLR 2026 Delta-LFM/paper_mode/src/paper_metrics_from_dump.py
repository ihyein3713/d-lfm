#!/usr/bin/env python
"""Score a finished evaluation with the paper's metrics, without re-running the flow model.

eval_fm.py already saved, for every test pair, the predicted latent change in unscaled latent
units (ae_runs/fm_eval/delta/<tag>/delta_NNNN.npy) plus the case list (pids.txt). Together with
the per-visit latents this is enough to rebuild the predicted image: decode(z0 + delta). So the
paper metrics can be added to the repo-version results with a few hundred decoder passes instead
of a 95-minute re-evaluation.

Self-check: PSNR and SSIM are recomputed here with the repo's own helpers and compared against
the values in the result JSON. If the two agree, the rebuilt images are the ones that were scored.

Region MAE needs the segmentation, which lives on the canonical 128x144x128 grid, so the decoded
112^3 volumes are resized up with the same trilinear resize eval_fm.py uses for its raw output.

Usage (run from the repo root):
    CUDA_VISIBLE_DEVICES=4 python paper_mode/src/paper_metrics_from_dump.py \
        --tag fm_rectified_ep9 --ae_ckpt ../../outputs/ae/all-ae-27-3D.pth
"""
import os
import sys
import json
import glob
import argparse

_PM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_PM_DIR))
for _p in (_REPO, _PM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

from paper_metrics import delta_rmae, region_mae, REGIONS

CANON = (128, 144, 128)


# psnr and ssim3d are copied verbatim in behaviour from ../../eval_fm.py (lines 354 and 365), so
# the self-check compares like with like. eval_fm.py cannot be imported: it parses argv on import.
def psnr(a, b, data_range=None):
    if data_range is None:
        data_range = float(b.max() - b.min()) or 1.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


_SSIM_CACHE = {}


def ssim3d(a, b, dr):
    from monai.metrics import SSIMMetric
    k = round(float(dr), 4)
    if k not in _SSIM_CACHE:
        _SSIM_CACHE[k] = SSIMMetric(spatial_dims=3, data_range=float(dr),
                                    kernel_type="gaussian", win_size=11, kernel_sigma=1.5)
    ta = torch.from_numpy(np.ascontiguousarray(a)).float()[None, None]
    tb = torch.from_numpy(np.ascontiguousarray(b)).float()[None, None]
    return float(_SSIM_CACHE[k](ta, tb).mean())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="evaluation tag, e.g. fm_rectified_ep9")
    p.add_argument("--eval_dir", default=os.path.join(_REPO, "ae_runs", "fm_eval"))
    p.add_argument("--latents", default="/mnt/aix22308/longi_v2/d_lfm/outputs/latents")
    p.add_argument("--ae_ckpt", default="/mnt/aix22308/longi_v2/d_lfm/outputs/ae/all-ae-27-3D.pth")
    p.add_argument("--out_dir", default=os.path.join(_PM_DIR, "..", "outputs", "paper_metrics"))
    p.add_argument("--limit", type=int, default=0, help="score only the first N pairs (0 = all)")
    p.add_argument("--batch", type=int, default=2)
    return p.parse_args()


def load_ae(ckpt):
    """Same construction eval_fm.py uses: MAISI definition, then our trained weights."""
    from src.autoencoder.MAISI_Unet3D import init_autoencoder
    ae = init_autoencoder()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    ae.load_state_dict(state)
    return ae.float().cuda().eval()


def latent_of(img_path, latent_root):
    """Image/<subject>/<visit>/t1.nii.gz -> <latent_root>/<subject>/<visit>/t1.npz"""
    parts = img_path.rstrip("/").split("/")
    sub, visit, stem = parts[-3], parts[-2], parts[-1].split(".")[0]
    f = os.path.join(latent_root, sub, visit, stem + ".npz")
    return np.load(f)["data"]


def seg_of(img_path):
    return os.path.join(os.path.dirname(img_path), "seg.nii.gz")


_PADCROP = None


def to_canonical_labels(seg):
    """Put a label volume on the canonical grid with the transform the image pipeline uses.

    The scans are 121x145x121 at 1.5 mm; eval_fm.py brings them to 128x144x128 with
    MONAI ResizeWithPadOrCrop(mode="constant"), which pads with 0 and centre-crops. Labels get the
    same treatment (no interpolation is involved, so nearest-neighbour questions do not arise).
    """
    global _PADCROP
    if tuple(seg.shape) == CANON:
        return seg
    if _PADCROP is None:
        from monai.transforms import ResizeWithPadOrCrop
        _PADCROP = ResizeWithPadOrCrop(spatial_size=CANON, mode="constant")
    out = _PADCROP(seg[None].astype(np.int16))
    return np.asarray(out)[0]


def up(vol):
    """Trilinear resize of one 3D volume from the model grid to the canonical grid."""
    t = torch.from_numpy(np.ascontiguousarray(vol))[None, None].float()
    t = F.interpolate(t, size=CANON, mode="trilinear", align_corners=False)
    return t[0, 0].numpy().astype(np.float64)


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dump = os.path.join(a.eval_dir, "delta", a.tag)
    pids = [l.strip() for l in open(os.path.join(dump, "pids.txt")) if l.strip()]
    deltas = sorted(glob.glob(os.path.join(dump, "delta_*.npy")))
    assert len(pids) == len(deltas), (len(pids), len(deltas))
    if a.limit:
        pids, deltas = pids[:a.limit], deltas[:a.limit]
    print(f"[paper-metrics] {a.tag}: {len(pids)} pairs", flush=True)

    ae = load_ae(a.ae_ckpt)

    rows = []
    for i, (pid, dfile) in enumerate(zip(pids, deltas)):
        p0, p1 = pid.split("->")
        z0 = torch.from_numpy(latent_of(p0, a.latents))[None].float()
        z1 = torch.from_numpy(latent_of(p1, a.latents))[None].float()
        dz = torch.from_numpy(np.load(dfile))[None].float()
        with torch.no_grad():
            ip = ae.decode(z0.cuda()).float().cpu()[0, 0].numpy().astype(np.float64)
            tg = ae.decode(z1.cuda()).float().cpu()[0, 0].numpy().astype(np.float64)
            pr = ae.decode((z0 + dz).cuda()).float().cpu()[0, 0].numpy().astype(np.float64)

        brain = (ip > 0.05) | (tg > 0.05)
        if brain.sum() == 0:
            brain = np.ones_like(ip, dtype=bool)
        dr = float(tg.max() - tg.min()) or 1.0

        row = {
            "pid": pid,
            "dRMAE_paper": delta_rmae(pr, tg, ip, brain),
            "dRMAE_paper_voxel": delta_rmae(pr, tg, ip, brain, mode="voxel"),
            "dRMAE_copy": delta_rmae(ip, tg, ip, brain),
            "PSNR": psnr(pr, tg, dr), "SSIM": ssim3d(pr, tg, dr),
            "PSNR_copy": psnr(ip, tg, dr), "SSIM_copy": ssim3d(ip, tg, dr),
        }

        segf = seg_of(p0)
        if os.path.exists(segf):
            seg = to_canonical_labels(np.asarray(nib.load(segf).dataobj))
            rm = region_mae(up(pr), up(tg), seg)
            rmc = region_mae(up(ip), up(tg), seg)
            row.update({"regionMAE_" + k: v for k, v in rm.items()})
            row.update({"regionMAE_copy_" + k: v for k, v in rmc.items()})
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(pids)}", flush=True)

    keys = [k for k in rows[0] if k != "pid" and isinstance(rows[0][k], float)]
    summary = {k: round(float(np.nanmean([r.get(k, np.nan) for r in rows])), 4) for k in keys}
    sd = {k: round(float(np.nanstd([r.get(k, np.nan) for r in rows])), 4) for k in keys}

    ref_file = os.path.join(a.eval_dir, a.tag + ".json")
    check = {}
    if os.path.exists(ref_file):
        ref = json.load(open(ref_file))
        check = {"reported_PSNR": ref["pred"]["PSNR"], "rebuilt_PSNR": summary["PSNR"],
                 "reported_SSIM": ref["pred"]["SSIM"], "rebuilt_SSIM": summary["SSIM"]}
        check["PSNR_delta"] = round(check["rebuilt_PSNR"] - check["reported_PSNR"], 4)
        check["SSIM_delta"] = round(check["rebuilt_SSIM"] - check["reported_SSIM"], 4)

    out = {"tag": a.tag, "n_pairs": len(rows), "ae_ckpt": a.ae_ckpt,
           "mean": summary, "sd": sd, "self_check": check, "rows": rows}
    dst = os.path.join(a.out_dir, a.tag + ".json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"mean": summary, "self_check": check}, indent=1), flush=True)
    print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()
