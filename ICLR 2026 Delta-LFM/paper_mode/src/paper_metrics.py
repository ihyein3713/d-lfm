"""Metrics as the paper defines them (arXiv 2512.09185v4), which the repo does not provide.

Delta-RMAE (eq 17, restated in eq 19 of Appendix E):

    Delta-RMAE = |D_gt - D_gen| / (0.5 * (|D_gt| + |D_gen|)),    D = follow-up - baseline

Appendix E calls this "defined voxel-wise", but the reported values only make sense for the
aggregate form, so both are implemented and `mode` selects one:

  mode="aggregate" (default)  sum|D_gt - D_gen| / (0.5 * (sum|D_gt| + sum|D_gen|))
  mode="voxel"                mean over voxels of the per-voxel ratio

Both give 2 for a copy baseline and 0 for an exact prediction, and Table 7's exact 2.000000
is consistent with either. They differ on real data: per-voxel, almost every voxel carries a
small noise residual that a model cannot predict, so the voxel-wise mean saturates near 2 and
could not produce the paper's 0.436 on ADNI. The aggregate form can. Reporting uses the
aggregate value, with the voxel-wise one alongside it; see ../PAPER_SETTINGS.md (open item M2).

This is NOT the repo's DRMAE in eval_fm.py, which divides by |D_gt| alone. The two are not
comparable: the repo metric is unbounded, this one lies in [0, 2] with 2 for a copy baseline.

Region MAE (Sec. 4.1): MAE restricted to hippocampus, amygdala, lateral ventricles, CSF and
thalamus, following BrLP. The paper does not say whether the error is over voxel intensities or
over regional volumes, and a volume version would need the generated images to be segmented.
`region_mae` therefore computes the voxel-intensity MAE inside each structure of the baseline
segmentation, per structure and averaged; see ../PAPER_SETTINGS.md (open item M1).

FreeSurfer labels in our seg.nii.gz files, verified present in all 753 visits.
"""
from __future__ import annotations
import numpy as np

# FreeSurfer / SynthSeg label ids, left and right where applicable
REGIONS = {
    "hippocampus": (17, 53),
    "amygdala": (18, 54),
    "lateral_ventricle": (4, 43),
    "csf": (24,),
    "thalamus": (10, 49),
}
_E = 1e-8


def delta_rmae(pred, target, input, mask=None, mode="aggregate", eps=1e-6):
    """Paper Delta-RMAE in [0, 2]. mask: boolean brain mask; None uses an intensity threshold.
    mode: "aggregate" (ratio of L1 norms, used for reporting) or "voxel" (mean of per-voxel ratios)."""
    pred, target, input = (np.asarray(a, dtype=np.float64) for a in (pred, target, input))
    brain = (np.asarray(mask) > 0.5) if mask is not None else ((input > 0.05) | (target > 0.05))
    if brain.sum() == 0:
        brain = np.ones_like(input, dtype=bool)
    dgt = (target - input)[brain]
    dgn = (pred - input)[brain]
    if mode == "aggregate":
        num = np.abs(dgt - dgn).sum()
        den = 0.5 * (np.abs(dgt).sum() + np.abs(dgn).sum())
        return float(num / (den + _E))
    if mode != "voxel":
        raise ValueError("mode must be 'aggregate' or 'voxel'")
    den = 0.5 * (np.abs(dgt) + np.abs(dgn))
    # voxels where both residuals vanish carry no information; eps keeps them at 0 instead of 0/0
    return float(np.mean(np.abs(dgt - dgn) / (den + eps)))


def region_mae(pred, target, seg, regions=None):
    """Voxel-intensity MAE inside each structure. Returns {structure: mae, 'mean': mean-of-structures}.

    seg: label volume aligned with pred/target (the baseline visit's segmentation).
    Structures absent from seg are skipped and reported as nan.
    """
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    seg = np.asarray(seg)
    out, vals = {}, []
    for name, labels in (regions or REGIONS).items():
        m = np.isin(seg, labels)
        if m.sum() == 0:
            out[name] = float("nan")
            continue
        v = float(np.abs(pred[m] - target[m]).mean())
        out[name] = v
        vals.append(v)
    out["mean"] = float(np.mean(vals)) if vals else float("nan")
    return out


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    shape = (24, 24, 24)
    x0 = np.clip(rng.rand(*shape) * 0.3 + 0.3, 0, 1)
    d = np.zeros(shape); d[6:12, 6:12, 6:12] = -0.2       # true change
    x1 = x0 + d
    brain = np.ones(shape, dtype=bool)
    print("copy      (expect 2.0)      ", round(delta_rmae(x0, x1, x0, brain), 6))
    print("exact     (expect 0.0)      ", round(delta_rmae(x1, x1, x0, brain), 6))
    print("half      (expect 0.667)    ", round(delta_rmae(x0 + 0.5 * d, x1, x0, brain), 6))
    print("double    (expect 0.667)    ", round(delta_rmae(x0 + 2.0 * d, x1, x0, brain), 6))
    print("reversed  (expect 2.0)      ", round(delta_rmae(x0 - d, x1, x0, brain), 6))
    print("copy, voxel-wise (synthetic) ", round(delta_rmae(x0, x1, x0, brain, mode="voxel"), 6))
    seg = np.zeros(shape, dtype=np.int32); seg[6:12, 6:12, 6:12] = 17; seg[0:4, 0:4, 0:4] = 24
    print("region_mae (copy)           ", {k: round(v, 4) for k, v in region_mae(x0, x1, seg).items()})
