"""ArcRank loss exactly as written in the paper (Sec. 3.2 of arXiv 2512.09185v4).

The repo ships a different variant in ../../src/trajectory_losses.py (rank-1 ratio, PC1 order
hinge, step cosine). This module follows the paper's equations instead:

    SVD(z) = U S V^T          angle(z) = U ,  ||z|| = S                     (eq 6, 7)
    L_Arc  = sum_{i<j, same patient} |U_i - U_j|                            (eq 8, left)
    L_Rank = sum_{i<j, same patient} max(0, m - (S_j - S_i)),  t_i < t_j    (eq 8, right)
    L_Pull = |S_j - S_i|       (temporally adjacent pairs, curbs runaway separation)  (eq 10)
    L_Rank_tilde = L_Rank + L_Pull, sharing the weight of L_Rank            (eq 11)
    L_ArcRank = lam_arc * L_Arc + lam_rank * L_Rank_tilde                   (eq 9)
    stop-gradient sg(.) on the earlier element i of every pair              (Sec. 3.2, last line)

Choices the paper leaves open, and why:

1. SVD of what matrix. Eq 6-7 take the SVD of a single latent z, so the decomposition is
   per timepoint. A 3D latent [C, H, W, D] is read as the matrix [C, H*W*D], giving U in
   R^{CxC} (orientation) and S in R^C (magnitudes). This is the literal reading and is cheap,
   since C is the latent channel count (4 here).
2. Magnitude as a scalar. ||z|| in eq 5 is a scalar, while S is a vector. ||S||_2 is used,
   which equals the Frobenius norm of z, so "magnitude" keeps its usual meaning.
3. Scale of m and of L_Pull. Both act on raw latent magnitudes, whose scale depends on the
   autoencoder, so a fixed m is not transferable. The magnitude terms are divided by the
   patient's mean magnitude (detached), which makes them dimensionless; `margin` is then the
   required growth per pair as a fraction of that mean. The paper does not state m.
4. Sign of U. Singular vectors are sign-ambiguous, so |U_i - U_j| is not well defined until
   the columns are aligned. Each column of U_j is flipped to agree with U_i before the
   difference is taken.

Every deviation above is recorded in ../PAPER_SETTINGS.md.
"""
from __future__ import annotations
import torch


def _by_patient(z: torch.Tensor, order: torch.Tensor, n_visits: int):
    """z:[n_visits*B, C, ...] stacked visit-major; order:[n_visits*B] time key.

    Returns Z:[B, n, C, S] and the per-patient visit ranks R:[B, n], both fp32.
    """
    N = z.shape[0]
    B = N // n_visits
    C = z.shape[1]
    Z = z.reshape(n_visits, B, C, -1).permute(1, 0, 2, 3).contiguous().float()
    R = order.reshape(n_visits, B).permute(1, 0).contiguous().float()
    return Z, R, B


def arcrank_loss(z: torch.Tensor,
                 order: torch.Tensor,
                 n_visits: int = 3,
                 lam_arc: float = 1.0,
                 lam_rank: float = 1.0,
                 margin: float = 0.05,
                 eps: float = 1e-8):
    """Paper ArcRank loss. Returns (total, parts) with parts unweighted, for logging."""
    Z, R, B = _by_patient(z, order, n_visits)          # [B,n,C,S], [B,n]
    n = Z.shape[1]
    dev = Z.device
    zero = torch.zeros((), device=dev)
    if n < 2:
        return zero, {"arc": 0.0, "rank": 0.0, "pull": 0.0, "mag_gap": 0.0}

    # time-order the visits (only the rank of `order` is used)
    idx = R.argsort(dim=1)                             # [B,n]
    Z = torch.gather(Z, 1, idx[:, :, None, None].expand_as(Z))

    # fp32 SVD per timepoint; bf16 SVD is numerically unstable
    U, S, _ = torch.linalg.svd(Z, full_matrices=False)  # U:[B,n,C,C]  S:[B,n,C]
    mag = S.norm(dim=-1)                               # [B,n] Frobenius norm of the latent
    scale = mag.mean(dim=1, keepdim=True).detach().clamp_min(eps)   # per-patient, detached

    arc_terms, rank_terms, pull_terms, gaps = [], [], [], []
    for i in range(n - 1):
        for j in range(i + 1, n):
            Ui = U[:, i].detach()                      # sg(.) on the earlier element
            Uj = U[:, j]
            # align column signs before differencing: singular vectors carry an arbitrary sign
            sgn = torch.sign((Ui * Uj).sum(dim=1, keepdim=True))     # [B,1,C]
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            arc_terms.append((Ui - Uj * sgn).abs().mean(dim=(1, 2)))  # [B]

            d = (mag[:, j] - mag[:, i].detach()) / scale[:, 0]        # [B] relative growth
            rank_terms.append(torch.relu(margin - d))
            gaps.append(d.detach())
            if j == i + 1:                             # L_Pull on temporally adjacent pairs
                pull_terms.append(d.abs())

    arc = torch.stack(arc_terms, 0).mean()
    rank = torch.stack(rank_terms, 0).mean()
    pull = torch.stack(pull_terms, 0).mean()
    total = lam_arc * arc + lam_rank * (rank + pull)
    parts = {"arc": float(arc.detach()), "rank": float(rank.detach()),
             "pull": float(pull.detach()), "mag_gap": float(torch.stack(gaps, 0).mean())}
    return total, parts
