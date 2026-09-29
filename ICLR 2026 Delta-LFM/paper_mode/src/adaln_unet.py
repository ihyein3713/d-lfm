"""AdaLN conditioning for the flow-matching UNet, as Appendix B of the paper specifies.

The paper injects the control signals (start time, current sample time, end query time and the
patient attributes) three possible ways -- additive bias, cross-attention, AdaLN -- and reports
AdaLN as the best with [0,T] sampling. The repo implements the additive variant only:
`emb = emb + cond_mlp(cond_vec)` in ../../src/model3D/unet.py (line 1806).

This module adds the AdaLN variant without touching the repo UNet. `AdaLNUNet` wraps the repo
model, and the condition vector no longer reaches the time embedding; instead

    h_i <- h_i * (1 + gamma_i) + beta_i        for the output of every decoder level i

with `[gamma_i, beta_i] = head_i(shared(cond_vec))`, i.e. the paper's "initial one-layer MLP that
processes the condition signals, followed by additional MLPs at each U-Net decoder layer to
generate the corresponding AdaLN modulation parameters". The heads are zero-initialised, so
training starts from the unmodulated network. Modulation is applied through forward hooks on the
decoder blocks, so the repo forward pass is untouched.

Selected with --pm_cond_mode adaln, which points the `flowmatching` config entry here.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _out_channels(block: nn.Module) -> int:
    """Channels a decoder block emits: the out_channels of its last 3D convolution."""
    convs = [m for m in block.modules() if isinstance(m, nn.Conv3d)]
    if not convs:
        raise ValueError("decoder block without a Conv3d; cannot size the AdaLN head")
    return int(convs[-1].out_channels)


class AdaLNUNet(nn.Module):
    """Wraps a repo DiffusionModelUNet and conditions it with AdaLN instead of an additive bias."""

    def __init__(self, unet: nn.Module, cond_dim: int, hidden: int | None = None):
        super().__init__()
        self.unet = unet
        self.cond_dim = int(cond_dim)
        blocks = list(unet.up_blocks)
        chans = [_out_channels(b) for b in blocks]
        h = int(hidden or max(chans))
        # "an initial one-layer MLP that processes the condition signals"
        self.cond_shared = nn.Sequential(nn.Linear(self.cond_dim, h), nn.SiLU())
        # "additional MLPs at each U-Net decoder layer" -> [gamma, beta] per level
        self.cond_heads = nn.ModuleList([nn.Linear(h, 2 * c) for c in chans])
        for lin in self.cond_heads:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._mod: list[torch.Tensor] | None = None
        for i, b in enumerate(blocks):
            b.register_forward_hook(self._make_hook(i))
        print("[paper_mode] AdaLN conditioning on %d decoder levels, channels %s"
              % (len(chans), chans), flush=True)

    def _make_hook(self, i: int):
        def hook(_module, _inputs, output):
            if self._mod is None:
                return output
            out = output[0] if isinstance(output, tuple) else output
            gb = self._mod[i].to(dtype=out.dtype)
            gamma, beta = gb.chunk(2, dim=1)
            shape = (out.shape[0], -1) + (1,) * (out.dim() - 2)
            out = out * (1.0 + gamma.reshape(shape)) + beta.reshape(shape)
            return (out,) + output[1:] if isinstance(output, tuple) else out
        return hook

    def forward(self, x, timesteps=None, context=None, class_labels=None, cond_vec=None, **kw):
        if cond_vec is None:
            self._mod = None
        else:
            s = self.cond_shared(cond_vec.to(dtype=x.dtype))
            self._mod = [head(s) for head in self.cond_heads]
        # cond_vec is deliberately NOT forwarded: in the AdaLN variant the condition must not also
        # enter the time embedding, otherwise the two mechanisms are mixed
        try:
            return self.unet(x=x, timesteps=timesteps, context=context,
                             class_labels=class_labels, **kw)
        finally:
            self._mod = None


def init_large_latent_diffusion_adaln(args=None, in_channels=4, use_image=True, out_channels=None,
                                      num_class_embeds=15, cond_dim=0, spade_dim=0) -> nn.Module:
    """Drop-in replacement for src.model3D.networks.init_large_latent_diffusion, AdaLN variant.

    The inner UNet is built with cond_dim=0 so it has no additive conditioning path at all.
    """
    from src.model3D.networks import init_large_latent_diffusion
    unet = init_large_latent_diffusion(args=args, in_channels=in_channels, use_image=use_image,
                                       out_channels=out_channels,
                                       num_class_embeds=num_class_embeds,
                                       cond_dim=0, spade_dim=spade_dim)
    if not cond_dim:
        print("[paper_mode] cond_dim=0: AdaLN has nothing to condition on, returning the plain UNet",
              flush=True)
        return unet
    return AdaLNUNet(unet, cond_dim=cond_dim)


if __name__ == "__main__":
    # Self-test. NOTE: the repo UNet's output convolution is zero-initialised
    # (zero_module in ../../src/model3D/unet.py line 1744), so an untrained network returns exactly
    # zeros and comparing final outputs proves nothing. The test therefore checks the DECODER
    # FEATURES, which is where AdaLN acts.
    import sys as _s
    _s.path.insert(0, ".")
    _s.path.insert(0, "paper_mode/src")
    seen = {}

    def _probe(model):
        for i, b in enumerate(model.unet.up_blocks):
            def mk(i):
                def h(_m, _in, out):
                    o = out[0] if isinstance(out, tuple) else out
                    seen[i] = float(o.abs().mean())
                return h
            b.register_forward_hook(mk(i))

    m = init_large_latent_diffusion_adaln(in_channels=8, use_image=False, out_channels=4,
                                          num_class_embeds=None, cond_dim=10).cuda().eval()
    _probe(m)
    x = torch.randn(2, 8, 8, 8, 8).cuda()
    t = torch.zeros(2).cuda()
    ctx = torch.randn(2, 1, 10).cuda()
    cv = torch.randn(2, 10).cuda()
    with torch.no_grad():
        m(x=x, timesteps=t, context=ctx, class_labels=None, cond_vec=cv)
    base = dict(seen)
    with torch.no_grad():
        for lin in m.cond_heads:
            lin.weight.normal_(std=0.1)
            lin.bias.normal_(std=0.1)
        m(x=x, timesteps=t, context=ctx, class_labels=None, cond_vec=cv)
    print("decoder feature magnitude, zero-init heads vs perturbed heads:")
    for i in sorted(base):
        print("  level %d: %.5f -> %.5f  %s" % (i, base[i], seen[i],
                                                "changed" if abs(seen[i] - base[i]) > 1e-6 else "UNCHANGED"))
    with torch.no_grad():
        for lin in m.cond_heads:
            lin.weight.zero_(); lin.bias.zero_()
        m(x=x, timesteps=t, context=ctx, class_labels=None, cond_vec=cv)
        restored = dict(seen)
        m(x=x, timesteps=t, context=ctx, class_labels=None, cond_vec=None)
    print("zero heads reproduce the unconditioned features:",
          all(abs(restored[i] - seen[i]) < 1e-6 for i in seen))
