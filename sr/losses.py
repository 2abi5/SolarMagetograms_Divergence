"""
Masked losses (training_plan.md sections 1 and 2d; decisions in reports/phase_0.md section 9).

Every loss takes (pred, target, valid_mask):
    pred, target  (B, 3, H, W) normalised (/3500) SHARP stacks, NaNs already set to 0
    valid_mask    (B, 1, H, W) bool, True where all three target channels were finite
and ignores invalid pixels.

  mse        mean over valid pixels and all 3 channels
  grad       Sobel gradients (raw 3x3 kernels, [-1 0 1; -2 0 2; -1 0 1]), squared
             difference of both components, averaged over pixels whose 3x3
             window is fully valid (Munoz-Jaramillo et al. 2024: L2 on Sobel)
  ssim       1 - SSIM per channel on a FIXED affine map to [0, 1] taken from the
             training split (never per image), 11x11 Gaussian window (sigma 1.5),
             C1 = 1e-4, C2 = 9e-3 as in the paper; only fully valid windows count
  hist       triangular soft histogram (Wang et al. 2016/2018 learnable-histogram
             kernel with FIXED centres), per channel, mean over channels.
             mode="counts": counts rescaled to the paper's 64 x 128^2 pixel budget,
             loss = sum_k |h_pred - h_true| (decision D4).
             mode="tv": counts normalised to distributions, loss = total variation
             distance 0.5 * sum_k |p_pred - p_true| in [0, 1].
             Applying it to 3 channels deviates from the published single-channel method.
  div        L_div on normalised data: "match" = mean[(div_h(pred) - div_h(true))^2],
             "zero" = mean[div_h(pred)^2] (ablation only); a pixel counts only if it
             and its 4 neighbours are valid.
             "match_ms" (amendment, reports/phase_6.md section 8) = mean over scales s of
             mean[(div_h(G_s*pred) - div_h(G_s*true))^2] / norm_s, with G_s a NaN-aware
             Gaussian of sigma s HR px (same as scripts/div_snr_by_scale.py) and norm_s the
             training-split mean square of div_h(G_s*true); pixel-scale div_h is
             noise-dominated in SHARP, div_h at >= ~0.7 Mm is not.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sr import physics

LOSS_TERMS = ("mse", "grad", "hist", "ssim", "div")


def masked_mean(x, mask):
    m = mask.to(x.dtype).expand_as(x)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def _window_valid(mask, k):
    """(B, 1, H-k+1, W-k+1): True where the whole k x k window is valid."""
    ones = torch.ones(1, 1, k, k, dtype=torch.float32, device=mask.device)
    return F.conv2d(mask.float(), ones) > k * k - 0.5


# --------------------------------------------------------------------------
# MSE, gradient, divergence
# --------------------------------------------------------------------------

def mse_loss(pred, target, mask):
    return masked_mean((pred - target) ** 2, mask)


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def sobel(x):
    """Raw Sobel gradients per channel, 'valid' region: (B, C, H-2, W-2) each."""
    c = x.shape[1]
    kx = _SOBEL_X.to(x)[None, None].repeat(c, 1, 1, 1)
    ky = _SOBEL_X.t().to(x)[None, None].repeat(c, 1, 1, 1)
    return F.conv2d(x, kx, groups=c), F.conv2d(x, ky, groups=c)


def gradient_loss(pred, target, mask):
    gx, gy = sobel(pred - target)          # Sobel is linear
    return masked_mean(gx ** 2 + gy ** 2, _window_valid(mask, 3))


def _gauss1d(sigma, dtype, device, truncate=4.0):
    r = int(truncate * sigma + 0.5)            # scipy.ndimage.gaussian_filter's kernel
    x = torch.arange(-r, r + 1, dtype=dtype, device=device)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum(), r


def _blur(x, sigma):
    """Separable Gaussian of (B, H, W) with zero padding (scipy mode='constant')."""
    g, r = _gauss1d(sigma, x.dtype, x.device)
    y = F.conv2d(x[:, None], g.view(1, 1, -1, 1), padding=(r, 0))
    y = F.conv2d(y, g.view(1, 1, 1, -1), padding=(0, r))
    return y[:, 0]


def smoothed_div(bp, bt, valid, sigma, min_weight=0.95):
    """div_h of NaN-aware Gaussian-smoothed Bp, Bt ((B, H, W) each, valid (B, H, W) bool),
    per pixel in normalised units, and the interior validity mask: the pixel and its
    4 neighbours valid with at least `min_weight` of the Gaussian weight on valid pixels."""
    if sigma > 0:
        m = valid.to(bp.dtype)
        w = _blur(m, sigma)
        bp = _blur(bp * m, sigma) / w.clamp_min(1e-12)
        bt = _blur(bt * m, sigma) / w.clamp_min(1e-12)
        valid = valid & (w >= min_weight)
    zero = torch.zeros((), dtype=bp.dtype, device=bp.device)
    bp, bt = torch.where(valid, bp, zero), torch.where(valid, bt, zero)
    return physics.div_h(bp, bt), physics.interior_valid(valid)


def divergence_loss(pred, target, mask, mode="match", scales=None, scale_norm=None):
    if mode == "match_ms":
        if not scales or scale_norm is None or len(scales) != len(scale_norm):
            raise ValueError("div mode 'match_ms' needs scales and one scale_norm per scale")
        valid = mask[:, 0]
        total = pred.new_zeros(())
        for s, norm in zip(scales, scale_norm):
            dp, v = smoothed_div(pred[:, 0], pred[:, 1], valid, float(s))
            dt, _ = smoothed_div(target[:, 0], target[:, 1], valid, float(s))
            total = total + masked_mean((dp - dt) ** 2, v) / float(norm)
        return total / len(scales)
    div_p = physics.div_h(pred[:, 0], pred[:, 1])
    valid = physics.interior_valid(mask[:, 0])
    if mode == "match":
        div_t = physics.div_h(target[:, 0], target[:, 1])
        return masked_mean((div_p - div_t) ** 2, valid)
    if mode == "zero":
        return masked_mean(div_p ** 2, valid)
    raise ValueError(f"div mode must be 'match', 'zero' or 'match_ms', got {mode!r}")


# --------------------------------------------------------------------------
# SSIM
# --------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    def __init__(self, lo, hi, win=11, sigma=1.5, C1=1e-4, C2=9e-3):
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer("scale", (torch.as_tensor(hi, dtype=torch.float32)
                                       - torch.as_tensor(lo, dtype=torch.float32)).view(-1, 1, 1))
        g = torch.exp(-((torch.arange(win, dtype=torch.float32) - (win - 1) / 2) ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        self.register_buffer("window", torch.outer(g, g)[None, None])
        self.win, self.C1, self.C2 = win, C1, C2

    def _filter(self, x):
        c = x.shape[1]
        return F.conv2d(x, self.window.repeat(c, 1, 1, 1), groups=c)

    def ssim_map(self, pred, target):
        a = (pred - self.lo) / self.scale
        b = (target - self.lo) / self.scale
        mu_a, mu_b = self._filter(a), self._filter(b)
        var_a = self._filter(a * a) - mu_a ** 2
        var_b = self._filter(b * b) - mu_b ** 2
        cov = self._filter(a * b) - mu_a * mu_b
        return ((2 * mu_a * mu_b + self.C1) * (2 * cov + self.C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + self.C1) * (var_a + var_b + self.C2))

    def forward(self, pred, target, mask):
        return 1.0 - masked_mean(self.ssim_map(pred, target), _window_valid(mask, self.win))


# --------------------------------------------------------------------------
# Soft histogram
# --------------------------------------------------------------------------

def log_bin_centers(noise, top, dl=0.15):
    """Symmetric centres: 0 and +-noise * 10**(dl*k) up to >= top (normalised
    units). Mirrors the converter config (noise_level 60 G, bin step dl=0.15
    dex), with the upper end taken from the training split."""
    n = int(math.ceil((math.log10(top) - math.log10(noise)) / dl - 1e-9)) + 1
    pos = noise * 10.0 ** (dl * torch.arange(n, dtype=torch.float64))
    return torch.cat([-pos.flip(0), torch.zeros(1, dtype=torch.float64), pos]).float()


def hat_weights(v, centers):
    """(N, K) triangular-kernel weights of values v on sorted centres. Each row
    sums to 1 (values beyond the outermost centres go to the edge bin)."""
    c = centers.to(v)
    inf = torch.full((1,), float("inf"), dtype=v.dtype, device=v.device)
    left = torch.cat([-inf, c[:-1]])
    right = torch.cat([c[1:], inf])
    v = v[:, None]
    up = torch.where(torch.isinf(left), torch.full_like(v, float("inf")).expand(-1, len(c)),
                     (v - left) / (c - left))
    down = torch.where(torch.isinf(right), torch.full_like(v, float("inf")).expand(-1, len(c)),
                       (right - v) / (right - c))
    return torch.clamp(torch.minimum(up, down), 0.0, 1.0)


class HistogramLoss(nn.Module):
    def __init__(self, centers, count_budget=64 * 128 * 128, mode="counts"):
        super().__init__()
        if mode not in ("counts", "tv"):
            raise ValueError(f"histogram mode must be 'counts' or 'tv', got {mode!r}")
        self.mode = mode
        for i, c in enumerate(centers):
            self.register_buffer(f"centers_{i}", torch.as_tensor(c, dtype=torch.float32))
        self.n_channels = len(centers)
        self.count_budget = float(count_budget)

    def _centers(self, i):
        return getattr(self, f"centers_{i}")

    def _counts(self, x, mask):
        valid = mask[:, 0].reshape(-1)
        n = valid.sum().clamp_min(1)
        out = []
        for i in range(self.n_channels):
            v = x[:, i].reshape(-1)[valid]
            scale = 1.0 / n if self.mode == "tv" else self.count_budget / n
            out.append(hat_weights(v, self._centers(i)).sum(0) * scale)
        return out

    def soft_counts(self, x, mask):
        counts = self._counts(x, mask)
        if len({c.numel() for c in counts}) == 1:
            return torch.stack(counts)
        return counts

    def forward(self, pred, target, mask):
        cp, ct = self._counts(pred, mask), self._counts(target, mask)
        factor = 0.5 if self.mode == "tv" else 1.0
        return torch.stack([(a - b).abs().sum() for a, b in zip(cp, ct)]).mean() * factor


# --------------------------------------------------------------------------
# Weighted sum, every term logged
# --------------------------------------------------------------------------

class CompoundLoss(nn.Module):
    """total = sum_k w_k * term_k. Every term in LOSS_TERMS is computed and
    returned for logging, including zero-weight ones (those without grad)."""

    def __init__(self, weights, div_mode="match", ssim=None, hist=None, div_scales=None, div_scale_norm=None):
        super().__init__()
        self.weights = {k: float(weights.get(k, 0.0)) for k in LOSS_TERMS}
        self.div_mode = div_mode
        self.div_scales, self.div_scale_norm = div_scales, div_scale_norm
        self.ssim = ssim
        self.hist = hist

    def _term(self, name, pred, target, mask):
        if name == "mse":
            return mse_loss(pred, target, mask)
        if name == "grad":
            return gradient_loss(pred, target, mask)
        if name == "div":
            return divergence_loss(pred, target, mask, self.div_mode, self.div_scales, self.div_scale_norm)
        module = self.ssim if name == "ssim" else self.hist
        if module is None:
            return torch.tensor(float("nan"), device=pred.device)
        return module(pred, target, mask)

    def forward(self, pred, target, mask):
        total = pred.new_zeros(())
        terms = {}
        for name in LOSS_TERMS:
            w = self.weights[name]
            if w != 0.0:
                val = self._term(name, pred, target, mask)
                total = total + w * val
            else:
                with torch.no_grad():
                    val = self._term(name, pred, target, mask)
            terms[name] = float(val.detach())
        terms["total"] = float(total.detach())
        return total, terms
