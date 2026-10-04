"""
Training-split statistics and divergence noise floor (training_plan.md 2b, Phase 4).

All inputs are normalised (/3500) (3, H, W) SHARP stacks (Bp, Bt, Br) with NaN
for missing pixels. Outputs are in Gauss or G/Mm.

  channel_stats     mean, std, 1st/99th percentile, 99.9th percentile of |B|
  ssim_ranges       fixed per-channel affine range [-a_c, a_c] for the SSIM
                    loss, a_c = 99.9th percentile of |B_c| (normalised units)
  spatial_div_noise mean |div_h(raw) - div_h(Gaussian-smoothed)|: the high-
                    frequency, noise-dominated part of the divergence
  temporal_div_noise mean |div_h(t) - div_h(t + 12 min)| over pixels whose
                    smoothed field barely changed between the two frames
"""

import math

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

from sr import physics as ph

CHANNELS = ("Bp", "Bt", "Br")


def channel_stats(hr_list, norm=3500.0):
    out = {}
    for c, name in enumerate(CHANNELS):
        v = np.concatenate([np.asarray(h[c], np.float64).ravel() for h in hr_list]) * norm
        v = v[np.isfinite(v)]
        out[name] = {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
                     "p1": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99)),
                     "abs_p999": float(np.percentile(np.abs(v), 99.9))}
    return out


def ssim_ranges(hr_list, q=99.9):
    a = np.array([np.nanpercentile(np.abs(np.concatenate([np.asarray(h[c]).ravel() for h in hr_list])), q)
                  for c in range(3)])
    return -a, a


def _nan_smooth(x, sigma):
    m = np.isfinite(x)
    num = gaussian_filter(np.where(m, x, 0.0), sigma)
    den = gaussian_filter(m.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _footprint_valid(valid, sigma):
    """Pixels whose whole Gaussian footprint (radius 4 sigma) is valid and
    inside the image."""
    r = int(4.0 * sigma + 0.5)
    v = np.pad(valid, 1, constant_values=False)
    v = binary_erosion(v, structure=np.ones((2 * r + 1, 2 * r + 1)), border_value=0)[1:-1, 1:-1]
    return v


def spatial_div_noise(hr, sigma=1.0, norm=3500.0, cdelt_deg=ph.CDELT_HR_DEG):
    hr = np.asarray(hr, np.float64)
    valid = np.isfinite(hr).all(axis=0)
    smooth = np.stack([_nan_smooth(hr[c], sigma) for c in range(3)])
    d_raw = ph.div_h(hr[0], hr[1])
    d_smooth = ph.div_h(smooth[0], smooth[1])
    sel = ph.interior_valid(valid) & ph.interior_valid(_footprint_valid(valid, sigma))
    if not sel.any():
        return float("nan")
    return float(np.mean(np.abs(d_raw - d_smooth)[sel]) * norm / ph.pixel_scale_mm(cdelt_deg))


def temporal_div_noise(hr_a, hr_b, norm=3500.0, max_change_gauss=50.0, smooth_sigma=2.0,
                       cdelt_deg=ph.CDELT_HR_DEG):
    """Divergence difference between two frames of one region on the same grid.
    A pixel is used only if the Gaussian-smoothed change of every component is
    below max_change_gauss (so real evolution is excluded but pixel noise is
    not) and the pixel and its 4 neighbours are valid in both frames."""
    a, b = np.asarray(hr_a, np.float64), np.asarray(hr_b, np.float64)
    valid = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
    change = np.max([np.abs(_nan_smooth(b[c] - a[c], smooth_sigma)) for c in range(3)], axis=0) * norm
    quiet = valid & (np.nan_to_num(change, nan=np.inf) < max_change_gauss)
    sel = ph.interior_valid(quiet)
    diff = (ph.div_h(a[0], a[1]) - ph.div_h(b[0], b[1])) * norm / ph.pixel_scale_mm(cdelt_deg)
    d = diff[sel]
    return {"mean_abs_g_per_mm": float(np.mean(np.abs(d))) if d.size else float("nan"),
            "rms_g_per_mm": float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan"),
            "n_pixels": int(d.size)}
