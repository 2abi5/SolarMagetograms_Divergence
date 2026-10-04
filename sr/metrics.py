"""
Per-region metrics (training_plan.md Phase 7 and section 2c). Inputs are
DE-normalised (3, H, W) stacks in Gauss, channel order Bp, Bt, Br, plus a
boolean (H, W) mask of valid pixels.

Pixel metrics per channel and on |B|: MAE, RMSE, Pearson r, PSNR, SSIM.
  PSNR = 20 log10(data_range / RMSE) with a FIXED per-channel data range
  (from the training split), SSIM on the same fixed affine map as the SSIM
  loss (11x11 Gaussian, sigma 1.5, C1 = 1e-4, C2 = 9e-3), averaged over
  windows that are fully valid.
Physics: divergence error mean|div_h(pred) - div_h(true)| (G/Mm), std of both
  divergence maps and their ratio, Jz error (mA/m^2), signed flux imbalance
  and unsigned flux relative error on Br, monopole-artifact count.
Degeneracy checks: std(pred)/std(true) per channel (use the strong-field mask).
Extremes: 1st/99th percentiles and the total-variation distance between
  soft histograms (fixed bin centres).
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, label

from sr import physics as ph

CHANNELS = ("Bp", "Bt", "Br")
SSIM_C1, SSIM_C2, SSIM_WIN, SSIM_SIGMA = 1e-4, 9e-3, 11, 1.5


def field_strength(stack):
    return np.sqrt((np.asarray(stack, float) ** 2).sum(axis=0))


def strong_mask(true, mask, threshold=200.0):
    return mask & (field_strength(true) > threshold)


def _valid(mask, *arrays):
    m = mask.copy()
    for a in arrays:
        m &= np.isfinite(a)
    return m


def ssim_masked(a, b, mask, lo, hi):
    """Mean SSIM over pixels whose whole 11x11 window is valid (fixed map to [0, 1])."""
    x = (np.where(mask, a, 0.0) - lo) / (hi - lo)
    y = (np.where(mask, b, 0.0) - lo) / (hi - lo)
    trunc = (SSIM_WIN - 1) / 2 / SSIM_SIGMA
    f = lambda z: gaussian_filter(z, SSIM_SIGMA, truncate=trunc, mode="constant")  # noqa: E731
    mx, my = f(x), f(y)
    vx, vy, cxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    s = ((2 * mx * my + SSIM_C1) * (2 * cxy + SSIM_C2)) / ((mx ** 2 + my ** 2 + SSIM_C1) * (vx + vy + SSIM_C2))
    r = SSIM_WIN // 2
    full = np.zeros_like(mask)
    full[r:-r, r:-r] = True
    ok = binary_erosion(mask, structure=np.ones((SSIM_WIN, SSIM_WIN)), border_value=0) & full
    return float(s[ok].mean()) if ok.any() else float("nan")


def _pixel(p, t, m, lo, hi):
    if not m.any():
        return {k: float("nan") for k in ("mae", "rmse", "pearson", "psnr", "ssim", "n")}
    d = p[m] - t[m]
    rmse = float(np.sqrt(np.mean(d ** 2)))
    pr = float(np.corrcoef(p[m], t[m])[0, 1]) if np.std(p[m]) > 0 and np.std(t[m]) > 0 else float("nan")
    psnr = float("inf") if rmse == 0 else float(20 * np.log10((hi - lo) / rmse))
    return {"mae": float(np.mean(np.abs(d))), "rmse": rmse, "pearson": pr, "psnr": psnr,
            "ssim": ssim_masked(p, t, m, lo, hi), "n": int(m.sum())}


def pixel_metrics(pred, true, mask, data_range):
    """data_range: (3,) fixed per-channel ranges in G (signed channels map
    [-r/2, r/2] -> [0, 1]); |B| uses [0, max(data_range)]."""
    out = {}
    for c, name in enumerate(CHANNELS):
        m = _valid(mask, pred[c], true[c])
        out[name] = _pixel(pred[c], true[c], m, -data_range[c] / 2, data_range[c] / 2)
    bp, bt = field_strength(pred), field_strength(true)
    out["B"] = _pixel(bp, bt, _valid(mask, bp, bt), 0.0, float(np.max(data_range)))
    return out


def variance_ratio(pred, true, mask):
    out = {}
    for c, name in enumerate(CHANNELS):
        m = _valid(mask, pred[c], true[c])
        st = np.std(true[c][m]) if m.any() else 0.0
        out[name] = float(np.std(pred[c][m]) / st) if st > 0 else float("nan")
    return out


def _div_maps(pred, true, mask, cdelt_deg):
    m = _valid(mask, *pred, *true)
    v = ph.interior_valid(m)
    dp = ph.div_g_per_mm(ph.div_h(np.nan_to_num(pred[0]), np.nan_to_num(pred[1])), cdelt_deg)
    dt = ph.div_g_per_mm(ph.div_h(np.nan_to_num(true[0]), np.nan_to_num(true[1])), cdelt_deg)
    return dp, dt, v


def physics_metrics(pred, true, mask, cdelt_deg=ph.CDELT_HR_DEG):
    dp, dt, v = _div_maps(pred, true, mask, cdelt_deg)
    jp = ph.jz_ma_per_m2(ph.jz(np.nan_to_num(pred[0]), np.nan_to_num(pred[1])), cdelt_deg)
    jt = ph.jz_ma_per_m2(ph.jz(np.nan_to_num(true[0]), np.nan_to_num(true[1])), cdelt_deg)
    if not v.any():
        return {k: float("nan") for k in ("div_err_g_per_mm", "div_std_pred_g_per_mm",
                                          "div_std_true_g_per_mm", "div_std_ratio", "jz_err_ma_per_m2")}
    s_true = float(np.std(dt[v]))
    return {"div_err_g_per_mm": float(np.mean(np.abs(dp - dt)[v])),
            "div_std_pred_g_per_mm": float(np.std(dp[v])),
            "div_std_true_g_per_mm": s_true,
            "div_std_ratio": float(np.std(dp[v]) / s_true) if s_true > 0 else float("nan"),
            "jz_err_ma_per_m2": float(np.mean(np.abs(jp - jt)[v]))}


def flux_metrics(pred, true, mask):
    m = _valid(mask, pred[2], true[2])
    bp, bt = pred[2][m], true[2][m]
    u_t = np.abs(bt).sum()
    if u_t == 0:
        return {"signed_flux_imbalance": float("nan"), "unsigned_flux_rel_err": float("nan")}
    return {"signed_flux_imbalance": float(abs(bp.sum() - bt.sum()) / u_t),
            "unsigned_flux_rel_err": float((np.abs(bp).sum() - u_t) / u_t)}


def monopole_artifacts(pred, true, mask, threshold_g_per_mm, cdelt_deg=ph.CDELT_HR_DEG):
    """Connected clusters of |div_h(pred) - div_h(true)| > threshold. The
    thresholded map is dilated by one pixel before labelling (8-connectivity),
    so the +/- lobes a single spurious source leaves in a central difference
    count as one cluster."""
    dp, dt, v = _div_maps(pred, true, mask, cdelt_deg)
    hot = (np.abs(dp - dt) > threshold_g_per_mm) & v
    if not hot.any():
        return 0
    _, n = label(binary_dilation(hot, structure=np.ones((3, 3))), structure=np.ones((3, 3)))
    return int(n)


def _soft_hist(v, centers):
    c = np.asarray(centers, float)
    left = np.concatenate([[-np.inf], c[:-1]])
    right = np.concatenate([c[1:], [np.inf]])
    v = v[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        up = np.where(np.isinf(left), np.inf, (v - left) / (c - left))
        down = np.where(np.isinf(right), np.inf, (right - v) / (right - c))
    w = np.clip(np.minimum(up, down), 0, 1)
    return w.sum(0) / max(len(v), 1)


def extreme_metrics(pred, true, mask, centers_gauss):
    out = {}
    for c, name in enumerate(CHANNELS):
        m = _valid(mask, pred[c], true[c])
        if not m.any():          # e.g. bicubic baselines, which have no Bp/Bt
            out[name] = {k: float("nan") for k in ("p1_pred", "p1_true", "p99_pred", "p99_true", "hist_tv")}
            continue
        p, t = pred[c][m], true[c][m]
        out[name] = {"p1_pred": float(np.percentile(p, 1)), "p1_true": float(np.percentile(t, 1)),
                     "p99_pred": float(np.percentile(p, 99)), "p99_true": float(np.percentile(t, 99)),
                     "hist_tv": float(0.5 * np.abs(_soft_hist(p, centers_gauss) - _soft_hist(t, centers_gauss)).sum())}
    return out


def radial_power_spectrum(img, mask=None):
    """Azimuthally averaged power spectrum (Hann-windowed, masked/NaN pixels
    set to the mean). Returns integer wavenumbers k >= 1 and P(k)."""
    a = np.asarray(img, float).copy()
    valid = np.isfinite(a) if mask is None else (mask & np.isfinite(a))
    a[~valid] = a[valid].mean()
    a -= a.mean()
    h, w = a.shape
    a *= np.outer(np.hanning(h), np.hanning(w))
    p = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    yy, xx = np.indices(p.shape)
    r = np.hypot((yy - h // 2) * (w / h), xx - w // 2).astype(int)
    kmax = min(h, w) // 2
    k = np.arange(1, kmax)
    pk = np.array([p[r == i].mean() for i in k])
    return k, pk


def bootstrap_ci(df, col, unit="harpnum", n_boot=10000, seed=0, alpha=0.05):
    """Mean over units (per-unit means first) and a percentile bootstrap CI
    resampling units (HARPs), never individual frames or patches."""
    per = df.groupby(unit)[col].mean().dropna().to_numpy()
    if per.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = per[rng.integers(0, per.size, size=(n_boot, per.size))].mean(axis=1)
    return float(per.mean()), float(np.percentile(boots, 100 * alpha / 2)), \
        float(np.percentile(boots, 100 * (1 - alpha / 2)))
