"""
SHARP active-region keywords (Bobra et al. 2014, Table 3) from a (3, H, W) Gauss stack
(Bp, Bt, Br) with NaN for missing pixels. Phase 9 (reports/phase_9.md).

Conventions (sr/physics.py): Bx = Bp (westward), By = -Bt (Bt is positive southward),
Bz = Br; rows increase with y, columns with x; derivatives are central differences on
interior pixels, so derivative quantities use pixels whose four neighbours are valid.
The CEA pixel is CDELT = 0.03 deg = 0.364 Mm.

    USFLUX   sum |Bz| dA                                  Mx
    MEANGAM  mean arctan(Bh / |Bz|)                       deg
    MEANGBT  mean |grad |B||                              G/Mm
    MEANGBZ  mean |grad Bz|                               G/Mm
    MEANGBH  mean |grad Bh|                               G/Mm
    MEANJZD  mean Jz                                      mA/m^2
    TOTUSJZ  sum |Jz| dA                                  A
    MEANALP  sum (curl B)_z Bz / sum Bz^2                 1/Mm
    MEANJZH  mean Bz (curl B)_z                           G^2/m
    TOTUSJH  sum |Bz (curl B)_z|                          G^2/m
    ABSNJZH  |sum Bz (curl B)_z|                          G^2/m
    SAVNCPP  |sum_{Bz>0} Jz dA| + |sum_{Bz<0} Jz dA|      A
    R_VALUE  log10 of Schrijver's (2007) R, see r_value() log10(G)

`mask` restricts every keyword except R_VALUE to the given pixels; R_VALUE needs the weak
field around polarity inversion lines, so it always uses all finite pixels. The SHARP
pipeline restricts its keywords to the active pixels of the bitmap segment with
high-confidence disambiguation; those segments are not stored in the region files, so
values differ from the header keywords through the mask (reports/phase_9.md, section 2).
ERR_KEY maps each keyword to the pipeline's error keyword in the stored header.
"""
import math

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from sr import physics as ph

KEYWORDS = ("USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZD", "TOTUSJZ",
            "MEANALP", "MEANJZH", "TOTUSJH", "ABSNJZH", "SAVNCPP", "R_VALUE")
ERR_KEY = {"USFLUX": "errvf", "MEANGAM": "errgam", "MEANGBT": "errbt", "MEANGBZ": "errbz",
           "MEANGBH": "errbh", "MEANJZD": "errjz", "TOTUSJZ": "errusi", "MEANALP": "erralp",
           "MEANJZH": "errmih", "TOTUSJH": "errtui", "ABSNJZH": "errtai", "SAVNCPP": "errjht"}

# R_VALUE constants (Schrijver 2007 as applied by the SHARP pipeline); fixed in reports/phase_9.md
R_THRESHOLD_G = 150.0   # polarity bitmaps: Bz > +150 G and Bz < -150 G
R_BIN = 4               # block average to MDI-like resolution before thresholding
R_FWHM_MM = 15.0        # Gaussian weight around the polarity inversion line


def _block_mean(a, b, min_valid):
    """b x b block means of `a` (NaN-aware); blocks with fewer than min_valid of their pixels finite are NaN."""
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    x = a[:h, :w].reshape(h // b, b, w // b, b)
    fin = np.isfinite(x)
    n = fin.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(fin, x, 0.0).sum(axis=(1, 3)) / n
    m[n < min_valid * b * b] = np.nan
    return m


def r_value(bz_gauss, cdelt_deg=ph.CDELT_HR_DEG, threshold_g=R_THRESHOLD_G, fwhm_mm=R_FWHM_MM,
            bin_px=R_BIN, min_valid=0.5):
    """log10 of R: the unsigned field near strong-gradient polarity inversion lines (Schrijver 2007).
    Bz is block-averaged bin_px x bin_px, the bitmaps Bz > +threshold and Bz < -threshold are each
    dilated by one binned pixel, their overlap is the inversion-line map, which is convolved with a
    Gaussian of the given FWHM; R = sum |Bz_binned| x that weight (G). Returns 0 when R = 0."""
    b = np.nan_to_num(_block_mean(np.asarray(bz_gauss, np.float64), bin_px, min_valid), nan=0.0)
    pos, neg = b > threshold_g, b < -threshold_g
    if not (pos.any() and neg.any()):
        return 0.0
    s = np.ones((3, 3), bool)
    pil = binary_dilation(pos, s) & binary_dilation(neg, s)
    if not pil.any():
        return 0.0
    sigma = fwhm_mm / (bin_px * ph.pixel_scale_mm(cdelt_deg)) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    w = gaussian_filter(pil.astype(float), sigma, mode="constant")
    r = float((np.abs(b) * w).sum())
    return float(np.log10(r)) if r > 0 else 0.0


def keywords(hr_gauss, mask=None, cdelt_deg=ph.CDELT_HR_DEG):
    """Dict of the KEYWORDS (plus n_pixels, n_interior) for one (3, H, W) Gauss stack (Bp, Bt, Br)."""
    hr = np.asarray(hr_gauss, np.float64)
    finite = np.isfinite(hr).all(axis=0)
    m = finite if mask is None else finite & np.asarray(mask, bool)
    pix_mm = ph.pixel_scale_mm(cdelt_deg)
    area_cm2, area_m2 = (pix_mm * 1e8) ** 2, (pix_mm * 1e6) ** 2
    out = {k: float("nan") for k in KEYWORDS}
    out["n_pixels"], out["n_interior"] = int(m.sum()), 0
    out["R_VALUE"] = r_value(np.where(finite, hr[2], np.nan), cdelt_deg)
    if not m.any():
        return out
    bx, by, bz = (np.where(m, a, 0.0) for a in ph.to_cartesian(hr))
    bh = np.hypot(bx, by)
    out["USFLUX"] = float(np.abs(bz[m]).sum() * area_cm2)
    out["MEANGAM"] = float(np.degrees(np.arctan2(bh[m], np.abs(bz[m]))).mean())
    v = ph.interior_valid(m)
    out["n_interior"] = int(v.sum())
    if not v.any():
        return out
    for name, f in (("MEANGBT", np.sqrt(bx ** 2 + by ** 2 + bz ** 2)), ("MEANGBZ", bz), ("MEANGBH", bh)):
        out[name] = float((np.hypot(ph.ddx(f), ph.ddy(f)) / pix_mm)[v].mean())
    curl_px = ph.jz(bx, -by)                       # (curl B)_z in G per pixel (Bt = -By)
    curl_mm = curl_px / pix_mm                     # G/Mm
    jz_ma = ph.jz_ma_per_m2(curl_px, cdelt_deg)    # mA/m^2
    jz_a = jz_ma * 1e-3
    bzi = bz[1:-1, 1:-1]
    out["MEANJZD"] = float(jz_ma[v].mean())
    out["TOTUSJZ"] = float(np.abs(jz_a[v]).sum() * area_m2)
    den = float((bzi[v] ** 2).sum())
    out["MEANALP"] = float((curl_mm[v] * bzi[v]).sum() / den) if den > 0 else float("nan")
    hc = bzi * curl_mm * 1e-6                      # G^2/m
    out["MEANJZH"] = float(hc[v].mean())
    out["TOTUSJH"] = float(np.abs(hc[v]).sum())
    out["ABSNJZH"] = float(abs(hc[v].sum()))
    pos, neg = v & (bzi > 0), v & (bzi < 0)
    out["SAVNCPP"] = float((abs(jz_a[pos].sum()) + abs(jz_a[neg].sum())) * area_m2)
    return out
