"""
Physics conventions for SHARP CEA vector data (training_plan.md, section 1).

Channel order everywhere: hr[..., 0, :, :] = Bp, hr[..., 1, :, :] = Bt,
hr[..., 2, :, :] = Br.  SHARP -> local Cartesian:

    Bx =  Bp   (westward)
    By = -Bt   (Bt is positive SOUTHward)
    Bz =  Br   (radial, up)

Array rows increase with y (FITS/sunpy origin lower-left), columns with x.
The CEA grid is uniform, so pixel-unit central differences are consistent.
Derivatives are central differences on interior pixels, so derivative maps
have shape (..., H-2, W-2); index (i, j) of the interior is pixel (i+1, j+1).

Every function works on numpy arrays and torch tensors alike.
"""

import math

import numpy as np
import torch

RSUN_MM = 696.0            # SHARP RSUN_REF = 6.96e8 m
MU0 = 4e-7 * math.pi       # T m / A
CDELT_HR_DEG = 0.03        # hmi.sharp_cea_720s pixel, degrees
CDELT_LR_DEG = 0.12        # x4 coarser LR grid


def _isfinite(x):
    return torch.isfinite(x) if torch.is_tensor(x) else np.isfinite(x)


def to_cartesian(hr):
    """(Bx, By, Bz) from a (..., 3, H, W) SHARP stack."""
    return hr[..., 0, :, :], -hr[..., 1, :, :], hr[..., 2, :, :]


def ddx(f):
    """Central difference along x (columns), interior pixels only."""
    return (f[..., 1:-1, 2:] - f[..., 1:-1, :-2]) / 2.0


def ddy(f):
    """Central difference along y (rows), interior pixels only."""
    return (f[..., 2:, 1:-1] - f[..., :-2, 1:-1]) / 2.0


def div_h(bp, bt):
    """Horizontal divergence dBx/dx + dBy/dy = dBp/dx - dBt/dy, per pixel."""
    return ddx(bp) - ddy(bt)


def jz(bp, bt):
    """Vertical current proxy dBy/dx - dBx/dy = -dBt/dx - dBp/dy, per pixel."""
    return -ddx(bt) - ddy(bp)


def interior_valid(valid):
    """Interior mask: a pixel is valid only if it and its 4 neighbours are."""
    return (valid[..., 1:-1, 1:-1] & valid[..., :-2, 1:-1] & valid[..., 2:, 1:-1]
            & valid[..., 1:-1, :-2] & valid[..., 1:-1, 2:])


def finite_mask(hr):
    """(..., H, W) mask of pixels where every channel of hr is finite."""
    f = _isfinite(hr)
    return f.all(dim=-3) if torch.is_tensor(f) else f.all(axis=-3)


# --------------------------------------------------------------------------
# Physical units
# --------------------------------------------------------------------------

def pixel_scale_mm(cdelt_deg=CDELT_HR_DEG, rsun_mm=RSUN_MM):
    """Length of one CEA pixel on the solar surface, in Mm (0.03 deg ~ 0.364 Mm)."""
    return math.radians(cdelt_deg) * rsun_mm


def div_g_per_mm(div_px, cdelt_deg=CDELT_HR_DEG):
    """Divergence from G per pixel to G per Mm."""
    return div_px / pixel_scale_mm(cdelt_deg)


def div_h_physical(hr, norm=3500.0, cdelt_deg=CDELT_HR_DEG):
    """Horizontal divergence of a normalised (..., 3, H, W) stack, in G/Mm."""
    return div_g_per_mm(div_h(hr[..., 0, :, :], hr[..., 1, :, :]) * norm, cdelt_deg)


def jz_ma_per_m2(jz_g_per_px, cdelt_deg=CDELT_HR_DEG):
    """Jz from G per pixel to mA/m^2:  (dB/dl [T/m]) / mu0 * 1e3."""
    return jz_g_per_px * 1e-4 / (pixel_scale_mm(cdelt_deg) * 1e6) / MU0 * 1e3


def jz_physical(hr, norm=3500.0, cdelt_deg=CDELT_HR_DEG):
    """Vertical current of a normalised (..., 3, H, W) stack, in mA/m^2."""
    return jz_ma_per_m2(jz(hr[..., 0, :, :], hr[..., 1, :, :]) * norm, cdelt_deg)


def _pixel_area_cm2(cdelt_deg):
    return (pixel_scale_mm(cdelt_deg) * 1e8) ** 2


def signed_flux_mx(br_gauss, cdelt_deg=CDELT_HR_DEG):
    """Sum of Br (G) x pixel area (cm^2) over finite pixels, in Mx."""
    b = br_gauss[_isfinite(br_gauss)]
    return float(b.sum()) * _pixel_area_cm2(cdelt_deg)


def unsigned_flux_mx(br_gauss, cdelt_deg=CDELT_HR_DEG):
    b = br_gauss[_isfinite(br_gauss)]
    return float(abs(b).sum()) * _pixel_area_cm2(cdelt_deg)
