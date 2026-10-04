"""Required physics tests (training_plan.md, Phase 4) plus unit checks.

Conventions (plan section 1): hr[0]=Bp, hr[1]=Bt, hr[2]=Br; Bx = Bp, By = -Bt,
Bz = Br; array rows increase with y, columns with x.
"""
import numpy as np
import pytest
import torch

from sr import physics as ph

H, W = 24, 32
Y, X = np.mgrid[0:H, 0:W].astype(float)


def _from_cartesian(bx, by):
    """(Bp, Bt) for a given in-plane Cartesian field."""
    return bx, -by


def test_div_of_sin_y_sin_x_is_zero():
    bp, bt = _from_cartesian(np.sin(0.3 * Y), np.sin(0.3 * X))
    d = ph.div_h(bp, bt)
    assert d.shape == (H - 2, W - 2)
    assert np.allclose(d, 0.0, atol=1e-12)


def test_div_of_x_y_is_two_on_interior():
    bp, bt = _from_cartesian(X, Y)
    assert np.allclose(ph.div_h(bp, bt), 2.0)


def test_sign_convention_by_is_minus_bt():
    # Bt = -y means By = +y, so div = dBy/dy = +1; Bt = +y gives -1
    assert np.allclose(ph.div_h(np.zeros_like(Y), -Y), 1.0)
    assert np.allclose(ph.div_h(np.zeros_like(Y), Y), -1.0)
    # Bp = x (westward, increasing with x) gives +1
    assert np.allclose(ph.div_h(X, np.zeros_like(X)), 1.0)


def test_to_cartesian_uses_by_minus_bt():
    hr = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 2.0), np.full((2, 2), 3.0)])
    bx, by, bz = ph.to_cartesian(hr)
    assert (bx == 1).all() and (by == -2).all() and (bz == 3).all()


def test_jz_of_rotation_is_two():
    # Bx = -y, By = x  ->  dBy/dx - dBx/dy = 2
    bp, bt = _from_cartesian(-Y, X)
    assert np.allclose(ph.jz(bp, bt), 2.0)


def test_jz_of_curl_free_field_is_zero():
    bp, bt = _from_cartesian(X, Y)
    assert np.allclose(ph.jz(bp, bt), 0.0)


def test_div_works_on_torch_batches():
    bp, bt = _from_cartesian(X, Y)
    t = torch.tensor(np.stack([bp, bt]))[None].repeat(3, 1, 1, 1)   # (B, 2, H, W)
    d = ph.div_h(t[:, 0], t[:, 1])
    assert d.shape == (3, H - 2, W - 2) and torch.allclose(d, torch.tensor(2.0, dtype=d.dtype))


def test_valid_mask_excludes_pixels_next_to_nan():
    bp = X.copy()
    bp[10, 10] = np.nan
    valid = ph.interior_valid(np.isfinite(bp) & np.isfinite(Y))
    assert valid.shape == (H - 2, W - 2)
    # interior index (i-1, j-1) <-> full index (i, j)
    for (i, j) in [(10, 10), (9, 10), (11, 10), (10, 9), (10, 11)]:
        assert not valid[i - 1, j - 1]
    assert valid[5, 5] and valid[8, 8]          # interior (8,8) = full (9,9): diagonal of the NaN
    d = ph.div_h(bp, -Y)
    assert np.isfinite(d[valid]).all()


def test_valid_mask_diagonal_neighbour_of_nan_stays_valid():
    v = np.ones((5, 5), bool)
    v[2, 2] = False
    iv = ph.interior_valid(v)                   # interior (3, 3), index (i-1, j-1)
    assert not iv[1, 1]                          # the NaN pixel itself
    assert not iv[0, 1] and not iv[2, 1] and not iv[1, 0] and not iv[1, 2]   # 4-neighbours
    assert iv[0, 0] and iv[0, 2] and iv[2, 0] and iv[2, 2]                    # diagonals


# ---------------------------------------------------------------- physical units

def test_cea_pixel_scale_is_about_364_km():
    assert ph.pixel_scale_mm(0.03) == pytest.approx(0.03 * np.pi / 180 * 696.0)
    assert ph.pixel_scale_mm(0.03) == pytest.approx(0.3644, abs=1e-4)


def test_divergence_is_converted_to_gauss_per_megametre():
    # a gradient of 1 G per pixel = 1 / 0.3644 G/Mm
    div_px = np.full((3, 3), 1.0)
    got = ph.div_g_per_mm(div_px, cdelt_deg=0.03)
    assert np.allclose(got, 1.0 / ph.pixel_scale_mm(0.03))
    # LR grid (0.12 deg/px) is 4x coarser
    assert np.allclose(ph.div_g_per_mm(div_px, cdelt_deg=0.12), got / 4)


def test_div_from_normalised_hr_in_gauss_per_megametre():
    bp, bt = _from_cartesian(X, Y)                     # div = 2 in normalised units per pixel
    hr = np.stack([bp, bt, np.zeros_like(X)]) / 3500.0
    d = ph.div_h_physical(hr, norm=3500.0, cdelt_deg=0.03)
    assert np.allclose(d, 2.0 / 3500.0 * 3500.0 / ph.pixel_scale_mm(0.03))


def test_jz_is_converted_to_milliampere_per_square_metre():
    # 1 G/px over 364.4 km: (1e-4 T / 3.644e5 m) / mu0 = 2.184e-4 A/m^2 = 0.2184 mA/m^2
    got = ph.jz_ma_per_m2(np.array([1.0]), cdelt_deg=0.03)
    expected = 1e-4 / (ph.pixel_scale_mm(0.03) * 1e6) / (4e-7 * np.pi) * 1e3
    assert got[0] == pytest.approx(expected)
    assert got[0] == pytest.approx(0.2184, abs=1e-3)


def test_flux_uses_pixel_area():
    br = np.ones((10, 10))                              # 1 G on 100 pixels
    area_cm2 = (ph.pixel_scale_mm(0.03) * 1e8) ** 2
    assert ph.unsigned_flux_mx(br, cdelt_deg=0.03) == pytest.approx(100 * area_cm2)
    br[:5] = -1
    assert ph.signed_flux_mx(br, cdelt_deg=0.03) == pytest.approx(0.0)
