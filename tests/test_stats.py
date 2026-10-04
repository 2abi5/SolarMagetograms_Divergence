"""Tests for sr/stats.py: training-split statistics and divergence noise floor."""
import numpy as np
import pytest

from sr import physics as ph
from sr import stats as st


def _hr(rng, h=40, w=48, scale=0.1):
    return (rng.normal(size=(3, h, w)) * scale).astype(np.float64)


def test_channel_stats_in_gauss_ignore_nan():
    hr = np.zeros((3, 10, 10))
    hr[0] = 1.0 / 3500          # 1 G
    hr[1] = -2.0 / 3500
    hr[2] = np.linspace(-1, 1, 100).reshape(10, 10) * 1000 / 3500
    hr[2, 0, 0] = np.nan
    s = st.channel_stats([hr], norm=3500.0)
    assert s["Bp"]["mean"] == pytest.approx(1.0) and s["Bp"]["std"] == pytest.approx(0.0)
    assert s["Bt"]["mean"] == pytest.approx(-2.0)
    assert s["Br"]["n"] == 99
    assert s["Br"]["p1"] < -950 and s["Br"]["p99"] > 950
    assert set(s["Br"]) >= {"mean", "std", "p1", "p99", "abs_p999"}


def test_ssim_ranges_are_symmetric_fixed_affine_per_channel():
    rng = np.random.default_rng(0)
    lo, hi = st.ssim_ranges([_hr(rng)])
    assert lo.shape == (3,) and np.allclose(lo, -hi) and (hi > 0).all()


def test_spatial_noise_floor_is_zero_for_smooth_fields_and_grows_with_noise():
    y, x = np.mgrid[0:40, 0:48].astype(float)
    smooth = np.stack([0.01 * x, -0.01 * y, np.zeros_like(x)])          # div constant
    rng = np.random.default_rng(1)
    noisy = smooth + rng.normal(size=smooth.shape) * 0.01
    f_smooth = st.spatial_div_noise(smooth, sigma=1.0, norm=3500.0)
    f_noisy = st.spatial_div_noise(noisy, sigma=1.0, norm=3500.0)
    assert f_smooth == pytest.approx(0.0, abs=1e-6)
    assert f_noisy > 1.0


def test_temporal_noise_floor_matches_injected_noise():
    rng = np.random.default_rng(2)
    base = _hr(rng, scale=0.0)
    noise = 0.002
    a = base + rng.normal(size=base.shape) * noise
    b = base + rng.normal(size=base.shape) * noise
    got = st.temporal_div_noise(a, b, norm=3500.0, max_change_gauss=1e9)
    # div of iid noise: var = 2 * (2 sigma^2 / 4) per frame -> difference of two frames
    sd_div = np.sqrt(2 * 2 * (noise ** 2) / 4) * 3500 / ph.pixel_scale_mm(0.03)
    expected_mean_abs = np.sqrt(2) * sd_div * np.sqrt(2 / np.pi)
    assert got["mean_abs_g_per_mm"] == pytest.approx(expected_mean_abs, rel=0.05)


def test_temporal_noise_floor_excludes_pixels_that_changed():
    rng = np.random.default_rng(3)
    a = _hr(rng, scale=0.001)
    b = a.copy()
    b[:, 10:20, 10:20] += 0.5                  # real evolution (1750 G) in one block
    got = st.temporal_div_noise(a, b, norm=3500.0, max_change_gauss=50.0)
    assert got["mean_abs_g_per_mm"] == pytest.approx(0.0, abs=1e-9)
    assert got["n_pixels"] < (40 - 2) * (48 - 2)
