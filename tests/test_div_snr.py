"""Tests for scripts/div_snr_by_scale.py (divergence signal vs noise by smoothing scale)."""
import os
import sys

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import div_snr_by_scale as dsn  # noqa: E402
from sr import physics as ph  # noqa: E402


def _signal(h=128, w=160, seed=0, amp=0.05):
    rng = np.random.default_rng(seed)
    s = np.stack([gaussian_filter(rng.normal(size=(h, w)), 6) for _ in range(3)])
    return amp * s / s.std()


def test_scale_zero_is_the_plain_pixel_divergence():
    hr = _signal()
    d, v = dsn.smoothed_div(hr, 0.0)
    np.testing.assert_allclose(d, ph.div_h_physical(hr), rtol=1e-12)
    assert v.all() and v.shape == d.shape


def test_smoothing_keeps_nan_pixels_out():
    hr = _signal()
    hr[:, 40:50, 60:70] = np.nan
    d, v = dsn.smoothed_div(hr, 2.0)
    assert not v[39:49, 59:69].any()          # interior coordinates are shifted by one pixel
    assert np.isfinite(d[v]).all()


def test_noise_dominated_at_pixel_scale_and_signal_dominated_when_smoothed():
    s = _signal()
    rng = np.random.default_rng(1)
    a = s + rng.normal(scale=0.02, size=s.shape)
    b = s + rng.normal(scale=0.02, size=s.shape)
    r0 = dsn.pooled([dsn.pair_moments(a, b, 0.0)])
    r4 = dsn.pooled([dsn.pair_moments(a, b, 4.0)])
    assert r0["corr"] < 0.3 and r4["corr"] > 0.9
    assert r4["snr"] > 3 * r0["snr"]


def test_identical_frames_have_correlation_one_and_no_noise():
    s = _signal()
    r = dsn.pooled([dsn.pair_moments(s, s.copy(), 1.0)])
    assert r["corr"] == pytest.approx(1.0) and r["noise_rms_g_per_mm"] == pytest.approx(0.0, abs=1e-9)


def test_strong_mask_uses_only_pixels_above_the_threshold():
    s = _signal()
    babs = np.sqrt((s ** 2).sum(axis=0)) * 3500
    thr = float(np.percentile(babs, 90))
    m_all = dsn.pair_moments(s, s.copy(), 0.0)
    m_strong = dsn.pair_moments(s, s.copy(), 0.0, strong_gauss=thr)
    assert 0 < m_strong["n"] < 0.2 * m_all["n"]
