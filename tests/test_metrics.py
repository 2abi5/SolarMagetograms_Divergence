"""Tests for sr/metrics.py (per-region metrics, Gauss units, (3, H, W) stacks)."""
import numpy as np
import pytest

from sr import metrics as M
from sr import physics as ph

rng = np.random.default_rng(0)
H, W = 40, 48


def _true():
    y, x = np.mgrid[0:H, 0:W].astype(float)
    br = 800 * np.exp(-((x - 15) ** 2 + (y - 20) ** 2) / 60) - 800 * np.exp(-((x - 33) ** 2 + (y - 20) ** 2) / 60)
    bp = 300 * np.sin(x / 6)
    bt = 200 * np.cos(y / 5)
    return np.stack([bp, bt, br])


def test_pixel_metrics_perfect_prediction():
    t = _true()
    m = M.pixel_metrics(t, t, np.ones((H, W), bool), data_range=np.array([1000.0, 1000.0, 2000.0]))
    for c in ("Bp", "Bt", "Br", "B"):
        assert m[c]["rmse"] == 0 and m[c]["mae"] == 0 and m[c]["pearson"] == pytest.approx(1.0)
        assert m[c]["ssim"] == pytest.approx(1.0)
        assert np.isinf(m[c]["psnr"])


def test_pixel_metrics_known_values_and_mask():
    t = _true()
    p = t + 10.0
    mask = np.ones((H, W), bool)
    mask[0, 0] = False
    p[:, 0, 0] = 1e6                       # outside the mask: ignored
    m = M.pixel_metrics(p, t, mask, data_range=np.array([1000.0, 1000.0, 2000.0]))
    assert m["Br"]["rmse"] == pytest.approx(10.0) and m["Br"]["mae"] == pytest.approx(10.0)
    assert m["Br"]["psnr"] == pytest.approx(20 * np.log10(2000 / 10))
    assert m["Br"]["pearson"] == pytest.approx(1.0)


def test_field_strength_channel_is_the_vector_magnitude():
    t = _true()
    b = M.field_strength(t)
    assert np.allclose(b, np.sqrt((t ** 2).sum(0)))


def test_strong_mask_uses_true_field_strength():
    t = _true()
    s = M.strong_mask(t, np.ones((H, W), bool), threshold=200.0)
    assert s.sum() == (M.field_strength(t) > 200).sum() and s.any() and not s.all()


def test_variance_ratio_flags_smoothing():
    t = _true()
    mask = np.ones((H, W), bool)
    assert M.variance_ratio(t, t, mask)["Bp"] == pytest.approx(1.0)
    flat = np.broadcast_to(t.mean(axis=(1, 2), keepdims=True), t.shape)
    assert M.variance_ratio(flat, t, mask)["Bp"] == pytest.approx(0.0)
    assert M.variance_ratio(0.3 * t, t, mask)["Br"] == pytest.approx(0.3)


def test_divergence_metrics_in_gauss_per_megametre():
    y, x = np.mgrid[0:H, 0:W].astype(float)
    true = np.stack([x, -y, np.zeros_like(x)])        # Bx = x, By = y: div = 2 G/px
    pred = np.zeros_like(true)                        # div = 0
    m = M.physics_metrics(pred, true, np.ones((H, W), bool))
    assert m["div_err_g_per_mm"] == pytest.approx(2 / ph.pixel_scale_mm())
    assert m["div_std_true_g_per_mm"] == pytest.approx(0.0, abs=1e-9)
    assert m["jz_err_ma_per_m2"] == pytest.approx(0.0)


def test_divergence_std_ratio_detects_flattening():
    t = _true() + rng.normal(size=(3, H, W)) * 50
    m_same = M.physics_metrics(t, t, np.ones((H, W), bool))
    assert m_same["div_std_ratio"] == pytest.approx(1.0) and m_same["div_err_g_per_mm"] == 0
    m_flat = M.physics_metrics(np.zeros_like(t), t, np.ones((H, W), bool))
    assert m_flat["div_std_ratio"] == pytest.approx(0.0)


def test_flux_metrics():
    t = _true()
    m = M.flux_metrics(t, t, np.ones((H, W), bool))
    assert m["signed_flux_imbalance"] == pytest.approx(0.0) and m["unsigned_flux_rel_err"] == pytest.approx(0.0)
    p = t.copy()
    p[2] *= 2
    m = M.flux_metrics(p, t, np.ones((H, W), bool))
    assert m["unsigned_flux_rel_err"] == pytest.approx(1.0)
    expected = abs(t[2].sum()) / np.abs(t[2]).sum()   # |2S - S| / sum|Br|
    assert m["signed_flux_imbalance"] == pytest.approx(expected)


def test_monopole_artifacts_count_connected_clusters_above_threshold():
    t = np.zeros((3, H, W))
    p = t.copy()
    p[0, 10, 10] = 100.0          # one isolated Bp spike -> a cluster of div error around it
    p[0, 30, 30] = 100.0          # another, far away
    n = M.monopole_artifacts(p, t, np.ones((H, W), bool), threshold_g_per_mm=10.0)
    assert n == 2
    assert M.monopole_artifacts(p, t, np.ones((H, W), bool), threshold_g_per_mm=1e6) == 0


def test_extreme_value_metrics():
    t = _true()
    centers = np.linspace(-1000, 1000, 21)
    m = M.extreme_metrics(t, t, np.ones((H, W), bool), centers)
    assert m["Br"]["p99_pred"] == pytest.approx(m["Br"]["p99_true"])
    assert m["Br"]["hist_tv"] == pytest.approx(0.0)
    m2 = M.extreme_metrics(0.5 * t, t, np.ones((H, W), bool), centers)
    assert m2["Br"]["p99_pred"] == pytest.approx(0.5 * m2["Br"]["p99_true"])
    assert 0 < m2["Br"]["hist_tv"] <= 1


def test_radial_power_spectrum_of_smooth_field_lacks_high_frequencies():
    t = _true()[2]
    k, p_true = M.radial_power_spectrum(t)
    noisy = t + rng.normal(size=t.shape) * 100
    _, p_noisy = M.radial_power_spectrum(noisy)
    assert len(k) == len(p_true)
    assert p_noisy[-5:].mean() > 10 * p_true[-5:].mean()


def test_bootstrap_ci_over_harps_contains_the_mean():
    import pandas as pd
    df = pd.DataFrame({"harpnum": np.repeat(np.arange(10), 3), "rmse": rng.normal(100, 10, 30)})
    mean, lo, hi = M.bootstrap_ci(df, "rmse", n_boot=2000, seed=0)
    per_harp = df.groupby("harpnum").rmse.mean()
    assert mean == pytest.approx(per_harp.mean()) and lo < mean < hi
