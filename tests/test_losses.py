"""Required loss tests (training_plan.md, Phase 4) plus behaviour checks.

Every loss takes (pred, target, valid_mask) with pred/target (B, 3, H, W) in
normalised units (NaNs already replaced by 0) and valid_mask (B, 1, H, W).
"""
import numpy as np
import pytest
import torch

from sr import losses as L

torch.manual_seed(0)
B, C, H, W = 2, 3, 20, 24


def _mask(b=B, h=H, w=W):
    return torch.ones(b, 1, h, w, dtype=torch.bool)


def _xy(h=H, w=W, dtype=torch.float64):
    y, x = torch.meshgrid(torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype), indexing="ij")
    return x, y


# ---------------------------------------------------------------- MSE

def test_mse_is_mean_over_valid_pixels_and_channels():
    t = torch.randn(B, C, H, W)
    assert L.mse_loss(t + 1.0, t, _mask()).item() == pytest.approx(1.0)


def test_mse_ignores_masked_pixels():
    t = torch.randn(B, C, H, W)
    p = t.clone()
    p[0, :, 3, 4] += 100.0
    m = _mask()
    m[0, 0, 3, 4] = False
    assert L.mse_loss(p, t, m).item() == pytest.approx(0.0)


# ---------------------------------------------------------------- gradient (Sobel + MSE)

def test_gradient_loss_is_zero_for_identical_and_offset_fields():
    t = torch.randn(B, C, H, W)
    assert L.gradient_loss(t, t, _mask()).item() == pytest.approx(0.0)
    assert L.gradient_loss(t + 3.0, t, _mask()).item() == pytest.approx(0.0, abs=1e-10)


def test_gradient_loss_uses_raw_sobel_scale():
    # f = x: raw Sobel Gx = (1 + 2 + 1) * 2 = 8, Gy = 0  ->  loss = 8^2 = 64
    x, _ = _xy(dtype=torch.float32)
    p = x.expand(B, C, H, W)
    assert L.gradient_loss(p, torch.zeros_like(p), _mask()).item() == pytest.approx(64.0)


def test_gradient_loss_ignores_windows_touching_a_masked_pixel():
    t = torch.zeros(B, C, H, W)
    p = t.clone()
    p[:, :, 8, 8] = 50.0
    m = _mask()
    m[:, :, 8, 8] = False
    assert L.gradient_loss(p, t, m).item() == pytest.approx(0.0)


# ---------------------------------------------------------------- divergence

def _bp_bt_from_xy(bx, by):
    return bx, -by


def test_divergence_match_is_zero_when_pred_equals_target():
    t = torch.randn(B, C, H, W)
    assert L.divergence_loss(t, t.clone(), _mask(), mode="match").item() == 0.0


def test_divergence_zero_mode_on_bx_x_by_y_is_four():
    x, y = _xy(dtype=torch.float32)
    bp, bt = _bp_bt_from_xy(x, y)
    p = torch.stack([bp, bt, torch.zeros_like(x)])[None]
    assert L.divergence_loss(p, torch.zeros_like(p), _mask(1), mode="zero").item() == pytest.approx(4.0)


def test_divergence_match_compares_to_true_divergence():
    x, y = _xy(dtype=torch.float32)
    bp, bt = _bp_bt_from_xy(x, y)                    # div = 2
    t = torch.stack([bp, bt, torch.zeros_like(x)])[None]
    p = torch.zeros_like(t)                          # div = 0
    assert L.divergence_loss(p, t, _mask(1), mode="match").item() == pytest.approx(4.0)


def test_divergence_mask_excludes_pixels_next_to_nan():
    t = torch.zeros(1, C, H, W)
    p = t.clone()
    p[0, 0, 10, 10] = 7.0          # would create divergence at (10, 9) and (10, 11)
    m = _mask(1)
    m[0, 0, 10, 10] = False        # (10, 10) was NaN in the data
    assert L.divergence_loss(p, t, m, mode="match").item() == 0.0
    assert L.divergence_loss(p, t, _mask(1), mode="match").item() > 0.0


def test_divergence_rejects_unknown_mode():
    t = torch.zeros(1, C, H, W)
    with pytest.raises(ValueError):
        L.divergence_loss(t, t, _mask(1), mode="other")


# ---------------------------------------------------------------- SSIM

def _ssim(win=11):
    return L.SSIMLoss(lo=torch.tensor([-0.2, -0.2, -0.5]), hi=torch.tensor([0.2, 0.2, 0.5]), win=win)


def test_ssim_loss_is_zero_for_identical_signed_fields():
    t = torch.randn(B, C, H, W) * 0.1               # signed data
    assert _ssim()(t, t, _mask()).item() == pytest.approx(0.0, abs=1e-6)


def test_ssim_loss_is_positive_for_different_fields():
    t = torch.randn(B, C, H, W) * 0.1
    assert _ssim()(torch.randn(B, C, H, W) * 0.1, t, _mask()).item() > 0.1


def test_ssim_uses_paper_constants_and_fixed_affine_map():
    s = _ssim()
    assert s.C1 == pytest.approx(1e-4) and s.C2 == pytest.approx(9e-3)
    t = torch.randn(1, C, H, W) * 0.1
    # the map is fixed, not per image: rescaling both images changes SSIM
    assert s(t, t * 0.5, _mask(1)).item() > 0.0


def test_ssim_ignores_windows_touching_masked_pixels():
    t = torch.randn(1, C, 30, 30) * 0.1
    p = t.clone()
    p[:, :, 0:3, 0:3] += 1.0
    m = _mask(1, 30, 30)
    m[:, :, 0:3, 0:3] = False
    assert _ssim()(p, t, m).item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- histogram

def _hist():
    centers = L.log_bin_centers(noise=60 / 3500, top=3000 / 3500, dl=0.15)
    return L.HistogramLoss([centers] * 3, count_budget=64 * 128 * 128)


def test_log_bin_centers_are_symmetric_and_include_zero():
    c = L.log_bin_centers(noise=60 / 3500, top=3000 / 3500, dl=0.15)
    assert torch.all(c[1:] > c[:-1])
    assert torch.allclose(c, -c.flip(0))
    assert (c == 0).sum() == 1
    assert c.max() >= 3000 / 3500 - 1e-9


def test_soft_counts_conserve_mass_and_are_rescaled_to_the_budget():
    h = _hist()
    x = torch.randn(B, C, H, W) * 0.3
    counts = h.soft_counts(x, _mask())
    assert counts.shape[0] == C
    assert torch.allclose(counts.sum(dim=1), torch.full((C,), 64.0 * 128 * 128), rtol=1e-5)


def test_histogram_loss_is_zero_for_identical_and_positive_for_shifted():
    h = _hist()
    t = torch.randn(B, C, H, W) * 0.1
    assert h(t, t, _mask()).item() == pytest.approx(0.0)
    assert h(t + 0.2, t, _mask()).item() > 0.0


def test_histogram_ignores_masked_pixels():
    h = _hist()
    t = torch.randn(B, C, H, W) * 0.1
    p = t.clone()
    p[0, :, 5, 5] = 0.9
    m = _mask()
    m[0, 0, 5, 5] = False
    assert h(p, t, m).item() == pytest.approx(0.0)


def _hist_tv():
    centers = L.log_bin_centers(noise=60 / 3500, top=3000 / 3500, dl=0.15)
    return L.HistogramLoss([centers] * 3, mode="tv")


def test_tv_mode_is_a_distance_between_distributions_in_0_1():
    h = _hist_tv()
    t = torch.full((B, C, H, W), -0.8)
    assert h(t, t, _mask()).item() == pytest.approx(0.0)
    assert h(-t, t, _mask()).item() == pytest.approx(1.0)       # disjoint distributions
    x = torch.randn(B, C, H, W) * 0.1
    assert 0.0 < h(x + 0.05, x, _mask()).item() < 1.0


def test_tv_mode_does_not_depend_on_the_number_of_pixels():
    h = _hist_tv()
    x = torch.randn(1, C, 16, 16) * 0.1
    big_x, big_t = x.repeat(4, 1, 1, 1), (x + 0.05).repeat(4, 1, 1, 1)
    assert h(big_x, big_t, _mask(4, 16, 16)).item() == pytest.approx(
        h(x, x + 0.05, _mask(1, 16, 16)).item(), rel=1e-5)


def test_histogram_rejects_unknown_mode():
    with pytest.raises(ValueError):
        L.HistogramLoss([torch.zeros(3)] * 3, mode="other")


# ---------------------------------------------------------------- differentiability

def _gc(fn, h=12, w=12):
    torch.manual_seed(1)
    p = (torch.randn(1, C, h, w, dtype=torch.float64) * 0.2).requires_grad_(True)
    t = torch.randn(1, C, h, w, dtype=torch.float64) * 0.2
    m = torch.ones(1, 1, h, w, dtype=torch.bool)
    return torch.autograd.gradcheck(lambda q: fn(q, t, m), (p,), eps=1e-6, atol=1e-5)


def test_gradcheck_mse():
    assert _gc(L.mse_loss)


def test_gradcheck_gradient_loss():
    assert _gc(L.gradient_loss)


def test_gradcheck_divergence_both_modes():
    assert _gc(lambda p, t, m: L.divergence_loss(p, t, m, mode="match"))
    assert _gc(lambda p, t, m: L.divergence_loss(p, t, m, mode="zero"))


def test_gradcheck_ssim():
    s = L.SSIMLoss(lo=torch.tensor([-0.5] * 3), hi=torch.tensor([0.5] * 3), win=5).double()
    assert _gc(s)


def test_gradcheck_soft_histogram():
    centers = L.log_bin_centers(noise=0.02, top=0.8, dl=0.15).double()
    hl = L.HistogramLoss([centers] * 3, count_budget=100.0)
    assert _gc(hl)
    assert _gc(L.HistogramLoss([centers] * 3, mode="tv"))


# ---------------------------------------------------------------- compound loss

def test_compound_loss_logs_every_term_and_sums_weighted_terms():
    ssim = _ssim()
    hist = _hist()
    loss = L.CompoundLoss({"mse": 1.0, "grad": 5.0, "hist": 0.0, "ssim": 5e-5, "div": 0.1},
                          div_mode="match", ssim=ssim, hist=hist)
    p, t = torch.randn(B, C, H, W) * 0.1, torch.randn(B, C, H, W) * 0.1
    total, terms = loss(p, t, _mask())
    assert set(terms) == {"mse", "grad", "hist", "ssim", "div", "total"}
    expected = sum(w * terms[k] for k, w in loss.weights.items())
    assert total.item() == pytest.approx(expected, rel=1e-6)
    assert terms["hist"] > 0          # logged although its weight is 0


def test_compound_loss_zero_weight_terms_do_not_get_gradients_of_their_own():
    loss = L.CompoundLoss({"mse": 1.0, "grad": 0.0, "hist": 0.0, "ssim": 0.0, "div": 0.0},
                          div_mode="match", ssim=_ssim(), hist=_hist())
    p = (torch.randn(1, C, H, W) * 0.1).requires_grad_(True)
    t = torch.randn(1, C, H, W) * 0.1
    total, _ = loss(p, t, _mask(1))
    total.backward()
    assert torch.allclose(p.grad, 2 * (p - t).detach() / p.numel(), atol=1e-7)


# ---------------------------------------------------------------- multi-scale divergence (match_ms; phase_6.md 8)

def _smooth_field(h=72, w=80, seed=0, amp=0.05):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    s = np.stack([gaussian_filter(rng.normal(size=(h, w)), 5) for _ in range(3)])
    return amp * s / s.std()


def test_torch_smoothed_divergence_equals_the_numpy_analysis():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import div_snr_by_scale as dsn
    from sr import physics as ph
    hr = _smooth_field()
    hr[:, 30:40, 20:34] = np.nan
    valid = torch.as_tensor(np.isfinite(hr).all(axis=0))[None]
    t = torch.as_tensor(np.nan_to_num(hr))[None]
    for sigma in (1.0, 2.0, 4.0):
        d_np, v_np = dsn.smoothed_div(hr, sigma)                      # G/Mm
        d_t, v_t = L.smoothed_div(t[:, 0], t[:, 1], valid, sigma)       # normalised units per pixel
        assert torch.equal(v_t[0], torch.as_tensor(v_np))
        d_t = d_t[0].numpy() * 3500.0 / ph.pixel_scale_mm()
        np.testing.assert_allclose(d_t[v_np], d_np[v_np], rtol=1e-9, atol=1e-9)


def test_match_ms_is_zero_when_pred_equals_target():
    t = torch.as_tensor(_smooth_field())[None]
    m = torch.ones(1, 1, *t.shape[-2:], dtype=torch.bool)
    assert L.divergence_loss(t, t.clone(), m, "match_ms", scales=(1, 2, 4), scale_norm=(1, 1, 1)).item() == 0.0


def test_match_ms_ignores_pixels_outside_the_mask():
    t = torch.as_tensor(_smooth_field())[None]
    m = torch.ones(1, 1, *t.shape[-2:], dtype=torch.bool)
    m[..., 10:20, 10:20] = False
    p = t + 0.01 * torch.randn_like(t)
    a = L.divergence_loss(p, t, m, "match_ms", scales=(1, 2), scale_norm=(1, 1))
    p2 = p.clone()
    p2[..., 12:18, 12:18] += 5.0                                         # garbage where the target is invalid
    b = L.divergence_loss(p2, t, m, "match_ms", scales=(1, 2), scale_norm=(1, 1))
    assert a.item() == pytest.approx(b.item(), rel=1e-12)


def test_match_ms_is_insensitive_to_pixel_scale_noise_unlike_pixel_match():
    t = torch.as_tensor(_smooth_field())[None]
    m = torch.ones(1, 1, *t.shape[-2:], dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    noisy = t + 0.02 * torch.randn(t.shape, generator=g, dtype=t.dtype)
    shifted = t.clone()
    shifted[:, 0] = torch.roll(t[:, 0], 3, dims=-1)                     # wrong large-scale structure
    pix = lambda p: L.divergence_loss(p, t, m, "match").item()
    ms = lambda p: L.divergence_loss(p, t, m, "match_ms", scales=(4,), scale_norm=(1,)).item()
    assert pix(noisy) > pix(shifted)          # the pixel loss is dominated by pixel noise
    assert ms(noisy) < ms(shifted)            # the multi-scale loss ranks real structure first


def test_match_ms_normalises_each_scale():
    t = torch.as_tensor(_smooth_field())[None]
    p = t + 0.01 * torch.randn_like(t)
    m = torch.ones(1, 1, *t.shape[-2:], dtype=torch.bool)
    a = L.divergence_loss(p, t, m, "match_ms", scales=(2,), scale_norm=(1.0,))
    b = L.divergence_loss(p, t, m, "match_ms", scales=(2,), scale_norm=(4.0,))
    assert a.item() == pytest.approx(4.0 * b.item())


def test_gradcheck_divergence_match_ms():
    t = torch.as_tensor(_smooth_field(h=24, w=28))[None]
    m = torch.ones(1, 1, 24, 28, dtype=torch.bool)
    m[..., :3, :5] = False
    p = (t + 0.01 * torch.randn_like(t)).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda x: L.divergence_loss(x, t, m, "match_ms", scales=(1, 2), scale_norm=(1e-4, 2e-5)), (p,))


def test_compound_loss_passes_the_scales_to_the_divergence_term():
    t = torch.as_tensor(_smooth_field())[None].float()
    m = torch.ones(1, 1, *t.shape[-2:], dtype=torch.bool)
    p = t + 0.01 * torch.randn_like(t)
    loss = L.CompoundLoss({"mse": 1.0, "div": 0.5}, div_mode="match_ms", div_scales=(1, 4), div_scale_norm=(1.0, 1.0))
    total, terms = loss(p, t, m)
    expect = L.divergence_loss(p, t, m, "match_ms", scales=(1, 4), scale_norm=(1.0, 1.0)).item()
    assert terms["div"] == pytest.approx(expect, rel=1e-6)
    assert total.item() == pytest.approx(terms["mse"] + 0.5 * expect, rel=1e-5)
