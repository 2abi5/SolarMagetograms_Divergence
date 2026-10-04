"""Tests for the Munoz-style metric helpers in scripts/eval_v2.py."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_v2 as ev  # noqa: E402


def test_tile_extremes_zero_for_perfect_prediction_and_count_max_and_min_errors():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(64, 96)) * 100
    mask = np.ones_like(t, bool)
    assert ev.tile_extreme_error(t, t, mask, tile=32) == 0.0
    p = t.copy()
    p[t == t[:32, :32].max()] += 50.0          # raise the maximum of one tile by 50 G
    assert ev.tile_extreme_error(p, t, mask, tile=32) == pytest.approx(50.0 / 6)   # 6 tiles, one off by 50


def test_tiles_with_invalid_pixels_are_skipped():
    t = np.ones((64, 64))
    mask = np.ones_like(t, bool)
    mask[0, 0] = False
    p = t.copy()
    p[:32, :32] += 100
    assert ev.tile_extreme_error(p, t, mask, tile=32) == 0.0


def test_gradient_error_is_zero_for_identical_and_offset_fields():
    t = np.random.default_rng(1).normal(size=(40, 50))
    mask = np.ones_like(t, bool)
    assert ev.sobel_error(t, t, mask) == 0.0
    assert ev.sobel_error(t + 7.0, t, mask) == pytest.approx(0.0, abs=1e-9)
    assert ev.sobel_error(t * 2, t, mask) > 0


def test_pooled_pearson_by_bins_matches_numpy():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=500), rng.normal(size=500)
    y = 0.6 * x + 0.8 * y
    b = rng.integers(0, 3, size=500)
    acc = {}
    ev.add_binned(acc, "m", "Br", x, y, b, n_bins=3)
    for k in range(3):
        sel = b == k
        assert ev.pearson_from(acc[("m", "Br", k)]) == pytest.approx(np.corrcoef(x[sel], y[sel])[0, 1], rel=1e-9)


def test_radius_bins_follow_munoz_table4_edges():
    # c_r = cos(theta) -> r = sin(theta); edges 1/3, 1/2, 3/4
    c_r = np.cos(np.arcsin(np.array([0.1, 0.4, 0.6, 0.8])))
    np.testing.assert_array_equal(ev.radius_bin(c_r), [0, 1, 2, 3])


def test_subsample_keeps_every_nth_frame_of_each_harp_in_time_order():
    import pandas as pd
    val = pd.DataFrame({"harpnum": [1] * 5 + [2] * 3,
                        "mdi_t_rec": ["t3", "t1", "t5", "t2", "t4", "a1", "a3", "a2"]})
    out = ev.subsample_regions(val, every=2)
    assert list(zip(out.harpnum, out.mdi_t_rec)) == [(1, "t1"), (1, "t3"), (1, "t5"), (2, "a1"), (2, "a3")]
    assert len(ev.subsample_regions(val, every=1)) == len(val)
