"""Tests for scripts/eval_small_flux.py (masks of small confined-flux regions, from the TRUE SHARP field)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_small_flux as sf  # noqa: E402


def test_compact_features_keep_small_strong_patches_and_drop_big_spots():
    br = np.zeros((80, 80))
    br[5:9, 5:9] = 500.0            # 16 px compact feature -> kept
    br[30:60, 30:60] = -900.0       # 900 px sunspot -> dropped (> 256 px)
    br[70, 70] = 800.0              # 1 px speck -> dropped (< 4 px)
    m = sf.compact_features(br, thr=300.0, min_px=4, max_px=256)
    assert m[5:9, 5:9].all() and not m[30:60, 30:60].any() and not m[70, 70]


def test_mixed_polarity_blocks_find_flux_that_cancels_inside_an_mdi_pixel():
    br = np.zeros((8, 8))
    br[0:2, 0:4] = 400.0; br[2:4, 0:4] = -400.0     # block (0,0): both polarities, cancels -> hidden flux
    br[0:4, 4:8] = 400.0                            # block (0,1): one polarity -> no
    br[4:6, 0:4] = 400.0; br[6:8, 0:4] = -50.0      # block (1,0): weak opposite polarity -> no
    m = sf.mixed_polarity_blocks(br, scale=4, thr=100.0, min_px=2, max_net=0.5)
    assert m[0:4, 0:4].all() and not m[0:4, 4:8].any() and not m[4:8, 0:4].any()


def test_pil_mask_marks_pixels_next_to_a_strong_sign_change():
    br = np.full((20, 20), 400.0)
    br[:, 10:] = -400.0
    m = sf.pil_mask(br, thr=150.0, width=2)
    assert m[:, 8:12].all() and not m[:, :6].any() and not m[:, 14:].any()


def test_flux_ratio_is_one_for_perfect_prediction():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(20, 20)) * 300
    m = np.ones_like(t, bool)
    assert sf.flux_ratio(t, t, m) == pytest.approx(1.0)
    assert sf.flux_ratio(t * 0.5, t, m) == pytest.approx(0.5)
