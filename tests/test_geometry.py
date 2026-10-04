"""Tests for sr/geometry.py: per-pixel viewing geometry (Munoz's location channel,
extended to vectors) and vector-aware augmentation."""
import os

import numpy as np
import pandas as pd
import pytest

from sr import geometry as geo
from sr import physics as ph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = os.path.join(ROOT, "data", "pilot")
needs_pilot = pytest.mark.skipif(not os.path.exists(os.path.join(PILOT, "regions.csv")), reason="pilot data absent")


def _pilot_regions(n=3):
    reg = pd.read_csv(os.path.join(PILOT, "regions.csv"))
    return reg.sample(n, random_state=1)


@needs_pilot
def test_coefficients_form_a_unit_line_of_sight_vector():
    for r in _pilot_regions().itertuples():
        c = geo.region_los_coefficients(os.path.join(PILOT, r.region_file))
        with np.load(os.path.join(PILOT, r.region_file)) as d:
            assert c.shape == (3,) + d["lr"].shape
        np.testing.assert_allclose((c.astype(np.float64) ** 2).sum(axis=0), 1.0, atol=1e-5)


@needs_pilot
def test_projected_sharp_matches_mdi_as_well_as_the_alignment_step_found():
    # sum_k c_k * B_k (4x4 block mean) is SHARP seen along MDI's line of sight; its correlation with
    # the MDI input must reproduce the stored post-refinement alignment correlation (validates signs)
    for r in _pilot_regions(5).itertuples():
        path = os.path.join(PILOT, r.region_file)
        c = geo.region_los_coefficients(path)
        with np.load(path) as d:
            lr, hr = d["lr"].astype(np.float64), d["hr"].astype(np.float64)
            stored = float(d["align_corr_los"])
        ny, nx = lr.shape
        bm = hr.reshape(3, ny, 4, nx, 4).mean(axis=(2, 4))
        los = (c * bm).sum(axis=0)
        ok = np.isfinite(lr) & np.isfinite(los)
        corr = np.corrcoef(lr[ok], los[ok])[0, 1]
        assert corr == pytest.approx(stored, abs=0.02), (r.region_file, corr, stored)


def _toy(seed=0, h=8, w=10):
    rng = np.random.default_rng(seed)
    lr = rng.normal(size=(4, h, w))
    lr[1:] = rng.uniform(-1, 1, size=(3, h, w))
    hr = rng.normal(size=(3, 4 * h, 4 * w))
    return lr, hr


def test_polarity_flip_negates_the_field_but_not_the_geometry():
    lr, hr = _toy()
    a_lr, a_hr = geo.augment(lr, hr, flip=True, mirror_x=False, mirror_y=False)
    np.testing.assert_array_equal(a_lr[0], -lr[0])
    np.testing.assert_array_equal(a_lr[1:], lr[1:])
    np.testing.assert_array_equal(a_hr, -hr)


@pytest.mark.parametrize("mx,my", [(True, False), (False, True), (True, True)])
def test_mirrors_keep_the_line_of_sight_relation_consistent(mx, my):
    lr, hr = _toy(1)
    h, w = lr.shape[1:]
    los = lambda l, b: (l[1:] * b.reshape(3, h, 4, w, 4).mean(axis=(2, 4))).sum(axis=0)
    a_lr, a_hr = geo.augment(lr, hr, flip=False, mirror_x=mx, mirror_y=my)
    expect = los(lr, hr)
    if mx:
        expect = expect[:, ::-1]
    if my:
        expect = expect[::-1, :]
    np.testing.assert_allclose(los(a_lr, a_hr), expect, atol=1e-12)


@pytest.mark.parametrize("mx,my", [(True, False), (False, True)])
def test_mirrors_carry_the_divergence_along(mx, my):
    _, hr = _toy(2)
    _, a_hr = geo.augment(np.zeros((4, 8, 10)), hr, flip=False, mirror_x=mx, mirror_y=my)
    d, da = ph.div_h(hr[0], hr[1]), ph.div_h(a_hr[0], a_hr[1])
    expect = d[:, ::-1] if mx else d[::-1, :]
    np.testing.assert_allclose(da, expect, atol=1e-12)


def test_augment_works_without_geometry_channels():
    rng = np.random.default_rng(3)
    lr, hr = rng.normal(size=(1, 8, 10)), rng.normal(size=(3, 32, 40))
    a_lr, a_hr = geo.augment(lr, hr, flip=True, mirror_x=True, mirror_y=False)
    np.testing.assert_array_equal(a_lr[0], -lr[0][:, ::-1])
    np.testing.assert_array_equal(a_hr[0], hr[0][:, ::-1])          # Bp: mirror (-1) and polarity (-1)
