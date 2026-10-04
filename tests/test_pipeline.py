"""Tests for the Phase 1 additions to data_active_region_created.py."""
import numpy as np
import pytest
from astropy.wcs import WCS

import data_active_region_created as pipe


# ---------------------------------------------------------------- CLI defaults

def test_new_flags_default_to_current_behaviour():
    args = pipe.build_parser().parse_args(
        ["--email", "x@y.z", "--start", "2010-06-01T00:00:00", "--end", "2010-06-01T02:00:00"])
    assert args.save_mode == "patches"
    assert args.dry_run is False
    assert args.max_center_angle is None
    assert args.min_align_corr is None
    assert args.lr_grid_fix is False
    assert args.align_refine == "none"
    assert args.summary_json is None


def test_quality_and_window_flags_default_off():
    args = pipe.build_parser().parse_args(
        ["--email", "x@y.z", "--start", "2010-06-01T00:00:00", "--end", "2010-06-01T02:00:00"])
    assert args.mdi_quality_mask is None
    assert args.strict_window is False


def test_mdi_quality_mask_accepts_hex():
    args = pipe.build_parser().parse_args(
        ["--email", "x@y.z", "--start", "a", "--end", "b", "--mdi_quality_mask", "0x80000000"])
    assert args.mdi_quality_mask == 0x80000000


# ---------------------------------------------------------------- MDI QUALITY filter

class _FakeClient:
    """Stands in for drms.Client: returns a fixed keyword table."""

    def __init__(self, rows):
        import pandas as pd
        self.table = pd.DataFrame(rows, columns=["T_REC", "QUALITY"])

    def query(self, query, key):
        return self.table


_ROWS = [("2011.02.01_00:00:00_TAI", 512), ("2011.02.01_01:36:00_TAI", 0),
         ("2011.02.01_03:12:00_TAI", 0x80000000), ("2011.02.01_04:48:00_TAI", 640)]


def test_mdi_quality_default_keeps_original_rule():
    # original rule: keep QUALITY == 0 if any exist, else keep everything
    got = pipe.query_mdi_series(_FakeClient(_ROWS), "mdi", pipe.datetime(2011, 2, 1), pipe.datetime(2011, 2, 2))
    assert got == ["2011.02.01_01:36:00_TAI"]
    no_zero = [r for r in _ROWS if r[1] != 0]
    got = pipe.query_mdi_series(_FakeClient(no_zero), "mdi", pipe.datetime(2011, 2, 1), pipe.datetime(2011, 2, 2))
    assert len(got) == 3


def test_mdi_quality_mask_drops_only_frames_with_masked_bits():
    got = pipe.query_mdi_series(_FakeClient(_ROWS), "mdi", pipe.datetime(2011, 2, 1),
                                pipe.datetime(2011, 2, 2), quality_mask=0x80000000)
    assert got == ["2011.02.01_00:00:00_TAI", "2011.02.01_01:36:00_TAI", "2011.02.01_04:48:00_TAI"]


# ---------------------------------------------------------------- strict time window

def test_filter_window_drops_records_outside_the_requested_range():
    t = ["2011.02.01_00:00:00_TAI", "2011.02.01_22:24:00_TAI", "2011.02.02_00:00:00_TAI"]
    got = pipe.filter_window(t, pipe.datetime(2011, 2, 1), pipe.datetime(2011, 2, 1, 23, 59, 59))
    assert got == t[:2]


# ---------------------------------------------------------------- LR grid CRPIX

def test_lr_crpix_default_is_the_original_formula():
    assert pipe.lr_crpix(120.5, 4, block_centred=False) == pytest.approx(120.5 / 4)


@pytest.mark.parametrize("crpix_hr", [1.0, 78.0, 120.5, 333.25])
def test_block_centred_lr_pixel_sits_at_centre_of_its_hr_block(crpix_hr):
    s, crval, cdelt = 4, 10.0, 0.03
    crpix_lr = pipe.lr_crpix(crpix_hr, s, block_centred=True)
    for j in (1, 2, 7):                                   # 1-based LR pixel
        hr = np.arange(s * (j - 1) + 1, s * j + 1)        # its 1-based HR pixels
        block_centre = np.mean(crval + cdelt * (hr - crpix_hr))
        lr_centre = crval + s * cdelt * (j - crpix_lr)
        assert lr_centre == pytest.approx(block_centre, abs=1e-12)


# ---------------------------------------------------------------- LOS projection

def test_los_projection_at_disk_centre_is_br():
    blos = pipe.los_projection(bp=3.0, bt=5.0, br=7.0, dlon=0.0, lat=0.0, b0=0.0)
    assert blos == pytest.approx(7.0)


def test_los_projection_at_west_limb_is_minus_bp():
    # at the west limb, westward (+Bp) points away from the observer
    blos = pipe.los_projection(bp=3.0, bt=5.0, br=7.0, dlon=np.pi / 2, lat=0.0, b0=0.0)
    assert blos == pytest.approx(-3.0)


def test_los_projection_at_north_pole_is_plus_bt():
    # at the north pole, southward (+Bt) points toward the observer
    blos = pipe.los_projection(bp=3.0, bt=5.0, br=7.0, dlon=0.0, lat=np.pi / 2, b0=0.0)
    assert blos == pytest.approx(5.0)


# ---------------------------------------------------------------- region centre

@pytest.mark.parametrize("lon,lat,b0,expected", [
    (0.0, 0.0, 0.0, 0.0), (60.0, 0.0, 0.0, 60.0), (0.0, -30.0, 0.0, 30.0), (0.0, 7.0, 7.0, 0.0)])
def test_center_angle(lon, lat, b0, expected):
    assert pipe.center_angle_deg(lon, lat, b0) == pytest.approx(expected, abs=1e-9)


def test_region_center_uses_flux_weighted_centre_when_present():
    meta = {"LON_FWT": 40.2, "LAT_FWT": 21.8, "CRVAL1": 240.9, "CRVAL2": 20.8, "CRLN_OBS": 201.3}
    assert pipe.region_center(meta) == (40.2, 21.8, "fwt")


def test_region_center_falls_back_to_cea_reference_when_fwt_missing():
    meta = {"LON_FWT": np.nan, "LAT_FWT": np.nan, "CRVAL1": 185.17, "CRVAL2": -33.36, "CRLN_OBS": 201.25}
    lon, lat, src = pipe.region_center(meta)
    assert (lon, lat, src) == (pytest.approx(-16.08), pytest.approx(-33.36), "crval")


def test_region_center_wraps_longitude_across_zero():
    meta = {"LON_FWT": None, "LAT_FWT": None, "CRVAL1": 5.0, "CRVAL2": 0.0, "CRLN_OBS": 355.0}
    assert pipe.region_center(meta)[0] == pytest.approx(10.0)


# ---------------------------------------------------------------- block mean / correlation

def test_block_mean_averages_4x4_blocks_and_propagates_nan():
    a = np.arange(64, dtype=float).reshape(8, 8)
    a[0, 0] = np.nan
    out = pipe.block_mean(a, 4)
    assert out.shape == (2, 2)
    assert np.isnan(out[0, 0])
    assert out[1, 1] == pytest.approx(a[4:, 4:].mean())


def test_pearson_finite_ignores_nan_pixels():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 20))
    b = 2 * a + 1
    b[3, 4] = np.nan
    assert pipe.pearson_finite(a, b) == pytest.approx(1.0)


def test_pearson_finite_returns_nan_with_too_few_pixels():
    assert np.isnan(pipe.pearson_finite(np.ones(5), np.ones(5)))


# ---------------------------------------------------------------- alignment refinement

def _smooth_field(n, seed=1):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.normal(size=(n, n)), 3.0)


def _car_wcs(crpix, cdelt, shape):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.crpix = crpix
    w.wcs.cdelt = [cdelt, cdelt]
    w.pixel_shape = shape[::-1]
    return w


@pytest.mark.parametrize("true_dx,true_dy", [(0.75, -1.25), (-1.75, 0.5), (0.0, 0.0)])
def test_refine_alignment_recovers_a_known_shift_with_correct_sign(true_dx, true_dy):
    # "MDI": a smooth field on a fine grid. The LR target is the same field
    # sampled on an LR grid displaced by (true_dx, true_dy) LR pixels.
    from scipy.ndimage import map_coordinates
    fine = _smooth_field(400)
    src_wcs = _car_wcs([200.5, 200.5], 0.01, fine.shape)
    ny, nx = 30, 40
    lr_wcs = _car_wcs([nx / 2 + 0.5, ny / 2 + 0.5], 0.04, (ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = lr_wcs.pixel_to_world_values(xx + true_dx, yy + true_dy)
    sx, sy = src_wcs.world_to_pixel_values(*world)
    target = map_coordinates(fine, [sy, sx], order=3)

    res = pipe.refine_alignment(fine, src_wcs, lr_wcs, (ny, nx), target, max_shift=3.0)

    assert res["shift_dx"] == pytest.approx(true_dx, abs=0.07)
    assert res["shift_dy"] == pytest.approx(true_dy, abs=0.07)
    assert res["at_edge"] is False
    assert res["corr_after"] > 0.99
    assert res["corr_after"] >= res["corr_before"] - 1e-9
    assert res["lr"].shape == (ny, nx)
    assert pipe.pearson_finite(res["lr"], target) == pytest.approx(res["corr_after"])


def test_refine_alignment_flags_a_shift_beyond_the_search_window():
    from scipy.ndimage import map_coordinates
    fine = _smooth_field(400)
    src_wcs = _car_wcs([200.5, 200.5], 0.01, fine.shape)
    ny, nx = 30, 40
    lr_wcs = _car_wcs([nx / 2 + 0.5, ny / 2 + 0.5], 0.04, (ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = lr_wcs.pixel_to_world_values(xx + 2.5, yy)
    sx, sy = src_wcs.world_to_pixel_values(*world)
    target = map_coordinates(fine, [sy, sx], order=3)

    res = pipe.refine_alignment(fine, src_wcs, lr_wcs, (ny, nx), target, max_shift=1.0)
    assert res["at_edge"] is True


# ---------------------------------------------------------------- region files

def test_save_region_round_trip(tmp_path):
    lr = np.random.default_rng(0).normal(size=(20, 30)).astype(np.float32)
    hr = np.random.default_rng(1).normal(size=(3, 80, 120)).astype(np.float32)
    meta = {"harpnum": 36, "mdi_t_rec": "2010.06.01_00:00:00_TAI", "align_corr_los": 0.9}
    path = tmp_path / "r.npz"
    pipe.save_region(str(path), lr, hr, meta)
    d = np.load(path, allow_pickle=False)
    assert d["lr"].shape == (20, 30) and d["hr"].shape == (3, 80, 120)
    assert d["lr"].dtype == np.float32 and d["hr"].dtype == np.float32
    assert int(d["harpnum"]) == 36
    assert str(d["mdi_t_rec"]) == "2010.06.01_00:00:00_TAI"
    assert float(d["align_corr_los"]) == pytest.approx(0.9)


def test_save_region_rejects_shape_mismatch(tmp_path):
    with pytest.raises(ValueError):
        pipe.save_region(str(tmp_path / "r.npz"), np.zeros((20, 30), np.float32),
                         np.zeros((3, 80, 100), np.float32), {})
