"""Tests for on-the-fly crops from full regions (Phase 8: --save_mode regions)."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from data_active_region_created import extract_patches
from sr import datasets as ds


def _region(run_dir, name, ny, nx, harp, seed=0, nan_block=None):
    rng = np.random.default_rng(seed)
    lr = rng.normal(size=(ny, nx)).astype(np.float32)
    hr = rng.normal(size=(3, 4 * ny, 4 * nx)).astype(np.float32)
    if nan_block is not None:
        r0, r1, c0, c1 = nan_block
        hr[:, r0:r1, c0:c1] = np.nan
        lr[r0 // 4:r1 // 4, c0 // 4:c1 // 4] = np.nan
    rel = os.path.join("days", "2011-02-01", "regions", f"{name}.npz")
    os.makedirs(os.path.join(run_dir, os.path.dirname(rel)), exist_ok=True)
    np.savez_compressed(os.path.join(run_dir, rel), lr=lr, hr=hr)
    return {"region_file": rel, "mdi_t_rec": f"t_{name}", "harpnum": harp, "lr_ny": ny, "lr_nx": nx}


@pytest.fixture
def run(tmp_path):
    rows = [_region(str(tmp_path), "a", 30, 45, 1, seed=1, nan_block=(0, 40, 0, 48)),
            _region(str(tmp_path), "b", 24, 20, 2, seed=2),
            _region(str(tmp_path), "c", 40, 60, 3, seed=3, nan_block=(60, 100, 100, 160)),
            _region(str(tmp_path), "tiny", 10, 30, 4, seed=4)]
    pd.DataFrame(rows).to_csv(tmp_path / "regions.csv", index=False)
    return tmp_path


def _brute_force_positions(lr, hr, p, min_finite=0.95):
    out = []
    for r in range(lr.shape[0] - p + 1):
        for c in range(lr.shape[1] - p + 1):
            if (np.isfinite(lr[r:r + p, c:c + p]).mean() >= min_finite
                    and np.isfinite(hr[:, 4 * r:4 * r + 4 * p, 4 * c:4 * c + 4 * p]).mean() >= min_finite):
                out.append((r, c))
    return np.array(out).reshape(-1, 2)


def test_valid_crop_positions_follow_the_pipeline_rule_at_every_offset(run):
    for name in ("a", "c"):
        lr, hr = ds._load_pair(str(run / "days/2011-02-01/regions" / f"{name}.npz"))
        got = ds.valid_crop_positions(lr, hr, 16)
        np.testing.assert_array_equal(got, _brute_force_positions(lr, hr, 16))


def test_grid_dataset_reproduces_the_pipeline_patches(run):
    grid = ds.RegionGridDataset(str(run), patch_lr=16, stride=8)
    got = {(int(grid[i]["harpnum"]), int(grid[i]["row"]), int(grid[i]["col"])): grid[i] for i in range(len(grid))}
    n_expected = 0
    for name, harp in (("a", 1), ("b", 2), ("c", 3), ("tiny", 4)):
        lr, hr = ds._load_pair(str(run / "days/2011-02-01/regions" / f"{name}.npz"))
        for lr_p, hr_p, r, c in extract_patches(lr, hr, 4, 16, 8):
            n_expected += 1
            item = got[(harp, r, c)]
            np.testing.assert_array_equal(item["lr"][0].numpy(), np.nan_to_num(lr_p))
            np.testing.assert_array_equal(item["hr"].numpy(), np.nan_to_num(hr_p))
            np.testing.assert_array_equal(item["mask"][0].numpy(), np.isfinite(hr_p).all(axis=0))
    assert len(grid) == n_expected


def test_random_crops_are_valid_slices_of_their_region(run):
    crops = ds.RegionCropDataset(str(run), patch_lr=16, samples_per_epoch=200, seed=0)
    assert len(crops) == 200
    regions = {r.harpnum: ds._load_pair(str(run / r.region_file)) for r in crops.rows.itertuples()}
    for i in range(0, 200, 7):
        it = crops[i]
        assert tuple(it["lr"].shape) == (1, 16, 16) and tuple(it["hr"].shape) == (3, 64, 64)
        lr, hr = regions[int(it["harpnum"])]
        r, c = int(it["row"]), int(it["col"])
        np.testing.assert_array_equal(it["lr"][0].numpy(), np.nan_to_num(lr[r:r + 16, c:c + 16]))
        np.testing.assert_array_equal(it["hr"].numpy(), np.nan_to_num(hr[:, 4 * r:4 * r + 64, 4 * c:4 * c + 64]))
        assert np.isfinite(hr[:, 4 * r:4 * r + 64, 4 * c:4 * c + 64]).mean() >= 0.95
    assert 4 not in set(crops.rows.harpnum)   # too small for one crop


def test_crops_are_deterministic_per_epoch_and_change_between_epochs(run):
    a = ds.RegionCropDataset(str(run), patch_lr=16, samples_per_epoch=50, seed=0)
    b = ds.RegionCropDataset(str(run), patch_lr=16, samples_per_epoch=50, seed=0)
    key = lambda d: [(int(d[i]["harpnum"]), int(d[i]["row"]), int(d[i]["col"])) for i in range(len(d))]
    assert key(a) == key(b)
    first = key(a)
    a.set_epoch(1)
    assert key(a) != first


def test_default_epoch_size_is_the_number_of_grid_patches(run):
    crops = ds.RegionCropDataset(str(run), patch_lr=16)
    assert len(crops) == len(ds.RegionGridDataset(str(run), patch_lr=16, stride=8))


def test_regions_are_drawn_in_proportion_to_their_valid_positions(run):
    crops = ds.RegionCropDataset(str(run), patch_lr=16, samples_per_epoch=20000, seed=1)
    n_pos = np.array([len(p) for p in crops.positions], float)
    expected = dict(zip(crops.rows.harpnum, n_pos / n_pos.sum()))
    drawn = pd.Series([int(crops.draw(i)[0]) for i in range(20000)]).map(dict(enumerate(crops.rows.harpnum)))
    share = drawn.value_counts(normalize=True)
    for h, p in expected.items():
        assert share.get(h, 0.0) == pytest.approx(p, abs=0.02)


def test_harp_cap_limits_the_largest_harp_share(run):
    crops = ds.RegionCropDataset(str(run), patch_lr=16, samples_per_epoch=20000, seed=2, max_share=0.4)
    drawn = pd.Series([int(crops.draw(i)[0]) for i in range(20000)]).map(dict(enumerate(crops.rows.harpnum)))
    assert drawn.value_counts(normalize=True).max() <= 0.4 + 0.02


# ---------------------------------------------------------------- Phase 8 v2: geometry channels + augmentation

@pytest.fixture
def geo_run(run):
    """The regions of `run` plus a random geometry map per region."""
    rng = np.random.default_rng(9)
    os.makedirs(run / "geometry_lr", exist_ok=True)
    regs = pd.read_csv(run / "regions.csv")
    maps = {}
    for r in regs.itertuples():
        m = rng.uniform(-1, 1, size=(3, r.lr_ny, r.lr_nx)).astype(np.float32)
        np.save(run / "geometry_lr" / (os.path.splitext(os.path.basename(r.region_file))[0] + ".npy"), m)
        maps[r.harpnum] = m
    return run, maps


def test_random_crops_carry_the_geometry_of_their_position(geo_run):
    run, maps = geo_run
    crops = ds.RegionCropDataset(str(run), patch_lr=16, samples_per_epoch=60, seed=0,
                                 geometry_dir=str(run / "geometry_lr"))
    for i in range(0, 60, 7):
        it = crops[i]
        r, c = int(it["row"]), int(it["col"])
        assert tuple(it["lr"].shape) == (4, 16, 16)
        np.testing.assert_array_equal(it["lr"][1:].numpy(), maps[int(it["harpnum"])][:, r:r + 16, c:c + 16])


def test_grid_crops_carry_the_geometry(geo_run):
    run, maps = geo_run
    grid = ds.RegionGridDataset(str(run), patch_lr=16, stride=8, geometry_dir=str(run / "geometry_lr"))
    it = grid[len(grid) // 2]
    r, c = int(it["row"]), int(it["col"])
    np.testing.assert_array_equal(it["lr"][1:].numpy(), maps[int(it["harpnum"])][:, r:r + 16, c:c + 16])


def test_region_crop_augmentation_is_one_of_the_eight_flip_combinations(geo_run):
    from sr import geometry as geo
    run, maps = geo_run
    kw = dict(patch_lr=16, samples_per_epoch=40, seed=3, geometry_dir=str(run / "geometry_lr"))
    plain = ds.RegionCropDataset(str(run), **kw)
    aug = ds.RegionCropDataset(str(run), augment=True, **kw)
    changed = 0
    for i in range(40):
        p, a = plain[i], aug[i]
        assert (int(p["row"]), int(p["col"]), int(p["harpnum"])) == (int(a["row"]), int(a["col"]), int(a["harpnum"]))
        combos = [geo.augment(p["lr"].numpy(), p["hr"].numpy(), f, mx, my) for f in (0, 1) for mx in (0, 1) for my in (0, 1)]
        assert any(np.allclose(a["lr"].numpy(), l) and np.allclose(a["hr"].numpy(), h) for l, h in combos)
        changed += not torch.equal(a["hr"], p["hr"])
    assert changed > 20                                   # 7/8 of the combinations change the patch
    again = ds.RegionCropDataset(str(run), augment=True, **kw)
    assert all(torch.equal(again[i]["hr"], aug[i]["hr"]) for i in range(10))


def test_grid_dataset_can_use_every_nth_frame_of_each_harp(tmp_path):
    rows = []
    for k in range(6):          # HARP 1: frames t0..t3, HARP 2: frames t0..t1
        h, t = (1, k) if k < 4 else (2, k - 4)
        rows.append(_region(str(tmp_path), f"r{k}", 20, 20, h, seed=k) | {"mdi_t_rec": f"t{t}"})
    pd.DataFrame(rows).to_csv(tmp_path / "regions.csv", index=False)
    full = ds.RegionGridDataset(str(tmp_path), patch_lr=16, stride=8)
    third = ds.RegionGridDataset(str(tmp_path), patch_lr=16, stride=8, every=2)
    assert sorted(zip(third.rows.harpnum, third.rows.mdi_t_rec)) == [(1, "t0"), (1, "t2"), (2, "t0")]
    assert len(third) == len(full) // 6 * 3
