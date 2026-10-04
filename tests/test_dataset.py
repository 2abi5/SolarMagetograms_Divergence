"""Dataset tests (training_plan.md Phase 4 + sampling balance, section 2b)."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from sr import datasets as ds


def _make_run(tmp_path, n_per_harp={1: 6, 2: 3}, with_nan=True):
    run = tmp_path / "run"
    (run / "patches").mkdir(parents=True)
    (run / "regions").mkdir()
    rng = np.random.default_rng(0)
    rows, regs = [], []
    for h, n in n_per_harp.items():
        for i in range(n):
            lr = rng.normal(size=(16, 16)).astype(np.float32) * 0.05
            hr = rng.normal(size=(3, 64, 64)).astype(np.float32) * 0.05
            if with_nan and i == 0:
                lr[0, 0] = np.nan
                hr[1, 5, 7] = np.nan
            p = f"patches/h{h}_{i}.npz"
            np.savez_compressed(run / p, lr=lr, hr=hr)
            rows.append({"patch_file": p, "harpnum": h, "mdi_t_rec": f"t{i}"})
        lr = rng.normal(size=(20, 28)).astype(np.float32)
        hr = rng.normal(size=(3, 80, 112)).astype(np.float32)
        hr[0, 0, 0] = np.nan
        r = f"regions/h{h}.npz"
        np.savez_compressed(run / r, lr=lr, hr=hr, harpnum=np.asarray(h), mdi_t_rec=np.asarray("t0"))
        regs.append({"region_file": r, "harpnum": h, "mdi_t_rec": "t0", "n_patches": n})
    pd.DataFrame(rows).to_csv(run / "manifest.csv", index=False)
    pd.DataFrame(regs).to_csv(run / "regions.csv", index=False)
    split = {"splits": {"train": {"harps": [1]}, "val": {"harps": [2]}, "test": {"harps": []}}}
    (tmp_path / "split.json").write_text(json.dumps(split))
    return str(run), str(tmp_path / "split.json")


@pytest.mark.parametrize("cache", [True, False])
def test_patch_item_shapes_and_finite_after_masking(tmp_path, cache):
    run, _ = _make_run(tmp_path)
    d = ds.PatchDataset(run, cache=cache)
    item = d[0]                                   # the patch that contains NaNs
    assert item["lr"].shape == (1, 16, 16) and item["hr"].shape == (3, 64, 64)
    assert item["mask"].shape == (1, 64, 64) and item["mask"].dtype == torch.bool
    assert torch.isfinite(item["lr"]).all() and torch.isfinite(item["hr"]).all()
    assert not item["mask"][0, 5, 7] and item["mask"].sum() == 64 * 64 - 1
    assert not item["lr_mask"][0, 0, 0]
    assert item["hr"][1, 5, 7] == 0 and item["lr"][0, 0, 0] == 0


def test_patch_dataset_filters_by_split(tmp_path):
    run, split = _make_run(tmp_path)
    assert len(ds.PatchDataset(run, harps=ds.split_harps(split, "train"))) == 6
    assert len(ds.PatchDataset(run, harps=ds.split_harps(split, "val"))) == 3


def test_region_dataset_returns_full_region_with_mask(tmp_path):
    run, split = _make_run(tmp_path)
    d = ds.RegionDataset(run, harps=ds.split_harps(split, "train"))
    item = d[0]
    assert item["lr"].shape == (1, 20, 28) and item["hr"].shape == (3, 80, 112)
    assert torch.isfinite(item["hr"]).all() and not item["mask"][0, 0, 0]
    assert item["harpnum"] == 1


def test_capped_sampler_limits_the_largest_region_share():
    # region 1 has 80% of patches; ten small regions share the rest
    harps = np.array([1] * 800 + [2] * 30 + [3] * 20 + sum(([h] * 20 for h in range(4, 12)), []))
    w = ds.capped_region_weights(harps, max_share=0.25)
    share = pd.Series(w).groupby(harps).sum() / w.sum()
    assert share[1] == pytest.approx(0.25, abs=1e-9)
    assert share[2] / share[3] == pytest.approx(30 / 20)      # others keep their proportions
    assert share.max() <= 0.25 + 1e-9
    assert share.sum() == pytest.approx(1.0)


def test_capped_sampler_caps_iteratively_when_redistribution_overflows():
    # capping region 1 pushes region 2 over the cap too: both end at the cap
    harps = np.array([1] * 800 + [2] * 150 + sum(([h] * 10 for h in range(3, 8)), []))
    w = ds.capped_region_weights(harps, max_share=0.25)
    share = pd.Series(w).groupby(harps).sum() / w.sum()
    assert share[1] == pytest.approx(0.25) and share[2] == pytest.approx(0.25)
    assert share.max() <= 0.25 + 1e-9


def test_capped_sampler_is_uniform_when_no_region_dominates():
    harps = np.array([1] * 100 + [2] * 90 + [3] * 110 + [4] * 100)
    w = ds.capped_region_weights(harps, max_share=0.25 + 1e-9 + 110 / 400)
    assert np.allclose(w, w[0])


# ---------------------------------------------------------------- pilot v2: geometry channels + augmentation

def _make_geo_run(tmp_path):
    """Region (20 x 28 LR) cut into 3 patches at known offsets, plus a geometry map for it."""
    run = tmp_path / "run"
    (run / "patches").mkdir(parents=True)
    (run / "regions").mkdir()
    (run / "geometry_lr").mkdir()
    rng = np.random.default_rng(0)
    lr = rng.normal(size=(20, 28)).astype(np.float32)
    hr = rng.normal(size=(3, 80, 112)).astype(np.float32)
    np.savez_compressed(run / "regions/r.npz", lr=lr, hr=hr)
    geo_map = rng.uniform(-1, 1, size=(3, 20, 28)).astype(np.float32)
    np.save(run / "geometry_lr/r.npy", geo_map)
    rows = []
    for k, (row, col) in enumerate([(0, 0), (4, 8), (2, 12)]):
        p = f"patches/p{k}.npz"
        np.savez_compressed(run / p, lr=lr[row:row + 16, col:col + 16], hr=hr[:, 4 * row:4 * row + 64, 4 * col:4 * col + 64])
        rows.append({"patch_file": p, "harpnum": 1, "mdi_t_rec": "t0", "row": row, "col": col})
    pd.DataFrame(rows).to_csv(run / "manifest.csv", index=False)
    pd.DataFrame([{"region_file": "regions/r.npz", "harpnum": 1, "mdi_t_rec": "t0"}]).to_csv(run / "regions.csv", index=False)
    return str(run), lr, hr, geo_map


def test_geometry_channels_are_the_region_map_at_the_patch_position(tmp_path):
    run, lr, hr, geo_map = _make_geo_run(tmp_path)
    d = ds.PatchDataset(run, geometry_dir=os.path.join(run, "geometry_lr"))
    for i, (row, col) in enumerate([(0, 0), (4, 8), (2, 12)]):
        item = d[i]
        assert item["lr"].shape == (4, 16, 16)
        np.testing.assert_array_equal(item["lr"][0].numpy(), lr[row:row + 16, col:col + 16])
        np.testing.assert_array_equal(item["lr"][1:].numpy(), geo_map[:, row:row + 16, col:col + 16])
        assert item["lr_mask"].shape == (1, 16, 16)


def test_default_patch_items_are_unchanged(tmp_path):
    run, lr, hr, _ = _make_geo_run(tmp_path)
    item = ds.PatchDataset(run)[1]
    assert item["lr"].shape == (1, 16, 16)
    np.testing.assert_array_equal(item["lr"][0].numpy(), lr[4:20, 8:24])


def test_augmentation_is_deterministic_per_epoch_and_consistent_with_geometry(tmp_path):
    from sr import geometry as geo
    run, lr, hr, geo_map = _make_geo_run(tmp_path)
    kw = dict(geometry_dir=os.path.join(run, "geometry_lr"), augment=True, seed=5)
    a, b = ds.PatchDataset(run, **kw), ds.PatchDataset(run, **kw)
    items = lambda d: [d[i] for i in range(3)]
    ia, ib = items(a), items(b)
    for x, y in zip(ia, ib):
        assert torch.equal(x["lr"], y["lr"]) and torch.equal(x["hr"], y["hr"])
    changed = False
    for epoch in range(1, 6):
        a.set_epoch(epoch)
        changed |= any(not torch.equal(x["hr"], y["hr"]) for x, y in zip(items(a), ib))
    assert changed
    # every augmented item is one of the 8 flag combinations applied to the raw patch
    raw_lr = np.concatenate([lr[None, 4:20, 8:24], geo_map[:, 4:20, 8:24]])
    raw_hr = hr[:, 16:80, 32:96]
    got = ia[1]
    combos = [geo.augment(raw_lr, raw_hr, f, mx, my) for f in (0, 1) for mx in (0, 1) for my in (0, 1)]
    assert any(np.allclose(got["lr"].numpy(), l) and np.allclose(got["hr"].numpy(), h) for l, h in combos)


def test_augmentation_is_off_by_default(tmp_path):
    run, *_ = _make_geo_run(tmp_path)
    d = ds.PatchDataset(run, geometry_dir=os.path.join(run, "geometry_lr"))
    first = d[0]["hr"].clone()
    d.set_epoch(3)
    assert torch.equal(d[0]["hr"], first)
