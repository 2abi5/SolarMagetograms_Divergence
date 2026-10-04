"""Tests for scripts/make_patch_set.py (re-patch saved regions at another patch size)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import make_patch_set as mps  # noqa: E402
from data_active_region_created import extract_patches  # noqa: E402
from sr.datasets import PatchDataset  # noqa: E402


def _region(run_dir, name, ny, nx, harp, seed=0, nan_corner=False):
    rng = np.random.default_rng(seed)
    lr = rng.normal(size=(ny, nx)).astype(np.float32)
    hr = rng.normal(size=(3, 4 * ny, 4 * nx)).astype(np.float32)
    if nan_corner:
        hr[:, :60, :60] = np.nan
    rel = os.path.join("days", "2011-02-01", "regions", f"{name}.npz")
    os.makedirs(os.path.join(run_dir, os.path.dirname(rel)), exist_ok=True)
    np.savez_compressed(os.path.join(run_dir, rel), lr=lr, hr=hr)
    return {"day": "2011-02-01", "region_file": rel, "mdi_t_rec": f"t_{name}", "harpnum": harp,
            "noaa_ar": 0, "lr_ny": ny, "lr_nx": nx}


@pytest.fixture
def src(tmp_path):
    run = tmp_path / "data" / "pilot"
    rows = [_region(str(run), "a", 50, 70, 1, seed=1, nan_corner=True),
            _region(str(run), "b", 40, 33, 2, seed=2),
            _region(str(run), "small", 20, 60, 3, seed=3)]
    pd.DataFrame(rows).to_csv(run / "regions.csv", index=False)
    return run


def test_patches_are_exactly_the_pipeline_patches_at_the_new_size(src, tmp_path):
    dst = tmp_path / "data" / "pilot_p32"
    mps.make_patch_set(str(src), str(dst), patch_size_lr=32, stride_lr=16)
    man = pd.read_csv(dst / "manifest.csv")
    with np.load(src / "days/2011-02-01/regions/a.npz") as d:
        expected = extract_patches(d["lr"], d["hr"], 4, 32, 16)
    got = man[man.harpnum == 1].sort_values(["row", "col"])
    assert len(got) == len(expected) > 0
    for (lr_p, hr_p, row, col), r in zip(expected, got.itertuples()):
        assert (r.row, r.col) == (row, col)
        with np.load(dst / r.patch_file) as d:
            np.testing.assert_array_equal(d["lr"], lr_p)
            np.testing.assert_array_equal(d["hr"], hr_p)
            assert d["lr"].shape == (32, 32) and d["hr"].shape == (3, 128, 128)


def test_regions_smaller_than_the_patch_give_no_patches(src, tmp_path):
    dst = tmp_path / "data" / "pilot_p32"
    info = mps.make_patch_set(str(src), str(dst), patch_size_lr=32, stride_lr=16)
    man = pd.read_csv(dst / "manifest.csv")
    assert 3 not in set(man.harpnum)
    assert info["regions_without_patches"] == 1 and info["regions"] == 3
    assert info["patches"] == len(man)


def test_manifest_is_readable_by_the_training_dataset(src, tmp_path):
    dst = tmp_path / "data" / "pilot_p32"
    mps.make_patch_set(str(src), str(dst), patch_size_lr=32, stride_lr=16)
    ds = PatchDataset(str(dst), harps=[2])
    assert len(ds) > 0 and set(ds.rows.harpnum) == {2}
    item = ds[0]
    assert tuple(item["lr"].shape) == (1, 32, 32) and tuple(item["hr"].shape) == (3, 128, 128)


def test_refuses_to_overwrite_an_existing_patch_set(src, tmp_path):
    dst = tmp_path / "data" / "pilot_p32"
    mps.make_patch_set(str(src), str(dst), patch_size_lr=32, stride_lr=16)
    with pytest.raises(FileExistsError):
        mps.make_patch_set(str(src), str(dst), patch_size_lr=32, stride_lr=16)
