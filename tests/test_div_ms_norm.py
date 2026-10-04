"""Tests for scripts/div_ms_norm.py (per-scale normalisation of the match_ms loss)."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import div_ms_norm as dmn  # noqa: E402
from sr import losses as L  # noqa: E402


def test_mean_square_matches_the_loss_of_a_zero_prediction():
    g = torch.Generator().manual_seed(0)
    hr = torch.randn(6, 3, 64, 64, generator=g, dtype=torch.float64) * 0.05
    mask = torch.ones(6, 1, 64, 64, dtype=torch.bool)
    mask[0, :, :10, :10] = False
    ms = dmn.scale_mean_squares([(hr, mask)], scales=(1.0, 4.0))
    for s, v in zip((1.0, 4.0), ms):
        # a zero prediction's match_ms loss with norm 1 is exactly the mean square of the true smoothed div
        zero = L.divergence_loss(torch.zeros_like(hr), hr, mask, "match_ms", scales=(s,), scale_norm=(1.0,))
        assert v == torch.tensor(zero.item()).item() or abs(v - zero.item()) < 1e-12 * max(1.0, v)


def test_region_batches_give_the_same_statistics_as_the_patches_they_contain(tmp_path):
    import numpy as np
    import pandas as pd
    from sr.datasets import RegionGridDataset
    rng = np.random.default_rng(0)
    run = tmp_path / "run"
    (run / "regions").mkdir(parents=True)
    rows = []
    for k in range(3):
        lr = rng.normal(size=(24, 32)).astype(np.float32)
        hr = (rng.normal(size=(3, 96, 128)) * 0.05).astype(np.float32)
        np.savez_compressed(run / f"regions/r{k}.npz", lr=lr, hr=hr)
        rows.append({"region_file": f"regions/r{k}.npz", "harpnum": k + 1, "mdi_t_rec": f"t{k}", "lr_ny": 24, "lr_nx": 32})
    pd.DataFrame(rows).to_csv(run / "regions.csv", index=False)
    grid = RegionGridDataset(str(run), patch_lr=16, stride=8)
    batches = list(dmn.grid_batches(grid, n=None, seed=0, batch=5))
    hr = torch.cat([b[0] for b in batches]); mask = torch.cat([b[1] for b in batches])
    assert hr.shape == (len(grid), 3, 64, 64)
    direct = dmn.scale_mean_squares([(hr.double(), mask)], scales=(2.0,))
    import pytest
    assert direct == pytest.approx(dmn.scale_mean_squares(dmn.grid_batches(grid, n=None, seed=0, batch=4, double=True),
                                                          scales=(2.0,)), rel=1e-12)
