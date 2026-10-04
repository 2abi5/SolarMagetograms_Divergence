"""Tests for sr/evaluate.py: full-region inference, baselines, tables, test-split guard."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from sr import evaluate as ev
from sr.models import build_model


def test_pad_and_crop_returns_exactly_four_times_the_region():
    m = build_model("unet").eval()                  # needs LR sizes divisible by 2
    lr = torch.randn(1, 1, 21, 33)
    out = ev.infer_region(m, lr)
    assert out.shape == (1, 3, 84, 132)


def test_padding_does_not_change_the_interior():
    torch.manual_seed(0)
    m = build_model("edsr").eval()
    lr = torch.randn(1, 1, 64, 72)
    # EDSR (8 blocks) sees ~19 LR px around each pixel: beyond that, padding is invisible
    assert ev.padding_interior_change(m, lr, extra=4, margin=24) < 1e-5
    assert ev.padding_interior_change(m, lr, extra=4, margin=0) > 1e-5


def test_calibration_fit_recovers_scale_and_offset():
    rng = np.random.default_rng(0)
    lr = rng.normal(size=(20, 24)) * 300
    up = ev.bicubic(lr)
    br = 1.5 * up + 10 + rng.normal(size=up.shape)
    a, b = ev.fit_calibration([(lr, br)])
    assert a == pytest.approx(1.5, rel=1e-2) and b == pytest.approx(10, abs=0.5)


def test_untrained_baselines():
    lr = np.ones((10, 12)) * 100.0
    means = np.array([1.0, -2.0, 3.0])
    e0 = ev.baseline_prediction("E0_raw", lr, means, (1.0, 0.0))
    assert np.isnan(e0[0]).all() and np.isnan(e0[1]).all() and np.allclose(e0[2], 100.0)
    e0b = ev.baseline_prediction("E0b", lr, means, (1.5, 5.0))
    assert np.allclose(e0b[0], 1.0) and np.allclose(e0b[1], -2.0) and np.allclose(e0b[2], 3.0)
    e0c = ev.baseline_prediction("E0c", lr, means, (1.5, 5.0))
    assert np.allclose(e0c[2], 155.0) and np.allclose(e0c[0], 1.0)
    assert e0b.shape == (3, 40, 48)


# ---------------------------------------------------------------- end to end

def _synthetic_run(root):
    rng = np.random.default_rng(1)
    run = os.path.join(root, "run")
    os.makedirs(os.path.join(run, "regions"))
    rows = []
    for i, h in enumerate([1, 1, 2, 2, 3, 3, 4, 4]):
        base = rng.normal(size=(6, 7)) * 0.05
        hr = np.kron(np.stack([0.5 * base, -0.3 * base, base]), np.ones((8, 8)))    # (3, 48, 56)
        lr = hr[2].reshape(12, 4, 14, 4).mean(axis=(1, 3))
        f = f"regions/r{i}.npz"
        np.savez_compressed(os.path.join(run, f), lr=lr.astype(np.float32), hr=hr.astype(np.float32))
        rows.append({"region_file": f, "harpnum": h, "mdi_t_rec": f"2011.02.0{i + 1}_00:00:00_TAI",
                     "day": "2011-02-01", "n_patches": 4})
    pd.DataFrame(rows).to_csv(os.path.join(run, "regions.csv"), index=False)
    split = {"splits": {"train": {"harps": [1, 2]}, "val": {"harps": [3]}, "test": {"harps": [4]}}}
    with open(os.path.join(root, "split.json"), "w") as f:
        json.dump(split, f)
    stats = {"norm": 3500.0, "ssim_lo": [-0.3] * 3, "ssim_hi": [0.3] * 3,
             "hist_centers": [-0.5, -0.1, 0.0, 0.1, 0.5],
             "noise_floor_spatial_g_per_mm": {"mean_over_regions": 50.0}}
    with open(os.path.join(run, "stats.json"), "w") as f:
        json.dump(stats, f)
    return run


def _checkpoint(root, exp):
    torch.manual_seed(0)
    m = build_model("edsr", n_feats=8, n_resblocks=1)
    out = os.path.join(root, "results", exp)
    os.makedirs(out)
    torch.save({"epoch": 1, "model_state_dict": m.state_dict(),
                "config": {"exp": exp, "model": {"name": "edsr", "kwargs": {"n_feats": 8, "n_resblocks": 1}}}},
               os.path.join(out, "best.pt"))
    return out


def test_evaluate_validation_split_end_to_end(tmp_path):
    run = _synthetic_run(str(tmp_path))
    _checkpoint(str(tmp_path), "E1")
    _checkpoint(str(tmp_path), "E3")
    out = str(tmp_path / "eval_val")
    ev.evaluate(run_dir=run, split_json=str(tmp_path / "split.json"), split="val",
                results_dir=str(tmp_path / "results"), exps=["E1", "E3"], out_dir=out, device="cpu",
                make_figures=False, n_boot=200)
    per = pd.read_csv(os.path.join(out, "per_region.csv"))
    assert set(per.model) == {"E0_raw", "E0_cal", "E0b", "E0c", "E1", "E3"}
    assert set(per["mask"]) == {"all", "strong"}
    assert (per.harpnum == 3).all() and per.region.nunique() == 2
    for col in ("Br_rmse", "Bp_rmse", "B_ssim", "div_err_g_per_mm", "jz_err_ma_per_m2",
                "var_ratio_Bp", "div_std_ratio", "signed_flux_imbalance", "Br_hist_tv", "monopoles"):
        assert col in per.columns, col
    assert per[per.model == "E0_raw"].Bp_rmse.isna().all()          # bicubic predicts Br only
    summary = pd.read_csv(os.path.join(out, "summary.csv"))
    assert {"model", "mask", "metric", "mean", "std", "ci_lo", "ci_hi", "n_harps"} <= set(summary.columns)
    info = json.load(open(os.path.join(out, "eval_info.json")))
    assert info["split"] == "val" and "calibration" in info and "padding_check" in info
    assert info["noise_floor_spatial_g_per_mm"] == 50.0


def test_test_split_requires_explicit_permission(tmp_path):
    run = _synthetic_run(str(tmp_path))
    with pytest.raises(PermissionError):
        ev.evaluate(run_dir=run, split_json=str(tmp_path / "split.json"), split="test",
                    results_dir=str(tmp_path / "results"), exps=[], out_dir=str(tmp_path / "e"),
                    device="cpu", make_figures=False)
