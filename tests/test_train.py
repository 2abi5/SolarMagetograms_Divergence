"""Tests for sr/train.py on a tiny synthetic dataset (CPU)."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from sr import train as tr


def _make_data(root, n=24, seed=0, bad=False):
    """Synthetic run: blocky HR fields whose three channels are fixed multiples of
    one random field, LR = 4x4 block mean of Br, so HR is a learnable function of LR."""
    rng = np.random.default_rng(seed)
    run = os.path.join(root, "run")
    os.makedirs(os.path.join(run, "patches"))
    rows = []
    for i in range(n):
        base = rng.normal(size=(8, 8)).astype(np.float32) * 0.05
        base = np.stack([0.5 * base, -0.3 * base, base])
        hr = np.kron(base, np.ones((8, 8), np.float32))          # (3, 64, 64) blocky field
        lr = hr[2].reshape(16, 4, 16, 4).mean(axis=(1, 3))
        if bad and i == 0:
            hr[0, 0, 0] = np.inf                                   # corrupt target
            hr[0, 0, 0] = 1e30
        p = f"patches/p{i}.npz"
        np.savez_compressed(os.path.join(run, p), lr=lr, hr=hr)
        rows.append({"patch_file": p, "harpnum": 1 + i % 4, "mdi_t_rec": f"t{i}"})
    pd.DataFrame(rows).to_csv(os.path.join(run, "manifest.csv"), index=False)
    split = {"splits": {"train": {"harps": [1, 2, 3]}, "val": {"harps": [4]}, "test": {"harps": []}},
             "largest_train_region": {"harpnum": 1, "share_of_train_patches": 0.34}}
    with open(os.path.join(root, "split.json"), "w") as f:
        json.dump(split, f)
    stats = {"norm": 3500.0, "ssim_lo": [-0.2] * 3, "ssim_hi": [0.2] * 3,
             "hist_centers": [-0.5, -0.1, -0.02, 0.0, 0.02, 0.1, 0.5]}
    with open(os.path.join(root, "stats.json"), "w") as f:
        json.dump(stats, f)
    return run


def _config(root, **over):
    cfg = {"exp": "T1", "data_dir": os.path.join(root, "run"), "split": os.path.join(root, "split.json"),
           "stats": os.path.join(root, "stats.json"), "out_dir": os.path.join(root, "out"),
           "model": {"name": "edsr", "kwargs": {"n_feats": 8, "n_resblocks": 1}},
           "loss": {"weights": {"mse": 1.0, "grad": 5.0, "hist": 1e-5, "ssim": 5e-5, "div": 0.1},
                    "div_mode": "match"},
           "train": {"batch_size": 8, "lr": 1e-3, "max_epochs": 3, "patience": 10, "seed": 0,
                     "num_workers": 0, "sampler": "auto", "max_share": 0.25}}
    for k, v in over.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and k in cfg else v
    return cfg


def test_training_writes_checkpoints_metrics_and_metadata(tmp_path):
    _make_data(str(tmp_path))
    cfg = _config(str(tmp_path))
    meta = tr.train(cfg, device="cpu")
    out = cfg["out_dir"]
    assert os.path.exists(os.path.join(out, "best.pt")) and os.path.exists(os.path.join(out, "last.pt"))
    m = pd.read_csv(os.path.join(out, "metrics.csv"))
    assert len(m) == 3
    for col in ("train_mse", "train_grad", "train_hist", "train_ssim", "train_div", "train_total",
                "val_mse", "val_div", "val_rmse_Bp", "val_rmse_Bt", "val_rmse_Br", "val_rmse_mean",
                "val_div_err_g_per_mm", "epoch_seconds"):
        assert col in m.columns, col
    saved = json.load(open(os.path.join(out, "meta.json")))
    assert saved["status"] == "finished" and saved == meta
    assert saved["seed"] == 0 and "git_commit" in saved and saved["stopping_epoch"] == 3
    assert saved["params_total"] > 0 and saved["best_val_rmse_mean"] == pytest.approx(m.val_rmse_mean.min())
    assert yaml.safe_load(open(os.path.join(out, "config.yaml")))["exp"] == "T1"


def test_best_checkpoint_is_selected_on_val_rmse_not_on_the_training_loss(tmp_path):
    _make_data(str(tmp_path))
    cfg = _config(str(tmp_path))
    tr.train(cfg, device="cpu")
    m = pd.read_csv(os.path.join(cfg["out_dir"], "metrics.csv"))
    best = torch.load(os.path.join(cfg["out_dir"], "best.pt"), weights_only=False)
    assert best["epoch"] == int(m.loc[m.val_rmse_mean.idxmin(), "epoch"])


def test_early_stopping_uses_patience(tmp_path):
    _make_data(str(tmp_path))
    cfg = _config(str(tmp_path), train={"lr": 0.0, "max_epochs": 20, "patience": 2})
    meta = tr.train(cfg, device="cpu")       # lr 0: val RMSE never improves after epoch 1
    assert meta["stopping_epoch"] == 3 and meta["hit_epoch_cap"] is False


def test_nonfinite_loss_stops_the_run_and_is_recorded(tmp_path):
    _make_data(str(tmp_path), bad=True)
    cfg = _config(str(tmp_path), train={"lr": 1e3})
    with pytest.raises(tr.NonFiniteLoss):
        tr.train(cfg, device="cpu")
    meta = json.load(open(os.path.join(cfg["out_dir"], "meta.json")))
    assert meta["status"] == "nonfinite"


def test_runs_are_reproducible_with_the_same_seed(tmp_path):
    _make_data(str(tmp_path))
    a = tr.train(_config(str(tmp_path), out_dir=str(tmp_path / "a")), device="cpu")
    b = tr.train(_config(str(tmp_path), out_dir=str(tmp_path / "b")), device="cpu")
    assert a["best_val_rmse_mean"] == pytest.approx(b["best_val_rmse_mean"], rel=1e-6)


def test_overfit_mode_drives_the_loss_towards_zero(tmp_path):
    _make_data(str(tmp_path), n=12)           # 9 training patches
    cfg = _config(str(tmp_path), loss={"weights": {"mse": 1.0}},
                  model={"name": "edsr", "kwargs": {"n_feats": 16, "n_resblocks": 2}},
                  train={"lr": 3e-3, "max_epochs": 150, "batch_size": 8})
    meta = tr.train(cfg, device="cpu", overfit_n=8)
    m = pd.read_csv(os.path.join(cfg["out_dir"], "metrics.csv"))
    assert m.train_mse.iloc[-1] < 0.02 * m.train_mse.iloc[0]
    assert meta["overfit_n"] == 8


def test_overfit_n_is_clamped_to_the_training_set(tmp_path):
    _make_data(str(tmp_path), n=8)            # 6 training patches
    cfg = _config(str(tmp_path), train={"max_epochs": 1})
    assert tr.train(cfg, device="cpu", overfit_n=64)["overfit_n"] == 6


# ---------------------------------------------------------------- regions mode (Phase 8: crops cut on the fly)

def _make_region_data(root, n=8, seed=0):
    """Same kind of learnable synthetic field as _make_data, stored as full regions."""
    rng = np.random.default_rng(seed)
    run = os.path.join(root, "run")
    os.makedirs(os.path.join(run, "regions"))
    rows = []
    for i in range(n):
        base = rng.normal(size=(12, 14)).astype(np.float32) * 0.05
        base = np.stack([0.5 * base, -0.3 * base, base])
        hr = np.kron(base, np.ones((8, 8), np.float32))           # (3, 96, 112)
        lr = hr[2].reshape(24, 4, 28, 4).mean(axis=(1, 3))
        p = f"regions/r{i}.npz"
        np.savez_compressed(os.path.join(run, p), lr=lr, hr=hr)
        rows.append({"region_file": p, "harpnum": 1 + i % 4, "mdi_t_rec": f"t{i}", "lr_ny": 24, "lr_nx": 28})
    pd.DataFrame(rows).to_csv(os.path.join(run, "regions.csv"), index=False)
    split = {"splits": {"train": {"harps": [1, 2, 3]}, "val": {"harps": [4]}, "test": {"harps": []}},
             "largest_train_region": {"harpnum": 1, "share_of_train_patches": 0.34}}
    with open(os.path.join(root, "split.json"), "w") as f:
        json.dump(split, f)
    stats = {"norm": 3500.0, "ssim_lo": [-0.2] * 3, "ssim_hi": [0.2] * 3,
             "hist_centers": [-0.5, -0.1, -0.02, 0.0, 0.02, 0.1, 0.5]}
    with open(os.path.join(root, "stats.json"), "w") as f:
        json.dump(stats, f)
    return run


def test_regions_mode_trains_on_random_crops_and_validates_on_the_grid(tmp_path):
    _make_region_data(str(tmp_path))
    cfg = _config(str(tmp_path), data={"mode": "regions"}, train={"samples_per_epoch": 40})
    meta = tr.train(cfg, device="cpu")
    m = pd.read_csv(os.path.join(cfg["out_dir"], "metrics.csv"))
    assert len(m) == 3 and m.val_rmse_mean.notna().all()
    assert meta["n_train"] == 40
    assert meta["n_val"] == 2 * 2 * 2     # 2 val regions x (2 x 2) stride-8 grid crops of a 24 x 28 LR region
    assert meta["sampler"] == "regions-capped"
    assert meta["data_mode"] == "regions"


def test_regions_mode_is_reproducible_with_the_same_seed(tmp_path):
    _make_region_data(str(tmp_path))
    a = _config(str(tmp_path), data={"mode": "regions"}, train={"samples_per_epoch": 40},
                out_dir=os.path.join(str(tmp_path), "a"))
    b = _config(str(tmp_path), data={"mode": "regions"}, train={"samples_per_epoch": 40},
                out_dir=os.path.join(str(tmp_path), "b"))
    tr.train(a, device="cpu")
    tr.train(b, device="cpu")
    ma, mb = (pd.read_csv(os.path.join(c["out_dir"], "metrics.csv")) for c in (a, b))
    np.testing.assert_allclose(ma.train_total, mb.train_total, rtol=1e-5)


def test_patch_mode_is_the_default_and_is_recorded(tmp_path):
    _make_data(str(tmp_path))
    meta = tr.train(_config(str(tmp_path)), device="cpu")
    assert meta["data_mode"] == "patches"


def test_training_with_the_multiscale_divergence_loss_runs_and_logs_it(tmp_path):
    _make_data(str(tmp_path))
    cfg = _config(str(tmp_path), loss={"weights": {"mse": 1.0, "grad": 5.0, "ssim": 5e-5, "div": 0.01},
                                       "div_mode": "match_ms", "div_scales": [1.0, 2.0],
                                       "div_scale_norm": [1e-4, 5e-5]})
    meta = tr.train(cfg, device="cpu")
    m = pd.read_csv(os.path.join(cfg["out_dir"], "metrics.csv"))
    assert meta["status"] == "finished" and np.isfinite(m.train_div).all() and (m.train_div > 0).all()


# ---------------------------------------------------------------- pilot v2: geometry channels + augmentation

def _make_geo_data(root, n=24, seed=0):
    """_make_data plus row/col, a regions.csv and a geometry map per region (4 regions, one per HARP)."""
    run = _make_data(root, n=n, seed=seed)
    man = pd.read_csv(os.path.join(run, "manifest.csv"))
    man["mdi_t_rec"] = "t0"
    man["row"], man["col"] = 0, 0
    man.to_csv(os.path.join(run, "manifest.csv"), index=False)
    os.makedirs(os.path.join(run, "geometry_lr"))
    rng = np.random.default_rng(seed)
    regs = []
    for h in sorted(man.harpnum.unique()):
        f = f"regions/h{h}.npz"
        np.save(os.path.join(run, "geometry_lr", f"h{h}.npy"), rng.uniform(-1, 1, size=(3, 16, 16)).astype(np.float32))
        regs.append({"region_file": f, "harpnum": h, "mdi_t_rec": "t0"})
    pd.DataFrame(regs).to_csv(os.path.join(run, "regions.csv"), index=False)
    return run


def test_training_with_geometry_and_augmentation(tmp_path):
    run = _make_geo_data(str(tmp_path))
    cfg = _config(str(tmp_path), data={"geometry": os.path.join(run, "geometry_lr"), "augment": True},
                  model={"name": "edsr", "kwargs": {"n_feats": 8, "n_resblocks": 1, "in_channels": 4}})
    meta = tr.train(cfg, device="cpu")
    m = pd.read_csv(os.path.join(cfg["out_dir"], "metrics.csv"))
    assert meta["status"] == "finished" and len(m) == 3 and m.val_rmse_mean.notna().all()
    assert meta["geometry"] is True and meta["augment"] is True


def test_geometry_config_needs_a_four_channel_model(tmp_path):
    run = _make_geo_data(str(tmp_path))
    cfg = _config(str(tmp_path), data={"geometry": os.path.join(run, "geometry_lr")})
    with pytest.raises(ValueError):
        tr.train(cfg, device="cpu")


def test_regions_mode_with_geometry_and_augmentation(tmp_path):
    run = _make_region_data(str(tmp_path))
    os.makedirs(os.path.join(run, "geometry_lr"))
    rng = np.random.default_rng(0)
    for f in pd.read_csv(os.path.join(run, "regions.csv")).region_file:
        np.save(os.path.join(run, "geometry_lr", os.path.splitext(os.path.basename(f))[0] + ".npy"),
                rng.uniform(-1, 1, size=(3, 24, 28)).astype(np.float32))
    cfg = _config(str(tmp_path), data={"mode": "regions", "geometry": os.path.join(run, "geometry_lr"), "augment": True},
                  train={"samples_per_epoch": 40},
                  model={"name": "edsr", "kwargs": {"n_feats": 8, "n_resblocks": 1, "in_channels": 4}})
    meta = tr.train(cfg, device="cpu")
    assert meta["status"] == "finished" and meta["geometry"] is True and meta["augment"] is True
    assert meta["data_mode"] == "regions"
