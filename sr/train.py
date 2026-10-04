"""
sr/train.py -- one config-driven training entry point for every experiment.

    python -m sr.train --config configs/pilot_E1.yaml [--lr 5e-5] [--out_dir ...]
                       [--max_epochs N] [--overfit 64] [--device cuda|cpu]

Identical for every model (training_plan.md Phase 5): Adam, batch 64, lr 1e-4,
max 100 epochs, fixed seed, no augmentation. Early stopping (patience 10) and
best.pt selection use ONE loss-independent metric for every model: validation
RMSE in Gauss averaged over Bp, Bt, Br. Every loss term (even zero-weight ones)
is logged each epoch for train and val, with val RMSE per channel and val
divergence error in G/Mm, to metrics.csv and TensorBoard. A non-finite loss
stops the run (status "nonfinite", exit code 3) so the run script can retry
once with half the learning rate.

Outputs in out_dir: best.pt, last.pt, metrics.csv, tb/, config.yaml, meta.json
(status, seed, git commit, parameter counts, stopping epoch, timings).
"""

import argparse
import copy
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sr import physics  # noqa: E402
from sr.datasets import (PatchDataset, RegionCropDataset, RegionGridDataset,  # noqa: E402
                         capped_region_sampler, split_harps)
from sr.losses import CompoundLoss, HistogramLoss, SSIMLoss, LOSS_TERMS  # noqa: E402
from sr.models import build_model, count_params  # noqa: E402

CHANNELS = ("Bp", "Bt", "Br")
DEFAULT_TRAIN = {"batch_size": 64, "lr": 1e-4, "max_epochs": 100, "patience": 10, "seed": 0,
                 "num_workers": 4, "sampler": "auto", "max_share": 0.25}


class NonFiniteLoss(RuntimeError):
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_state():
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD") or None, bool(run("status", "--porcelain", "--", "sr", "pipeline",
                                                        "configs", "scripts"))


def build_loss(loss_cfg, stats):
    ssim = SSIMLoss(lo=stats["ssim_lo"], hi=stats["ssim_hi"])
    centers = stats["hist_centers"]
    if not isinstance(centers[0], (list, tuple)):
        centers = [centers] * 3
    hist = HistogramLoss([torch.tensor(c, dtype=torch.float32) for c in centers],
                         count_budget=64 * 128 * 128, mode=loss_cfg.get("hist_mode", "counts"))
    return CompoundLoss(loss_cfg["weights"], div_mode=loss_cfg.get("div_mode", "match"),
                        ssim=ssim, hist=hist, div_scales=loss_cfg.get("div_scales"),
                        div_scale_norm=loss_cfg.get("div_scale_norm"))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, norm):
    model.eval()
    sums, n = defaultdict(float), 0
    se = torch.zeros(3, dtype=torch.float64)
    npx = 0.0
    div_abs, div_n = 0.0, 0.0
    for batch in loader:
        lr, hr, mask = (batch[k].to(device) for k in ("lr", "hr", "mask"))
        pred = model(lr)
        _, terms = loss_fn(pred, hr, mask)
        for k, v in terms.items():
            sums[k] += v * len(lr)
        n += len(lr)
        m = mask.to(pred.dtype)
        se += (((pred - hr) ** 2) * m).sum(dim=(0, 2, 3)).double().cpu()
        npx += float(m.sum())
        valid = physics.interior_valid(mask[:, 0])
        d = (physics.div_h(pred[:, 0], pred[:, 1]) - physics.div_h(hr[:, 0], hr[:, 1])).abs()
        div_abs += float((d * valid).sum())
        div_n += float(valid.sum())
    out = {f"val_{k}": v / max(n, 1) for k, v in sums.items()}
    rmse = torch.sqrt(se / max(npx, 1.0)).numpy() * norm
    for c, name in enumerate(CHANNELS):
        out[f"val_rmse_{name}"] = float(rmse[c])
    out["val_rmse_mean"] = float(rmse.mean())
    out["val_div_err_g_per_mm"] = div_abs / max(div_n, 1.0) * norm / physics.pixel_scale_mm()
    return out


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def train(cfg, device=None, overfit_n=None):
    cfg = copy.deepcopy(cfg)
    tcfg = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    cfg["train"] = tcfg
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    set_seed(tcfg["seed"])

    with open(cfg["stats"]) as f:
        stats = json.load(f)
    norm = float(stats.get("norm", 3500.0))
    data_mode = cfg.get("data", {}).get("mode", "patches")
    dcfg = cfg.get("data", {})
    geometry, augment = dcfg.get("geometry"), bool(dcfg.get("augment", False))
    in_ch = int(cfg["model"].get("kwargs", {}).get("in_channels", 1))
    if (geometry is not None) != (in_ch == 4):
        raise ValueError(f"geometry channels need model in_channels=4 (got geometry={geometry}, in_channels={in_ch})")
    if data_mode == "regions":   # Phase 8: random crops from full regions, fixed grid crops for validation
        if overfit_n:
            raise ValueError("--overfit needs a patch dataset")
        with open(cfg["split"]) as f:
            largest = json.load(f).get("largest_train_region", {}).get("share_of_train_patches", 0.0)
        region_cap = tcfg["sampler"] == "capped" or (tcfg["sampler"] == "auto" and largest > tcfg["max_share"])
        train_ds = RegionCropDataset(cfg["data_dir"], harps=split_harps(cfg["split"], "train"),
                                     samples_per_epoch=tcfg.get("samples_per_epoch"),
                                     max_share=tcfg["max_share"] if region_cap else None, seed=tcfg["seed"],
                                     geometry_dir=geometry, augment=augment)
        val_ds = RegionGridDataset(cfg["data_dir"], harps=split_harps(cfg["split"], "val"), geometry_dir=geometry,
                                   every=int(dcfg.get("val_every", 1)))
    else:
        train_ds = PatchDataset(cfg["data_dir"], harps=split_harps(cfg["split"], "train"),
                                geometry_dir=geometry, augment=augment, seed=tcfg["seed"])
        val_ds = PatchDataset(cfg["data_dir"], harps=split_harps(cfg["split"], "val"), geometry_dir=geometry)
    train_harps = train_ds.harpnums
    patience = tcfg["patience"]
    if overfit_n:
        overfit_n = min(overfit_n, len(train_ds))
        train_ds = Subset(train_ds, list(range(overfit_n)))
        val_ds = train_ds
        train_harps = train_harps[:overfit_n]
        patience = float("inf")

    with open(cfg["split"]) as f:
        largest = json.load(f).get("largest_train_region", {}).get("share_of_train_patches", 0.0)
    use_cap = data_mode == "patches" and (
        tcfg["sampler"] == "capped" or (tcfg["sampler"] == "auto" and largest > tcfg["max_share"] and not overfit_n))
    g = torch.Generator().manual_seed(tcfg["seed"])
    loader_kw = {"batch_size": tcfg["batch_size"], "num_workers": tcfg["num_workers"],
                 "pin_memory": device.startswith("cuda"),
                 "worker_init_fn": lambda w: np.random.seed(tcfg["seed"] + w)}
    if use_cap:
        train_loader = DataLoader(train_ds, sampler=capped_region_sampler(
            train_harps, tcfg["max_share"], seed=tcfg["seed"]), **loader_kw)
    elif data_mode == "regions":
        train_loader = DataLoader(train_ds, shuffle=False, **loader_kw)   # every index is already a random draw
    else:
        train_loader = DataLoader(train_ds, shuffle=True, generator=g, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = build_model(cfg["model"]["name"], **cfg["model"].get("kwargs", {})).to(device)
    loss_fn = build_loss(cfg["loss"], stats).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    writer = SummaryWriter(os.path.join(out, "tb"))
    commit, dirty = git_state()
    meta = {"exp": cfg.get("exp"), "status": "running", "seed": tcfg["seed"], "lr": tcfg["lr"],
            "git_commit": commit, "git_dirty": dirty, "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
            "params_total": count_params(model), "params_active": count_params(model, active_only=True),
            "n_train": len(train_ds), "n_val": len(val_ds),
            "sampler": ("regions-capped" if region_cap else "regions") if data_mode == "regions"
            else ("capped" if use_cap else "uniform"), "data_mode": data_mode, "overfit_n": overfit_n,
            "geometry": cfg.get("data", {}).get("geometry") is not None,
            "augment": bool(cfg.get("data", {}).get("augment", False)),
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "stopping_epoch": None, "best_epoch": None, "best_val_rmse_mean": None,
            "hit_epoch_cap": None, "first_epoch_seconds": None}
    _write_json(os.path.join(out, "meta.json"), meta)

    metrics_path = os.path.join(out, "metrics.csv")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)   # a fresh run owns this file
    best, bad, t_start = float("inf"), 0, time.time()
    epoch = 0
    for epoch in range(1, tcfg["max_epochs"] + 1):
        t0 = time.time()
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        sums, n = defaultdict(float), 0
        for batch in train_loader:
            lr, hr, mask = (batch[k].to(device, non_blocking=True) for k in ("lr", "hr", "mask"))
            pred = model(lr)
            total, terms = loss_fn(pred, hr, mask)
            if not torch.isfinite(total):
                meta.update(status="nonfinite", stopping_epoch=epoch,
                            end_time=datetime.now().isoformat(timespec="seconds"))
                _write_json(os.path.join(out, "meta.json"), meta)
                writer.close()
                raise NonFiniteLoss(f"non-finite loss at epoch {epoch}: {terms}")
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            for k, v in terms.items():
                sums[k] += v * len(lr)
            n += len(lr)
        row = {"epoch": epoch, "lr": tcfg["lr"]}
        row.update({f"train_{k}": sums[k] / max(n, 1) for k in (*LOSS_TERMS, "total")})
        val = evaluate(model, val_loader, loss_fn, device, norm)
        row.update(val)
        row["epoch_seconds"] = time.time() - t0
        if not all(np.isfinite(v) for k, v in row.items() if isinstance(v, float) and not k.startswith("val_hist")):
            meta.update(status="nonfinite", stopping_epoch=epoch)
            _write_json(os.path.join(out, "meta.json"), meta)
            writer.close()
            raise NonFiniteLoss(f"non-finite metrics at epoch {epoch}")
        if epoch == 1:
            meta["first_epoch_seconds"] = row["epoch_seconds"]
            _write_json(os.path.join(out, "meta.json"), meta)
        with open(metrics_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
        for k, v in row.items():
            if k != "epoch":
                writer.add_scalar(k, v, epoch)
        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": opt.state_dict(), "metrics": row, "config": cfg}
        torch.save(state, os.path.join(out, "last.pt"))
        if row["val_rmse_mean"] < best:
            best, bad = row["val_rmse_mean"], 0
            meta.update(best_epoch=epoch, best_val_rmse_mean=best)
            torch.save(state, os.path.join(out, "best.pt"))
        else:
            bad += 1
        print(f"[{cfg.get('exp')}] epoch {epoch} train_total={row['train_total']:.4g} "
              f"val_rmse(G) Bp={row['val_rmse_Bp']:.1f} Bt={row['val_rmse_Bt']:.1f} "
              f"Br={row['val_rmse_Br']:.1f} mean={row['val_rmse_mean']:.1f} "
              f"div_err={row['val_div_err_g_per_mm']:.1f} G/Mm ({row['epoch_seconds']:.0f}s)", flush=True)
        if bad >= patience:
            break
    writer.close()
    early = bad >= patience
    meta.update(status="finished", stopping_epoch=epoch,
                hit_epoch_cap=bool(epoch == tcfg["max_epochs"] and not early),
                total_seconds=time.time() - t_start,
                end_time=datetime.now().isoformat(timespec="seconds"))
    _write_json(os.path.join(out, "meta.json"), meta)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--overfit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("train", {})
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    try:
        train(cfg, device=args.device, overfit_n=args.overfit)
    except NonFiniteLoss as e:
        print(f"NONFINITE: {e}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
