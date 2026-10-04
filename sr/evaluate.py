"""
sr/evaluate.py -- full-region inference and every metric, for every model.

    python -m sr.evaluate --run pilot --split val --exps E1 E2 E3_l0.1 ... [--figures]
    python -m sr.evaluate --run pilot --split test --allow_test ...     # Phase 7 only

Models run on FULL regions (all networks are fully convolutional). The LR
region is reflect-padded at the bottom/right up to the model's required
multiple, inferred, and the output cropped back to exactly 4 x (LR size).
Everything is de-normalised (x norm = 3500 G) before any metric.

Untrained baselines (training_plan.md 2, 2b, 2c):
  E0_raw  bicubic x4 MDI compared with Br only (cannot produce Bp, Bt)
  E0_cal  same, with Br ~ a * bicubic + b fitted on the TRAINING split
  E0b     per-channel training mean everywhere
  E0c     calibrated bicubic MDI for Br, training mean for Bp and Bt
Metrics: sr/metrics.py, on two masks: all valid pixels and strong field
(|B_true| > 200 G). The monopole threshold is the 99th percentile of E1's
|div_h error| on VALIDATION; it is computed on --split val and reused later.
Tables: per_region.csv (one row per region x model x mask), summary.csv
(mean, std over regions; mean and 95% bootstrap CI over HARPs), stats_tests.csv
(paired Wilcoxon on per-HARP means, E3 vs E1 and E3 vs E2), eval_info.json.
The test split is refused unless --allow_test is given (Phase 7).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, label
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sr import metrics as M  # noqa: E402
from sr import physics as ph  # noqa: E402
from sr.datasets import _load_pair, split_harps  # noqa: E402
from sr.models import build_model  # noqa: E402

BASELINES = ("E0_raw", "E0_cal", "E0b", "E0c")
KEY_METRICS = ("Br_rmse", "Bp_rmse", "Bt_rmse", "B_rmse", "div_err_g_per_mm", "jz_err_ma_per_m2")


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def infer_region(model, lr):
    """lr (1, 1, H, W) -> (1, 3, 4H, 4W): reflect-pad to the model's multiple, crop back."""
    k = getattr(model, "lr_multiple", 1)
    h, w = lr.shape[-2:]
    ph_, pw = (-h) % k, (-w) % k
    x = F.pad(lr, (0, pw, 0, ph_), mode="reflect") if (ph_ or pw) else lr
    with torch.no_grad():
        out = model(x)
    return out[..., :4 * h, :4 * w]


def padding_interior_change(model, lr, extra=4, margin=24):
    """Max |difference| between the model's output on the region and on the
    region reflect-padded by `extra` LR px on every side, over pixels more than
    `margin` LR px from the region edge."""
    a = infer_region(model, lr)
    b = infer_region(model, F.pad(lr, (extra, extra, extra, extra), mode="reflect"))
    h, w = lr.shape[-2:]
    b = b[..., 4 * extra:4 * (extra + h), 4 * extra:4 * (extra + w)]
    m = 4 * margin
    return float((a - b)[..., m:4 * h - m, m:4 * w - m].abs().max()) if 4 * h > 2 * m and 4 * w > 2 * m \
        else float("nan")


def bicubic(lr):
    t = torch.as_tensor(np.nan_to_num(np.asarray(lr, np.float64)))[None, None]
    return F.interpolate(t, scale_factor=4, mode="bicubic", align_corners=False)[0, 0].numpy()


def fit_calibration(pairs):
    """Least squares Br_true ~ a * bicubic(MDI) + b over valid pixels (training split)."""
    xs, ys = [], []
    for lr, br in pairs:
        up = bicubic(lr)
        h, w = min(up.shape[0], br.shape[0]), min(up.shape[1], br.shape[1])
        up, br = up[:h, :w], br[:h, :w]
        m = np.isfinite(br) & np.isfinite(up)
        xs.append(up[m])
        ys.append(br[m])
    x, y = np.concatenate(xs), np.concatenate(ys)
    (a, b), *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
    return float(a), float(b)


def baseline_prediction(name, lr_gauss, means, calib):
    up = bicubic(lr_gauss)
    a, b = calib
    shape = (3,) + up.shape
    if name == "E0_raw":
        out = np.full(shape, np.nan)
        out[2] = up
    elif name == "E0_cal":
        out = np.full(shape, np.nan)
        out[2] = a * up + b
    elif name == "E0b":
        out = np.broadcast_to(np.asarray(means, float)[:, None, None], shape).copy()
    elif name == "E0c":
        out = np.broadcast_to(np.asarray(means, float)[:, None, None], shape).copy()
        out[2] = a * up + b
    else:
        raise ValueError(name)
    return out


def load_model(results_dir, exp, device):
    ckpt = torch.load(os.path.join(results_dir, exp, "best.pt"), map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]
    model = build_model(cfg["name"], **cfg.get("kwargs", {})).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model, int(ckpt.get("epoch", -1))


# --------------------------------------------------------------------------
# Metrics per region
# --------------------------------------------------------------------------

def region_rows(pred, true, mask, data_range, centers_gauss):
    rows = {}
    for mname, mk in (("all", mask), ("strong", M.strong_mask(true, mask))):
        r = {}
        for ch, vals in M.pixel_metrics(pred, true, mk, data_range).items():
            for k in ("mae", "rmse", "pearson", "psnr", "ssim"):
                r[f"{ch}_{k}"] = vals[k]
        r["n_pixels"] = int(mk.sum())
        for ch, v in M.variance_ratio(pred, true, mk).items():
            r[f"var_ratio_{ch}"] = v
        r.update(M.physics_metrics(pred, true, mk))
        r.update(M.flux_metrics(pred, true, mk))
        if mk.sum() > 10:
            for ch, vals in M.extreme_metrics(pred, true, mk, centers_gauss).items():
                for k, v in vals.items():
                    r[f"{ch}_{k}"] = v
        rows[mname] = r
    return rows


def div_error_map(pred, true, mask):
    dp, dt, v = M._div_maps(pred, true, mask, ph.CDELT_HR_DEG)
    e = np.abs(dp - dt)
    e[~v] = np.nan
    return e.astype(np.float32)


def summarise(per, n_boot, seed=0):
    out = []
    metrics = [c for c in per.columns if c not in ("model", "mask", "region", "harpnum", "mdi_t_rec")]
    for (model, mask), g in per.groupby(["model", "mask"], sort=False):
        for met in metrics:
            vals = g[met].astype(float)
            if vals.notna().sum() == 0:
                continue
            mean_h, lo, hi = M.bootstrap_ci(g, met, n_boot=n_boot, seed=seed)
            out.append({"model": model, "mask": mask, "metric": met, "mean": float(vals.mean()),
                        "std": float(vals.std()), "mean_over_harps": mean_h, "ci_lo": lo, "ci_hi": hi,
                        "n_regions": int(vals.notna().sum()), "n_harps": int(g.loc[vals.notna(), "harpnum"].nunique())})
    return pd.DataFrame(out)


def paired_tests(per, exps):
    out = []
    e3s = [e for e in exps if e.startswith("E3")]
    for a in e3s:
        for b in ("E1", "E2"):
            if b not in exps:
                continue
            for mask in ("all", "strong"):
                for met in KEY_METRICS:
                    x = per[(per.model == a) & (per["mask"] == mask)].groupby("harpnum")[met].mean()
                    y = per[(per.model == b) & (per["mask"] == mask)].groupby("harpnum")[met].mean()
                    d = (x - y).dropna()
                    if len(d) < 2 or (d == 0).all():
                        p = float("nan")
                    else:
                        p = float(wilcoxon(d).pvalue)
                    out.append({"a": a, "b": b, "mask": mask, "metric": met, "n_harps": int(len(d)),
                                "median_diff": float(d.median()) if len(d) else float("nan"), "p_wilcoxon": p})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

def evaluate(run_dir, split_json, split, results_dir, exps, out_dir, device=None, make_figures=True,
             n_boot=10000, allow_test=False, save_predictions=None, monopole_threshold=None):
    if split == "test" and not allow_test:
        raise PermissionError("The test split is for Phase 7 only; pass allow_test=True (--allow_test).")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_predictions = (split == "test") if save_predictions is None else save_predictions
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(run_dir, "stats.json")) as f:
        stats = json.load(f)
    norm = float(stats.get("norm", 3500.0))
    data_range = 2 * np.asarray(stats["ssim_hi"], float) * norm
    centers_gauss = np.asarray(stats["hist_centers"], float) * norm

    regs = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    ev_regs = regs[regs.harpnum.isin(split_harps(split_json, split))].reset_index(drop=True)
    train_regs = regs[regs.harpnum.isin(split_harps(split_json, "train"))]
    train_pairs = [_load_pair(os.path.join(run_dir, f)) for f in train_regs.region_file]
    calib = fit_calibration([(lr * norm, hr[2] * norm) for lr, hr in train_pairs])
    if "channel_stats_gauss" in stats:
        means = np.array([stats["channel_stats_gauss"][c]["mean"] for c in M.CHANNELS])
    else:
        means = np.array([np.nanmean(np.concatenate([hr[c].ravel() for _, hr in train_pairs])) * norm
                          for c in range(3)])

    models = {e: load_model(results_dir, e, device) for e in exps}
    pad_check = {}
    if len(ev_regs) and models:
        lr0, _ = _load_pair(os.path.join(run_dir, ev_regs.region_file[0]))
        t = torch.as_tensor(np.nan_to_num(lr0))[None, None].to(device)
        for e, (m, _) in models.items():
            pad_check[e] = {str(mg): padding_interior_change(m, t, extra=4, margin=mg) for mg in (0, 8, 16, 24, 32)}

    rows, div_maps = [], {}
    for r in ev_regs.itertuples():
        lr, hr = _load_pair(os.path.join(run_dir, r.region_file))
        true = hr * norm
        mask = np.isfinite(true).all(axis=0)
        lr_g = lr * norm
        region = os.path.splitext(os.path.basename(r.region_file))[0]
        preds = {b: baseline_prediction(b, lr_g, means, calib) for b in BASELINES}
        t_lr = torch.as_tensor(np.nan_to_num(lr))[None, None].to(device)
        for e, (m, _) in models.items():
            preds[e] = infer_region(m, t_lr)[0].cpu().numpy().astype(np.float64) * norm
        for name, pred in preds.items():
            for mname, vals in region_rows(pred, true, mask, data_range, centers_gauss).items():
                rows.append({"model": name, "mask": mname, "region": region, "harpnum": int(r.harpnum),
                             "mdi_t_rec": r.mdi_t_rec, **vals})
            div_maps[(name, region)] = div_error_map(pred, true, mask)
            if save_predictions and name in models:
                d = os.path.join(out_dir, "predictions", name)
                os.makedirs(d, exist_ok=True)
                np.savez_compressed(os.path.join(d, f"{region}.npz"), pred=pred.astype(np.float32),
                                    true=true.astype(np.float32), region=region, harpnum=int(r.harpnum),
                                    mdi_t_rec=r.mdi_t_rec)

    thr_path = os.path.join(results_dir, "monopole_threshold.json")
    if monopole_threshold is None:
        if split == "val" and "E1" in models:
            vals = np.concatenate([v[np.isfinite(v)].ravel() for (n, _), v in div_maps.items() if n == "E1"])
            monopole_threshold = float(np.percentile(vals, 99))
            with open(thr_path, "w") as f:
                json.dump({"threshold_g_per_mm": monopole_threshold,
                           "source": "99th percentile of E1 validation |div_h error|"}, f)
        elif os.path.exists(thr_path):
            with open(thr_path) as f:
                monopole_threshold = json.load(f)["threshold_g_per_mm"]
    # Monopole clusters are counted once per (model, region) over all valid
    # pixels (same rule as metrics.monopole_artifacts) and reported on both mask rows.
    counts = {}
    for key, e in div_maps.items():
        if monopole_threshold is None or not np.isfinite(e).any():
            counts[key] = float("nan")
            continue
        hot = np.nan_to_num(e, nan=0.0) > monopole_threshold
        counts[key] = int(label(binary_dilation(hot, np.ones((3, 3))), np.ones((3, 3)))[1]) if hot.any() else 0
    for row in rows:
        row["monopoles"] = counts[(row["model"], row["region"])]

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(out_dir, "per_region.csv"), index=False)
    summary = summarise(per, n_boot)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    paired_tests(per, list(models)).to_csv(os.path.join(out_dir, "stats_tests.csv"), index=False)

    temporal = None
    tpath = os.path.join(run_dir, "noise_floor_temporal.json")
    if os.path.exists(tpath):
        with open(tpath) as f:
            temporal = json.load(f).get("mean_abs_g_per_mm_median_over_pairs")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    info = {"split": split, "n_regions": int(len(ev_regs)), "n_harps": int(ev_regs.harpnum.nunique()),
            "harps": sorted(int(h) for h in ev_regs.harpnum.unique()),
            "models": {e: {"best_epoch": ep} for e, (_, ep) in models.items()},
            "calibration": {"a": calib[0], "b_gauss": calib[1], "fit_on": "training split"},
            "training_means_gauss": dict(zip(M.CHANNELS, means.tolist())),
            "data_range_gauss": dict(zip(M.CHANNELS, data_range.tolist())),
            "padding_check": pad_check, "monopole_threshold_g_per_mm": monopole_threshold,
            "noise_floor_spatial_g_per_mm": (stats.get("noise_floor_spatial_g_per_mm") or {}).get("mean_over_regions"),
            "noise_floor_temporal_g_per_mm": temporal, "git_commit": commit}
    with open(os.path.join(out_dir, "eval_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    if make_figures:
        from sr import figures
        figures.make_all(run_dir, ev_regs, models, out_dir, norm, means, calib, per, summary, device)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--exps", nargs="*", default=[])
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--split_json", default=None)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--allow_test", action="store_true")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    results_dir = args.results_dir or os.path.join(ROOT, "results", args.run)
    info = evaluate(run_dir=os.path.join(args.data_root, args.run),
                    split_json=args.split_json or os.path.join(ROOT, "splits", f"{args.run}.json"),
                    split=args.split, results_dir=results_dir, exps=args.exps,
                    out_dir=args.out_dir or os.path.join(results_dir, f"eval_{args.split}"),
                    device=args.device, make_figures=args.figures, n_boot=args.n_boot,
                    allow_test=args.allow_test)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
