"""Phase 9: SHARP keywords and vertical current of the trained models (reports/phase_9.md).

    python scripts/eval_keywords.py --run full --split val --every 3 --device cuda:1 \
        --exps F_U_E1_s0 F_U_E2_s0 F_U_E3_s0 F_U_E4_s0 F_U_E5_s0 F_E1_s0 F_E2_s0 F_E3_s0 \
        --out_dir results/full/keywords_val_s0

Per frame (every N-th frame of each HARP, as in Phase 8) and model: the SHARP keywords of
sr/sharp_keywords.py from the predicted and from the true (Bp, Bt, Br) on two masks shared by
prediction and truth (all finite pixels; |B_true| > 200 G), with the pipeline's own error
keyword from the stored header; and the correlation of the smoothed vertical current J_z
between prediction and truth per scale and HARP (same recipe as the divergence metric).
Baselines without transverse information: E0c (calibrated bicubic MDI as Br, training means
for Bp and Bt) and E0_mu (radial-field assumption Br = B_LOS / c_r, Bp = Bt = 0).
Writes <out_dir>/per_frame_keywords.csv, jz_by_scale_per_harp.csv and eval_info.json.
The test split is refused unless --allow_test is given (one evaluation, after validation).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from div_snr_by_scale import SIGMAS, smoothed_jz  # noqa: E402
from eval_v2 import _corr, subsample_regions  # noqa: E402
from sr import physics as ph  # noqa: E402
from sr import sharp_keywords as sk  # noqa: E402
from sr.datasets import _load_pair, split_harps  # noqa: E402
from sr.evaluate import baseline_prediction, bicubic, fit_calibration, infer_region, load_model  # noqa: E402
from sr.geometry import geometry_path  # noqa: E402

CHANNELS = ("Bp", "Bt", "Br")
BASELINE_NAMES = ("E0c", "E0_mu")


def shared_masks(pred, true, strong_gauss=200.0):
    """(all, strong): pixels finite in prediction and truth; and those with |B_true| > strong_gauss."""
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    allm = np.isfinite(pred).all(axis=0) & np.isfinite(true).all(axis=0)
    babs = np.sqrt(np.nansum(true ** 2, axis=0))
    return allm, allm & (babs > strong_gauss)


def mu_baseline(lr_gauss, c_r_lr, min_c=0.1):
    """Radial-field assumption: Br = B_LOS / c_r (bicubic x4), Bp = Bt = 0. (3, 4h, 4w) in Gauss."""
    br = bicubic(np.asarray(lr_gauss, np.float64) / np.clip(np.asarray(c_r_lr, np.float64), min_c, None))
    out = np.zeros((3,) + br.shape)
    out[2] = br
    return out


def frame_rows(preds, true, header, cdelt_deg, strong_gauss=200.0):
    """Keyword rows (model, mask, keyword, pred, true, err_header, n_pixels, n_interior) for one frame."""
    rows, truth_cache = [], {}
    for name, pred in preds.items():
        pred = np.where(np.isfinite(true), pred, np.nan)          # same valid pixels as the truth
        allm, strong = shared_masks(pred, true, strong_gauss)
        for mname, mk in (("all", allm), ("strong", strong)):
            kp = sk.keywords(pred, mask=mk, cdelt_deg=cdelt_deg)
            key = (mname, mk.tobytes())
            if key not in truth_cache:
                truth_cache[key] = sk.keywords(true, mask=mk, cdelt_deg=cdelt_deg)
            kt = truth_cache[key]
            for kw in sk.KEYWORDS:
                err = header.get(sk.ERR_KEY[kw]) if kw in sk.ERR_KEY else None
                rows.append({"model": name, "mask": mname, "keyword": kw, "pred": kp[kw], "true": kt[kw],
                             "err_header": float(err) if err is not None else float("nan"),
                             "n_pixels": kp["n_pixels"], "n_interior": kp["n_interior"]})
    return rows


def load_or_fit_calibration(res_dir, run_dir, train_files, norm):
    """Br ~ a * bicubic(MDI) + b on the training split (for E0c); cached in results/<run>/e0_calibration.json."""
    path = os.path.join(res_dir, "e0_calibration.json")
    if os.path.exists(path):
        with open(path) as f:
            c = json.load(f)
        return float(c["a"]), float(c["b_gauss"])
    calib = fit_calibration([(lr * norm, hr[2] * norm) for lr, hr in
                             (_load_pair(os.path.join(run_dir, f)) for f in train_files)])
    with open(path, "w") as f:
        json.dump({"a": calib[0], "b_gauss": calib[1], "fit_on": "training split", "n_regions": len(train_files)}, f, indent=1)
    return calib


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="full")
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--every", type=int, default=3, help="use every N-th frame of each HARP (Phase 8: 3)")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--allow_test", action="store_true", help="required for --split test (one evaluation, after validation)")
    ap.add_argument("--max_regions", type=int, default=0, help="smoke test: only the first N frames")
    ap.add_argument("--strong_gauss", type=float, default=200.0)
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        raise PermissionError("The test split is evaluated once, after validation; pass --allow_test.")
    run_dir, res_dir = os.path.join(ROOT, "data", a.run), os.path.join(ROOT, "results", a.run)
    out_dir = a.out_dir or os.path.join(res_dir, f"keywords_{a.split}")
    os.makedirs(out_dir, exist_ok=True)
    split = os.path.join(ROOT, "splits", f"{a.run}.json")
    with open(os.path.join(run_dir, "stats.json")) as f:
        stats = json.load(f)
    norm = float(stats.get("norm", 3500.0))
    means = np.array([stats["channel_stats_gauss"][c]["mean"] for c in CHANNELS])
    regs = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    train = regs[regs.harpnum.isin(split_harps(split, "train"))]
    calib = load_or_fit_calibration(res_dir, run_dir, list(train.region_file), norm)
    frames = regs[regs.harpnum.isin(split_harps(split, a.split))].reset_index(drop=True)
    frames = subsample_regions(frames, a.every)
    if a.max_regions:
        frames = frames.head(a.max_regions).reset_index(drop=True)
    geo_dir = os.path.join(run_dir, "geometry_lr")
    models = {}
    for e in a.exps:
        with open(os.path.join(res_dir, e, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        model, epoch = load_model(res_dir, e, a.device)
        models[e] = (model, int(cfg["model"].get("kwargs", {}).get("in_channels", 1)), epoch)

    rows, jz_harp = [], {}
    for i, r in enumerate(frames.itertuples()):
        path = os.path.join(run_dir, r.region_file)
        with np.load(path, allow_pickle=False) as d:
            lr, hr = d["lr"].astype(np.float32), d["hr"].astype(np.float32)
            header = json.loads(str(d["sharp_header_json"])) if "sharp_header_json" in d.files else {}
        cdelt = float(header.get("cdelt1", ph.CDELT_HR_DEG))
        true = hr.astype(np.float64) * norm
        geom = np.load(geometry_path(geo_dir, r.region_file))
        preds = {"E0c": baseline_prediction("E0c", lr * norm, means, calib), "E0_mu": mu_baseline(lr * norm, geom[2])}
        x1 = torch.as_tensor(np.nan_to_num(lr))[None, None].to(a.device)
        x4 = torch.as_tensor(np.nan_to_num(np.concatenate([lr[None], geom])))[None].to(a.device)
        with torch.no_grad():
            for e, (m, cin, _) in models.items():
                preds[e] = infer_region(m, x4 if cin == 4 else x1)[0].cpu().numpy().astype(np.float64) * norm
        for row in frame_rows(preds, true, header, cdelt, a.strong_gauss):
            rows.append({"region": os.path.basename(r.region_file), "harpnum": int(r.harpnum),
                         "mdi_t_rec": r.mdi_t_rec, **row})
        finite = np.isfinite(true)
        truth_jz = {s: smoothed_jz(true / norm, s) for s in SIGMAS}
        strong_int = (np.sqrt(np.nansum(true ** 2, axis=0)) > a.strong_gauss)[1:-1, 1:-1]
        for e in models:
            pred = np.where(finite, preds[e], np.nan) / norm
            for s in SIGMAS:
                jt, vt = truth_jz[s]
                jp, vp = smoothed_jz(pred, s)
                for mk, extra in (("all", True), ("strong", strong_int)):
                    sel = vt & vp & extra
                    x, y = jp[sel], jt[sel]
                    jz_harp.setdefault((e, s, mk, int(r.harpnum)), np.zeros(6))[:] += \
                        [x.size, x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]
        if i % 50 == 0:
            print(f"{i}/{len(frames)} frames", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_frame_keywords.csv"), index=False)
    pd.DataFrame([{"model": k[0], "sigma_px": k[1], "mask": k[2], "harpnum": k[3], "n": int(c[0]),
                   "corr_pred_true": _corr(c)} for k, c in jz_harp.items() if c[0] > 10]) \
        .to_csv(os.path.join(out_dir, "jz_by_scale_per_harp.csv"), index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    info = {"split": a.split, "every": a.every, "n_frames": int(len(frames)), "n_harps": int(frames.harpnum.nunique()),
            "models": {e: {"best_epoch": ep, "in_channels": cin} for e, (_, cin, ep) in models.items()},
            "baselines": list(BASELINE_NAMES), "strong_gauss": a.strong_gauss, "keywords": list(sk.KEYWORDS),
            "r_value": {"threshold_g": sk.R_THRESHOLD_G, "bin_px": sk.R_BIN, "fwhm_mm": sk.R_FWHM_MM},
            "jz_sigmas_px": list(SIGMAS), "calibration": {"a": calib[0], "b_gauss": calib[1]},
            "device": a.device, "git_commit": commit, "preregistration": "reports/phase_9.md"}
    with open(os.path.join(out_dir, "eval_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
