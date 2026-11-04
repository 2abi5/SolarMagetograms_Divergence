"""Overlap-pilot stage E1: SHARP-keyword time series of the frozen models on ALL 233 overlap ARs
(reports/phase_12.md 3). Every frame of every HARP; no training, selection or tuning happens here.

    python scripts/c23_features.py --run full --device cuda:1 --shard 0/6 \
        --exps F_U_E3_s0_e200 F_U_E3_s1_e200 F_U_E3_s2 F_U_E1_s0 F_U_E1_s1 F_U_E1_s2_e200 \
        --out_dir results/full/overlap_pilot

Per frame and model one WIDE row per mask variant with the 13 keywords of sr/sharp_keywords.py,
n_pixels and n_interior, plus frame metadata (harpnum, t_recs, noaa_ar, center angle/lon, month).
Mask variants per model: "all" and "strong" (|B_true| > 200 G) as in Phase 9, both on the truth's
valid pixels; "strong_pred" (|B_pred| > 200 G on the prediction's own pixels) = the deployment
condition of a Cycle-23 record, where no truth exists. Truth rows (model "true_hmi"): all, strong.
Baselines E0c and E0_mu as in Phase 9. Shards split the sorted HARP list round-robin (k/n);
each shard writes features_shard<k>of<n>.csv; `merge` concatenates shards and writes extract_info.json.
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
from eval_keywords import load_or_fit_calibration, mu_baseline  # noqa: E402
from sr import physics as ph  # noqa: E402
from sr import sharp_keywords as sk  # noqa: E402
from sr.evaluate import baseline_prediction, infer_region, load_model  # noqa: E402
from sr.geometry import geometry_path  # noqa: E402

CHANNELS = ("Bp", "Bt", "Br")
META = ("harpnum", "mdi_t_rec", "sharp_t_rec", "noaa_ar", "center_angle", "center_lon", "month")


def shard_harps(harps, shard):
    """Round-robin split of the sorted HARP list: 'k/n' -> the k-th of n shards."""
    k, n = (int(x) for x in shard.split("/"))
    if not 0 <= k < n:
        raise ValueError(f"shard {shard}: need 0 <= k < n")
    return sorted(harps)[k::n]


def frame_rows(preds, true, cdelt_deg, strong_gauss=200.0):
    """One wide row per (model, mask variant) for one frame; truth rows under model 'true_hmi'."""
    true = np.asarray(true, np.float64)
    tfin = np.isfinite(true).all(axis=0)
    tabs = np.sqrt(np.nansum(true ** 2, axis=0))
    rows = [{"model": "true_hmi", "mask": m, "n_finite": int(tfin.sum()),
             **sk.keywords(true, mask=mk, cdelt_deg=cdelt_deg)}
            for m, mk in (("all", tfin), ("strong", tfin & (tabs > strong_gauss)))]
    for name, pred in preds.items():
        pred = np.asarray(pred, np.float64)
        pfin = np.isfinite(pred).all(axis=0)
        pabs = np.sqrt(np.nansum(np.where(pfin, pred, 0.0) ** 2, axis=0))
        on_truth = np.where(tfin, pred, np.nan)           # Phase 9 convention: truth's valid pixels
        for m, arr, mk, nf in (("all", on_truth, tfin, tfin),
                               ("strong", on_truth, tfin & (tabs > strong_gauss), tfin),
                               ("strong_pred", pred, pfin & (pabs > strong_gauss), pfin)):
            rows.append({"model": name, "mask": m, "n_finite": int(nf.sum()),
                         **sk.keywords(arr, mask=mk, cdelt_deg=cdelt_deg)})
    return rows


def extract(frames, run_dir, models, calib, means, norm, device, strong_gauss=200.0, log=print):
    """All wide rows for the given frames (a DataFrame slice of regions.csv)."""
    geo_dir = os.path.join(run_dir, "geometry_lr")
    out = []
    for i, r in enumerate(frames.itertuples()):
        with np.load(os.path.join(run_dir, r.region_file), allow_pickle=False) as d:
            lr, hr = d["lr"].astype(np.float32), d["hr"].astype(np.float32)
        true = hr.astype(np.float64) * norm
        geom = np.load(geometry_path(geo_dir, r.region_file))
        preds = {"E0c": baseline_prediction("E0c", lr * norm, means, calib),
                 "E0_mu": mu_baseline(lr * norm, geom[2])}
        x1 = torch.as_tensor(np.nan_to_num(lr))[None, None].to(device)
        x4 = torch.as_tensor(np.nan_to_num(np.concatenate([lr[None], geom])))[None].to(device)
        with torch.no_grad():
            for e, (m, cin, _) in models.items():
                preds[e] = infer_region(m, x4 if cin == 4 else x1)[0].cpu().numpy().astype(np.float64) * norm
        meta = {"harpnum": int(r.harpnum), "mdi_t_rec": r.mdi_t_rec, "sharp_t_rec": r.sharp_t_rec,
                "noaa_ar": int(r.noaa_ar), "center_angle": float(r.center_angle),
                "center_lon": float(r.center_lon), "month": str(r.sharp_t_rec)[:7]}
        for row in frame_rows(preds, true, ph.CDELT_HR_DEG, strong_gauss):
            out.append({**meta, **row})
        if i % 200 == 0:
            log(f"{i}/{len(frames)} frames")
    return pd.DataFrame(out)


def cmd_extract(a):
    run_dir, res_dir = os.path.join(ROOT, "data", a.run), os.path.join(ROOT, "results", a.run)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(run_dir, "stats.json")) as f:
        stats = json.load(f)
    norm = float(stats.get("norm", 3500.0))
    means = np.array([stats["channel_stats_gauss"][c]["mean"] for c in CHANNELS])
    regs = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    from sr.datasets import split_harps
    split = os.path.join(ROOT, "splits", f"{a.run}.json")
    train = regs[regs.harpnum.isin(split_harps(split, "train"))]
    calib = load_or_fit_calibration(res_dir, run_dir, list(train.region_file), norm)
    harps = shard_harps(regs.harpnum.unique().tolist(), a.shard)
    frames = regs[regs.harpnum.isin(harps)].reset_index(drop=True)
    if a.max_regions:
        frames = frames.head(a.max_regions).reset_index(drop=True)
    models = {}
    for e in a.exps:
        with open(os.path.join(res_dir, e, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        model, epoch = load_model(res_dir, e, a.device)
        models[e] = (model, int(cfg["model"].get("kwargs", {}).get("in_channels", 1)), epoch)
    k = a.shard.replace("/", "of")
    df = extract(frames, run_dir, models, calib, means, norm, a.device, a.strong_gauss)
    df.to_csv(os.path.join(a.out_dir, f"features_shard{k}.csv"), index=False)
    with open(os.path.join(a.out_dir, f"shard{k}_info.json"), "w") as f:
        json.dump({"shard": a.shard, "n_frames": int(len(frames)), "n_harps": int(frames.harpnum.nunique()),
                   "models": {e: {"best_epoch": ep, "in_channels": c} for e, (_, c, ep) in models.items()},
                   "device": a.device}, f, indent=1)
    print(f"wrote {a.out_dir}/features_shard{k}.csv ({len(df)} rows)")


def cmd_merge(a):
    parts = sorted(f for f in os.listdir(a.out_dir) if f.startswith("features_shard") and f.endswith(".csv"))
    df = pd.concat([pd.read_csv(os.path.join(a.out_dir, p)) for p in parts], ignore_index=True)
    df = df.sort_values(["harpnum", "mdi_t_rec", "model", "mask"]).reset_index(drop=True)
    df.to_csv(os.path.join(a.out_dir, "features.csv"), index=False)
    infos = {}
    for p in sorted(f for f in os.listdir(a.out_dir) if f.startswith("shard") and f.endswith("_info.json")):
        with open(os.path.join(a.out_dir, p)) as f:
            infos[p] = json.load(f)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(os.path.join(a.out_dir, "extract_info.json"), "w") as f:
        json.dump({"shards": infos, "n_rows": int(len(df)), "n_frames": int(len(df) // df.groupby(["model", "mask"]).ngroups),
                   "n_harps": int(df.harpnum.nunique()), "keywords": list(sk.KEYWORDS),
                   "strong_gauss": 200.0, "git_commit": commit, "preregistration": "reports/phase_12.md"}, f, indent=1)
    print(f"wrote {a.out_dir}/features.csv ({len(df)} rows, {df.harpnum.nunique()} HARPs)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="one shard of the extraction")
    ex.add_argument("--run", default="full")
    ex.add_argument("--exps", nargs="+", required=True)
    ex.add_argument("--device", default="cpu")
    ex.add_argument("--out_dir", default=os.path.join("results", "full", "overlap_pilot"))
    ex.add_argument("--shard", default="0/1", help="k/n round-robin over the sorted HARP list")
    ex.add_argument("--max_regions", type=int, default=0, help="smoke test: only the first N frames")
    ex.add_argument("--strong_gauss", type=float, default=200.0)
    ex.set_defaults(fn=cmd_extract)
    mg = sub.add_parser("merge", help="concatenate shards -> features.csv")
    mg.add_argument("--out_dir", default=os.path.join("results", "full", "overlap_pilot"))
    mg.set_defaults(fn=cmd_merge)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
