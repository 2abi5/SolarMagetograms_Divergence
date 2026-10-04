"""Targeted evaluation on small confined-flux regions (reports/phase_8.md 5), VALIDATION regions only.

    python scripts/eval_small_flux.py --run pilot --device cuda --exps V_E7 V_U_E2 V_U_E3 V_E1 V_E2 V_E3

Masks come from the TRUE SHARP Br of each region:
  compact  connected |Br| >= 300 G components of 4..256 HR px (smaller than 4x4 MDI pixels)
  mixed    MDI-pixel (4x4 HR) blocks holding both polarities (>= 2 px at >= 100 G each) that largely cancel
  pil      pixels within 2 px of both >= +150 G and <= -150 G (polarity inversion lines)
Per mask: Bp/Bt/Br RMSE, Br flux recovery |sum|pred|/sum|true| - 1|, peak recovery of compact features,
and divergence correlation at sigma 1 and 2 px. Paired Wilcoxon over HARPs (per-HARP means; seeds averaged
when model names end in _s<k>). Writes <out_dir>/per_region.csv, per_harp.csv and tests.json.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from div_snr_by_scale import smoothed_div  # noqa: E402
from sr.datasets import _load_pair, split_harps  # noqa: E402
from sr.evaluate import infer_region, load_model  # noqa: E402
from sr.geometry import geometry_path  # noqa: E402

NORM = 3500.0
EIGHT = np.ones((3, 3), bool)


def compact_features(br, thr=300.0, min_px=4, max_px=256):
    lab, n = ndimage.label(np.abs(np.nan_to_num(br)) >= thr, structure=EIGHT)
    if n == 0:
        return np.zeros(br.shape, bool)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = (sizes >= min_px) & (sizes <= max_px)
    return keep[lab]


def mixed_polarity_blocks(br, scale=4, thr=100.0, min_px=2, max_net=0.5):
    b = np.nan_to_num(br)
    ny, nx = b.shape[0] // scale, b.shape[1] // scale
    blk = b[:ny * scale, :nx * scale].reshape(ny, scale, nx, scale)
    pos = (blk >= thr).sum(axis=(1, 3)) >= min_px
    neg = (blk <= -thr).sum(axis=(1, 3)) >= min_px
    net = np.abs(blk.sum(axis=(1, 3))) < max_net * np.abs(blk).sum(axis=(1, 3))
    m = np.zeros(b.shape, bool)
    m[:ny * scale, :nx * scale] = np.kron(pos & neg & net, np.ones((scale, scale), bool)).astype(bool)
    return m


def pil_mask(br, thr=150.0, width=2):
    b = np.nan_to_num(br)
    st = np.ones((2 * width + 1, 2 * width + 1), bool)
    return ndimage.binary_dilation(b >= thr, st) & ndimage.binary_dilation(b <= -thr, st)


def flux_ratio(pred, true, mask):
    t = np.abs(true[mask]).sum()
    return float(np.abs(pred[mask]).sum() / t) if t > 0 else float("nan")


def peak_ratio(pred, true, mask):
    lab, n = ndimage.label(mask, structure=EIGHT)
    if n == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    pt = ndimage.maximum(np.abs(true), lab, idx)
    pp = ndimage.maximum(np.abs(pred), lab, idx)
    return float(np.mean(np.asarray(pp) / np.maximum(np.asarray(pt), 1e-9)))


def family(m):
    return re.sub(r"_s\d+$", "", m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--allow_test", action="store_true", help="Phase 7 only: required for --split test")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pairs", nargs="*", default=["V_U_E3:V_U_E2", "V_U_E3:V_E7", "V_E3:V_E2", "V_E3:V_E1"],
                    help="a:b family pairs to test")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        raise PermissionError("The test split is for Phase 7 only; pass --allow_test.")
    run_dir, res_dir = os.path.join(ROOT, "data", a.run), os.path.join(ROOT, "results", a.run)
    out_dir = a.out_dir or os.path.join(res_dir, "eval_small_flux")
    os.makedirs(out_dir, exist_ok=True)
    regs = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    val = regs[regs.harpnum.isin(split_harps(os.path.join(ROOT, "splits", f"{a.run}.json"), a.split))]
    if a.every > 1:
        val = pd.concat([g.sort_values("mdi_t_rec").iloc[::a.every] for _, g in val.groupby("harpnum")])
    val = val.reset_index(drop=True)
    models = {}
    for e in a.exps:
        cfg = yaml.safe_load(open(os.path.join(res_dir, e, "config.yaml")))
        models[e] = (load_model(res_dir, e, a.device)[0], int(cfg["model"].get("kwargs", {}).get("in_channels", 1)))
    geo_dir = os.path.join(run_dir, "geometry_lr")
    rows = []
    for i, r in enumerate(val.itertuples()):
        lr, hr = _load_pair(os.path.join(run_dir, r.region_file))
        true = hr.astype(np.float64) * NORM
        valid = np.isfinite(true).all(axis=0)
        br = np.where(valid, true[2], 0.0)
        masks = {"compact": compact_features(br) & valid, "mixed": mixed_polarity_blocks(br) & valid,
                 "pil": pil_mask(br) & valid}
        if not any(m.any() for m in masks.values()):
            continue
        x1 = torch.as_tensor(np.nan_to_num(lr))[None, None].to(a.device)
        x4 = None
        if any(c == 4 for _, c in models.values()):
            geom = np.load(geometry_path(geo_dir, r.region_file))
            x4 = torch.as_tensor(np.nan_to_num(np.concatenate([lr[None], geom])))[None].to(a.device)
        truth_div = {s: smoothed_div(true / NORM, s) for s in (1.0, 2.0)}
        with torch.no_grad():
            preds = {e: infer_region(m, x4 if c == 4 else x1)[0].cpu().numpy().astype(np.float64) * NORM
                     for e, (m, c) in models.items()}
        for e, p in preds.items():
            pdiv = {s: smoothed_div(np.where(np.isfinite(true), p, np.nan) / NORM, s) for s in (1.0, 2.0)}
            for mk, m in masks.items():
                if m.sum() < 4:
                    continue
                rec = {"model": e, "family": family(e), "mask": mk, "region": os.path.basename(r.region_file),
                       "harpnum": int(r.harpnum), "n_px": int(m.sum())}
                for c, ch in enumerate(("Bp", "Bt", "Br")):
                    rec[f"{ch}_rmse"] = float(np.sqrt(np.mean((p[c][m] - true[c][m]) ** 2)))
                fr = flux_ratio(p[2], true[2], m)
                rec["flux_ratio"], rec["flux_err"] = fr, abs(fr - 1)
                if mk == "compact":
                    pr = peak_ratio(p[2], true[2], m)
                    rec["peak_ratio"], rec["peak_err"] = pr, abs(pr - 1)
                mi = m[1:-1, 1:-1]
                for s in (1.0, 2.0):
                    dt, vt = truth_div[s]
                    dp, vp = pdiv[s]
                    sel = vt & vp & mi
                    rec[f"div_corr_s{int(s)}"] = (float(np.corrcoef(dp[sel], dt[sel])[0, 1])
                                                  if sel.sum() > 10 and dp[sel].std() > 0 and dt[sel].std() > 0 else float("nan"))
                rows.append(rec)
        if i % 50 == 0:
            print(f"{i}/{len(val)} regions", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(out_dir, "per_region.csv"), index=False)
    metrics = [c for c in ("Br_rmse", "Bp_rmse", "Bt_rmse", "flux_err", "peak_err", "div_corr_s1", "div_corr_s2") if c in per]
    ph = per.groupby(["model", "family", "mask", "harpnum"])[metrics].mean().reset_index()
    fh = ph.groupby(["family", "mask", "harpnum"])[metrics].mean()          # average over seeds
    fh.reset_index().to_csv(os.path.join(out_dir, "per_harp.csv"), index=False)
    tests = {"n_harps": int(per.harpnum.nunique()), "means": {}, "tests": {}}
    for (f, mk), g in fh.groupby(level=["family", "mask"]):
        tests["means"][f"{f}|{mk}"] = {m: float(g[m].mean()) for m in metrics}
    for pair in a.pairs:
        fa, fb = pair.split(":")
        for mk in ("compact", "mixed", "pil"):
            if (fa, mk) not in fh.index.droplevel("harpnum") or (fb, mk) not in fh.index.droplevel("harpnum"):
                continue
            xa, xb = fh.loc[(fa, mk)], fh.loc[(fb, mk)]
            res = {}
            for m in metrics:
                d = (xa[m] - xb[m]).dropna()
                if len(d) < 2 or (d == 0).all():
                    continue
                better = (d > 0) if m.startswith("div_corr") else (d < 0)
                res[m] = {"median_diff": float(d.median()), "a_better_in": int(better.sum()), "n": int(len(d)),
                          "p_wilcoxon": float(wilcoxon(d).pvalue), "mean_a": float(xa[m].mean()), "mean_b": float(xb[m].mean())}
            tests["tests"][f"{fa} vs {fb} | {mk}"] = res
    json.dump(tests, open(os.path.join(out_dir, "tests.json"), "w"), indent=1)
    for k, res in tests["tests"].items():
        s = "  ".join(f"{m}: {v['median_diff']:+.3g} ({v['a_better_in']}/{v['n']}, p={v['p_wilcoxon']:.3f})"
                      for m, v in res.items() if m in ("flux_err", "Br_rmse", "peak_err", "div_corr_s2"))
        print(f"{k}: {s}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
