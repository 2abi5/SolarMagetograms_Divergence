"""Pilot v2 evaluation (reports/phase_6.md 11) on VALIDATION regions only (never test).

    python scripts/eval_v2.py --run pilot --device cuda --exps V_E1 V_E2 V_E3 ... E1 E7 ...

Models whose config has model.kwargs.in_channels = 4 get MDI + the viewing-geometry
channels (sr/geometry.py); the others get MDI only. Per region and model:
  - the standard metrics of sr/evaluate.region_rows (all pixels and |B| > 200 G);
  - the multi-scale divergence metric of reports/phase_6.md 8 (div_h correlation after
    Gaussian smoothing, sigma in HR px);
  - Munoz-Jaramillo et al. (2024) metrics: pooled Pearson per component by disk position
    (r/Rsun bins 0-1/3, 1/3-1/2, 1/2-3/4, 3/4-1; their Table 4) and by field strength
    (|B| < 600 G, >= 600 G), extreme-value error per fully valid 32x32 HR tile
    (|max_pred - max_true| + |min_pred - min_true|), and the Sobel-gradient error
    |grad(pred - true)| per pixel.
Writes <out_dir>/per_region.csv, summary.csv, munoz_table4.csv, munoz_small_scale.csv,
div_by_scale.csv, div_by_scale_per_harp.csv and hypotheses.json.
"""
import argparse
import json
import os
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
from div_snr_by_scale import SIGMAS, smoothed_div  # noqa: E402
from sr.datasets import _load_pair, split_harps  # noqa: E402
from sr.evaluate import baseline_prediction, fit_calibration, infer_region, load_model, region_rows, summarise  # noqa: E402
from sr.geometry import geometry_path  # noqa: E402

CHANNELS = ("Bp", "Bt", "Br")
R_EDGES = (1 / 3, 1 / 2, 3 / 4)
R_LABELS = ("0-1/3", "1/3-1/2", "1/2-3/4", "3/4-1")
B_EDGES = (600.0,)
B_LABELS = ("|B|<600G", "|B|>=600G")
BASELINE_NAMES = ("E0_raw", "E0_cal", "E0c")


def subsample_regions(val, every):
    """Every `every`-th frame of each HARP in time order (consecutive 96-min frames are near-duplicates)."""
    if every <= 1:
        return val.reset_index(drop=True)
    parts = [g.sort_values("mdi_t_rec").iloc[::every] for _, g in val.groupby("harpnum", sort=True)]
    return pd.concat(parts).reset_index(drop=True)


def radius_bin(c_r):
    """Bin index of r/Rsun = sin(heliocentric angle) = sqrt(1 - c_r^2)."""
    return np.digitize(np.sqrt(np.clip(1.0 - np.asarray(c_r, float) ** 2, 0.0, None)), R_EDGES)


def add_binned(acc, model, ch, x, y, bins, n_bins):
    for k in range(n_bins):
        sel = bins == k
        xs, ys = x[sel], y[sel]
        c = acc.setdefault((model, ch, k), np.zeros(6))
        c += [xs.size, xs.sum(), ys.sum(), (xs * xs).sum(), (ys * ys).sum(), (xs * ys).sum()]


def pearson_from(c):
    n, sx, sy, sxx, syy, sxy = c
    if n < 3:
        return float("nan")
    vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
    return float((sxy / n - sx * sy / n ** 2) / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def tile_extreme_error(p, t, mask, tile=32):
    """Mean over fully valid tile x tile tiles of |max_p - max_t| + |min_p - min_t| (Munoz Eq. 4-5)."""
    errs = []
    for i in range(0, t.shape[0] - tile + 1, tile):
        for j in range(0, t.shape[1] - tile + 1, tile):
            if mask[i:i + tile, j:j + tile].all():
                pt, tt = p[i:i + tile, j:j + tile], t[i:i + tile, j:j + tile]
                errs.append(abs(pt.max() - tt.max()) + abs(pt.min() - tt.min()))
    return float(np.mean(errs)) if errs else float("nan")


def sobel_error(p, t, mask):
    """Mean |Sobel gradient of (p - t)| over pixels whose 3x3 window is valid (Munoz Eq. 6)."""
    d = np.where(mask, p - t, 0.0)
    g = np.hypot(ndimage.sobel(d, axis=1, mode="nearest"), ndimage.sobel(d, axis=0, mode="nearest"))
    v = ndimage.binary_erosion(mask, np.ones((3, 3)), border_value=0)
    return float(g[v].mean()) if v.any() else float("nan")


def _corr(c):
    n, sx, sy, sxx, syy, sxy = c[:6]
    vx = sxx / n - (sx / n) ** 2
    return (sxy / n - sx * sy / n ** 2) / np.sqrt(vx * (syy / n - (sy / n) ** 2)) if vx > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--max_regions", type=int, default=0, help="smoke test: only the first N validation regions")
    ap.add_argument("--every", type=int, default=1, help="use every N-th frame of each HARP (Phase 8: 3)")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--allow_test", action="store_true", help="Phase 7 only: required for --split test")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        raise PermissionError("The test split is for Phase 7 only; pass --allow_test.")
    run_dir, res_dir = os.path.join(ROOT, "data", a.run), os.path.join(ROOT, "results", a.run)
    out_dir = a.out_dir or os.path.join(res_dir, "eval_val_v2")
    os.makedirs(out_dir, exist_ok=True)
    split = os.path.join(ROOT, "splits", f"{a.run}.json")
    stats = json.load(open(os.path.join(run_dir, "stats.json")))
    norm = float(stats.get("norm", 3500.0))
    data_range = 2 * np.asarray(stats["ssim_hi"], float) * norm
    centers = np.asarray(stats["hist_centers"], float) * norm
    means = np.array([stats["channel_stats_gauss"][c]["mean"] for c in CHANNELS])
    regs = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    train = regs[regs.harpnum.isin(split_harps(split, "train"))]
    calib = fit_calibration([(lr * norm, hr[2] * norm) for lr, hr in
                             (_load_pair(os.path.join(run_dir, f)) for f in train.region_file)])
    val = regs[regs.harpnum.isin(split_harps(split, a.split))].reset_index(drop=True)
    val = subsample_regions(val, a.every)
    if a.max_regions:
        val = val.groupby("harpnum").head(max(1, a.max_regions // val.harpnum.nunique())).reset_index(drop=True)
    geo_dir = os.path.join(run_dir, "geometry_lr")
    models = {}
    for e in a.exps:
        cfg = yaml.safe_load(open(os.path.join(res_dir, e, "config.yaml")))
        models[e] = (load_model(res_dir, e, a.device)[0], int(cfg["model"].get("kwargs", {}).get("in_channels", 1)))

    rows, t4, small, div_acc, div_harp = [], {}, [], {}, {}
    for i, r in enumerate(val.itertuples()):
        lr, hr = _load_pair(os.path.join(run_dir, r.region_file))
        true = hr.astype(np.float64) * norm
        mask = np.isfinite(true).all(axis=0)
        geom = np.load(geometry_path(geo_dir, r.region_file))
        c_r_hr = np.kron(geom[2], np.ones((4, 4)))[:true.shape[1], :true.shape[2]]
        rbin = radius_bin(c_r_hr)
        babs = np.sqrt(np.nansum(true ** 2, axis=0))
        bbin = np.digitize(babs, B_EDGES)
        preds = {b: baseline_prediction(b, lr * norm, means, calib) for b in BASELINE_NAMES}
        x1 = torch.as_tensor(np.nan_to_num(lr))[None, None].to(a.device)
        x4 = torch.as_tensor(np.nan_to_num(np.concatenate([lr[None], geom])))[None].to(a.device)
        with torch.no_grad():
            for e, (m, cin) in models.items():
                preds[e] = infer_region(m, x4 if cin == 4 else x1)[0].cpu().numpy().astype(np.float64) * norm
        truth_div = {s: smoothed_div(true / norm, s) for s in SIGMAS}
        strong_int = (babs > 200.0)[1:-1, 1:-1]
        for name, pred in preds.items():
            for mname, vals in region_rows(pred, true, mask, data_range, centers).items():
                rows.append({"model": name, "mask": mname, "region": os.path.basename(r.region_file),
                             "harpnum": int(r.harpnum), "mdi_t_rec": r.mdi_t_rec, **vals})
            rec = {"model": name, "region": os.path.basename(r.region_file), "harpnum": int(r.harpnum)}
            for c, ch in enumerate(CHANNELS):
                pc = pred[c]
                ok = mask & np.isfinite(pc)
                if not ok.any():
                    continue
                add_binned(t4, name, ch + "|radius", pc[ok], true[c][ok], rbin[ok], len(R_LABELS))
                add_binned(t4, name, ch + "|field", pc[ok], true[c][ok], bbin[ok], len(B_LABELS))
                add_binned(t4, name, ch + "|all", pc[ok], true[c][ok], np.zeros(ok.sum(), int), 1)
                rec[f"{ch}_extreme_err_32"] = tile_extreme_error(np.where(ok, pc, 0), true[c], ok)
                rec[f"{ch}_sobel_err"] = sobel_error(np.where(ok, pc, 0), true[c], ok)
            small.append(rec)
            if name in BASELINE_NAMES:
                continue
            for s in SIGMAS:
                dt, vt = truth_div[s]
                dp, vp = smoothed_div(np.where(np.isfinite(true), pred, np.nan) / norm, s)
                for mk, extra in (("all", True), ("strong", strong_int)):
                    sel = vt & vp & extra
                    x, y = dp[sel], dt[sel]
                    row = [x.size, x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]
                    div_acc.setdefault((name, s, mk), np.zeros(6))[:] += row
                    div_harp.setdefault((name, s, mk, int(r.harpnum)), np.zeros(6))[:] += row
        if i % 50 == 0:
            print(f"{i}/{len(val)} regions", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(out_dir, "per_region.csv"), index=False)
    summarise(per, a.n_boot).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    t4rows = []
    for (m, key, k), c in t4.items():
        ch, kind = key.split("|")
        label = {"radius": R_LABELS, "field": B_LABELS, "all": ("all",)}[kind][k]
        t4rows.append({"model": m, "channel": ch, "bin_type": kind, "bin": label, "n": int(c[0]), "pearson": pearson_from(c)})
    pd.DataFrame(t4rows).to_csv(os.path.join(out_dir, "munoz_table4.csv"), index=False)
    sm = pd.DataFrame(small)
    sm.to_csv(os.path.join(out_dir, "munoz_small_scale_per_region.csv"), index=False)
    sm.drop(columns=["region"]).groupby(["model", "harpnum"]).mean().groupby("model").mean() \
        .to_csv(os.path.join(out_dir, "munoz_small_scale.csv"))
    pd.DataFrame([{"model": k[0], "sigma_px": k[1], "mask": k[2], "n": int(c[0]), "corr_pred_true": _corr(c)}
                  for k, c in div_acc.items()]).to_csv(os.path.join(out_dir, "div_by_scale.csv"), index=False)
    dh = pd.DataFrame([{"model": k[0], "sigma_px": k[1], "mask": k[2], "harpnum": k[3], "n": int(c[0]),
                        "corr_pred_true": _corr(c)} for k, c in div_harp.items() if c[0] > 10])
    dh.to_csv(os.path.join(out_dir, "div_by_scale_per_harp.csv"), index=False)

    # pre-registered tests (reports/phase_6.md 11)
    def harp_div(m, mk="all"):
        x = dh[(dh.model == m) & (dh["mask"] == mk) & dh.sigma_px.isin([2.0, 4.0])]
        return x.groupby("harpnum").corr_pred_true.mean()

    def harp_metric(m, met, mk="all"):
        return per[(per.model == m) & (per["mask"] == mk)].groupby("harpnum")[met].mean()

    def test(xa, xb, higher_better=True):
        d = (xa - xb).dropna()
        if len(d) < 2 or (d == 0).all():
            return {"n_harps": int(len(d))}
        better = (d > 0) if higher_better else (d < 0)
        return {"median_diff": float(d.median()), "a_better_in": int(better.sum()), "n_harps": int(len(d)),
                "p_wilcoxon": float(wilcoxon(d).pvalue), "mean_a": float(xa.mean()), "mean_b": float(xb.mean())}

    have = set(models)
    meta = {e: json.load(open(os.path.join(res_dir, e, "meta.json"))).get("best_val_rmse_mean") for e in have}
    hyp = {}
    for key, a_, b_ in (("H1_V_E3_vs_V_E2", "V_E3", "V_E2"), ("H2_V_E3_vs_V_E1", "V_E3", "V_E1"),
                        ("H3_V_U_E3_vs_V_E7", "V_U_E3", "V_E7"), ("secondary_V_E6_vs_V_E2", "V_E6", "V_E2"),
                        ("H5_V_U_E3_vs_V_U_E2", "V_U_E3", "V_U_E2")):
        if a_ in have and b_ in have:
            res = {"div_corr_all": test(harp_div(a_), harp_div(b_)), "div_corr_strong": test(harp_div(a_, "strong"), harp_div(b_, "strong"))}
            for met in ("Bp_rmse", "Bt_rmse", "Br_rmse", "Bp_hist_tv", "Bt_hist_tv"):
                res[met] = test(harp_metric(a_, met), harp_metric(b_, met), higher_better=False)
            res["val_rmse"] = {a_: meta.get(a_), b_: meta.get(b_)}
            t = res["div_corr_all"]
            res["supported"] = bool(t.get("p_wilcoxon", 1) < 0.05 and t.get("median_diff", 0) > 0)
            if key.startswith("H3"):
                res["supported"] = bool(res["supported"] and meta[a_] <= 1.01 * meta[b_])
            hyp[key] = res
    if "V_E1" in have and "E1" in have:
        res = {m: test(harp_metric("V_E1", m), harp_metric("E1", m)) for m in ("Bp_pearson", "Bt_pearson", "Br_pearson")}
        res["supported"] = bool(all(res[m].get("p_wilcoxon", 1) < 0.05 and res[m].get("median_diff", 0) > 0
                                    for m in ("Bp_pearson", "Bt_pearson")))
        hyp["H4_V_E1_vs_E1_setup"] = res
    json.dump(hyp, open(os.path.join(out_dir, "hypotheses.json"), "w"), indent=1)
    print(json.dumps({k: v.get("supported") for k, v in hyp.items()}, indent=1))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
