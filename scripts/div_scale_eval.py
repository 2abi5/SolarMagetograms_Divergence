"""Multi-scale divergence error of trained models on VALIDATION regions (never test).

    python scripts/div_scale_eval.py --run pilot --exps E1 E2 E3_l0.01 E3_l100 E7 --device cpu

For each smoothing scale sigma (HR px) and model: pooled correlation between the
predicted and true div_h at that scale, and mean |div_pred - div_true| next to
mean |div_true| (the error of a zero-divergence prediction). "zero" is the
prediction Bp = Bt = 0. Writes results/<run>/div_by_scale_val.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from div_snr_by_scale import SIGMAS, smoothed_div  # noqa: E402
from sr.datasets import RegionDataset, split_harps  # noqa: E402
from sr.evaluate import infer_region, load_model  # noqa: E402

NORM = 3500.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strong_gauss", type=float, default=200.0)
    ap.add_argument("--out_name", default="div_by_scale_val", help="results/<run>/<out_name>.csv (+ _per_harp.csv)")
    a = ap.parse_args()
    run_dir = os.path.join(ROOT, "data", a.run)
    res_dir = os.path.join(ROOT, "results", a.run)
    regs = RegionDataset(run_dir, harps=split_harps(os.path.join(ROOT, "splits", f"{a.run}.json"), "val")).rows
    models = {e: load_model(res_dir, e, a.device)[0] for e in a.exps}
    acc, per_harp = {}, {}
    for i, r in enumerate(regs.itertuples()):
        with np.load(os.path.join(run_dir, r.region_file)) as d:
            lr, hr = d["lr"].astype(np.float32), d["hr"].astype(np.float64)
        preds = {"zero": np.zeros_like(hr)}
        t = torch.as_tensor(np.nan_to_num(lr))[None, None].to(a.device)
        with torch.no_grad():
            for e, m in models.items():
                preds[e] = infer_region(m, t)[0].cpu().numpy().astype(np.float64)
        babs = np.sqrt(np.nansum(hr ** 2, axis=0))[1:-1, 1:-1] * NORM
        for s in SIGMAS:
            dt, vt = smoothed_div(hr, s)
            for name, p in preds.items():
                p = np.where(np.isfinite(hr), p, np.nan)          # same valid pixels as the truth
                dp, vp = smoothed_div(p, s)
                for mask_name, extra in (("all", True), ("strong", babs > a.strong_gauss)):
                    sel = vt & vp & extra
                    x, y = dp[sel], dt[sel]
                    k = (name, s, mask_name)
                    row = [x.size, x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum(),
                           np.abs(x - y).sum(), np.abs(y).sum()]
                    acc.setdefault(k, np.zeros(8))[:] += row
                    per_harp.setdefault(k + (int(r.harpnum),), np.zeros(8))[:] += row
        if i % 50 == 0:
            print(f"{i}/{len(regs)} regions", flush=True)
    def summary(c):
        n, sx, sy, sxx, syy, sxy, sae, sat = c
        mx, my = sx / n, sy / n
        vx = sxx / n - mx * mx
        corr = (sxy / n - mx * my) / np.sqrt(vx * (syy / n - my * my)) if vx > 0 else 0.0
        return {"n": int(n), "corr_pred_true": corr, "mean_abs_err_g_per_mm": sae / n,
                "mean_abs_true_g_per_mm": sat / n}
    rows = [{"model": k[0], "mask": k[2], "sigma_px": k[1], **summary(c)} for k, c in acc.items()]
    ph = pd.DataFrame([{"model": k[0], "mask": k[2], "sigma_px": k[1], "harpnum": k[3], **summary(c)}
                       for k, c in per_harp.items() if c[0] > 10])
    ph.to_csv(os.path.join(res_dir, f"{a.out_name}_per_harp.csv"), index=False)
    df = pd.DataFrame(rows).sort_values(["mask", "sigma_px", "model"])
    df["err_vs_zero"] = df.mean_abs_err_g_per_mm / df.mean_abs_true_g_per_mm
    out = os.path.join(res_dir, f"{a.out_name}.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
