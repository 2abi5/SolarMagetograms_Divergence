"""Phase 8 hypothesis tests (reports/phase_8.md 4): per-HARP values averaged over the 3 seeds of each
model, paired Wilcoxon over the validation HARPs.

    python scripts/full_hypotheses.py --evals results/full/eval_val_s0 results/full/eval_val_s1 results/full/eval_val_s2
Writes results/full/hypotheses_full.json and prints a summary.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def family(model):
    return re.sub(r"_s\d+$", "", model)


def family_harp_mean(df, col):
    """(family, harpnum) -> mean over seeds of the per-HARP mean of `col`."""
    per = df.groupby(["model", "harpnum"])[col].mean().reset_index()
    per["family"] = per.model.map(family)
    return per.groupby(["family", "harpnum"])[col].mean()


def paired(a, b, higher_better=True):
    d = (a - b).dropna()
    if len(d) < 2 or (d == 0).all():
        return {"n_harps": int(len(d))}
    better = d > 0 if higher_better else d < 0
    return {"median_diff": float(d.median()), "a_better_in": int(better.sum()), "n_harps": int(len(d)),
            "p_wilcoxon": float(wilcoxon(d).pvalue), "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def non_inferior(a_rmse, b_rmse, margin=0.01):
    """a's per-HARP RMSE is at most `margin` (relative) worse than b's (median of the relative difference)."""
    rel = ((a_rmse - b_rmse) / b_rmse).dropna()
    return bool(rel.median() <= margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "full", "hypotheses_full.json"))
    a = ap.parse_args()
    div = pd.concat([pd.read_csv(os.path.join(d, "div_by_scale_per_harp.csv")) for d in a.evals])
    per = pd.concat([pd.read_csv(os.path.join(d, "per_region.csv")) for d in a.evals])
    small = pd.concat([pd.read_csv(os.path.join(d, "munoz_small_scale_per_region.csv")) for d in a.evals])
    divs = {mk: family_harp_mean(div[(div["mask"] == mk) & div.sigma_px.isin([2.0, 4.0])], "corr_pred_true")
            for mk in ("all", "strong")}
    allp = per[per["mask"] == "all"].copy()
    allp["rmse_mean"] = allp[["Bp_rmse", "Bt_rmse", "Br_rmse"]].mean(axis=1)
    metric = {m: family_harp_mean(allp, m) for m in ("rmse_mean", "Bp_rmse", "Bt_rmse", "Br_rmse",
                                                     "Bp_pearson", "Bt_pearson", "Br_pearson", "Bp_hist_tv", "Bt_hist_tv")}
    for m in ("Br_extreme_err_32", "Bp_extreme_err_32", "Bt_extreme_err_32", "Br_sobel_err", "Bp_sobel_err", "Bt_sobel_err"):
        metric[m] = family_harp_mean(small, m)
    g = lambda s, f: s.xs(f, level="family")
    out = {"n_val_harps": int(div.harpnum.nunique()), "comparisons": {}}
    for key, x, y in (("P8-H1 F_U_E3 vs F_U_E1", "F_U_E3", "F_U_E1"), ("P8-H2 F_U_E3 vs F_U_E2", "F_U_E3", "F_U_E2"),
                      ("P8-H4a F_E3 vs F_E1", "F_E3", "F_E1"), ("P8-H4b F_E3 vs F_E2", "F_E3", "F_E2"),
                      ("U-Net Munoz vs MSE", "F_U_E2", "F_U_E1")):
        res = {"div_corr_all": paired(g(divs["all"], x), g(divs["all"], y)),
               "div_corr_strong": paired(g(divs["strong"], x), g(divs["strong"], y))}
        for m, s in metric.items():
            res[m] = paired(g(s, x), g(s, y), higher_better=m.endswith("pearson"))
        t = res["div_corr_all"]
        res["div_supported"] = bool(t.get("p_wilcoxon", 1) < 0.05 and t.get("median_diff", 0) > 0)
        out["comparisons"][key] = res
    out["P8-H3 non-inferior RMSE (F_U_E3 vs F_U_E1, 1%)"] = non_inferior(g(metric["rmse_mean"], "F_U_E3"),
                                                                        g(metric["rmse_mean"], "F_U_E1"))
    # run-to-run spread: std over seeds of each model's mean divergence correlation
    seed_means = div[(div["mask"] == "all") & div.sigma_px.isin([2.0, 4.0])].groupby("model").corr_pred_true.mean()
    out["seed_spread_div_corr"] = {f: {"mean": float(v.mean()), "std": float(v.std())}
                                   for f, v in seed_means.groupby(seed_means.index.map(family))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, r in out["comparisons"].items():
        t = r["div_corr_all"]
        print(f"{k}: div {t.get('median_diff', np.nan):+.4f} ({t.get('a_better_in')}/{t.get('n_harps')}, p={t.get('p_wilcoxon', np.nan):.3g}) "
              f"supported={r['div_supported']} | rmse_mean {r['rmse_mean'].get('median_diff', np.nan):+.2f} G")
    print("P8-H3:", out["P8-H3 non-inferior RMSE (F_U_E3 vs F_U_E1, 1%)"])
    print("seed spread:", {k: round(v["std"], 4) for k, v in out["seed_spread_div_corr"].items()})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
