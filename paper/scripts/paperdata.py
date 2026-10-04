"""Shared loaders for the paper's tables and figures.

Every number in the paper is computed here from the evaluation outputs in
results/ (never typed by hand). Aggregation follows scripts/full_hypotheses.py:
a model's value for a HARP is the mean over that HARP's evaluated frames; a loss
family's value is the mean of that over the 3 seeds; paired Wilcoxon tests run
over HARPs on the seed-averaged values.
"""
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PAPER)
RES = os.path.join(ROOT, "results")
SEEDS = (0, 1, 2)

# Loss families: E1 = MSE, E2 = Munoz-Jaramillo et al. (2024) loss, E3 = ours,
# E4 = MSE + divergence only, E5 = MSE + gradient + SSIM (E3 without divergence).
FAMILIES = {
    "F_U_E1": ("U-Net", "MSE"), "F_U_E2": ("U-Net", "M24"), "F_U_E3": ("U-Net", "Ours"),
    "F_U_E4": ("U-Net", "MSE + Div"), "F_U_E5": ("U-Net", "MSE + Grad + SSIM"),
    "F_E1": ("HighRes-net", "MSE"), "F_E2": ("HighRes-net", "M24"), "F_E3": ("HighRes-net", "Ours"),
}
# Test HARPs that also appeared in the one-month pilot (splits/pilot.json), where
# the loss weight and smoothing scales were chosen: excluded in a robustness check.
PILOT_OVERLAP_TEST = (355, 358, 361, 378, 394)

HIGHER_BETTER = {"div_all", "div_strong", "div_s0", "div_s1", "div_s2", "div_s4", "div_s8",
                 "Bp_pearson", "Bt_pearson", "Br_pearson"}


def family(model):
    return re.sub(r"_s\d+$", "", model)


def seed_of(model):
    return int(re.search(r"_s(\d+)$", model).group(1))


def eval_dirs(split):
    d = [os.path.join(RES, "full", f"eval_{split}_s{s}") for s in SEEDS]
    d += [os.path.join(RES, "full", f"eval_controls_{split}_s{s}") for s in SEEDS]
    return d


def harp_table(split):
    """(family, seed, harpnum) -> per-HARP metrics for one split ('val' or 'test')."""
    dirs = eval_dirs(split)
    per = pd.concat([pd.read_csv(os.path.join(d, "per_region.csv")) for d in dirs])
    div = pd.concat([pd.read_csv(os.path.join(d, "div_by_scale_per_harp.csv")) for d in dirs])
    small = pd.concat([pd.read_csv(os.path.join(d, "munoz_small_scale_per_region.csv")) for d in dirs])
    per = per[per.model.str.startswith("F_") & (per["mask"] == "all")].copy()
    div = div[div.model.str.startswith("F_")].copy()
    small = small[small.model.str.startswith("F_")].copy()
    per["rmse_mean"] = per[["Bp_rmse", "Bt_rmse", "Br_rmse"]].mean(axis=1)
    cols = ["rmse_mean", "Bp_rmse", "Bt_rmse", "Br_rmse", "Bp_pearson", "Bt_pearson", "Br_pearson",
            "Bp_hist_tv", "Bt_hist_tv", "Br_hist_tv", "signed_flux_imbalance", "unsigned_flux_rel_err"]
    t = per.groupby(["model", "harpnum"])[cols].mean()
    scols = ["Br_extreme_err_32", "Bp_extreme_err_32", "Bt_extreme_err_32",
             "Br_sobel_err", "Bp_sobel_err", "Bt_sobel_err"]
    t = t.join(small.groupby(["model", "harpnum"])[scols].mean(), how="outer")
    for mk, name in (("all", "div_all"), ("strong", "div_strong")):
        x = div[(div["mask"] == mk) & div.sigma_px.isin([2.0, 4.0])]
        t[name] = x.groupby(["model", "harpnum"]).corr_pred_true.mean()
    for s in (0.0, 1.0, 2.0, 4.0, 8.0):
        x = div[(div["mask"] == "all") & (div.sigma_px == s)]
        t[f"div_s{int(s)}"] = x.set_index(["model", "harpnum"]).corr_pred_true
    t = t.reset_index()
    t["family"] = t.model.map(family)
    t["seed"] = t.model.map(seed_of)
    return t.set_index(["family", "seed", "harpnum"]).drop(columns="model").sort_index()


def family_harp(t):
    """(family, harpnum) -> metric, averaged over seeds."""
    return t.groupby(level=["family", "harpnum"]).mean()


def seed_stats(t, metric):
    """{family: (mean over seeds of the HARP-mean, std over seeds)}."""
    m = t[metric].groupby(level=["family", "seed"]).mean()
    return {f: (float(v.mean()), float(v.std(ddof=1))) for f, v in m.groupby(level="family")}


def paired(fh, metric, a, b, exclude=()):
    """Paired Wilcoxon over HARPs, same conventions as scripts/full_hypotheses.py."""
    x = fh[metric].xs(a, level="family")
    y = fh[metric].xs(b, level="family")
    d = (x - y).dropna()
    if exclude:
        d = d[~d.index.isin(exclude)]
    hb = metric in HIGHER_BETTER
    better = (d > 0) if hb else (d < 0)
    return {"median_diff": float(d.median()), "a_better_in": int(better.sum()), "n_harps": int(len(d)),
            "p_wilcoxon": float(wilcoxon(d).pvalue), "mean_a": float(x.mean()), "mean_b": float(y.mean())}


def pilot_snr():
    d = json.load(open(os.path.join(ROOT, "data", "pilot", "div_snr_by_scale.json")))
    return d, pd.DataFrame(d["rows"])


def pilot_summary(sub):
    return pd.read_csv(os.path.join(RES, "pilot", sub, "summary.csv"))


def pilot_div_per_harp(sub):
    return pd.read_csv(os.path.join(RES, "pilot", sub, "div_by_scale_per_harp.csv"))
