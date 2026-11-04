"""Supplementary validation computations (results/full/extra_checks):

  testonly_agreement.json   screen agreement recomputed on the 52 TEST HARPs alone
                            (the 92 'held-out' pool includes the 40 early-stopping
                            validation regions; blocking item S2)
  deploy_same_window.json   deployment benchmark on the common window set of the two
                            arms (vector vs merged-LOS, same windows; blocking item B3)
  ratio_decomposition.json  reconciles the 1.08 recon/LOS flux ratio with the known
                            ~1.4 MDI/HMI scale: full-mask scale x strong-mask share

    python scripts/c23_supplementary_checks.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from c23_screening import SOURCES, agreement, derive, seed_mean  # noqa: E402
from c23_benchmark import (                                            # noqa: E402
    LOS_B, VEC_B, build_windows, fit_eval, label_windows, load_flares, los_frames_c23,
    sharp_era_frames, vec_frames_c23)

OUT = os.path.join(ROOT, "results", "full", "extra_checks")
PILOT = os.path.join(ROOT, "results", "full", "overlap_pilot", "features.csv")
C23F = os.path.join(ROOT, "results", "full", "c23_build", "features.csv")
KWS = ("USFLUX", "TOTUSJZ", "TOTUSJH", "R_VALUE", "AREA", "STRONG_FRAC",
       "MEANGBZ", "J_PER_FLUX", "H_PER_FLUX")


def testonly():
    splits = json.load(open(os.path.join(ROOT, "splits", "full.json")))["splits"]
    test_h = set(splits["test"]["harps"])
    df = pd.read_csv(PILOT)
    sub = df[df.harpnum.isin(test_h)]
    held = json.load(open(os.path.join(OUT, "heldout_agreement.json")))["keywords"]
    out = {}
    for kw in KWS:
        rho, n = agreement(sub, SOURCES["ours"], kw)
        out[kw] = {"testonly": round(float(rho), 4), "n_frames": int(n),
                   "heldout92": held[kw]["heldout"]["agreement"]}
    screened = [k for k in KWS if k not in ("J_PER_FLUX", "H_PER_FLUX")]
    out["max_abs_diff_vs_heldout_screened"] = round(
        max(abs(out[k]["testonly"] - out[k]["heldout92"]) for k in screened), 4)
    out["n_test_harps"] = int(sub.harpnum.nunique())
    json.dump(out, open(os.path.join(OUT, "testonly_agreement.json"), "w"), indent=1)
    print("testonly:", {k: out[k]["testonly"] for k in KWS},
          "| max diff:", out["max_abs_diff_vs_heldout_screened"])


def deploy_same_window():
    flares = load_flares(years=range(1996, 2017))
    era = sharp_era_frames(os.path.join(ROOT, "data", "c23", "sharp_era.csv"))
    tr_v = label_windows(build_windows(era, list(VEC_B)), flares)
    tr_l = label_windows(build_windows(era, list(LOS_B)), flares)
    te_v = label_windows(build_windows(vec_frames_c23(C23F), list(VEC_B)), flares)
    te_l = label_windows(build_windows(los_frames_c23(
        os.path.join(ROOT, "data", "c23", "frames_all.csv")), list(LOS_B)), flares)
    kv = set(map(tuple, te_v[["region", "t_end"]].astype(str).to_numpy().tolist()))
    kl = set(map(tuple, te_l[["region", "t_end"]].astype(str).to_numpy().tolist()))
    both = kv & kl
    sel = lambda d: d[[tuple(x) in both for x in d[["region", "t_end"]].astype(str).to_numpy().tolist()]]
    cols = lambda feats: [f"{f}_{s}" for f in feats for s in ("max", "mean", "last")]
    out = {"n_windows": len(both),
           "vector": fit_eval(tr_v, sel(te_v), cols(VEC_B), n_boot=1000, per_set_scale=True),
           "los_record": fit_eval(tr_l, sel(te_l), cols(LOS_B), n_boot=1000, per_set_scale=True)}
    json.dump(out, open(os.path.join(OUT, "deploy_same_window.json"), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("deploy same-window:", {a: {q: round(out[a][q], 4) for q in ("auc", "tss")}
                                  for a in ("vector", "los_record")}, "n", out["n_windows"])


def ratio_decomposition():
    use = {"tarpnum", "t_rec", "model", "mask", "USFLUX", "usfluxl"}
    df = pd.read_csv(C23F, usecols=lambda c: c in use)
    sm = df[df.model == "seed_mean"]
    a = sm[sm["mask"] == "all"].set_index(["tarpnum", "t_rec"])
    s = sm[sm["mask"] == "strong_pred"].set_index(["tarpnum", "t_rec"])
    j = a[["USFLUX", "usfluxl"]].join(s[["USFLUX"]], rsuffix="_strong").dropna()
    j = j[(j.usfluxl > 0) & (j.USFLUX > 0)]
    out = {"median_all_over_usfluxl": round(float((j.USFLUX / j.usfluxl).median()), 3),
           "median_strong_over_usfluxl": round(float((j.USFLUX_strong / j.usfluxl).median()), 3),
           "median_strong_over_all": round(float((j.USFLUX_strong / j.USFLUX).median()), 3),
           "n_frames": int(len(j))}
    json.dump(out, open(os.path.join(OUT, "ratio_decomposition.json"), "w"), indent=1)
    print("ratio decomposition:", out)


if __name__ == "__main__":
    testonly()
    ratio_decomposition()
    deploy_same_window()
