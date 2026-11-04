"""Phase 13 / F3: cross-cycle flare-forecasting transfer benchmark (reports/phase_13.md 4, F3).

    python scripts/c23_benchmark.py c24los   # SHARP-header LOS keywords -> c24_los_keywords.csv
    python scripts/c23_benchmark.py run      # both arms, train C24 -> test C23

Rolling windows per region: 24 h observation (15 frames at 96 min, >= 10 observed), stride 24 h,
label = the region's NOAA number produces a GOES >= M1.0 flare in the 24 h after window end
(NGDC catalogs 1996-2011, one parser). L2 logistic regression (C = 1, class-balanced) on
max/mean/last of each feature, z-scored on the training set. AUC + peak TSS (threshold from the
training set), 95 % bootstrap over test regions.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from c23_external import parse_xrs  # noqa: E402
from c23_screening import transform  # noqa: E402

WINDOW_FRAMES = 15          # 24 h at 96 min
MIN_OBSERVED = 10
HORIZON_H = 24.0
M1 = 1e-5
RAW_DIR = os.path.join(ROOT, "data", "full", "c23_external", "raw")
VEC_FEATURES = ("USFLUX", "TOTUSJZ", "TOTUSJH", "R_VALUE", "AREA", "STRONG_FRAC")
LOS_FEATURES = ("USFLUXL", "MEANGBL", "AREA_LOS", "R_VALUE_LOS")
LOS_LOG10 = frozenset({"USFLUXL", "AREA_LOS"})


def tvalue(series):
    return pd.to_datetime(series.astype(str).str.replace("_TAI", "", regex=False),
                          format="%Y.%m.%d_%H:%M:%S")


def load_flares(years=range(1996, 2012)):
    """One consistently parsed GOES event table over all years."""
    parts = [parse_xrs(os.path.join(RAW_DIR, f"goes-xrs-report_{y}.txt")) for y in years]
    f = pd.concat(parts, ignore_index=True)
    return f[f.noaa_ar > 0].reset_index(drop=True)


def feature_transform(name, v):
    """Analysis transform for benchmark features (vector names via E2's rule; LOS names here)."""
    if name in LOS_LOG10:
        v = np.asarray(v, np.float64)
        return np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
    if name.endswith("_LOS"):
        return np.asarray(v, np.float64)
    return transform(name, v)


def build_windows(frames, features, window=WINDOW_FRAMES, stride=WINDOW_FRAMES, min_obs=MIN_OBSERVED):
    """Per-region rolling windows over the time-ordered frame table (region, t, noaa_ar, features).
    Returns one row per window: region, noaa_ar, t_end, n_obs, and max/mean/last per feature
    (on the transform scale)."""
    rows = []
    for region, g in frames.groupby("region"):
        g = g.sort_values("_t").reset_index(drop=True)
        vals = {f: feature_transform(f, g[f]) for f in features}
        for s in range(0, max(1, len(g) - min_obs + 1), stride):
            w = slice(s, s + window)
            tw = g["_t"].iloc[w]
            if len(tw) < min_obs:
                continue
            row = {"region": region, "noaa_ar": int(g.noaa_ar.iloc[0]), "t_end": tw.iloc[-1],
                   "n_obs": int(len(tw))}
            ok_all = True
            for f in features:
                v = vals[f][w.start:w.stop]
                v = v[np.isfinite(v)]
                if v.size < min_obs // 2:
                    ok_all = False
                    break
                row[f"{f}_max"], row[f"{f}_mean"], row[f"{f}_last"] = \
                    float(v.max()), float(v.mean()), float(v[-1])
            if ok_all:
                rows.append(row)
    return pd.DataFrame(rows)


def label_windows(wins, flares, horizon_h=HORIZON_H, wm2_min=M1):
    """True when the window's NOAA number has a flare >= wm2_min peaking in (t_end, t_end + horizon]."""
    f = flares[flares.wm2 >= wm2_min]
    by = {n: g.peak.to_numpy() for n, g in f.groupby("noaa_ar")}
    lab = []
    for r in wins.itertuples():
        peaks = by.get(int(r.noaa_ar))
        if peaks is None:
            lab.append(False)
            continue
        t0 = np.datetime64(r.t_end)
        lab.append(bool(((peaks > t0) & (peaks <= t0 + np.timedelta64(int(horizon_h * 60), "m"))).any()))
    out = wins.copy()
    out["label"] = lab
    return out


def tss(y, p, thr):
    yhat = p >= thr
    pos, neg = y.astype(bool), ~y.astype(bool)
    tpr = yhat[pos].mean() if pos.any() else 0.0
    fpr = yhat[neg].mean() if neg.any() else 0.0
    return float(tpr - fpr)


def peak_tss_threshold(y, p):
    cands = np.unique(p)
    if len(cands) > 512:
        cands = np.quantile(p, np.linspace(0, 1, 512))
    scores = [(tss(y, p, t), t) for t in cands]
    return max(scores)[1]


def fit_eval(train, test, feature_cols, n_boot=2000, seed=0, per_set_scale=False):
    """Logistic arm: fit on train windows, evaluate on test windows; AUC + TSS at the
    train-chosen threshold, bootstrap over TEST regions. per_set_scale: z-score test on its
    own statistics (per-era standardization, phase_13 F3b rule)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    Xtr = train[feature_cols].to_numpy(np.float64)
    Xte = test[feature_cols].to_numpy(np.float64)
    ytr = train.label.to_numpy(bool)
    yte = test.label.to_numpy(bool)
    sc = StandardScaler().fit(Xtr)
    sct = StandardScaler().fit(Xte) if per_set_scale else sc
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(Xtr), ytr)
    ptr = clf.predict_proba(sc.transform(Xtr))[:, 1]
    pte = clf.predict_proba(sct.transform(Xte))[:, 1]
    thr = peak_tss_threshold(ytr, ptr)
    out = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
           "base_rate_train": float(ytr.mean()), "base_rate_test": float(yte.mean()),
           "auc": float(roc_auc_score(yte, pte)) if 0 < yte.mean() < 1 else float("nan"),
           "tss": tss(yte, pte, thr), "threshold": float(thr),
           "coef": {c: float(w) for c, w in zip(feature_cols, clf.coef_[0])}}
    rng = np.random.default_rng(seed)
    regions = test.region.unique()
    by = {r: np.flatnonzero((test.region == r).to_numpy()) for r in regions}
    aucs, tsss = [], []
    for _ in range(n_boot):
        idx = np.concatenate([by[r] for r in rng.choice(regions, size=len(regions), replace=True)])
        yb, pb = yte[idx], pte[idx]
        if 0 < yb.mean() < 1:
            from sklearn.metrics import roc_auc_score as ras
            aucs.append(ras(yb, pb))
            tsss.append(tss(yb, pb, thr))
    if aucs:
        out["auc_ci"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
        out["tss_ci"] = [float(np.percentile(tsss, 2.5)), float(np.percentile(tsss, 97.5))]
    return out


# ---------------------------------------------------------------- data assembly

def cmd_c24los(a):
    """SHARP-header LOS keywords per C24 frame -> results/full/c23_build/c24_los_keywords.csv."""
    regs = pd.read_csv(os.path.join(ROOT, "data", "full", "regions.csv"))
    rows = []
    for i, r in enumerate(regs.itertuples()):
        with np.load(os.path.join(ROOT, "data", "full", r.region_file), allow_pickle=False) as d:
            h = json.loads(str(d["sharp_header_json"])) if "sharp_header_json" in d.files else {}
        rows.append({"harpnum": int(r.harpnum), "mdi_t_rec": r.mdi_t_rec, "noaa_ar": int(r.noaa_ar),
                     "USFLUXL": h.get("usfluxl"), "MEANGBL": h.get("meangbl"),
                     "AREA_LOS": h.get("area"), "R_VALUE_LOS": h.get("r_value")})
        if i % 2000 == 0:
            print(f"{i}/{len(regs)}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "c24_los_keywords.csv"), index=False)
    print(f"wrote {a.out_dir}/c24_los_keywords.csv ({len(rows)} frames)")


def vec_frames_c24(pilot_features):
    from c23_screening import SOURCES, derive, seed_mean
    df = pd.read_csv(pilot_features)
    sm = derive(seed_mean(df[df["mask"] == "strong_pred"], SOURCES["ours"]))
    sm["region"] = sm.harpnum
    sm["_t"] = tvalue(sm.mdi_t_rec)
    meta = df[["harpnum", "mdi_t_rec", "noaa_ar"]].drop_duplicates()
    sm = sm.merge(meta, on=["harpnum", "mdi_t_rec"], how="left")
    return sm


def vec_frames_c23(c23_features):
    from c23_screening import derive
    df = pd.read_csv(c23_features)
    sm = df[(df.model == "seed_mean") & (df["mask"] == "strong_pred")].copy()
    sm = derive(sm)
    sm["region"] = sm.tarpnum
    sm["_t"] = tvalue(sm.t_rec)
    return sm


def los_frames_c24(path):
    df = pd.read_csv(path)
    df = df[df.noaa_ar >= 0].copy()
    df["region"] = df.harpnum
    df["_t"] = tvalue(df.mdi_t_rec)
    return df


def los_frames_c23(frames_all):
    df = pd.read_csv(frames_all)
    df = df.rename(columns={"TARPNUM": "region", "NOAA_AR": "noaa_ar",
                            "AREA": "AREA_LOS", "R_VALUE": "R_VALUE_LOS"})
    df["_t"] = tvalue(df.T_REC)
    return df


def cmd_run(a):
    flares = load_flares()
    print(f"{len(flares)} AR-attributed flares, {int((flares.wm2 >= M1).sum())} >= M1")
    arms = {}
    cols = lambda feats: [f"{f}_{s}" for f in feats for s in ("max", "mean", "last")]
    vec_feats = [f for f in VEC_FEATURES]
    tr = label_windows(build_windows(vec_frames_c24(a.pilot_features), vec_feats), flares)
    te = label_windows(build_windows(vec_frames_c23(a.c23_features), vec_feats), flares)
    arms["vector"] = fit_eval(tr, te, cols(vec_feats))
    tr_l = label_windows(build_windows(los_frames_c24(os.path.join(a.out_dir, "c24_los_keywords.csv")),
                                       list(LOS_FEATURES)), flares)
    te_l = label_windows(build_windows(los_frames_c23(a.frames_all), list(LOS_FEATURES)), flares)
    arms["los_record"] = fit_eval(tr_l, te_l, cols(LOS_FEATURES))
    with open(os.path.join(a.out_dir, "f3_benchmark.json"), "w") as f:
        json.dump({"design": "phase_13.md 4 F3 (rolling windows, 24h obs -> 24h horizon, M1+)",
                   "arms": arms}, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for k, v in arms.items():
        print(k, {q: v.get(q) for q in ("n_train", "n_test", "base_rate_test", "auc", "auc_ci", "tss", "tss_ci")})
    print(f"wrote {a.out_dir}/f3_benchmark.json")


def cmd_robust(a):
    """F3-R1/R2 (phase_13 F3 addendum): same-window comparison, per-seed arms, reverse
    direction, and a >= C1-label variant. The primary 9.6 numbers are not touched."""
    flares = load_flares()
    cols = lambda feats: [f"{f}_{s}" for f in feats for s in ("max", "mean", "last")]
    vf = list(VEC_FEATURES)
    out = {}
    tr_v = label_windows(build_windows(vec_frames_c24(a.pilot_features), vf), flares)
    te_v = label_windows(build_windows(vec_frames_c23(a.c23_features), vf), flares)
    tr_l = label_windows(build_windows(los_frames_c24(os.path.join(a.out_dir, "c24_los_keywords.csv")),
                                       list(LOS_FEATURES)), flares)
    te_l = label_windows(build_windows(los_frames_c23(a.frames_all), list(LOS_FEATURES)), flares)
    # R1: evaluate both arms on the intersection of test windows (same regions, same t_end)
    kv = set(map(tuple, te_v[["region", "t_end"]].astype(str).to_numpy().tolist()))
    kl = set(map(tuple, te_l[["region", "t_end"]].astype(str).to_numpy().tolist()))
    both = kv & kl
    sel = lambda d: d[[tuple(x) in both for x in d[["region", "t_end"]].astype(str).to_numpy().tolist()]]
    out["same_window"] = {"n_windows": len(both),
                          "vector": fit_eval(tr_v, sel(te_v), cols(vf), n_boot=a.n_boot),
                          "los_record": fit_eval(tr_l, sel(te_l), cols(LOS_FEATURES), n_boot=a.n_boot)}
    # R2a: per-seed vector arms
    df = pd.read_csv(a.c23_features)
    pilot = pd.read_csv(a.pilot_features)
    from c23_screening import derive
    seeds = {}
    for s24, s23 in (("F_U_E3_s0_e200",) * 2, ("F_U_E3_s1_e200",) * 2, ("F_U_E3_s2",) * 2):
        from c23_screening import seed_mean
        sm24 = derive(seed_mean(pilot[pilot["mask"] == "strong_pred"], (s24,)))
        sm24["region"], sm24["_t"] = sm24.harpnum, tvalue(sm24.mdi_t_rec)
        meta = pilot[["harpnum", "mdi_t_rec", "noaa_ar"]].drop_duplicates()
        sm24 = sm24.merge(meta, on=["harpnum", "mdi_t_rec"], how="left")
        sm23 = derive(df[(df.model == s23) & (df["mask"] == "strong_pred")].copy())
        sm23["region"], sm23["_t"] = sm23.tarpnum, tvalue(sm23.t_rec)
        r = fit_eval(label_windows(build_windows(sm24, vf), flares),
                     label_windows(build_windows(sm23, vf), flares), cols(vf), n_boot=200)
        seeds[s23] = {q: r[q] for q in ("auc", "tss", "n_test")}
    out["per_seed_vector"] = seeds
    # R2b: reverse direction (train C23 -> test C24)
    out["reverse"] = {"vector": fit_eval(te_v, tr_v, cols(vf), n_boot=a.n_boot),
                      "los_record": fit_eval(te_l, tr_l, cols(LOS_FEATURES), n_boot=a.n_boot)}
    # R2c: >= C1 labels
    trc = label_windows(build_windows(vec_frames_c24(a.pilot_features), vf), flares, wm2_min=1e-6)
    tec = label_windows(build_windows(vec_frames_c23(a.c23_features), vf), flares, wm2_min=1e-6)
    trcl = label_windows(build_windows(los_frames_c24(os.path.join(a.out_dir, "c24_los_keywords.csv")),
                                       list(LOS_FEATURES)), flares, wm2_min=1e-6)
    tecl = label_windows(build_windows(los_frames_c23(a.frames_all), list(LOS_FEATURES)), flares, wm2_min=1e-6)
    out["ge_c1"] = {"vector": fit_eval(trc, tec, cols(vf), n_boot=a.n_boot),
                    "los_record": fit_eval(trcl, tecl, cols(LOS_FEATURES), n_boot=a.n_boot)}
    with open(os.path.join(a.out_dir, "f3_robustness.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for k in ("same_window", "reverse", "ge_c1"):
        v = out[k]
        print(k, {arm: {q: round(v[arm][q], 3) for q in ("auc", "tss") if q in v[arm] and np.isfinite(v[arm][q])}
                  for arm in ("vector", "los_record")})
    print("per_seed:", {k2: {q: round(x, 3) for q, x in v.items() if q != "n_test"} for k2, v in seeds.items()})
    print(f"wrote {a.out_dir}/f3_robustness.json")


VEC_B = ("USFLUX", "TOTUSJZ", "TOTUSJH", "R_VALUE")       # F3b vector set (header-consistent)
LOS_B = ("USFLUXL", "MEANGBL", "AREA_LOS", "R_VALUE_LOS")


def sharp_era_frames(path, end="2016-12-31"):
    """Training frame table from the harvested SHARP headers: QUALITY == 0, |lon| <= 65,
    through `end` (the single-catalog rule)."""
    df = pd.read_csv(path)
    q = pd.to_numeric(df.QUALITY.astype(str).str.replace("0x", "", regex=False), errors="coerce")
    df = df[(q == 0) & (df.LON_FWT.abs() <= 65) & (df.NOAA_AR > 0)].copy()
    df["_t"] = tvalue(df.T_REC)
    df = df[df["_t"] <= pd.Timestamp(end)]
    df = df.rename(columns={"HARPNUM": "region", "NOAA_AR": "noaa_ar",
                            "AREA": "AREA_LOS", "R_VALUE": "R_VALUE_LOS"})
    df["R_VALUE"] = df["R_VALUE_LOS"]                      # header R_VALUE serves both arms
    return df


def cmd_runb(a):
    """F3b: full-era-trained benchmark (train SHARP headers 2010-2016, test C23; per-era z)."""
    flares = load_flares(years=range(1996, 2017))
    print(f"{len(flares)} flares (1996-2016), {int((flares.wm2 >= M1).sum())} >= M1")
    era = sharp_era_frames(a.sharp_era)
    cols = lambda feats: [f"{f}_{s}" for f in feats for s in ("max", "mean", "last")]
    out = {}
    tr_v = label_windows(build_windows(era, list(VEC_B)), flares)
    te_v = label_windows(build_windows(vec_frames_c23(a.c23_features), list(VEC_B)), flares)
    print(f"vector: {len(tr_v)} train windows ({int(tr_v.label.sum())} positive), {len(te_v)} test")
    out["vector"] = fit_eval(tr_v, te_v, cols(VEC_B), n_boot=a.n_boot, per_set_scale=True)
    tr_l = label_windows(build_windows(era, list(LOS_B)), flares)
    te_l = label_windows(build_windows(los_frames_c23(a.frames_all), list(LOS_B)), flares)
    print(f"los: {len(tr_l)} train windows ({int(tr_l.label.sum())} positive), {len(te_l)} test")
    out["los_record"] = fit_eval(tr_l, te_l, cols(LOS_B), n_boot=a.n_boot, per_set_scale=True)
    with open(os.path.join(a.out_dir, "f3b_fullera.json"), "w") as f:
        json.dump({"design": "phase_13 F3b: train SHARP headers 2010-05..2016-12, test C23, per-era z",
                   "arms": out}, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for k, v in out.items():
        print(k, {q: v.get(q) for q in ("n_train", "n_test", "base_rate_train", "auc", "auc_ci", "tss", "tss_ci")})
    print(f"wrote {a.out_dir}/f3b_fullera.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c2 = sub.add_parser("c24los")
    c2.add_argument("--out_dir", default=os.path.join(ROOT, "results", "full", "c23_build"))
    c2.set_defaults(fn=cmd_c24los)
    rn = sub.add_parser("run")
    rn.add_argument("--pilot_features", default=os.path.join(ROOT, "results", "full", "overlap_pilot", "features.csv"))
    rn.add_argument("--c23_features", default=os.path.join(ROOT, "results", "full", "c23_build", "features.csv"))
    rn.add_argument("--frames_all", default=os.path.join(ROOT, "data", "c23", "frames_all.csv"))
    rn.add_argument("--out_dir", default=os.path.join(ROOT, "results", "full", "c23_build"))
    rn.set_defaults(fn=cmd_run)
    rb = sub.add_parser("robust")
    rb.add_argument("--pilot_features", default=os.path.join(ROOT, "results", "full", "overlap_pilot", "features.csv"))
    rb.add_argument("--c23_features", default=os.path.join(ROOT, "results", "full", "c23_build", "features.csv"))
    rb.add_argument("--frames_all", default=os.path.join(ROOT, "data", "c23", "frames_all.csv"))
    rb.add_argument("--out_dir", default=os.path.join(ROOT, "results", "full", "c23_build"))
    rb.add_argument("--n_boot", type=int, default=1000)
    rb.set_defaults(fn=cmd_robust)
    rc = sub.add_parser("runb")
    rc.add_argument("--sharp_era", default=os.path.join(ROOT, "data", "c23", "sharp_era.csv"))
    rc.add_argument("--c23_features", default=os.path.join(ROOT, "results", "full", "c23_build", "features.csv"))
    rc.add_argument("--frames_all", default=os.path.join(ROOT, "data", "c23", "frames_all.csv"))
    rc.add_argument("--out_dir", default=os.path.join(ROOT, "results", "full", "c23_build"))
    rc.add_argument("--n_boot", type=int, default=1000)
    rc.set_defaults(fn=cmd_runb)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
