"""Phase 15 / reports/phase_14.md section 3: the full flare-verification skill-score suite,
and a Bobra & Couvidat (2015, ApJ 798, 135)-style point-in-time analysis of the record.

    python scripts/c23_skill_scores.py suite   # re-score every frozen benchmark arm
    python scripts/c23_skill_scores.py bobra   # B&C sampling + LR/SVM + Fisher ranking

Frozen pipeline files are imported, never modified; outputs to results/full/skill_scores/.
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
from c23_benchmark import (                                   # noqa: E402
    LOS_B, LOS_FEATURES, M1, VEC_B, VEC_FEATURES, build_windows, label_windows, load_flares,
    los_frames_c23, los_frames_c24, peak_tss_threshold, sharp_era_frames, tvalue,
    vec_frames_c23, vec_frames_c24)
from c23_screening import transform                    # noqa: E402

OUT = os.path.join(ROOT, "results", "full", "skill_scores")
C23F = os.path.join(ROOT, "results", "full", "c23_build", "features.csv")
PILOT = os.path.join(ROOT, "results", "full", "overlap_pilot", "features.csv")
FRAMES_ALL = os.path.join(ROOT, "data", "c23", "frames_all.csv")
C24_LOS = os.path.join(ROOT, "results", "full", "c23_build", "c24_los_keywords.csv")
SHARP_ERA = os.path.join(ROOT, "data", "c23", "sharp_era.csv")
FROZEN = os.path.join(ROOT, "results", "full", "c23_build")

cols = lambda f: [f"{x}_{s}" for x in f for s in ("max", "mean", "last")]


# ------------------------------------------------------------------ metrics

def skill_suite(y, yhat):
    """Every standard verification metric from one confusion matrix (phase_14 section 3.1)."""
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    TP = int((y & yhat).sum()); FN = int((y & ~yhat).sum())
    TN = int((~y & ~yhat).sum()); FP = int((~y & yhat).sum())
    P, N, T = TP + FN, TN + FP, TP + FN + TN + FP
    d = lambda a, b: float(a / b) if b else float("nan")
    rec, spec = d(TP, P), d(TN, N)
    prec, prec_n = d(TP, TP + FP), d(TN, TN + FN)
    f1 = d(2 * prec * rec, prec + rec) if np.isfinite(prec + rec) and (prec + rec) else float("nan")
    E = d((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP), T)
    CH = d((TP + FP) * (TP + FN), T)
    return {"TP": TP, "FN": FN, "TN": TN, "FP": FP, "P": P, "N": N,
            "imbalance_N_over_P": d(N, P),
            "accuracy": d(TP + TN, T),
            "precision_pos": prec, "precision_neg": prec_n,
            "recall_pos": rec, "recall_neg": spec,
            "f1_pos": f1,
            "TSS": rec - d(FP, N),
            "HSS1": d(TP + TN - N, P),
            "HSS2": d(TP + TN - E, T - E),
            "gilbert_skill_score": d(TP - CH, TP + FP + FN - CH)}


def boot_ci(y, p, thr, regions, keys=("TSS", "HSS2"), n_boot=1000, seed=0):
    """Bootstrap over test regions, as everywhere else in the project."""
    y, p, regions = np.asarray(y, bool), np.asarray(p), np.asarray(regions)
    uniq = pd.unique(regions)
    by = {r: np.flatnonzero(regions == r) for r in uniq}
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([by[r] for r in rng.choice(uniq, size=len(uniq), replace=True)])
        yb = y[idx]
        if not 0 < yb.mean() < 1:
            continue
        m = skill_suite(yb, p[idx] >= thr)
        for k in keys:
            acc[k].append(m[k])
    return {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None)
            for k, v in acc.items()}


def peak_metric_threshold(y, p, metric="TSS"):
    """Threshold maximizing the named skill score on the TRAINING predictions."""
    c = np.unique(p)
    if len(c) > 512:
        c = np.quantile(p, np.linspace(0, 1, 512))
    best = (-np.inf, float(c[0]))
    for thr in c:
        v = skill_suite(y, p >= thr)[metric]
        if np.isfinite(v) and v > best[0]:
            best = (v, float(thr))
    return best[1]


def fit_score(train, test, feature_cols, per_set_scale=False, model="lr", seed=0,
              thr_metric="TSS", svm_params=None):
    """Frozen-equivalent fit, returning the full suite instead of only AUC/TSS."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    Xtr = train[feature_cols].to_numpy(np.float64)
    Xte = test[feature_cols].to_numpy(np.float64)
    ytr, yte = train.label.to_numpy(bool), test.label.to_numpy(bool)
    sc = StandardScaler().fit(Xtr)
    sct = StandardScaler().fit(Xte) if per_set_scale else sc
    if model == "lr":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    else:
        sp = svm_params or {"C": 4.0, "gamma": 0.075}
        clf = SVC(class_weight="balanced", probability=True, random_state=seed, **sp)
    clf.fit(sc.transform(Xtr), ytr)
    ptr = clf.predict_proba(sc.transform(Xtr))[:, 1]
    pte = clf.predict_proba(sct.transform(Xte))[:, 1]
    thr = (peak_tss_threshold(ytr, ptr) if thr_metric == "TSS"
           else peak_metric_threshold(ytr, ptr, thr_metric))
    out = skill_suite(yte, pte >= thr)
    out["threshold_rule"] = f"peak {thr_metric} on training predictions"
    out["AUC"] = float(roc_auc_score(yte, pte)) if 0 < yte.mean() < 1 else float("nan")
    out["threshold"] = float(thr)
    out["model"] = model
    return out, pte, thr


def save(name, payload):
    os.makedirs(OUT, exist_ok=True)
    import sklearn
    payload["sklearn_version"] = sklearn.__version__
    payload["preregistration"] = "reports/phase_14.md section 3"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(payload, f, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {os.path.join(OUT, name)}")


# ------------------------------------------------------------------ suite

def cmd_suite(a):
    flares_all = load_flares(years=range(1996, 2017))
    flares_dm = load_flares()
    out = {"design": "phase_14 section 3.1: full verification suite on the frozen arms"}
    era = sharp_era_frames(SHARP_ERA)
    c23v = vec_frames_c23(C23F)
    c23l = los_frames_c23(FRAMES_ALL)

    W = lambda fr, fe, fl, **k: label_windows(build_windows(fr, list(fe)), fl, **k)
    arms = {}
    # domain-matched (F3): train recon C24 -> test C23
    tr = W(vec_frames_c24(PILOT), VEC_FEATURES, flares_dm)
    te = W(c23v, VEC_FEATURES, flares_dm)
    arms["domain_matched_vector"] = (tr, te, VEC_FEATURES, False)
    trl = W(los_frames_c24(C24_LOS), LOS_FEATURES, flares_dm)
    tel = W(c23l, LOS_FEATURES, flares_dm)
    arms["domain_matched_los"] = (trl, tel, LOS_FEATURES, False)
    # deployment (F3b): train real SHARP era -> test C23, per-era z
    arms["deployment_vector"] = (W(era, VEC_B, flares_all), W(c23v, VEC_B, flares_all),
                                 VEC_B, True)
    arms["deployment_los"] = (W(era, LOS_B, flares_all), W(c23l, LOS_B, flares_all),
                              LOS_B, True)
    res = {}
    for name, (tr_, te_, fe, pss) in arms.items():
        res[name] = {}
        for tm in ("TSS", "HSS2"):
            m, pte, thr = fit_score(tr_, te_, cols(fe), per_set_scale=pss, thr_metric=tm)
            m["ci"] = boot_ci(te_.label.to_numpy(bool), pte, thr, te_.region.to_numpy())
            res[name][f"threshold_max_{tm}"] = m
            print(f"{name:24s} [thr max {tm:4s}] TSS {m['TSS']:.4f}  HSS1 {m['HSS1']:+.3f}  "
                  f"HSS2 {m['HSS2']:.4f}  GS {m['gilbert_skill_score']:.4f}  "
                  f"AUC {m['AUC']:.4f}  N/P {m['imbalance_N_over_P']:.1f}")
    # frozen reproduction check
    fz = json.load(open(os.path.join(FROZEN, "f3_benchmark.json")))["arms"]
    fb = json.load(open(os.path.join(FROZEN, "f3b_fullera.json")))["arms"]
    checks = {"domain_matched_vector": fz["vector"], "domain_matched_los": fz["los_record"],
              "deployment_vector": fb["vector"], "deployment_los": fb["los_record"]}
    for k, ref in checks.items():
        m = res[k]["threshold_max_TSS"]
        for q, qq in (("TSS", "tss"), ("AUC", "auc")):
            assert abs(m[q] - ref[qq]) < 1e-6, f"{k}.{q} drifted: {m[q]} vs {ref[qq]}"
    out["arms"] = res
    out["frozen_reproduction"] = "TSS and AUC reproduce f3_benchmark.json / f3b_fullera.json to 1e-6"
    save("suite.json", out)


# ------------------------------------------------------------------ bobra

BC_SHARED = ("USFLUX", "TOTUSJZ", "TOTUSJH", "ABSNJZH", "SAVNCPP", "R_VALUE", "AREA")
BC_EXTRA = ("MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZD", "MEANALP", "MEANJZH",
            "STRONG_FRAC")
BC_ABSENT = ("TOTPOT", "MEANPOT", "MEANSHR", "SHRGT45", "TOTBSQ", "TOTFX", "TOTFY", "TOTFZ",
             "EPSX", "EPSY", "EPSZ")
NP_RATIO = 16.5
TOL_MIN = 48


def bc_samples(frames, flares, mode, seed=0):
    """B&C sampling: positives 24 h before each >=M1 peak; negatives at random quiet times."""
    rng = np.random.default_rng(seed)
    fl = flares[flares.wm2 >= M1]
    fr = frames.sort_values(["region", "_t"]).reset_index(drop=True)
    by_noaa = {int(n): g.peak.to_numpy() for n, g in fl.groupby("noaa_ar")}
    tol = np.timedelta64(TOL_MIN, "m")
    pos_idx, used = [], set()
    for noaa, peaks in by_noaa.items():
        g = fr[fr.noaa_ar == noaa]
        if g.empty:
            continue
        t = g["_t"].to_numpy()
        for pk in peaks:
            target = np.datetime64(pk) - np.timedelta64(24, "h")
            j = int(np.argmin(np.abs(t - target)))
            if abs(t[j] - target) <= tol and g.index[j] not in used:
                used.add(g.index[j]); pos_idx.append(g.index[j])
    # negatives: no >=M1 flare in the forward (operational) or symmetric (segmented) window
    back = np.timedelta64(48, "h") if mode == "segmented" else np.timedelta64(0, "h")
    fwd = np.timedelta64(48, "h") if mode == "segmented" else np.timedelta64(24, "h")
    cand = []
    for r, g in fr.groupby("region"):
        noaa = int(g.noaa_ar.iloc[0])
        peaks = by_noaa.get(noaa)
        tt = g["_t"].to_numpy()
        if peaks is None:
            cand.extend(g.index.tolist()); continue
        pk = np.array([np.datetime64(p) for p in peaks])
        ok = ~np.any((pk[None, :] >= tt[:, None] - back) & (pk[None, :] <= tt[:, None] + fwd), 1)
        cand.extend(g.index[ok].tolist())
    cand = [i for i in cand if i not in used]
    n_neg = int(round(NP_RATIO * len(pos_idx)))
    neg_idx = list(rng.choice(cand, size=min(n_neg, len(cand)), replace=False))
    s = fr.loc[pos_idx + neg_idx].copy()
    s["label"] = [True] * len(pos_idx) + [False] * len(neg_idx)
    return s.reset_index(drop=True)


def fisher(X, y):
    """B&C Eq. (1): univariate Fisher score per feature."""
    X = np.asarray(X, np.float64); y = np.asarray(y, bool)
    out = []
    for j in range(X.shape[1]):
        v = X[:, j]; m = np.isfinite(v)
        vp, vn, va = v[m & y], v[m & ~y], v[m]
        if len(vp) < 2 or len(vn) < 2:
            out.append(np.nan); continue
        num = (vp.mean() - va.mean()) ** 2 + (vn.mean() - va.mean()) ** 2
        den = vp.var(ddof=1) + vn.var(ddof=1)
        out.append(float(num / den) if den > 0 else np.nan)
    return np.array(out)


def cmd_bobra(a):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    flares = load_flares(years=range(1996, 2012))
    frames = vec_frames_c23(C23F)
    feats = [f for f in BC_SHARED + BC_EXTRA if f in frames.columns]
    print(f"features available: {len(feats)} of B&C's 25; absent (need potential field or "
          f"Lorentz force): {len(BC_ABSENT)}")
    out = {"design": "phase_14 section 3.2 (B&C 2015-style point-in-time analysis)",
           "features_used": feats, "features_absent_vs_bc": list(BC_ABSENT),
           "np_ratio_target": NP_RATIO, "modes": {}}
    for mode in ("operational", "segmented"):
        s = bc_samples(frames, flares, mode)
        X = np.column_stack([transform(f, s[f].to_numpy()) for f in feats])
        y = s.label.to_numpy(bool)
        keep = np.isfinite(X).all(1)
        X, y, grp = X[keep], y[keep], s.noaa_ar.to_numpy()[keep]
        med, sd = np.nanmedian(X, 0), np.nanstd(X, 0)
        Xs = (X - med) / np.where(sd > 0, sd, 1.0)      # B&C scaling: median / sd
        F = fisher(Xs, y)
        order = np.argsort(-np.nan_to_num(F, nan=-1))
        print(f"\n[{mode}] {int(y.sum())} positives, {int((~y).sum())} negatives, "
              f"N/P {(~y).sum()/max(y.sum(),1):.1f}")
        print("  Fisher ranking:", ", ".join(f"{feats[i]} {F[i]:.3f}" for i in order[:8]))
        res = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()),
               "fisher": {feats[i]: (None if not np.isfinite(F[i]) else round(float(F[i]), 1))
                          for i in order},
               "splits": {}}
        for split in ("region_disjoint", "random"):
            if split == "random":
                tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0,
                                          stratify=y)
            else:
                tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                              .split(X, y, groups=grp))
            d = {}
            trd = pd.DataFrame(Xs[tr], columns=feats); trd["label"] = y[tr]
            ted = pd.DataFrame(Xs[te], columns=feats); ted["label"] = y[te]
            # SVM hyperparameters: B&C's values, plus a small grid chosen INSIDE the
            # training split only (their C=4, gamma=0.075 were tuned on their feature set)
            itr, ite = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
                            .split(Xs[tr], y[tr], groups=grp[tr]))
            i_tr = pd.DataFrame(Xs[tr][itr], columns=feats); i_tr["label"] = y[tr][itr]
            i_te = pd.DataFrame(Xs[tr][ite], columns=feats); i_te["label"] = y[tr][ite]
            best = (-np.inf, {"C": 4.0, "gamma": 0.075})
            for C in (1.0, 4.0, 16.0):
                for gam in (0.02, 0.075, 0.2):
                    try:
                        mm, _, _ = fit_score(i_tr, i_te, feats, model="svm",
                                             svm_params={"C": C, "gamma": gam})
                    except Exception:
                        continue
                    if np.isfinite(mm["TSS"]) and mm["TSS"] > best[0]:
                        best = (mm["TSS"], {"C": C, "gamma": gam})
            d["svm_tuned_params"] = best[1]
            for model, sp in (("lr", None), ("svm_bc_params", {"C": 4.0, "gamma": 0.075}),
                              ("svm_tuned", best[1])):
                for tm in ("TSS", "HSS2"):
                    m, pte, thr = fit_score(trd, ted, feats,
                                            model="lr" if model == "lr" else "svm",
                                            svm_params=sp, thr_metric=tm)
                    m["ci"] = boot_ci(y[te], pte, thr, grp[te])
                    d[f"{model}__thr_max_{tm}"] = m
                    print(f"  {split:15s} {model:14s} thr={tm:4s}  TSS {m['TSS']:.3f}  "
                          f"HSS1 {m['HSS1']:+.3f}  HSS2 {m['HSS2']:.3f}  "
                          f"GS {m['gilbert_skill_score']:.3f}  recall {m['recall_pos']:.3f}  "
                          f"prec {m['precision_pos']:.3f}")
            res["splits"][split] = d
        out["modes"][mode] = res
    rows = [{"mode": m, "feature": f, "fisher": v}
            for m, r in out["modes"].items() for f, v in r["fisher"].items()]
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "fisher_ranking.csv"), index=False)
    save("bobra.json", out)


def cmd_repeat(a):
    """Bobra-style repeated resampling: N region-disjoint splits, mean +- sd per metric.
    Uses decision_function / decision margin for thresholding (no Platt scaling), which is
    equivalent for rank-based skill scores and much faster."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.svm import SVC
    flares = load_flares(years=range(1996, 2012))
    frames = vec_frames_c23(C23F)
    feats = [f for f in BC_SHARED + BC_EXTRA if f in frames.columns]
    out = {"design": "phase_14 section 3.2 addendum: %d repeated %s splits" % (a.n_splits, a.split),
           "n_splits": a.n_splits, "features_used": feats, "modes": {}}
    for mode in ("operational", "segmented"):
        s = bc_samples(frames, flares, mode)
        X = np.column_stack([transform(f, s[f].to_numpy()) for f in feats])
        y = s.label.to_numpy(bool); keep = np.isfinite(X).all(1)
        X, y, grp = X[keep], y[keep], s.noaa_ar.to_numpy()[keep]
        med, sd = np.nanmedian(X, 0), np.nanstd(X, 0)
        X = (X - med) / np.where(sd > 0, sd, 1.0)
        acc = {}
        if a.split == "random":
            from sklearn.model_selection import StratifiedShuffleSplit
            gss = StratifiedShuffleSplit(n_splits=a.n_splits, test_size=0.3, random_state=0)
        else:
            gss = GroupShuffleSplit(n_splits=a.n_splits, test_size=0.3, random_state=0)
        for i, (tr, te) in enumerate(gss.split(X, y, groups=grp)):
            for name, clf in (("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                                        max_iter=2000)),
                              ("svm", SVC(C=4.0, gamma=0.02, class_weight="balanced"))):
                try:
                    clf.fit(X[tr], y[tr])
                    str_, ste = clf.decision_function(X[tr]), clf.decision_function(X[te])
                except Exception:
                    continue
                for tm in ("TSS", "HSS2"):
                    thr = peak_metric_threshold(y[tr], str_, tm)
                    m = skill_suite(y[te], ste >= thr)
                    for k in ("TSS", "HSS2", "HSS1", "gilbert_skill_score", "recall_pos",
                              "precision_pos"):
                        acc.setdefault(f"{name}__thr_max_{tm}", {}).setdefault(k, []).append(m[k])
            if (i + 1) % 25 == 0:
                print(f"  [{mode}] {i+1}/{a.n_splits}", flush=True)
        res = {}
        for arm, d in acc.items():
            res[arm] = {k: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                            "p5": float(np.nanpercentile(v, 5)),
                            "p95": float(np.nanpercentile(v, 95)), "n": len(v)}
                        for k, v in d.items()}
            r = res[arm]
            print(f"[{mode}] {arm:22s} TSS {r['TSS']['mean']:.3f} +- {r['TSS']['sd']:.3f}   "
                  f"HSS2 {r['HSS2']['mean']:.3f} +- {r['HSS2']['sd']:.3f}")
        out["modes"][mode] = res
    out["split_rule"] = a.split
    save(a.out, out)


SHARED5 = ("USFLUX", "TOTUSJZ", "TOTUSJH", "R_VALUE", "AREA")


def real_hmi_frames(end="2016-12-31"):
    """Real SHARP header keywords as a B&C-style frame table (phase_14 section 3.3).
    Note: read straight from sharp_era.csv so AREA/R_VALUE keep their SHARP-header meaning
    (sharp_era_frames() renames them for the LOS benchmark arm)."""
    df = pd.read_csv(os.path.join(ROOT, "data", "c23", "sharp_era.csv"))
    q = pd.to_numeric(df.QUALITY.astype(str).str.replace("0x", "", regex=False), errors="coerce")
    df = df[(q == 0) & (df.LON_FWT.abs() <= 65) & (df.NOAA_AR > 0)].copy()
    df["_t"] = tvalue(df.T_REC)
    df = df[df["_t"] <= pd.Timestamp(end)]
    df["region"] = df.HARPNUM
    df["noaa_ar"] = df.NOAA_AR
    return df


def cmd_realhmi(a):
    """Same pipeline, same features, same protocol: real HMI vs this record (section 3.3)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.svm import SVC
    feats = list(SHARED5)
    sources = {"REAL_hmi": (real_hmi_frames(), range(1996, 2017)),
               "RECON_c23": (vec_frames_c23(C23F), range(1996, 2012))}
    out = {"design": "phase_14 section 3.3: real HMI vs reconstruction, identical 5 features",
           "features": feats, "n_splits": a.n_splits, "modes": {}}
    for mode in ("operational", "segmented"):
        out["modes"][mode] = {}
        for name, (frames, yrs) in sources.items():
            fl = load_flares(years=yrs)
            miss = [f for f in feats if f not in frames.columns]
            assert not miss, f"{name} missing {miss}"
            s = bc_samples(frames, fl, mode)
            X = np.column_stack([transform(f, s[f].to_numpy()) for f in feats])
            y = s.label.to_numpy(bool); keep = np.isfinite(X).all(1)
            X, y, grp = X[keep], y[keep], s.noaa_ar.to_numpy()[keep]
            med, sd = np.nanmedian(X, 0), np.nanstd(X, 0)
            X = (X - med) / np.where(sd > 0, sd, 1.0)
            acc = {}
            for tr, te in GroupShuffleSplit(n_splits=a.n_splits, test_size=0.3,
                                            random_state=0).split(X, y, groups=grp):
                if y[te].sum() < 8 or y[tr].sum() < 8:
                    continue
                for mdl, clf in (("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                                           max_iter=2000)),
                                 ("svm", SVC(C=4.0, gamma=0.02, class_weight="balanced"))):
                    clf.fit(X[tr], y[tr])
                    str_, ste = clf.decision_function(X[tr]), clf.decision_function(X[te])
                    thr = peak_tss_threshold(y[tr], str_)
                    m = skill_suite(y[te], ste >= thr)
                    for k in ("TSS", "HSS2"):
                        acc.setdefault(mdl, {}).setdefault(k, []).append(m[k])
            res = {"n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                   "np_ratio": round(float((~y).sum() / max(y.sum(), 1)), 2)}
            for mdl, d in acc.items():
                res[mdl] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
                            for k, v in d.items()}
            out["modes"][mode][name] = res
            print(f"[{mode:12s}] {name:10s} P={res['n_pos']:4d} N={res['n_neg']:5d} "
                  f"(N/P {res['np_ratio']}) | "
                  + " | ".join(f"{m} TSS {res[m]['TSS']['mean']:.3f}+-{res[m]['TSS']['sd']:.3f}"
                               for m in ("lr", "svm") if m in res))
        r = out["modes"][mode]
        if "REAL_hmi" in r and "RECON_c23" in r:
            for mdl in ("lr", "svm"):
                if mdl in r["REAL_hmi"] and mdl in r["RECON_c23"]:
                    d = r["REAL_hmi"][mdl]["TSS"]["mean"] - r["RECON_c23"][mdl]["TSS"]["mean"]
                    r.setdefault("cost_of_reconstruction_TSS", {})[mdl] = round(float(d), 4)
                    print(f"    -> {mdl}: real minus reconstruction = {d:+.3f} TSS")
    out["caveat"] = ("Five features only (both sides); B&C selected 13, including TOTBSQ and "
                     "TOTPOT which this record cannot compute. The REAL arm is therefore NOT "
                     "expected to reach their published 0.761 and its shortfall is not evidence "
                     "about our pipeline (phase_14 section 3.3 reading b).")
    save("realhmi_control.json", out)


BC_TOP13_JSOC = ("TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX", "AREA_ACR",
                 "MEANPOT", "R_VALUE", "SHRGT45")          # 10 of their selected 13
BC18_JSOC = BC_TOP13_JSOC + ("MEANSHR", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZH",
                             "MEANJZD", "MEANALP")


def bobra_window_frames(end="2014-05-31"):
    """Real HMI keywords over B&C's own window, harvested by c23_bobra_harvest.py."""
    df = pd.read_csv(os.path.join(ROOT, "data", "c23", "bobra_era_96m.csv"))
    q = pd.to_numeric(df.QUALITY.astype(str).str.replace("0x", "", regex=False), errors="coerce")
    df = df[(q == 0) & (df.LON_FWT.abs() <= 65) & (df.NOAA_AR > 0)].copy()
    df["_t"] = tvalue(df.T_REC)
    df = df[df["_t"] <= pd.Timestamp(end)]
    df["region"], df["noaa_ar"] = df.HARPNUM, df.NOAA_AR
    return df


def cmd_herfeatures(a):
    """Does OUR pipeline reach B&C's published score when given (most of) THEIR features on
    THEIR instrument in THEIR window? (phase_14 section 3.4)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.svm import SVC
    frames = bobra_window_frames()
    fl = load_flares(years=range(2010, 2015))
    sets = {"their_top13_available_10": list(BC_TOP13_JSOC),
            "all_18_jsoc": list(BC18_JSOC),
            "our_5_shared": ["USFLUX", "TOTUSJZ", "TOTUSJH", "R_VALUE", "AREA_ACR"]}
    out = {"design": "phase_14 3.4: our pipeline, real HMI, B&C window and features",
           "window": "2010-05..2014-05", "n_splits": a.n_splits,
           "published_bobra": {"operational_TSS": 0.761, "segmented_TSS": 0.817},
           "feature_sets": {k: v for k, v in sets.items()}, "modes": {}}
    for mode in ("operational", "segmented"):
        s = bc_samples(frames, fl, mode)
        out["modes"][mode] = {"n_pos": int(s.label.sum()), "n_neg": int((~s.label).sum())}
        print(f"\n[{mode}] {int(s.label.sum())} positives, {int((~s.label).sum())} negatives")
        for name, feats in sets.items():
            X = np.column_stack([transform(f.replace("AREA_ACR", "AREA"), s[f].to_numpy())
                                 for f in feats])
            y = s.label.to_numpy(bool); keep = np.isfinite(X).all(1)
            X, y, grp = X[keep], y[keep], s.noaa_ar.to_numpy()[keep]
            med, sd = np.nanmedian(X, 0), np.nanstd(X, 0)
            X = (X - med) / np.where(sd > 0, sd, 1.0)
            acc = {}
            for tr, te in GroupShuffleSplit(n_splits=a.n_splits, test_size=0.3,
                                            random_state=0).split(X, y, groups=grp):
                if y[te].sum() < 8 or y[tr].sum() < 8:
                    continue
                for mdl, clf in (("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                                           max_iter=5000)),
                                 ("svm", SVC(C=4.0, gamma=0.02, class_weight="balanced"))):
                    clf.fit(X[tr], y[tr])
                    thr = peak_tss_threshold(y[tr], clf.decision_function(X[tr]))
                    m = skill_suite(y[te], clf.decision_function(X[te]) >= thr)
                    acc.setdefault(mdl, []).append(m["TSS"])
            res = {"n_features": len(feats)}
            for mdl, v in acc.items():
                res[mdl] = {"TSS_mean": float(np.mean(v)), "TSS_sd": float(np.std(v)), "n": len(v)}
            out["modes"][mode][name] = res
            print(f"  {name:26s} ({len(feats):2d} feats) " + "  ".join(
                f"{mdl} {res[mdl]['TSS_mean']:.3f}+-{res[mdl]['TSS_sd']:.3f}"
                for mdl in ("lr", "svm") if mdl in res))
        pub = out["published_bobra"][f"{mode}_TSS"]
        print(f"  {'Bobra published (13 feats)':26s}        {pub:.3f}")
    save("her_features.json", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, fn in (("suite", cmd_suite), ("bobra", cmd_bobra)):
        sub.add_parser(n).set_defaults(fn=fn)
    rp = sub.add_parser("repeat")
    rp.add_argument("--n_splits", type=int, default=100)
    rp.add_argument("--split", choices=("region_disjoint", "random"), default="region_disjoint")
    rp.add_argument("--out", default="repeat.json")
    hf = sub.add_parser("herfeatures")
    hf.add_argument("--n_splits", type=int, default=50)
    hf.set_defaults(fn=cmd_herfeatures)
    rh = sub.add_parser("realhmi")
    rh.add_argument("--n_splits", type=int, default=50)
    rh.set_defaults(fn=cmd_realhmi)
    rp.set_defaults(fn=cmd_repeat)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
