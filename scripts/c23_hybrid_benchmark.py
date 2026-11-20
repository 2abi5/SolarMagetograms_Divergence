"""Phase 14 / F3c: hybrid real+synthetic training benchmark (reports/phase_14.md section 1).

    python scripts/c23_hybrid_benchmark.py gate     # H0: reproduce frozen F3/F3b; mixed-path equality
    python scripts/c23_hybrid_benchmark.py real     # H1: real-era augmentation, test held-out real 2014-16
    python scripts/c23_hybrid_benchmark.py deploy   # H2: deployment augmentation, test recon C23

Pre-registered in reports/phase_14.md before the first run. Frozen pipeline files are imported,
never modified; all outputs go to results/full/hybrid/.
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
    LOS_B, LOS_FEATURES, M1, VEC_B, VEC_FEATURES, build_windows, fit_eval, label_windows,
    load_flares, los_frames_c23, los_frames_c24, peak_tss_threshold, sharp_era_frames, tss,
    vec_frames_c23, vec_frames_c24)

SHARP_ERA = os.path.join(ROOT, "data", "c23", "sharp_era.csv")
C23_FEATURES = os.path.join(ROOT, "results", "full", "c23_build", "features.csv")
PILOT_FEATURES = os.path.join(ROOT, "results", "full", "overlap_pilot", "features.csv")
FRAMES_ALL = os.path.join(ROOT, "data", "c23", "frames_all.csv")
C24_LOS = os.path.join(ROOT, "results", "full", "c23_build", "c24_los_keywords.csv")
FROZEN_DIR = os.path.join(ROOT, "results", "full", "c23_build")
OUT_DIR = os.path.join(ROOT, "results", "full", "hybrid")

PRIMARY_BOUNDARY = "2013-12-31"     # train t_end <= boundary - 24 h (label purge)
SECONDARY_BOUNDARY = "2014-12-31"
TEST_END = "2016-12-30 23:59:59"    # label horizon stays inside the 1996-2016 catalog
C23_TRAIN_CUT = "2010-05-01"        # recon C23 training rows must predate the SHARP era
OVERLAP_END = "2010-11-01"          # C23 record coverage ends 2010-10-27
EQUIV_MARGIN = 0.05                 # pre-registered TSS equivalence margin (parity rule)

cols = lambda feats: [f"{f}_{s}" for f in feats for s in ("max", "mean", "last")]


def windows(frames, feats, flares):
    return label_windows(build_windows(frames, list(feats)), flares)


def prep(wins, source, prefix):
    """Tag a window table with its source and namespace its region ids before any concat."""
    w = wins.copy()
    w["source"] = source
    w["region"] = prefix + w["region"].astype(str)
    return w


def split_real(wins, boundary, test_end=TEST_END, purge_h=24.0):
    """Time split with a label purge: train t_end <= boundary - purge; test t_end in
    (boundary, test_end] minus windows whose noaa_ar appears in train."""
    b = pd.Timestamp(boundary)
    train = wins[wins.t_end <= b - pd.Timedelta(hours=purge_h)].copy()
    pool = wins[(wins.t_end > b) & (wins.t_end <= pd.Timestamp(test_end))].copy()
    shared = pool.noaa_ar.isin(set(train.noaa_ar))
    return train, pool[~shared].copy(), int(shared.sum())


def _z_train(X, src, mode):
    """Standardize the training block: one scaler per source ('per_source') or pooled."""
    from sklearn.preprocessing import StandardScaler
    if mode == "pooled":
        sc = StandardScaler().fit(X)
        return sc.transform(X), {"pooled": sc}
    out, scalers = np.empty_like(X), {}
    for s in np.unique(src):
        m = src == s
        scalers[s] = StandardScaler().fit(X[m])
        out[m] = scalers[s].transform(X[m])
    return out, scalers


def fit_predict(train, test, feature_cols, w_syn=1.0, z="per_source", thr_rows="real",
                test_z="self", model="lr"):
    """One arm: fit on the (possibly mixed) train table, predict the test table.
    Returns (point-metric dict, test probabilities, threshold). Train rows need columns
    feature_cols + label + source; test needs feature_cols + label + region."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    Xtr_raw = train[feature_cols].to_numpy(np.float64)
    ytr = train.label.to_numpy(bool)
    src = train.source.to_numpy()
    Xtr, scalers = _z_train(Xtr_raw, src, z)
    Xte_raw = test[feature_cols].to_numpy(np.float64)
    if test_z == "self":                                   # frozen per-era rule
        Xte = StandardScaler().fit(Xte_raw).transform(Xte_raw)
    else:                                                  # 'train_real' ablation
        Xte = scalers["real" if "real" in scalers else "pooled"].transform(Xte_raw)
    yte = test.label.to_numpy(bool)
    sw = np.where(src == "real", 1.0, float(w_syn))
    assert (sw > 0).all(), "w=0 must be row exclusion, never zero weights (phase_14 rule)"
    if model == "lr":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        clf.fit(Xtr, ytr, sample_weight=None if np.all(sw == 1.0) else sw)
        coef = {c: float(v) for c, v in zip(feature_cols, clf.coef_[0])}
    else:                                                  # capacity check (appendix)
        n, counts = len(ytr), np.bincount(ytr.astype(int), minlength=2)
        cw = n / (2.0 * np.maximum(counts, 1))
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        clf.fit(Xtr, ytr, sample_weight=sw * np.where(ytr, cw[1], cw[0]))
        coef = None
    ptr = clf.predict_proba(Xtr)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    tm = (src == "real") if (thr_rows == "real" and (src == "real").any()) else np.ones(len(ytr), bool)
    thr = peak_tss_threshold(ytr[tm], ptr[tm])
    real, syn = src == "real", src != "real"
    point = {"n_train": int(len(ytr)), "n_train_real": int(real.sum()), "n_train_syn": int(syn.sum()),
             "pos_real": int(ytr[real].sum()), "pos_syn": int(ytr[syn].sum()),
             "n_test": int(len(yte)), "base_rate_test": float(yte.mean()),
             "w_syn": float(w_syn), "z": z, "thr_rows": thr_rows, "test_z": test_z, "model": model,
             "auc": float(roc_auc_score(yte, pte)), "tss": tss(yte, pte, thr),
             "threshold": float(thr), "coef": coef}
    return point, pte, thr


def boot_all(test, arms, n_boot=2000, seed=0):
    """Bootstrap over test regions, SHARED resamples across arms (paired deltas).
    arms: {name: (pte, thr)} -> ({name: {'auc': [...], 'tss': [...]}}, skipped)."""
    from sklearn.metrics import roc_auc_score
    yte = test.label.to_numpy(bool)
    regions = test.region.unique()
    by = {r: np.flatnonzero((test.region == r).to_numpy()) for r in regions}
    rng = np.random.default_rng(seed)
    res = {k: {"auc": [], "tss": []} for k in arms}
    skipped = 0
    for _ in range(n_boot):
        idx = np.concatenate([by[r] for r in rng.choice(regions, size=len(regions), replace=True)])
        yb = yte[idx]
        if not 0 < yb.mean() < 1:
            skipped += 1
            continue
        for k, (p, thr) in arms.items():
            res[k]["auc"].append(float(roc_auc_score(yb, p[idx])))
            res[k]["tss"].append(tss(yb, p[idx], thr))
    return res, skipped


def ci(xs):
    return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if len(xs) else None


def paired_delta(res, b, a):
    """Per-resample paired differences arm b minus arm a (shared resamples)."""
    out = {}
    for m in ("tss", "auc"):
        d = np.asarray(res[b][m]) - np.asarray(res[a][m])
        out[f"d{m}"] = float(np.mean(d))
        out[f"d{m}_ci"] = ci(d)
    return out


def verdict_h1(d):
    """Pre-registered H1 decision rules (phase_14 section 1.2)."""
    tlo, thi = d["dtss_ci"]
    alo, _ = d["dauc_ci"]
    if tlo > 0 and alo > 0:
        return "IMPROVEMENT"
    if -EQUIV_MARGIN <= tlo and thi <= EQUIV_MARGIN:
        return "PARITY (no-harm, within +-0.05)"
    if thi < 0:
        return "DEGRADATION"
    return "INCONCLUSIVE at this sample size"


def save(name, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    import sklearn
    payload["sklearn_version"] = sklearn.__version__
    payload["preregistration"] = "reports/phase_14.md"
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {path}")


# ---------------------------------------------------------------- H0 gate

def cmd_gate(a):
    flares = load_flares(years=range(1996, 2017))
    era = sharp_era_frames(SHARP_ERA)
    frozen_b = json.load(open(os.path.join(FROZEN_DIR, "f3b_fullera.json")))["arms"]
    frozen_f3 = json.load(open(os.path.join(FROZEN_DIR, "f3_benchmark.json")))["arms"]
    checks = {}

    tr_v = windows(era, VEC_B, flares)
    te_v = windows(vec_frames_c23(C23_FEATURES), VEC_B, flares)
    r = fit_eval(tr_v, te_v, cols(VEC_B), n_boot=10, per_set_scale=True)
    checks["f3b_vector"] = {k: [r[k], frozen_b["vector"][k]] for k in ("auc", "tss", "threshold")}
    tr_l = windows(era, LOS_B, flares)
    te_l = windows(los_frames_c23(FRAMES_ALL), LOS_B, flares)
    r = fit_eval(tr_l, te_l, cols(LOS_B), n_boot=10, per_set_scale=True)
    checks["f3b_los"] = {k: [r[k], frozen_b["los_record"][k]] for k in ("auc", "tss", "threshold")}

    tr_dm = windows(vec_frames_c24(PILOT_FEATURES), VEC_FEATURES, flares)
    te_dm = windows(vec_frames_c23(C23_FEATURES), VEC_FEATURES, flares)
    r = fit_eval(tr_dm, te_dm, cols(VEC_FEATURES), n_boot=10)
    checks["f3_vector"] = {k: [r[k], frozen_f3["vector"][k]] for k in ("auc", "tss", "threshold")}
    tr_dl = windows(los_frames_c24(C24_LOS), LOS_FEATURES, flares)
    r = fit_eval(tr_dl, te_l, cols(LOS_FEATURES), n_boot=10)
    checks["f3_los"] = {k: [r[k], frozen_f3["los_record"][k]] for k in ("auc", "tss", "threshold")}

    for name, c in checks.items():
        for k, (new, old) in c.items():
            assert abs(new - old) < 1e-6, f"H0 FAIL {name}.{k}: {new} vs frozen {old}"

    # mixed-path equality: fit_predict(all-real) must equal fit_eval to 1e-9
    frozen_eq = fit_eval(tr_v, te_v, cols(VEC_B), n_boot=10, per_set_scale=True)
    mine, _, thr = fit_predict(prep(tr_v, "real", "H"), prep(te_v, "synC23", "T"), cols(VEC_B))
    eq = {k: [mine[k], frozen_eq[k]] for k in ("auc", "tss", "threshold")}
    for k, (new, old) in eq.items():
        assert abs(new - old) < 1e-9, f"H0 FAIL mixed-path equality on {k}: {new} vs {old}"
    checks["mixed_path_equality_1e-9"] = eq
    save("hybrid_gate.json", {"design": "phase_14 H0 gate", "checks": checks, "passed": True})
    print("H0 gate PASSED: frozen F3/F3b reproduced to 1e-6; mixed path equals fit_eval to 1e-9")


# ---------------------------------------------------------------- H1 real-era augmentation

def cmd_real(a):
    flares = load_flares(years=range(1996, 2017))
    era = sharp_era_frames(SHARP_ERA)
    wins_v = windows(era, VEC_B, flares)
    syn_all = windows(vec_frames_c23(C23_FEATURES), VEC_B, flares)
    syn23 = syn_all[syn_all.t_end < pd.Timestamp(C23_TRAIN_CUT)].copy()
    pilot = windows(vec_frames_c24(PILOT_FEATURES), VEC_B, flares)
    out = {"design": "phase_14 section 1.2 (H1 real-era augmentation)", "boundaries": {}}

    for tag, boundary, confirm in (("primary", PRIMARY_BOUNDARY, True),
                                   ("secondary", SECONDARY_BOUNDARY, False)):
        train_r, test, n_drop = split_real(wins_v, boundary)
        counts = {"train": [len(train_r), int(train_r.label.sum())],
                  "test": [len(test), int(test.label.sum())],
                  "test_noaa_dropped": n_drop,
                  "syn23_pool": [len(syn23), int(syn23.label.sum())],
                  "pilot_pool": [len(pilot), int(pilot.label.sum())]}
        print(f"[{tag} {boundary}] train {counts['train']}, test {counts['test']} "
              f"(dropped {n_drop}), synC23 {counts['syn23_pool']}, pilot {counts['pilot_pool']}")
        assert set(syn23.noaa_ar).isdisjoint(set(test.noaa_ar)), "C23/test NOAA overlap"
        assert set(pilot.noaa_ar).isdisjoint(set(test.noaa_ar)), "pilot/test NOAA overlap"
        TR = prep(train_r, "real", "H")
        SY = prep(syn23, "synC23", "T")
        PI = prep(pilot, "synC24", "P")
        TE = prep(test, "real", "H")
        fc = cols(VEC_B)
        arms, preds = {}, {}

        def run(name, train, **kw):
            point, pte, thr = fit_predict(train, TE, fc, **kw)
            arms[name], preds[name] = point, (pte, thr)

        run("a_real_only", TR)
        run("b_plus_c23", pd.concat([TR, SY], ignore_index=True))
        if confirm:
            for w in (0.1, 0.25, 0.5):
                run(f"c_w{w}", pd.concat([TR, SY], ignore_index=True), w_syn=w)
            run("d_pooled_z", pd.concat([TR, SY], ignore_index=True), z="pooled")
            run("e_c23_only", SY)
            run("b_pooled_thr", pd.concat([TR, SY], ignore_index=True), thr_rows="pooled")
            run("a_train_scaler", TR, test_z="train_real")
            run("b_train_scaler", pd.concat([TR, SY], ignore_index=True), test_z="train_real")
            run("h5i_plus_reconC24", pd.concat([TR, PI], ignore_index=True))
            run("a_gbm", TR, model="gbm")
            run("b_gbm", pd.concat([TR, SY], ignore_index=True), model="gbm")

        res, skipped = boot_all(TE, preds, n_boot=a.n_boot)
        for k in arms:
            arms[k]["auc_ci"], arms[k]["tss_ci"] = ci(res[k]["auc"]), ci(res[k]["tss"])
        deltas = {k: paired_delta(res, k, "a_real_only") for k in arms if k != "a_real_only"
                  and arms[k]["model"] == "lr" and arms[k]["test_z"] == "self"}
        if confirm and "b_gbm" in res:
            deltas["b_gbm_vs_a_gbm"] = paired_delta(res, "b_gbm", "a_gbm")
        v = verdict_h1(deltas["b_plus_c23"])
        out["boundaries"][tag] = {"boundary": boundary, "counts": counts, "arms": arms,
                                  "paired_deltas_vs_a": deltas, "skipped_resamples": skipped,
                                  "confirmatory": confirm,
                                  "verdict_b_vs_a": v if confirm else f"(exploratory) {v}"}
        print(f"[{tag}] a TSS {arms['a_real_only']['tss']:.4f} {arms['a_real_only']['tss_ci']}, "
              f"b TSS {arms['b_plus_c23']['tss']:.4f} {arms['b_plus_c23']['tss_ci']}, "
              f"dTSS {deltas['b_plus_c23']['dtss']:+.4f} {deltas['b_plus_c23']['dtss_ci']} -> {v}")

    # LOS positive control (qualitative; era extension with REAL LOS data)
    wins_l = windows(era, LOS_B, flares)
    smarp = windows(los_frames_c23(FRAMES_ALL), LOS_B, flares)
    smarp23 = smarp[smarp.t_end < pd.Timestamp(C23_TRAIN_CUT)].copy()
    ltr, lte, ldrop = split_real(wins_l, PRIMARY_BOUNDARY)
    LTR, LSY, LTE = prep(ltr, "real", "H"), prep(smarp23, "smarpC23", "T"), prep(lte, "real", "H")
    lfc = cols(LOS_B)
    lp = {}
    la, pa, ta = fit_predict(LTR, LTE, lfc)
    lb, pb, tb = fit_predict(pd.concat([LTR, LSY], ignore_index=True), LTE, lfc)
    lp = {"los_a": (pa, ta), "los_b_plus_smarp": (pb, tb)}
    lres, lskip = boot_all(LTE, lp, n_boot=1000)
    for k, d in (("los_a", la), ("los_b_plus_smarp", lb)):
        d["auc_ci"], d["tss_ci"] = ci(lres[k]["auc"]), ci(lres[k]["tss"])
    out["los_control"] = {"note": "qualitative era-only control (real LOS data); own test set",
                          "counts": {"train": [len(ltr), int(ltr.label.sum())],
                                     "test": [len(lte), int(lte.label.sum())],
                                     "smarp23_pool": [len(smarp23), int(smarp23.label.sum())],
                                     "test_noaa_dropped": ldrop},
                          "arms": {"los_a": la, "los_b_plus_smarp": lb},
                          "paired_delta": paired_delta(lres, "los_b_plus_smarp", "los_a"),
                          "skipped_resamples": lskip}
    out["leak_checklist"] = {
        "threshold_rows": "real train rows only (pooled variant labeled sensitivity)",
        "per_source_scalers": "fit on own-source train rows only",
        "label_purge_h": 24, "c23_train_cut": C23_TRAIN_CUT, "test_end": TEST_END,
        "noaa_disjointness_asserted": True, "regions_namespaced": "H/P/T",
        "equivalence_margin_tss": EQUIV_MARGIN}
    save("hybrid_real.json", out)
    rows = []
    for tag, b in out["boundaries"].items():
        for k, v in b["arms"].items():
            d = b["paired_deltas_vs_a"].get(k, {})
            rows.append({"block": f"H1_{tag}", "arm": k, "n_train": v["n_train"],
                         "n_train_syn": v["n_train_syn"], "auc": v["auc"], "tss": v["tss"],
                         "tss_ci_lo": v["tss_ci"][0], "tss_ci_hi": v["tss_ci"][1],
                         "dtss": d.get("dtss"), "dtss_lo": (d.get("dtss_ci") or [None, None])[0],
                         "dtss_hi": (d.get("dtss_ci") or [None, None])[1]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "hybrid_summary.csv"), index=False)
    print(f"wrote {OUT_DIR}/hybrid_summary.csv")


# ---------------------------------------------------------------- H1-T tuned mixing

INNER_BOUNDARY = "2012-12-31"       # train-inner <= boundary - 24 h; val-inner = 2013
VAL_END = "2013-12-30 23:59:59"
W_GRID = (0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def cmd_tuned(a):
    """Phase_14 section 1.3b: weight tuned on an internal validation year, GBM with a
    val-calibrated threshold, balanced subsample. The frozen test set is scored once per arm."""
    from sklearn.metrics import roc_auc_score
    flares = load_flares(years=range(1996, 2017))
    era = sharp_era_frames(SHARP_ERA)
    wins_v = windows(era, VEC_B, flares)
    syn_all = windows(vec_frames_c23(C23_FEATURES), VEC_B, flares)
    syn23 = syn_all[syn_all.t_end < pd.Timestamp(C23_TRAIN_CUT)].copy()
    train_r, test, n_drop = split_real(wins_v, PRIMARY_BOUNDARY)
    train_in, val_in, n_drop_val = split_real(wins_v, INNER_BOUNDARY, test_end=VAL_END)
    assert set(syn23.noaa_ar).isdisjoint(set(val_in.noaa_ar)), "C23/val NOAA overlap"
    assert set(syn23.noaa_ar).isdisjoint(set(test.noaa_ar)), "C23/test NOAA overlap"
    counts = {"train_inner": [len(train_in), int(train_in.label.sum())],
              "val_inner": [len(val_in), int(val_in.label.sum())], "val_noaa_dropped": n_drop_val,
              "train_full": [len(train_r), int(train_r.label.sum())],
              "test": [len(test), int(test.label.sum())], "test_noaa_dropped": n_drop,
              "syn23_pool": [len(syn23), int(syn23.label.sum())]}
    print(f"counts: {counts}")
    TRI, VAL = prep(train_in, "real", "H"), prep(val_in, "real", "H")
    TRA, TE = prep(train_r, "real", "H"), prep(test, "real", "H")
    SY = prep(syn23, "synC23", "T")
    fc = cols(VEC_B)
    mix = lambda base, w: base if w == 0 else pd.concat([base, SY], ignore_index=True)

    # T1 grid: LR, selection on val-inner TSS, ties to the smaller w
    grid_lr = {}
    for w in W_GRID:
        p, _, _ = fit_predict(mix(TRI, w), VAL, fc, w_syn=max(w, 1.0) if w == 0 else w)
        grid_lr[w] = {"val_tss": p["tss"], "val_auc": p["auc"]}
    w_star = max(W_GRID, key=lambda w: (round(grid_lr[w]["val_tss"], 6), -w))
    print(f"T1 grid val TSS: { {w: round(v['val_tss'], 4) for w, v in grid_lr.items()} } -> w*={w_star}")

    # T2 grid: GBM fit on train-inner, threshold chosen on val-inner predictions
    yv = VAL.label.to_numpy(bool)
    yte = TE.label.to_numpy(bool)
    grid_gbm, gbm_cache = {}, {}
    for w in W_GRID:
        tr = mix(TRI, w)
        _, pval, _ = fit_predict(tr, VAL, fc, w_syn=max(w, 1.0) if w == 0 else w, model="gbm")
        thr_v = peak_tss_threshold(yv, pval)
        grid_gbm[w] = {"val_tss": tss(yv, pval, thr_v), "val_thr": float(thr_v)}
        gbm_cache[w] = (tr, thr_v)
    w_star_g = max(W_GRID, key=lambda w: (round(grid_gbm[w]["val_tss"], 6), -w))
    print(f"T2 GBM grid val TSS: { {w: round(v['val_tss'], 4) for w, v in grid_gbm.items()} } "
          f"-> w*_gbm={w_star_g}")

    arms, preds = {}, {}

    def run(name, train, **kw):
        point, pte, thr = fit_predict(train, TE, fc, **kw)
        arms[name], preds[name] = point, (pte, thr)

    run("a_real_only", TRA)
    run("b_naive_w1", mix(TRA, 1.0))
    if w_star == 0:
        arms["t1_tuned"], preds["t1_tuned"] = dict(arms["a_real_only"]), preds["a_real_only"]
        arms["t1_tuned"]["note"] = "tuning selected w*=0 (no mixing)"
    else:
        run("t1_tuned", mix(TRA, w_star), w_syn=w_star)
    arms["t1_tuned"]["w_star"] = w_star

    for name, w in (("gbm_real_valthr", 0), ("gbm_tuned_valthr", w_star_g)):
        tr, thr_v = gbm_cache[w]
        _, pte, _ = fit_predict(tr, TE, fc, w_syn=max(w, 1.0) if w == 0 else w, model="gbm")
        arms[name] = {"model": "gbm", "w_syn": w, "threshold": float(thr_v),
                      "thr_rows": "val_inner", "n_train": int(len(tr)),
                      "n_train_syn": int((tr.source != "real").sum()),
                      "auc": float(roc_auc_score(yte, pte)), "tss": tss(yte, pte, thr_v),
                      "note": "fit on train-inner only; threshold calibrated on val-inner"}
        preds[name] = (pte, thr_v)

    n_sub = len(TRA) // 2
    for s in (0, 1, 2):
        sub = pd.concat([TRA, SY.sample(n=min(n_sub, len(SY)), random_state=s)], ignore_index=True)
        run(f"t3_subsample_s{s}", sub)

    res, skipped = boot_all(TE, preds, n_boot=a.n_boot)
    for k in arms:
        arms[k]["auc_ci"], arms[k]["tss_ci"] = ci(res[k]["auc"]), ci(res[k]["tss"])
    deltas = {k: paired_delta(res, k, "a_real_only") for k in arms if k != "a_real_only"}
    d = deltas["t1_tuned"]
    if d["dtss_ci"][0] > 0 and d["dauc_ci"][0] > 0:
        verdict = "IMPROVEMENT"
    elif d["dtss_ci"][0] > -EQUIV_MARGIN:
        verdict = "SAFE (non-inferior, dTSS CI lower bound > -0.05)"
    else:
        verdict = "naive-mixing degradation stands unqualified"
    save("hybrid_tuned.json", {
        "design": "phase_14 section 1.3b (H1-T tuned mixing)", "counts": counts,
        "grid_lr": grid_lr, "w_star": w_star, "grid_gbm": grid_gbm, "w_star_gbm": w_star_g,
        "arms": arms, "paired_deltas_vs_a": deltas, "skipped_resamples": skipped,
        "confirmatory": "t1_tuned vs a_real_only", "verdict_t1": verdict})
    for k in ("a_real_only", "b_naive_w1", "t1_tuned", "gbm_real_valthr", "gbm_tuned_valthr",
              "t3_subsample_s0", "t3_subsample_s1", "t3_subsample_s2"):
        dd = deltas.get(k, {})
        print(f"{k:20s} tss {arms[k]['tss']:.4f} {[round(x, 3) for x in arms[k]['tss_ci']]}"
              + (f"  dtss {dd['dtss']:+.4f} {[round(x, 4) for x in dd['dtss_ci']]}" if dd else ""))
    print(f"T1 verdict: {verdict}")


# ---------------------------------------------------------------- H1-F twist-augmented features

C23_CAL = os.path.join(ROOT, "results", "full", "intensive", "c23_corrected_ratios.csv")
PILOT_CAL = os.path.join(ROOT, "results", "full", "intensive", "pilot_corrected_frames.csv")
VEC_T = VEC_B + ("J_PER_FLUX", "H_PER_FLUX")


def with_header_ratios(frames):
    """Real SHARP side: the per-flux ratios computed directly from the header keywords."""
    f = frames.copy()
    ok = np.isfinite(f.USFLUX) & (f.USFLUX > 0)
    f["J_PER_FLUX"] = np.where(ok, f.TOTUSJZ / np.where(ok, f.USFLUX, 1.0), np.nan)
    f["H_PER_FLUX"] = np.where(ok, f.TOTUSJH / np.where(ok, f.USFLUX, 1.0), np.nan)
    return f


def with_cal_ratios(frames, cal, keys):
    """Recon side: REPLACE the raw per-flux ratios with the calibrated estimates (stored on the
    log10 transform scale; re-linearized here so the frozen transform re-logs them)."""
    f = frames.merge(cal, on=keys, how="left")
    f["J_PER_FLUX"] = np.power(10.0, f["_calJ"])
    f["H_PER_FLUX"] = np.power(10.0, f["_calH"])
    return f.drop(columns=["_calJ", "_calH"])


def cmd_augment(a):
    """Phase_14 section 1.3c: add twist-per-flux to the forecaster's features (real ratios on
    the real side, calibrated estimates on the recon side)."""
    flares = load_flares(years=range(1996, 2017))
    era = with_header_ratios(sharp_era_frames(SHARP_ERA))
    c23_cal = pd.read_csv(C23_CAL).rename(columns={
        "J_PER_FLUX_CAL": "_calJ", "H_PER_FLUX_CAL": "_calH"})[["tarpnum", "t_rec", "_calJ", "_calH"]]
    pil_cal = pd.read_csv(PILOT_CAL).rename(columns={
        "J_PER_FLUX_CAL_oof": "_calJ", "H_PER_FLUX_CAL_oof": "_calH"})[
        ["harpnum", "mdi_t_rec", "_calJ", "_calH"]]
    c23 = with_cal_ratios(vec_frames_c23(C23_FEATURES), c23_cal, ["tarpnum", "t_rec"])
    pilot = with_cal_ratios(vec_frames_c24(PILOT_FEATURES), pil_cal, ["harpnum", "mdi_t_rec"])
    arms, out = {}, {"design": "phase_14 section 1.3c (twist-augmented features)"}

    def pair(tag, train_b, train_t, test_b, test_t, feats_b, feats_t, n_boot=2000):
        """Baseline vs twist-augmented arm on the same test windows, shared resamples."""
        res_pair, preds = {}, {}
        for name, tr, te, fc in ((f"{tag}_base", train_b, test_b, cols(feats_b)),
                                 (f"{tag}_twist", train_t, test_t, cols(feats_t))):
            point, pte, thr = fit_predict(tr, te, fc)
            res_pair[name], preds[name] = point, (pte, thr)
        # the two arms share window construction keyed by (region, t_end); align on the
        # intersection so the paired bootstrap compares identical test windows
        tb, tt = preds[f"{tag}_base"], preds[f"{tag}_twist"]
        assert len(test_b) == len(test_t), f"{tag}: test window sets differ"
        boot, skipped = boot_all(test_b, {f"{tag}_base": tb, f"{tag}_twist": tt}, n_boot=n_boot)
        for k in res_pair:
            res_pair[k]["auc_ci"], res_pair[k]["tss_ci"] = ci(boot[k]["auc"]), ci(boot[k]["tss"])
        d = paired_delta(boot, f"{tag}_twist", f"{tag}_base")
        arms.update(res_pair)
        out[f"{tag}_delta"] = d
        out[f"{tag}_skipped"] = skipped
        print(f"[{tag}] base {res_pair[f'{tag}_base']['tss']:.4f} -> twist "
              f"{res_pair[f'{tag}_twist']['tss']:.4f}  dTSS {d['dtss']:+.4f} {d['dtss_ci']}  "
              f"dAUC {d['dauc']:+.4f} {d['dauc_ci']}")
        return d

    # P-A deployment: train real era -> test recon C23 (calibrated ratios on the test side)
    te_b = prep(windows(c23, VEC_B, flares), "synC23", "T")
    te_t = prep(windows(c23, VEC_T, flares), "synC23", "T")
    tr_b = prep(windows(era, VEC_B, flares), "real", "H")
    tr_t = prep(windows(era, VEC_T, flares), "real", "H")
    pa = pair("f3b", tr_b, tr_t, te_b, te_t, VEC_B, VEC_T)

    # P-B real era: the 1.2 primary split, header ratios on both sides
    wins_b = windows(era, VEC_B, flares)
    wins_t = windows(era, VEC_T, flares)
    trb, teb, _ = split_real(wins_b, PRIMARY_BOUNDARY)
    trt, tet, _ = split_real(wins_t, PRIMARY_BOUNDARY)
    pb = pair("h1", prep(trb, "real", "H"), prep(trt, "real", "H"),
              prep(teb, "real", "H"), prep(tet, "real", "H"), VEC_B, VEC_T)

    # exploratory: domain-matched with calibrated ratios, 6 and 9 features
    tr_dm_b = prep(windows(pilot, VEC_B, flares), "synC24", "P")
    tr_dm_t = prep(windows(pilot, VEC_T, flares), "synC24", "P")
    pair("f3dm", tr_dm_b, tr_dm_t, te_b, te_t, VEC_B, VEC_T, n_boot=1000)
    full_b, full_t = VEC_FEATURES, VEC_FEATURES + ("J_PER_FLUX", "H_PER_FLUX", "MEANGBZ")
    te_fb = prep(windows(c23, full_b, flares), "synC23", "T")
    te_ft = prep(windows(c23, full_t, flares), "synC23", "T")
    tr_fb = prep(windows(pilot, full_b, flares), "synC24", "P")
    tr_ft = prep(windows(pilot, full_t, flares), "synC24", "P")
    pair("f3dm_full", tr_fb, tr_ft, te_fb, te_ft, full_b, full_t, n_boot=1000)

    out["arms"] = arms
    out["primary"] = {"P-A_deployment": pa, "P-B_real_era": pb}
    save("hybrid_twist_features.json", out)


# ---------------------------------------------------------------- H2 deployment augmentation

def cmd_deploy(a):
    flares = load_flares(years=range(1996, 2017))
    era = sharp_era_frames(SHARP_ERA)
    wins_v = windows(era, VEC_B, flares)
    pilot = windows(vec_frames_c24(PILOT_FEATURES), VEC_B, flares)
    te = windows(vec_frames_c23(C23_FEATURES), VEC_B, flares)
    tail_noaa = set(te[te.t_end >= pd.Timestamp(C23_TRAIN_CUT)].noaa_ar)
    shared = sorted(tail_noaa & (set(wins_v.noaa_ar) | set(pilot.noaa_ar)))
    print(f"shared-NOAA guard set ({len(shared)}): {shared}")
    TE = prep(te, "synC23", "T")
    fc = cols(VEC_B)
    frozen_b = json.load(open(os.path.join(FROZEN_DIR, "f3b_fullera.json")))["arms"]
    arms, preds = {}, {}

    def run(name, train, **kw):
        point, pte, thr = fit_predict(train, TE, fc, **kw)
        arms[name], preds[name] = point, (pte, thr)

    for guard in (False, True):
        g = "guarded" if guard else "unguarded"
        r = wins_v[~wins_v.noaa_ar.isin(shared)] if guard else wins_v
        p = pilot[~pilot.noaa_ar.isin(shared)] if guard else pilot
        TR, PI = prep(r, "real", "H"), prep(p, "synC24", "P")
        for w in (0, 0.25, 0.5, 1, 2):
            name = f"{g}_w{w}"
            if w == 0:
                run(name, TR)                                   # row exclusion, never zero weight
            else:
                run(name, pd.concat([TR, PI], ignore_index=True), w_syn=w)
    # sensitivity: drop every train window inside the C23 overlap coverage, any NOAA
    in_overlap = lambda d: (d.t_end >= pd.Timestamp(C23_TRAIN_CUT)) & (d.t_end < pd.Timestamp(OVERLAP_END))
    r_s, p_s = wins_v[~in_overlap(wins_v)], pilot[~in_overlap(pilot)]
    run("sens_overlap_drop_w1", pd.concat([prep(r_s, "real", "H"), prep(p_s, "synC24", "P")],
                                          ignore_index=True), w_syn=1)
    print(f"sensitivity overlap drop: real {len(wins_v)}->{len(r_s)}, pilot {len(pilot)}->{len(p_s)}")

    res, skipped = boot_all(TE, preds, n_boot=a.n_boot)
    for k in arms:
        arms[k]["auc_ci"], arms[k]["tss_ci"] = ci(res[k]["auc"]), ci(res[k]["tss"])
    deltas = {}
    for g in ("unguarded", "guarded"):
        for w in (0.25, 0.5, 1, 2):
            deltas[f"{g}_w{w}_vs_w0"] = paired_delta(res, f"{g}_w{w}", f"{g}_w0")
    deltas["sens_overlap_drop_w1_vs_guarded_w0"] = paired_delta(res, "sens_overlap_drop_w1", "guarded_w0")

    # frozen reproduction (unguarded w0 is the frozen F3b vector arm)
    for k in ("auc", "tss", "threshold"):
        d = abs(arms["unguarded_w0"][k] - frozen_b["vector"][k])
        assert d < 1e-6, f"unguarded_w0.{k} drifted from frozen f3b: {d}"

    # LOS comparator, guarded and unguarded (test = SMARP C23 LOS record)
    smarp_te = windows(los_frames_c23(FRAMES_ALL), LOS_B, flares)
    LTE = prep(smarp_te, "smarpC23", "T")
    wins_l = windows(era, LOS_B, flares)
    lfc = cols(LOS_B)
    los_arms, los_preds = {}, {}
    for guard in (False, True):
        g = "guarded" if guard else "unguarded"
        lt = wins_l[~wins_l.noaa_ar.isin(shared)] if guard else wins_l
        point, pte, thr = fit_predict(prep(lt, "real", "H"), LTE, lfc)
        los_arms[f"los_{g}"], los_preds[f"los_{g}"] = point, (pte, thr)
    lres, lskip = boot_all(LTE, los_preds, n_boot=1000)
    for k in los_arms:
        los_arms[k]["auc_ci"], los_arms[k]["tss_ci"] = ci(lres[k]["auc"]), ci(lres[k]["tss"])
    for k in ("auc", "tss", "threshold"):
        d = abs(los_arms["los_unguarded"][k] - frozen_b["los_record"][k])
        assert d < 1e-6, f"los_unguarded.{k} drifted from frozen f3b: {d}"

    gap = {"guarded_vector_w0_tss": arms["guarded_w0"]["tss"],
           "guarded_los_tss": los_arms["los_guarded"]["tss"],
           "note": "gap-closure statements compare guarded-to-guarded only (phase_14 rule); "
                   "any gain is attributed to added training coverage, never to reconstruction "
                   "physics quality"}
    save("hybrid_deploy.json", {
        "design": "phase_14 section 1.3 (H2 deployment augmentation)",
        "shared_noaa_guard": {"n": len(shared), "noaa": shared},
        "overlap_sensitivity_counts": {"real": [len(wins_v), len(r_s)], "pilot": [len(pilot), len(p_s)]},
        "arms": arms, "los_arms": los_arms, "paired_deltas": deltas,
        "guarded_gap_comparison": gap, "skipped_resamples": skipped,
        "los_skipped_resamples": lskip})
    for g in ("unguarded", "guarded"):
        row = {f"w{w}": round(arms[f"{g}_w{w}"]["tss"], 4) for w in (0, 0.25, 0.5, 1, 2)}
        print(f"{g} TSS(w): {row}")
    print(f"guarded LOS TSS {los_arms['los_guarded']['tss']:.4f}, "
          f"sens_overlap_drop_w1 TSS {arms['sens_overlap_drop_w1']['tss']:.4f}")


def cmd_repeat(a):
    """Is the mixing effect real or a lucky split? Two stress tests (phase_14 section 1.2
    addendum): (A) several time boundaries, the operationally meaningful split; (B) many
    random NOAA-region-disjoint splits of the real era. Reports mean +- sd of the paired
    delta TSS of mixing vs real-only, so the sign can be judged against its own spread."""
    from sklearn.model_selection import GroupShuffleSplit
    flares = load_flares(years=range(1996, 2017))
    era = sharp_era_frames(SHARP_ERA)
    wins = windows(era, VEC_B, flares)
    syn = windows(vec_frames_c23(C23_FEATURES), VEC_B, flares)
    syn = syn[syn.t_end < pd.Timestamp(C23_TRAIN_CUT)].copy()
    SY = prep(syn, "synC23", "T")
    fc = cols(VEC_B)
    out = {"design": "phase_14 1.2 addendum: repeated splits for the mixing effect",
           "syn_pool": [len(syn), int(syn.label.sum())], "A_time_boundaries": {},
           "B_random_region_splits": {}}

    def arm_delta(TR, TE, weights):
        """paired dTSS of each mixed arm vs real-only, on shared bootstrap resamples."""
        preds, pts = {}, {}
        p0, pte0, thr0 = fit_predict(TR, TE, fc)
        preds["w0"], pts["w0"] = (pte0, thr0), p0
        for w in weights:
            pw, ptew, thrw = fit_predict(pd.concat([TR, SY], ignore_index=True), TE, fc, w_syn=w)
            preds[f"w{w}"], pts[f"w{w}"] = (ptew, thrw), pw
        res, _ = boot_all(TE, preds, n_boot=1000)
        return ({f"w{w}": paired_delta(res, f"w{w}", "w0") for w in weights},
                {k: v["tss"] for k, v in pts.items()})

    WEIGHTS = (0.25, 1.0)
    # (A) time boundaries
    for b in ("2012-12-31", "2013-12-31", "2014-12-31", "2015-12-31"):
        tr, te, drop = split_real(wins, b)
        if len(te) < 200 or te.label.sum() < 15 or tr.label.sum() < 15:
            print(f"[A] {b}: too few, skipped ({len(tr)}/{int(tr.label.sum())} -> "
                  f"{len(te)}/{int(te.label.sum())})"); continue
        d, tss = arm_delta(prep(tr, "real", "H"), prep(te, "real", "H"), WEIGHTS)
        out["A_time_boundaries"][b] = {"counts": {"train": [len(tr), int(tr.label.sum())],
                                                  "test": [len(te), int(te.label.sum())]},
                                       "tss": tss, "deltas": d}
        print(f"[A] boundary {b}: real-only TSS {tss['w0']:.3f} | "
              + " | ".join(f"w={w} dTSS {d[f'w{w}']['dtss']:+.3f} "
                           f"{[round(x,3) for x in d[f'w{w}']['dtss_ci']]}" for w in WEIGHTS))
    # (B) random region-disjoint splits
    acc = {f"w{w}": [] for w in WEIGHTS}; base = []
    gss = GroupShuffleSplit(n_splits=a.n_splits, test_size=0.3, random_state=0)
    grp = wins.noaa_ar.to_numpy()
    for i, (itr, ite) in enumerate(gss.split(wins, wins.label, groups=grp)):
        tr, te = wins.iloc[itr], wins.iloc[ite]
        if te.label.sum() < 10 or tr.label.sum() < 10:
            continue
        d, tss = arm_delta(prep(tr, "real", "H"), prep(te, "real", "H"), WEIGHTS)
        base.append(tss["w0"])
        for w in WEIGHTS:
            acc[f"w{w}"].append(d[f"w{w}"]["dtss"])
        if (i + 1) % 5 == 0:
            print(f"  [B] {i+1}/{a.n_splits}", flush=True)
    out["B_random_region_splits"] = {
        "n_splits": len(base),
        "real_only_tss": {"mean": float(np.mean(base)), "sd": float(np.std(base))},
        "delta_tss": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                          "n_negative": int(sum(1 for x in v if x < 0)), "n": len(v)}
                      for k, v in acc.items()}}
    b = out["B_random_region_splits"]
    print(f"\n[B] {b['n_splits']} region-disjoint splits: real-only TSS "
          f"{b['real_only_tss']['mean']:.3f} +- {b['real_only_tss']['sd']:.3f}")
    for k, v in b["delta_tss"].items():
        print(f"    {k}: dTSS {v['mean']:+.3f} +- {v['sd']:.3f}  "
              f"({v['n_negative']}/{v['n']} splits negative)")
    save("hybrid_repeat.json", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.set_defaults(fn=cmd_gate)
    r = sub.add_parser("real")
    r.add_argument("--n_boot", type=int, default=2000)
    r.set_defaults(fn=cmd_real)
    t = sub.add_parser("tuned")
    t.add_argument("--n_boot", type=int, default=2000)
    t.set_defaults(fn=cmd_tuned)
    g2 = sub.add_parser("augment")
    g2.set_defaults(fn=cmd_augment)
    rp = sub.add_parser("repeat")
    rp.add_argument("--n_splits", type=int, default=30)
    rp.set_defaults(fn=cmd_repeat)
    d = sub.add_parser("deploy")
    d.add_argument("--n_boot", type=int, default=2000)
    d.set_defaults(fn=cmd_deploy)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
