"""Phase 14 / E-INT: calibrated per-flux estimator for the intensive keywords
(reports/phase_14.md section 2). The C2 screening result stands: J_PER_FLUX / H_PER_FLUX from the
SR pixels agree with truth at only 0.320 / 0.610 (bar 0.70). This script asks whether a
SEPARATELY VALIDATED keyword-space estimator (Ridge on the 21 audited recon features) produces
calibrated per-flux estimates that pass the same screen plus four blind guardrails. Wording rule:
"statistically calibrated estimates", never "recovery".

    python scripts/c23_intensive_calibration.py corrector   # P0: protocols A/B, ablation, seeds, g1-g4
    python scripts/c23_intensive_calibration.py weights     # keyword_reliability_intensive_corr.csv
    python scripts/c23_intensive_calibration.py demo        # R0: quantile mapping cannot move Spearman

Frozen pipeline files imported, never modified; outputs to results/full/intensive/.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from c23_screening import (                            # noqa: E402
    LOG10, LON_MAX, MIN_FRAMES, SOURCES, clv_detrend, clv_fit, derive, lag_pairs, seed_mean,
    transform)
from sr import sharp_keywords as sk                               # noqa: E402

PILOT_FEATURES = os.path.join(ROOT, "results", "full", "overlap_pilot", "features.csv")
SPLITS = os.path.join(ROOT, "splits", "full.json")
REGIONS_CSV = os.path.join(ROOT, "data", "full", "regions.csv")
PILOT_DIR = os.path.join(ROOT, "results", "full", "overlap_pilot")
OUT_DIR = os.path.join(ROOT, "results", "full", "intensive")

KEYWORDS = list(sk.KEYWORDS)                       # the 13 SHARP keywords
COUNTS = ["n_pixels", "n_finite", "n_interior"]
GEOM = ["center_angle", "center_lon"]
SEED_STD = ["std_log10_TOTUSJZ", "std_log10_TOTUSJH", "std_log10_USFLUX"]
FEATURES = KEYWORDS + COUNTS + GEOM + SEED_STD     # 21, frozen by the phase_14 audit disclosure
ABLATION = ["USFLUX", "n_pixels", "n_finite", "n_interior", "center_angle", "center_lon"]
TARGETS = {"J_PER_FLUX": ("TOTUSJZ",), "H_PER_FLUX": ("TOTUSJH",)}
ALPHA = 1.0                                        # frozen (audited)
SEED = 0
ERA_SPLIT = "2010-11-01"
BAR_POINT, BAR_CI_LO, BAR_G2, BAR_G3, BAR_G4 = 0.70, 0.60, 0.60, 0.50, 0.60
BAR_G1_DIFF, BAR_G4_DROP = 0.15, 0.10
RAW_BASELINE = {"J_PER_FLUX": 0.320, "H_PER_FLUX": 0.610}


def spearman(x, y):
    """Exactly the E2 agreement() statistic: rank -> Pearson over finite pairs, n >= 8."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return float("nan"), int(ok.sum())
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def harp_boot_ci(df, xcol, ycol, n_boot=1000, seed=SEED):
    """95% cluster-bootstrap CI for spearman(xcol, ycol), resampling HARPs with replacement."""
    rng = np.random.default_rng(seed)
    harps = df.harpnum.unique()
    by = {h: g for h, g in df.groupby("harpnum")}
    rs = []
    for _ in range(n_boot):
        take = rng.choice(harps, size=len(harps), replace=True)
        m = pd.concat([by[h] for h in take])
        r, _ = spearman(m[xcol], m[ycol])
        if np.isfinite(r):
            rs.append(r)
    return [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))] if rs else None


def rank_partial(x, y, controls):
    """Spearman partial correlation: rank-transform everything, residualize x and y on the
    controls (with intercept), Pearson of the residuals."""
    cols = [np.asarray(c, np.float64) for c in controls]
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    for c in cols:
        ok &= np.isfinite(c)
    if ok.sum() < 8:
        return float("nan")
    rk = lambda v: pd.Series(v[ok]).rank().to_numpy()
    C = np.stack([np.ones(int(ok.sum()))] + [rk(c) for c in cols], axis=1)
    resid = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return float(np.corrcoef(resid(rk(x)), resid(rk(y)))[0, 1])


def center_angle_b0(lat_deg, lon_deg, times):
    """B0-corrected heliocentric angle from FWT lat/lon: the C23 substitution path."""
    from sunpy.coordinates.sun import B0
    b0 = np.array([B0(t).to_value("deg") for t in times])
    lat, lon, b = np.radians(lat_deg), np.radians(lon_deg), np.radians(b0)
    cosang = np.sin(b) * np.sin(lat) + np.cos(b) * np.cos(lat) * np.cos(lon)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def load_pairs():
    """The paired overlap-era table: recon features (X, transform scale) + true targets."""
    use = ["harpnum", "mdi_t_rec", "model", "mask", "center_angle", "center_lon",
           "n_pixels", "n_finite", "n_interior"] + KEYWORDS
    df = pd.read_csv(PILOT_FEATURES, usecols=lambda c: c in set(use + ["sharp_t_rec"]))
    ours = df[df.model.isin(SOURCES["ours"]) & (df["mask"] == "strong_pred")]
    X = derive(seed_mean(ours, SOURCES["ours"]))
    meta = ours[["harpnum", "mdi_t_rec", "center_lon"]].drop_duplicates(["harpnum", "mdi_t_rec"])
    X = X.merge(meta, on=["harpnum", "mdi_t_rec"], how="left")
    # per-seed std of the log10 current/helicity/flux keywords (the seed-disagreement channel)
    for kw in ("TOTUSJZ", "TOTUSJH", "USFLUX"):
        s = ours.copy()
        s[f"log_{kw}"] = transform(kw, s[kw])
        g = s.groupby(["harpnum", "mdi_t_rec"])[f"log_{kw}"].std().rename(f"std_log10_{kw}")
        X = X.merge(g.reset_index(), on=["harpnum", "mdi_t_rec"], how="left")
    for kw in KEYWORDS:                                   # analysis transforms, frozen rules
        X[kw] = transform(kw, X[kw])
    true = derive(seed_mean(df[(df.model == "true_hmi") & (df["mask"] == "strong")], ("true_hmi",)))
    true = true[["harpnum", "mdi_t_rec", "J_PER_FLUX", "H_PER_FLUX", "USFLUX", "MEANGBZ",
                 "STRONG_FRAC"]].rename(columns=lambda c: c if c in ("harpnum", "mdi_t_rec")
                                        else f"true_{c}")
    m = X.merge(true, on=["harpnum", "mdi_t_rec"], how="inner")
    for t in TARGETS:
        m[f"y_{t}"] = transform(t, m[f"true_{t}"])        # log10 true ratio
        m[f"raw_{t}"] = transform(t, m[t])                # log10 recon ratio (the C2 loser)
    m["true_MEANGBZ_t"] = m["true_MEANGBZ"]
    m["true_STRONG_FRAC_t"] = m["true_STRONG_FRAC"]
    m["_t"] = pd.to_datetime(m.mdi_t_rec.str.replace("_TAI", "", regex=False),
                             format="%Y.%m.%d_%H:%M:%S")
    return m


def load_groups():
    sp = json.load(open(SPLITS))
    h2g = {}
    for gname, harps in sp["groups"].items():
        for h in harps:
            h2g[int(h)] = gname
    return h2g, sp["splits"]


def fit_ridge(Xtr, ytr, Xte):
    """Median-impute + standardize (train stats only) + Ridge(ALPHA); returns predictions + model."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ok = np.isfinite(ytr)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ALPHA))
    pipe.fit(Xtr[ok], ytr[ok])
    return pipe.predict(Xte), pipe


def protocol_a(m, feats, h2g):
    """5-fold GroupKFold by NOAA-transitive group -> OOF predictions per target."""
    from sklearn.model_selection import GroupKFold
    groups = m.harpnum.map(h2g).to_numpy()
    X = m[feats].to_numpy(np.float64)
    oof = {t: np.full(len(m), np.nan) for t in TARGETS}
    for tr, te in GroupKFold(n_splits=5).split(X, groups=groups):
        for t in TARGETS:
            oof[t][te] = fit_ridge(X[tr], m[f"y_{t}"].to_numpy()[tr], X[te])[0]
    return oof


def protocol_b(m, feats, splits):
    """Frozen-split confirmation: fit on train+val HARPs, predict the 52 test HARPs."""
    fit_h = set(splits["train"]["harps"]) | set(splits["val"]["harps"])
    test_h = set(splits["test"]["harps"])
    tr, te = m[m.harpnum.isin(fit_h)], m[m.harpnum.isin(test_h)]
    X, Xte = tr[feats].to_numpy(np.float64), te[feats].to_numpy(np.float64)
    preds, models = {}, {}
    for t in TARGETS:
        preds[t], models[t] = fit_ridge(X, tr[f"y_{t}"].to_numpy(), Xte)
    return tr, te, preds, models


def guardrails(m, oof, h2g, feats):
    """g1-g4 + era split, per target (phase_14 section 2.1 bars)."""
    out = {}
    luf = m["USFLUX"].to_numpy()                      # log10 recon USFLUX (already transformed)
    for t in TARGETS:
        y, p, raw = m[f"y_{t}"].to_numpy(), oof[t], m[f"raw_{t}"].to_numpy()
        g = {}
        # g1 size trap: corrected must track size the way TRUTH tracks size
        r_corr, _ = spearman(p, luf)
        r_true, _ = spearman(y, luf)
        g["g1"] = {"rho_corrected_vs_logflux": r_corr, "rho_truth_vs_logflux": r_true,
                   "abs_diff": abs(r_corr - r_true), "bar": BAR_G1_DIFF,
                   "pass": bool(abs(r_corr - r_true) <= BAR_G1_DIFF)}
        # g2 era split, NOAA-group-disjoint at the majority era
        tmp = m[["harpnum", "_t"]].copy()
        tmp["grp"] = tmp.harpnum.map(h2g)
        tmp["late"] = tmp._t >= pd.Timestamp(ERA_SPLIT)
        maj = tmp.groupby("grp")["late"].mean()
        keep = tmp.late.eq(tmp.grp.map(maj >= 0.5)).to_numpy()     # drop minority-era frames
        late = tmp.grp.map(maj >= 0.5).to_numpy()
        X = m[feats].to_numpy(np.float64)
        g2 = {}
        for d, (trm, tem) in {"early_to_late": (keep & ~late, keep & late),
                              "late_to_early": (keep & late, keep & ~late)}.items():
            pe = fit_ridge(X[trm], y[trm], X[tem])[0]
            g2[d] = spearman(pe, y[tem])[0]
        g2.update({"bar": BAR_G2, "n_early": int((keep & ~late).sum()),
                   "n_late": int((keep & late).sum()),
                   "n_minority_frames_dropped": int((~keep).sum()),
                   "pass": bool(min(g2["early_to_late"], g2["late_to_early"]) >= BAR_G2),
                   "caveat": "a within-year split cannot prove C24-to-C23 stationarity"})
        g["g2"] = g2
        # g3 within-trajectory fidelity (decisive for clustering use)
        def per_harp(med_of):
            vals = []
            for _, grp in m.assign(_p=med_of).groupby("harpnum"):
                ok = np.isfinite(grp._p) & np.isfinite(grp[f"y_{t}"])
                if ok.sum() >= MIN_FRAMES:
                    vals.append(spearman(grp._p[ok], grp[f"y_{t}"][ok])[0])
            v = np.asarray(vals)
            return {"median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)),
                    float(np.percentile(v, 75))], "n_harps": int(len(v))}
        g3 = {"corrected": per_harp(p), "raw": per_harp(raw)}
        g3["bar"], g3["pass"] = BAR_G3, bool(g3["corrected"]["median"] >= BAR_G3)
        g["g3"] = g3
        # g4 size-partial control
        controls = [luf, m["n_pixels"].to_numpy(np.float64), m["center_angle"].to_numpy(np.float64)]
        uncond, _ = spearman(p, y)
        part = rank_partial(p, y, controls)
        g["g4"] = {"unconditional": uncond, "partial": part, "drop": uncond - part,
                   "bars": {"partial": BAR_G4, "max_drop": BAR_G4_DROP},
                   "pass": bool(part >= BAR_G4 and (uncond - part) <= BAR_G4_DROP)}
        out[t] = g
    # context rows for g3: how well do the PASSING intensive keywords track truth over time
    ctx = {}
    for kw, tcol in (("MEANGBZ", "true_MEANGBZ_t"), ("STRONG_FRAC", "true_STRONG_FRAC_t")):
        vals = []
        for _, grp in m.groupby("harpnum"):
            ok = np.isfinite(grp[kw]) & np.isfinite(grp[tcol])
            if ok.sum() >= MIN_FRAMES:
                vals.append(spearman(grp[kw][ok], grp[tcol][ok])[0])
        v = np.asarray(vals)
        ctx[kw] = {"median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)),
                   float(np.percentile(v, 75))], "n_harps": int(len(v))}
    out["g3_context_passing_keywords"] = ctx
    return out


def cmd_corrector(a):
    os.makedirs(OUT_DIR, exist_ok=True)
    m = load_pairs()
    h2g, splits = load_groups()
    n_harps, n_groups = m.harpnum.nunique(), m.harpnum.map(h2g).nunique()
    counts = {"paired_frames": len(m), "harps": int(n_harps), "noaa_groups": int(n_groups),
              "finite_targets": {t: int(np.isfinite(m[f"y_{t}"]).sum()) for t in TARGETS}}
    print(f"pairs: {counts}")
    assert n_harps == 233, f"expected 233 overlap HARPs, got {n_harps}"

    # sanity: the C23 substitution path for center_angle (lat/lon + per-time B0)
    sanity = {}
    try:
        regs = pd.read_csv(REGIONS_CSV, usecols=["mdi_t_rec", "center_lat", "center_lon",
                                                 "center_angle"]).sample(200, random_state=0)
        times = pd.to_datetime(regs.mdi_t_rec.str.replace("_TAI", "", regex=False),
                               format="%Y.%m.%d_%H:%M:%S")
        ca = center_angle_b0(regs.center_lat.to_numpy(), regs.center_lon.to_numpy(), times)
        sanity = {"max_abs_delta_deg": float(np.max(np.abs(ca - regs.center_angle.to_numpy()))),
                  "n": 200, "formula": "cos th = sin B0 sin lat + cos B0 cos lat cos lon"}
        assert sanity["max_abs_delta_deg"] < 0.1, f"center_angle formula mismatch: {sanity}"
        print(f"center_angle sanity: max |delta| {sanity['max_abs_delta_deg']:.4f} deg over 200 frames")
    except ImportError as e:
        sanity = {"skipped": f"sunpy unavailable ({e}); c23-apply blocked until checked"}
        print(f"WARNING center_angle sanity skipped: {e}")

    oof = protocol_a(m, FEATURES, h2g)
    tr_b, te_b, pred_b, models_b = protocol_b(m, FEATURES, splits)
    oof_ab = protocol_a(m, ABLATION, h2g)
    _, te_ab, pred_ab, _ = protocol_b(m, ABLATION, splits)

    # single-seed feature frames for the seed channel (target-independent)
    use = ["harpnum", "mdi_t_rec", "sharp_t_rec", "model", "mask", "center_angle", "center_lon",
           "n_pixels", "n_finite", "n_interior"] + KEYWORDS
    raw_df = pd.read_csv(PILOT_FEATURES, usecols=lambda c: c in set(use))
    seed_frames = {}
    for sd in SOURCES["ours"]:
        Xs = derive(seed_mean(raw_df[(raw_df.model == sd) & (raw_df["mask"] == "strong_pred")], (sd,)))
        meta = raw_df[["harpnum", "mdi_t_rec", "center_lon"]].drop_duplicates(["harpnum", "mdi_t_rec"])
        Xs = Xs.merge(meta, on=["harpnum", "mdi_t_rec"], how="left")
        for kw in KEYWORDS:
            Xs[kw] = transform(kw, Xs[kw])
        for c in SEED_STD:
            Xs[c] = 0.0                                     # single seed: no spread channel
        seed_frames[sd] = Xs

    report = {"date": "2026-10-31", "alpha": ALPHA, "seed": SEED, "features": FEATURES,
              "ablation_features": ABLATION, "counts": counts, "center_angle_sanity": sanity,
              "audit_disclosure": "hyperparameters frozen on a design-time audit; see "
                                  "reports/phase_14.md section 0", "targets": {}}
    for t in TARGETS:
        y = m[f"y_{t}"].to_numpy()
        da = m.assign(_p=oof[t])
        ra, na = spearman(oof[t], y)
        ra_ci = harp_boot_ci(da, "_p", f"y_{t}")
        db = te_b.assign(_p=pred_b[t])
        rb, nb = spearman(pred_b[t], te_b[f"y_{t}"])
        rb_ci = harp_boot_ci(db, "_p", f"y_{t}")
        raw_a = spearman(m[f"raw_{t}"], y)[0]
        raw_b = spearman(te_b[f"raw_{t}"], te_b[f"y_{t}"])[0]
        abl_a = spearman(oof_ab[t], y)[0]
        abl_b = spearman(pred_ab[t], te_b[f"y_{t}"])[0]
        abl_part = rank_partial(oof_ab[t], y, [m["USFLUX"].to_numpy(np.float64),
                                               m["n_pixels"].to_numpy(np.float64),
                                               m["center_angle"].to_numpy(np.float64)])
        # seed channel: protocol B refit per single seed
        seed_rs, seed_preds = [], []
        for sd in SOURCES["ours"]:
            ms = seed_frames[sd].merge(m[["harpnum", "mdi_t_rec", f"y_{t}"]],
                                       on=["harpnum", "mdi_t_rec"])
            fit_h = set(splits["train"]["harps"]) | set(splits["val"]["harps"])
            test_h = set(splits["test"]["harps"])
            trs, tes = ms[ms.harpnum.isin(fit_h)], ms[ms.harpnum.isin(test_h)]
            ps = fit_ridge(trs[FEATURES].to_numpy(np.float64), trs[f"y_{t}"].to_numpy(),
                           tes[FEATURES].to_numpy(np.float64))[0]
            seed_rs.append(spearman(ps, tes[f"y_{t}"])[0])
            seed_preds.append(pd.Series(ps, index=pd.MultiIndex.from_frame(tes[["harpnum", "mdi_t_rec"]])))
        sp3 = pd.concat(seed_preds, axis=1)
        per_frame_sd = float(sp3.std(axis=1).median())
        seed_channel = {"test_spearman_per_seed": seed_rs,
                        "spread": float(np.max(seed_rs) - np.min(seed_rs)),
                        "median_per_frame_sd": per_frame_sd}
        report["targets"][t] = {
            "seed_channel": seed_channel,
            "protocol_a_oof": {"spearman": ra, "ci": ra_ci, "n": na,
                               "pass": bool(ra >= BAR_POINT and ra_ci[0] >= BAR_CI_LO)},
            "protocol_b_test": {"spearman": rb, "ci": rb_ci, "n": nb,
                                "pass": bool(rb >= BAR_POINT and rb_ci[0] >= BAR_CI_LO)},
            "raw_baseline": {"oof_frames": raw_a, "test_frames": raw_b,
                             "frozen_screen": RAW_BASELINE[t]},
            "ablation_density_geometry": {"oof": abl_a, "test": abl_b, "oof_partial": abl_part},
        }
    g = guardrails(m, oof, h2g, FEATURES)
    report["guardrails"] = g
    passes = {t: (report["targets"][t]["protocol_a_oof"]["pass"]
                  and report["targets"][t]["protocol_b_test"]["pass"]
                  and all(g[t][k]["pass"] for k in ("g1", "g2", "g3", "g4"))) for t in TARGETS}
    m1_only = {t: (report["targets"][t]["protocol_a_oof"]["pass"]
                   and report["targets"][t]["protocol_b_test"]["pass"]) for t in TARGETS}
    report["reading"] = ("PASS" if all(passes.values()) else
                         "PARTIAL" if any(m1_only.values()) else "NULL")
    report["reading_rule"] = ("PASS = M1+g1+g2+g3+g4 both ratios; PARTIAL = M1 holds but a "
                              "guardrail fails; NULL = no material improvement (phase_14 2.1)")
    with open(os.path.join(OUT_DIR, "corrector_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    # frozen coefficients for the (gated) one-way c23-apply
    rows = []
    for t, pipe in models_b.items():
        imp, sc, rg = pipe.named_steps.values()
        for fcol, med, mu, sdv, w in zip(FEATURES, imp.statistics_, sc.mean_, sc.scale_, rg.coef_):
            rows.append({"target": t, "feature": fcol, "impute_median": med, "scale_mean": mu,
                         "scale_sd": sdv, "coef": w})
        rows.append({"target": t, "feature": "__intercept__", "impute_median": np.nan,
                     "scale_mean": np.nan, "scale_sd": np.nan, "coef": rg.intercept_})
    coefs = pd.DataFrame(rows)
    cpath = os.path.join(OUT_DIR, "corrector_coefs.csv")
    coefs.to_csv(cpath, index=False)
    sha = hashlib.sha256(open(cpath, "rb").read()).hexdigest()
    print(f"corrector_coefs.csv sha256 {sha}")
    # per-frame output table
    outf = m[["harpnum", "mdi_t_rec", "center_angle", "center_lon"]].copy()
    for t in TARGETS:
        outf[f"{t}_CAL_oof"] = oof[t]
        outf[f"raw_{t}"] = m[f"raw_{t}"]
        outf[f"true_{t}"] = m[f"y_{t}"]
    bmap = te_b[["harpnum", "mdi_t_rec"]].copy()
    for t in TARGETS:
        bmap[f"{t}_CAL_frozen"] = pred_b[t]
    outf = outf.merge(bmap, on=["harpnum", "mdi_t_rec"], how="left")
    outf.to_csv(os.path.join(OUT_DIR, "pilot_corrected_frames.csv"), index=False)
    with open(os.path.join(OUT_DIR, "corrector_coefs.sha256"), "w") as f:
        f.write(sha + "\n")
    print(json.dumps({t: {"A": report["targets"][t]["protocol_a_oof"],
                          "B": report["targets"][t]["protocol_b_test"]} for t in TARGETS},
                     indent=1, default=str))
    print("guardrails:", json.dumps({t: {k: g[t][k]["pass"] for k in ("g1", "g2", "g3", "g4")}
                                     for t in TARGETS}))
    print(f"READING: {report['reading']}")
    print(f"wrote {OUT_DIR}/corrector_report.json, corrector_coefs.csv, pilot_corrected_frames.csv")


def cmd_weights(a):
    """_CAL rows in the exact schema of keyword_reliability_intensive.csv; the OOF-based noise
    model replaces noise_var() (no 12-min real pairs exist for a corrector output) - the one
    documented deviation (phase_14 section 2.2)."""
    rep = json.load(open(os.path.join(OUT_DIR, "corrector_report.json")))
    frames = pd.read_csv(os.path.join(OUT_DIR, "pilot_corrected_frames.csv"))
    old = pd.read_csv(os.path.join(PILOT_DIR, "keyword_reliability_intensive.csv"))
    r12 = {"J_PER_FLUX": float(old.set_index("keyword").loc["J_PER_FLUX", "r12_real"]),
           "H_PER_FLUX": float(old.set_index("keyword").loc["H_PER_FLUX", "r12_real"])}
    rows, meta = [], {}
    for t in TARGETS:
        cal, true = frames[f"{t}_CAL_oof"], frames[f"true_{t}"]
        agr = rep["targets"][t]["protocol_a_oof"]["spearman"]
        n = rep["targets"][t]["protocol_a_oof"]["n"]
        resid = (cal - true).to_numpy()
        nv_oof = float(np.nanvar(resid))
        sub = frames.rename(columns={f"{t}_CAL_oof": f"{t}_CAL"})[
            ["harpnum", "mdi_t_rec", f"{t}_CAL", "center_angle", "center_lon"]].dropna()
        pairs = lag_pairs(sub, 96)
        d = (pairs[f"{t}_CAL_b"] - pairs[f"{t}_CAL_a"]).to_numpy(np.float64)
        nv_lag = float(np.nanvar(d) / 2.0)
        nv = max(nv_oof, nv_lag)
        sub2 = sub[sub.center_lon.abs() <= LON_MAX]
        sub2 = clv_detrend(sub2, clv_fit(sub2, [f"{t}_CAL"]))
        vs = [np.var(g[f"{t}_CAL"].dropna().to_numpy(), ddof=1)
              for _, g in sub2.groupby("harpnum") if g[f"{t}_CAL"].notna().sum() >= MIN_FRAMES]
        wv = float(np.mean(vs))
        evo = wv / nv if nv > 0 else float("nan")
        inc = bool(r12[t] >= 0.70 and agr >= 0.70)
        rows.append({"keyword": f"{t}_CAL", "r12_real": r12[t], "agreement_ours": agr,
                     "n_frames": n, "noise_var": nv, "within_var": wv, "snr2_evo": evo,
                     "included": inc, "snr2_r12": r12[t] / (1 - r12[t]),
                     "rho_eff": r12[t] * agr})
        meta[f"{t}_CAL"] = {"noise_var_oof_resid": nv_oof, "noise_var_lag96_halfdiff": nv_lag,
                            "noise_rule": "max of the two (phase_14 2.2)",
                            "within_var_rule": "frozen CLV-detrend machinery on the corrected series"}
    keep = old[old.keyword.isin(["MEANGBZ", "STRONG_FRAC"])].to_dict("records")
    w = pd.DataFrame(rows + keep)
    inc = w[w.included & np.isfinite(w.snr2_evo)]
    norm = float(inc.snr2_evo.mean()) if len(inc) else float("nan")
    w["weight"] = np.where(w.included & np.isfinite(w.snr2_evo), w.snr2_evo / norm, 0.0)
    w.to_csv(os.path.join(OUT_DIR, "keyword_reliability_intensive_corr.csv"), index=False)
    with open(os.path.join(OUT_DIR, "keyword_reliability_intensive_corr.json"), "w") as f:
        json.dump({"bars": {"r12": 0.70, "agreement": 0.70}, "noise_model_deviation": meta,
                   "agreement_source": "Protocol-A OOF (never resubstitution)",
                   "rows": w.to_dict("records"), "preregistration": "reports/phase_14.md"},
                  f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(w.to_string(index=False))
    print(f"wrote {OUT_DIR}/keyword_reliability_intensive_corr.{{csv,json}}")


def cmd_demo(a):
    """R0: marginal quantile mapping is monotone, so it cannot move the Spearman screen."""
    m = load_pairs()
    x, y = m["raw_J_PER_FLUX"].to_numpy(), m["y_J_PER_FLUX"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    before, _ = spearman(x[ok], y[ok])
    qs = np.linspace(0, 1, 2001)
    mapped = np.interp(x[ok], np.quantile(x[ok], qs), np.quantile(y[ok], qs))
    after, _ = spearman(mapped, y[ok])
    print(f"J_PER_FLUX agreement before quantile mapping {before:.6f}, after {after:.6f}, "
          f"|delta| {abs(after - before):.2e}")
    assert abs(after - before) < 1e-6
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "r0_quantile_demo.json"), "w") as f:
        json.dump({"spearman_before": before, "spearman_after": after,
                   "conclusion": "monotone marginal recalibration cannot change the rank-based "
                                 "agreement screen; the fix must condition on other observables"},
                  f, indent=1)
    print(f"wrote {OUT_DIR}/r0_quantile_demo.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("corrector", cmd_corrector), ("weights", cmd_weights), ("demo", cmd_demo)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
