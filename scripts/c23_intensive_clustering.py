"""Phase 14 / E-INT P1: intensive clustering with the calibrated per-flux estimates
(reports/phase_14.md section 2.4b; P0 = scripts/c23_intensive_calibration.py).

    python scripts/c23_intensive_clustering.py cluster   # pilot _CAL clustering + truth match + external
    python scripts/c23_intensive_clustering.py jitter    # stability under the corrector's error model
    python scripts/c23_intensive_clustering.py c23       # covariate check, apply, C23 clustering + match

Frozen pipeline reused by import; all outputs to results/full/intensive/.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from c23_clustering import (                                     # noqa: E402
    K_RANGE, bootstrap_ari, distance_matrix, pam, silhouette_k, size_auc)
from c23_intensive_calibration import (                             # noqa: E402
    FEATURES, OUT_DIR, PILOT_FEATURES, SPLITS, TARGETS, center_angle_b0, load_pairs, spearman)
from c23_screening import (                            # noqa: E402
    LON_MAX, MIN_FRAMES, clv_detrend, clv_fit, transform)

PILOT_DIR = os.path.join(ROOT, "results", "full", "overlap_pilot")
C23_FEATURES = os.path.join(ROOT, "results", "full", "c23_build", "features.csv")
CORR_FRAMES = os.path.join(OUT_DIR, "pilot_corrected_frames.csv")
CORR_WEIGHTS = os.path.join(OUT_DIR, "keyword_reliability_intensive_corr.csv")
CAL_FEATURES = ("J_PER_FLUX_CAL", "H_PER_FLUX_CAL", "MEANGBZ", "STRONG_FRAC")
N_JITTER, N_IID = 20, 10
SEED_MODELS = ("F_U_E3_s0_e200", "F_U_E3_s1_e200", "F_U_E3_s2")


def load_cal_weights():
    rel = pd.read_csv(CORR_WEIGHTS).set_index("keyword")
    feats = [f for f in CAL_FEATURES if f in rel.index and bool(rel.loc[f, "included"])]
    w = rel.loc[feats, "weight"].to_numpy(np.float64)
    return feats, w / w.mean()


def pilot_tables():
    """(recon frame table with _CAL OOF columns, truth frame table named like the _CAL set),
    both |center_lon| <= 60, one row per (harpnum, mdi_t_rec)."""
    m = load_pairs()
    cf = pd.read_csv(CORR_FRAMES)
    m = m.merge(cf[["harpnum", "mdi_t_rec", "J_PER_FLUX_CAL_oof", "H_PER_FLUX_CAL_oof"]],
                on=["harpnum", "mdi_t_rec"], how="left")
    m = m[m.center_lon.abs() <= LON_MAX].copy()
    rec = m.rename(columns={"J_PER_FLUX_CAL_oof": "J_PER_FLUX_CAL",
                            "H_PER_FLUX_CAL_oof": "H_PER_FLUX_CAL"})
    true = pd.DataFrame({"harpnum": m.harpnum, "mdi_t_rec": m.mdi_t_rec, "_t": m._t,
                         "center_angle": m.center_angle,
                         "J_PER_FLUX_CAL": m.y_J_PER_FLUX, "H_PER_FLUX_CAL": m.y_H_PER_FLUX,
                         "MEANGBZ": m.true_MEANGBZ, "STRONG_FRAC": m.true_STRONG_FRAC})
    return rec, true


def trajectories(tab, feats, coefs, min_frames=MIN_FRAMES):
    """Mirror of c23_clustering.build_trajectories on an in-memory frame table."""
    sm = clv_detrend(tab.copy(), coefs)
    stats = {f: (float(np.nanmean(sm[f])), float(np.nanstd(sm[f]) or 1.0)) for f in feats}
    for f in feats:
        sm[f] = (sm[f] - stats[f][0]) / stats[f][1]
    out = {}
    for h, g in sm.groupby("harpnum"):
        g = g.sort_values("_t")
        arr = g[list(feats)].to_numpy(np.float64)
        arr = arr[np.isfinite(arr).all(1)]
        if len(arr) >= min_frames:
            out[int(h)] = arr
    return out, stats


def hungarian_match(lab_a, lab_b):
    from scipy.optimize import linear_sum_assignment
    ka, kb = int(lab_a.max()) + 1, int(lab_b.max()) + 1
    conf = np.zeros((ka, kb))
    for i in range(ka):
        for j in range(kb):
            conf[i, j] = ((lab_a == i) & (lab_b == j)).sum()
    ri, cj = linear_sum_assignment(-conf)
    return float(conf[ri, cj].sum() / conf.sum()), {int(i): int(j) for i, j in zip(ri, cj)}


def cluster_run(rec, true_ref, feats, weights, size_series, tag, out_dir, boots=True):
    """Frozen clustering recipe on an in-memory table; returns (result dict, labels df, trajs)."""
    coefs = clv_fit(true_ref, feats)
    trajs, stats = trajectories(rec, feats, coefs)
    harps = sorted(trajs)
    T = [trajs[h] for h in harps]
    size_vec = np.array([size_series.get(h, np.nan) for h in harps])
    print(f"[{tag}] {len(T)} trajectories, features {feats}, weights {np.round(weights, 3)}")
    mats = {}
    for name, w in (("weighted", weights), ("unweighted", None)):
        path = os.path.join(out_dir, f"dtw_{tag}_{name}.npy")
        if os.path.exists(path):
            mats[name] = np.load(path)
        else:
            mats[name] = distance_matrix(T, w)
            np.save(path, mats[name])
    k, sil = silhouette_k(mats["unweighted"])
    out = {"tag": tag, "n_trajectories": len(T), "features": list(feats),
           "weights": [float(x) for x in weights], "k": int(k),
           "k_by_silhouette_unweighted": {str(q): round(v, 4) for q, v in sil.items()},
           "clv_coefs": {f: list(c) for f, c in coefs.items()},
           "z_stats": {f: list(s) for f, s in stats.items()}, "variants": {}}
    labs = {}
    for name, D in mats.items():
        lab, med = pam(D, k)
        labs[name] = lab
        v = {"medoid_harps": [int(harps[m]) for m in med],
             "cluster_sizes": [int((lab == c).sum()) for c in range(k)],
             "size_auc": size_auc(lab, size_vec)}
        if boots:
            b = bootstrap_ari(D, k)
            v["self_ari_boot"] = {"p5": float(np.percentile(b, 5)),
                                  "median": float(np.median(b)), "n": len(b)}
        out["variants"][name] = v
    ldf = pd.DataFrame({"harpnum": harps, **{f"label_{n}": labs[n] for n in labs}})
    ldf.to_csv(os.path.join(out_dir, f"clustering_{tag}.csv"), index=False)
    return out, ldf, {h: trajs[h] for h in harps}, mats


# ---------------------------------------------------------------- pilot

def cmd_cluster(a):
    from sklearn.metrics import adjusted_rand_score
    os.makedirs(OUT_DIR, exist_ok=True)
    rec, true = pilot_tables()
    feats, weights = load_cal_weights()
    assert list(feats) == list(CAL_FEATURES), f"B1 FAIL: features entering = {feats}"
    size_series = rec.groupby("harpnum").USFLUX.mean().to_dict()     # transformed log10 flux
    out, ldf, trajs, _ = cluster_run(rec, true, feats, weights, size_series,
                                     "ours_cal_intensive", OUT_DIR)
    # B3 truth correspondence against the frozen true-HMI intensive run
    tl = pd.read_csv(os.path.join(PILOT_DIR, "clustering_true_hmi_intensive.csv"))
    m = ldf.merge(tl, on="harpnum", suffixes=("_cal", "_true"))
    ari = float(adjusted_rand_score(m.label_weighted_cal, m.label_weighted_true))
    matched, mapping = hungarian_match(m.label_weighted_cal.to_numpy(),
                                       m.label_weighted_true.to_numpy())
    out["truth_match"] = {"n_common": int(len(m)), "ari_weighted": ari,
                          "hungarian_matched_fraction": matched, "map": mapping}
    # external validation through the frozen validator (same ar_external.csv)
    shutil.copy(os.path.join(PILOT_DIR, "ar_external.csv"), os.path.join(OUT_DIR, "ar_external.csv"))
    import c23_external
    c23_external.main(["validate", "--source", "ours_cal", "--tag", "intensive",
                        "--out_dir", OUT_DIR])
    ext = json.load(open(os.path.join(OUT_DIR, "external_ours_cal_intensive.json")))
    wv = ext["label_weighted"]
    bars = {
        "B1_features": True,
        "B2_structure": bool(out["k_by_silhouette_unweighted"][str(out["k"])] >= 0.25
                             and out["variants"]["weighted"]["self_ari_boot"]["p5"] >= 0.15),
        "B3_truth": bool(ari >= 0.40 and matched >= 0.75),
        "B4_size_trap": bool(out["variants"]["weighted"]["size_auc"] <= 0.65),
        "B5_external": bool(wv.get("kruskal_p_rate_c", 1.0) <= 0.05
                            and wv.get("rate_ratio_c", 0.0) >= 10
                            and wv.get("cramers_v_mount_wilson", 0.0) >= 0.25),
    }
    out["bars"] = bars
    out["external_weighted"] = {q: wv.get(q) for q in
                                ("rate_ratio_c", "kruskal_p_rate_c", "cramers_v_mount_wilson")}
    out["reading"] = "PASS" if all(bars.values()) else "FAIL: " + \
        ",".join(k for k, v in bars.items() if not v)
    with open(os.path.join(OUT_DIR, "clustering_ours_cal_intensive.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"k": out["k"], "silhouette_k": out["k_by_silhouette_unweighted"][str(out["k"])],
                      "self_ari_p5_weighted": out["variants"]["weighted"]["self_ari_boot"]["p5"],
                      "size_auc_weighted": out["variants"]["weighted"]["size_auc"],
                      "truth_ari": ari, "matched": matched,
                      "external": out["external_weighted"], "bars": bars,
                      "reading": out["reading"]}, indent=1))
    print(f"wrote {OUT_DIR}/clustering_ours_cal_intensive.{{json,csv}}")


# ---------------------------------------------------------------- jitter stability

def residual_model():
    """Per-target per-HARP offset sd + within-HARP AR(1) (rho, sd) from the OOF residuals."""
    out = {}
    cf = pd.read_csv(CORR_FRAMES)
    for t in TARGETS:
        cf[f"res_{t}"] = cf[f"{t}_CAL_oof"] - cf[f"true_{t}"]
    for t in TARGETS:
        g = cf.dropna(subset=[f"res_{t}"]).groupby("harpnum")[f"res_{t}"]
        offsets = g.mean()
        within = cf[f"res_{t}"] - cf.harpnum.map(offsets)
        num, den, n = 0.0, 0.0, 0
        for _, grp in cf.dropna(subset=[f"res_{t}"]).groupby("harpnum"):
            e = (grp[f"res_{t}"] - grp[f"res_{t}"].mean()).to_numpy()
            if len(e) > 2:
                num += float((e[1:] * e[:-1]).sum())
                den += float((e[:-1] ** 2).sum())
                n += len(e)
        rho = num / den if den > 0 else 0.0
        out[t] = {"sd_offset": float(offsets.std()), "rho": float(rho),
                  "sd_within": float(np.nanstd(within)), "n": int(n)}
    return out


def cmd_jitter(a):
    from sklearn.metrics import adjusted_rand_score
    rec, true = pilot_tables()
    feats, weights = load_cal_weights()
    if a.variant == "unweighted":
        weights = None                                    # post-hoc diagnostic, not the prereg bar
    base = json.load(open(os.path.join(OUT_DIR, "clustering_ours_cal_intensive.json")))
    k = int(base["k"])
    ldf = pd.read_csv(os.path.join(OUT_DIR, "clustering_ours_cal_intensive.csv"))
    base_lab = ldf.set_index("harpnum")[f"label_{a.variant}"]
    model = residual_model()
    print("residual model:", json.dumps(model, indent=1))
    coefs = clv_fit(true, feats)
    rng = np.random.default_rng(0)
    results = {"model": model, "draws": {"ar1": [], "iid": []}}
    for kind, n_draws in (("ar1", N_JITTER), ("iid", N_IID)):
        for d in range(n_draws):
            jit = rec.copy()
            for t, cal in (("J_PER_FLUX", "J_PER_FLUX_CAL"), ("H_PER_FLUX", "H_PER_FLUX_CAL")):
                mm = model[t]
                tot_sd = float(np.hypot(mm["sd_offset"], mm["sd_within"]))
                parts = []
                for h, grp in jit.groupby("harpnum", sort=True):
                    n = len(grp)
                    if kind == "iid":
                        noise = rng.normal(0, tot_sd, n)
                    else:
                        e = np.empty(n)
                        e[0] = rng.normal(0, mm["sd_within"])
                        innov = rng.normal(0, mm["sd_within"] * np.sqrt(max(1 - mm["rho"] ** 2, 1e-9)), n)
                        for i in range(1, n):
                            e[i] = mm["rho"] * e[i - 1] + innov[i]
                        noise = rng.normal(0, mm["sd_offset"]) + e
                    parts.append(pd.Series(noise, index=grp.sort_values("_t").index))
                jit[cal] = jit[cal] + pd.concat(parts)
            trajs, _ = trajectories(jit, feats, coefs)
            harps = sorted(trajs)
            D = distance_matrix([trajs[h] for h in harps], weights, log=lambda s: None)
            lab, _ = pam(D, k)
            ref = base_lab.reindex(harps).to_numpy()
            ok = np.isfinite(ref)
            ari = float(adjusted_rand_score(ref[ok].astype(int), lab[ok]))
            results["draws"][kind].append(ari)
            print(f"{kind} draw {d}: ARI {ari:.3f}")
    for kind in ("ar1", "iid"):
        v = np.array(results["draws"][kind])
        results[f"{kind}_summary"] = {"median": float(np.median(v)), "p5": float(np.percentile(v, 5)),
                                      "n": len(v)}
    results["variant"] = a.variant
    results["bar"] = {"median_ar1_ari_ge": 0.60,
                      "pass": bool(results["ar1_summary"]["median"] >= 0.60),
                      "preregistered": a.variant == "weighted"}
    suffix = "" if a.variant == "weighted" else f"_{a.variant}"
    with open(os.path.join(OUT_DIR, f"jitter_stability{suffix}.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k2: results[k2] for k2 in ("ar1_summary", "iid_summary", "bar")}, indent=1))
    print(f"wrote {OUT_DIR}/jitter_stability{suffix}.json")


# ---------------------------------------------------------------- C23

def c23_frames():
    """C23 per-frame corrector inputs + clustering features, mirrored on the pilot build."""
    use = ["tarpnum", "t_rec", "lat_fwt", "lon_fwt", "model", "mask", "n_pixels", "n_finite",
           "n_interior", "USFLUX", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZD",
           "TOTUSJZ", "MEANALP", "MEANJZH", "TOTUSJH", "ABSNJZH", "SAVNCPP", "R_VALUE"]
    df = pd.read_csv(C23_FEATURES, usecols=lambda c: c in set(use))
    sp = df[df["mask"] == "strong_pred"]
    sm = sp[sp.model == "seed_mean"].copy()
    from c23_screening import derive
    sm = derive(sm)
    for kw in [c for c in use if c.isupper()]:
        sm[kw] = transform(kw, sm[kw])
    seeds = sp[sp.model.isin(SEED_MODELS)].copy()
    for kw in ("TOTUSJZ", "TOTUSJH", "USFLUX"):
        seeds[f"log_{kw}"] = transform(kw, seeds[kw])
        g = seeds.groupby(["tarpnum", "t_rec"])[f"log_{kw}"].std().rename(f"std_log10_{kw}")
        sm = sm.merge(g.reset_index(), on=["tarpnum", "t_rec"], how="left")
    sm["center_lon"] = sm.lon_fwt
    sm["_t"] = pd.to_datetime(sm.t_rec.str.replace("_TAI", "", regex=False),
                              format="%Y.%m.%d_%H:%M:%S")
    days = sm._t.dt.floor("D")
    uniq = days.drop_duplicates()
    b0_by_day = {d: None for d in uniq}
    from sunpy.coordinates.sun import B0
    for d in uniq:
        b0_by_day[d] = float(B0(d).to_value("deg"))
    b0 = days.map(b0_by_day).to_numpy()
    lat, lon, b = np.radians(sm.lat_fwt), np.radians(sm.lon_fwt), np.radians(b0)
    sm["center_angle"] = np.degrees(np.arccos(np.clip(
        np.sin(b) * np.sin(lat) + np.cos(b) * np.cos(lat) * np.cos(lon), -1, 1)))
    sm["harpnum"] = sm.tarpnum                      # trajectory grouping key
    return sm


def cmd_c23(a):
    rec23 = c23_frames()
    # covariate compatibility: C23 medians inside the pilot train+val 1st-99th percentiles
    m = load_pairs()
    splits = json.load(open(SPLITS))["splits"]
    fit_h = set(splits["train"]["harps"]) | set(splits["val"]["harps"])
    pil = m[m.harpnum.isin(fit_h)]
    coefs_df = pd.read_csv(os.path.join(OUT_DIR, "corrector_coefs.csv"))
    sha = hashlib.sha256(open(os.path.join(OUT_DIR, "corrector_coefs.csv"), "rb").read()).hexdigest()
    assert sha == open(os.path.join(OUT_DIR, "corrector_coefs.sha256")).read().strip()
    compat, fail = {}, []
    for f in FEATURES:
        lo, hi = np.nanpercentile(pil[f], [1, 99])
        med = float(np.nanmedian(rec23[f])) if f in rec23 else float("nan")
        ok = bool(lo <= med <= hi)
        compat[f] = {"pilot_p1": float(lo), "pilot_p99": float(hi), "c23_median": med, "ok": ok}
        if not ok:
            fail.append(f)
    if fail:
        with open(os.path.join(OUT_DIR, "c23_covariate_check.json"), "w") as f:
            json.dump({"pass": False, "failed": fail, "table": compat}, f, indent=1)
        raise SystemExit(f"covariate check FAILED for {fail}; aborting c23 apply (phase_14 2.4b)")
    # apply the frozen Protocol-B coefficients
    for t in TARGETS:
        cd = coefs_df[coefs_df.target == t].set_index("feature")
        x = np.stack([np.where(np.isfinite(rec23[f].to_numpy(np.float64)),
                               rec23[f].to_numpy(np.float64), cd.loc[f, "impute_median"])
                      for f in FEATURES], axis=1)
        z = (x - cd.loc[list(FEATURES), "scale_mean"].to_numpy()) / \
            cd.loc[list(FEATURES), "scale_sd"].to_numpy()
        rec23[f"{t}_CAL"] = z @ cd.loc[list(FEATURES), "coef"].to_numpy() + \
            float(cd.loc["__intercept__", "coef"])
    keep = ["tarpnum", "t_rec", "center_angle", "center_lon", "J_PER_FLUX_CAL", "H_PER_FLUX_CAL"]
    rec23[keep].to_csv(os.path.join(OUT_DIR, "c23_corrected_ratios.csv"), index=False)
    with open(os.path.join(OUT_DIR, "c23_covariate_check.json"), "w") as f:
        json.dump({"pass": True, "coefs_sha256": sha, "table": compat}, f, indent=1)
    print("covariate check PASSED; applied frozen coefficients to "
          f"{len(rec23)} C23 frames / {rec23.tarpnum.nunique()} TARPs")

    # C23 intensive _CAL clustering (CLV reference = pilot truth fit, frozen rule)
    _, true = pilot_tables()
    feats, weights = load_cal_weights()
    rec23 = rec23[rec23.center_lon.abs() <= LON_MAX]
    size_series = rec23.groupby("harpnum").USFLUX.mean().to_dict()
    out, ldf, trajs, mats = cluster_run(rec23, true, feats, weights, size_series,
                                        "c23_cal_intensive", OUT_DIR, boots=not a.no_boot)
    # cross-cycle regime correspondence: medoid-to-medoid DTW, Hungarian on the cost matrix
    pilot_out = json.load(open(os.path.join(OUT_DIR, "clustering_ours_cal_intensive.json")))
    rec_p, _ = pilot_tables()
    coefs = clv_fit(true, feats)
    ptrajs, _ = trajectories(rec_p, feats, coefs)
    from c23_clustering import dtw
    from scipy.optimize import linear_sum_assignment
    pm = [ptrajs[h] for h in pilot_out["variants"]["weighted"]["medoid_harps"]]
    cm = [trajs[h] for h in out["variants"]["weighted"]["medoid_harps"]]
    cost = np.array([[dtw(x, y, weights) for y in pm] for x in cm])
    ri, cj = linear_sum_assignment(cost)
    out["crosscycle"] = {"cost_matrix": cost.tolist(),
                         "matched_costs": [float(cost[i, j]) for i, j in zip(ri, cj)],
                         "off_costs": [float(cost[i, j]) for i in range(cost.shape[0])
                                       for j in range(cost.shape[1]) if j != dict(zip(ri, cj))[i]],
                         "match": {int(i): int(j) for i, j in zip(ri, cj)}}
    out["crosscycle"]["pass_matched_lt_off"] = bool(
        max(out["crosscycle"]["matched_costs"]) < min(out["crosscycle"]["off_costs"]))
    # simple-window external validation: >= C1 flare rate per observed day, per cluster
    from c23_benchmark import load_flares
    flares = load_flares(years=range(1996, 2012))
    fl = flares[flares.wm2 >= 1e-6]
    frames_per = rec23.groupby("harpnum").agg(t0=("_t", "min"), t1=("_t", "max"),
                                              n=("_t", "size"))
    noaa_path = pd.read_csv(C23_FEATURES, usecols=lambda c: c in {"tarpnum", "noaa_ar"})
    noaa = noaa_path.drop_duplicates("tarpnum").set_index("tarpnum").noaa_ar
    rows = []
    for h in ldf.harpnum:
        w = frames_per.loc[h]
        n_ar = int(noaa.get(h, 0))
        cnt = 0
        if n_ar > 0:
            sel = fl[(fl.noaa_ar == n_ar) & (fl.peak >= w.t0) & (fl.peak <= w.t1)]
            cnt = int(len(sel))
        rows.append({"harpnum": h, "flares_c": cnt, "days": float(w.n) * 96 / 1440})
    exdf = pd.DataFrame(rows).merge(ldf, on="harpnum")
    per = exdf.groupby("label_weighted").agg(n_ars=("harpnum", "size"),
                                             flares=("flares_c", "sum"), days=("days", "sum"))
    per["rate_c"] = per.flares / per.days
    pos = per.rate_c[per.rate_c > 0]
    ratio = float(per.rate_c.max() / pos.min()) if len(pos) else float("nan")
    out["external_simple"] = {"clusters": json.loads(per.to_json(orient="index")),
                              "rate_ratio_c": ratio}
    out["bars"] = {"size_trap": bool(out["variants"]["weighted"]["size_auc"] <= 0.65),
                   "crosscycle": out["crosscycle"]["pass_matched_lt_off"],
                   "flare_ratio_ge5": bool(ratio >= 5)}
    out["reading"] = "PASS" if all(out["bars"].values()) else "FAIL: " + \
        ",".join(k for k, v in out["bars"].items() if not v)
    with open(os.path.join(OUT_DIR, "clustering_c23_cal_intensive.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"k": out["k"], "size_auc": out["variants"]["weighted"]["size_auc"],
                      "matched": out["crosscycle"]["matched_costs"],
                      "off_min": min(out["crosscycle"]["off_costs"]),
                      "rate_ratio": ratio, "bars": out["bars"], "reading": out["reading"]},
                     indent=1))
    print(f"wrote {OUT_DIR}/clustering_c23_cal_intensive.{{json,csv}}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cluster")
    c.set_defaults(fn=cmd_cluster)
    j = sub.add_parser("jitter")
    j.add_argument("--variant", choices=("weighted", "unweighted"), default="weighted")
    j.set_defaults(fn=cmd_jitter)
    t = sub.add_parser("c23")
    t.add_argument("--no_boot", action="store_true", help="skip the 200-resample self-ARI")
    t.set_defaults(fn=cmd_c23)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
