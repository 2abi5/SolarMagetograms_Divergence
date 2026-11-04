"""Phase 13 / F1 + F2: reliability bracket and cross-cycle clustering on the Cycle-23 record
(reports/phase_13.md 4).

    python scripts/c23_validation.py f1   # 96/192-min brackets on the C23 seed-mean record
    python scripts/c23_validation.py f2   # C23 trajectories -> clusters -> match to C24 (pilot)

f1: per candidate keyword the consecutive-frame correlations (lag 96 and 192 min) of the
    reconstructed Cycle-23 record, bootstrap over TARPs, next to the Cycle-24 reconstructed
    values from the pilot (phase_12 E2b, 'ours' rows) -> results/full/c23_build/f1_reliability.csv
f2: Cycle-23 trajectories under the phase_12 rules (|lon| <= 60, >= 45 frames, CLV coefficients
    fitted on the Cycle-24 true-HMI series, per-cycle z), DTW k-medoids (weighted + unweighted,
    k by silhouette on the unweighted matrix), Hungarian match of the weighted clusters to the
    pilot's C24 'ours' weighted clusters on cross-cycle medoid distances
    -> results/full/c23_build/f2_crosscycle.json, clustering_c23.csv
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
import c23_clustering as icc  # noqa: E402
import c23_screening as ikr  # noqa: E402

C23_FEATURES = os.path.join(ROOT, "results", "full", "c23_build", "features.csv")
PILOT = os.path.join(ROOT, "results", "full", "overlap_pilot")
OUT = os.path.join(ROOT, "results", "full", "c23_build")


def center_angle_deg(lat_deg, lon_deg):
    """Heliocentric angle from disk centre: cos(theta) = cos(lat) cos(lon)."""
    la, lo = np.radians(np.asarray(lat_deg, np.float64)), np.radians(np.asarray(lon_deg, np.float64))
    return np.degrees(np.arccos(np.clip(np.cos(la) * np.cos(lo), -1.0, 1.0)))


def c23_as_pilot_schema(df, model="seed_mean", mask="strong_pred"):
    """C23 extraction rows -> the pilot frame schema the E2/E3 machinery consumes:
    harpnum <- tarpnum, mdi_t_rec/sharp_t_rec <- t_rec, center_lon <- lon_fwt,
    center_angle from (lat_fwt, lon_fwt); keyword and count columns pass through."""
    sub = df[(df.model == model) & (df["mask"] == mask)].copy()
    sub["harpnum"] = sub.tarpnum.astype(int)
    sub["mdi_t_rec"] = sub.t_rec
    sub["sharp_t_rec"] = sub.t_rec
    sub["center_lon"] = sub.lon_fwt
    sub["center_angle"] = center_angle_deg(sub.lat_fwt, sub.lon_fwt)
    sub["model"] = "c23"
    return sub


def candidates(weights_csv=None):
    rel = pd.read_csv(weights_csv or os.path.join(PILOT, "keyword_reliability.csv"))
    inc = rel[rel.included]
    return list(inc.keyword), inc.weight.to_numpy(np.float64)


def cmd_f1(a):
    feats, _ = candidates()
    df = ikr.derive(c23_as_pilot_schema(pd.read_csv(a.features)))
    pilot_recon = pd.read_csv(os.path.join(PILOT, "keyword_reliability_recon.csv"))
    rows = []
    for lag in (96, 192):
        m = ikr.lag_pairs(df, lag)
        for kw in feats:
            r, n = ikr.corr_by(m, f"{kw}_a", f"{kw}_b", kw)
            lo, hi = ikr.boot_r(m, f"{kw}_a", f"{kw}_b", kw, n_boot=a.n_boot)
            c24 = pilot_recon[(pilot_recon.source == "ours") & (pilot_recon.lag_min == lag) &
                              (pilot_recon.keyword == kw)]
            rows.append({"keyword": kw, "lag_min": lag, "r_c23": r, "n_pairs": n,
                         "r_lo": lo, "r_hi": hi,
                         "r_c24_ours": float(c24.r.iloc[0]) if len(c24) else float("nan")})
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "f1_reliability.csv"), index=False)
    print(out.round(4).to_string(index=False))
    print(f"wrote {OUT}/f1_reliability.csv")


def cmd_f2(a):
    from scipy.optimize import linear_sum_assignment
    feats, weights = candidates(a.weights)
    tag = f"_{a.tag}" if a.tag else ""
    c23 = ikr.derive(c23_as_pilot_schema(pd.read_csv(a.features)))
    for f in feats:
        c23[f] = ikr.transform(f, c23[f])
    pilot_feat = pd.read_csv(os.path.join(PILOT, "features.csv"))
    coefs = icc.clv_fit(icc.source_frames(pilot_feat, "true_hmi", feats), feats)  # frozen: C24 truth
    c23 = icc.clv_detrend(c23, coefs)
    c23 = c23[c23.center_lon.abs() <= icc.LON_MAX]
    stats = {f: (float(np.nanmean(c23[f])), float(np.nanstd(c23[f]) or 1.0)) for f in feats}
    for f in feats:
        c23[f] = (c23[f] - stats[f][0]) / stats[f][1]      # per-cycle standardization (frozen)
    c23["_t"] = pd.to_datetime(c23.mdi_t_rec.str.replace("_TAI", "", regex=False),
                               format="%Y.%m.%d_%H:%M:%S")
    trajs = {}
    for h, g in c23.groupby("harpnum"):
        arr = g.sort_values("_t")[feats].to_numpy(np.float64)
        arr = arr[np.isfinite(arr).all(1)]
        if len(arr) >= icc.MIN_FRAMES:
            trajs[int(h)] = arr
    harps = sorted(trajs)
    T = [trajs[h] for h in harps]
    print(f"C23 trajectories: {len(T)} of {c23.harpnum.nunique()} TARPs")
    mats = {}
    for name, w in (("weighted", weights), ("unweighted", None)):
        path = os.path.join(OUT, f"dtw_c23{tag}_{name}.npy")
        if os.path.exists(path) and not a.recompute:
            mats[name] = np.load(path)
        else:
            mats[name] = icc.distance_matrix(T, w)
            np.save(path, mats[name])
    k, sil = icc.silhouette_k(mats["unweighted"])
    labs = {}
    out = {"n_trajectories": len(T), "k": int(k),
           "k_by_silhouette_unweighted": {str(q): v for q, v in sil.items()}, "variants": {}}
    for name, D in mats.items():
        lab, med = icc.pam(D, k)
        labs[name] = lab
        boots = icc.bootstrap_ari(D, k, n_boot=100)
        out["variants"][name] = {"cluster_sizes": [int((lab == c).sum()) for c in range(k)],
                                 "medoid_tarps": [int(harps[m]) for m in med],
                                 "self_ari_boot_p5": float(np.percentile(boots, 5))}
    # Hungarian match of C23 weighted clusters to the pilot C24 'ours' weighted clusters:
    # distance between cluster centres = DTW between medoid trajectories across cycles
    with open(os.path.join(PILOT, f"clustering_ours{tag}.json")) as f:
        c24run = json.load(f)
    c24lab = pd.read_csv(os.path.join(PILOT, f"clustering_ours{tag}.csv"))
    c24feats = c24run["features"]
    if c24feats != feats:
        raise SystemExit(f"feature mismatch: pilot {c24feats} vs {feats}")
    c24traj, _ = icc.build_trajectories(pilot_feat, "ours", feats,
                                        coefs=icc.clv_fit(icc.source_frames(pilot_feat, "true_hmi", feats), feats))
    k24 = c24run["k"]
    med24 = [c24run["variants"]["weighted"]["medoid_harps"][i] for i in range(k24)]
    cost = np.zeros((k, k24))
    for i, mh in enumerate(out["variants"]["weighted"]["medoid_tarps"]):
        for j, h24 in enumerate(med24):
            cost[i, j] = icc.dtw(trajs[mh], c24traj[int(h24)], weights)
    ri, cj = linear_sum_assignment(cost)
    out["match_to_c24"] = {"pairs": [[int(x), int(y)] for x, y in zip(ri, cj)],
                           "medoid_dtw_cost": [[float(c) for c in row] for row in cost]}
    pd.DataFrame({"tarpnum": harps, **{f"label_{n}": labs[n] for n in labs}}) \
        .to_csv(os.path.join(OUT, f"clustering_c23{tag}.csv"), index=False)
    with open(os.path.join(OUT, f"f2_crosscycle{tag}.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({q: out[q] for q in ("n_trajectories", "k")}, default=str))
    print(f"wrote {OUT}/f2_crosscycle{tag}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f1 = sub.add_parser("f1")
    f1.add_argument("--features", default=C23_FEATURES)
    f1.add_argument("--n_boot", type=int, default=500)
    f1.set_defaults(fn=cmd_f1)
    f2 = sub.add_parser("f2")
    f2.add_argument("--features", default=C23_FEATURES)
    f2.add_argument("--recompute", action="store_true")
    f2.add_argument("--weights", default=None)
    f2.add_argument("--tag", default="")
    f2.set_defaults(fn=cmd_f2)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
