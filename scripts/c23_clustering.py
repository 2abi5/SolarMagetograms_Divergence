"""Overlap-pilot stage E3: reliability-weighted DTW k-medoids over AR keyword trajectories
(reports/phase_12.md 5).

    python scripts/c23_clustering.py run --source ours --out_dir results/full/overlap_pilot
    python scripts/c23_clustering.py run --source true_hmi ...
    python scripts/c23_clustering.py confound --out_dir results/full/overlap_pilot

run:      build trajectories for one source (seed-mean, E2-included features, transform, z-score,
          |center_lon| <= 60 deg, center-to-limb detrend fit on the true_hmi series, >= 45 frames),
          DTW distance matrices for the frozen variants (weighted / unweighted / flux-only),
          k by silhouette on the unweighted matrix (k = 2..8, then fixed), PAM k-medoids,
          bootstrap stability over ARs (ARI), -> clustering_<source>.{json,csv}, dtw_<source>.npz
confound: cluster ours and true_hmi separately (their saved runs), Hungarian-match the clusters
          on cross-source medoid distances, report cross-source ARI and per-feature z offsets
          (W1 instrument-confound control) -> confound.json
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
from c23_screening import (LON_MAX, MIN_FRAMES, SOURCES, clv_detrend, clv_fit,  # noqa: E402
                                      derive, seed_mean, transform)

K_RANGE = range(2, 9)
N_BOOT = 200


# ---------------------------------------------------------------- DTW

def dtw(a, b, w=None):
    """DTW distance between (Ta, K) and (Tb, K); local cost sqrt(sum_k w_k (a_k - b_k)^2).
    Anti-diagonal dynamic program (no window), NaN-free inputs required."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.ndim == 1:
        a, b = a[:, None], b[:, None]
    d = a[:, None, :] - b[None, :, :]
    if w is not None:
        d = d * np.sqrt(np.asarray(w, np.float64))
    c = np.sqrt((d * d).sum(-1))
    ta, tb = c.shape
    big = np.inf
    prev2 = np.full(min(ta, tb) + 2, big)      # diagonal s-2
    prev1 = np.full(min(ta, tb) + 2, big)      # diagonal s-1
    # cells on diagonal s: i + j = s; i in [max(0, s-tb+1), min(ta-1, s)]
    for s in range(ta + tb - 1):
        i0, i1 = max(0, s - tb + 1), min(ta - 1, s)
        i = np.arange(i0, i1 + 1)
        cur = np.full(i.size + 2, big)
        cost = c[i, s - i]
        if s == 0:
            cur[1] = cost[0]
        else:
            p0, pi0 = prev1, max(0, s - 1 - tb + 1)
            up = np.full(i.size, big)          # D[i-1, j]   (diagonal s-1, index i-1)
            left = np.full(i.size, big)        # D[i, j-1]   (diagonal s-1, index i)
            diag = np.full(i.size, big)        # D[i-1, j-1] (diagonal s-2, index i-1)
            k = i - 1 - pi0
            ok = (i - 1 >= 0) & (k >= 0)
            up[ok] = p0[1 + k[ok]]
            k = i - pi0
            ok = (s - i - 1 >= 0) & (k >= 0) & (k < p0.size - 2)
            left[ok] = p0[1 + k[ok]]
            pi2 = max(0, s - 2 - tb + 1)
            k = i - 1 - pi2
            ok = (i - 1 >= 0) & (s - i - 1 >= 0) & (k >= 0) & (k < prev2.size - 2)
            diag[ok] = prev2[1 + k[ok]]
            cur[1:-1] = cost + np.minimum(np.minimum(up, left), diag)
        prev2, prev1 = prev1, cur
    return float(prev1[1 + (ta - 1) - max(0, (ta + tb - 2) - tb + 1)])


def distance_matrix(trajs, w=None, log=print):
    """Condensed-to-square DTW matrix over a list of (T_i, K) arrays."""
    n = len(trajs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw(trajs[i], trajs[j], w)
        if i % 25 == 0:
            log(f"dtw row {i}/{n}")
    return D


# ---------------------------------------------------------------- PAM k-medoids

def pam(D, k, seed=0, max_iter=100):
    """k-medoids (BUILD + SWAP) on a precomputed distance matrix; returns (labels, medoids)."""
    n = len(D)
    rng = np.random.default_rng(seed)
    med = [int(np.argmin(D.sum(1)))]                       # BUILD: most central first
    while len(med) < k:
        dmin = D[:, med].min(1)
        gain = np.maximum(dmin[None, :] - D, 0.0).sum(1)   # how much each candidate would reduce cost
        gain[med] = -1
        med.append(int(np.argmax(gain)))
    med = np.array(sorted(med))
    for _ in range(max_iter):
        lab = np.argmin(D[:, med], 1)
        cost = D[np.arange(n), med[lab]].sum()
        best = (cost, None)
        for mi in range(k):
            others = med[np.arange(k) != mi]
            for h in range(n):
                if h in med:
                    continue
                trial = np.sort(np.append(others, h))
                c = D[np.arange(n), trial[np.argmin(D[:, trial], 1)]].sum()
                if c < best[0] - 1e-12:
                    best = (c, trial)
        if best[1] is None:
            break
        med = best[1]
    lab = np.argmin(D[:, med], 1)
    _ = rng                                                # seed kept for signature stability
    return lab, med


def size_auc(labels, size_vec):
    """AUC with which per-trajectory mean (transformed) USFLUX separates the TWO LARGEST clusters
    (phase_13 size-trap statistic: ~0.5 = clusters are not size classes)."""
    from sklearn.metrics import roc_auc_score
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        return float("nan")
    a, b = counts.index[0], counts.index[1]
    sel = np.isin(labels, [a, b]) & np.isfinite(size_vec)
    y = (np.asarray(labels)[sel] == a)
    if y.all() or not y.any():
        return float("nan")
    auc = roc_auc_score(y, np.asarray(size_vec)[sel])
    return float(max(auc, 1.0 - auc))


def silhouette_k(D, k_range=K_RANGE, seed=0):
    """(best_k, {k: silhouette}) on a precomputed distance matrix."""
    from sklearn.metrics import silhouette_score
    scores = {}
    for k in k_range:
        lab, _ = pam(D, k, seed)
        scores[k] = float(silhouette_score(D, lab, metric="precomputed")) if len(set(lab)) > 1 else -1.0
    best = max(scores, key=scores.get)
    return best, scores


def bootstrap_ari(D, k, n_boot=N_BOOT, seed=0):
    """Self-stability: resample ARs with replacement, recluster the (deduplicated) sample, ARI
    against the full-sample labels on the sampled ARs. Returns the list of ARIs."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    full, _ = pam(D, k)
    out = []
    n = len(D)
    for _ in range(n_boot):
        idx = np.unique(rng.choice(n, size=n, replace=True))
        if len(idx) <= k:
            continue
        lab, _ = pam(D[np.ix_(idx, idx)], k)
        out.append(float(adjusted_rand_score(full[idx], lab)))
    return out


# ---------------------------------------------------------------- trajectories

def source_frames(feat, source, features):
    """Seed-mean frame table of one source with transformed candidate features."""
    mask = "strong" if source == "true_hmi" else "strong_pred"
    sub = feat[feat["mask"] == mask]
    sub = sub[sub.center_lon.abs() <= LON_MAX] if "center_lon" in sub.columns else sub
    sm = seed_mean(sub, SOURCES[source])
    sm = derive(sm.merge(sub[["harpnum", "mdi_t_rec", "center_lon"]].drop_duplicates(),
                         on=["harpnum", "mdi_t_rec"], how="left"))
    for f in features:
        sm[f] = transform(f, sm[f])
    return sm


def build_trajectories(feat, source, features, min_frames=MIN_FRAMES, coefs=None):
    """{harpnum: (T, K) z-scored arrays}; z over all frames of the source set; CLV-detrended."""
    sm = source_frames(feat, source, features)
    if coefs is not None:
        sm = clv_detrend(sm, coefs)
    stats = {f: (float(np.nanmean(sm[f])), float(np.nanstd(sm[f]) or 1.0)) for f in features}
    for f in features:
        sm[f] = (sm[f] - stats[f][0]) / stats[f][1]
    sm["_t"] = pd.to_datetime(sm.mdi_t_rec.str.replace("_TAI", "", regex=False), format="%Y.%m.%d_%H:%M:%S")
    out = {}
    for h, g in sm.groupby("harpnum"):
        g = g.sort_values("_t")
        arr = g[list(features)].to_numpy(np.float64)
        arr = arr[np.isfinite(arr).all(1)]
        if len(arr) >= min_frames:
            out[int(h)] = arr
    return out, stats


# ---------------------------------------------------------------- commands

def cmd_run(a):
    from sklearn.metrics import adjusted_rand_score
    feat = pd.read_csv(a.features)
    rel = pd.read_csv(a.weights)
    inc = rel[rel.included] if "included" in rel.columns else rel
    features = list(inc.keyword)
    weights = inc.weight.to_numpy(np.float64)
    if not features:
        raise SystemExit("no features passed the E2 bar; see keyword_reliability.csv")
    # a source keeps only the features it can measure (E0 baselines have no transverse field,
    # so the current keywords are identically zero -> NaN on the log transform); weights are
    # renormalized to mean 1 over the kept set and the drop is recorded.
    sf = source_frames(feat, a.source, features)
    feasible = [f for f in features if np.isfinite(sf[f]).mean() >= 0.5]
    dropped = sorted(set(features) - set(feasible))
    if dropped:
        keep = [features.index(f) for f in feasible]
        weights = weights[keep]
        weights = weights / weights.mean() if weights.size else weights
        features = feasible
        print(f"{a.source}: dropped infeasible features {dropped}")
    if not features:
        raise SystemExit(f"{a.source}: no feasible features")
    coefs = clv_fit(source_frames(feat, "true_hmi", features), features)
    trajs, stats = build_trajectories(feat, a.source, features, coefs=coefs)
    harps = sorted(trajs)
    T = [trajs[h] for h in harps]
    if len(T) <= max(K_RANGE):
        raise SystemExit(f"{a.source}: only {len(T)} trajectories; nothing to cluster")
    # per-trajectory mean log10 USFLUX for the size-separation statistic (phase_13 addition)
    sfl = source_frames(feat, a.source, ["USFLUX"])
    size_of = sfl.groupby("harpnum").USFLUX.mean().to_dict()
    size_vec = np.array([size_of.get(h, np.nan) for h in harps])
    print(f"{a.source}: {len(T)} trajectories, features {features}, weights {np.round(weights, 3)}")
    variants = {"weighted": (T, weights), "unweighted": (T, None)}
    if "USFLUX" in features:
        fi = features.index("USFLUX")
        variants["flux_only"] = ([t[:, [fi]] for t in T], None)
    tag = f"_{a.tag}" if a.tag else ""
    mats = {}
    for name, (trs, w) in variants.items():
        path = os.path.join(a.out_dir, f"dtw_{a.source}{tag}_{name}.npy")
        if os.path.exists(path) and not a.recompute:
            mats[name] = np.load(path)
        else:
            mats[name] = distance_matrix(trs, w)
            np.save(path, mats[name])
    k, sil = silhouette_k(mats["unweighted"])
    out = {"source": a.source, "n_trajectories": len(T), "features": features,
           "weights": [float(x) for x in weights], "k_by_silhouette_unweighted": {str(q): v for q, v in sil.items()},
           "k": int(k), "clv_coefs": {f: list(c) for f, c in coefs.items()},
           "z_stats": {f: list(s) for f, s in stats.items()}, "variants": {}}
    labs = {}
    for name, D in mats.items():
        lab, med = pam(D, k)
        boots = bootstrap_ari(D, k)
        labs[name] = lab
        out["variants"][name] = {"medoid_harps": [int(harps[m]) for m in med],
                                 "cluster_sizes": [int((lab == c).sum()) for c in range(k)],
                                 "self_ari_boot": {"p5": float(np.percentile(boots, 5)),
                                                   "median": float(np.median(boots)), "n": len(boots)},
                                 "size_auc": size_auc(lab, size_vec)}
    out["ari_weighted_vs_unweighted"] = float(adjusted_rand_score(labs["weighted"], labs["unweighted"]))
    if "flux_only" in labs:
        out["ari_weighted_vs_flux_only"] = float(adjusted_rand_score(labs["weighted"], labs["flux_only"]))
    p5 = out["variants"]["unweighted"]["self_ari_boot"]["p5"]
    out["weighting_changes_clusters"] = bool(out["ari_weighted_vs_unweighted"] < p5)
    pd.DataFrame({"harpnum": harps, **{f"label_{n}": labs[n] for n in labs}}) \
        .to_csv(os.path.join(a.out_dir, f"clustering_{a.source}{tag}.csv"), index=False)
    with open(os.path.join(a.out_dir, f"clustering_{a.source}{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({q: out[q] for q in ("k", "ari_weighted_vs_unweighted", "weighting_changes_clusters")}))
    print(f"wrote {a.out_dir}/clustering_{a.source}{tag}.{{json,csv}}")


def cmd_confound(a):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score
    runs, labs, feats = {}, {}, None
    for s in ("ours", "true_hmi"):
        with open(os.path.join(a.out_dir, f"clustering_{s}.json")) as f:
            runs[s] = json.load(f)
        labs[s] = pd.read_csv(os.path.join(a.out_dir, f"clustering_{s}.csv"))
        feats = runs[s]["features"]
    m = labs["ours"].merge(labs["true_hmi"], on="harpnum", suffixes=("_ours", "_true"))
    ari = float(adjusted_rand_score(m.label_weighted_ours, m.label_weighted_true))
    k = runs["ours"]["k"]
    conf = np.zeros((k, runs["true_hmi"]["k"]))
    for i in range(k):
        for j in range(runs["true_hmi"]["k"]):
            conf[i, j] = ((m.label_weighted_ours == i) & (m.label_weighted_true == j)).sum()
    ri, cj = linear_sum_assignment(-conf)
    matched = float(conf[ri, cj].sum() / conf.sum()) if conf.sum() else float("nan")
    feat = pd.read_csv(a.features)
    shift = {}
    for f in feats:
        o = source_frames(feat, "ours", feats)[f]
        t = source_frames(feat, "true_hmi", feats)[f]
        sd = float(np.nanstd(t)) or 1.0
        shift[f] = float((np.nanmean(o) - np.nanmean(t)) / sd)
    out = {"n_common": int(len(m)), "ari_ours_vs_true_weighted": ari,
           "hungarian_matched_fraction": matched,
           "match": {int(i): int(j) for i, j in zip(ri, cj)},
           "feature_domain_shift_z": shift}
    with open(os.path.join(a.out_dir, "confound.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run")
    rn.add_argument("--source", choices=sorted(SOURCES), required=True)
    rn.add_argument("--features", default=os.path.join("results", "full", "overlap_pilot", "features.csv"))
    rn.add_argument("--weights", default=os.path.join("results", "full", "overlap_pilot", "keyword_reliability.csv"))
    rn.add_argument("--out_dir", default=os.path.join("results", "full", "overlap_pilot"))
    rn.add_argument("--recompute", action="store_true")
    rn.add_argument("--tag", default="", help="suffix for output names (e.g. 'intensive')")
    rn.set_defaults(fn=cmd_run)
    cf = sub.add_parser("confound")
    cf.add_argument("--features", default=os.path.join("results", "full", "overlap_pilot", "features.csv"))
    cf.add_argument("--out_dir", default=os.path.join("results", "full", "overlap_pilot"))
    cf.set_defaults(fn=cmd_confound)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
