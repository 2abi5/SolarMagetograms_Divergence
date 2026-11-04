"""Overlap-pilot stage E2: keyword-level test-retest reliability -> the clustering weights
(reports/phase_12.md 4).

    python scripts/c23_screening.py real    --run full
    python scripts/c23_screening.py recon   --features results/full/overlap_pilot/features.csv
    python scripts/c23_screening.py weights --run full --features results/full/overlap_pilot/features.csv

real:    from the Phase 10 raw SHARP pairs (data/<run>/reliability/raw), each record on its OWN
         strong mask (deployment condition; keywords are patch integrals, so no pixel alignment):
         per keyword the test-retest r12 and r24 across frames on the analysis transform, with
         95 % bootstrap intervals over HARPs -> data/<run>/reliability/keyword_records.csv and
         results/full/overlap_pilot/keyword_reliability_real.csv
recon:   from E1's features.csv: per source (ours, mse, e0c, e0_mu seed-means on strong_pred;
         true_hmi on strong) the lag-96 and lag-192 min consecutive-frame correlations per
         keyword (the plan's MDI-side lag bracket; W4 smoothing check = ours vs true_hmi)
         -> results/full/overlap_pilot/keyword_reliability_recon.csv
weights: merge real r12 with the reconstruction agreement Spearman(pred, true) across frames,
         apply the frozen entry bar (r12 >= 0.70 AND agreement >= 0.70), weight w_k prop. to
         r12/(1-r12) normalized to mean 1 over the included candidates
         -> results/full/overlap_pilot/keyword_reliability.{csv,json}
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
from reliability_pairs import fits_name, lag_t_recs, shared_noise_upper  # noqa: E402
from sr import physics as ph  # noqa: E402
from sr import sharp_keywords as sk  # noqa: E402

SEGMENTS = ("Bp", "Bt", "Br")
NORM = 3500.0
LAGS = (12, 24)
CANDIDATES = ("USFLUX", "TOTUSJZ", "TOTUSJH", "R_VALUE", "AREA", "STRONG_FRAC")
CANDIDATES_INTENSIVE = ("J_PER_FLUX", "H_PER_FLUX", "MEANGBZ", "STRONG_FRAC")   # phase_13 addition
LOG10 = frozenset({"USFLUX", "TOTUSJZ", "TOTUSJH", "SAVNCPP", "ABSNJZH", "AREA",
                   "J_PER_FLUX", "H_PER_FLUX"})
DERIVED = ("AREA", "STRONG_FRAC", "J_PER_FLUX", "H_PER_FLUX")
R12_BAR, AGREE_BAR = 0.70, 0.70
LON_MAX = 60.0
MIN_FRAMES = 45            # 3 days at 96 min: the trajectory entry condition
SOURCES = {"ours": ("F_U_E3_s0_e200", "F_U_E3_s1_e200", "F_U_E3_s2"),
           "mse": ("F_U_E1_s0", "F_U_E1_s1", "F_U_E1_s2_e200"),
           "e0c": ("E0c",), "e0_mu": ("E0_mu",), "true_hmi": ("true_hmi",)}


def transform(keyword, v):
    """Analysis transform (frozen in phase_12.md 4): log10 for extensive positive keywords, else raw."""
    v = np.asarray(v, np.float64)
    if keyword in LOG10:
        return np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
    return v


def derive(df):
    """Add AREA (Mm^2, from the strong-pixel count), STRONG_FRAC, and the intensive alpha-like
    ratios J_PER_FLUX = TOTUSJZ/USFLUX, H_PER_FLUX = TOTUSJH/USFLUX (phase_13; size-free)."""
    px_mm2 = ph.pixel_scale_mm(ph.CDELT_HR_DEG) ** 2
    df = df.copy()
    df["AREA"] = df["n_pixels"] * px_mm2
    df["STRONG_FRAC"] = np.where(df["n_finite"] > 0, df["n_pixels"] / df["n_finite"], np.nan)
    if "TOTUSJZ" in df.columns and "USFLUX" in df.columns:
        flux = df["USFLUX"].to_numpy(np.float64)
        ok = np.isfinite(flux) & (flux > 0)
        df["J_PER_FLUX"] = np.where(ok, df["TOTUSJZ"] / np.where(ok, flux, 1.0), np.nan)
        if "TOTUSJH" in df.columns:
            df["H_PER_FLUX"] = np.where(ok, df["TOTUSJH"] / np.where(ok, flux, 1.0), np.nan)
    return df


def corr_by(df, col_a, col_b, keyword):
    """Pearson r between transformed col_a and col_b over rows with both finite; (r, n)."""
    a, b = transform(keyword, df[col_a]), transform(keyword, df[col_b])
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def boot_r(df, col_a, col_b, keyword, n_boot=2000, seed=0):
    """95 % bootstrap interval over HARPs for corr_by."""
    rng = np.random.default_rng(seed)
    harps = df.harpnum.unique()
    by = {h: g for h, g in df.groupby("harpnum")}
    rs = []
    for _ in range(n_boot):
        take = rng.choice(harps, size=len(harps), replace=True)
        r, _ = corr_by(pd.concat([by[h] for h in take]), col_a, col_b, keyword)
        if np.isfinite(r):
            rs.append(r)
    if len(rs) < n_boot // 2:
        return float("nan"), float("nan")
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def snr2(r):
    """Squared test-retest SNR = rho / (1 - rho); the weight before normalization."""
    return float(r / (1.0 - r)) if np.isfinite(r) and r < 1.0 else float("nan")


# ---------------------------------------------------------------- real

def record_keywords(stack_gauss, strong_gauss=200.0):
    """Keywords of one record on its own strong mask, plus the finite-pixel count."""
    st = np.asarray(stack_gauss, np.float64)
    fin = np.isfinite(st).all(axis=0)
    babs = np.sqrt(np.nansum(np.where(fin, st, 0.0) ** 2, axis=0))
    out = sk.keywords(st, mask=fin & (babs > strong_gauss), cdelt_deg=ph.CDELT_HR_DEG)
    out["n_finite"] = int(fin.sum())
    return out


def cmd_real(a):
    import noise_floor_temporal as nft
    rel_dir = os.path.join(ROOT, "data", a.run, "reliability")
    raw = os.path.join(rel_dir, "raw")
    frames = pd.read_csv(os.path.join(rel_dir, "frames.csv"))
    rows = []
    for i, r in enumerate(frames.itertuples()):
        for lag, t in zip((0,) + LAGS, [r.sharp_t_rec] + lag_t_recs(r.sharp_t_rec)):
            paths = {s: os.path.join(raw, fits_name(r.harpnum, t, s)) for s in SEGMENTS}
            if not all(os.path.exists(p) for p in paths.values()):
                continue
            _, st = nft.load_stack(paths)
            rows.append({"harpnum": int(r.harpnum), "t_rec": r.sharp_t_rec, "month": r.month,
                         "lag_min": lag, **record_keywords(st * NORM, a.strong_gauss)})
        if i % 40 == 0:
            print(f"{i}/{len(frames)} frames", flush=True)
    rec = pd.DataFrame(rows)
    rec.to_csv(os.path.join(rel_dir, "keyword_records.csv"), index=False)

    rec = derive(rec)
    base = rec[rec.lag_min == 0]
    out = []
    for kw in list(sk.KEYWORDS) + list(DERIVED):
        row = {"keyword": kw, "transform": "log10" if kw in LOG10 else "raw"}
        for lag in LAGS:
            m = base.merge(rec[rec.lag_min == lag], on=["harpnum", "t_rec"], suffixes=("_a", "_b"))
            r, n = corr_by(m, f"{kw}_a", f"{kw}_b", kw)
            lo, hi = boot_r(m, f"{kw}_a", f"{kw}_b", kw, a.n_boot)
            row.update({f"r{lag}": r, f"n{lag}": n, f"r{lag}_lo": lo, f"r{lag}_hi": hi})
        row["c_upper"] = shared_noise_upper(row["r12"], row["r24"]) \
            if np.isfinite(row["r12"]) and np.isfinite(row["r24"]) else float("nan")
        row["snr2_r12"] = snr2(row["r12"])
        out.append(row)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "keyword_reliability_real.csv")
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"wrote {path}")


# ---------------------------------------------------------------- recon

def seed_mean(df, exps):
    """Seed-mean keyword values per frame: mean over the exps' rows (the record the dataset releases)."""
    g = df[df.model.isin(exps)].groupby(["harpnum", "mdi_t_rec"], as_index=False)
    cols = [c for c in df.columns if c in sk.KEYWORDS or c in ("n_pixels", "n_finite", "n_interior")]
    return g.agg({**{c: "mean" for c in cols}, "sharp_t_rec": "first", "center_angle": "first"})


def lag_pairs(df, lag_min, tol_min=2.0):
    """Within-HARP row pairs whose mdi_t_rec differ by lag_min (+- tol); returns a merged frame (_a, _b)."""
    df = df.copy()
    df["_t"] = pd.to_datetime(df.mdi_t_rec.str.replace("_TAI", "", regex=False), format="%Y.%m.%d_%H:%M:%S")
    out = []
    for h, g in df.groupby("harpnum"):
        g = g.sort_values("_t").reset_index(drop=True)
        t = g["_t"].to_numpy()
        for i in range(len(g)):
            d = (t[i + 1:] - t[i]) / np.timedelta64(1, "m")
            for j in np.nonzero(np.abs(d - lag_min) <= tol_min)[0]:
                out.append((g.iloc[i], g.iloc[i + 1 + j]))
    if not out:
        return pd.DataFrame()
    aa = pd.DataFrame([x[0] for x in out]).reset_index(drop=True).add_suffix("_a")
    bb = pd.DataFrame([x[1] for x in out]).reset_index(drop=True).add_suffix("_b")
    m = pd.concat([aa, bb], axis=1)
    m["harpnum"] = m["harpnum_a"]
    return m


def cmd_recon(a):
    df = pd.read_csv(a.features)
    out = []
    for source, exps in SOURCES.items():
        mask = "strong" if source == "true_hmi" else "strong_pred"
        sub = derive(seed_mean(df[df["mask"] == mask], exps))
        for lag in (96, 192):
            m = lag_pairs(sub, lag)
            if m.empty:
                continue
            for kw in CANDIDATES:
                r, n = corr_by(m, f"{kw}_a", f"{kw}_b", kw)
                lo, hi = boot_r(m, f"{kw}_a", f"{kw}_b", kw, a.n_boot)
                out.append({"source": source, "mask": mask, "lag_min": lag, "keyword": kw,
                            "r": r, "n_pairs": n, "r_lo": lo, "r_hi": hi})
    os.makedirs(os.path.dirname(a.features), exist_ok=True)
    path = os.path.join(os.path.dirname(a.features), "keyword_reliability_recon.csv")
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"wrote {path}")


# ---------------------------------------------------------------- weights

def clv_fit(df, features):
    """Center-to-limb detrend coefficients per feature: value ~ a + b mu + c mu^2 (mu = cos angle),
    fit pooled over the given frames; returns {feature: (b, c)} for the mu-dependent part."""
    mu = np.cos(np.radians(df["center_angle"].to_numpy(np.float64)))
    out = {}
    for f in features:
        v = df[f].to_numpy(np.float64)
        ok = np.isfinite(v) & np.isfinite(mu)
        X = np.stack([np.ones(int(ok.sum())), mu[ok], mu[ok] ** 2], 1)
        beta = np.linalg.lstsq(X, v[ok], rcond=None)[0]
        out[f] = (float(beta[1]), float(beta[2]))
    return out


def clv_detrend(df, coefs):
    """Subtract the mu-dependent part, referenced to disk center (mu = 1)."""
    df = df.copy()
    mu = np.cos(np.radians(df["center_angle"].to_numpy(np.float64)))
    for f, (b, c) in coefs.items():
        df[f] = df[f] - (b * (mu - 1.0) + c * (mu ** 2 - 1.0))
    return df


def noise_var(rec, kw, lag=12):
    """Measurement-noise variance on the transform: Var(v_{t+lag} - v_t) / 2 over the real pairs."""
    base = rec[rec.lag_min == 0]
    m = base.merge(rec[rec.lag_min == lag], on=["harpnum", "t_rec"], suffixes=("_a", "_b"))
    d = transform(kw, m[f"{kw}_b"]) - transform(kw, m[f"{kw}_a"])
    d = d[np.isfinite(d)]
    return (float(np.var(d) / 2.0), int(d.size)) if d.size >= 8 else (float("nan"), int(d.size))


def within_var(df, kw, min_frames=MIN_FRAMES, lon_max=LON_MAX):
    """Pooled within-HARP variance of the transformed, CLV-detrended true-HMI series
    (strong mask, |lon| <= lon_max, HARPs with >= min_frames frames): the evolution-signal pool."""
    sub = df[(df.model == "true_hmi") & (df["mask"] == "strong")].copy()
    sub = derive(sub)
    sub = sub[sub.center_lon.abs() <= lon_max]
    sub[kw] = transform(kw, sub[kw])
    sub = clv_detrend(sub, clv_fit(sub, [kw]))
    vs, ns = [], 0
    for _, g in sub.groupby("harpnum"):
        v = g[kw].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        if v.size >= min_frames:
            vs.append(np.var(v, ddof=1))
            ns += v.size
    return (float(np.mean(vs)), int(ns)) if vs else (float("nan"), 0)


def agreement(df, exps, kw, n_boot=0):
    """Spearman(pred on strong_pred, true on strong) across frames, seed-mean, on the transform."""
    pred = derive(seed_mean(df[df["mask"] == "strong_pred"], exps))
    true = derive(seed_mean(df[df["mask"] == "strong"], ("true_hmi",)))
    m = pred.merge(true, on=["harpnum", "mdi_t_rec"], suffixes=("_p", "_t"))
    x, y = transform(kw, m[f"{kw}_p"]), transform(kw, m[f"{kw}_t"])
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return float("nan"), int(ok.sum())
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def real_r12(rec, kw):
    """Real-side r12 of one (possibly derived) keyword straight from the Phase-10 records."""
    base = rec[rec.lag_min == 0]
    m = base.merge(rec[rec.lag_min == 12], on=["harpnum", "t_rec"], suffixes=("_a", "_b"))
    r, _ = corr_by(m, f"{kw}_a", f"{kw}_b", kw)
    return r


def cmd_weights(a):
    real = pd.read_csv(os.path.join(a.out_dir, "keyword_reliability_real.csv")).set_index("keyword")
    rec = derive(pd.read_csv(os.path.join(ROOT, "data", a.run, "reliability", "keyword_records.csv")))
    df = pd.read_csv(a.features)
    cands = CANDIDATES_INTENSIVE if a.candidates == "intensive" else CANDIDATES
    suffix = "_intensive" if a.candidates == "intensive" else ""
    rows = []
    for kw in cands:
        r12 = float(real.loc[kw, "r12"]) if kw in real.index else real_r12(rec, kw)
        agr, n = agreement(df, SOURCES["ours"], kw)
        nv, n_noise = noise_var(rec, kw)
        wv, n_within = within_var(df, kw)
        evo = wv / nv if np.isfinite(wv) and np.isfinite(nv) and nv > 0 else float("nan")
        inc = bool(np.isfinite(r12) and r12 >= R12_BAR and np.isfinite(agr) and agr >= AGREE_BAR)
        rows.append({"keyword": kw, "r12_real": r12, "agreement_ours": agr, "n_frames": n,
                     "noise_var": nv, "within_var": wv, "snr2_evo": evo,
                     "included": inc, "snr2_r12": snr2(r12),
                     "rho_eff": r12 * agr if np.isfinite(agr) else float("nan")})
    w = pd.DataFrame(rows)
    inc = w[w.included & np.isfinite(w.snr2_evo)]
    if len(inc):
        norm = float(inc.snr2_evo.mean())
        w["weight"] = np.where(w.included & np.isfinite(w.snr2_evo), w.snr2_evo / norm, 0.0)
    else:
        w["weight"] = 0.0
    w.to_csv(os.path.join(a.out_dir, f"keyword_reliability{suffix}.csv"), index=False)
    with open(os.path.join(a.out_dir, f"keyword_reliability{suffix}.json"), "w") as f:
        json.dump({"bars": {"r12": R12_BAR, "agreement": AGREE_BAR},
                   "weight_rule": "snr2_evo = within-HARP var / noise var (phase_12.md 4, amended), "
                                  "normalized to mean 1 over included",
                   "candidates": {r["keyword"]: {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                                 for k, v in r.items() if k != "keyword"}
                                  for r in w.to_dict("records")},
                   "preregistration": "reports/phase_12.md"}, f, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(w.to_string(index=False))
    print(f"wrote {a.out_dir}/keyword_reliability{suffix}.{{csv,json}}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    re_ = sub.add_parser("real")
    re_.add_argument("--run", default="full")
    re_.add_argument("--out_dir", default=os.path.join("results", "full", "overlap_pilot"))
    re_.add_argument("--strong_gauss", type=float, default=200.0)
    re_.add_argument("--n_boot", type=int, default=2000)
    re_.set_defaults(fn=cmd_real)
    rc = sub.add_parser("recon")
    rc.add_argument("--features", default=os.path.join("results", "full", "overlap_pilot", "features.csv"))
    rc.add_argument("--n_boot", type=int, default=500)
    rc.set_defaults(fn=cmd_recon)
    wt = sub.add_parser("weights")
    wt.add_argument("--run", default="full")
    wt.add_argument("--features", default=os.path.join("results", "full", "overlap_pilot", "features.csv"))
    wt.add_argument("--out_dir", default=os.path.join("results", "full", "overlap_pilot"))
    wt.add_argument("--candidates", choices=("standard", "intensive"), default="standard")
    wt.set_defaults(fn=cmd_weights)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
