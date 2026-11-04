"""Supplementary validation checks for the Cycle-23 record: every check that needs a new
measurement, computed from the frozen artifacts alone (no new training, no new downloads).

    python scripts/c23_extra_checks.py heldout     # screening agreement off the model's training regions
    python scripts/c23_extra_checks.py fluxrank    # what the current columns add over a flux ranking
    python scripts/c23_extra_checks.py helicity    # hemispheric sign rule on the released signed columns
    python scripts/c23_extra_checks.py epochs      # MDI epoch robustness: monthly LOS ratio + era lag-96 r
    python scripts/c23_extra_checks.py f4          # pre-registered F4: Halloween 2003 rank positions
    python scripts/c23_extra_checks.py samplecheck # sample-vs-population representativeness
    python scripts/c23_extra_checks.py all         # the six analyses above
    python scripts/c23_extra_checks.py figures     # vector-PDF f5 figures + epoch figure for the paper
    python scripts/c23_extra_checks.py example     # SMARP input vs reconstructed Bp/Bt/Br example

JSON outputs -> results/full/extra_checks/; figures -> figures/.
The U-Net's training regions are splits/full.json train HARPs; "held-out" = val + test.
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
import c23_screening as ikr  # noqa: E402

OUT = os.path.join(ROOT, "results", "full", "extra_checks")
FIG = os.path.join(ROOT, "figures")
PILOT_FEATURES = os.path.join(ROOT, "results", "full", "overlap_pilot", "features.csv")
C23_FEATURES = os.path.join(ROOT, "results", "full", "c23_build", "features.csv")
SPLIT = os.path.join(ROOT, "splits", "full.json")
FRAMES_ALL = os.path.join(ROOT, "data", "c23", "frames_all.csv")
FRAMES_SAMPLE = os.path.join(ROOT, "data", "c23", "frames.csv")
KWS_STANDARD = ikr.CANDIDATES
KWS_INTENSIVE = tuple(k for k in ikr.CANDIDATES_INTENSIVE if k not in ikr.CANDIDATES)
HALLOWEEN = ("2003.10.19", "2003.11.06")       # frozen F4 window (phase_13.md 4): Oct/Nov 2003 storms


def spearman(x, y):
    """Spearman rho over finite pairs, average ranks (same recipe as ikr.agreement); (rho, n)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return float("nan"), int(ok.sum())
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def partial_spearman(x, y, z):
    """Rank-based partial correlation of x and y given z (Pearson on average ranks)."""
    x, y, z = (np.asarray(v, np.float64) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    r = [pd.Series(v[ok]).rank().to_numpy() for v in (x, y, z)]
    c = np.corrcoef(np.stack(r))
    rxy, rxz, ryz = c[0, 1], c[0, 2], c[1, 2]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def wilson_ci(k, n, z=1.959964):
    """Wilson 95 % interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return float(c - h), float(c + h)


def binom_p(k, n):
    """Two-sided exact binomial p against 0.5."""
    from scipy.stats import binomtest
    return float(binomtest(k, n, 0.5).pvalue)


def load_split():
    with open(SPLIT) as f:
        s = json.load(f)["splits"]
    return {k: list(map(int, s[k]["harps"])) for k in ("train", "val", "test")}


def merged_pred_true(df, exps):
    """Frame-level merge of the seed-mean prediction (strong_pred) with HMI truth (strong):
    the frame ikr.agreement ranks; columns <kw>_p / <kw>_t plus harpnum."""
    pred = ikr.derive(ikr.seed_mean(df[df["mask"] == "strong_pred"], exps))
    true = ikr.derive(ikr.seed_mean(df[df["mask"] == "strong"], ("true_hmi",)))
    return pred.merge(true, on=["harpnum", "mdi_t_rec"], suffixes=("_p", "_t"))


def agreement_on(df, exps, kw, harps=None):
    """ikr.agreement restricted to a HARP subset; (rho, n_frames)."""
    if harps is not None:
        df = df[df.harpnum.isin(set(harps))]
    return ikr.agreement(df, exps, kw)


def boot_merged(m, fn, n_boot=1000, seed=0):
    """95 % bootstrap interval over HARPs of a statistic fn(merged_frame) -> float."""
    rng = np.random.default_rng(seed)
    by = {h: g for h, g in m.groupby("harpnum")}
    harps = np.array(list(by))
    vals = []
    for _ in range(n_boot):
        take = rng.choice(harps, size=len(harps), replace=True)
        v = fn(pd.concat([by[h] for h in take]))
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot // 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _agree_stat(kw):
    def fn(m):
        return spearman(ikr.transform(kw, m[f"{kw}_p"]), ikr.transform(kw, m[f"{kw}_t"]))[0]
    return fn


def cmd_heldout(a):
    df = pd.read_csv(a.features)
    split = load_split()
    heldout = split["val"] + split["test"]
    m_all = merged_pred_true(df, ikr.SOURCES["ours"])
    out = {"split": {k: len(v) for k, v in split.items()},
           "n_heldout_harps": len(heldout),
           "flux_domain_true_mx": [float(np.nanmin(m_all["USFLUX_t"])),
                                   float(np.nanmax(m_all["USFLUX_t"]))],
           "keywords": {}}
    for kw in KWS_STANDARD + KWS_INTENSIVE:
        row = {}
        for name, harps in (("all", None), ("train", split["train"]), ("heldout", heldout)):
            r, n = agreement_on(df, ikr.SOURCES["ours"], kw, harps)
            row[name] = {"agreement": r, "n_frames": n}
        mh = m_all[m_all.harpnum.isin(set(heldout))]
        row["heldout"]["ci"] = list(boot_merged(mh, _agree_stat(kw), a.n_boot))
        row["delta_all_minus_heldout"] = row["all"]["agreement"] - row["heldout"]["agreement"]
        out["keywords"][kw] = row
        print(f"{kw}: all {row['all']['agreement']:.3f}  train {row['train']['agreement']:.3f}  "
              f"heldout {row['heldout']['agreement']:.3f} {row['heldout']['ci']}", flush=True)
    _write(a.out, "heldout_agreement.json", out)


def cmd_fluxrank(a):
    df = pd.read_csv(a.features)
    split = load_split()
    heldout = set(split["val"] + split["test"])
    m = merged_pred_true(df, ikr.SOURCES["ours"])
    e0c = ikr.derive(ikr.seed_mean(df[df["mask"] == "strong_pred"], ikr.SOURCES["e0c"]))
    m = m.merge(e0c[["harpnum", "mdi_t_rec", "USFLUX"]].rename(columns={"USFLUX": "USFLUX_e0c"}),
                on=["harpnum", "mdi_t_rec"], how="left")
    out = {}
    for scope, mm in (("all", m), ("heldout", m[m.harpnum.isin(heldout)])):
        res = {}
        for kw in ("TOTUSJZ", "TOTUSJH"):
            t = {c: ikr.transform(kw if "_e0c" not in c and "USFLUX" not in c else "USFLUX", mm[c])
                 for c in (f"{kw}_p", f"{kw}_t", "USFLUX_p", "USFLUX_t", "USFLUX_e0c")}
            rho_ours, n = spearman(t[f"{kw}_p"], t[f"{kw}_t"])
            rho_flux_pred, _ = spearman(t["USFLUX_p"], t[f"{kw}_t"])
            rho_flux_los, _ = spearman(t["USFLUX_e0c"], t[f"{kw}_t"])
            rho_flux_true, _ = spearman(t["USFLUX_t"], t[f"{kw}_t"])
            part = partial_spearman(t[f"{kw}_p"], t[f"{kw}_t"], t["USFLUX_t"])

            def delta(g, kw=kw):
                return (spearman(ikr.transform(kw, g[f"{kw}_p"]), ikr.transform(kw, g[f"{kw}_t"]))[0]
                        - spearman(ikr.transform("USFLUX", g["USFLUX_p"]),
                                   ikr.transform(kw, g[f"{kw}_t"]))[0])
            lo, hi = boot_merged(mm, delta, a.n_boot)
            res[kw] = {"n_frames": n, "rho_ours": rho_ours, "rho_flux_proxy_pred": rho_flux_pred,
                       "rho_flux_proxy_e0c": rho_flux_los, "rho_flux_ceiling_true": rho_flux_true,
                       "partial_given_true_flux": part,
                       "delta_ours_minus_flux_proxy": rho_ours - rho_flux_pred,
                       "delta_ci": [lo, hi]}
            print(f"[{scope}] {kw}: ours {rho_ours:.3f} vs flux-proxy {rho_flux_pred:.3f} "
                  f"(LOS-e0c {rho_flux_los:.3f}, true-flux ceiling {rho_flux_true:.3f}); "
                  f"partial {part:.3f}; delta [{lo:.3f},{hi:.3f}]", flush=True)
        out[scope] = res
    _write(a.out, "fluxrank.json", out)


def _c23_record(usecols=None):
    cols = usecols or ["tarpnum", "t_rec", "noaa_ar", "lat_fwt", "lon_fwt", "usfluxl", "model",
                       "mask", "USFLUX", "TOTUSJZ", "TOTUSJH", "R_VALUE", "MEANJZH", "MEANALP",
                       "n_pixels", "n_finite"]
    df = pd.read_csv(C23_FEATURES, usecols=cols)
    return df[(df.model == "seed_mean") & (df["mask"] == "strong_pred")].reset_index(drop=True)


def helicity_compliance(reg, col):
    """Hemispheric sign rule over per-region medians: north -> negative, south -> positive.
    reg needs columns lat and <col>; zero-latitude or zero-signed rows are dropped."""
    v = reg[col].to_numpy(np.float64)
    lat = reg["lat"].to_numpy(np.float64)
    ok = np.isfinite(v) & np.isfinite(lat) & (v != 0) & (lat != 0)
    v, lat = v[ok], lat[ok]
    obey = ((lat > 0) & (v < 0)) | ((lat < 0) & (v > 0))
    n, k = int(ok.sum()), int(obey.sum())
    return {"n": n, "k": k, "frac": k / n if n else float("nan"),
            "ci": list(wilson_ci(k, n)), "p_vs_chance": binom_p(k, n) if n else float("nan")}


def cmd_helicity(a):
    df = _c23_record()
    reg = df.groupby("tarpnum").agg(lat=("lat_fwt", "median"), mjzh=("MEANJZH", "median"),
                                    malp=("MEANALP", "median"), n_frames=("t_rec", "size"),
                                    noaa=("noaa_ar", "first")).reset_index()
    reg = reg[reg.n_frames >= a.min_frames]
    out = {"min_frames": a.min_frames, "n_regions": int(len(reg)),
           "MEANJZH": helicity_compliance(reg, "mjzh"),
           "MEANALP": helicity_compliance(reg, "malp"),
           "MEANJZH_lat5": helicity_compliance(reg[reg.lat.abs() >= 5.0], "mjzh"),
           "MEANALP_lat5": helicity_compliance(reg[reg.lat.abs() >= 5.0], "malp")}
    print(json.dumps(out, indent=1))
    _write(a.out, "helicity_rule.json", out)


def monthly_medians(df):
    """Per calendar month: frame count, median USFLUX/USFLUXL (both > 0), median TOTUSJZ/USFLUX."""
    d = df.copy()
    d["month"] = d.t_rec.str[:7]
    ok = (d.USFLUX > 0) & (d.usfluxl > 0)
    d["ratio"] = np.where(ok, d.USFLUX / d.usfluxl.where(d.usfluxl > 0), np.nan)
    jz_ok = (d.USFLUX > 0) & (d.TOTUSJZ > 0)
    d["jz_per_flux"] = np.where(jz_ok, d.TOTUSJZ / d.USFLUX.where(d.USFLUX > 0), np.nan)
    g = d.groupby("month").agg(n_frames=("t_rec", "size"), ratio_med=("ratio", "median"),
                               jz_per_flux_med=("jz_per_flux", "median")).reset_index()
    return g


def era_lag_r(df, kws, boundary="2004-01-01", lag=96):
    """Lag-96-min correlation per keyword split at a calendar boundary (pre/post)."""
    d = df.rename(columns={"tarpnum": "harpnum", "t_rec": "mdi_t_rec"}).copy()
    if "harpnum" not in d.columns:
        raise ValueError("need tarpnum/harpnum")
    m = ikr.lag_pairs(d, lag)
    if m.empty:
        return []
    t = pd.to_datetime(m.mdi_t_rec_a.str.replace("_TAI", "", regex=False),
                       format="%Y.%m.%d_%H:%M:%S")
    era = np.where(t < pd.Timestamp(boundary), "pre", "post")
    out = []
    for e in ("pre", "post"):
        sub = m[era == e]
        for kw in kws:
            r, n = ikr.corr_by(sub, f"{kw}_a", f"{kw}_b", kw)
            out.append({"keyword": kw, "era": e, "r": r, "n_pairs": n})
    return out


def cmd_epochs(a):
    df = ikr.derive(_c23_record())
    mm = monthly_medians(df)
    kws = ("USFLUX", "TOTUSJZ", "TOTUSJH", "R_VALUE", "STRONG_FRAC")
    eras = era_lag_r(df, kws, boundary=a.boundary)
    ratio = df.USFLUX / df.usfluxl.where(df.usfluxl > 0)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    yearly = mm.assign(year=mm.month.str[:4]).groupby("year") \
               .apply(lambda g: float(np.nanmedian(g.ratio_med)), include_groups=False)
    out = {"boundary": a.boundary,
           "ratio_usflux_over_usfluxl": {"median": float(np.median(ratio)),
                                         "p10": float(np.percentile(ratio, 10)),
                                         "p90": float(np.percentile(ratio, 90))},
           "yearly_ratio_median": {k: v for k, v in yearly.items()},
           "yearly_ratio_minmax": [float(yearly.min()), float(yearly.max())],
           "era_lag96": eras}
    mm.to_csv(os.path.join(_ensure(a.out), "epoch_monthly.csv"), index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "yearly_ratio_median"}, indent=1))
    _write(a.out, "epochs.json", out)


def percentile_rank(values, v):
    """Percentage of values <= v (percentile rank of v within the population)."""
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    return float(100.0 * np.mean(values <= v))


def cmd_f4(a):
    df = _c23_record()
    peaks = df[df.USFLUX > 0].sort_values("USFLUX").groupby("tarpnum").last().reset_index()
    peaks = peaks.rename(columns={"USFLUX": "peak_usflux", "t_rec": "peak_t_rec"})
    hw = df[(df.t_rec >= HALLOWEEN[0]) & (df.t_rec < HALLOWEEN[1])]
    hw_tarps = sorted(hw.tarpnum.unique())
    pilot = pd.read_csv(PILOT_FEATURES)
    true_usflux = pilot[(pilot.model == "true_hmi") & (pilot["mask"] == "strong")].USFLUX
    support_max = float(np.nanmax(true_usflux))
    rows = []
    for tarp in hw_tarps:
        p = peaks[peaks.tarpnum == tarp]
        if p.empty:
            continue
        p = p.iloc[0]
        frames = hw[hw.tarpnum == tarp]
        rows.append({"tarpnum": int(tarp), "noaa_ar": int(p.noaa_ar),
                     "peak_usflux_mx": float(p.peak_usflux),
                     "percentile_in_c23": percentile_rank(peaks.peak_usflux, p.peak_usflux),
                     "n_frames_in_window": int(len(frames)),
                     "share_frames_above_support": float(np.mean(frames.USFLUX > support_max))})
    rows.sort(key=lambda r: -r["peak_usflux_mx"])
    out = {"window": list(HALLOWEEN), "n_halloween_tarps": len(rows),
           "c24_training_support_max_usflux_mx": support_max,
           "share_all_c23_frames_above_support": float(np.mean(df.USFLUX > support_max)),
           "share_all_c23_regions_peak_above_support":
               float(np.mean(peaks.peak_usflux > support_max)),
           "regions": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "regions"}, indent=1))
    print(pd.DataFrame(rows).head(12).to_string(index=False))
    _write(a.out, "f4_halloween.json", out)


def cmd_samplecheck(a):
    from scipy.stats import ks_2samp
    pop = pd.read_csv(FRAMES_ALL)
    sam = pd.read_csv(FRAMES_SAMPLE)
    pop_peak = pop[pop.USFLUXL > 0].groupby("TARPNUM").USFLUXL.max()
    sam_peak = sam[sam.USFLUXL > 0].groupby("TARPNUM").USFLUXL.max()
    ks = ks_2samp(np.log10(pop_peak), np.log10(sam_peak))
    pop_y = pop.groupby(pop.T_REC.str[:4]).TARPNUM.nunique()
    sam_y = sam.groupby(sam.T_REC.str[:4]).TARPNUM.nunique()
    share = (sam_y / pop_y).round(3)
    out = {"ks_stat_log_peak_usfluxl": float(ks.statistic), "ks_p": float(ks.pvalue),
           "n_pop_regions": int(pop.TARPNUM.nunique()), "n_sample_regions": int(sam.TARPNUM.nunique()),
           "sampled_region_share_by_year": {k: float(v) for k, v in share.items()}}
    print(json.dumps(out, indent=1))
    _write(a.out, "samplecheck.json", out)


# ---------------------------------------------------------------- paper figures

def cmd_figures(a):
    """Vector-PDF copies of the physics figures + the epoch-robustness figure (paper_dataset),
    exported at their main.tex placement widths (textwidth 345 pt) so fonts render 1:1,
    with TrueType (type 42) fonts embedded for the publisher."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import c23_physics_checks as ipc
    os.makedirs(FIG, exist_ok=True)
    sources = {"c24_true": ipc.c24_frames("c24_true"), "c24_recon": ipc.c24_frames("c24_recon"),
               "c23_recon": ipc.c23_frames()}
    small = {"pdf.fonttype": 42, "font.size": 7, "axes.labelsize": 7,
             "xtick.labelsize": 6.5, "ytick.labelsize": 6.5}
    decays = {k: ipc.decay_points(v) for k, v in sources.items()}
    with plt.rc_context(small):                     # placed at 0.49\textwidth = 2.34 in
        ipc.scatter_fit_figure(decays, "peak_flux", "decay_rate", "peak unsigned flux [Mx]",
                               "flux decay rate [Mx/h]", "Decay law: rate vs peak flux",
                               os.path.join(FIG, "f5a_decay_law.pdf"),
                               figsize=(2.4, 2.1), title_fs=7.5, legend_fs=5.8, marker_s=5,
                               line_w=1.2)
    from c23_benchmark import load_flares
    flares = load_flares()
    rates = {k: ipc.flare_rates(v, flares) for k, v in sources.items()}
    rates_pos = {k: v[v.rate_c > 0] for k, v in rates.items()}
    with plt.rc_context(small):
        ipc.scatter_fit_figure(rates_pos, "peak_flux", "rate_c", "peak unsigned flux [Mx]",
                               ">= C1 flares per observed day",
                               "Flux-flare scaling (flaring regions)",
                               os.path.join(FIG, "f5c_flux_flare.pdf"),
                               figsize=(2.4, 2.1), title_fs=7.5, legend_fs=5.8, marker_s=5,
                               line_w=1.2)
    mid = {"pdf.fonttype": 42, "font.size": 7.5, "axes.labelsize": 7.5,
           "xtick.labelsize": 7, "ytick.labelsize": 7}
    with plt.rc_context(mid):                       # placed at 0.8\textwidth = 3.82 in
        ipc.butterfly_figure(ipc.butterfly_points(), os.path.join(FIG, "f5b_butterfly.pdf"),
                             figsize=(3.85, 2.0), title_fs=8, marker_s=3)
    _epoch_figure(a)
    print(f"figures in {FIG}")


def _epoch_figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mm = pd.read_csv(os.path.join(OUT, "epoch_monthly.csv"), dtype={"month": str})
    with open(os.path.join(OUT, "epochs.json")) as f:
        ep = json.load(f)
    t = pd.PeriodIndex(mm.month.str.replace(".", "-", regex=False), freq="M").to_timestamp()
    rc = {"pdf.fonttype": 42, "font.size": 7.5, "axes.labelsize": 7.5,
          "xtick.labelsize": 7, "ytick.labelsize": 7}
    with plt.rc_context(rc):                        # placed at 0.85\textwidth = 4.06 in
        fig, axes = plt.subplots(2, 1, figsize=(4.05, 3.3), sharex=False)
        ax = axes[0]
        solid = mm.ratio_med.where(mm.n_frames >= 50)
        ax.plot(t, solid, color="#2a78d6", lw=1.0)
        sparse = mm[mm.n_frames < 50]
        ax.scatter(t[mm.n_frames < 50], sparse.ratio_med, s=5, color="#2a78d6", alpha=0.35, lw=0)
        ax.axhline(float(np.nanmedian(mm.ratio_med)), color="#9b9890", lw=0.7, ls="--")
        ipc_style(ax, "year", "median USFLUX / USFLUXL",
                  "Monthly reconstructed-to-LOS flux ratio (dots: < 50 frames)", 8)
        ax = axes[1]
        eras = pd.DataFrame(ep["era_lag96"])
        kws = eras.keyword.unique()
        xs = np.arange(len(kws))
        for off, era, color in ((-0.16, "pre", "#2a78d6"), (0.16, "post", "#eb6834")):
            sub = eras[eras.era == era].set_index("keyword").reindex(kws)
            ax.bar(xs + off, sub.r, width=0.3, color=color, label=f"{era} {ep['boundary'][:4]}")
        ax.set_xticks(xs, kws, rotation=0, fontsize=6.2)
        ax.set_ylim(0.8, 1.0)
        ipc_style(ax, "", "lag-96-min correlation", "Temporal reliability by MDI era", 8)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "epoch_robustness.pdf"))
        plt.close(fig)


def ipc_style(ax, xlabel, ylabel, title, title_fs=11):
    import c23_physics_checks as ipc
    ipc._style(ax, xlabel, ylabel, title, title_fs)


def cmd_example(a):
    """SMARP LOS input next to the reconstructed (Bp, Bt, Br) for one sampled frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from sr.evaluate import infer_region, load_model
    from c23_build import NORM, fits_name, load_smarp_frame, model_input
    frames = pd.read_csv(FRAMES_SAMPLE)
    raw = os.path.join(ROOT, "data", "c23", "raw")
    sel = frames[frames.NOAA_AR == a.noaa] if a.noaa else frames
    sel = sel[sel.USFLUXL > 0].copy()
    sel["dist"] = sel.LON_FWT.abs() + sel.LAT_FWT.abs() * 0.1
    sel = sel.sort_values(["dist"])
    path, row = None, None
    for r in sel.itertuples():
        p = os.path.join(raw, fits_name(r.TARPNUM, r.T_REC))
        if os.path.exists(p):
            path, row = p, r
            break
    if path is None:
        raise SystemExit(f"no downloaded frame for NOAA {a.noaa}")
    lr, geom, _ = load_smarp_frame(path)
    x = torch.as_tensor(model_input(lr, geom))[None]
    res_dir = os.path.join(ROOT, "results", "full")
    preds = []
    for e in ("F_U_E3_s0_e200", "F_U_E3_s1_e200", "F_U_E3_s2"):
        m, _ = load_model(res_dir, e, "cpu")
        with torch.no_grad():
            preds.append(infer_region(m, x)[0].cpu().numpy().astype(np.float64) * NORM)
    pred = np.mean(preds, 0)
    los = lr * NORM
    v = min(1000.0, float(np.nanpercentile(np.abs(los), 99.5)))
    ny, nx = pred[2].shape
    plt.rcParams["pdf.fonttype"] = 42
    # exported at the main.tex placement width (\textwidth = 4.77 in) so fonts render 1:1
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 1.0 * 2 * max(1.0, 5.0 * ny / nx)))
    panels = [("SMARP LOS input (0.12$^\\circ$ px)", los),
              ("reconstructed $B_r$ (0.03$^\\circ$ px)", pred[2]),
              ("reconstructed $B_p$", pred[0]), ("reconstructed $B_t$", pred[1])]
    for ax, (title, img) in zip(axes.ravel(), panels):
        im = ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower",
                       interpolation="antialiased", rasterized=True)
        ax.set_title(title, fontsize=7, color="#0b0b0b", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("field strength [G]", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    date = row.T_REC.split("_")[0].replace(".", "-")
    fig.suptitle(f"NOAA {int(row.NOAA_AR)} (TARP {int(row.TARPNUM)}), {date}", fontsize=8,
                 x=0.12, ha="left")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "example_reconstruction.pdf"), bbox_inches="tight", dpi=400)
    plt.close(fig)
    meta = {"tarpnum": int(row.TARPNUM), "noaa_ar": int(row.NOAA_AR), "t_rec": row.T_REC,
            "lon_fwt": float(row.LON_FWT), "lat_fwt": float(row.LAT_FWT), "clip_gauss": v}
    _write(OUT, "example_frame.json", meta)
    print(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- plumbing

def _ensure(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write(out_dir, name, obj):
    path = os.path.join(_ensure(out_dir), name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {path}")


def cmd_all(a):
    for fn in (cmd_heldout, cmd_fluxrank, cmd_helicity, cmd_epochs, cmd_f4, cmd_samplecheck):
        print(f"== {fn.__name__} ==", flush=True)
        fn(a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=PILOT_FEATURES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--min_frames", type=int, default=10)
    ap.add_argument("--boundary", default="2004-01-01")
    ap.add_argument("--noaa", type=int, default=9077)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("heldout", cmd_heldout), ("fluxrank", cmd_fluxrank),
                     ("helicity", cmd_helicity), ("epochs", cmd_epochs), ("f4", cmd_f4),
                     ("samplecheck", cmd_samplecheck), ("all", cmd_all),
                     ("figures", cmd_figures), ("example", cmd_example)):
        sub.add_parser(name).set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
