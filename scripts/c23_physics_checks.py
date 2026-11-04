"""Phase 13 / F5: physics-law validation of the record (reports/phase_13.md 4, F5).

    python scripts/c23_physics_checks.py run            # C24 sides + butterfly (now)
    python scripts/c23_physics_checks.py run --with_c23 # after the C23 extraction exists

F5a decay law: per region with a >= 24 h decline after its USFLUX peak (>= 10 frames),
    decay rate = -slope of a least-squares line through USFLUX(t) from peak to end;
    log-log regression of rate on peak flux; exponent per record with bootstrap CI over regions.
F5b butterfly: |latitude| at first appearance vs time (metadata; validates the record's sampling).
F5c flux-flare: log-log slope of the >= C1 flare rate on peak USFLUX over flaring regions.
Figures (palette: slots 1-3 of the validated default; one axis; thin marks) ->
results/full/c23_build/figures/; numbers -> f5_physics.json.
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

OUT = os.path.join(ROOT, "results", "full", "c23_build")
FIG = os.path.join(OUT, "figures")
COLORS = {"c24_true": "#2a78d6", "c24_recon": "#eb6834", "c23_recon": "#1baf7a"}
LABELS = {"c24_true": "Cycle 24, HMI truth", "c24_recon": "Cycle 24, reconstructed",
          "c23_recon": "Cycle 23, reconstructed"}
TEXT, GRID = "#0b0b0b", "#d9d8d4"
MIN_DECLINE_H = 24.0
MIN_DECLINE_FRAMES = 10


def decay_points(frames):
    """Per region: (peak_flux, decay_rate Mx/h) from the post-peak decline of USFLUX(t);
    regions kept per the frozen rule (decline >= 24 h and >= 10 frames)."""
    rows = []
    for region, g in frames.groupby("region"):
        g = g.sort_values("_t")
        flux = g.USFLUX.to_numpy(np.float64)
        t_h = (g["_t"] - g["_t"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        ok = np.isfinite(flux) & (flux > 0)
        if ok.sum() < MIN_DECLINE_FRAMES:
            continue
        flux, t_h = flux[ok], t_h[ok]
        ip = int(np.argmax(flux))
        td, fd = t_h[ip:], flux[ip:]
        if len(td) < MIN_DECLINE_FRAMES or (td[-1] - td[0]) < MIN_DECLINE_H:
            continue
        slope = np.polyfit(td, fd, 1)[0]
        if slope >= 0:                                     # no net decline -> not a decay segment
            continue
        rows.append({"region": region, "peak_flux": float(flux[ip]), "decay_rate": float(-slope)})
    return pd.DataFrame(rows)


def loglog_slope(x, y, n_boot=2000, seed=0):
    """Slope of log10 y on log10 x with a bootstrap CI over rows; (slope, lo, hi, n)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log10(x[ok]), np.log10(y[ok])
    if lx.size < 8:
        return float("nan"), float("nan"), float("nan"), int(lx.size)
    slope = float(np.polyfit(lx, ly, 1)[0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, lx.size, lx.size)
        if np.ptp(lx[i]) > 0:
            bs.append(np.polyfit(lx[i], ly[i], 1)[0])
    return slope, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(lx.size)


def _style(ax, xlabel, ylabel, title, title_fs=11):
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.set_title(title, color=TEXT, loc="left", fontsize=title_fs)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT)


def scatter_fit_figure(datasets, xcol, ycol, xlabel, ylabel, title, path,
                       figsize=(6.4, 4.6), title_fs=11, legend_fs=9, marker_s=14, line_w=2):
    """Log-log scatter + fitted line per record; direct labels, no legend box duplication."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    for name, df in datasets.items():
        x, y = df[xcol].to_numpy(), df[ycol].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        ax.scatter(x[ok], y[ok], s=marker_s, alpha=0.55, lw=0, color=COLORS[name],
                   label=LABELS[name])
        s, lo, hi, n = loglog_slope(x, y, n_boot=200)
        if np.isfinite(s):
            lx = np.log10(x[ok])
            b = np.polyfit(lx, np.log10(y[ok]), 1)[1]
            xs = np.array([x[ok].min(), x[ok].max()])
            ax.plot(xs, 10 ** (s * np.log10(xs) + b), color=COLORS[name], lw=line_w)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, xlabel, ylabel, title, title_fs)
    ax.legend(frameon=False, fontsize=legend_fs, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def butterfly_figure(points, path, figsize=(7.6, 4.0), title_fs=11, marker_s=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.scatter(points["year"], points["lat"], s=marker_s, alpha=0.5, lw=0,
               color=COLORS["c24_true"])
    _style(ax, "year", "latitude [deg]", "Butterfly diagram of the record's regions (metadata)",
           title_fs)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- data

def c24_frames(source):
    from c23_benchmark import vec_frames_c24
    from c23_screening import SOURCES, derive, seed_mean
    if source == "c24_recon":
        return vec_frames_c24(os.path.join(ROOT, "results", "full", "overlap_pilot", "features.csv"))
    df = pd.read_csv(os.path.join(ROOT, "results", "full", "overlap_pilot", "features.csv"))
    sm = derive(seed_mean(df[df["mask"] == "strong"], SOURCES["true_hmi"]))
    sm["region"] = sm.harpnum
    from c23_benchmark import tvalue
    sm["_t"] = tvalue(sm.mdi_t_rec)
    meta = df[["harpnum", "mdi_t_rec", "noaa_ar"]].drop_duplicates()
    return sm.merge(meta, on=["harpnum", "mdi_t_rec"], how="left")


def c23_frames():
    from c23_benchmark import vec_frames_c23
    return vec_frames_c23(os.path.join(OUT, "features.csv"))


def butterfly_points():
    c24 = pd.read_csv(os.path.join(ROOT, "data", "full", "regions.csv")) \
        .sort_values("mdi_t_rec").groupby("harpnum").first().reset_index()
    c24["year"] = c24.mdi_t_rec.str[:4].astype(int) + (pd.to_datetime(
        c24.mdi_t_rec.str.replace("_TAI", "", regex=False), format="%Y.%m.%d_%H:%M:%S").dt.dayofyear / 365.0)
    c23 = pd.read_csv(os.path.join(ROOT, "data", "c23", "frames_all.csv")) \
        .sort_values("T_REC").groupby("TARPNUM").first().reset_index()
    c23["year"] = c23.T_REC.str[:4].astype(int) + (pd.to_datetime(
        c23.T_REC.str.replace("_TAI", "", regex=False), format="%Y.%m.%d_%H:%M:%S").dt.dayofyear / 365.0)
    return pd.DataFrame({"year": np.concatenate([c23.year, c24.year]),
                         "lat": np.concatenate([c23.LAT_FWT, c24.lat_fwt])})


def flare_rates(frames, flares):
    """Per region: peak USFLUX and >= C1 flare rate inside the region's observed window."""
    from c23_benchmark import M1
    rows = []
    for region, g in frames.groupby("region"):
        noaa = int(g.noaa_ar.iloc[0]) if "noaa_ar" in g else 0
        flux = g.USFLUX.to_numpy(np.float64)
        ok = np.isfinite(flux) & (flux > 0)
        if not ok.any() or noaa <= 0:
            continue
        days = len(g) * 96.0 / 1440.0
        f = flares[(flares.noaa_ar == noaa) & (flares.wm2 >= 1e-6) &
                   (flares.peak >= g["_t"].min()) & (flares.peak <= g["_t"].max())]
        rows.append({"region": region, "peak_flux": float(flux[ok].max()),
                     "rate_c": len(f) / days})
    return pd.DataFrame(rows)


def cmd_run(a):
    from c23_benchmark import load_flares
    os.makedirs(FIG, exist_ok=True)
    sources = {"c24_true": c24_frames("c24_true"), "c24_recon": c24_frames("c24_recon")}
    if a.with_c23:
        sources["c23_recon"] = c23_frames()
    out = {"decay": {}, "flux_flare": {}}
    decays = {k: decay_points(v) for k, v in sources.items()}
    for k, d in decays.items():
        s, lo, hi, n = loglog_slope(d.peak_flux, d.decay_rate)
        out["decay"][k] = {"exponent": s, "ci": [lo, hi], "n_regions": n}
    scatter_fit_figure(decays, "peak_flux", "decay_rate", "peak unsigned flux [Mx]",
                       "flux decay rate [Mx/h]", "F5a - decay law, rate vs peak flux",
                       os.path.join(FIG, "f5a_decay_law.png"))
    flares = load_flares()
    rates = {k: flare_rates(v, flares) for k, v in sources.items()}
    rates_pos = {k: v[v.rate_c > 0] for k, v in rates.items()}
    for k, d in rates_pos.items():
        s, lo, hi, n = loglog_slope(d.peak_flux, d.rate_c)
        out["flux_flare"][k] = {"slope": s, "ci": [lo, hi], "n_flaring_regions": n}
    scatter_fit_figure(rates_pos, "peak_flux", "rate_c", "peak unsigned flux [Mx]",
                       ">= C1 flares per observed day", "F5c - flux-flare scaling (flaring regions)",
                       os.path.join(FIG, "f5c_flux_flare.png"))
    butterfly_figure(butterfly_points(), os.path.join(FIG, "f5b_butterfly.png"))
    with open(os.path.join(OUT, "f5_physics.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(out, indent=1, default=str))
    print(f"figures in {FIG}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run")
    rn.add_argument("--with_c23", action="store_true")
    rn.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
