"""Phase 10: test–retest reliability of div_h and J_z over the full year (reports/phase_10.md).

    python scripts/reliability_pairs.py sample   --run full --per_month 20 --seed 0
    python scripts/reliability_pairs.py download --run full --email you@example.com --part 0/4
    python scripts/reliability_pairs.py analyze  --run full --n_boot 2000

sample:   stratified frame list from the TRAINING HARPs (per_month frames per month, distinct HARPs first)
          -> data/<run>/reliability/frames.csv
download: for the HARPs of one part (round-robin split of the sorted HARP list over n parts, one part per
          JSOC email), export the SHARP records at t, t + 12 min and t + 24 min (Bp, Bt, Br) with the
          pipeline's one-export-per-HARP routine -> data/<run>/reliability/raw/
analyze:  per frame and lag the pooled-moment sums of the smoothed divergence and vertical current at
          sigma = 0, 1, 2, 4, 8 px on all pixels and above 200 / 500 / 1000 G (same code as the paper's
          Table 5), pooled over frames, with 95 % bootstrap intervals over HARPs, by field strength,
          center-to-limb angle and month; the 24-min lag brackets the shared-noise correction
          -> data/<run>/reliability/moments.csv, reliability_by_scale.{csv,json}
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import div_snr_by_scale as dsn  # noqa: E402
from sr import physics as ph  # noqa: E402

SERIES = "hmi.sharp_cea_720s"
SEGMENTS = ("Bp", "Bt", "Br")
LAGS = (12, 24)
QUANTITIES = ("div", "jz")
ANGLE_BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0))
MOMENT_KEYS = ("n", "sx", "sy", "sxx", "syy", "sxy", "sabs", "sdd")


# ---------------------------------------------------------------- sample

def stratified_sample(regs, per_month=20, seed=0):
    """per_month frames per calendar month, one frame per HARP first (random frame), then repeats."""
    rng = np.random.default_rng(seed)
    regs = regs.copy()
    regs["month"] = regs.sharp_t_rec.str[:7]
    out = []
    for _, g in regs.groupby("month", sort=True):
        g = g.sample(frac=1.0, random_state=int(rng.integers(2 ** 31 - 1)))
        take = g.drop_duplicates("harpnum").head(per_month)
        if len(take) < per_month:
            take = pd.concat([take, g.drop(take.index).head(per_month - len(take))])
        out.append(take)
    return pd.concat(out).reset_index(drop=True)


def _parse(t_rec):
    s = t_rec.replace("_TAI", "")
    for fmt in ("%Y.%m.%d_%H:%M:%S", "%Y.%m.%d_%H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(t_rec)


def lag_t_recs(t_rec, lags_min=LAGS):
    t = _parse(t_rec)
    return [(t + timedelta(minutes=m)).strftime("%Y.%m.%d_%H:%M:%S_TAI") for m in lags_min]


def part_of(harps, k, n_parts):
    return sorted(set(int(h) for h in harps))[k::n_parts]


def fits_name(harpnum, t_rec, segment):
    return f"{SERIES}.{int(harpnum)}.{_parse(t_rec):%Y%m%d_%H%M%S}_TAI.{segment}.fits"


# ---------------------------------------------------------------- statistics

def shared_noise_upper(r12, r24):
    """Upper bound on the shared-noise correlation of consecutive records: r12 = rho + c (1 - rho) with
    rho >= r24 gives c <= (r12 - r24) / (1 - r24); it also absorbs the extra 12 min of evolution."""
    return float("nan") if r24 >= 1.0 else float((r12 - r24) / (1.0 - r24))


def _sum_moments(ms):
    return {k: float(sum(m[k] for m in ms)) for k in MOMENT_KEYS}


def bootstrap_corr(per_harp, n_boot=2000, seed=0):
    """95 % percentile interval of the pooled correlation when HARPs are resampled with replacement
    (per_harp: harpnum -> moments summed over that HARP's frames)."""
    keys = list(per_harp)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(len(keys), size=len(keys))
        vals.append(dsn.pooled([per_harp[keys[i]] for i in pick])["corr"])
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


# ---------------------------------------------------------------- commands

def cmd_sample(a):
    from sr.datasets import split_harps
    run_dir = os.path.join(ROOT, "data", a.run)
    out_dir = os.path.join(run_dir, "reliability")
    os.makedirs(out_dir, exist_ok=True)
    regs = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    train = regs[regs.harpnum.isin(split_harps(os.path.join(ROOT, "splits", f"{a.run}.json"), "train"))]
    s = stratified_sample(train, a.per_month, a.seed)
    cols = [c for c in ("harpnum", "sharp_t_rec", "month", "center_angle", "noaa_ar", "region_file", "day") if c in s.columns]
    s = s[cols]
    s.to_csv(os.path.join(out_dir, "frames.csv"), index=False)
    print(f"{len(s)} frames from {s.harpnum.nunique()} HARPs over {s.month.nunique()} months -> {out_dir}/frames.csv")
    print(s.groupby("month").agg(frames=("harpnum", "size"), harps=("harpnum", "nunique")).to_string())


def download_with_retry(export, by_harp, max_retries=6, wait_s=600, sleep=time.sleep, log=print):
    """Call export(harp, pairs) for every HARP in turn. On JSOC's "pending export" refusal (one outstanding
    export per email) wait `wait_s` seconds and retry the same HARP, up to max_retries times; any other
    error is recorded and the next HARP is tried. Returns the list of (harp, error) failures."""
    failures = []
    for harp, pairs in sorted(by_harp.items()):
        for attempt in range(max_retries + 1):
            try:
                export(harp, pairs)
                break
            except Exception as e:   # noqa: BLE001 - every JSOC or network error must be survivable here
                if "pending export" in str(e).lower() and attempt < max_retries:
                    log(f"HARP {harp}: JSOC still has a pending export for this email; waiting {wait_s} s "
                        f"(attempt {attempt + 1}/{max_retries})")
                    sleep(wait_s)
                else:
                    log(f"HARP {harp}: giving up: {e}")
                    failures.append((harp, str(e)))
                    break
    return failures


def cmd_download(a):
    import drms
    import data_active_region_created as pipe
    from pipeline.build_dataset import JsocLock
    run_dir = os.path.join(ROOT, "data", a.run)
    rel = os.path.join(run_dir, "reliability")
    raw = os.path.join(rel, "raw")
    os.makedirs(raw, exist_ok=True)
    frames = pd.read_csv(os.path.join(rel, "frames.csv"))
    k, n = (int(x) for x in a.part.split("/"))
    harps = part_of(frames.harpnum, k, n)
    by_harp = {}
    for r in frames[frames.harpnum.isin(harps)].itertuples():
        for t in [r.sharp_t_rec] + lag_t_recs(r.sharp_t_rec):
            if not all(os.path.exists(os.path.join(raw, fits_name(r.harpnum, t, s))) for s in SEGMENTS):
                by_harp.setdefault(int(r.harpnum), []).append((int(r.harpnum), t))
    n_pairs = sum(len(v) for v in by_harp.values())
    print(f"part {k}/{n}: {len(harps)} HARPs, {n_pairs} records still to fetch ({a.email})", flush=True)
    if not by_harp:
        return
    client = drms.Client(email=a.email)

    def export(harp, pairs):
        pipe.batched_export_sharp(client, SERIES, pairs, raw, segments=SEGMENTS, label=f"REL part {k} HARP {harp}")

    with JsocLock(os.path.join(ROOT, "data"), email=a.email):
        failures = download_with_retry(export, by_harp, max_retries=a.max_retries, wait_s=a.wait_s,
                                       log=lambda m: print(f"  [REL part {k}] {m}", flush=True))
    missing = [p for ps in by_harp.values() for p in ps
               if not all(os.path.exists(os.path.join(raw, fits_name(p[0], p[1], s))) for s in SEGMENTS)]
    print(f"part {k}/{n} done; {len(missing)} records still missing; {len(failures)} HARP export(s) failed", flush=True)
    with open(os.path.join(rel, f"download_part{k}of{n}.json"), "w") as f:
        json.dump({"email": a.email, "harps": harps, "requested": n_pairs, "missing": missing, "failures": failures},
                  f, indent=1, default=str)


def frame_moments(run_dir, frames):
    """Per frame and lag: moments for every quantity, scale and field threshold. Returns a DataFrame."""
    import noise_floor_temporal as nft
    raw = os.path.join(run_dir, "reliability", "raw")
    rows, skipped = [], []
    for i, r in enumerate(frames.itertuples()):
        paths = {}
        for t in [r.sharp_t_rec] + lag_t_recs(r.sharp_t_rec):
            p = {s: os.path.join(raw, fits_name(r.harpnum, t, s)) for s in SEGMENTS}
            if all(os.path.exists(x) for x in p.values()):
                paths[t] = p
        if r.sharp_t_rec not in paths:
            skipped.append((int(r.harpnum), r.sharp_t_rec, "missing t"))
            continue
        ma, sa = nft.load_stack(paths[r.sharp_t_rec])
        for lag, t in zip(LAGS, lag_t_recs(r.sharp_t_rec)):
            if t not in paths:
                skipped.append((int(r.harpnum), r.sharp_t_rec, f"missing t+{lag}"))
                continue
            mb, sb = nft.load_stack(paths[t])
            x, y, off = nft.common_grid(ma["Br"], sa, mb["Br"], sb)
            if x is None:
                skipped.append((int(r.harpnum), r.sharp_t_rec, f"grid offset {off} at lag {lag}"))
                continue
            for q in QUANTITIES:
                for s in dsn.SIGMAS:
                    for thr in dsn.STRONG:
                        m = dsn.pair_moments(x, y, s, strong_gauss=thr, quantity=q)
                        rows.append({"harpnum": int(r.harpnum), "t_rec": r.sharp_t_rec, "month": r.month,
                                     "center_angle": float(r.center_angle), "lag_min": lag, "quantity": q,
                                     "sigma_px": s, "pixels": "all" if thr is None else f"|B|>{thr:g}G", **m})
        if i % 20 == 0:
            print(f"{i}/{len(frames)} frames", flush=True)
    return pd.DataFrame(rows), skipped


def matched_bracket(mom, pixels="all"):
    """Lag bracket on the frames that have BOTH lags (the 24-min pairs fail the grid tolerance more often, so the
    unmatched r12 and r24 are computed on different frames): per quantity and scale, pooled r12 and r24 over the
    common frames, their count, and the shared-noise upper bound."""
    m = mom[mom.pixels == pixels]
    keys12 = set(map(tuple, m[m.lag_min == 12][["harpnum", "t_rec"]].drop_duplicates().to_numpy().tolist()))
    keys24 = set(map(tuple, m[m.lag_min == 24][["harpnum", "t_rec"]].drop_duplicates().to_numpy().tolist()))
    both = keys12 & keys24
    m = m[[tuple(x) in both for x in m[["harpnum", "t_rec"]].to_numpy().tolist()]]
    rows = []
    for (q, s), g in m.groupby(["quantity", "sigma_px"]):
        r = {int(lag): dsn.pooled([dict(zip(MOMENT_KEYS, v)) for v in gg[list(MOMENT_KEYS)].to_numpy()])["corr"]
             for lag, gg in g.groupby("lag_min")}
        if 12 in r and 24 in r:
            rows.append({"quantity": q, "sigma_px": s, "pixels": pixels, "n_frames": int(g[g.lag_min == 12].t_rec.nunique()),
                         "r12_matched": r[12], "r24": r[24], "c_upper": shared_noise_upper(r[12], r[24])})
    return pd.DataFrame(rows)


def summarize(mom, n_boot=2000, seed=0):
    """Pooled statistics per (quantity, lag, sigma, pixels) with bootstrap intervals over HARPs (all pixels),
    plus the lag bracket, and pooled r12 by center-angle bin and by month (all pixels)."""
    rows = []
    for (q, lag, s, px), g in mom.groupby(["quantity", "lag_min", "sigma_px", "pixels"], sort=True):
        pooled = dsn.pooled([dict(zip(MOMENT_KEYS, v)) for v in g[list(MOMENT_KEYS)].to_numpy()])
        row = {"quantity": q, "lag_min": lag, "sigma_px": s, "sigma_mm": round(s * ph.pixel_scale_mm(), 3),
               "pixels": px, "n_frames": int(len(g)), "n_harps": int(g.harpnum.nunique()), **pooled}
        if px == "all":
            per_harp = {h: _sum_moments([dict(zip(MOMENT_KEYS, v)) for v in gh[list(MOMENT_KEYS)].to_numpy()])
                        for h, gh in g.groupby("harpnum")}
            row["corr_ci_lo"], row["corr_ci_hi"] = bootstrap_corr(per_harp, n_boot, seed)
        rows.append(row)
    table = pd.DataFrame(rows)
    bracket = []
    for (q, s, px), g in table.groupby(["quantity", "sigma_px", "pixels"]):
        r = {int(l): float(c) for l, c in zip(g.lag_min, g["corr"])}
        if 12 in r and 24 in r:
            bracket.append({"quantity": q, "sigma_px": s, "pixels": px, "r12": r[12], "r24": r[24],
                            "c_upper": shared_noise_upper(r[12], r[24])})
    by_angle, by_month = [], []
    g12 = mom[(mom.lag_min == 12) & (mom.pixels == "all")]
    for (q, s), g in g12.groupby(["quantity", "sigma_px"]):
        for lo, hi in ANGLE_BINS:
            sel = g[(g.center_angle >= lo) & (g.center_angle < hi)]
            if len(sel):
                p = dsn.pooled([dict(zip(MOMENT_KEYS, v)) for v in sel[list(MOMENT_KEYS)].to_numpy()])
                by_angle.append({"quantity": q, "sigma_px": s, "angle_bin": f"{lo:g}-{hi:g}", "n_frames": int(len(sel)), "corr": p["corr"], "snr": p["snr"]})
        for month, sel in g.groupby("month"):
            p = dsn.pooled([dict(zip(MOMENT_KEYS, v)) for v in sel[list(MOMENT_KEYS)].to_numpy()])
            by_month.append({"quantity": q, "sigma_px": s, "month": month, "n_frames": int(len(sel)), "corr": p["corr"], "snr": p["snr"]})
    return table, pd.DataFrame(bracket), pd.DataFrame(by_angle), pd.DataFrame(by_month)


def cmd_analyze(a):
    run_dir = os.path.join(ROOT, "data", a.run)
    rel = os.path.join(run_dir, "reliability")
    frames = pd.read_csv(os.path.join(rel, "frames.csv"))
    mpath = os.path.join(rel, "moments.csv")
    jpath = os.path.join(rel, "reliability_by_scale.json")
    if os.path.exists(mpath) and not a.recompute:
        mom = pd.read_csv(mpath)
        skipped = []
        if os.path.exists(jpath):            # keep the skip list of the run that computed the cached moments
            with open(jpath) as f:
                skipped = json.load(f).get("skipped", [])
    else:
        mom, skipped = frame_moments(run_dir, frames)
        mom.to_csv(mpath, index=False)
    table, bracket, by_angle, by_month = summarize(mom, a.n_boot, a.seed)
    matched = matched_bracket(mom)
    table.to_csv(os.path.join(rel, "reliability_by_scale.csv"), index=False)
    out = {"run": a.run, "n_frames_sampled": int(len(frames)), "n_frames_used_lag12": int(mom[mom.lag_min == 12].t_rec.nunique()),
           "n_frames_used_lag24": int(mom[mom.lag_min == 24].t_rec.nunique()), "n_harps": int(mom.harpnum.nunique()),
           "skipped": skipped, "table": table.to_dict(orient="records"), "bracket": bracket.to_dict(orient="records"),
           "matched_bracket": matched.to_dict(orient="records"),
           "by_angle": by_angle.to_dict(orient="records"), "by_month": by_month.to_dict(orient="records"),
           "note": "corr = frame-to-frame correlation of the smoothed quantity; SNR = sqrt(corr/(1-corr)); "
                   "the *_g_per_mm columns are mA/m^2 for quantity jz"}
    with open(os.path.join(rel, "reliability_by_scale.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    pd.set_option("display.width", 220)
    show = table[table.pixels == "all"][["quantity", "lag_min", "sigma_px", "sigma_mm", "n_frames", "n_harps", "corr", "corr_ci_lo", "corr_ci_hi", "snr"]]
    print(show.round(3).to_string(index=False))
    print("\nbracket (all pixels, all frames of each lag):")
    print(bracket[bracket.pixels == "all"].round(3).to_string(index=False))
    print("\nbracket on the frames that have both lags (all pixels):")
    print(matched.round(3).to_string(index=False))
    print(f"\nwrote {rel}/reliability_by_scale.{{csv,json}} ({len(skipped)} frame-lags skipped)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--run", default="full")
    s.add_argument("--per_month", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("download")
    d.add_argument("--run", default="full")
    d.add_argument("--email", required=True)
    d.add_argument("--part", default="0/1", help="k/n: this process takes every n-th HARP starting at k")
    d.add_argument("--max_retries", type=int, default=6, help="retries per HARP on JSOC's 'pending export' refusal")
    d.add_argument("--wait_s", type=int, default=600, help="seconds to wait before such a retry")
    z = sub.add_parser("analyze")
    z.add_argument("--run", default="full")
    z.add_argument("--n_boot", type=int, default=2000)
    z.add_argument("--seed", type=int, default=0)
    z.add_argument("--recompute", action="store_true")
    a = ap.parse_args(argv)
    {"sample": cmd_sample, "download": cmd_download, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
