"""Divergence signal vs noise by smoothing scale (reports/phase_6.md, follow-up).

    python scripts/div_snr_by_scale.py --run pilot

Uses the TRAINING-split SHARP pairs already on disk for the temporal noise floor
(a frame and the frame 12 min later, same sample as scripts/noise_floor_temporal.py).
For each Gaussian smoothing scale sigma (HR pixels) the horizontal divergence
div_h of the smoothed Bp, Bt is computed in both frames. Real structure is
common to both frames and noise is not, so over the pooled valid pixels

    corr(div_t, div_t+12)  = signal variance / (signal + noise variance)
    SNR                    = sqrt(corr / (1 - corr))
    signal rms             = sqrt(cov(div_t, div_t+12))
    per-frame noise rms    = rms(div_t+12 - div_t) / sqrt(2)

(real evolution within 12 min counts as noise, so the SNR is conservative).
Reported for all valid pixels and for strong-field pixels (|B| > threshold at t).
Writes <run_dir>/div_snr_by_scale.json.
"""
import argparse
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sr import physics as ph  # noqa: E402

NORM = 3500.0
SIGMAS = (0.0, 1.0, 2.0, 4.0, 8.0)
STRONG = (None, 200.0, 500.0, 1000.0)


def smoothed_div(hr, sigma, norm=NORM, min_weight=0.95):
    """div_h (G/Mm, interior pixels) of Bp, Bt after NaN-aware Gaussian smoothing,
    and the interior validity mask (pixel and 4 neighbours finite, and at least
    `min_weight` of the Gaussian weight on valid pixels)."""
    hr = np.asarray(hr, np.float64)
    valid = np.isfinite(hr).all(axis=0)
    if sigma > 0:
        w = gaussian_filter(valid.astype(float), sigma, mode="constant")
        comps = [gaussian_filter(np.where(valid, hr[c], 0.0), sigma, mode="constant") for c in (0, 1)]
        with np.errstate(invalid="ignore", divide="ignore"):
            bp, bt = (x / w for x in comps)
        valid = valid & (w >= min_weight)
    else:
        bp, bt = hr[0], hr[1]
    field = np.stack([np.where(valid, bp, 0.0), np.where(valid, bt, 0.0), np.zeros_like(bp)])
    return ph.div_h_physical(field, norm=norm), ph.interior_valid(valid)


def pair_moments(a, b, sigma, strong_gauss=None, norm=NORM):
    """Sums for pooled statistics of div_h(a) and div_h(b) at scale sigma."""
    da, va = smoothed_div(a, sigma, norm)
    db, vb = smoothed_div(b, sigma, norm)
    sel = va & vb
    if strong_gauss is not None:
        babs = np.sqrt(np.nansum(np.asarray(a, np.float64) ** 2, axis=0))[1:-1, 1:-1] * norm
        sel &= babs > strong_gauss
    x, y = da[sel], db[sel]
    return {"n": int(x.size), "sx": float(x.sum()), "sy": float(y.sum()), "sxx": float((x * x).sum()),
            "syy": float((y * y).sum()), "sxy": float((x * y).sum()), "sabs": float(np.abs(x).sum()),
            "sdd": float(((y - x) ** 2).sum())}


def pooled(moments):
    m = {k: sum(d[k] for d in moments) for k in moments[0]}
    n = m["n"]
    if n < 2:
        return {"n": n, "corr": float("nan"), "snr": float("nan"), "signal_rms_g_per_mm": float("nan"),
                "noise_rms_g_per_mm": float("nan"), "mean_abs_div_g_per_mm": float("nan")}
    mx, my = m["sx"] / n, m["sy"] / n
    vx, vy = m["sxx"] / n - mx * mx, m["syy"] / n - my * my
    cov = m["sxy"] / n - mx * my
    corr = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
    snr = float(np.sqrt(corr / (1 - corr))) if 0 < corr < 1 else (float("inf") if corr >= 1 else 0.0)
    return {"n": n, "corr": float(corr), "snr": snr, "signal_rms_g_per_mm": float(np.sqrt(max(cov, 0.0))),
            "noise_rms_g_per_mm": float(np.sqrt(m["sdd"] / n / 2.0)),
            "mean_abs_div_g_per_mm": float(m["sabs"] / n)}


def load_pairs(run_dir, split_json, n_pairs=30, seed=0):
    """(frame at t, frame at t+12 min) stacks on a common grid, same sample as the temporal floor."""
    import noise_floor_temporal as nft
    import data_active_region_created as pipe
    from sr.datasets import split_harps
    reg = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    reg = reg[reg.harpnum.isin(split_harps(split_json, "train"))]
    sample = reg.sample(min(n_pairs, len(reg)), random_state=seed)
    pairs, skipped = [], []
    for r in sample.itertuples():
        stamp = pipe.parse_jsoc_time(r.sharp_t_rec)
        pat = f"hmi.sharp_cea_720s.{r.harpnum}.{{:%Y%m%d_%H%M%S}}_TAI.{{}}.fits"
        now = {c: os.path.join(run_dir, "days", r.day, "raw", "sharp", pat.format(stamp, c)) for c in ("Bp", "Bt", "Br")}
        nxt = {c: os.path.join(run_dir, "noise_floor_raw", pat.format(stamp + timedelta(minutes=12), c))
               for c in ("Bp", "Bt", "Br")}
        if not all(os.path.exists(p) for p in (*now.values(), *nxt.values())):
            skipped.append((int(r.harpnum), r.sharp_t_rec, "missing files"))
            continue
        ma, sa = nft.load_stack(now)
        mb, sb = nft.load_stack(nxt)
        a, b, off = nft.common_grid(ma["Br"], sa, mb["Br"], sb)
        if a is None:
            skipped.append((int(r.harpnum), r.sharp_t_rec, f"grid offset {off}"))
            continue
        pairs.append((int(r.harpnum), r.sharp_t_rec, a, b))
    return pairs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--split", default=None)
    args = ap.parse_args()
    run_dir = os.path.join(args.data_root, args.run)
    split = args.split or os.path.join(ROOT, "splits", f"{args.run}.json")
    pairs, skipped = load_pairs(run_dir, split)
    print(f"{len(pairs)} frame pairs used, {len(skipped)} skipped")
    rows = []
    for thr in STRONG:
        for s in SIGMAS:
            r = pooled([pair_moments(a, b, s, strong_gauss=thr) for _, _, a, b in pairs])
            rows.append({"pixels": "all" if thr is None else f"|B|>{thr:g}G", "sigma_px": s,
                         "sigma_mm": round(s * ph.pixel_scale_mm(), 3), **r})
    df = pd.DataFrame(rows)
    out = {"run": args.run, "n_pairs": len(pairs), "skipped": skipped,
           "note": "corr = frame-to-frame (12 min) correlation of div_h at each scale; "
                   "SNR = sqrt(corr/(1-corr)); evolution counts as noise (conservative)",
           "rows": df.to_dict(orient="records")}
    with open(os.path.join(run_dir, "div_snr_by_scale.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    pd.set_option("display.width", 200)
    print(df[["pixels", "sigma_px", "sigma_mm", "n", "mean_abs_div_g_per_mm", "signal_rms_g_per_mm",
              "noise_rms_g_per_mm", "corr", "snr"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
