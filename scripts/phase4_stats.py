"""Phase 4: training-split statistics -> <run_dir>/stats.json (used by sr/train.py).

    python scripts/phase4_stats.py --run pilot [--split splits/pilot.json]

TRAINING split only (never val/test):
  - per-channel Gauss statistics (mean, std, 1st/99th pct, 99.9th pct of |B|)
  - fixed SSIM affine ranges [-a_c, a_c], a_c = 99.9th pct of |B_c| (normalised)
  - soft-histogram centres: 0 and +-60 G * 10**(0.15 k) up to the largest
    per-channel 99.9th pct of |B| (shared by all channels and all models)
  - spatial divergence noise floor: mean |div_h(raw) - div_h(Gaussian sigma=1)|
    per region, and the typical |div_h(true)| for scale
The temporal noise floor needs extra SHARP frames: scripts/noise_floor_temporal.py.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sr import physics as ph  # noqa: E402
from sr import stats as st  # noqa: E402
from sr.datasets import RegionDataset, split_harps, _load_pair  # noqa: E402
from sr.losses import log_bin_centers  # noqa: E402

NORM = 3500.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--split", default=None)
    ap.add_argument("--noise_level_g", type=float, default=60.0)
    ap.add_argument("--dl", type=float, default=0.15)
    ap.add_argument("--out", default=None, help="default: <run_dir>/stats.json")
    args = ap.parse_args()
    run_dir = os.path.join(args.data_root, args.run)
    split = args.split or os.path.join(ROOT, "splits", f"{args.run}.json")

    reg = RegionDataset(run_dir, harps=split_harps(split, "train")).rows
    hrs = [_load_pair(os.path.join(run_dir, f))[1] for f in reg.region_file]
    print(f"training regions: {len(hrs)} (HARPs {sorted(reg.harpnum.unique().tolist())})")

    cstats = st.channel_stats(hrs, norm=NORM)
    lo, hi = st.ssim_ranges(hrs)
    top = max(cstats[c]["abs_p999"] for c in st.CHANNELS) / NORM
    centers = log_bin_centers(noise=args.noise_level_g / NORM, top=top, dl=args.dl)

    spatial = np.array([st.spatial_div_noise(h, sigma=1.0, norm=NORM) for h in hrs])
    true_div = []
    for h in hrs:
        v = ph.interior_valid(np.isfinite(h).all(axis=0))
        d = np.abs(ph.div_h_physical(np.nan_to_num(h), norm=NORM))[v]
        true_div.append(float(d.mean()))
    out = {
        "run": args.run, "split": os.path.relpath(split, ROOT), "norm": NORM,
        "n_train_regions": len(hrs), "channel_stats_gauss": cstats,
        "ssim_lo": lo.tolist(), "ssim_hi": hi.tolist(),
        "ssim_note": "x' = (x - lo_c) / (hi_c - lo_c) per channel, fixed from the training split",
        "hist_centers": centers.tolist(),
        "hist_note": f"0 and +-{args.noise_level_g:g} G * 10**({args.dl} k) up to max_c p99.9(|B_c|) "
                     f"= {top * NORM:.0f} G; triangular kernel; shared by all channels",
        "noise_floor_spatial_g_per_mm": {"mean_over_regions": float(np.nanmean(spatial)),
                                         "median_over_regions": float(np.nanmedian(spatial)),
                                         "p90_over_regions": float(np.nanpercentile(spatial, 90))},
        "true_mean_abs_div_g_per_mm": {"mean_over_regions": float(np.mean(true_div)),
                                       "median_over_regions": float(np.median(true_div))},
    }
    path = args.out or os.path.join(run_dir, "stats.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "hist_centers"}, indent=1))
    print(f"{len(centers)} histogram centres; wrote {path}")


if __name__ == "__main__":
    main()
