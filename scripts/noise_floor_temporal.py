"""Temporal divergence noise floor (training_plan.md 2b), TRAINING split only.

    python scripts/noise_floor_temporal.py --run pilot --n_pairs 30 --email you@example.com

For a sample of training (HARP, T_REC) pairs, download the SHARP frame 12 min
later (Bp, Bt, Br; one JSOC export per HARP, strictly sequential, under the
same lock as the dataset build), put both frames on the same CEA grid (the
Carrington CEA grids of consecutive frames differ by ~0.06 px in 12 min; offsets
within 0.1 px of a whole pixel are rounded, others skipped -- interpolating would
smooth the noise away) and
compute mean |div_h(t) - div_h(t+12 min)| over pixels whose smoothed field
changed by < 50 G. Real evolution in 12 min is small, so this is mostly noise.
Writes <run_dir>/noise_floor_temporal.json; raw files go to <run_dir>/noise_floor_raw/.
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import drms  # noqa: E402
import sunpy.map  # noqa: E402

import data_active_region_created as pipe  # noqa: E402
from pipeline.build_dataset import JsocLock  # noqa: E402
from sr import stats as st  # noqa: E402
from sr.datasets import split_harps  # noqa: E402

NORM = 3500.0


def load_stack(paths):
    maps = {c: sunpy.map.Map(paths[c]) for c in ("Bp", "Bt", "Br")}
    return maps, np.stack([maps[c].data.astype(float) for c in ("Bp", "Bt", "Br")]) / NORM


def common_grid(map_a, stack_a, map_b, stack_b):
    """Crop both stacks to their overlap if the grids differ by whole pixels."""
    # Plain header arithmetic on the Carrington CEA grids (no sunpy frames: a
    # frame transform between obstimes would add the Sun's rotation, ~0.9 deg per
    # 96 min, as if the surface were fixed in inertial space).
    lon, lat = map_b.wcs.wcs_pix2world([[0.0, 0.0]], 0)[0]
    x0, y0 = map_a.wcs.wcs_world2pix([[lon, lat]], 0)[0]
    dx, dy = float(x0), float(y0)            # position of b's pixel (0,0) in a's pixel frame
    # SHARP grids drift ~0.06 px per 12 min relative to Carrington; round offsets
    # below 0.1 px (the residual shift makes the floor slightly conservative)
    if abs(dx - round(dx)) > 0.1 or abs(dy - round(dy)) > 0.1:
        return None, None, (dx, dy)
    dx, dy = int(round(dx)), int(round(dy))
    ha, wa = stack_a.shape[1:]
    hb, wb = stack_b.shape[1:]
    ya0, xa0 = max(0, dy), max(0, dx)
    ya1, xa1 = min(ha, dy + hb), min(wa, dx + wb)
    if ya1 - ya0 < 20 or xa1 - xa0 < 20:
        return None, None, (dx, dy)
    a = stack_a[:, ya0:ya1, xa0:xa1]
    b = stack_b[:, ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
    return a, b, (dx, dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--email", required=True)
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--split", default=None)
    ap.add_argument("--n_pairs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_change_gauss", type=float, default=50.0)
    args = ap.parse_args()
    run_dir = os.path.join(args.data_root, args.run)
    split = args.split or os.path.join(ROOT, "splits", f"{args.run}.json")
    reg = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    reg = reg[reg.harpnum.isin(split_harps(split, "train"))]
    sample = reg.sample(min(args.n_pairs, len(reg)), random_state=args.seed)
    raw_dir = os.path.join(run_dir, "noise_floor_raw")

    later = {}
    for r in sample.itertuples():
        t = pipe.parse_jsoc_time(r.sharp_t_rec) + timedelta(minutes=12)
        later.setdefault(int(r.harpnum), []).append(t.strftime("%Y.%m.%d_%H:%M:%S_TAI"))
    client = drms.Client(email=args.email)
    with JsocLock(args.data_root):
        files = pipe.batched_export_sharp(client, "hmi.sharp_cea_720s",
                                          [(h, t) for h, ts in later.items() for t in ts],
                                          raw_dir, segments=("Bp", "Bt", "Br"), label="SHARP+12min")

    rows = []
    for r in sample.itertuples():
        day_raw = os.path.join(run_dir, "days", r.day, "raw", "sharp")
        stamp = pipe.parse_jsoc_time(r.sharp_t_rec)
        pat = f"hmi.sharp_cea_720s.{r.harpnum}.{stamp:%Y%m%d_%H%M%S}_TAI"
        now = {c: os.path.join(day_raw, f"{pat}.{c}.fits") for c in ("Bp", "Bt", "Br")}
        t12 = (stamp + timedelta(minutes=12)).strftime("%Y.%m.%d_%H:%M:%S_TAI")
        nxt = files.get((int(r.harpnum), t12), {})
        if not all(os.path.exists(p) for p in now.values()) or len(nxt) < 3:
            rows.append({"harpnum": r.harpnum, "t": r.sharp_t_rec, "status": "missing files"})
            continue
        ma, sa = load_stack(now)
        mb, sb = load_stack(nxt)
        a, b, off = common_grid(ma["Br"], sa, mb["Br"], sb)
        if a is None:
            rows.append({"harpnum": r.harpnum, "t": r.sharp_t_rec, "status": f"grid offset {off}"})
            continue
        res = st.temporal_div_noise(a, b, norm=NORM, max_change_gauss=args.max_change_gauss)
        rows.append({"harpnum": r.harpnum, "t": r.sharp_t_rec, "status": "ok", "offset": off, **res})
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    out = {"run": args.run, "n_requested": len(sample), "n_used": int(len(ok)),
           "max_change_gauss": args.max_change_gauss,
           "mean_abs_g_per_mm_mean_over_pairs": float(ok.mean_abs_g_per_mm.mean()) if len(ok) else None,
           "mean_abs_g_per_mm_median_over_pairs": float(ok.mean_abs_g_per_mm.median()) if len(ok) else None,
           "pairs": df.to_dict(orient="records")}
    with open(os.path.join(run_dir, "noise_floor_temporal.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    print(df.status.value_counts().to_string())


if __name__ == "__main__":
    main()
