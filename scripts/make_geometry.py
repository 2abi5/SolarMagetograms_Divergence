"""Cache the per-pixel viewing geometry (c_p, c_t, c_r) of every saved region
(sr/geometry.py) as <run_dir>/geometry_lr/<region stem>.npy (float32, (3, ny, nx)).

    python scripts/make_geometry.py --run pilot
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sr import geometry as geo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run_dir = os.path.join(ROOT, "data", a.run)
    out = os.path.join(run_dir, "geometry_lr")
    os.makedirs(out, exist_ok=True)
    regs = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    for i, f in enumerate(regs.region_file):
        p = geo.geometry_path(out, f)
        if not os.path.exists(p):
            np.save(p, geo.region_los_coefficients(os.path.join(run_dir, f)))
        if i % 200 == 0:
            print(f"{i}/{len(regs)}", flush=True)
    print(f"wrote {len(regs)} geometry maps to {out}")


if __name__ == "__main__":
    main()
