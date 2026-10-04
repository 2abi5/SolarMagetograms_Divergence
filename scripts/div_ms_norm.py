"""Per-scale normalisation for the match_ms divergence loss (reports/phase_6.md 8):
mean square of div_h of the Gaussian-smoothed TRUE field on training patches.

    python scripts/div_ms_norm.py --run pilot --scales 1 2 4

Writes <run_dir>/div_ms_stats.json; with it a zero prediction scores 1 on every scale.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sr import losses as L  # noqa: E402


def scale_mean_squares(batches, scales):
    """batches: iterable of (hr (B,3,H,W), mask (B,1,H,W)); pooled masked mean of div_h(G_s*hr)^2 per scale."""
    num, den = np.zeros(len(scales)), np.zeros(len(scales))
    for hr, mask in batches:
        for i, s in enumerate(scales):
            d, v = L.smoothed_div(hr[:, 0], hr[:, 1], mask[:, 0], float(s))
            num[i] += float((d ** 2 * v).sum())
            den[i] += float(v.sum())
    return (num / np.maximum(den, 1)).tolist()


def grid_batches(ds, n=None, seed=0, batch=256, double=False):
    """(hr, mask) batches of a dataset's items (all of them, or n random ones)."""
    idx = np.arange(len(ds)) if n is None or n >= len(ds) else np.random.default_rng(seed).choice(len(ds), n, replace=False)
    for k in range(0, len(idx), batch):
        items = [ds[int(i)] for i in idx[k:k + batch]]
        hr = torch.stack([it["hr"] for it in items])
        yield (hr.double() if double else hr), torch.stack([it["mask"] for it in items])


def main():
    from sr.datasets import PatchDataset, RegionGridDataset, split_harps
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("patches", "regions"), default="patches",
                    help="regions: stride-8 grid crops of the saved regions (Phase 8, no patch files)")
    a = ap.parse_args()
    run_dir = os.path.join(ROOT, "data", a.run)
    harps = split_harps(os.path.join(ROOT, "splits", f"{a.run}.json"), "train")
    ds = PatchDataset(run_dir, harps=harps) if a.mode == "patches" else RegionGridDataset(run_dir, harps=harps)
    n = min(a.n, len(ds))
    if a.mode == "patches":   # unchanged sampling (the pilot's div_ms_stats.json was made this way)
        idx = np.random.default_rng(a.seed).choice(len(ds), n, replace=False)

        def batches(bs=256):
            for k in range(0, len(idx), bs):
                items = [ds[int(i)] for i in idx[k:k + bs]]
                yield (torch.stack([it["hr"] for it in items]).double(), torch.stack([it["mask"] for it in items]))
        ms = scale_mean_squares(batches(), a.scales)
    else:
        ms = scale_mean_squares(grid_batches(ds, n=n, seed=a.seed, double=True), a.scales)
    out = {"run": a.run, "split": "train", "mode": a.mode, "n_patches": int(n), "scales_px": a.scales,
           "mean_square_div": ms, "units": "normalised field (/3500) per HR pixel, squared"}
    with open(os.path.join(run_dir, "div_ms_stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
