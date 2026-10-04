"""Re-patch the saved full regions of a run at another patch size
(training_plan.md 2b, patch-size caveat: one extra pilot pair, E1 and E3 at
32x32 LR / 128x128 HR, on the regions large enough to allow it).

    python scripts/make_patch_set.py --src_run pilot --dst_run pilot_p32 --patch_size_lr 32 --stride_lr 16

Uses the pipeline's own extract_patches (same validity rule: >= 95% finite in
LR and in HR) on the same aligned regions the 16x16 patches were cut from, so
only the patch size differs (the stride stays at half the patch). Writes
data/<dst_run>/patches/*.npz and data/<dst_run>/manifest.csv (the columns the
training code reads, plus the source region's metadata). Regions smaller than
one patch are skipped. Never overwrites an existing patch set.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data_active_region_created import extract_patches  # noqa: E402
from sr.datasets import _load_pair  # noqa: E402

SCALE = 4


def _patch_region(src_dir, dst_dir, region, patch_size_lr, stride_lr):
    lr, hr = _load_pair(os.path.join(src_dir, region["region_file"]))
    stem = os.path.splitext(os.path.basename(region["region_file"]))[0]
    rows = []
    for lr_p, hr_p, row, col in extract_patches(lr, hr, SCALE, patch_size_lr, stride_lr):
        rel = os.path.join("patches", f"{stem}_{row}_{col}.npz")
        np.savez_compressed(os.path.join(dst_dir, rel), lr=lr_p, hr=hr_p)
        rows.append({**region, "patch_file": rel, "row": row, "col": col,
                     "patch_size_lr": patch_size_lr, "stride_lr": stride_lr})
    return rows


def make_patch_set(src_dir, dst_dir, patch_size_lr=32, stride_lr=16, n_threads=8):
    manifest = os.path.join(dst_dir, "manifest.csv")
    if os.path.exists(manifest):
        raise FileExistsError(f"{manifest} exists; not overwriting")
    os.makedirs(os.path.join(dst_dir, "patches"), exist_ok=True)
    regions = pd.read_csv(os.path.join(src_dir, "regions.csv")).to_dict("records")
    with ThreadPoolExecutor(n_threads) as ex:
        per_region = list(ex.map(lambda r: _patch_region(src_dir, dst_dir, r, patch_size_lr, stride_lr), regions))
    rows = [row for rs in per_region for row in rs]
    pd.DataFrame(rows).to_csv(manifest, index=False)
    info = {"src": os.path.abspath(src_dir), "patch_size_lr": patch_size_lr, "stride_lr": stride_lr,
            "scale": SCALE, "regions": len(regions),
            "regions_without_patches": sum(1 for rs in per_region if not rs), "patches": len(rows)}
    with open(os.path.join(dst_dir, "patch_set.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_run", required=True)
    ap.add_argument("--dst_run", required=True)
    ap.add_argument("--patch_size_lr", type=int, default=32)
    ap.add_argument("--stride_lr", type=int, default=16)
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    info = make_patch_set(os.path.join(a.data_root, a.src_run), os.path.join(a.data_root, a.dst_run),
                          a.patch_size_lr, a.stride_lr)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
