"""Combine two build runs of the same period split between two JSOC emails (Phase 8):
move every finished window folder of --src into --dst, then rebuild dst's tables.

    python scripts/merge_runs.py --dst data/full --src data/full_b
Refuses (and moves nothing) if a window is finished in both. A window that dst started but did not
finish (interrupted when the download was split) is moved aside to dst/days_incomplete/, never deleted.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.build_dataset import is_day_done, merge_outputs  # noqa: E402


def merge_runs(dst, src):
    src_days, dst_days = os.path.join(src, "days"), os.path.join(dst, "days")
    names = sorted(n for n in (os.listdir(src_days) if os.path.isdir(src_days) else [])
                   if is_day_done(os.path.join(src_days, n)))     # unfinished src windows stay where they are
    clash = [n for n in names if is_day_done(os.path.join(dst_days, n))]
    if clash:
        raise FileExistsError(f"windows finished in both runs: {clash}")
    os.makedirs(dst_days, exist_ok=True)
    for n in names:
        if os.path.exists(os.path.join(dst_days, n)):      # unfinished in dst: keep it, out of the way
            aside = os.path.join(dst, "days_incomplete")
            os.makedirs(aside, exist_ok=True)
            os.rename(os.path.join(dst_days, n), os.path.join(aside, n))
        os.rename(os.path.join(src_days, n), os.path.join(dst_days, n))
    merge_outputs(dst)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True)
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    moved = merge_runs(a.dst, a.src)
    print(f"moved {len(moved)} window(s) from {a.src} into {a.dst}; tables rebuilt")


if __name__ == "__main__":
    main()
