"""
build_dataset.py -- day-by-day, resumable wrapper around data_active_region_created.py

Runs the pipeline one day at a time over [start, end). Each day goes to
data/<run>/days/YYYY-MM-DD/ and ends with a summary.json written by the
pipeline. Days whose summary says they finished are skipped on re-runs, so the
build can be stopped and restarted at any time.

JSOC allows ONE pending export per email, so days run strictly one after
another and a lock file under the data root stops a second build from starting.
When JSOC reports a pending export, the day is retried after 10 minutes (up to
3 times). After each day, daily outputs are merged into data/<run>/manifest.csv,
regions.csv, days.csv and skipped.csv, and data/ + results/ disk usage is
checked against --max_disk_gb.

Usage (from the project root, after `source env.sh`):
    python pipeline/build_dataset.py --run pilot --start 2011-02-01 --end 2011-03-01 \
        --email you@example.com --save_mode both \
        --max_center_angle 60 --min_align_corr 0.5 --lr_grid_fix --align_refine xcorr

    add --dry_run to query and match only (writes data/<run>/dry_run.csv),
    add --delete_raw to delete a day's raw FITS once its outputs are verified.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(ROOT, "data_active_region_created.py")
DONE_STATUSES = {"ok", "no_frames", "no_matches"}
SUMMARY_KEYS = ("status", "mdi_frames", "sharp_records", "harps", "matched_pairs",
                "pairs_after_limb", "pairs_saved", "patches", "patch_files_saved", "regions_saved")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def day_windows(start, end, days=1):
    """[(YYYY-MM-DD, window_start, inclusive_window_end)] covering [start, end).

    Each window spans `days` calendar days (the last one may be shorter) and ends
    one second before the following midnight (or before `end`). Multi-day windows
    cut the number of JSOC exports (one MDI export and one per HARP per window),
    which dominate the build time; the label is the window's first day.
    JSOC still rounds range ends to the series' time slots, so the pipeline is
    always run with --strict_window, which keeps a 00:00 frame on one day only.
    """
    windows = []
    t = start
    while t < end:
        next_midnight = datetime(t.year, t.month, t.day) + timedelta(days=days)
        stop = min(next_midnight, end)
        windows.append((t.strftime("%Y-%m-%d"), t, stop - timedelta(seconds=1)))
        t = stop
    return windows


def read_summary(day_dir):
    path = os.path.join(day_dir, "summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def is_day_done(day_dir):
    s = read_summary(day_dir)
    return bool(s) and s.get("status") in DONE_STATUSES


def is_pending_export_error(text):
    return "pending export" in text.lower()


def pipeline_command(args, day_dir, win_start, win_end, dry_run):
    cmd = [sys.executable, "-u", PIPELINE,  # -u: progress reaches the day log live
           "--email", args.email,
           "--start", win_start.isoformat(), "--end", win_end.isoformat(),
           "--out_dir", day_dir,
           "--summary_json", os.path.join(day_dir, "summary.json"),
           "--save_mode", args.save_mode,
           "--align_metric", args.align_metric,
           "--align_refine", args.align_refine,
           "--max_shift_lr", str(args.max_shift_lr),
           "--strict_window"]  # always: JSOC slot rounding would give 00:00 frames to two days
    if args.max_center_angle is not None:
        cmd += ["--max_center_angle", str(args.max_center_angle)]
    if args.min_align_corr is not None:
        cmd += ["--min_align_corr", str(args.min_align_corr)]
    if args.lr_grid_fix:
        cmd.append("--lr_grid_fix")
    if args.mdi_quality_mask is not None:
        cmd += ["--mdi_quality_mask", args.mdi_quality_mask]
    if dry_run:
        cmd.append("--dry_run")
    cmd += shlex.split(args.pipeline_args)
    return cmd


def _pair_scale(lr, hr):
    if lr.ndim != 2 or hr.ndim != 3 or hr.shape[0] != 3:
        return None
    s1, s2 = hr.shape[1] / lr.shape[0], hr.shape[2] / lr.shape[1]
    return int(s1) if s1 == s2 and s1 == int(s1) and s1 >= 1 else None


def verify_day(day_dir):
    """Every listed patch/region file exists, loads, and has an lr/hr shape
    pair with one integer scale; counts match the day's summary."""
    s = read_summary(day_dir)
    if not s or s.get("status") not in DONE_STATUSES:
        return False, "day not finished"
    for csv_name, col, count_key in (("manifest.csv", "patch_file", "patch_files_saved"),
                                     ("regions.csv", "region_file", "regions_saved")):
        path = os.path.join(day_dir, csv_name)
        n_listed = 0
        if os.path.exists(path):
            files = pd.read_csv(path)[col]
            n_listed = len(files)
            for rel in files:
                full = os.path.join(day_dir, rel)
                if not os.path.exists(full):
                    return False, f"missing {rel}"
                try:
                    with np.load(full, allow_pickle=False) as d:
                        if _pair_scale(d["lr"], d["hr"]) is None:
                            return False, f"bad shapes in {rel}: {d['lr'].shape} {d['hr'].shape}"
                except Exception as e:
                    return False, f"cannot load {rel}: {e}"
        expected = s.get(count_key)
        if expected is not None and int(expected) != n_listed:
            return False, f"{csv_name} lists {n_listed} files, summary says {expected}"
    return True, "ok"


def delete_raw(day_dir):
    """Delete the day's raw MDI/SHARP FITS files (re-downloadable). Only
    regular *.fits files directly inside <day>/raw/mdi and <day>/raw/sharp."""
    freed = 0
    for sub in ("mdi", "sharp"):
        folder = os.path.join(day_dir, "raw", sub)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if name.endswith(".fits") and os.path.isfile(path) and not os.path.islink(path):
                freed += os.path.getsize(path)
                os.remove(path)
    return freed


def disk_usage_bytes(paths):
    total = 0
    for p in paths:
        if os.path.exists(p):
            out = subprocess.run(["du", "-sb", p], capture_output=True, text=True).stdout
            total += int(out.split()[0]) if out else 0
    return total


def merge_outputs(run_dir):
    """Concatenate every day's manifest.csv / regions.csv (paths re-rooted at
    run_dir) and summaries into run_dir/{manifest,regions,days,skipped}.csv."""
    days_root = os.path.join(run_dir, "days")
    if not os.path.isdir(days_root):
        return
    manifests, regions, day_rows, skipped = [], [], [], []
    for day in sorted(os.listdir(days_root)):
        day_dir = os.path.join(days_root, day)
        s = read_summary(day_dir)
        if not s:
            continue
        row = {"day": day, **{k: s.get(k) for k in SUMMARY_KEYS}}
        for reason, n in (s.get("skipped_counts") or {}).items():
            row[f"skipped_{reason}"] = n
        day_rows.append(row)
        skipped += [dict(r, day=day) for r in s.get("skipped", [])]
        if s.get("status") != "ok":
            continue
        for csv_name, col, bucket in (("manifest.csv", "patch_file", manifests),
                                      ("regions.csv", "region_file", regions)):
            path = os.path.join(day_dir, csv_name)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df[col] = f"days/{day}/" + df[col].astype(str)
                df.insert(0, "day", day)
                bucket.append(df)
    for name, frames in (("manifest.csv", manifests), ("regions.csv", regions)):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(os.path.join(run_dir, name), index=False)
    if day_rows:
        pd.DataFrame(day_rows).fillna({c: 0 for c in pd.DataFrame(day_rows).columns
                                       if c.startswith("skipped_")}).to_csv(
            os.path.join(run_dir, "days.csv"), index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(run_dir, "skipped.csv"), index=False)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def claim_window(claims_dir, day, email):
    """Claim a window for this builder (several JSOC emails sharing one date range). Returns False
    if a live builder already holds it; a claim left by a dead builder is taken over. The claim file
    is created atomically (O_EXCL) and re-read to confirm ownership."""
    os.makedirs(claims_dir, exist_ok=True)
    path = os.path.join(claims_dir, f"{day}.claim")
    mine = f"{email} {os.getpid()}"
    if os.path.exists(path):
        try:
            owner_pid = int(open(path).read().split()[1])
        except (ValueError, IndexError, OSError):
            owner_pid = -1
        if owner_pid == os.getpid():
            return True
        if owner_pid > 0 and _pid_alive(owner_pid):
            return False
        try:
            os.remove(path)          # stale coordination file of a builder that no longer runs
        except FileNotFoundError:
            pass
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(mine)
    return open(path).read().strip() == mine


class JsocLock:
    """One build at a time per data root and JSOC email (JSOC allows one pending export per
    registered email, so builds with different emails may run side by side)."""

    def __init__(self, data_root, email=None):
        name = ".jsoc_build.lock" if email is None else ".jsoc_build." + re.sub(r"[^A-Za-z0-9]+", "_", email) + ".lock"
        self.path = os.path.join(data_root, name)

    def __enter__(self):
        if os.path.exists(self.path):
            try:
                pid = int(open(self.path).read().strip())
                os.kill(pid, 0)
                raise SystemExit(f"Another build (pid {pid}) holds {self.path}; not starting.")
            except (ValueError, ProcessLookupError):
                pass  # stale lock from a dead process
        with open(self.path, "w") as f:
            f.write(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        if os.path.exists(self.path):
            os.remove(self.path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Dataset name: output goes to <data_root>/<run>/")
    p.add_argument("--start", required=True, help="Inclusive start, ISO date/time")
    p.add_argument("--end", required=True, help="Exclusive end, ISO date/time")
    p.add_argument("--email", required=True)
    p.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    p.add_argument("--save_mode", choices=("patches", "regions", "both"), default="patches")
    p.add_argument("--max_center_angle", type=float, default=None)
    p.add_argument("--min_align_corr", type=float, default=None)
    p.add_argument("--align_metric", choices=("los", "br"), default="los")
    p.add_argument("--lr_grid_fix", action="store_true")
    p.add_argument("--align_refine", choices=("none", "xcorr"), default="none")
    p.add_argument("--max_shift_lr", type=float, default=3.0)
    p.add_argument("--mdi_quality_mask", default=None,
                   help="Passed to the pipeline, e.g. 0x80000000 (drop MDI no-data frames).")
    p.add_argument("--pipeline_args", default="",
                   help="Extra arguments passed verbatim to data_active_region_created.py")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_done_in", nargs="*", default=None,
                   help="other run dirs: skip windows already done there (parallel builds with several JSOC emails)")
    p.add_argument("--claims_dir", default=None,
                   help="shared claims dir: builders with different emails take the next unclaimed window")
    p.add_argument("--window_days", type=int, default=1,
                   help="days per pipeline call / JSOC export window (default 1)")
    p.add_argument("--delete_raw", action="store_true",
                   help="After a day's outputs are verified, delete its raw FITS files.")
    p.add_argument("--max_disk_gb", type=float, default=50.0,
                   help="Stop if data/ + results/ exceed this many GB (checked after each day).")
    p.add_argument("--pending_wait_min", type=float, default=10.0)
    p.add_argument("--max_pending_retries", type=int, default=3)
    return p


def run_day(args, day_dir, win_start, win_end, log):
    """Run the pipeline for one day, retrying on JSOC 'pending export' (up to
    --max_pending_retries, waiting --pending_wait_min) and once on any other
    failure. Returns the final status string."""
    os.makedirs(day_dir, exist_ok=True)
    day_log = os.path.join(day_dir, "build.log")
    cmd = pipeline_command(args, day_dir, win_start, win_end, dry_run=False)
    pending_retries, other_retries = 0, 0
    while True:
        with open(day_log, "a") as f:
            f.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} $ {' '.join(cmd)}\n")
            f.flush()
            start_pos = f.tell()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        with open(day_log) as f:
            f.seek(start_pos)
            text = f.read()
        if is_day_done(day_dir):
            return read_summary(day_dir)["status"]
        if is_pending_export_error(text) and pending_retries < args.max_pending_retries:
            pending_retries += 1
            log(f"  JSOC pending export; waiting {args.pending_wait_min:g} min "
                f"(retry {pending_retries}/{args.max_pending_retries})")
            time.sleep(args.pending_wait_min * 60)
            continue
        if not is_pending_export_error(text) and other_retries < 1:
            other_retries += 1
            log(f"  pipeline exited with code {proc.returncode}; retrying once in 2 min "
                f"(see {day_log})")
            time.sleep(120)
            continue
        return "failed"


def format_day(day, s):
    if not s:
        return f"{day}: no summary"
    skipped = ", ".join(f"{k}={v}" for k, v in sorted((s.get("skipped_counts") or {}).items())) or "none"
    return (f"{day}: status={s.get('status')} mdi_frames={s.get('mdi_frames')} "
            f"harps={s.get('harps')} pairs={s.get('matched_pairs')} "
            f"(after limb {s.get('pairs_after_limb')}, saved {s.get('pairs_saved')}) "
            f"patches={s.get('patches')} regions={s.get('regions_saved')} | skipped: {skipped}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    run_dir = os.path.join(args.data_root, args.run)
    os.makedirs(run_dir, exist_ok=True)
    main_log = os.path.join(run_dir, "dry_run.log" if args.dry_run else "build.log")

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(main_log, "a") as f:
            f.write(line + "\n")

    windows = day_windows(start, end, days=args.window_days)
    log(f"{'DRY RUN' if args.dry_run else 'BUILD'} {args.run}: {len(windows)} window(s) of {args.window_days} day(s) "
        f"{args.start} -> {args.end} | {' '.join(sys.argv[1:])}")

    with JsocLock(args.data_root, email=args.email):
        if args.dry_run:
            rows = []
            for day, ws, we in windows:
                out = os.path.join(run_dir, "dry_run", day)
                os.makedirs(out, exist_ok=True)
                cmd = pipeline_command(args, out, ws, we, dry_run=True)
                with open(os.path.join(out, "dry_run.log"), "w") as f:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
                s = read_summary(out) or {}
                rows.append({"day": day, **{k: s.get(k) for k in (
                    "status", "mdi_frames", "sharp_records", "harps", "matched_pairs",
                    "pairs_after_limb", "exports_needed")}, "skipped_limb":
                    (s.get("skipped_counts") or {}).get("limb", 0),
                    "pairs_per_harp": json.dumps(s.get("pairs_per_harp", {}))})
                log(f"{day}: status={s.get('status')} mdi_frames={s.get('mdi_frames')} "
                    f"harps={s.get('harps')} pairs={s.get('matched_pairs')} "
                    f"after_limb={s.get('pairs_after_limb')}")
            pd.DataFrame(rows).to_csv(os.path.join(run_dir, "dry_run.csv"), index=False)
            log(f"Wrote {os.path.join(run_dir, 'dry_run.csv')}")
            return

        for day, ws, we in windows:
            day_dir = os.path.join(run_dir, "days", day)
            if is_day_done(day_dir):
                log(f"{day}: already done, skipping")
                continue
            other = [d for d in (args.skip_done_in or []) if is_day_done(os.path.join(d, "days", day))]
            if other:
                log(f"{day}: already done in {other[0]}, skipping")
                continue
            if args.claims_dir and not claim_window(args.claims_dir, day, args.email):
                log(f"{day}: claimed by another builder, skipping")
                continue
            t0 = time.time()
            log(f"{day}: starting ({ws.isoformat()} .. {we.isoformat()})")
            status = run_day(args, day_dir, ws, we, log)
            log(format_day(day, read_summary(day_dir)) + f" [{(time.time() - t0) / 60:.1f} min]")
            if status == "failed":
                log(f"{day}: FAILED after retries; it will be retried on the next run.")
            elif args.delete_raw:
                ok, msg = verify_day(day_dir)
                if ok:
                    log(f"{day}: outputs verified; deleted {delete_raw(day_dir) / 1e6:.1f} MB of raw FITS")
                else:
                    log(f"{day}: NOT deleting raw FITS, verification failed: {msg}")
            merge_outputs(run_dir)
            used_gb = disk_usage_bytes([args.data_root, os.path.join(ROOT, "results")]) / 1e9
            log(f"disk: data/ + results/ = {used_gb:.2f} GB (limit {args.max_disk_gb:g} GB)")
            if used_gb > args.max_disk_gb:
                log("STOP: disk budget exceeded. Report before continuing.")
                break
        merge_outputs(run_dir)
        log("done")


if __name__ == "__main__":
    main()
