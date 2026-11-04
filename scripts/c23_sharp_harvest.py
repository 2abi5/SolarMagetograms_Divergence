"""Phase 13 / F3b: harvest hmi.sharp_cea_720s keyword headers 2010-2026 at 96-min cadence
(reports/phase_13.md 4, F3 addendum). JSOC metadata queries only - no images.

    python scripts/c23_sharp_harvest.py run [--start 2010-05 --end 2026-09]

Monthly queries (resumable: one CSV per month in data/c23/sharp_era/), keys = identity +
vector keywords + Bobra LOS keywords; QUALITY == 0 and |LON_FWT| <= 65 applied at load time
by the consumer, not here (raw rows kept). Concatenated manifest -> data/c23/sharp_era.csv.
"""
import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "c23", "sharp_era")
KEYS = ("HARPNUM,T_REC,NOAA_AR,LAT_FWT,LON_FWT,QUALITY,"
        "USFLUX,TOTUSJZ,TOTUSJH,R_VALUE,AREA,USFLUXL,MEANGBL")


def months(start, end):
    cur = pd.Period(start, "M")
    last = pd.Period(end, "M")
    while cur <= last:
        yield str(cur)
        cur += 1


def cmd_run(a):
    import drms
    os.makedirs(OUT_DIR, exist_ok=True)
    c = drms.Client()
    done, failed = 0, []
    for m in months(a.start, a.end):
        path = os.path.join(OUT_DIR, f"{m}.csv")
        if os.path.exists(path):
            done += 1
            continue
        p = pd.Period(m, "M")
        t0 = f"{p.year}.{p.month:02d}.01_00:00:00_TAI"
        q = (p + 1)
        t1 = f"{q.year}.{q.month:02d}.01_00:00:00_TAI"
        try:
            k = c.query(f"hmi.sharp_cea_720s[][{t0}-{t1}@96m]", key=KEYS)
            k.to_csv(path, index=False)
            print(f"{m}: {len(k)} rows", flush=True)
        except Exception as e:
            print(f"{m}: FAILED {str(e)[:70]}", flush=True)
            failed.append(m)
        done += 1
        time.sleep(a.sleep_s)
    parts = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".csv"))
    df = pd.concat([pd.read_csv(os.path.join(OUT_DIR, f)) for f in parts], ignore_index=True)
    df.to_csv(os.path.join(ROOT, "data", "c23", "sharp_era.csv"), index=False)
    print(f"wrote data/c23/sharp_era.csv: {len(df)} rows, {df.HARPNUM.nunique()} HARPs, "
          f"{len(parts)} months, {len(failed)} failed months: {failed[:10]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run")
    rn.add_argument("--start", default="2010-05")
    rn.add_argument("--end", default="2026-09")
    rn.add_argument("--sleep_s", type=float, default=2.0)
    rn.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
