"""Phase 14 / section 3.4: harvest the Bobra & Couvidat (2015) keyword set for THEIR window
from JSOC (metadata only, no images, no export email).

    python scripts/c23_bobra_harvest.py run [--start 2010-05 --end 2014-05] [--cadence 96m]

Of their 25 parameters, 18 exist as JSOC SHARP keywords; the seven Lorentz-force terms
(TOTBSQ, TOTF[XYZ], EPS[XYZ]) are NOT in the series - Bobra computed those herself from the
vector data, following Fisher et al. (2012). Ten of their SELECTED thirteen are available here.
Monthly queries, resumable (one CSV per month); 12m cadence must be chunked smaller than a
month or JSOC returns status=6. Output: data/c23/bobra_era{suffix}.csv - a NEW file; the frozen
data/c23/sharp_era.csv is never touched.
"""
import argparse
import os
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# their 25 minus the seven Lorentz-force terms JSOC does not carry
BC18 = ("TOTUSJH,TOTPOT,TOTUSJZ,ABSNJZH,SAVNCPP,USFLUX,AREA_ACR,MEANPOT,R_VALUE,SHRGT45,"
        "MEANSHR,MEANGAM,MEANGBT,MEANGBZ,MEANGBH,MEANJZH,MEANJZD,MEANALP")
KEYS = "HARPNUM,T_REC,NOAA_AR,LAT_FWT,LON_FWT,QUALITY," + BC18
ABSENT_FROM_JSOC = ("TOTBSQ", "TOTFX", "TOTFY", "TOTFZ", "EPSX", "EPSY", "EPSZ")


def months(start, end):
    cur, last = pd.Period(start, "M"), pd.Period(end, "M")
    while cur <= last:
        yield str(cur)
        cur += 1


def spans(period, cadence):
    """One (t0, t1) per query; sub-month chunks when the cadence is fine enough to blow the
    JSOC per-query record limit."""
    step = {"12m": 7, "24m": 14}.get(cadence, 0)      # days per chunk, 0 = whole month
    s = pd.Timestamp(period.start_time)
    e = pd.Timestamp((period + 1).start_time)
    if not step:
        return [(s, e)]
    out, cur = [], s
    while cur < e:
        nxt = min(cur + pd.Timedelta(days=step), e)
        out.append((cur, nxt)); cur = nxt
    return out


fmt = lambda t: f"{t.year}.{t.month:02d}.{t.day:02d}_{t.hour:02d}:00:00_TAI"


def cmd_run(a):
    import drms
    out_dir = os.path.join(ROOT, "data", "c23", f"bobra_era_{a.cadence}")
    os.makedirs(out_dir, exist_ok=True)
    c = drms.Client()
    failed = []
    for m in months(a.start, a.end):
        path = os.path.join(out_dir, f"{m}.csv")
        if os.path.exists(path):
            continue
        parts = []
        ok = True
        for t0, t1 in spans(pd.Period(m, "M"), a.cadence):
            try:
                k = c.query(f"hmi.sharp_cea_720s[][{fmt(t0)}-{fmt(t1)}@{a.cadence}]", key=KEYS)
                parts.append(k)
            except Exception as e:
                print(f"{m} {t0.date()}: FAILED {str(e)[:60]}", flush=True)
                ok = False
            time.sleep(a.sleep_s)
        if ok and parts:
            pd.concat(parts, ignore_index=True).to_csv(path, index=False)
            print(f"{m}: {sum(len(p) for p in parts)} rows", flush=True)
        else:
            failed.append(m)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".csv"))
    df = pd.concat([pd.read_csv(os.path.join(out_dir, f)) for f in files], ignore_index=True)
    dest = os.path.join(ROOT, "data", "c23", f"bobra_era_{a.cadence}.csv")
    df.to_csv(dest, index=False)
    print(f"wrote {dest}: {len(df)} rows, {df.HARPNUM.nunique()} HARPs, {len(files)} months, "
          f"{len(failed)} failed: {failed[:8]}")
    print(f"keywords held: 18 of Bobra's 25; absent from JSOC: {', '.join(ABSENT_FROM_JSOC)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--start", default="2010-05")
    r.add_argument("--end", default="2014-05")        # Bobra & Couvidat's own window
    r.add_argument("--cadence", default="96m")
    r.add_argument("--sleep_s", type=float, default=1.5)
    r.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
