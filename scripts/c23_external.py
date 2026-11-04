"""Overlap-pilot stage E4: external validation of the trajectory clusters
(reports/phase_12.md 6).

    python scripts/c23_external.py match    --out_dir results/full/overlap_pilot
    python scripts/c23_external.py validate --source ours --out_dir results/full/overlap_pilot

match:    parse the NGDC GOES XRS reports (data/full/c23_external/raw/goes-xrs-report_{2010,2011}.txt,
          fixed width, full 5-digit NOAA numbers) and the SWPC SRS daily files (same dir, SRS/ or
          *_SRS.tar.gz extracted; 4-digit Nmbr + 10000 in this era), build the per-AR table:
          observation window and disk time from features.csv (|lon| <= 60 frames), flare counts
          >= C1 and >= M1 whose peak falls inside the window and whose NOAA number matches,
          modal Mount Wilson and McIntosh class over the window days
          -> results/full/overlap_pilot/ar_external.csv
validate: join with clustering_<source>.csv; per variant: per-cluster flare rates (per observed
          day), Kruskal-Wallis over per-AR rates, extreme-cluster rate ratio, Cramer's V against
          the modal Mount Wilson class -> external_<source>.json
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

CLASS_W = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
LON_MAX = 60.0


def parse_xrs(path):
    """NGDC GOES XRS report -> DataFrame(peak, class_letter, class_mag, wm2, noaa_ar)."""
    rows = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line.startswith("31777") or len(line) < 63:
            continue
        d = datetime.strptime(line[5:11], "%y%m%d")
        hhmm = line[23:27].strip() or line[13:17].strip()
        if not re.fullmatch(r"\d{4}", hhmm):
            continue
        h, m = int(hhmm[:2]), int(hhmm[2:])
        peak = d + timedelta(hours=h, minutes=m)
        start = line[13:17].strip()
        if re.fullmatch(r"\d{4}", start) and hhmm < start:   # crossed midnight
            peak += timedelta(days=1)
        letter = line[59].upper()
        if letter not in CLASS_W:
            continue
        mag = line[60:63].strip()
        mag = float(mag) / 10.0 if mag.isdigit() else float("nan")
        region = line[80:85].strip() if len(line) >= 81 else ""
        rows.append({"peak": peak, "class_letter": letter, "class_mag": mag,
                     "wm2": CLASS_W[letter] * (mag if np.isfinite(mag) else 1.0),
                     "noaa_ar": int(region) if region.isdigit() else 0})
    return pd.DataFrame(rows)


def parse_srs(path):
    """One SWPC SRS file -> rows (date, noaa_ar full 5-digit, mag_type, mcintosh)."""
    m = re.search(r"(\d{8})SRS", os.path.basename(path))
    date = datetime.strptime(m.group(1), "%Y%m%d").date() if m else None
    rows, in_sec1 = [], False
    for line in open(path, errors="replace"):
        u = line.upper()
        if u.startswith("I.") and "REGIONS WITH SUNSPOTS" in u:
            in_sec1 = True
            continue
        if in_sec1 and (u.startswith("IA.") or u.startswith("II.")):
            break
        if not in_sec1:
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[0].isdigit():
            nmbr = int(parts[0])
            nmbr += 10000 if nmbr < 4000 else 0            # era rule: SRS drops the leading 1
            rows.append({"date": date, "noaa_ar": nmbr,
                         "mcintosh": parts[4].upper(), "mag_type": parts[-1].upper()})
    return rows


def load_srs_dir(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "**", "*SRS.txt"), recursive=True))
    rows = []
    for f in files:
        rows.extend(parse_srs(f))
    return pd.DataFrame(rows)


def ar_windows(features_csv, lon_max=LON_MAX):
    """Per HARP: NOAA number (modal nonzero), observation window and disk time of the |lon|<=60 frames."""
    df = pd.read_csv(features_csv)
    df = df[(df.model == "true_hmi") & (df["mask"] == "strong")]
    df = df[df.center_lon.abs() <= lon_max]
    df["_t"] = pd.to_datetime(df.mdi_t_rec.str.replace("_TAI", "", regex=False), format="%Y.%m.%d_%H:%M:%S")
    out = []
    for h, g in df.groupby("harpnum"):
        noaa = g.noaa_ar[g.noaa_ar > 0]
        out.append({"harpnum": int(h), "noaa_ar": int(noaa.mode().iloc[0]) if len(noaa) else 0,
                    "t_start": g._t.min(), "t_end": g._t.max(),
                    "n_frames": int(len(g)), "observed_days": float(len(g) * 96.0 / 1440.0)})
    return pd.DataFrame(out)


def match_flares(wins, flares):
    """Flare counts per HARP: NOAA number equal and peak inside the window."""
    wins = wins.copy()
    for col, thr in (("n_ge_c", "C"), ("n_ge_m", "M")):
        counts = []
        lo = {"C": 1e-6, "M": 1e-5}[thr]
        for r in wins.itertuples():
            if r.noaa_ar <= 0:
                counts.append(0)
                continue
            f = flares[(flares.noaa_ar == r.noaa_ar) & (flares.wm2 >= lo) &
                       (flares.peak >= r.t_start) & (flares.peak <= r.t_end)]
            counts.append(int(len(f)))
        wins[col] = counts
    wins["rate_c_per_day"] = wins.n_ge_c / wins.observed_days
    wins["rate_m_per_day"] = wins.n_ge_m / wins.observed_days
    return wins


def modal_classes(wins, srs):
    """Modal Mount Wilson and McIntosh class per HARP over its window days."""
    mw, mc = [], []
    for r in wins.itertuples():
        if r.noaa_ar <= 0 or srs.empty:
            mw.append("")
            mc.append("")
            continue
        s = srs[(srs.noaa_ar == r.noaa_ar) &
                (srs.date >= r.t_start.date()) & (srs.date <= r.t_end.date())]
        mw.append(s.mag_type.mode().iloc[0] if len(s) else "")
        mc.append(s.mcintosh.mode().iloc[0] if len(s) else "")
    wins["mount_wilson"], wins["mcintosh"] = mw, mc
    return wins


def cramers_v(a, b):
    """Cramer's V of two label series (rows with both nonempty)."""
    ok = (a != "") & (b.notna() if hasattr(b, "notna") else True)
    t = pd.crosstab(a[ok], b[ok])
    if t.size == 0 or min(t.shape) < 2:
        return float("nan"), int(ok.sum())
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(t, correction=False)[0]       # no Yates correction: V is an effect size
    n = t.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1)))), int(n)


def cmd_match(a):
    raw = os.path.join(ROOT, "data", "full", "c23_external", "raw")
    flares = pd.concat([parse_xrs(os.path.join(raw, f"goes-xrs-report_{y}.txt")) for y in (2010, 2011)],
                       ignore_index=True)
    srs = load_srs_dir(raw)
    wins = ar_windows(a.features)
    wins = match_flares(wins, flares)
    wins = modal_classes(wins, srs)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "ar_external.csv")
    wins.to_csv(path, index=False)
    print(f"{len(flares)} flares parsed ({(flares.wm2 >= 1e-6).sum()} >= C1, {(flares.wm2 >= 1e-5).sum()} >= M1), "
          f"{len(srs)} SRS rows, {len(wins)} ARs, {wins.n_ge_c.sum()} matched >= C1 flares")
    print(f"wrote {path}")


def cmd_validate(a):
    from scipy.stats import kruskal
    tag = f"_{a.tag}" if getattr(a, "tag", "") else ""
    ext = pd.read_csv(os.path.join(a.out_dir, "ar_external.csv"))
    lab = pd.read_csv(os.path.join(a.out_dir, f"clustering_{a.source}{tag}.csv"))
    m = lab.merge(ext, on="harpnum", how="left")
    out = {"source": a.source, "n": int(len(m))}
    for var in [c for c in lab.columns if c.startswith("label_")]:
        per = {}
        groups = [g.rate_c_per_day.dropna().to_numpy() for _, g in m.groupby(var)]
        rates = m.groupby(var).agg(n_ars=("harpnum", "size"), flares_c=("n_ge_c", "sum"),
                                   flares_m=("n_ge_m", "sum"), days=("observed_days", "sum"))
        rates["rate_c"] = rates.flares_c / rates.days
        rates["rate_m"] = rates.flares_m / rates.days
        per["clusters"] = json.loads(rates.to_json(orient="index"))
        pos = rates.rate_c[rates.rate_c > 0]
        per["rate_ratio_c"] = float(rates.rate_c.max() / pos.min()) if len(pos) and rates.rate_c.max() > 0 else float("nan")
        if sum(len(g) > 0 for g in groups) >= 2 and sum(len(g) for g in groups) > len(groups):
            try:
                per["kruskal_p_rate_c"] = float(kruskal(*[g for g in groups if len(g)]).pvalue)
            except ValueError:
                per["kruskal_p_rate_c"] = float("nan")
        v, n = cramers_v(m.mount_wilson.fillna(""), m[var])
        per["cramers_v_mount_wilson"], per["n_with_class"] = v, n
        out[var] = per
    path = os.path.join(a.out_dir, f"external_{a.source}{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: (v if not isinstance(v, dict) else {q: v[q] for q in ("rate_ratio_c", "kruskal_p_rate_c", "cramers_v_mount_wilson") if q in v})
                      for k, v in out.items()}, indent=1, default=str))
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mt = sub.add_parser("match")
    mt.add_argument("--features", default=os.path.join("results", "full", "overlap_pilot", "features.csv"))
    mt.add_argument("--out_dir", default=os.path.join("results", "full", "overlap_pilot"))
    mt.set_defaults(fn=cmd_match)
    vl = sub.add_parser("validate")
    vl.add_argument("--source", required=True)
    vl.add_argument("--out_dir", default=os.path.join("results", "full", "overlap_pilot"))
    vl.add_argument("--tag", default="")
    vl.set_defaults(fn=cmd_validate)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
