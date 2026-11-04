"""Phase 13 / D1: assemble the dataset release-candidate documentation (reports/phase_13.md 3).

    python scripts/c23_dataset_docs.py build

Generates results/full/c23_build/DATASET.md from the existing artifacts: data dictionary,
per-column reliability/agreement screening tables, per-track gap statistics (a frozen plan
rule), physics-validation summary, provenance, and completeness statement. Numbers are read
from the result files, never retyped.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUT = os.path.join(ROOT, "results", "full", "c23_build")
PILOT = os.path.join(ROOT, "results", "full", "overlap_pilot")


def gap_stats(features_csv):
    """Per-track coverage: observed frames / expected frames at 96-min cadence over the
    track's observed span (|lon| window). Returns the per-track frame and summary dicts."""
    df = pd.read_csv(features_csv, usecols=["tarpnum", "t_rec", "model", "mask"])
    f = df[(df.model == "seed_mean") & (df["mask"] == "strong_pred")]
    t = pd.to_datetime(f.t_rec.str.replace("_TAI", "", regex=False), format="%Y.%m.%d_%H:%M:%S")
    g = pd.DataFrame({"tarpnum": f.tarpnum, "t": t}).groupby("tarpnum").agg(
        n=("t", "size"), span_h=("t", lambda s: (s.max() - s.min()).total_seconds() / 3600.0))
    g["expected"] = (g.span_h / 1.6).round() + 1
    g["coverage"] = (g.n / g.expected).clip(upper=1.0)
    s = {"n_tracks": int(len(g)),
         "coverage_median": float(g.coverage.median()),
         "coverage_p10": float(g.coverage.quantile(0.10)),
         "tracks_below_80pct": int((g.coverage < 0.8).sum()),
         "frames_total": int(g.n.sum())}
    return g.reset_index(), s


def cmd_build(a):
    rel = pd.read_csv(os.path.join(PILOT, "keyword_reliability.csv"))
    rel_i = pd.read_csv(os.path.join(PILOT, "keyword_reliability_intensive.csv"))
    f1 = pd.read_csv(os.path.join(OUT, "f1_reliability.csv"))
    with open(os.path.join(OUT, "f5_physics.json")) as f:
        f5 = json.load(f)
    with open(os.path.join(OUT, "extract_info.json")) as f:
        info = json.load(f)
    g, gs = gap_stats(os.path.join(OUT, "features.csv"))
    g.to_csv(os.path.join(OUT, "track_gap_stats.csv"), index=False)

    def table(df, cols, fmt="%.3f"):
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            cells = [(fmt % v if isinstance(v, float) and np.isfinite(v) else str(v)) for v in r[cols]]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    md = f"""# A Vector-Derived Active-Region Keyword Record for Solar Cycle 23 (release candidate)

Reconstructed SHARP-like physics keywords for {gs['n_tracks']} NOAA-numbered active regions,
1996–2010, built by applying a frozen MDI→HMI super-resolution model (U-Net, divergence-regularized;
AISTATS 2027 submission) to `mdi.smarp_cea_96m` magnetograms at 96-minute cadence, with the SMARP
LOS keywords carried per frame as the built-in comparison column.

## Files
- `features.csv` — {info['n_rows']:,} rows: one row per (frame × model × mask). Models: three
  training seeds and their mean (`seed_mean`, the reference column). Masks: `all` and
  `strong_pred` (|B_pred| > 200 G — the deployment convention; no truth exists for Cycle 23).
- `track_gap_stats.csv` — per-track coverage at 96-min cadence (SOHO gaps, QUALITY exclusions).
- Companion keyword files: `c24_los_keywords.csv` (SHARP-era LOS twins), `data/c23/frames_all.csv`
  (full SMARP LOS record, 2,195 TARPs — population frame of this sample).

## What the columns are good for — measured, not asserted
Screening on the 233-region MDI/HMI overlap sample (test–retest repeats + agreement with HMI truth;
pre-registered in reports/phase_12.md §4):

{table(rel, ['keyword', 'r12_real', 'agreement_ours', 'snr2_evo', 'included'])}

Intensive (size-free) candidates — **the α-like ratios are NOT recovered by the reconstruction**
(reliable in real data, agreement below the 0.70 bar): use the current/twist columns to rank
regions, not to measure twist-per-flux.

{table(rel_i, ['keyword', 'r12_real', 'agreement_ours', 'included'])}

## Temporal reliability of this record (lag bracket, {int(f1[f1.lag_min == 96].n_pairs.iloc[0]):,} pairs)
{table(f1[f1.lag_min == 96], ['keyword', 'r_c23', 'r_lo', 'r_hi', 'r_c24_ours'])}

## Physics-law validation (reports/phase_13.md §9.7)
- Flux decay law: exponent {f5['decay']['c23_recon']['exponent']:.3f} [{f5['decay']['c23_recon']['ci'][0]:.3f}, {f5['decay']['c23_recon']['ci'][1]:.3f}] ({f5['decay']['c23_recon']['n_regions']} regions) vs HMI-truth {f5['decay']['c24_true']['exponent']:.3f} [{f5['decay']['c24_true']['ci'][0]:.3f}, {f5['decay']['c24_true']['ci'][1]:.3f}].
- Flux–flare scaling: slope {f5['flux_flare']['c23_recon']['slope']:.3f} [{f5['flux_flare']['c23_recon']['ci'][0]:.3f}, {f5['flux_flare']['c23_recon']['ci'][1]:.3f}] ({f5['flux_flare']['c23_recon']['n_flaring_regions']} flaring regions) vs truth {f5['flux_flare']['c24_true']['slope']:.3f}.
- Butterfly diagram: figures/f5b_butterfly.png.

## Coverage and gaps
{gs['frames_total']:,} frames; median track coverage {gs['coverage_median']:.2f} of the 96-min grid
(10th percentile {gs['coverage_p10']:.2f}); {gs['tracks_below_80pct']} of {gs['n_tracks']} tracks
below 80 % (SOHO 1998 contact loss, keyhole periods, QUALITY mask, one JSOC outage window during
the build — completeness pass scheduled before the DOI freeze).

## Provenance
- Checkpoints: converged Phase-11 set (F_U_E3 s0/s1 at epoch-200 continuation, s2 original).
- Extraction commit: `{info.get('git_commit', 'n/a')}`; pre-registrations: reports/phase_12.md, phase_13.md.
- Known limits: signed/directional quantities are not recovered and are excluded; MEANGAM degrades;
  intensive ratios not recovered (above); claims should stay rank-based and within-cycle.
- License and DOI: to be assigned at release (Zenodo); contact the authors.
"""
    path = os.path.join(OUT, "DATASET.md")
    with open(path, "w") as f:
        f.write(md)
    print(f"wrote {path} and track_gap_stats.csv ({gs})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.set_defaults(fn=cmd_build)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
