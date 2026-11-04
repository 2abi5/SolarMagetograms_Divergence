"""Cycle-23 build stage: SMARP deployment of the frozen models (reports/phase_13.md).

    python scripts/c23_build.py probe    --email you@example.com
    python scripts/c23_build.py frames   --out data/c23/frames.csv
    python scripts/c23_build.py download --email you@example.com --part 0/4
    (extract over the downloaded archive is added once the probe and download land)

probe:    the §6 implementation check: two TARPs (NOAA 10486 Halloween 2003; an ordinary 1999 one),
          a handful of frames each: geometry equivalence vs a stored training geometry_lr array,
          frozen-model inference, keywords, USFLUX-vs-USFLUXL rank gate, bytes/frame.
frames:   per-TARP keyword query over the NOAA-numbered TARPs -> the frame manifest
          (|LON_FWT| <= 65, QUALITY == 0, 96-min cadence), with the SMARP LOS keywords.
download: export the manifest's magnetograms (part k/n of the TARP list, one pending export
          per email, resumable by file existence) -> data/c23/raw/.
extract:  frozen models over the downloaded frames -> results/full/c23_build/features.csv
          (same wide schema as phase_12 E1; pred-mask conventions only — Cycle 23 has no truth).
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
from sr.geometry import los_coefficients  # noqa: E402

SERIES = "mdi.smarp_cea_96m"
SEGMENT = "magnetogram"
NORM = 3500.0
LON_MAX_C23 = 65.0
CATALOG = os.path.join(ROOT, "data", "full", "c23_external", "smarp_daily_catalog.csv")
C23_DIR = os.path.join(ROOT, "data", "c23")
KEYS = "TARPNUM,T_REC,NOAA_AR,LAT_FWT,LON_FWT,USFLUXL,MEANGBL,R_VALUE,AREA,QUALITY,CRLN_OBS,CRLT_OBS"


def wrap_deg(x):
    """Wrap degrees to [-180, 180)."""
    return (np.asarray(x, np.float64) + 180.0) % 360.0 - 180.0


def smarp_geometry(header, shape):
    """(3, ny, nx) float32 (c_p, c_t, c_r) from a SMARP CEA header: pure header arithmetic
    (CRLN/CRLT-CEA grid; dlon = CRLN_pixel - CRLN_OBS wrapped, lat = CRLT_pixel, b0 = CRLT_OBS)."""
    ny, nx = shape
    from astropy.wcs import WCS
    w = WCS(header)
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = w.wcs_pix2world(np.stack([xx.ravel(), yy.ravel()], 1), 0)
    crln = world[:, 0].reshape(ny, nx)
    crlt = world[:, 1].reshape(ny, nx)
    dlon = np.deg2rad(wrap_deg(crln - float(header["CRLN_OBS"])))
    lat = np.deg2rad(crlt)
    b0 = np.deg2rad(float(header["CRLT_OBS"]))
    return los_coefficients(dlon, lat, b0).astype(np.float32)


def fits_name(tarpnum, t_rec):
    t = str(t_rec).replace(".", "").replace(":", "").replace("-", "")
    return f"{SERIES}.{int(tarpnum)}.{t.replace('_TAI', '')}_TAI.{SEGMENT}.fits"


QUALITY_MASK = 0x80000000   # the training build's --mdi_quality_mask: drop only the no-data bit
                            # (MDI 96m carries QUALITY 512 routinely; training frames include it)


def select_frames(k, lon_max=LON_MAX_C23, quality_mask=QUALITY_MASK):
    """Manifest rows from a per-TARP keyword query: |LON_FWT| <= lon_max and no masked QUALITY bit
    (same rule as the training build, so deployment matches the training distribution)."""
    k = k.copy()
    q = k.QUALITY.astype(str).str.strip()
    is_hex = q.str.startswith("0x")
    k["QUALITY"] = pd.concat([pd.to_numeric(q[~is_hex], errors="coerce"),
                              q[is_hex].apply(lambda s: int(s, 16))]).reindex(k.index).fillna(-1).astype("int64")
    ok = (k.LON_FWT.abs() <= lon_max) & (k.QUALITY >= 0) & ((k.QUALITY & quality_mask) == 0)
    return k[ok].reset_index(drop=True)


def load_smarp_frame(path):
    """(lr_norm, geometry, header) for one SMARP magnetogram FITS: Gauss/NORM, NaN-kept."""
    import astropy.io.fits as fits
    with fits.open(path) as f:
        hdu = f[1] if len(f) > 1 and f[1].data is not None else f[0]
        data = np.asarray(hdu.data, np.float64)
        header = hdu.header
    geom = smarp_geometry(header, data.shape)
    return (data / NORM).astype(np.float32), geom, header


def model_input(lr_norm, geom):
    """(4, ny, nx) float32 network input: nan_to_num(MDI/NORM) + geometry channels."""
    return np.concatenate([np.nan_to_num(lr_norm)[None], geom]).astype(np.float32)


# ---------------------------------------------------------------- frames

def cmd_frames(a):
    import drms
    cat = pd.read_csv(CATALOG)
    tarps = sorted(cat[cat.NOAA_AR > 0].TARPNUM.unique())
    if a.max_tarps:
        tarps = tarps[:a.max_tarps]
    c = drms.Client()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows, done = [], 0
    for t in tarps:
        try:
            k = c.query(f"{SERIES}[{int(t)}][]", key=KEYS)
            rows.append(select_frames(k))
        except Exception as e:
            print(f"TARP {t} query failed: {str(e)[:80]}", flush=True)
        done += 1
        if done % 50 == 0:
            print(f"{done}/{len(tarps)} TARPs queried", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} frames, {df.TARPNUM.nunique()} TARPs")


# ---------------------------------------------------------------- sample

def stratified_tarp_sample(frames, budget_frames, seed=0):
    """Whole-TARP sample under a frame budget: strata = first-frame year x tercile of the TARP's
    peak USFLUXL (terciles over all TARPs); within each stratum TARPs are shuffled (fixed seed)
    and taken round-robin across strata until the budget is reached. Deterministic."""
    per = frames.groupby("TARPNUM").agg(n=("T_REC", "size"), peak=("USFLUXL", "max"),
                                        year=("T_REC", lambda s: int(str(s.iloc[0])[:4])))
    q1, q2 = per.peak.quantile([1 / 3, 2 / 3])
    per["tercile"] = np.where(per.peak <= q1, 0, np.where(per.peak <= q2, 1, 2))
    rng = np.random.default_rng(seed)
    strata = []
    for _, g in per.groupby(["year", "tercile"], sort=True):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        strata.append(list(idx))
    chosen, total = [], 0
    progressed = True
    while progressed and total < budget_frames:
        progressed = False
        for s in strata:
            if s and total < budget_frames:
                t = s.pop(0)
                chosen.append(t)
                total += int(per.loc[t, "n"])
                progressed = True
    return frames[frames.TARPNUM.isin(chosen)].reset_index(drop=True)


def cmd_sample(a):
    frames = pd.read_csv(a.frames_all)
    budget_frames = int(a.budget_gb * 1e9 / a.bytes_per_frame)
    out = stratified_tarp_sample(frames, budget_frames, seed=a.seed)
    out.to_csv(a.out, index=False)
    print(f"budget {a.budget_gb} GB @ {a.bytes_per_frame} B/frame -> {budget_frames} frames; "
          f"drew {out.TARPNUM.nunique()} TARPs, {len(out)} frames "
          f"({out.T_REC.str[:4].min()}-{out.T_REC.str[:4].max()})")
    print(f"wrote {a.out}")


# ---------------------------------------------------------------- download

def cmd_download(a):
    import drms
    import data_active_region_created as pipe
    from pipeline.build_dataset import JsocLock
    from reliability_pairs import download_with_retry, part_of
    raw = os.path.join(C23_DIR, "raw")
    os.makedirs(raw, exist_ok=True)
    frames = pd.read_csv(a.frames)
    k, n = (int(x) for x in a.part.split("/"))
    tarps = part_of(frames.TARPNUM, k, n)
    by_tarp = {}
    for r in frames[frames.TARPNUM.isin(tarps)].itertuples():
        if not os.path.exists(os.path.join(raw, fits_name(r.TARPNUM, r.T_REC))):
            by_tarp.setdefault(int(r.TARPNUM), []).append((int(r.TARPNUM), r.T_REC))
    print(f"part {k}/{n}: {len(tarps)} TARPs, {sum(len(v) for v in by_tarp.values())} frames to fetch ({a.email})",
          flush=True)
    if not by_tarp:
        return
    client = drms.Client(email=a.email)

    def export(tarp, pairs):
        pipe.batched_export_sharp(client, SERIES, pairs, raw, segments=(SEGMENT,),
                                  label=f"C23 part {k} TARP {tarp}")

    with JsocLock(os.path.join(ROOT, "data"), email=a.email):
        failures = download_with_retry(export, by_tarp, max_retries=a.max_retries, wait_s=a.wait_s,
                                       log=lambda m: print(f"  [C23 part {k}] {m}", flush=True))
    missing = [p for ps in by_tarp.values() for p in ps
               if not os.path.exists(os.path.join(raw, fits_name(p[0], p[1])))]
    print(f"part {k}/{n} done; {len(missing)} frames still missing; {len(failures)} TARP export(s) failed", flush=True)


# ---------------------------------------------------------------- extract

def c23_frame_rows(pred_gauss, strong_gauss=200.0):
    """Wide keyword rows for one Cycle-23 frame: pred-mask conventions only (no truth exists)."""
    from sr import sharp_keywords as sk
    pred = np.asarray(pred_gauss, np.float64)
    fin = np.isfinite(pred).all(axis=0)
    babs = np.sqrt(np.nansum(np.where(fin, pred, 0.0) ** 2, axis=0))
    rows = []
    for m, mk in (("all", fin), ("strong_pred", fin & (babs > strong_gauss))):
        rows.append({"mask": m, "n_finite": int(fin.sum()),
                     **sk.keywords(pred, mask=mk, cdelt_deg=0.03)})
    return rows


def cmd_extract(a):
    import torch
    import yaml
    from sr.evaluate import infer_region, load_model
    raw = os.path.join(C23_DIR, "raw")
    frames = pd.read_csv(a.frames)
    tarps = sorted(frames.TARPNUM.unique())[int(a.shard.split("/")[0])::int(a.shard.split("/")[1])]
    frames = frames[frames.TARPNUM.isin(tarps)].reset_index(drop=True)
    res_dir = os.path.join(ROOT, "results", "full")
    models = {}
    for e in a.exps:
        with open(os.path.join(res_dir, e, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        m, ep = load_model(res_dir, e, a.device)
        models[e] = (m, int(cfg["model"].get("kwargs", {}).get("in_channels", 1)), ep)
    os.makedirs(a.out_dir, exist_ok=True)
    rows, missing = [], 0
    for i, r in enumerate(frames.itertuples()):
        path = os.path.join(raw, fits_name(r.TARPNUM, r.T_REC))
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            lr, geom, _ = load_smarp_frame(path)
        except Exception as e:
            print(f"load failed {path}: {str(e)[:60]}", flush=True)
            continue
        x = torch.as_tensor(model_input(lr, geom))[None].to(a.device)
        meta = {"tarpnum": int(r.TARPNUM), "t_rec": r.T_REC, "noaa_ar": int(r.NOAA_AR),
                "lat_fwt": float(r.LAT_FWT), "lon_fwt": float(r.LON_FWT),
                "usfluxl": float(r.USFLUXL), "meangbl": float(r.MEANGBL),
                "r_value_los": float(r.R_VALUE), "area_los": float(r.AREA)}
        with torch.no_grad():
            preds = {e: infer_region(m, x)[0].cpu().numpy().astype(np.float64) * NORM
                     for e, (m, cin, _) in models.items()}
        preds["seed_mean"] = np.mean(list(preds.values()), 0)
        for name, pred in preds.items():
            for row in c23_frame_rows(pred, a.strong_gauss):
                rows.append({**meta, "model": name, **row})
        if i % 500 == 0:
            print(f"{i}/{len(frames)} frames", flush=True)
    k = a.shard.replace("/", "of")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, f"features_shard{k}.csv"), index=False)
    print(f"wrote shard {a.shard}: {len(df)} rows, {missing} frames missing")


def cmd_merge(a):
    import subprocess
    parts = sorted(f for f in os.listdir(a.out_dir) if f.startswith("features_shard") and f.endswith(".csv"))
    df = pd.concat([pd.read_csv(os.path.join(a.out_dir, p)) for p in parts], ignore_index=True)
    df = df.sort_values(["tarpnum", "t_rec", "model", "mask"]).reset_index(drop=True)
    df.to_csv(os.path.join(a.out_dir, "features.csv"), index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(os.path.join(a.out_dir, "extract_info.json"), "w") as f:
        json.dump({"n_rows": int(len(df)), "n_tarps": int(df.tarpnum.nunique()),
                   "n_frames": int(df[["tarpnum", "t_rec"]].drop_duplicates().shape[0]),
                   "shards": parts, "git_commit": commit,
                   "preregistration": "reports/phase_13.md"}, f, indent=1)
    print(f"wrote {a.out_dir}/features.csv ({len(df)} rows, {df.tarpnum.nunique()} TARPs)")


# ---------------------------------------------------------------- probe

def probe_geometry_equivalence(tol=0.01):
    """Max |difference| between the stored training geometry of one region and the header-arithmetic
    recipe applied to the same LR grid (observer Carrington position from sunpy at the frame time)."""
    import astropy.io.fits as afits
    import sunpy.coordinates.sun as sun
    from astropy.time import Time
    regs = pd.read_csv(os.path.join(ROOT, "data", "full", "regions.csv"))
    r = regs.iloc[0]
    npz = np.load(os.path.join(ROOT, "data", "full", r.region_file), allow_pickle=False)
    stored = np.load(os.path.join(ROOT, "data", "full", "geometry_lr",
                                  os.path.splitext(os.path.basename(r.region_file))[0] + ".npy"))
    header = afits.Header.fromstring(str(npz["lr_wcs_header"]))
    t = Time(str(npz["mdi_t_rec"]).replace("_TAI", "").replace(".", "-", 2).replace("_", "T"), scale="tai")
    # observer Carrington longitude: Earth's L0 plus the stored Stonyhurst offset of SOHO
    header = header.copy()
    header["CRLN_OBS"] = sun.L0(t).to_value("deg") + float(npz["mdi_obs_lon_deg"])
    header["CRLT_OBS"] = float(npz["mdi_obs_lat_deg"])
    mine = smarp_geometry(header, npz["lr"].shape)
    return float(np.max(np.abs(mine - stored))), stored.shape


def cmd_probe(a):
    import drms
    import torch
    import yaml
    from sr import sharp_keywords as sk
    from sr.evaluate import infer_region, load_model
    out_dir = os.path.join(C23_DIR, "probe")
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    diff, shape = probe_geometry_equivalence()
    report["geometry_max_abs_diff"] = diff
    print(f"geometry equivalence: max |diff| = {diff:.5f} on {shape} (tolerance 0.01)")
    c = drms.Client(email=a.email)
    cat = pd.read_csv(CATALOG)
    # between-region rank gate: one near-centre frame from each of ~12 NOAA TARPs spanning the
    # USFLUXL range (within-passage comparisons are confounded by centre-to-limb physics: the
    # reconstruction corrects viewing geometry, the LOS keyword does not)
    cm = cat[(cat.NOAA_AR > 0) & (cat.LON_FWT.abs() <= 20) & (cat.USFLUXL > 0)].copy()
    cm = cm.sort_values("USFLUXL")
    pick = cm.iloc[np.linspace(0, len(cm) - 1, a.n_tarps).astype(int)].drop_duplicates("TARPNUM")
    halloween = cat[(cat.NOAA_AR == 10486) & (cat.LON_FWT.abs() <= 30)]
    if len(halloween):
        pick = pd.concat([pick, halloween.iloc[[0]]]).drop_duplicates("TARPNUM")
    frames = []
    for r in pick.itertuples():
        k = c.query(f"{SERIES}[{int(r.TARPNUM)}][]", key=KEYS)
        sel = select_frames(k)
        if sel.empty:
            print(f"TARP {r.TARPNUM}: no usable frame")
            continue
        sel = sel.iloc[(sel.LON_FWT.abs()).argsort()].head(1)   # the most disk-centre frame
        frames.append(sel.assign(TARPNUM=int(r.TARPNUM)))
    frames = pd.concat(frames, ignore_index=True)
    print(f"probe frames: {len(frames)} TARPs, USFLUXL span {frames.USFLUXL.min():.2e}..{frames.USFLUXL.max():.2e}")
    import data_active_region_created as pipe
    from reliability_pairs import download_with_retry
    pairs = {int(t): [(int(t), r.T_REC) for r in g.itertuples()] for t, g in frames.groupby("TARPNUM")}
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    todo = {t: [p for p in ps if not os.path.exists(os.path.join(raw, fits_name(p[0], p[1])))]
            for t, ps in pairs.items()}
    todo = {t: ps for t, ps in todo.items() if ps}
    if todo:
        download_with_retry(lambda tarp, ps: pipe.batched_export_sharp(
            c, SERIES, ps, raw, segments=(SEGMENT,), label=f"C23 probe {tarp}"),
            todo, max_retries=6, wait_s=120, log=lambda m: print(f"  [probe] {m}", flush=True))
    res_dir = os.path.join(ROOT, "results", "full")
    models = {}
    for e in a.exps:
        with open(os.path.join(res_dir, e, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        m, ep = load_model(res_dir, e, a.device)
        models[e] = (m, int(cfg["model"].get("kwargs", {}).get("in_channels", 1)), ep)
    rows, sizes = [], []
    for r in frames.itertuples():
        path = os.path.join(raw, fits_name(r.TARPNUM, r.T_REC))
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        sizes.append(os.path.getsize(path))
        lr, geom, _ = load_smarp_frame(path)
        x = torch.as_tensor(model_input(lr, geom))[None].to(a.device)
        preds = []
        with torch.no_grad():
            for e, (m, cin, _) in models.items():
                pred = infer_region(m, x)[0].cpu().numpy().astype(np.float64) * NORM
                preds.append(pred)
        pred = np.mean(preds, 0)
        fin = np.isfinite(pred).all(axis=0)
        babs = np.sqrt(np.nansum(np.where(fin, pred, 0.0) ** 2, axis=0))
        kw = sk.keywords(pred, mask=fin & (babs > 200.0))
        rows.append({"tarpnum": int(r.TARPNUM), "t_rec": r.T_REC, "usfluxl_smarp": float(r.USFLUXL),
                     "lr_ny": lr.shape[0], "lr_nx": lr.shape[1], **{q: kw[q] for q in sk.KEYWORDS},
                     "n_pixels": kw["n_pixels"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "probe_keywords.csv"), index=False)
    ok = df[(df.USFLUX > 0) & (df.usfluxl_smarp > 0)]
    rho = float(pd.Series(np.log10(ok.USFLUX)).corr(pd.Series(np.log10(ok.usfluxl_smarp)), method="spearman"))
    report.update({"n_frames": int(len(df)), "bytes_per_frame_mean": float(np.mean(sizes)),
                   "spearman_usflux_vs_usfluxl": rho,
                   "models": {e: ep for e, (_, _, ep) in models.items()},
                   "gates": {"geometry": diff <= 0.01, "rank": bool(rho >= 0.8)}})
    with open(os.path.join(out_dir, "probe_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(report, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--email", required=True)
    pr.add_argument("--device", default="cuda:1")
    pr.add_argument("--n_tarps", type=int, default=12)
    pr.add_argument("--exps", nargs="+", default=["F_U_E3_s0_e200", "F_U_E3_s1_e200", "F_U_E3_s2"])
    pr.set_defaults(fn=cmd_probe)
    fr = sub.add_parser("frames")
    fr.add_argument("--out", default=os.path.join(C23_DIR, "frames.csv"))
    fr.add_argument("--max_tarps", type=int, default=0)
    fr.set_defaults(fn=cmd_frames)
    sm = sub.add_parser("sample")
    sm.add_argument("--frames_all", default=os.path.join(C23_DIR, "frames_all.csv"))
    sm.add_argument("--out", default=os.path.join(C23_DIR, "frames.csv"))
    sm.add_argument("--budget_gb", type=float, default=12.0)
    sm.add_argument("--bytes_per_frame", type=float, default=120e3)
    sm.add_argument("--seed", type=int, default=0)
    sm.set_defaults(fn=cmd_sample)
    dl = sub.add_parser("download")
    dl.add_argument("--email", required=True)
    dl.add_argument("--part", default="0/4")
    dl.add_argument("--frames", default=os.path.join(C23_DIR, "frames.csv"))
    dl.add_argument("--max_retries", type=int, default=6)
    dl.add_argument("--wait_s", type=int, default=600)
    dl.set_defaults(fn=cmd_download)
    ex = sub.add_parser("extract")
    ex.add_argument("--device", default="cuda:1")
    ex.add_argument("--shard", default="0/6")
    ex.add_argument("--frames", default=os.path.join(C23_DIR, "frames.csv"))
    ex.add_argument("--out_dir", default=os.path.join(ROOT, "results", "full", "c23_build"))
    ex.add_argument("--strong_gauss", type=float, default=200.0)
    ex.add_argument("--exps", nargs="+", default=["F_U_E3_s0_e200", "F_U_E3_s1_e200", "F_U_E3_s2"])
    ex.set_defaults(fn=cmd_extract)
    mg = sub.add_parser("merge")
    mg.add_argument("--out_dir", default=os.path.join(ROOT, "results", "full", "c23_build"))
    mg.set_defaults(fn=cmd_merge)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
