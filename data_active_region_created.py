"""
data_set_creation.py

Builds a paired MDI (low-res, scalar LOS) / HMI SHARP (high-res, 3-channel
vector) magnetogram patch dataset for Step 1 of the physics-guided
super-resolution project (MDI -> SHARP), where "Step 1" now means the
version of MDI->HMI that actually supports a real divergence loss.

WHY THIS VERSION EXISTS (read before changing --hmi_series back)
------------------------------------------------------------------
The previous version of this script paired MDI against hmi.M_720s, HMI's
plain line-of-sight magnetogram. That product is a single scalar per pixel
(the field component along the line of sight). The divergence operator

    div(B) = dBx/dx + dBy/dy + dBz/dz

needs three independently measured field components. There is no way to
compute a real divergence from a single scalar channel -- differentiating
one scalar field in x and y and calling it "divergence" is not the physics,
it's just two partial derivatives of the same number.

hmi.sharp_cea_720s is HMI's vector-field product: three independently
measured, co-registered components (Bp, Bt, Br -- the in-plane and radial
field in a local Lambert cylindrical equal-area (CEA) projection centered
on each active region). That is genuine vector data, so a divergence loss
computed on it is real physics, not a proxy.

The trade-off: SHARP is patch-based (one cutout per tracked active region),
not full-disk. You lose full-disk / quiet-Sun coverage. There is no HMI
product that is simultaneously full-disk AND vector, so this trade is
unavoidable if the divergence loss needs to be real.

WHAT CHANGED vs the hmi.M_720s version
------------------------------------------------------------------
  1. --hmi_series default -> hmi.sharp_cea_720s.
  2. Querying and export now carry HARPNUM alongside T_REC, and the
     export requests three segments (Br, Bt, Bp) per record instead of
     one ("magnetogram").
  3. align_pair() now builds its target grid from the SHARP map's own
     native CEA WCS (local heliographic projection, degrees), and
     reprojects the full-disk MDI map (helioprojective, arcsec) DOWN
     onto a coarsened version of that same CEA grid. This is the
     opposite direction from the old version, which built a synthetic
     upscaled grid from MDI and reprojected HMI onto it -- that only
     worked because both were full-disk frames on a shared projection.
     SHARP tiles are not full-disk and are not in that projection, so
     the target grid now has to come from SHARP, not MDI.
  4. disk_mask() is removed. It assumed full-disk geometry (rsun_obs,
     pixel-to-world across the whole solar disk), which doesn't apply
     to an active-region cutout. SHARP already pads everything outside
     the tracked region as NaN, so validity is handled by the existing
     per-patch min_finite_fraction check in extract_patches() instead
     of a separate whole-frame mask.
  5. The HR side is now a (3, H, W) stack (Bp, Bt, Br) instead of a
     single 2D array. extract_patches() and the .npz save both handle
     this. Channel order is kept as (Bp, Bt, Br) throughout so that a
     downstream 2D in-plane divergence loss can treat channels 0 and 1
     as the Bx/By analogues (Br, the radial/vertical component, has no
     height information in a single photospheric slice anyway, so only
     the in-plane divergence dBp/dx + dBt/dy is physically computable
     from this data -- that's a training-loop decision, not something
     this script needs to resolve).

Usage:
    python data_set_creation.py \
        --email you@example.com \
        --start 2010-06-01T00:00:00 \
        --end   2010-06-02T00:00:00 \
        --out_dir ./dataset_sharp_day1 \
        --scale_factor 4 \
        --patch_size_lr 16 \
        --max_time_offset_minutes 6 \
        --max_frames 3

Requirements:
    pip install drms sunpy astropy numpy scikit-image reproject
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import requests

import drms
import sunpy.coordinates
import sunpy.map
from astropy import units as u
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy.ndimage import map_coordinates, spline_filter


# --------------------------------------------------------------------------
# Querying and matching
# --------------------------------------------------------------------------

def query_mdi_series(client, series, t_start, t_end, quality_mask=None):
    """Unchanged from the scalar version: MDI has no HARPNUM concept.

    quality_mask=None keeps the original rule (QUALITY == 0 if any, else all).
    With a mask, frames with any masked QUALITY bit set are dropped instead;
    0x80000000 is MDI's "no data" bit. (Most 2010-11 MDI 96m frames have
    QUALITY 512, so the original rule either drops them or keeps no-data frames.)
    """
    t_start_str = t_start.strftime("%Y.%m.%d_%H:%M:%S")
    t_end_str = t_end.strftime("%Y.%m.%d_%H:%M:%S")
    query = f"{series}[{t_start_str}-{t_end_str}]"
    keys = client.query(query, key="T_REC, QUALITY")
    if keys is None or len(keys) == 0:
        return []
    if "QUALITY" in keys.columns and quality_mask is not None:
        keys = keys[(keys["QUALITY"].astype("int64") & int(quality_mask)) == 0]
    elif "QUALITY" in keys.columns:
        good = keys[keys["QUALITY"] == 0]
        if len(good) > 0:
            keys = good
    return list(keys["T_REC"])


def filter_window(t_recs, t_start, t_end):
    """Keep T_RECs inside [t_start, t_end]. JSOC rounds range ends to the
    series' time slots, so a query ending 23:59:59 also returns next-day
    00:00; without this, day-by-day builds would process that frame twice."""
    return [t for t in t_recs if t_start <= parse_jsoc_time(t) <= t_end]


SHARP_META_KEYS = ("NOAA_AR", "LON_FWT", "LAT_FWT", "CRVAL1", "CRVAL2", "CRLN_OBS", "CRLT_OBS")


def query_sharp_series(client, series, t_start, t_end, with_meta=False):
    """SHARP records carry a HARPNUM (one per tracked active region) in
    addition to T_REC. A single time window can contain several HARPs
    simultaneously, so we return (t_rec, harpnum) pairs, not just times.

    with_meta=True also fetches SHARP_META_KEYS in the same query and returns
    (records, meta), where meta maps (harpnum, t_rec) -> {key: value}.
    """
    t_start_str = t_start.strftime("%Y.%m.%d_%H:%M:%S")
    t_end_str = t_end.strftime("%Y.%m.%d_%H:%M:%S")
    # "[]" for the HARPNUM slot means "all HARPs active in this window".
    query = f"{series}[][{t_start_str}-{t_end_str}]"
    key_str = "T_REC, HARPNUM, QUALITY"
    if with_meta:
        key_str += ", " + ", ".join(SHARP_META_KEYS)
    keys = client.query(query, key=key_str)
    if keys is None or len(keys) == 0:
        return ([], {}) if with_meta else []
    if "QUALITY" in keys.columns:
        good = keys[keys["QUALITY"] == 0]
        if len(good) > 0:
            keys = good
    records = [(str(t), int(h)) for t, h in zip(keys["T_REC"], keys["HARPNUM"])]
    if not with_meta:
        return records
    meta = {}
    for i, (t, h) in enumerate(records):
        row = keys.iloc[i]
        meta[(h, t)] = {k: _to_float(row[k]) if k in keys.columns else float("nan")
                        for k in SHARP_META_KEYS}
    return records, meta


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def parse_jsoc_time(t_rec_str):
    core = t_rec_str.replace("_TAI", "").replace("_UTC", "")
    return datetime.strptime(core, "%Y.%m.%d_%H:%M:%S")


def find_nearest_sharp_matches(mdi_time, sharp_records, max_offset_minutes):
    """For one MDI timestamp, find the nearest SHARP record independently
    for EVERY HARPNUM present in the query window. A single MDI frame can
    therefore produce zero, one, or several matched pairs (one per active
    region visible at that time).
    """
    by_harp = {}
    for t_rec, harpnum in sharp_records:
        by_harp.setdefault(harpnum, []).append(t_rec)

    matches = []
    for harpnum, t_recs in by_harp.items():
        best, best_offset = None, None
        for t_rec in t_recs:
            t = parse_jsoc_time(t_rec)
            offset = abs((t - mdi_time).total_seconds()) / 60.0
            if best_offset is None or offset < best_offset:
                best_offset, best = offset, t_rec
        if best is not None and best_offset <= max_offset_minutes:
            matches.append((harpnum, best, best_offset))
    return matches


# --------------------------------------------------------------------------
# Batched download
# --------------------------------------------------------------------------

def batched_export_mdi(client, series, t_recs, out_dir, label="MDI",
                        poll_interval=10, max_wait_minutes=30):
    """Unchanged in spirit from the scalar version: one export, one segment
    ("data"), keyed by T_REC only.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not t_recs:
        return {}

    record_list = ",".join(t_recs)
    record_set = f"{series}[{record_list}]{{data}}"

    print(f"  [{label}] Submitting one batched export for {len(t_recs)} frames ...")
    r = _submit_and_wait(client, record_set, label, max_wait_minutes)

    if r.urls is None or len(r.urls) == 0:
        print(f"  [{label}] Export finished but returned no files.")
        return {}

    t_rec_pattern = re.compile(r"\[([0-9\.\_\:]+_TAI)\]")
    mapping = {}
    for _, row in r.urls.iterrows():
        url = row["url"]
        record_str = row.get("record", "")
        match = t_rec_pattern.search(str(record_str))
        t_rec = match.group(1) if match else None

        filename = os.path.basename(url)
        filepath = os.path.join(out_dir, filename)

        if not os.path.exists(filepath):
            try:
                resp = requests.get(url, timeout=180)
                resp.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                print(f"  [{label}] Failed to download {url}: {e}")
                continue

        if t_rec:
            mapping[t_rec] = filepath
        else:
            mapping[filename] = filepath

    print(f"  [{label}] Downloaded {len(mapping)} files.")
    return mapping


def _submit_and_wait(client, record_set, label, max_wait_minutes):
    """Shared export/wait logic with a clearer message on the JSOC
    one-pending-export-per-user limit, and on a bare, message-less
    export failure (JSOC status=4), which usually means the record-set
    string itself was rejected rather than anything about staging.
    """
    r = client.export(record_set, method="url", protocol="fits")
    print(f"  [{label}] Export submitted (id={r.id}). Waiting for JSOC to stage it ...")
    try:
        r.wait(timeout=max_wait_minutes * 60)
    except drms.exceptions.DrmsExportError as e:
        msg = str(e).strip()
        if "pending export" in msg.lower():
            print(f"\n  [{label}] JSOC says you already have a pending export "
                  f"under this email. JSOC only allows ONE outstanding export "
                  f"per registered address at a time -- this is very likely "
                  f"an earlier export (from this run or an interrupted one) "
                  f"still not fully closed out on JSOC's end.\n"
                  f"  Fix: wait 10-15 minutes for it to clear, or check its "
                  f"status at http://jsoc.stanford.edu/ajax/lookdata.html "
                  f"before re-running. Avoid launching the script twice while "
                  f"an earlier run is still polling.\n")
        elif not msg:
            print(f"\n  [{label}] JSOC rejected this export with no error "
                  f"message (a bare status=4). This almost always means the "
                  f"record-set string itself was malformed -- e.g. an "
                  f"unrecognized segment name, or a HARPNUM/T_REC JSOC has no "
                  f"data for. The record-set string submitted was:\n"
                  f"    {record_set}\n"
                  f"  Worth checking by hand at "
                  f"http://jsoc.stanford.edu/ajax/lookdata.html -- paste this "
                  f"same string into the 'RecordSet' field there and see what "
                  f"error (if any) the web UI reports; it's often more "
                  f"specific than the API is.\n")
        raise
    return r


def batched_export_sharp(client, series, harp_trec_pairs, out_dir,
                          segments=("Bp", "Bt", "Br"), label="SHARP",
                          max_wait_minutes=30):
    """Submit ONE export PER HARPNUM (all of that HARP's matched T_RECs
    listed together in one bracket group: series[harpnum][t1,t2,...]),
    submitted sequentially. Returns a dict keyed by
    (harpnum, t_rec) -> {"Bp": path, "Bt": path, "Br": path}.

    Earlier versions of this function tried to chain every (harpnum,
    t_rec) pair into ONE compound record-set string
    (series[h1][t1]{...},series[h2][t2]{...},...) to keep it to a single
    export call. That produced a bare, message-less JSOC failure
    (status=4) in practice -- the compound-record-set comma-chaining
    JSOC documents is not reliably honored this way across series and
    segment filters. Grouping by HARPNUM and listing that HARP's times
    in one bracket is exactly the pattern shown in JSOC's own drms
    tutorial examples, so it's the safer bet even though it means one
    export call per HARP instead of one overall. Submitting them
    sequentially (each wait() finishes before the next export() fires)
    also means this never collides with JSOC's one-pending-export-per-
    user limit.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not harp_trec_pairs:
        return {}

    seg_str = ",".join(segments)

    # Parse from the downloaded FILENAME, not the "record" metadata column.
    # In practice, JSOC's returned "record" string for this series comes
    # back as e.g. "hmi.sharp_cea_720s[36][2010.06.01_00:00:00_TAI]" with
    # no {segment} tag at all, even though three different per-segment
    # files were exported for that same record. The segment is only
    # recoverable from the filename, which reliably follows:
    #   {series}.{harpnum}.{YYYYMMDD}_{HHMMSS}_TAI.{segment}.fits
    filename_pattern = re.compile(
        rf"^{re.escape(series)}\.(\d+)\.(\d{{8}}_\d{{6}}_TAI)\.(\w+)\.fits$"
    )

    def _filename_time_to_trec(compact_time):
        # "20100601_000000_TAI" -> "2010.06.01_00:00:00_TAI"
        date_part, time_part, tai = compact_time.split("_")
        y, mo, d = date_part[:4], date_part[4:6], date_part[6:8]
        hh, mm, ss = time_part[:2], time_part[2:4], time_part[4:6]
        return f"{y}.{mo}.{d}_{hh}:{mm}:{ss}_{tai}"

    by_harp = {}
    for harpnum, t_rec in harp_trec_pairs:
        by_harp.setdefault(harpnum, set()).add(t_rec)

    mapping = {}
    for harp_idx, (harpnum, t_rec_set) in enumerate(sorted(by_harp.items()), start=1):
        t_rec_list = ",".join(sorted(t_rec_set))
        record_set = f"{series}[{harpnum}][{t_rec_list}]{{{seg_str}}}"

        print(f"  [{label}] HARP {harpnum} ({harp_idx}/{len(by_harp)}): "
              f"submitting export for {len(t_rec_set)} time(s) x "
              f"{len(segments)} segments ...")
        r = _submit_and_wait(client, record_set, f"{label} HARP {harpnum}",
                              max_wait_minutes)

        if r.urls is None or len(r.urls) == 0:
            print(f"  [{label}] HARP {harpnum}: export finished but returned no files.")
            continue

        for _, row in r.urls.iterrows():
            url = row["url"]
            filename = os.path.basename(url)
            filepath = os.path.join(out_dir, filename)

            if not os.path.exists(filepath):
                try:
                    resp = requests.get(url, timeout=180)
                    resp.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                except Exception as e:
                    print(f"  [{label}] Failed to download {url}: {e}")
                    continue

            m = filename_pattern.match(filename)
            if not m:
                print(f"  [{label}] Warning: filename '{filename}' didn't match "
                      f"the expected pattern for series '{series}'. Skipping "
                      f"this file for pairing purposes.")
                continue

            parsed_harp = int(m.group(1))
            t_rec = _filename_time_to_trec(m.group(2))
            segment = m.group(3)
            key = (parsed_harp, t_rec)
            mapping.setdefault(key, {})[segment] = filepath

    n_complete = sum(1 for v in mapping.values() if set(v.keys()) >= set(segments))
    print(f"  [{label}] Downloaded files for {len(mapping)} (HARP, time) "
          f"records total; {n_complete} have all {len(segments)} segments.")
    return mapping


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------

def load_map_safely(path):
    try:
        return sunpy.map.Map(path)
    except Exception as e:
        raise RuntimeError(f"Could not load {path} as a SunPy Map: {e}")


def align_pair(mdi_map, sharp_maps, scale_factor, lr_grid_fix=False, return_wcs=False):
    """Build the HR stack directly from the SHARP maps' own native CEA
    grid (Bp, Bt, Br, already co-registered with each other by JSOC), then
    reproject the full-disk MDI map DOWN onto a coarsened version of that
    same grid to produce the LR side.

    This is the reverse of the old full-disk version, where the target
    grid was synthesized from MDI and HMI was reprojected up onto it. That
    only worked because both MDI and HMI were full-disk frames sharing a
    compatible projection. SHARP's CEA grid is local to one active region
    and in a different projection (local heliographic, not helioprojective),
    so the target has to be defined by SHARP; MDI is the one being
    reprojected here.

    Note: reprojecting between MDI's helioprojective WCS and SHARP's
    heliographic CEA WCS requires reproject/sunpy to resolve the frame
    transform via each map's embedded observer metadata. This works with
    sunpy Map-derived WCS objects but has not been exercised against live
    JSOC data in this rewrite -- confirm on your first real batch before
    trusting the alignment blindly.

    lr_grid_fix=True centres each LR pixel on its HR block (see lr_crpix);
    the default keeps the original CRPIX/scale_factor. return_wcs=True also
    returns the LR WCS as a third value.
    """
    ref_map = sharp_maps["Br"]
    ny_hr, nx_hr = ref_map.data.shape

    if ny_hr < scale_factor or nx_hr < scale_factor:
        raise ValueError(
            f"SHARP cutout ({ny_hr}x{nx_hr}) is smaller than scale_factor "
            f"({scale_factor}); cannot build an LR grid from it.")

    ny_lr, nx_lr = ny_hr // scale_factor, nx_hr // scale_factor
    ny_hr_crop, nx_hr_crop = ny_lr * scale_factor, nx_lr * scale_factor

    # Coarsened copy of SHARP's own WCS to use as the LR target grid.
    lr_header = ref_map.wcs.to_header()
    lr_header["NAXIS1"] = nx_lr
    lr_header["NAXIS2"] = ny_lr
    lr_header["CRPIX1"] = lr_crpix(ref_map.wcs.wcs.crpix[0], scale_factor, lr_grid_fix)
    lr_header["CRPIX2"] = lr_crpix(ref_map.wcs.wcs.crpix[1], scale_factor, lr_grid_fix)
    lr_header["CDELT1"] = ref_map.wcs.wcs.cdelt[0] * scale_factor
    lr_header["CDELT2"] = ref_map.wcs.wcs.cdelt[1] * scale_factor
    lr_wcs = WCS(lr_header)

    mdi_reprojected, _ = reproject_interp(
        (mdi_map.data, mdi_map.wcs), lr_wcs, shape_out=(ny_lr, nx_lr)
    )

    # Stack in a fixed channel order: (Bp, Bt, Br).
    hr_stack = np.stack(
        [sharp_maps["Bp"].data, sharp_maps["Bt"].data, sharp_maps["Br"].data],
        axis=0,
    )
    hr_stack = hr_stack[:, :ny_hr_crop, :nx_hr_crop]

    if return_wcs:
        return mdi_reprojected, hr_stack, lr_wcs
    return mdi_reprojected, hr_stack


# --------------------------------------------------------------------------
# Registration checks, limb filter, region files (optional; all opt-in)
# --------------------------------------------------------------------------

def lr_crpix(crpix_hr, scale_factor, block_centred=False):
    """FITS (1-based) CRPIX of the LR grid obtained by binning the HR grid.

    The original formula, crpix_hr / s, places LR pixel J at HR pixel s*J,
    i.e. (s-1)/2 HR pixels past the centre of the s x s block it is paired
    with. block_centred=True uses (crpix_hr - 0.5)/s + 0.5, which puts it on
    the block centre.
    """
    if block_centred:
        return (crpix_hr - 0.5) / scale_factor + 0.5
    return crpix_hr / scale_factor


def los_projection(bp, bt, br, dlon, lat, b0):
    """Project a SHARP vector (Bp westward, Bt SOUTHward, Br radial) onto the
    line of sight of an observer at heliographic latitude b0. dlon is the
    pixel's longitude relative to the observer's central meridian and lat its
    latitude, all in radians. Positive = toward the observer.
    """
    cl, sl = np.cos(lat), np.sin(lat)
    cd, sd = np.cos(dlon), np.sin(dlon)
    cb, sb = np.cos(b0), np.sin(b0)
    return (br * (cl * cd * cb + sl * sb)
            - bp * (sd * cb)
            - bt * (-sl * cd * cb + cl * sb))


def center_angle_deg(lon, lat, b0=0.0):
    """Heliocentric angle (degrees) between a point at Stonyhurst (lon, lat)
    and disk centre for an observer at latitude b0 (all degrees)."""
    lon, lat, b0 = np.deg2rad(lon), np.deg2rad(lat), np.deg2rad(b0)
    point = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    observer = np.array([np.cos(b0), 0.0, np.sin(b0)])
    # atan2(|a x b|, a.b) stays accurate near 0 deg, unlike arccos(a.b)
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(point, observer)),
                                       np.dot(point, observer))))


def region_center(meta):
    """(lon, lat, source) of a SHARP region centre in Stonyhurst degrees.

    Uses the flux-weighted centre LON_FWT/LAT_FWT when present. Some HARPs
    (e.g. weak plage) have it empty; then fall back to the CEA reference
    point: lon = CRVAL1 - CRLN_OBS (wrapped to [-180, 180)), lat = CRVAL2.
    """
    lon, lat = _to_float(meta.get("LON_FWT")), _to_float(meta.get("LAT_FWT"))
    if np.isfinite(lon) and np.isfinite(lat):
        return lon, lat, "fwt"
    lon = (_to_float(meta.get("CRVAL1")) - _to_float(meta.get("CRLN_OBS")) + 180.0) % 360.0 - 180.0
    return lon, _to_float(meta.get("CRVAL2")), "crval"


def block_mean(a, s):
    """Mean over non-overlapping s x s blocks (NaN if any pixel is NaN)."""
    ny, nx = a.shape[0] // s, a.shape[1] // s
    return a[:ny * s, :nx * s].reshape(ny, s, nx, s).mean(axis=(1, 3))


def pearson_finite(a, b, min_pixels=10):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_pixels or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def sharp_los_lr(sharp_maps, observer, scale_factor):
    """SHARP vector projected onto the MDI line of sight, averaged over
    scale_factor x scale_factor HR blocks (the comparison target for MDI).
    Geometry is evaluated at the block centres."""
    ref = sharp_maps["Br"]
    s = scale_factor
    bp, bt, br = (block_mean(sharp_maps[c].data.astype(float), s) for c in ("Bp", "Bt", "Br"))
    ny, nx = br.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    hgs = ref.wcs.pixel_to_world(s * xx + (s - 1) / 2.0, s * yy + (s - 1) / 2.0).transform_to(
        sunpy.coordinates.frames.HeliographicStonyhurst(obstime=observer.obstime))
    dlon = np.deg2rad(hgs.lon.to_value(u.deg) - observer.lon.to_value(u.deg))
    lat = np.deg2rad(hgs.lat.to_value(u.deg))
    b0 = np.deg2rad(observer.lat.to_value(u.deg))
    return los_projection(bp, bt, br, dlon, lat, b0)


def refine_alignment(src_data, src_wcs, lr_wcs, shape, target, max_shift=3.0,
                     coarse_step=0.25, fine_step=0.0625):
    """Find the sub-pixel translation of the LR grid that maximises the
    Pearson correlation between the reprojected source (MDI) and `target`,
    then reproject once more onto the shifted grid.

    The source is reprojected once onto the LR grid padded by a margin; trial
    shifts are evaluated by cubic-spline resampling of that padded image
    (coarse grid, then a fine grid around the best coarse shift). A shift of
    (dx, dy) LR pixels means LR pixel p takes the source value at grid
    position p + (dx, dy), i.e. CRPIX_new = CRPIX - (dx, dy).

    Returns dict: lr, shift_dx, shift_dy, corr_before, corr_after, at_edge.
    """
    ny, nx = shape
    m = int(math.ceil(max_shift)) + 2
    pad_header = lr_wcs.to_header()
    pad_header["CRPIX1"] = lr_wcs.wcs.crpix[0] + m
    pad_header["CRPIX2"] = lr_wcs.wcs.crpix[1] + m
    pad_header["NAXIS1"], pad_header["NAXIS2"] = nx + 2 * m, ny + 2 * m
    padded, _ = reproject_interp((src_data, src_wcs), WCS(pad_header),
                                 shape_out=(ny + 2 * m, nx + 2 * m))
    corr_before = pearson_finite(padded[m:m + ny, m:m + nx], target)

    valid = np.isfinite(padded).astype(float)
    coeffs = spline_filter(np.where(np.isfinite(padded), padded, 0.0), order=3)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)

    def corr_at(dx, dy):
        coords = [yy + m + dy, xx + m + dx]
        sample = map_coordinates(coeffs, coords, order=3, prefilter=False)
        ok = map_coordinates(valid, coords, order=1) > 0.999
        return pearson_finite(np.where(ok, sample, np.nan), target)

    def search(cx, cy, half, step):
        best = (-np.inf, cx, cy)
        offsets = np.arange(-half, half + step / 2, step)
        for dy in np.clip(cy + offsets, -max_shift, max_shift):
            for dx in np.clip(cx + offsets, -max_shift, max_shift):
                r = corr_at(dx, dy)
                if np.isfinite(r) and r > best[0]:
                    best = (r, dx, dy)
        return best

    _, cx, cy = search(0.0, 0.0, max_shift, coarse_step)
    r_best, dx, dy = search(cx, cy, coarse_step, fine_step)
    at_edge = bool(max(abs(dx), abs(dy)) >= max_shift - fine_step / 2)

    shifted = lr_wcs.deepcopy()
    shifted.wcs.crpix = [lr_wcs.wcs.crpix[0] - dx, lr_wcs.wcs.crpix[1] - dy]
    shifted.wcs.set()
    lr, _ = reproject_interp((src_data, src_wcs), shifted, shape_out=(ny, nx))
    return {"lr": lr, "shift_dx": float(dx), "shift_dy": float(dy),
            "corr_before": corr_before, "corr_after": pearson_finite(lr, target),
            "at_edge": at_edge, "wcs": shifted}


def save_region(path, lr, hr, meta):
    """One matched pair as a compressed float32 .npz: lr (H, W), hr (3, sH, sW)
    and scalar/string metadata (loadable with allow_pickle=False)."""
    lr = np.asarray(lr, np.float32)
    hr = np.asarray(hr, np.float32)
    if hr.ndim != 3 or hr.shape[1] % lr.shape[0] or hr.shape[2] % lr.shape[1] \
            or hr.shape[1] // lr.shape[0] != hr.shape[2] // lr.shape[1]:
        raise ValueError(f"lr {lr.shape} and hr {hr.shape} are not an integer-scale pair")
    extras = {k: np.asarray(v) for k, v in meta.items()}
    np.savez_compressed(path, lr=lr, hr=hr, **extras)


# --------------------------------------------------------------------------
# Patching
# --------------------------------------------------------------------------

def extract_patches(lr, hr, scale_factor, patch_size_lr, stride_lr,
                     min_finite_fraction=0.95):
    """hr is now (3, H, W) instead of (H, W). Validity is checked across
    all three channels, since a pixel that's NaN in only one component
    still isn't usable for a divergence computation that needs all three
    (or at minimum both in-plane components).
    """
    patches = []
    ny, nx = lr.shape
    for row in range(0, ny - patch_size_lr + 1, stride_lr):
        for col in range(0, nx - patch_size_lr + 1, stride_lr):
            lr_patch = lr[row:row + patch_size_lr, col:col + patch_size_lr]
            hr_row, hr_col = row * scale_factor, col * scale_factor
            hr_size = patch_size_lr * scale_factor
            hr_patch = hr[:, hr_row:hr_row + hr_size, hr_col:hr_col + hr_size]

            if hr_patch.shape != (3, hr_size, hr_size):
                continue
            if np.isfinite(lr_patch).mean() < min_finite_fraction:
                continue
            if np.isfinite(hr_patch).mean() < min_finite_fraction:
                continue

            patches.append((lr_patch.copy(), hr_patch.copy(), row, col))
    return patches


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out_dir", default="./dataset_sharp")
    parser.add_argument("--mdi_series", default="mdi.fd_M_96m_lev182")
    parser.add_argument("--hmi_series", default="hmi.sharp_cea_720s")
    parser.add_argument("--segments", default="Bp,Bt,Br",
                         help="Comma-separated SHARP vector segments to pull.")
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument("--max_time_offset_minutes", type=float, default=6.0)
    parser.add_argument("--patch_size_lr", type=int, default=16,
                         help="LR patch size in pixels. NOTE: this default was "
                              "lowered from the old full-disk script's 64, "
                              "since SHARP active-region cutouts are much "
                              "smaller than a full-disk frame -- many regions "
                              "(especially small/newly-tracked ones) won't "
                              "fit a 64px LR patch at all. 16 fits comfortably "
                              "in even fairly small regions; raise it if you "
                              "want fewer, larger patches from bigger regions.")
    parser.add_argument("--stride_lr", type=int, default=8)
    parser.add_argument("--normalisation", type=float, default=3500.0)
    parser.add_argument("--min_finite_fraction", type=float, default=0.95)
    parser.add_argument("--max_frames", type=int, default=None)

    # ---- Optional additions (defaults reproduce the original behaviour) ----
    parser.add_argument("--save_mode", choices=("patches", "regions", "both"), default="patches",
                         help="patches: one .npz per patch (original). regions: one .npz per "
                              "matched pair with the full aligned LR (H,W) and HR (3,sH,sW) "
                              "region plus metadata, listed in regions.csv. both: both.")
    parser.add_argument("--dry_run", action="store_true",
                         help="Query and match only (and apply the limb filter); print frame, "
                              "HARP and pair counts; no export or download.")
    parser.add_argument("--max_center_angle", type=float, default=None,
                         help="Skip pairs whose region centre (LON_FWT/LAT_FWT, falling back to "
                              "the CEA reference point) is more than this many degrees from "
                              "disk centre. Applied before export.")
    parser.add_argument("--min_align_corr", type=float, default=None,
                         help="Skip pairs whose MDI/SHARP correlation (see --align_metric) is "
                              "below this value.")
    parser.add_argument("--align_metric", choices=("los", "br"), default="los",
                         help="Correlation used by --min_align_corr: 'los' = MDI vs SHARP vector "
                              "projected on the MDI line of sight (align_corr_los), 'br' = MDI vs "
                              "Br (align_corr). Both are always written to the manifest.")
    parser.add_argument("--lr_grid_fix", action="store_true",
                         help="Centre each LR pixel on its HR block: CRPIX_LR = "
                              "(CRPIX_HR-0.5)/s+0.5 instead of CRPIX_HR/s.")
    parser.add_argument("--align_refine", choices=("none", "xcorr"), default="none",
                         help="xcorr: per pair, find the sub-pixel LR-grid translation that "
                              "maximises the MDI vs SHARP-LOS correlation (template matching) "
                              "and reproject MDI onto the shifted grid.")
    parser.add_argument("--max_shift_lr", type=float, default=3.0,
                         help="Search radius for --align_refine, in LR pixels. Pairs whose best "
                              "shift sits on this boundary are skipped (reason align_edge).")
    parser.add_argument("--summary_json", default=None,
                         help="Write per-run counts, skip reasons and per-HARP patch counts here.")
    parser.add_argument("--mdi_quality_mask", type=lambda s: int(s, 0), default=None,
                         help="Drop MDI frames with any of these QUALITY bits set (e.g. 0x80000000 "
                              "= no data) instead of the original QUALITY==0-if-any rule.")
    parser.add_argument("--strict_window", action="store_true",
                         help="Drop MDI frames outside [--start, --end] (JSOC slot rounding can "
                              "return the next 00:00 frame).")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = {"args": vars(args), "status": "started", "skipped": []}

    def write_summary(status):
        summary["status"] = status
        summary["skipped_counts"] = dict(Counter(s["reason"] for s in summary["skipped"]))
        if args.summary_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
            with open(args.summary_json, "w") as f:
                json.dump(summary, f, indent=1, default=str)

    def skip(pair, reason, detail=""):
        mdi_t, harp, sharp_t = pair[0], pair[1], pair[2]
        summary["skipped"].append({"mdi_t_rec": mdi_t, "harpnum": harp, "sharp_t_rec": sharp_t,
                                   "reason": reason, "detail": detail})

    if args.scale_factor not in (1, 2, 4):
        print("scale_factor must be 1, 2, or 4", file=sys.stderr)
        sys.exit(1)

    segments = tuple(s.strip() for s in args.segments.split(",") if s.strip())

    raw_dir = os.path.join(args.out_dir, "raw")
    mdi_raw_dir = os.path.join(raw_dir, "mdi")
    sharp_raw_dir = os.path.join(raw_dir, "sharp")
    patch_dir = os.path.join(args.out_dir, "patches")
    os.makedirs(patch_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest_rows = []
    save_patches = args.save_mode in ("patches", "both")
    save_regions = args.save_mode in ("regions", "both")
    region_dir = os.path.join(args.out_dir, "regions")
    if save_regions:
        os.makedirs(region_dir, exist_ok=True)
    region_rows = []

    print(f"Connecting to JSOC as {args.email} ...")
    client = drms.Client(email=args.email)

    t_start = datetime.fromisoformat(args.start)
    t_end = datetime.fromisoformat(args.end)

    print(f"Querying {args.mdi_series} from {t_start} to {t_end} ...")
    mdi_t_recs = query_mdi_series(client, args.mdi_series, t_start, t_end,
                                  quality_mask=args.mdi_quality_mask)
    if args.strict_window:
        mdi_t_recs = filter_window(mdi_t_recs, t_start, t_end)
    print(f"  Found {len(mdi_t_recs)} MDI frames.")

    sharp_query_start = t_start - timedelta(minutes=args.max_time_offset_minutes + 15)
    sharp_query_end = t_end + timedelta(minutes=args.max_time_offset_minutes + 15)
    print(f"Querying {args.hmi_series} from {sharp_query_start} to {sharp_query_end} ...")
    sharp_records, sharp_meta = query_sharp_series(
        client, args.hmi_series, sharp_query_start, sharp_query_end, with_meta=True)
    n_harps = len({h for _, h in sharp_records})
    print(f"  Found {len(sharp_records)} SHARP records across {n_harps} HARP(s).")
    summary.update(mdi_frames=len(mdi_t_recs), sharp_records=len(sharp_records), harps=n_harps)

    if not mdi_t_recs or not sharp_records:
        print("No frames found. Check dates fall within HMI/SHARP's coverage "
              "(SHARP starts ~2010-05), and that your JSOC registration is confirmed.")
        write_summary("no_frames")
        sys.exit(1)

    if args.max_frames is not None:
        mdi_t_recs = mdi_t_recs[:args.max_frames]

    # --- Match every MDI frame to every HARP's nearest SHARP record ---
    # One MDI frame can yield several matches (one per active region present).
    matched_pairs = []  # (mdi_t_rec, harpnum, sharp_t_rec, offset_min)
    for mdi_t_rec_raw in mdi_t_recs:
        mdi_time = parse_jsoc_time(mdi_t_rec_raw)
        matches = find_nearest_sharp_matches(mdi_time, sharp_records, args.max_time_offset_minutes)
        if not matches:
            print(f"  Skipping MDI {mdi_time}: no HARP within "
                  f"{args.max_time_offset_minutes} min tolerance.")
            continue
        for harpnum, sharp_t_rec, offset_min in matches:
            matched_pairs.append((mdi_t_rec_raw, harpnum, sharp_t_rec, offset_min))

    print(f"\n{len(matched_pairs)} MDI/SHARP (region) pairs matched within tolerance.")
    summary["matched_pairs"] = len(matched_pairs)
    if not matched_pairs:
        print("Nothing to download. Try a larger --max_time_offset_minutes.")
        write_summary("no_matches")
        sys.exit(1)

    # --- Region centre / limb filter (before export, so skipped pairs are never downloaded) ---
    pair_info = {}
    kept_pairs = []
    for pair in matched_pairs:
        _, harpnum, sharp_t_rec, _ = pair
        meta = sharp_meta.get((harpnum, sharp_t_rec), {})
        lon, lat, src = region_center(meta)
        b0 = meta.get("CRLT_OBS", 0.0)
        angle = center_angle_deg(lon, lat, b0 if np.isfinite(b0) else 0.0) \
            if np.isfinite(lon) and np.isfinite(lat) else float("nan")
        noaa = meta.get("NOAA_AR", float("nan"))
        pair_info[pair] = {"noaa_ar": int(noaa) if np.isfinite(noaa) else 0,
                           "lon_fwt": meta.get("LON_FWT", float("nan")),
                           "lat_fwt": meta.get("LAT_FWT", float("nan")),
                           "center_lon": round(lon, 4), "center_lat": round(lat, 4),
                           "center_angle": round(angle, 4), "center_source": src}
        if args.max_center_angle is not None and not (angle <= args.max_center_angle):
            skip(pair, "limb", f"center_angle={angle:.1f} ({src})")
            continue
        kept_pairs.append(pair)
    if args.max_center_angle is not None:
        n_limb = len(matched_pairs) - len(kept_pairs)
        print(f"Limb filter (--max_center_angle {args.max_center_angle}): skipped {n_limb} of "
              f"{len(matched_pairs)} pairs; {len(kept_pairs)} remain.")
    matched_pairs = kept_pairs
    summary["pairs_after_limb"] = len(matched_pairs)

    if args.dry_run:
        per_harp = Counter(p[1] for p in matched_pairs)
        print(f"\nDRY RUN: {summary['mdi_frames']} MDI frames, {summary['harps']} HARPs, "
              f"{summary['matched_pairs']} matched pairs, {len(matched_pairs)} after limb filter.")
        for h, n in sorted(per_harp.items()):
            info = next(pair_info[p] for p in pair_info if p[1] == h)
            print(f"  HARP {h:>5}  NOAA {info['noaa_ar']:>5}  pairs {n:>3}  "
                  f"centre angle {info['center_angle']:.1f} ({info['center_source']})")
        summary["pairs_per_harp"] = {str(h): n for h, n in per_harp.items()}
        summary["exports_needed"] = 1 + len(per_harp)
        write_summary("dry_run")
        return

    if not matched_pairs:
        print("Nothing to download after the limb filter.")
        write_summary("no_matches")
        return

    # --- ONE batched export each for MDI and SHARP ---
    mdi_list = sorted({p[0] for p in matched_pairs})
    harp_trec_pairs = sorted({(p[1], p[2]) for p in matched_pairs})

    mdi_map_paths = batched_export_mdi(client, args.mdi_series, mdi_list, mdi_raw_dir, "MDI")
    sharp_map_paths = batched_export_sharp(
        client, args.hmi_series, harp_trec_pairs, sharp_raw_dir,
        segments=segments, label="SHARP")

    # --- Now process each matched pair locally, no more JSOC calls ---
    total_patches = 0
    patches_per_harp = Counter()
    for pair in matched_pairs:
        mdi_t_rec_raw, harpnum, sharp_t_rec, offset_min = pair
        print(f"\nProcessing pair: MDI {mdi_t_rec_raw}  <->  SHARP HARP {harpnum} "
              f"{sharp_t_rec} (offset {offset_min:.2f} min)")

        mdi_path = mdi_map_paths.get(mdi_t_rec_raw)
        seg_paths = sharp_map_paths.get((harpnum, sharp_t_rec), {})

        if mdi_path is None:
            print("  Skipped: missing downloaded MDI file (check export/download logs above).")
            skip(pair, "missing_mdi")
            continue
        if not set(segments) <= set(seg_paths.keys()):
            missing = set(segments) - set(seg_paths.keys())
            print(f"  Skipped: missing SHARP segment(s) {missing} for this record.")
            skip(pair, "missing_sharp", str(sorted(missing)))
            continue

        try:
            mdi_map = load_map_safely(mdi_path)
            sharp_maps = {seg: load_map_safely(seg_paths[seg]) for seg in segments}

            mdi_data, hr_stack, lr_wcs = align_pair(
                mdi_map, sharp_maps, args.scale_factor,
                lr_grid_fix=args.lr_grid_fix, return_wcs=True)

            # --- Registration check (and optional refinement) against SHARP ---
            align = {"align_corr": float("nan"), "align_corr_los": float("nan"),
                     "align_corr_los_before": float("nan"), "shift_dx": float("nan"),
                     "shift_dy": float("nan"), "shift_at_edge": False}
            try:
                target_los = sharp_los_lr(sharp_maps, mdi_map.observer_coordinate,
                                          args.scale_factor)[:mdi_data.shape[0], :mdi_data.shape[1]]
            except Exception as e:
                target_los = None
                print(f"  [align] could not build the SHARP line-of-sight target: {e}")
            if target_los is not None:
                align["align_corr_los_before"] = pearson_finite(mdi_data, target_los)
            if args.align_refine == "xcorr":
                if target_los is None:
                    raise RuntimeError("--align_refine xcorr needs the SHARP line-of-sight target")
                res = refine_alignment(mdi_map.data, mdi_map.wcs, lr_wcs, mdi_data.shape,
                                       target_los, max_shift=args.max_shift_lr)
                mdi_data, lr_wcs = res["lr"], res["wcs"]
                align.update(shift_dx=round(res["shift_dx"], 4), shift_dy=round(res["shift_dy"], 4),
                             shift_at_edge=res["at_edge"])
            align["align_corr"] = pearson_finite(mdi_data, block_mean(hr_stack[2], args.scale_factor))
            if target_los is not None:
                align["align_corr_los"] = pearson_finite(mdi_data, target_los)
            print(f"  [align] corr vs Br={align['align_corr']:.3f}, vs SHARP-LOS="
                  f"{align['align_corr_los']:.3f} (before refinement "
                  f"{align['align_corr_los_before']:.3f}), shift (dx,dy)=("
                  f"{align['shift_dx']}, {align['shift_dy']}) LR px"
                  f"{' AT SEARCH EDGE' if align['shift_at_edge'] else ''}")

            # --- Diagnostics: pin down WHERE data is going missing before ---
            # --- blaming the whole pipeline when patch count comes out 0. ---
            mdi_finite = np.isfinite(mdi_data).mean()
            hr_finite_per_channel = [np.isfinite(hr_stack[c]).mean() for c in range(hr_stack.shape[0])]
            print(f"  [diag] MDI reprojected (LR) shape={mdi_data.shape}, "
                  f"finite_fraction={mdi_finite:.3f}, "
                  f"min/max (finite only)="
                  f"{np.nanmin(mdi_data) if mdi_finite > 0 else 'nan'}/"
                  f"{np.nanmax(mdi_data) if mdi_finite > 0 else 'nan'}")
            print(f"  [diag] SHARP (HR) shape={hr_stack.shape}, "
                  f"finite_fraction per channel (Bp,Bt,Br)={[round(f, 3) for f in hr_finite_per_channel]}")
            if mdi_finite < 0.5:
                print(f"  [diag] MDI reprojection looks broken (finite fraction "
                      f"{mdi_finite:.3f} < 0.5) -- this points at align_pair()'s "
                      f"reproject_interp call, not the patch extraction logic. "
                      f"Likely causes: the MDI FITS export wasn't recognized as "
                      f"an MDI-specific sunpy map (check `type(mdi_map)` and "
                      f"`mdi_map.wcs.to_header()` by hand), or the two WCS "
                      f"frames (MDI helioprojective vs SHARP heliographic CEA) "
                      f"aren't resolving a shared coordinate transform, so "
                      f"reproject_interp silently returns NaN everywhere "
                      f"instead of raising an error.")
            elif min(hr_finite_per_channel) < 0.5:
                print(f"  [diag] SHARP data itself looks mostly NaN -- this "
                      f"points at the downloaded FITS files or segment "
                      f"selection, not the reprojection step.")

            lr_norm = mdi_data / args.normalisation
            hr_norm = hr_stack / args.normalisation

            ny_lr, nx_lr = lr_norm.shape
            if ny_lr < args.patch_size_lr or nx_lr < args.patch_size_lr:
                print(f"  Skipped: this region's LR grid ({ny_lr}x{nx_lr}) is "
                      f"smaller than --patch_size_lr ({args.patch_size_lr}) in "
                      f"at least one dimension -- no patch of that size fits. "
                      f"This is an active-region-size limitation, not a bug: "
                      f"small/newly-tracked HARPs are physically smaller than "
                      f"larger ones. Lower --patch_size_lr to fit more regions, "
                      f"or accept that the smallest regions will be skipped.")
                skip(pair, "too_small", f"lr={ny_lr}x{nx_lr}")
                continue

            if args.align_refine == "xcorr" and align["shift_at_edge"]:
                print(f"  Skipped: best alignment shift is at the search boundary "
                      f"(+-{args.max_shift_lr} LR px); registration not trusted.")
                skip(pair, "align_edge", f"shift=({align['shift_dx']},{align['shift_dy']})")
                continue
            if args.min_align_corr is not None:
                key = "align_corr_los" if args.align_metric == "los" else "align_corr"
                if not (align[key] >= args.min_align_corr):
                    print(f"  Skipped: {key}={align[key]:.3f} < --min_align_corr "
                          f"{args.min_align_corr} (misaligned or bad data).")
                    skip(pair, "align_corr", f"{key}={align[key]:.3f}")
                    continue

            patches = extract_patches(
                lr_norm, hr_norm, args.scale_factor,
                args.patch_size_lr, args.stride_lr, args.min_finite_fraction)
            print(f"  Extracted {len(patches)} valid patches.")
            if not patches:
                skip(pair, "nan", f"no patch with finite fraction >= {args.min_finite_fraction}")
                continue

            extra = dict(pair_info[pair], **align)
            safe_t_rec = mdi_t_rec_raw.replace(":", "").replace(".", "")
            for (lr_patch, hr_patch, row, col) in patches:
                if not save_patches:
                    break
                patch_id = f"harp{harpnum}_{safe_t_rec}_{row}_{col}"
                patch_path = os.path.join(patch_dir, f"{patch_id}.npz")
                np.savez_compressed(patch_path,
                                     lr=lr_patch.astype(np.float32),
                                     hr=hr_patch.astype(np.float32))  # hr shape (3, H, W)
                manifest_rows.append({
                    "patch_file": os.path.relpath(patch_path, args.out_dir),
                    "mdi_t_rec": mdi_t_rec_raw,
                    "harpnum": harpnum,
                    "sharp_t_rec": sharp_t_rec,
                    "time_offset_minutes": round(offset_min, 3),
                    "row": row, "col": col,
                    "scale_factor": args.scale_factor,
                    "normalisation": args.normalisation,
                    "hr_channels": "Bp,Bt,Br",
                    **extra,
                })
                total_patches += 1
            patches_per_harp[harpnum] += len(patches)

            if save_regions:
                region_path = os.path.join(region_dir, f"harp{harpnum}_{safe_t_rec}.npz")
                obs = mdi_map.observer_coordinate
                region_meta = {
                    "harpnum": harpnum, "mdi_t_rec": mdi_t_rec_raw, "sharp_t_rec": sharp_t_rec,
                    "time_offset_minutes": offset_min, "scale_factor": args.scale_factor,
                    "normalisation": args.normalisation, "hr_channels": "Bp,Bt,Br",
                    "n_patches": len(patches), **extra,
                    "mdi_obs_lon_deg": obs.lon.to_value(u.deg),
                    "mdi_obs_lat_deg": obs.lat.to_value(u.deg),
                    "mdi_obs_dist_m": obs.radius.to_value(u.m),
                    "lr_wcs_header": lr_wcs.to_header_string(),
                    "hr_wcs_header": sharp_maps["Br"].wcs.to_header_string(),
                    "sharp_header_json": json.dumps(dict(sharp_maps["Br"].meta), default=str),
                }
                save_region(region_path, lr_norm, hr_norm, region_meta)
                region_rows.append({
                    "region_file": os.path.relpath(region_path, args.out_dir),
                    "mdi_t_rec": mdi_t_rec_raw, "harpnum": harpnum, "sharp_t_rec": sharp_t_rec,
                    "time_offset_minutes": round(offset_min, 3),
                    "lr_ny": ny_lr, "lr_nx": nx_lr, "n_patches": len(patches),
                    "scale_factor": args.scale_factor, "normalisation": args.normalisation,
                    **extra,
                })

        except Exception as e:
            print(f"  ERROR processing this pair: {e}")
            traceback.print_exc()
            skip(pair, "error", str(e)[:200])
            continue

    if manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nWrote manifest with {len(manifest_rows)} patches to {manifest_path}")
    elif save_patches:
        print("\nNo patches were produced. Check the log above.")

    if region_rows:
        regions_path = os.path.join(args.out_dir, "regions.csv")
        with open(regions_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(region_rows[0].keys()))
            writer.writeheader()
            writer.writerows(region_rows)
        print(f"Wrote {len(region_rows)} regions to {regions_path}")

    counts = Counter(s["reason"] for s in summary["skipped"])
    if counts:
        print("Skipped pairs by reason: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    summary.update(pairs_saved=len({(r["mdi_t_rec"], r["harpnum"]) for r in manifest_rows}
                                   | {(r["mdi_t_rec"], r["harpnum"]) for r in region_rows}),
                   patches=sum(patches_per_harp.values()), patch_files_saved=total_patches,
                   regions_saved=len(region_rows),
                   patches_per_harp={str(h): n for h, n in sorted(patches_per_harp.items())},
                   align=[{k: r[k] for k in ("mdi_t_rec", "harpnum", "align_corr", "align_corr_los",
                                             "align_corr_los_before", "shift_dx", "shift_dy")}
                          for r in (region_rows or _one_row_per_pair(manifest_rows))])
    write_summary("ok")

    print(f"\nDone. Total patches saved: {total_patches}")
    print(f"Patches directory: {patch_dir}")


def _one_row_per_pair(rows):
    seen, out = set(), []
    for r in rows:
        key = (r["mdi_t_rec"], r["harpnum"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


if __name__ == "__main__":
    main()