"""Tests for pipeline/build_dataset.py (no JSOC access needed)."""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pipeline import build_dataset as bd


# ---------------------------------------------------------------- day windows

def test_day_windows_split_at_midnight_with_inclusive_end_one_second_early():
    w = bd.day_windows(datetime(2011, 2, 1), datetime(2011, 2, 3))
    assert [d for d, _, _ in w] == ["2011-02-01", "2011-02-02"]
    assert w[0][1] == datetime(2011, 2, 1, 0, 0, 0)
    assert w[0][2] == datetime(2011, 2, 1, 23, 59, 59)   # 00:00 of the next day belongs to the next day
    assert w[1][2] == datetime(2011, 2, 2, 23, 59, 59)


def test_day_windows_partial_window_inside_one_day():
    w = bd.day_windows(datetime(2010, 6, 1, 0, 0), datetime(2010, 6, 1, 2, 0))
    assert w == [("2010-06-01", datetime(2010, 6, 1, 0, 0), datetime(2010, 6, 1, 1, 59, 59))]


# ---------------------------------------------------------------- resume / retry

def _write_summary(day_dir, status, **extra):
    os.makedirs(day_dir, exist_ok=True)
    with open(os.path.join(day_dir, "summary.json"), "w") as f:
        json.dump({"status": status, **extra}, f)


@pytest.mark.parametrize("status,done", [("ok", True), ("no_frames", True), ("no_matches", True),
                                         ("started", False), ("dry_run", False)])
def test_is_day_done_depends_on_summary_status(tmp_path, status, done):
    _write_summary(tmp_path / "d", status)
    assert bd.is_day_done(str(tmp_path / "d")) is done


def test_is_day_done_false_without_summary(tmp_path):
    assert bd.is_day_done(str(tmp_path)) is False


def test_pending_export_error_is_detected():
    log = "drms.exceptions.DrmsExportError: [status=7] User someone has pending export"
    assert bd.is_pending_export_error(log) is True
    assert bd.is_pending_export_error("ConnectionError: timed out") is False


# ---------------------------------------------------------------- merging

def _fake_day(run_dir, day, harp, n_patches, with_region=True):
    d = os.path.join(run_dir, "days", day)
    os.makedirs(os.path.join(d, "patches"), exist_ok=True)
    os.makedirs(os.path.join(d, "regions"), exist_ok=True)
    rows = []
    for i in range(n_patches):
        p = f"patches/harp{harp}_{day}_{i}.npz"
        np.savez_compressed(os.path.join(d, p), lr=np.zeros((16, 16), np.float32),
                            hr=np.zeros((3, 64, 64), np.float32))
        rows.append({"patch_file": p, "harpnum": harp, "mdi_t_rec": day})
    pd.DataFrame(rows).to_csv(os.path.join(d, "manifest.csv"), index=False)
    if with_region:
        r = f"regions/harp{harp}_{day}.npz"
        np.savez_compressed(os.path.join(d, r), lr=np.zeros((20, 24), np.float32),
                            hr=np.zeros((3, 80, 96), np.float32))
        pd.DataFrame([{"region_file": r, "harpnum": harp, "n_patches": n_patches}]).to_csv(
            os.path.join(d, "regions.csv"), index=False)
    _write_summary(d, "ok", mdi_frames=15, harps=1, matched_pairs=3, patches=n_patches,
                   patch_files_saved=n_patches, regions_saved=int(with_region),
                   skipped=[{"mdi_t_rec": day, "harpnum": harp, "sharp_t_rec": day,
                             "reason": "limb", "detail": ""}],
                   skipped_counts={"limb": 1})
    return d


def test_merge_outputs_prefixes_paths_and_concatenates_days(tmp_path):
    run = str(tmp_path)
    _fake_day(run, "2011-02-01", 1, 2)
    _fake_day(run, "2011-02-02", 2, 3)
    bd.merge_outputs(run)
    man = pd.read_csv(os.path.join(run, "manifest.csv"))
    assert len(man) == 5
    assert all(os.path.exists(os.path.join(run, p)) for p in man["patch_file"])
    assert set(man["day"]) == {"2011-02-01", "2011-02-02"}
    reg = pd.read_csv(os.path.join(run, "regions.csv"))
    assert len(reg) == 2 and all(os.path.exists(os.path.join(run, p)) for p in reg["region_file"])
    days = pd.read_csv(os.path.join(run, "days.csv"))
    assert list(days["patches"]) == [2, 3]
    assert list(days["skipped_limb"]) == [1, 1]
    skipped = pd.read_csv(os.path.join(run, "skipped.csv"))
    assert len(skipped) == 2 and set(skipped["reason"]) == {"limb"}


# ---------------------------------------------------------------- verify / delete raw

def test_verify_day_accepts_consistent_outputs(tmp_path):
    d = _fake_day(str(tmp_path), "2011-02-01", 1, 2)
    ok, msg = bd.verify_day(d)
    assert ok, msg


def test_verify_day_rejects_missing_patch_file(tmp_path):
    d = _fake_day(str(tmp_path), "2011-02-01", 1, 2)
    os.remove(os.path.join(d, "patches", "harp1_2011-02-01_0.npz"))
    ok, _ = bd.verify_day(d)
    assert ok is False


def test_verify_day_rejects_wrong_shapes(tmp_path):
    d = _fake_day(str(tmp_path), "2011-02-01", 1, 1)
    np.savez_compressed(os.path.join(d, "patches", "harp1_2011-02-01_0.npz"),
                        lr=np.zeros((16, 16), np.float32), hr=np.zeros((3, 32, 64), np.float32))
    ok, _ = bd.verify_day(d)
    assert ok is False


def test_delete_raw_removes_only_fits_in_that_days_raw_folder(tmp_path):
    d = _fake_day(str(tmp_path), "2011-02-01", 1, 1)
    other = _fake_day(str(tmp_path), "2011-02-02", 1, 1)
    for day_dir in (d, other):
        for sub in ("mdi", "sharp"):
            os.makedirs(os.path.join(day_dir, "raw", sub), exist_ok=True)
            with open(os.path.join(day_dir, "raw", sub, "x.fits"), "wb") as f:
                f.write(b"0" * 100)
    keep = os.path.join(d, "raw", "notes.txt")
    open(keep, "w").close()

    freed = bd.delete_raw(d)

    assert freed == 200
    assert not os.path.exists(os.path.join(d, "raw", "mdi", "x.fits"))
    assert os.path.exists(keep)                                        # non-FITS untouched
    assert os.path.exists(os.path.join(other, "raw", "mdi", "x.fits"))  # other day untouched
    assert os.path.exists(os.path.join(d, "manifest.csv"))


# ---------------------------------------------------------------- pipeline command

def test_pipeline_command_passes_window_and_flags():
    args = bd.build_parser().parse_args([
        "--run", "pilot", "--start", "2011-02-01", "--end", "2011-02-02",
        "--email", "a@b.c", "--save_mode", "both", "--max_center_angle", "60",
        "--min_align_corr", "0.5", "--lr_grid_fix", "--align_refine", "xcorr"])
    cmd = bd.pipeline_command(args, "/x/days/2011-02-01", datetime(2011, 2, 1),
                              datetime(2011, 2, 1, 23, 59, 59), dry_run=False)
    s = " ".join(cmd)
    assert "--start 2011-02-01T00:00:00" in s and "--end 2011-02-01T23:59:59" in s
    assert "--out_dir /x/days/2011-02-01" in s
    assert "--summary_json /x/days/2011-02-01/summary.json" in s
    for flag in ("--save_mode both", "--max_center_angle 60.0", "--min_align_corr 0.5",
                 "--lr_grid_fix", "--align_refine xcorr", "--email a@b.c"):
        assert flag in s
    assert "--dry_run" not in s
    assert "--strict_window" in s          # always on: JSOC slot rounding would duplicate 00:00 frames
    assert "--mdi_quality_mask" not in s   # only when requested


def test_pipeline_command_passes_quality_mask_when_given():
    args = bd.build_parser().parse_args([
        "--run", "pilot", "--start", "2011-02-01", "--end", "2011-02-02", "--email", "a@b.c",
        "--mdi_quality_mask", "0x80000000"])
    cmd = bd.pipeline_command(args, "/x", datetime(2011, 2, 1), datetime(2011, 2, 1, 23, 59, 59),
                              dry_run=True)
    s = " ".join(cmd)
    assert "--mdi_quality_mask 0x80000000" in s and "--dry_run" in s


# ---------------------------------------------------------------- multi-day windows (Phase 8: fewer JSOC exports)

def test_multi_day_windows_cover_the_range_in_blocks_and_end_one_second_early():
    w = bd.day_windows(datetime(2010, 5, 1), datetime(2010, 5, 11), days=4)
    assert [x[0] for x in w] == ["2010-05-01", "2010-05-05", "2010-05-09"]
    assert w[0][1] == datetime(2010, 5, 1) and w[0][2] == datetime(2010, 5, 4, 23, 59, 59)
    assert w[1][1] == datetime(2010, 5, 5) and w[1][2] == datetime(2010, 5, 8, 23, 59, 59)
    assert w[2][1] == datetime(2010, 5, 9) and w[2][2] == datetime(2010, 5, 10, 23, 59, 59)


def test_one_day_windows_are_the_default():
    a = bd.day_windows(datetime(2011, 2, 1), datetime(2011, 2, 4))
    b = bd.day_windows(datetime(2011, 2, 1), datetime(2011, 2, 4), days=1)
    assert a == b and len(a) == 3


def test_window_days_flag_reaches_the_windows(monkeypatch, tmp_path):
    seen = {}

    def fake_windows(start, end, days=1):
        seen["days"] = days
        return []
    monkeypatch.setattr(bd, "day_windows", fake_windows)
    bd.main(["--run", "x", "--start", "2010-05-01", "--end", "2010-05-09", "--email", "a@b.c",
             "--data_root", str(tmp_path), "--window_days", "4"])
    assert seen["days"] == 4


# ---------------------------------------------------------------- one JSOC build per email (parallel emails allowed)

def test_builds_with_different_emails_can_run_side_by_side(tmp_path):
    with bd.JsocLock(str(tmp_path), email="a@x.org"):
        with bd.JsocLock(str(tmp_path), email="b@y.org"):
            pass


def test_a_second_build_with_the_same_email_is_refused(tmp_path):
    with bd.JsocLock(str(tmp_path), email="a@x.org"):
        with pytest.raises(SystemExit):
            with bd.JsocLock(str(tmp_path), email="a@x.org"):
                pass


def test_lock_without_email_keeps_the_old_path(tmp_path):
    lock = bd.JsocLock(str(tmp_path))
    assert lock.path == os.path.join(str(tmp_path), ".jsoc_build.lock")


def test_windows_done_in_another_run_are_skipped(monkeypatch, tmp_path):
    other = tmp_path / "data" / "full"
    (other / "days" / "2010-11-06").mkdir(parents=True)
    (other / "days" / "2010-11-06" / "summary.json").write_text(json.dumps({"status": "ok"}))
    ran = []
    monkeypatch.setattr(bd, "run_day", lambda args, day_dir, ws, we, log: ran.append(os.path.basename(day_dir)) or "failed")
    monkeypatch.setattr(bd, "merge_outputs", lambda run_dir: None)
    bd.main(["--run", "full_b", "--start", "2010-10-30", "--end", "2010-11-20", "--window_days", "7",
             "--email", "b@y.org", "--data_root", str(tmp_path / "data"), "--skip_done_in", str(other)])
    assert ran == ["2010-10-30", "2010-11-13"]


# ---------------------------------------------------------------- week claiming (several emails share one range)

def test_a_free_window_is_claimed_and_a_live_claim_blocks_others(tmp_path):
    claims = str(tmp_path / "claims")
    assert bd.claim_window(claims, "2010-06-19", "a@x.org") is True
    other_pid = os.getppid()                                   # a live process that is not us
    with open(os.path.join(claims, "2010-06-26.claim"), "w") as f:
        f.write(f"b@y.org {other_pid}")
    assert bd.claim_window(claims, "2010-06-26", "a@x.org") is False


def test_a_claim_by_a_dead_process_is_taken_over(tmp_path):
    claims = tmp_path / "claims"
    claims.mkdir()
    (claims / "2010-07-03.claim").write_text("b@y.org 999999999")   # no such pid
    assert bd.claim_window(str(claims), "2010-07-03", "a@x.org") is True
    assert (claims / "2010-07-03.claim").read_text().split()[0] == "a@x.org"


def test_builders_skip_windows_claimed_by_a_live_builder_or_done_anywhere(monkeypatch, tmp_path):
    data = tmp_path / "data"
    done_elsewhere = data / "full_b" / "days" / "2010-05-08"
    done_elsewhere.mkdir(parents=True)
    (done_elsewhere / "summary.json").write_text(json.dumps({"status": "ok"}))
    claims = data / "claims"
    claims.mkdir(parents=True)
    (claims / "2010-05-15.claim").write_text(f"c@z.org {os.getppid()}")
    ran = []
    monkeypatch.setattr(bd, "run_day", lambda args, day_dir, ws, we, log: ran.append(os.path.basename(day_dir)) or "failed")
    monkeypatch.setattr(bd, "merge_outputs", lambda run_dir: None)
    bd.main(["--run", "full", "--start", "2010-05-01", "--end", "2010-05-29", "--window_days", "7", "--email", "a@x.org",
             "--data_root", str(data), "--skip_done_in", str(data / "full_b"), str(data / "full_c"),
             "--claims_dir", str(claims)])
    assert ran == ["2010-05-01", "2010-05-22"]
