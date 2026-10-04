"""Tests for scripts/merge_runs.py (combine two build runs made with different JSOC emails)."""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import merge_runs as mr  # noqa: E402


def _day(run, day, harp):
    d = run / "days" / day
    (d / "regions").mkdir(parents=True)
    (d / "regions" / f"h{harp}.npz").write_bytes(b"x")
    pd.DataFrame([{"region_file": f"regions/h{harp}.npz", "harpnum": harp, "mdi_t_rec": day}]).to_csv(d / "regions.csv", index=False)
    (d / "summary.json").write_text(json.dumps({"status": "ok", "skipped": [], "skipped_counts": {}}))


def test_windows_of_the_second_run_are_moved_and_the_tables_merged(tmp_path):
    a, b = tmp_path / "full", tmp_path / "full_b"
    _day(a, "2010-05-01", 1)
    _day(b, "2010-10-30", 2)
    _day(b, "2010-11-06", 3)
    mr.merge_runs(str(a), str(b))
    regs = pd.read_csv(a / "regions.csv")
    assert sorted(regs.harpnum) == [1, 2, 3]
    assert all((a / f).exists() for f in regs.region_file)
    assert not (b / "days" / "2010-10-30").exists()
    assert len(pd.read_csv(a / "days.csv")) == 3


def test_overlapping_windows_are_refused(tmp_path):
    a, b = tmp_path / "full", tmp_path / "full_b"
    _day(a, "2010-05-01", 1)
    _day(b, "2010-05-01", 2)
    with pytest.raises(FileExistsError):
        mr.merge_runs(str(a), str(b))
    assert (b / "days" / "2010-05-01").exists()      # nothing moved


def test_an_unfinished_window_in_dst_is_moved_aside_not_deleted(tmp_path):
    a, b = tmp_path / "full", tmp_path / "full_b"
    _day(a, "2010-05-01", 1)
    part = a / "days" / "2010-10-30"
    part.mkdir(parents=True)
    (part / "summary.json").write_text(json.dumps({"status": "failed"}))
    _day(b, "2010-10-30", 2)
    mr.merge_runs(str(a), str(b))
    assert (a / "days_incomplete" / "2010-10-30" / "summary.json").exists()
    assert sorted(pd.read_csv(a / "regions.csv").harpnum) == [1, 2]


def test_unfinished_windows_in_src_stay_where_they_are(tmp_path):
    a, b = tmp_path / "full", tmp_path / "full_c"
    _day(a, "2010-06-19", 1)                       # finished by A
    part = b / "days" / "2010-06-19"               # C was interrupted on the same week
    part.mkdir(parents=True)
    (part / "summary.json").write_text(json.dumps({"status": "failed"}))
    _day(b, "2010-07-03", 2)
    moved = mr.merge_runs(str(a), str(b))
    assert moved == ["2010-07-03"]
    assert part.exists()                           # left in place, not deleted
    assert sorted(pd.read_csv(a / "regions.csv").harpnum) == [1, 2]
