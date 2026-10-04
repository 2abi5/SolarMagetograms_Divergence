"""Tests for pipeline/split_dataset.py (region-group split, no leakage)."""
import json

import pandas as pd
import pytest

from pipeline import split_dataset as sd


def _regions(rows):
    """rows: (harpnum, noaa_ar, n_patches) per (region, time) pair."""
    return pd.DataFrame([{"harpnum": h, "noaa_ar": a, "n_patches": n, "mdi_t_rec": f"t{i}"}
                         for i, (h, a, n) in enumerate(rows)])


def test_harps_sharing_a_noaa_ar_form_one_group():
    reg = _regions([(1, 11158, 10), (2, 11158, 10), (3, 0, 5), (4, 11160, 7)])
    groups = sd.region_groups(reg)
    assert groups[1] == groups[2]
    assert groups[3] != groups[1] and groups[4] != groups[1]


def test_harp_without_noaa_ar_is_its_own_group():
    reg = _regions([(3, 0, 5), (5, 0, 6)])
    groups = sd.region_groups(reg)
    assert groups[3] != groups[5]


def test_harp_with_noaa_ar_only_at_some_times_still_groups_by_it():
    reg = _regions([(7, 0, 5), (7, 11163, 5), (8, 11163, 4)])
    groups = sd.region_groups(reg)
    assert groups[7] == groups[8]


def test_group_chains_are_merged_transitively():
    # HARP 1 carries AR A, HARP 2 carries A and B, HARP 3 carries B -> one group
    reg = _regions([(1, 100, 1), (2, 100, 1), (2, 200, 1), (3, 200, 1)])
    groups = sd.region_groups(reg)
    assert groups[1] == groups[2] == groups[3]


def test_shared_noaa_report_lists_harps_with_a_common_ar():
    reg = _regions([(1, 11158, 10), (2, 11158, 10), (3, 0, 5)])
    assert sd.shared_noaa_report(reg) == {11158: [1, 2]}


def _many(n_groups=30, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    return _regions([(h, 0, int(n)) for h, n in zip(range(1, n_groups + 1),
                                                     rng.integers(20, 400, n_groups))])


def test_split_has_no_group_in_two_splits_and_covers_everything():
    reg = _many()
    split = sd.make_split(reg, seed=0)
    harps = [set(split[s]["harps"]) for s in ("train", "val", "test")]
    assert not (harps[0] & harps[1]) and not (harps[0] & harps[2]) and not (harps[1] & harps[2])
    assert set().union(*harps) == set(reg.harpnum)


def test_split_is_balanced_by_patch_count():
    reg = _many()
    split = sd.make_split(reg, seed=0)
    total = reg.n_patches.sum()
    for s, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert split[s]["patches"] / total == pytest.approx(target, abs=0.05)
    assert all(len(split[s]["harps"]) >= 2 for s in ("val", "test"))


def test_split_is_deterministic_for_a_seed_and_changes_with_it():
    reg = _many()
    a, b, c = sd.make_split(reg, seed=0), sd.make_split(reg, seed=0), sd.make_split(reg, seed=1)
    assert a == b
    assert a["test"]["harps"] != c["test"]["harps"]


def test_split_keeps_grouped_harps_together():
    reg = pd.concat([_many(), _regions([(101, 11158, 50), (102, 11158, 50)])], ignore_index=True)
    split = sd.make_split(reg, seed=3)
    where = {h: s for s in ("train", "val", "test") for h in split[s]["harps"]}
    assert where[101] == where[102]


def test_write_split_json_round_trip(tmp_path):
    reg = _many()
    path = tmp_path / "splits" / "pilot.json"
    info = sd.write_split(reg, str(path), seed=0, run="pilot")
    data = json.loads(path.read_text())
    assert data["seed"] == 0 and data["run"] == "pilot"
    assert set(data["splits"]) == {"train", "val", "test"}
    assert data == info


# ---------------------------------------------------------------- returning regions (Carrington position)

def _pos(rows):
    """rows: (harpnum, carr_lon, carr_lat, first_day, last_day)"""
    return pd.DataFrame([{"harpnum": h, "carr_lon": lo, "carr_lat": la,
                          "t_first": pd.Timestamp("2011-01-01") + pd.Timedelta(days=a),
                          "t_last": pd.Timestamp("2011-01-01") + pd.Timedelta(days=b)}
                         for h, lo, la, a, b in rows])


def test_region_returning_at_the_same_carrington_position_is_linked():
    pos = _pos([(1, 100.0, 20.0, 0, 9), (2, 104.0, 18.0, 25, 34), (3, 250.0, -15.0, 0, 9)])
    assert sd.returning_links(pos) == [(1, 2)]


def test_longitude_wraps_at_360():
    pos = _pos([(1, 355.0, 10.0, 0, 9), (2, 3.0, 12.0, 25, 34)])
    assert sd.returning_links(pos) == [(1, 2)]


def test_overlapping_or_far_apart_regions_are_not_linked():
    pos = _pos([(1, 100.0, 20.0, 0, 9), (2, 101.0, 20.0, 5, 12),      # simultaneous: different regions
                (3, 100.0, 20.0, 60, 69),                             # two rotations later: too late
                (4, 100.0, -20.0, 25, 34)])                           # other hemisphere
    assert sd.returning_links(pos) == []


def test_returning_links_merge_groups_in_the_split():
    reg = _regions([(1, 0, 10), (2, 0, 10), (3, 0, 5)])
    groups = sd.region_groups(reg, extra_links=[(1, 2)])
    assert groups[1] == groups[2] != groups[3]


# ---------------------------------------------------------------- month stratification (Phase 8)

def _year(n_groups=120, seed=0):
    """Groups that each live in one month; month sizes differ."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(1, n_groups + 1):
        month = f"2010.{1 + (h * 7) % 12:02d}"
        for k in range(int(rng.integers(1, 4))):
            rows.append({"harpnum": h, "noaa_ar": 0, "n_patches": int(rng.integers(20, 400)),
                         "mdi_t_rec": f"{month}.1{k}_00:00:00_TAI", "month": month})
    return pd.DataFrame(rows)


def _month_tv(reg, split):
    """Mean over months of the fraction of that month's patches that are not where the targets say."""
    import numpy as np
    where = {h: s for s in ("train", "val", "test") for h in split[s]["harps"]}
    reg = reg.assign(split=reg.harpnum.map(where))
    tab = reg.pivot_table(index="month", columns="split", values="n_patches", aggfunc="sum", fill_value=0)
    frac = tab.div(tab.sum(axis=1), axis=0)
    tgt = pd.Series({"train": 0.70, "val": 0.15, "test": 0.15})
    return float((0.5 * (frac.reindex(columns=tgt.index, fill_value=0) - tgt).abs().sum(axis=1)).mean())


def test_default_split_is_unchanged_by_the_stratification_option():
    reg = _many()
    assert sd.make_split(reg, seed=0) == sd.make_split(reg, seed=0, stratify=None)


def test_month_stratification_spreads_every_split_across_months():
    reg = _year()
    plain = sd.make_split(reg, seed=0)
    strat = sd.make_split(reg, seed=0, stratify="month")
    assert _month_tv(reg, strat) < _month_tv(reg, plain)
    assert _month_tv(reg, strat) < 0.10
    for s in ("val", "test"):
        months = set(reg[reg.harpnum.isin(strat[s]["harps"])].month)
        assert len(months) == 12, (s, sorted(months))


def test_stratified_split_still_balances_patches_and_has_no_leakage():
    reg = _year()
    split = sd.make_split(reg, seed=0, stratify="month")
    harps = [set(split[s]["harps"]) for s in ("train", "val", "test")]
    assert not (harps[0] & harps[1]) and not (harps[0] & harps[2]) and not (harps[1] & harps[2])
    assert set().union(*harps) == set(reg.harpnum)
    total = reg.n_patches.sum()
    for s, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert split[s]["patches"] / total == pytest.approx(target, abs=0.03)


def test_month_column_is_derived_from_the_mdi_time():
    reg = pd.DataFrame({"mdi_t_rec": ["2010.05.01_00:00:00_TAI", "2011.02.28_22:24:00_TAI"]})
    assert list(sd.month_of(reg)) == ["2010.05", "2011.02"]
