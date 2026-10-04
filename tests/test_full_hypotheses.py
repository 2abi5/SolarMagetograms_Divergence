"""Tests for scripts/full_hypotheses.py (seed-averaged per-HARP tests)."""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import full_hypotheses as fh  # noqa: E402


def test_family_strips_the_seed_suffix():
    assert fh.family("F_U_E3_s2") == "F_U_E3" and fh.family("F_E1_s0") == "F_E1" and fh.family("E0c") == "E0c"


def test_per_harp_values_are_averaged_over_seeds():
    df = pd.DataFrame({"model": ["F_A_s0", "F_A_s1", "F_A_s0", "F_A_s1", "F_B_s0"],
                       "harpnum": [1, 1, 2, 2, 1], "x": [1.0, 3.0, 5.0, 7.0, 9.0]})
    got = fh.family_harp_mean(df, "x")
    assert got.loc[("F_A", 1)] == 2.0 and got.loc[("F_A", 2)] == 6.0 and got.loc[("F_B", 1)] == 9.0


def test_paired_test_direction_and_non_inferiority():
    a = pd.Series([0.80, 0.81, 0.79, 0.82, 0.80, 0.83, 0.81, 0.80], index=range(8))
    b = a - 0.01
    r = fh.paired(a, b, higher_better=True)
    assert r["median_diff"] == pytest.approx(0.01) and r["a_better_in"] == 8 and r["p_wilcoxon"] < 0.05
    assert fh.non_inferior(a * 100, a * 100 * 1.005, margin=0.01)       # a 0.5% worse RMSE is within 1%
    assert not fh.non_inferior(a * 100 * 1.02, a * 100, margin=0.01)    # a 2% worse RMSE is not
