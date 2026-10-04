"""Smoke test: evaluate(..., make_figures=True) writes every figure type."""
import os

from sr import evaluate as ev
from tests.test_evaluate import _checkpoint, _synthetic_run


def test_all_figures_are_written(tmp_path):
    run = _synthetic_run(str(tmp_path))
    for e in ("E1", "E2", "E3_l0.1"):
        _checkpoint(str(tmp_path), e)
    out = str(tmp_path / "eval_val")
    ev.evaluate(run_dir=run, split_json=str(tmp_path / "split.json"), split="val",
                results_dir=str(tmp_path / "results"), exps=["E1", "E2", "E3_l0.1"], out_dir=out,
                device="cpu", make_figures=True, n_boot=100)
    figs = sorted(os.listdir(os.path.join(out, "figures")))
    for prefix in ("panels_", "div_error_", "histograms", "scatter", "metrics_bar", "power_spectra_"):
        assert any(f.startswith(prefix) and f.endswith(".png") for f in figs), (prefix, figs)
