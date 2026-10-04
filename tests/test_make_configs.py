"""Tests for scripts/make_configs.py (one config per experiment)."""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import make_configs as mc  # noqa: E402


def test_patch32_pair_is_added_only_with_a_lambda_and_a_patch_set():
    assert not any(k.endswith("_p32") for k in mc.experiments(0.1))
    assert not any(k.endswith("_p32") for k in mc.experiments(None, patch32_data="data/pilot_p32"))
    ex = mc.experiments(0.1, patch32_data="data/pilot_p32")
    assert ex["E1_p32"][:3] == ex["E1"]
    assert ex["E3_p32"][:3] == ex["E3_l0.1"]
    assert ex["E1_p32"][3] == ex["E3_p32"][3] == {"data_dir": "data/pilot_p32"}


def test_patch32_set_also_gets_e3_at_the_largest_lambda_as_a_diagnostic():
    ex = mc.experiments(0.01, patch32_data="data/pilot_p32")
    assert ex["E3_p32_l100"][:3] == ex["E3_l100"]
    assert ex["E3_p32_l100"][3] == {"data_dir": "data/pilot_p32"}
    # when the chosen lambda already is the largest, E3_p32 covers it
    assert not any(k.startswith("E3_p32_l") for k in mc.experiments(100.0, patch32_data="data/pilot_p32"))


def test_patch32_configs_use_the_patch_set_but_the_main_run_statistics(tmp_path):
    mc.main(["--stage", "pilot", "--run", "pilot", "--lambda_div", "1", "--patch32_run", "pilot_p32",
             "--out", str(tmp_path)])
    p32 = yaml.safe_load((tmp_path / "pilot_E3_p32.yaml").read_text())
    e3 = yaml.safe_load((tmp_path / "pilot_E3_l1.yaml").read_text())
    assert p32["data_dir"] == "data/pilot_p32"
    assert p32["stats"] == "data/pilot/stats.json" and p32["split"] == "splits/pilot.json"
    assert p32["out_dir"] == "results/pilot/E3_p32" and p32["exp"] == "E3_p32"
    assert p32["loss"] == e3["loss"] and p32["train"] == e3["train"] and p32["model"] == e3["model"]


def test_other_configs_are_unchanged_by_the_patch32_option(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    mc.main(["--stage", "pilot", "--run", "pilot", "--lambda_div", "1", "--patch32_run", "none", "--out", str(a)])
    mc.main(["--stage", "pilot", "--run", "pilot", "--lambda_div", "1", "--patch32_run", "pilot_p32", "--out", str(b)])
    names_a = sorted(os.listdir(a))
    assert sorted(set(os.listdir(b)) - set(names_a)) == ["pilot_E1_p32.yaml", "pilot_E3_p32.yaml",
                                                         "pilot_E3_p32_l100.yaml"]
    for n in names_a:
        assert (a / n).read_text() == (b / n).read_text()


def test_auto_adds_the_pair_only_when_the_patch_set_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "ROOT", str(tmp_path))
    assert mc.patch32_data("auto", "pilot", 1.0, "pilot") is None
    (tmp_path / "data" / "pilot_p32").mkdir(parents=True)
    (tmp_path / "data" / "pilot_p32" / "manifest.csv").write_text("harpnum,patch_file\n")
    assert mc.patch32_data("auto", "pilot", 1.0, "pilot") == "data/pilot_p32"
    assert mc.patch32_data("auto", "pilot", None, "pilot") is None
    assert mc.patch32_data("auto", "pilot", 1.0, "full") is None
    assert mc.patch32_data("none", "pilot", 1.0, "pilot") is None


def test_hist_sweep_adds_e2_variants_that_differ_only_in_the_histogram_weight():
    assert not any(k.startswith("E2_h") for k in mc.experiments(0.01))
    ex = mc.experiments(0.01, hist_sweep=True)
    names = sorted(k for k in ex if k.startswith("E2_h"))
    assert names == sorted(f"E2_h{w:g}" for w in mc.HIST_WEIGHTS)
    model, weights, mode = ex["E2"]
    for w in mc.HIST_WEIGHTS:
        m2, w2, mode2 = ex[f"E2_h{w:g}"]
        assert (m2, mode2) == (model, mode)
        assert w2 == {**weights, "hist": w}


def test_hist_sweep_leaves_the_other_configs_unchanged(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    mc.main(["--stage", "pilot", "--run", "pilot", "--lambda_div", "0.01", "--patch32_run", "none", "--out", str(a)])
    mc.main(["--stage", "pilot", "--run", "pilot", "--lambda_div", "0.01", "--patch32_run", "none", "--hist_sweep",
             "--out", str(b)])
    names_a = sorted(os.listdir(a))
    assert sorted(set(os.listdir(b)) - set(names_a)) == sorted(f"pilot_E2_h{w:g}.yaml" for w in mc.HIST_WEIGHTS)
    for n in names_a:
        assert (a / n).read_text() == (b / n).read_text()


# ---------------------------------------------------------------- full stage (Phase 8)

def _full(tmp_path, *extra):
    mc.main(["--stage", "full", "--run", "full", "--lambda_div", "0.01", "--seeds", "0", "1", "2",
             "--samples_per_epoch", "262144", "--out", str(tmp_path), *extra])
    return {p.name: yaml.safe_load(p.read_text()) for p in tmp_path.iterdir()}


def test_full_stage_writes_the_plan_experiments_with_three_seeds(tmp_path):
    cfgs = _full(tmp_path)
    exps = sorted({c["exp"].rsplit("_s", 1)[0] for c in cfgs.values()})
    assert exps == ["E1", "E2", "E3", "E4", "E7", "E8"]
    assert len(cfgs) == 6 * 3
    assert sorted(c["train"]["seed"] for c in cfgs.values() if c["exp"].startswith("E1_")) == [0, 1, 2]


def test_full_stage_optional_e5_e6(tmp_path):
    cfgs = _full(tmp_path, "--with_e5_e6")
    exps = sorted({c["exp"].rsplit("_s", 1)[0] for c in cfgs.values()})
    assert exps == ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]


def test_full_stage_uses_regions_mode_and_the_chosen_weights(tmp_path):
    cfgs = _full(tmp_path, "--hist_weight", "0.001")
    e3 = cfgs["full_E3_s0.yaml"]
    assert e3["data"] == {"mode": "regions"} and e3["train"]["samples_per_epoch"] == 262144
    assert e3["data_dir"] == "data/full" and e3["split"] == "splits/full.json"
    assert e3["out_dir"] == "results/full/E3_s0"
    pilot_e3 = mc.experiments(0.01)["E3_l0.01"]
    assert e3["loss"]["weights"] == {k: pilot_e3[1].get(k, 0.0) for k in ("mse", "grad", "hist", "ssim", "div")}
    assert cfgs["full_E2_s1.yaml"]["loss"]["weights"]["hist"] == 0.001
    assert cfgs["full_E4_s2.yaml"]["loss"]["weights"] == {"mse": 1.0, "grad": 0.0, "hist": 0.0, "ssim": 0.0,
                                                          "div": 0.01}


def test_full_stage_needs_a_lambda(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        mc.main(["--stage", "full", "--run", "full", "--out", str(tmp_path)])


def test_pilot_configs_have_no_data_section(tmp_path):
    mc.main(["--stage", "pilot", "--run", "pilot", "--out", str(tmp_path)])
    assert all("data" not in yaml.safe_load(p.read_text()) for p in tmp_path.iterdir())


# ---------------------------------------------------------------- multi-scale divergence sweep (phase_6.md 8)

def test_ms_sweep_writes_e3_and_e4_with_the_multiscale_divergence(tmp_path):
    import json
    stats = tmp_path / "div_ms_stats.json"
    stats.write_text(json.dumps({"scales_px": [1.0, 2.0, 4.0], "mean_square_div": [6e-5, 2e-5, 9e-6]}))
    out = tmp_path / "cfg"
    mc.main(["--stage", "pilot", "--run", "pilot", "--patch32_run", "none", "--ms_sweep", str(stats),
             "--out", str(out)])
    names = sorted(p.name for p in out.iterdir() if "_ms_" in p.name)
    assert names == sorted(f"pilot_{e}_ms_l{l:g}.yaml" for e in ("E3", "E4") for l in mc.MS_LAMBDAS)
    e3 = yaml.safe_load((out / "pilot_E3_ms_l0.01.yaml").read_text())
    e3_pix = yaml.safe_load((out / "pilot_E3_l0.01.yaml").read_text())
    assert e3["loss"]["div_mode"] == "match_ms"
    assert e3["loss"]["div_scales"] == [1.0, 2.0, 4.0] and e3["loss"]["div_scale_norm"] == [6e-5, 2e-5, 9e-6]
    assert e3["loss"]["weights"] == e3_pix["loss"]["weights"]           # same terms, only the divergence differs
    e4 = yaml.safe_load((out / "pilot_E4_ms_l0.1.yaml").read_text())
    assert e4["loss"]["weights"] == {"mse": 1.0, "grad": 0.0, "hist": 0.0, "ssim": 0.0, "div": 0.1}
    assert e4["out_dir"] == "results/pilot/E4_ms_l0.1"


def test_ms_sweep_leaves_the_other_configs_unchanged(tmp_path):
    import json
    stats = tmp_path / "s.json"
    stats.write_text(json.dumps({"scales_px": [1.0], "mean_square_div": [1e-5]}))
    a, b = tmp_path / "a", tmp_path / "b"
    mc.main(["--stage", "pilot", "--run", "pilot", "--patch32_run", "none", "--out", str(a)])
    mc.main(["--stage", "pilot", "--run", "pilot", "--patch32_run", "none", "--ms_sweep", str(stats), "--out", str(b)])
    for n in os.listdir(a):
        assert (a / n).read_text() == (b / n).read_text()


# ---------------------------------------------------------------- U-Net test (phase_6.md 9)

def test_unet_test_configs_differ_from_the_highresnet_ones_only_in_the_model(tmp_path):
    import json
    stats = tmp_path / "s.json"
    stats.write_text(json.dumps({"scales_px": [1.0, 2.0, 4.0], "mean_square_div": [6e-5, 2e-5, 9e-6]}))
    out = tmp_path / "cfg"
    mc.main(["--stage", "pilot", "--run", "pilot", "--patch32_run", "none", "--hist_sweep", "--ms_sweep", str(stats),
             "--unet_test", "--out", str(out)])
    load = lambda n: yaml.safe_load((out / f"pilot_{n}.yaml").read_text())
    for u, h in (("U_E3_ms_l0.01", "E3_ms_l0.01"), ("U_E2_h0.001", "E2_h0.001")):
        cu, ch = load(u), load(h)
        assert cu["model"]["name"] == "unet" and ch["model"]["name"] == "highresnet"
        assert cu["loss"] == ch["loss"] and cu["train"] == ch["train"] and cu["out_dir"] == f"results/pilot/{u}"
    assert load("U_E2_h0.001")["model"] == load("E7")["model"]


# ---------------------------------------------------------------- pilot v2 (geometry + augmentation, Munoz setup)

def test_v2_configs_share_geometry_augmentation_and_four_input_channels(tmp_path):
    import json
    stats = tmp_path / "s.json"
    stats.write_text(json.dumps({"scales_px": [1.0, 2.0, 4.0], "mean_square_div": [6e-5, 2e-5, 9e-6]}))
    out = tmp_path / "cfg"
    mc.main(["--stage", "pilot", "--run", "pilot", "--patch32_run", "none", "--v2", str(stats), "--out", str(out)])
    v2 = {p.name[len("pilot_"):-len(".yaml")]: yaml.safe_load(p.read_text()) for p in out.iterdir() if p.name.startswith("pilot_V_")}
    assert sorted(v2) == sorted(list(mc.V2_EXPERIMENTS) + list(mc.V2_ABLATION) + list(mc.V2_CONTROLS))
    v2 = {n: c for n, c in v2.items() if n in mc.V2_EXPERIMENTS}
    for name, c in v2.items():
        assert c["data"] == {"geometry": "data/pilot/geometry_lr", "augment": True}
        assert c["model"]["kwargs"] == {"in_channels": 4}
        assert c["out_dir"] == f"results/pilot/{name}"
    w = lambda n: v2[n]["loss"]["weights"]
    assert w("V_E1") == {"mse": 1.0, "grad": 0.0, "hist": 0.0, "ssim": 0.0, "div": 0.0}
    assert w("V_E2") == {"mse": 1.0, "grad": 5.0, "hist": 1e-3, "ssim": 5e-4, "div": 0.0}
    assert w("V_E3") == {"mse": 1.0, "grad": 5.0, "hist": 0.0, "ssim": 5e-4, "div": 0.01}
    assert w("V_E6") == {"mse": 1.0, "grad": 5.0, "hist": 1e-3, "ssim": 5e-4, "div": 0.01}
    assert w("V_Grad") == {"mse": 1.0, "grad": 5.0, "hist": 0.0, "ssim": 0.0, "div": 0.0}
    assert w("V_GradHist") == {"mse": 1.0, "grad": 5.0, "hist": 1e-3, "ssim": 0.0, "div": 0.0}
    for n in ("V_E3", "V_E6", "V_U_E3"):
        assert v2[n]["loss"]["div_mode"] == "match_ms" and v2[n]["loss"]["div_scales"] == [1.0, 2.0, 4.0]
    assert v2["V_E7"]["model"]["name"] == v2["V_U_E3"]["model"]["name"] == "unet"
    assert w("V_U_E3") == w("V_E3") and w("V_E7") == w("V_E1")
    assert all(v2[n]["model"]["name"] == "highresnet" for n in v2 if n not in ("V_E7", "V_U_E3"))


def test_v2_leaves_the_other_configs_unchanged(tmp_path):
    import json
    stats = tmp_path / "s.json"
    stats.write_text(json.dumps({"scales_px": [1.0], "mean_square_div": [1e-5]}))
    a, b = tmp_path / "a", tmp_path / "b"
    mc.main(["--stage", "pilot", "--run", "pilot", "--patch32_run", "none", "--out", str(a)])
    mc.main(["--stage", "pilot", "--run", "pilot", "--patch32_run", "none", "--v2", str(stats), "--out", str(b)])
    for n in os.listdir(a):
        assert (a / n).read_text() == (b / n).read_text()


def test_v2_ablation_configs_separate_geometry_and_augmentation(tmp_path):
    import json
    stats = tmp_path / "s.json"
    stats.write_text(json.dumps({"scales_px": [1.0], "mean_square_div": [1e-5]}))
    out = tmp_path / "cfg"
    mc.main(["--stage", "pilot", "--run", "pilot", "--patch32_run", "none", "--v2", str(stats), "--out", str(out)])
    load = lambda n: yaml.safe_load((out / f"pilot_{n}.yaml").read_text())
    geo, aug, both = load("V_E1_geo"), load("V_E1_aug"), load("V_E1")
    assert geo["data"] == {"geometry": "data/pilot/geometry_lr", "augment": False}
    assert geo["model"]["kwargs"] == {"in_channels": 4}
    assert aug["data"] == {"augment": True} and aug["model"]["kwargs"] == {}
    for c in (geo, aug):
        assert c["loss"] == both["loss"] and c["train"] == both["train"] and c["model"]["name"] == both["model"]["name"]


def test_v2_has_the_unet_munoz_control(tmp_path):
    import json
    stats = tmp_path / "s.json"
    stats.write_text(json.dumps({"scales_px": [1.0], "mean_square_div": [1e-5]}))
    out = tmp_path / "cfg"
    mc.main(["--stage", "pilot", "--run", "pilot", "--patch32_run", "none", "--v2", str(stats), "--out", str(out)])
    load = lambda n: yaml.safe_load((out / f"pilot_{n}.yaml").read_text())
    u2, v2 = load("V_U_E2"), load("V_E2")
    assert u2["model"] == {"name": "unet", "kwargs": {"in_channels": 4}}
    assert u2["loss"] == v2["loss"] and u2["data"] == v2["data"] and u2["train"] == v2["train"]


# ---------------------------------------------------------------- Phase 8 with the v2 setup

def test_full_v2_configs(tmp_path):
    import json
    stats = tmp_path / "s.json"
    stats.write_text(json.dumps({"scales_px": [1.0, 2.0, 4.0], "mean_square_div": [6e-5, 2e-5, 9e-6]}))
    out = tmp_path / "cfg"
    mc.main(["--stage", "full", "--run", "full", "--seeds", "0", "1", "2", "--samples_per_epoch", "131072",
             "--full_v2", str(stats), "--out", str(out)])
    cfgs = {p.name[len("full_"):-len(".yaml")]: yaml.safe_load(p.read_text()) for p in out.iterdir()}
    assert sorted(cfgs) == sorted(f"{e}_s{s}" for e in mc.FULL_V2 for s in (0, 1, 2))
    for name, c in cfgs.items():
        assert c["data"] == {"mode": "regions", "geometry": "data/full/geometry_lr", "augment": True, "val_every": 3}
        assert c["model"]["kwargs"] == {"in_channels": 4} and c["train"]["samples_per_epoch"] == 131072
        assert c["split"] == "splits/full.json" and c["stats"] == "data/full/stats.json"
    pilot = {n: mc.V2_EXPERIMENTS[n] for n in ("V_E1", "V_E2", "V_E3", "V_E7", "V_U_E3")}
    same = {"F_E1": "V_E1", "F_E2": "V_E2", "F_E3": "V_E3", "F_U_E1": "V_E7", "F_U_E3": "V_U_E3"}
    for f, v in same.items():
        c = cfgs[f"{f}_s1"]
        assert c["model"]["name"] == pilot[v][0]
        assert c["loss"]["weights"] == {k: pilot[v][1].get(k, 0.0) for k in ("mse", "grad", "hist", "ssim", "div")}
    assert cfgs["F_U_E2_s0"]["model"]["name"] == "unet" and cfgs["F_U_E2_s0"]["loss"]["weights"] == cfgs["F_E2_s0"]["loss"]["weights"]
    assert cfgs["F_U_E3_s2"]["loss"]["div_mode"] == "match_ms" and cfgs["F_U_E3_s2"]["loss"]["div_scale_norm"] == [6e-5, 2e-5, 9e-6]
