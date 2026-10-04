"""Write one YAML config per trained experiment from a single table, so every
run is identical except for the loss (and, for E7/E8, the architecture).

    python scripts/make_configs.py --stage pilot --run pilot
    python scripts/make_configs.py --stage pilot --run pilot --lambda_div 0.1   # adds E4, E5, E6
    python scripts/make_configs.py --stage full  --run full --lambda_div 0.1 --seeds 0 1 2

Patch-size check (plan 2b): with --lambda_div at the pilot stage, E1_p32 and
E3_p32 (E1 and E3 at the chosen lambda, trained on 32x32 LR / 128x128 HR
patches from scripts/make_patch_set.py) are added when data/<run>_p32/manifest.csv
exists (--patch32_run auto) or when a patch-set run is named explicitly. They
use the main run's split and training statistics. E3_p32_l100 (E3 at the
largest grid lambda on the same patches) is added as a diagnostic unless the
chosen lambda already is the largest.

Loss weights (training_plan.md 2, fairness protocol): E2 uses the converter
repo's MDI weights (grad 5, hist 1e-5, ssim 5e-5; reports/phase_0.md D2);
E3/E5/E6 reuse exactly those for the shared terms; lambda_div is tuned only
for E3 over {0.01, 0.1, 1, 10, 100} (plan grid widened: at <= 1 the term is
< ~2.5% of the loss); E4/E5/E6 reuse E3's chosen value. Histogram loss uses the
normalised total-variation distance (hist_mode tv) by default.
"""
import argparse
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUNOZ = {"grad": 5.0, "hist": 1e-5, "ssim": 5e-5}
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)   # plan grid + 10, 100 (decision 2026-09-23, reports/phase_0.md 9b)
# E2 histogram-weight sweep (reports/phase_6.md 4b): with hist_mode tv the repo weight 1e-5 makes the
# term ~0.01% of the loss; TV ~0.4, so w ~1e-3 puts it at the order of the MSE (~5e-4).
HIST_WEIGHTS = (1e-4, 1e-3, 1e-2, 1e-1)
# multi-scale divergence (reports/phase_6.md 8): L_div_ms is ~0.5-1 when normalised per scale; E3's
# other terms sum to ~0.047, so these put it at ~1.5%, ~13% and ~60% of E3's loss
MS_LAMBDAS = (0.001, 0.01, 0.1)
# pilot v2 (reports/phase_6.md 11): viewing-geometry input + vector-aware augmentation for every model;
# Munoz's paper weights (Table 2: grad 5, SSIM 5e-4) with the histogram at the paper's balance rule (1e-3 for
# the TV histogram); his ablation steps; blurred divergence (match_ms, lambda 0.01) replacing / adding to the histogram
MUNOZ_PAPER = {"grad": 5.0, "hist": 1e-3, "ssim": 5e-4}
V2_EXPERIMENTS = {
    "V_E1": ("highresnet", {"mse": 1.0}, "match"),
    "V_E2": ("highresnet", {"mse": 1.0, **MUNOZ_PAPER}, "match"),
    "V_E3": ("highresnet", {"mse": 1.0, "grad": 5.0, "ssim": 5e-4, "div": 0.01}, "match_ms"),
    "V_E7": ("unet", {"mse": 1.0}, "match"),
    "V_U_E3": ("unet", {"mse": 1.0, "grad": 5.0, "ssim": 5e-4, "div": 0.01}, "match_ms"),
    "V_E6": ("highresnet", {"mse": 1.0, **MUNOZ_PAPER, "div": 0.01}, "match_ms"),
    "V_Grad": ("highresnet", {"mse": 1.0, "grad": 5.0}, "match"),
    "V_GradHist": ("highresnet", {"mse": 1.0, "grad": 5.0, "hist": 1e-3}, "match"),
}
# Phase 8 with the pilot-v2 setup (regions mode, geometry, augmentation): the three losses on both networks
FULL_V2 = {
    "F_U_E1": ("unet", {"mse": 1.0}, "match"),
    "F_U_E2": ("unet", {"mse": 1.0, **MUNOZ_PAPER}, "match"),
    "F_U_E3": ("unet", {"mse": 1.0, "grad": 5.0, "ssim": 5e-4, "div": 0.01}, "match_ms"),
    "F_E1": ("highresnet", {"mse": 1.0}, "match"),
    "F_E2": ("highresnet", {"mse": 1.0, **MUNOZ_PAPER}, "match"),
    "F_E3": ("highresnet", {"mse": 1.0, "grad": 5.0, "ssim": 5e-4, "div": 0.01}, "match_ms"),
}
# Phase 8 controls (added 2026-09-26 after the validation results): F_U_E3 bundles Sobel + SSIM + divergence, so
# E4 = MSE + divergence only, E5 = MSE + Sobel + SSIM only (no divergence). Same setup as FULL_V2.
FULL_V2_CONTROLS = {
    "F_U_E4": ("unet", {"mse": 1.0, "div": 0.01}, "match_ms"),
    "F_U_E5": ("unet", {"mse": 1.0, "grad": 5.0, "ssim": 5e-4}, "match"),
}
# added control (phase_6.md 11b): Munoz's paper loss on the U-Net, v2 setup
V2_CONTROLS = {"V_U_E2": ("unet", {"mse": 1.0, **MUNOZ_PAPER}, "match")}
# which of Munoz's two setup changes helps (phase_6.md 11): V_E1 with only one of them
V2_ABLATION = {
    "V_E1_geo": {"geometry": True, "augment": False},
    "V_E1_aug": {"geometry": False, "augment": True},
}


def experiments(lambda_div=None, patch32_data=None, hist_sweep=False):
    ex = {
        "E1": ("highresnet", {"mse": 1.0}, "match"),
        "E2": ("highresnet", {"mse": 1.0, **MUNOZ}, "match"),
        "E7": ("unet", {"mse": 1.0}, "match"),
        "E8": ("edsr", {"mse": 1.0}, "match"),
    }
    if hist_sweep:
        for w in HIST_WEIGHTS:
            ex[f"E2_h{w:g}"] = ("highresnet", {"mse": 1.0, **MUNOZ, "hist": w}, "match")
    for lam in LAMBDAS:
        ex[f"E3_l{lam:g}"] = ("highresnet", {"mse": 1.0, "grad": MUNOZ["grad"], "ssim": MUNOZ["ssim"],
                                             "div": lam}, "match")
    if lambda_div is not None:
        ex["E4"] = ("highresnet", {"mse": 1.0, "div": lambda_div}, "match")
        ex["E5"] = ("highresnet", {"mse": 1.0, "grad": MUNOZ["grad"], "ssim": MUNOZ["ssim"],
                                   "div": lambda_div}, "zero")
        ex["E6"] = ("highresnet", {"mse": 1.0, **MUNOZ, "div": lambda_div}, "match")
        if patch32_data:
            ex["E1_p32"] = ex["E1"] + ({"data_dir": patch32_data},)
            ex["E3_p32"] = ex[f"E3_l{lambda_div:g}"] + ({"data_dir": patch32_data},)
            lmax = max(LAMBDAS)   # diagnostic: does more context let the strongest L_div act?
            if lambda_div != lmax:
                ex[f"E3_p32_l{lmax:g}"] = ex[f"E3_l{lmax:g}"] + ({"data_dir": patch32_data},)
    return ex


def full_experiments(lambda_div, hist_weight=None, with_e5_e6=False):
    """Phase 8: E1, E2, E3, E4, E7, E8 (+ E5, E6), E3/E4/E5/E6 at the pilot's chosen
    lambda and E2/E6 at the chosen histogram weight (default: the repo weight)."""
    pilot = experiments(lambda_div)
    hist = MUNOZ["hist"] if hist_weight is None else hist_weight
    ex = {"E1": pilot["E1"],
          "E2": ("highresnet", {**pilot["E2"][1], "hist": hist}, "match"),
          "E3": pilot[f"E3_l{lambda_div:g}"] if f"E3_l{lambda_div:g}" in pilot else
          ("highresnet", {"mse": 1.0, "grad": MUNOZ["grad"], "ssim": MUNOZ["ssim"], "div": lambda_div}, "match"),
          "E4": pilot["E4"], "E7": pilot["E7"], "E8": pilot["E8"]}
    if with_e5_e6:
        ex["E5"] = pilot["E5"]
        ex["E6"] = ("highresnet", {**pilot["E6"][1], "hist": hist}, "match")
    return ex


def patch32_data(option, run, lambda_div, stage):
    """data_dir of the 32x32 patch set to add (or None): 'none', 'auto' (data/<run>_p32
    if its manifest exists, pilot stage with a chosen lambda only), or a run name."""
    if option == "none" or lambda_div is None:
        return None
    if option == "auto":
        if stage != "pilot" or not os.path.exists(os.path.join(ROOT, "data", f"{run}_p32", "manifest.csv")):
            return None
        return f"data/{run}_p32"
    return f"data/{option}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pilot", "full"), required=True)
    ap.add_argument("--run", required=True, help="dataset name under data/")
    ap.add_argument("--lambda_div", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--results_run", default=None, help="results/<results_run>/<exp> (default: --run)")
    ap.add_argument("--out", default=os.path.join(ROOT, "configs"))
    ap.add_argument("--hist_mode", choices=("counts", "tv"), default="tv",
                    help="histogram loss scale: 'counts' (D4, raw counts on a 64x128^2 budget) "
                         "or 'tv' (normalised total-variation distance)")
    ap.add_argument("--hist_sweep", action="store_true",
                    help="also write E2_h<w> (E2 with histogram weight w in HIST_WEIGHTS)")
    ap.add_argument("--ms_sweep", default=None,
                    help="path to div_ms_stats.json: also write E3_ms_l<lam>, E4_ms_l<lam> (match_ms divergence)")
    ap.add_argument("--v2", default=None,
                    help="path to div_ms_stats.json: also write the pilot-v2 configs (V2_EXPERIMENTS)")
    ap.add_argument("--full_controls", action="store_true", help="with --full_v2: also write F_U_E4 / F_U_E5")
    ap.add_argument("--full_v2", default=None,
                    help="stage full: path to the run's div_ms_stats.json; writes FULL_V2 (no --lambda_div needed)")
    ap.add_argument("--unet_test", action="store_true",
                    help="also write U_E3_ms_l0.01 and U_E2_h0.001 (the E3_ms / E2_h0.001 losses on the U-Net; "
                         "needs --ms_sweep)")
    ap.add_argument("--hist_weight", type=float, default=None,
                    help="full stage: E2/E6 histogram weight (from results/pilot/hist_selection.json)")
    ap.add_argument("--with_e5_e6", action="store_true", help="full stage: also E5 and E6")
    ap.add_argument("--data_mode", choices=("patches", "regions"), default=None,
                    help="default: patches for pilot, regions for full")
    ap.add_argument("--samples_per_epoch", type=int, default=None,
                    help="regions mode: random crops per epoch (default: number of grid patches)")
    ap.add_argument("--patch32_run", default="auto",
                    help="'auto' (default), 'none', or the run name of a 32x32 patch set")
    args = ap.parse_args(argv)
    if args.stage == "full" and args.lambda_div is None and not args.full_v2:
        ap.error("--stage full needs --lambda_div (the pilot's chosen value)")
    os.makedirs(args.out, exist_ok=True)
    data_dir = args.data_dir or f"data/{args.run}"
    data_mode = args.data_mode or ("regions" if args.stage == "full" else "patches")
    written = []
    if args.stage == "full" and args.full_v2:
        with open(args.full_v2) as f:
            ms = json.load(f)
        ms_loss = {"div_scales": [float(x) for x in ms["scales_px"]],
                   "div_scale_norm": [float(x) for x in ms["mean_square_div"]]}
        exps = {}
        for name, (model, weights, mode) in {**FULL_V2, **(FULL_V2_CONTROLS if args.full_controls else {})}.items():
            over = {"data": {"geometry": f"data/{args.run}/geometry_lr", "augment": True, "val_every": 3},
                    "model_kwargs": {"in_channels": 4}}
            if mode == "match_ms":
                over["loss"] = ms_loss
            exps[name] = (model, weights, mode, over)
    elif args.stage == "full":
        exps = full_experiments(args.lambda_div, args.hist_weight, args.with_e5_e6)
    else:
        p32 = patch32_data(args.patch32_run, args.run, args.lambda_div, args.stage)
        exps = experiments(args.lambda_div, p32, args.hist_sweep)
        if args.v2:
            with open(args.v2) as f:
                ms = json.load(f)
            ms_loss = {"div_scales": [float(x) for x in ms["scales_px"]],
                       "div_scale_norm": [float(x) for x in ms["mean_square_div"]]}
            v2_over = {"data": {"geometry": f"data/{args.run}/geometry_lr", "augment": True},
                       "model_kwargs": {"in_channels": 4}}
            for name, (model, weights, mode) in V2_EXPERIMENTS.items():
                over = dict(v2_over)
                if mode == "match_ms":
                    over["loss"] = ms_loss
                exps[name] = (model, weights, mode, over)
            for name, (model, weights, mode) in V2_CONTROLS.items():
                exps[name] = (model, weights, mode, dict(v2_over))
            for name, flags in V2_ABLATION.items():
                model, weights, mode = V2_EXPERIMENTS["V_E1"]
                data = {"geometry": f"data/{args.run}/geometry_lr"} if flags["geometry"] else {}
                data["augment"] = flags["augment"]
                exps[name] = (model, weights, mode, {"data": data,
                                                     "model_kwargs": {"in_channels": 4} if flags["geometry"] else {}})
        if args.ms_sweep:
            with open(args.ms_sweep) as f:
                ms = json.load(f)
            ms_loss = {"div_mode": "match_ms", "div_scales": [float(x) for x in ms["scales_px"]],
                       "div_scale_norm": [float(x) for x in ms["mean_square_div"]]}
            for lam in MS_LAMBDAS:
                exps[f"E3_ms_l{lam:g}"] = ("highresnet", {"mse": 1.0, "grad": MUNOZ["grad"], "ssim": MUNOZ["ssim"],
                                                          "div": lam}, "match_ms", {"loss": ms_loss})
                exps[f"E4_ms_l{lam:g}"] = ("highresnet", {"mse": 1.0, "div": lam}, "match_ms", {"loss": ms_loss})
            if args.unet_test:
                exps["U_E3_ms_l0.01"] = ("unet",) + exps["E3_ms_l0.01"][1:]
                exps["U_E2_h0.001"] = ("unet", {"mse": 1.0, **MUNOZ, "hist": 1e-3}, "match")
    for exp, (model, weights, div_mode, *over) in exps.items():
        over = over[0] if over else {}
        for seed in args.seeds:
            name = exp if len(args.seeds) == 1 else f"{exp}_s{seed}"
            cfg = {"exp": name, "stage": args.stage,
                   "data_dir": over.get("data_dir", data_dir),
                   "split": args.split or f"splits/{args.run}.json",
                   "stats": args.stats or f"{data_dir}/stats.json",
                   "out_dir": f"results/{args.results_run or args.run}/{name}",
                   "model": {"name": model, "kwargs": dict(over.get("model_kwargs", {}))},
                   "loss": {"weights": {k: weights.get(k, 0.0) for k in ("mse", "grad", "hist", "ssim", "div")},
                            "div_mode": div_mode, "hist_mode": args.hist_mode, **over.get("loss", {})},
                   "train": {"batch_size": 64, "lr": 1.0e-4, "max_epochs": 100, "patience": 10,
                             "seed": seed, "num_workers": 4, "sampler": "auto", "max_share": 0.25}}
            if "data" in over:
                cfg["data"] = dict(over["data"])
            if data_mode == "regions":
                cfg["data"] = {"mode": "regions", **cfg.get("data", {})}
                if args.samples_per_epoch:
                    cfg["train"]["samples_per_epoch"] = args.samples_per_epoch
            path = os.path.join(args.out, f"{args.stage}_{name}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            written.append(os.path.relpath(path, ROOT))
    print("\n".join(written))


if __name__ == "__main__":
    main()
