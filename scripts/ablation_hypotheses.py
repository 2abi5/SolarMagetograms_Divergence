"""Loss-term ablation (F_U_E4 = MSE+divergence only; F_U_E5 = MSE+grad+SSIM, no divergence, no histogram):
isolates whether the divergence term, not just the grad+SSIM regularisers, drives the Phase 8 result.
Reuses the paired-Wilcoxon machinery from scripts/full_hypotheses.py.

    python scripts/ablation_hypotheses.py --evals results/full/eval_val_s0 ... results/full/eval_controls_val_s0 ... --out results/full/hypotheses_ablation_val.json
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_hypotheses import family, family_harp_mean, paired, non_inferior  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    div = pd.concat([pd.read_csv(os.path.join(d, "div_by_scale_per_harp.csv")) for d in a.evals])
    per = pd.concat([pd.read_csv(os.path.join(d, "per_region.csv")) for d in a.evals])
    divs = {mk: family_harp_mean(div[(div["mask"] == mk) & div.sigma_px.isin([2.0, 4.0])], "corr_pred_true")
            for mk in ("all", "strong")}
    allp = per[per["mask"] == "all"].copy()
    allp["rmse_mean"] = allp[["Bp_rmse", "Bt_rmse", "Br_rmse"]].mean(axis=1)
    metric = {m: family_harp_mean(allp, m) for m in ("rmse_mean", "Br_rmse", "Bp_rmse", "Bt_rmse",
                                                      "Br_pearson", "Bp_pearson", "Bt_pearson")}
    g = lambda s, f: s.xs(f, level="family")
    out = {"n_harps": int(div.harpnum.nunique()), "comparisons": {}}
    comparisons = [
        ("E3 vs E5 (divergence's own effect, grad+SSIM held fixed)", "F_U_E3", "F_U_E5"),
        ("E4 vs E1 (divergence alone vs plain MSE)", "F_U_E4", "F_U_E1"),
        ("E4 vs E3 (does adding grad+SSIM on top of divergence help further)", "F_U_E3", "F_U_E4"),
        ("E5 vs E1 (grad+SSIM alone, no divergence, no histogram, vs plain MSE)", "F_U_E5", "F_U_E1"),
    ]
    for key, x, y in comparisons:
        res = {"div_corr_all": paired(g(divs["all"], x), g(divs["all"], y)),
               "div_corr_strong": paired(g(divs["strong"], x), g(divs["strong"], y))}
        for m, s in metric.items():
            res[m] = paired(g(s, x), g(s, y), higher_better=m.endswith("pearson"))
        t = res["div_corr_all"]
        res["div_higher_for_first"] = bool(t.get("p_wilcoxon", 1) < 0.05 and t.get("median_diff", 0) > 0)
        out["comparisons"][key] = res
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, r in out["comparisons"].items():
        t = r["div_corr_all"]
        print(f"{k}: div {t.get('median_diff', float('nan')):+.4f} "
              f"({t.get('a_better_in')}/{t.get('n_harps')}, p={t.get('p_wilcoxon', float('nan')):.3g}) "
              f"| rmse_mean {r['rmse_mean'].get('median_diff', float('nan')):+.2f} G "
              f"({r['rmse_mean'].get('a_better_in')}/{r['rmse_mean'].get('n_harps')})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
