#!/usr/bin/env python3
"""results/ -> tables/*.tex, tables/numbers.tex and figures/data/*.csv.

    cd paper && ../.venv-mag/bin/python scripts/make_tables.py     (or: make tables)

Never hand-type a number into the paper: every table cell and every number in
the text is a macro from tables/numbers.tex, written here. Before writing, the
recomputed paired tests are checked against the pre-registered analysis outputs
(results/full/hypotheses_*.json); any disagreement stops the script.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperdata as P  # noqa: E402

TAB = os.path.join(P.PAPER, "tables")
FIGDATA = os.path.join(P.PAPER, "figures", "data")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIGDATA, exist_ok=True)

MACROS = {}


def mac(name, value):
    assert name.isalpha(), name
    assert name not in MACROS, f"duplicate macro {name}"
    MACROS[name] = value
    return value


# ---------------------------------------------------------------- formatting
def minus(s):
    return s.replace("-", "\\textminus{}") if s.startswith("-") else s


def f(x, nd):
    return minus(f"{x:.{nd}f}")


def sf(x, nd):
    """Signed, e.g. +0.0089 / \\textminus 0.17."""
    return minus(f"{x:+.{nd}f}")


def fp(p):
    if p >= 0.01:
        return f"{p:.2f}"
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"\\ensuremath{{{m:.1f}{{\\times}}10^{{{e}}}}}"


def frac(r):
    return f"{r['a_better_in']}/{r['n_harps']}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def ints(n):
    return f"{n:,}".replace(",", "{,}")


# ---------------------------------------------------------------- data
print("loading evaluations ...")
T = {s: P.harp_table(s) for s in ("val", "test")}
FH = {s: P.family_harp(T[s]) for s in ("val", "test")}
SPL = {"val": "Val", "test": "Test"}

# ---------------------------------------------------------------- consistency with the pre-registered outputs
COMP = {  # macro key -> (a, b, key in hypotheses_*.json)
    "UOursMSE": ("F_U_E3", "F_U_E1", "P8-H1 F_U_E3 vs F_U_E1"),
    "UOursMZ": ("F_U_E3", "F_U_E2", "P8-H2 F_U_E3 vs F_U_E2"),
    "HOursMSE": ("F_E3", "F_E1", "P8-H4a F_E3 vs F_E1"),
    "HOursMZ": ("F_E3", "F_E2", "P8-H4b F_E3 vs F_E2"),
    "UMZMSE": ("F_U_E2", "F_U_E1", "U-Net Munoz vs MSE"),
}
ABL = {
    "AblDivOwn": ("F_U_E3", "F_U_E5", "E3 vs E5 (divergence's own effect, grad+SSIM held fixed)"),
    "AblDivOnly": ("F_U_E4", "F_U_E1", "E4 vs E1 (divergence alone vs plain MSE)"),
    "AblAddGS": ("F_U_E3", "F_U_E4", "E4 vs E3 (does adding grad+SSIM on top of divergence help further)"),
    "AblGS": ("F_U_E5", "F_U_E1", "E5 vs E1 (grad+SSIM alone, no divergence, no histogram, vs plain MSE)"),
}
HYP = {"val": json.load(open(os.path.join(P.RES, "full", "hypotheses_full.json"))),
       "test": json.load(open(os.path.join(P.RES, "full", "hypotheses_test.json")))}
AHYP = {"val": json.load(open(os.path.join(P.RES, "full", "hypotheses_ablation_val.json"))),
        "test": json.load(open(os.path.join(P.RES, "full", "hypotheses_ablation_test.json")))}


def check(mine, ref, what):
    for k in ("median_diff", "a_better_in", "n_harps", "p_wilcoxon"):
        a, b = mine[k], ref[k]
        if not np.isclose(a, b, rtol=1e-9, atol=1e-12):
            raise SystemExit(f"MISMATCH {what} {k}: recomputed {a} vs pre-registered output {b}")


JKEY = {"div_all": "div_corr_all", "div_strong": "div_corr_strong"}  # our name -> key in the json
n_checked = 0
for s in ("val", "test"):
    for key, (a, b, hk) in COMP.items():
        for m in ("div_all", "div_strong", "rmse_mean", "Br_rmse", "Bp_hist_tv", "Bt_hist_tv", "Br_extreme_err_32"):
            check(P.paired(FH[s], m, a, b), HYP[s]["comparisons"][hk][JKEY.get(m, m)], f"{s} {hk} {m}")
            n_checked += 1
    for key, (a, b, hk) in ABL.items():
        # note: the json entry labelled "E4 vs E3" is computed as E3 - E4 (scripts/ablation_hypotheses.py)
        ref = AHYP[s]["comparisons"][hk]
        for m in ("div_all", "div_strong", "rmse_mean", "Br_rmse"):
            check(P.paired(FH[s], m, a, b), ref[JKEY.get(m, m)], f"{s} {hk} {m}")
            n_checked += 1
print(f"recomputed tests match the pre-registered outputs ({n_checked} checks)")

# ---------------------------------------------------------------- data set numbers
split = json.load(open(os.path.join(P.ROOT, "splits", "full.json")))
sp = split["splits"]
for k, name in (("train", "Train"), ("val", "Val"), ("test", "Test")):
    mac(f"nHarps{name}", str(len(sp[k]["harps"])))
    mac(f"nPairs{name}", ints(int(sp[k]["pairs"]) if not isinstance(sp[k]["pairs"], list) else len(sp[k]["pairs"])))
mac("nHarpsAll", str(sum(len(sp[k]["harps"]) for k in sp)))
mac("nPairsAll", ints(sum(int(sp[k]["pairs"]) if not isinstance(sp[k]["pairs"], list) else len(sp[k]["pairs"]) for k in sp)))
regs = pd.read_csv(os.path.join(P.ROOT, "data", "full", "regions.csv"))
skip = pd.read_csv(os.path.join(P.ROOT, "data", "full", "skipped.csv"))
assert len(regs) == sum(int(sp[k]["pairs"]) if not isinstance(sp[k]["pairs"], list) else len(sp[k]["pairs"]) for k in sp)
mac("nMdiFrames", ints(regs.mdi_t_rec.nunique()))
mac("dateFirst", str(regs.day.min()))
mac("dateLast", str(regs.day.max()))
pregs = pd.read_csv(os.path.join(P.ROOT, "data", "pilot", "regions.csv"))
psplit = json.load(open(os.path.join(P.ROOT, "splits", "pilot.json")))["splits"]
mac("nPilotPairs", ints(len(pregs)))
mac("nPilotHarps", str(pregs.harpnum.nunique()))
mac("nPilotVal", str(len(psplit["val"]["harps"])))
mac("alignBefore", f(regs.align_corr_los_before.median(), 2))
mac("alignAfter", f(regs.align_corr_los.median(), 2))
mac("nMatched", ints(len(regs) + len(skip)))
for reason, n in skip.reason.value_counts().items():
    mac("nSkip" + "".join(w.capitalize() for w in str(reason).replace("-", "_").split("_") if w.isalpha()), ints(int(n)))
print("skip reasons:", skip.reason.value_counts().to_dict())
for s in ("val", "test"):
    summ = pd.read_csv(os.path.join(P.RES, "full", f"eval_{s}_s0", "summary.csv"))
    r = summ[(summ.model == "F_U_E3_s0") & (summ["mask"] == "all") & (summ.metric == "Br_rmse")].iloc[0]
    mac(f"nFrames{SPL[s]}", ints(int(r.n_regions)))
    assert int(r.n_harps) == len(sp[s]["harps"])

# ---------------------------------------------------------------- reliability of the target's divergence (pilot training pairs)
snr_json, snr = P.pilot_snr()
mac("nSnrPairs", str(snr_json["n_pairs"]))
WORD = {0: "Zero", 1: "One", 2: "Two", 4: "Four", 8: "Eight"}
for _, r in snr.iterrows():
    tag = {"all": "", "|B|>200G": "Strong"}.get(r.pixels)
    if tag is None:
        continue
    w = WORD[int(r.sigma_px)]
    mac(f"rel{tag}{w}", f(r["corr"], 2))
    mac(f"snr{tag}{w}", f(r.snr, 1))
    if tag == "":
        mac(f"sigMm{w}", f(r.sigma_mm, 2))
        mac(f"sigRms{w}", f(r.signal_rms_g_per_mm, 0))
        mac(f"noiseRms{w}", f(r.noise_rms_g_per_mm, 0))
        mac(f"ceil{w}", f(math.sqrt(r["corr"]), 2))
nf = json.load(open(os.path.join(P.ROOT, "data", "pilot", "noise_floor_temporal.json")))
mac("noiseFloorTemporal", f(nf["mean_abs_g_per_mm_mean_over_pairs"], 0))

with open(os.path.join(TAB, "snr.tex"), "w") as fh:
    fh.write("% GENERATED by scripts/make_tables.py from data/pilot/div_snr_by_scale.json -- do not edit\n")
    fh.write("\\begin{tabular}{rr cc cc rr}\n\\toprule\n")
    fh.write("\\multicolumn{2}{c}{Scale $\\sigma$} & \\multicolumn{2}{c}{All pixels} & "
             "\\multicolumn{2}{c}{$|\\mathbf{B}|>200$\\,G} & \\multicolumn{2}{c}{rms (G\\,Mm$^{-1}$)} \\\\\n")
    fh.write("\\cmidrule(lr){1-2}\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n")
    fh.write("px & Mm & $\\hat\\rho_\\sigma$ & SNR & $\\hat\\rho_\\sigma$ & SNR & signal & noise \\\\\n\\midrule\n")
    for s in (0.0, 1.0, 2.0, 4.0, 8.0):
        a = snr[(snr.pixels == "all") & (snr.sigma_px == s)].iloc[0]
        b = snr[(snr.pixels == "|B|>200G") & (snr.sigma_px == s)].iloc[0]
        fh.write(f"{int(s)} & {a.sigma_mm:.2f} & {a["corr"]:.2f} & {a.snr:.1f} & {b["corr"]:.2f} & {b.snr:.1f} & "
                 f"{a.signal_rms_g_per_mm:.0f} & {a.noise_rms_g_per_mm:.0f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- pilot (one month, 8 validation HARPs, seed 0)
pv1 = P.pilot_summary("eval_val")
dv = pv1[(pv1["mask"] == "all") & (pv1.metric == "div_err_g_per_mm")].set_index("model").mean_over_harps
lam = [m for m in dv.index if m.startswith("E3_l")]
mac("pilotDivErrLamLo", f(dv[lam].min(), 1))
mac("pilotDivErrLamHi", f(dv[lam].max(), 1))
mac("pilotDivErrMSE", f(dv["E1"], 1))
mac("pilotDivErrZero", f(dv["E0c"], 1))
mac("pilotDivErrBestNet", f(dv[[m for m in dv.index if not m.startswith("E0")]].min(), 1))

abl = P.pilot_summary("eval_val_v2_ablation")
pdh = P.pilot_div_per_harp("eval_val_v2_ablation")


def pilot_row(model):
    a = abl[(abl["mask"] == "all") & (abl.model == model)].set_index("metric").mean_over_harps
    x = pdh[(pdh.model == model) & (pdh["mask"] == "all") & pdh.sigma_px.isin([2.0, 4.0])]
    return a, float(x.groupby("harpnum").corr_pred_true.mean().mean())


pilot_rows = [("E1", "--", "--"), ("V_E1_geo", "\\checkmark", "--"), ("V_E1_aug", "--", "\\checkmark"),
              ("V_E1", "\\checkmark", "\\checkmark")]
with open(os.path.join(TAB, "pilot_setup.tex"), "w") as fh:
    fh.write("% GENERATED by scripts/make_tables.py from results/pilot/eval_val_v2_ablation -- do not edit\n")
    fh.write("\\begin{tabular}{cc ccc ccc c}\n\\toprule\n")
    fh.write("Geom. & Aug. & \\multicolumn{3}{c}{Pearson} & \\multicolumn{3}{c}{RMSE (G)} & Div.\\ corr. \\\\\n")
    fh.write("\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n")
    fh.write(" & & $B_p$ & $B_t$ & $B_r$ & $B_p$ & $B_t$ & $B_r$ & \\\\\n\\midrule\n")
    for m, g, au in pilot_rows:
        a, d = pilot_row(m)
        fh.write(f"{g} & {au} & {a.Bp_pearson:.2f} & {a.Bt_pearson:.2f} & {a.Br_pearson:.2f} & "
                 f"{a.Bp_rmse:.1f} & {a.Bt_rmse:.1f} & {a.Br_rmse:.1f} & {d:.3f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
a0, _ = pilot_row("E1")
a1, _ = pilot_row("V_E1")
mac("pilotBpBase", f(a0.Bp_pearson, 2)); mac("pilotBtBase", f(a0.Bt_pearson, 2))
mac("pilotBpSetup", f(a1.Bp_pearson, 2)); mac("pilotBtSetup", f(a1.Bt_pearson, 2))

# ---------------------------------------------------------------- main results table (per split)
ORDER = ["F_U_E1", "F_U_E2", "F_U_E3", "F_E1", "F_E2", "F_E3"]


def main_table(s, path):
    t, fh_ = T[s], FH[s]
    div = P.seed_stats(t, "div_all")
    divs = P.seed_stats(t, "div_strong")
    g = fh_.groupby(level="family").mean()
    best = {}
    for m, hb in (("div_all", True), ("div_strong", True), ("rmse_mean", False), ("tv", False),
                  ("Br_extreme_err_32", False), ("signed_flux_imbalance", False)):
        col = g[["Bp_hist_tv", "Bt_hist_tv"]].mean(axis=1) if m == "tv" else g[m]
        col = col[ORDER]
        best[m] = (col.idxmax() if hb else col.idxmin())
    with open(path, "w") as out:
        out.write(f"% GENERATED by scripts/make_tables.py from results/full/eval_{s}_s* -- do not edit\n")
        out.write("\\begin{tabular}{ll cc c c ccc}\n\\toprule\n")
        out.write(" & & \\multicolumn{2}{c}{Div.\\ corr.\\ $\\uparrow$} & RMSE $\\downarrow$ & Pearson $\\uparrow$ "
                  "& TV $\\downarrow$ & Ext.\\ $B_r$ $\\downarrow$ & Net flux $\\downarrow$ \\\\\n")
        out.write("\\cmidrule(lr){3-4}\n")
        out.write("Network & Loss & all & strong & (G) & $B_p$\\,/\\,$B_t$\\,/\\,$B_r$ & $B_p$,$B_t$ & (G) & (\\%) \\\\\n\\midrule\n")
        for i, fam in enumerate(ORDER):
            net, loss = P.FAMILIES[fam]
            r = g.loc[fam]
            tv = (r.Bp_hist_tv + r.Bt_hist_tv) / 2

            def bold(m, txt):
                return f"\\textbf{{{txt}}}" if best[m] == fam else txt
            cells = [net if i % 3 == 0 else "", loss,
                     bold("div_all", f"{div[fam][0]:.3f}\\,{{\\scriptsize$\\pm${div[fam][1]:.3f}}}"),
                     bold("div_strong", f"{divs[fam][0]:.3f}"),
                     bold("rmse_mean", f"{r.rmse_mean:.1f}"),
                     f"{r.Bp_pearson:.2f}\\,/\\,{r.Bt_pearson:.2f}\\,/\\,{r.Br_pearson:.2f}",
                     bold("tv", f"{tv:.3f}"),
                     bold("Br_extreme_err_32", f"{r.Br_extreme_err_32:.0f}"),
                     bold("signed_flux_imbalance", pct(r.signed_flux_imbalance))]
            out.write(" & ".join(cells) + " \\\\\n")
            if i == 2:
                out.write("\\midrule\n")
        out.write("\\bottomrule\n\\end{tabular}\n")
    return g


G = {}
for s in ("val", "test"):
    G[s] = main_table(s, os.path.join(TAB, f"main_{s}.tex"))
    for fam, key in (("F_U_E1", "UMSE"), ("F_U_E2", "UMZ"), ("F_U_E3", "UOurs"),
                     ("F_E1", "HMSE"), ("F_E2", "HMZ"), ("F_E3", "HOurs"),
                     ("F_U_E4", "UDivOnly"), ("F_U_E5", "UGS")):
        r = G[s].loc[fam]
        st = P.seed_stats(T[s], "div_all")[fam]
        mac(f"div{key}{SPL[s]}", f(st[0], 3))
        mac(f"divSd{key}{SPL[s]}", f(st[1], 4))
        mac(f"rmse{key}{SPL[s]}", f(r.rmse_mean, 1))
        mac(f"flux{key}{SPL[s]}", pct(r.signed_flux_imbalance))
        mac(f"uflux{key}{SPL[s]}", pct(-r.unsigned_flux_rel_err))
        mac(f"tvBp{key}{SPL[s]}", f(r.Bp_hist_tv, 2))
        mac(f"extBr{key}{SPL[s]}", f(r.Br_extreme_err_32, 0))
        mac(f"pearBp{key}{SPL[s]}", f(r.Bp_pearson, 2))
        mac(f"pearBt{key}{SPL[s]}", f(r.Bt_pearson, 2))
        mac(f"pearBr{key}{SPL[s]}", f(r.Br_pearson, 2))
    sd = [P.seed_stats(T[s], "div_all")[fam][1] for fam in ORDER]
    mac(f"seedSdMin{SPL[s]}", f(min(sd), 4))
    mac(f"seedSdMax{SPL[s]}", f(max(sd), 4))

# ---------------------------------------------------------------- paired tests
RES = {}
for s in ("val", "test"):
    for key, (a, b, _) in {**COMP, **ABL}.items():
        for m in ("div_all", "div_strong", "rmse_mean", "Br_rmse", "Bp_hist_tv", "Bt_hist_tv",
                  "Br_extreme_err_32", "Bp_extreme_err_32", "Br_sobel_err", "signed_flux_imbalance",
                  "unsigned_flux_rel_err", "Br_pearson", "div_s0", "div_s1", "div_s2", "div_s4", "div_s8"):
            RES[(s, key, m)] = r = P.paired(FH[s], m, a, b)
        for m, tag, nd in (("div_all", "Div", 4), ("div_strong", "DivStrong", 4), ("rmse_mean", "Rmse", 2),
                           ("Br_rmse", "BrRmse", 2), ("Br_extreme_err_32", "ExtBr", 1),
                           ("Bp_extreme_err_32", "ExtBp", 1), ("signed_flux_imbalance", "Flux", 4),
                           ("Bp_hist_tv", "TvBp", 3), ("Bt_hist_tv", "TvBt", 3), ("Br_sobel_err", "Sobel", 2),
                           ("Br_pearson", "PearBr", 4)):
            r = RES[(s, key, m)]
            mac(f"d{tag}{key}{SPL[s]}", sf(r["median_diff"], nd))
            mac(f"n{tag}{key}{SPL[s]}", frac(r))
            mac(f"p{tag}{key}{SPL[s]}", fp(r["p_wilcoxon"]))

with open(os.path.join(TAB, "paired.tex"), "w") as out:
    out.write("% GENERATED by scripts/make_tables.py -- do not edit\n")
    out.write("\\begin{tabular}{ll r r l r}\n\\toprule\n")
    out.write("Comparison & Split & $\\Delta$ div.\\ corr. & better & $p$ & $\\Delta$RMSE (G) \\\\\n\\midrule\n")
    rows = [("UOursMSE", "U-Net: Ours $-$ MSE"), ("UOursMZ", "U-Net: Ours $-$ M24"),
            ("HOursMSE", "HRN: Ours $-$ MSE"), ("HOursMZ", "HRN: Ours $-$ M24"),
            ("UMZMSE", "U-Net: M24 $-$ MSE")]
    for i, (key, label) in enumerate(rows):
        for j, s in enumerate(("val", "test")):
            d, rm = RES[(s, key, "div_all")], RES[(s, key, "rmse_mean")]
            out.write(f"{label if j == 0 else ''} & {SPL[s]} & {sf(d['median_diff'], 4)} & {frac(d)} & "
                      f"{fp(d['p_wilcoxon'])} & {sf(rm['median_diff'], 2)} \\\\\n")
        if i < len(rows) - 1:
            out.write("\\addlinespace[2pt]\n")
    out.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(TAB, "ablation.tex"), "w") as out:
    out.write("% GENERATED by scripts/make_tables.py -- do not edit\n")
    out.write("\\begin{tabular}{l l r r l r}\n\\toprule\n")
    out.write("Contrast (U-Net) & Split & $\\Delta$ div.\\ corr. & better & $p$ & $\\Delta$RMSE (G) \\\\\n\\midrule\n")
    rows = [("AblGS", "(E5) $-$ (E1): Grad+SSIM"), ("AblDivOnly", "(E4) $-$ (E1): Div alone"),
            ("AblDivOwn", "(E3) $-$ (E5): Div on top"), ("AblAddGS", "(E3) $-$ (E4): Grad+SSIM on top")]
    for i, (key, label) in enumerate(rows):
        for j, s in enumerate(("val", "test")):
            d, rm = RES[(s, key, "div_all")], RES[(s, key, "rmse_mean")]
            out.write(f"{label if j == 0 else ''} & {SPL[s]} & {sf(d['median_diff'], 4)} & {frac(d)} & "
                      f"{fp(d['p_wilcoxon'])} & {sf(rm['median_diff'], 2)} \\\\\n")
        if i < len(rows) - 1:
            out.write("\\addlinespace[2pt]\n")
    out.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- one table: loss comparisons + U-Net ablation, both splits, with bootstrap CIs
def boot_ci(fh_, metric, a, b, n_boot=10000, seed=0):
    """95% percentile CI of the median per-HARP difference, resampling HARPs."""
    d = (fh_[metric].xs(a, level="family") - fh_[metric].xs(b, level="family")).dropna().values
    rng = np.random.default_rng(seed)
    meds = np.median(d[rng.integers(0, len(d), size=(n_boot, len(d)))], axis=1)
    return np.percentile(meds, [2.5, 97.5])


ALL_ROWS = [("UOursMSE", "U-Net", "Ours vs MSE"), ("UOursMZ", "U-Net", "Ours vs M24"),
            ("HOursMSE", "HighRes-net", "Ours vs MSE"), ("HOursMZ", "HighRes-net", "Ours vs M24"),
            ("UMZMSE", "U-Net", "M24 vs MSE"),
            ("AblGS", "U-Net", "E5 vs MSE"), ("AblDivOnly", "U-Net", "E4 vs MSE"),
            ("AblDivOwn", "U-Net", "Ours vs E5"), ("AblAddGS", "U-Net", "Ours vs E4")]
with open(os.path.join(TAB, "paired_all.tex"), "w") as out:
    out.write("% GENERATED by scripts/make_tables.py -- do not edit\n")
    out.write("\\begin{tabular}{l rcrlr rcrlr}\n\\toprule\n")
    out.write(" & \\multicolumn{5}{c}{Validation (\\nHarpsVal{} HARPs)} & \\multicolumn{5}{c}{Test (\\nHarpsTest{} HARPs)} \\\\\n")
    out.write("\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n")
    out.write("Comparison & $\\Delta$ corr. & 95\\% CI & better & $p$ & $\\Delta$RMSE "
              "& $\\Delta$ corr. & 95\\% CI & better & $p$ & $\\Delta$RMSE \\\\\n\\midrule\n")
    for i, (key, net, lab) in enumerate(ALL_ROWS):
        a, b, _ = {**COMP, **ABL}[key]
        cells = [f"{'HRN' if net == 'HighRes-net' else net}: {lab}"]
        for s in ("val", "test"):
            d, rm = RES[(s, key, "div_all")], RES[(s, key, "rmse_mean")]
            lo, hi = boot_ci(FH[s], "div_all", a, b)
            mac(f"ci{key}{SPL[s]}", f"[{f(lo, 4)}, {f(hi, 4)}]")
            cells += [sf(d["median_diff"], 4), f"{f(lo, 4)}--{f(hi, 4)}", frac(d), fp(d["p_wilcoxon"]),
                      sf(rm["median_diff"], 2)]
        out.write(" & ".join(cells) + " \\\\\n")
        if key == "UMZMSE":
            out.write("\\midrule\n")
    out.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- divergence gain by smoothing scale, per seed
rows = []
for s in ("val", "test"):
    for sig in (0, 1, 2, 4, 8):
        m = f"div_s{sig}"
        r = RES[(s, "UOursMSE", m)]
        rows.append({"split": s, "sigma_px": sig, "seed": "avg", "median_diff": r["median_diff"],
                     "better": r["a_better_in"], "n": r["n_harps"], "p": r["p_wilcoxon"]})
        for seed in P.SEEDS:
            x = T[s][m].xs(("F_U_E3", seed), level=("family", "seed"))
            y = T[s][m].xs(("F_U_E1", seed), level=("family", "seed"))
            d = (x - y).dropna()
            rows.append({"split": s, "sigma_px": sig, "seed": str(seed), "median_diff": float(d.median()),
                         "better": int((d > 0).sum()), "n": int(len(d)), "p": np.nan})
scale = pd.DataFrame(rows)
scale.to_csv(os.path.join(FIGDATA, "scale_gain.csv"), index=False)
for s in ("val", "test"):
    for sig in (0, 8):
        r = scale[(scale.split == s) & (scale.sigma_px == sig) & (scale.seed == "avg")].iloc[0]
        per = scale[(scale.split == s) & (scale.sigma_px == sig) & (scale.seed != "avg")]
        w = WORD[sig]
        mac(f"gainSig{w}{SPL[s]}", sf(r.median_diff, 4))
        mac(f"gainSig{w}N{SPL[s]}", f"{int(r.better)}/{int(r.n)}")
        mac(f"gainSig{w}P{SPL[s]}", fp(r.p))
        mac(f"gainSig{w}SeedMin{SPL[s]}", sf(per.median_diff.min(), 4))
        mac(f"gainSig{w}SeedMax{SPL[s]}", sf(per.median_diff.max(), 4))
    per = scale[(scale.split == s) & scale.sigma_px.isin([0, 1, 2, 4]) & (scale.seed != "avg")]
    mac(f"gainSeedMinFine{SPL[s]}", sf(per.median_diff.min(), 4))

with open(os.path.join(TAB, "scale.tex"), "w") as out:
    out.write("% GENERATED by scripts/make_tables.py -- do not edit\n")
    out.write("\\begin{tabular}{l rr " + "r" * 3 + " rr " + "r" * 3 + "}\n\\toprule\n")
    out.write(" & \\multicolumn{5}{c}{Validation (40 HARPs)} & \\multicolumn{5}{c}{Test (52 HARPs)} \\\\\n")
    out.write("\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n")
    out.write("$\\sigma$ (px) & avg & better & s0 & s1 & s2 & avg & better & s0 & s1 & s2 \\\\\n\\midrule\n")
    for sig in (0, 1, 2, 4, 8):
        cells = [f"{sig}" + (" (trained)" if sig in (1, 2, 4) else "")]
        for s in ("val", "test"):
            x = scale[(scale.split == s) & (scale.sigma_px == sig)].set_index("seed")
            cells += [sf(x.loc["avg"].median_diff, 4), f"{int(x.loc['avg'].better)}/{int(x.loc['avg'].n)}"]
            cells += [sf(x.loc[str(k)].median_diff, 4) for k in P.SEEDS]
        out.write(" & ".join(cells) + " \\\\\n")
    out.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- per-HARP paired differences (figure data)
rows = []
for s in ("val", "test"):
    for key in ("UOursMSE", "UOursMZ", "HOursMSE", "HOursMZ", "UMZMSE"):
        a, b, _ = COMP[key]
        x = FH[s]["div_all"].xs(a, level="family")
        y = FH[s]["div_all"].xs(b, level="family")
        for h, d in (x - y).dropna().items():
            rows.append({"split": s, "comparison": key, "harpnum": int(h), "diff": float(d)})
pd.DataFrame(rows).to_csv(os.path.join(FIGDATA, "paired_div.csv"), index=False)

# ---------------------------------------------------------------- net flux: per-seed spread for the U-Net
for s in ("val", "test"):
    per_seed = T[s]["signed_flux_imbalance"].groupby(level=["family", "seed"]).mean()
    u3 = per_seed.xs("F_U_E3", level="family")
    mac(f"fluxUOursSeedMin{SPL[s]}", pct(u3.min()))
    mac(f"fluxUOursSeedMax{SPL[s]}", pct(u3.max()))

# ---------------------------------------------------------------- robustness: test split without the pilot HARPs
rows = []
for key in ("UOursMSE", "UOursMZ", "HOursMSE", "HOursMZ"):
    a, b, _ = COMP[key]
    r = P.paired(FH["test"], "div_all", a, b, exclude=P.PILOT_OVERLAP_TEST)
    rows.append((key, r))
    mac(f"robDiv{key}", sf(r["median_diff"], 4))
    mac(f"robN{key}", frac(r))
    mac(f"robP{key}", fp(r["p_wilcoxon"]))
mac("nPilotOverlap", str(len(P.PILOT_OVERLAP_TEST)))
mac("nTestNoOverlap", str(rows[0][1]["n_harps"]))
mac("pilotOverlapList", ", ".join(str(h) for h in P.PILOT_OVERLAP_TEST))

# ---------------------------------------------------------------- small confined-flux regions (validation only; secondary)
sfx = json.load(open(os.path.join(P.RES, "full", "eval_small_flux", "tests.json")))
SF_METRICS = (("Br_rmse", "$B_r$ RMSE", 2), ("flux_err", "flux err.", 4), ("peak_err", "peak err.", 4),
              ("div_corr_s2", "div.\\ corr.", 4))
with open(os.path.join(TAB, "smallflux.tex"), "w") as out:
    out.write("% GENERATED by scripts/make_tables.py from results/full/eval_small_flux/tests.json -- do not edit\n")
    out.write("\\begin{tabular}{ll " + "rr" * len(SF_METRICS) + "}\n\\toprule\n")
    out.write(" & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab, _ in SF_METRICS) + " \\\\\n")
    out.write("".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(SF_METRICS))) + "\n")
    out.write("U-Net & mask & " + " & ".join(["$\\Delta$ & better"] * len(SF_METRICS)) + " \\\\\n\\midrule\n")
    for comp, lab, key in (("F_U_E3 vs F_U_E1", "Ours $-$ MSE", "SfMSE"), ("F_U_E3 vs F_U_E2", "Ours $-$ M24", "SfMZ")):
        for mask in ("compact", "mixed", "pil"):
            r = sfx["tests"][f"{comp} | {mask}"]
            cells = [lab if mask == "compact" else "", mask]
            for m, _, nd in SF_METRICS:
                x = r.get(m, {})
                if "median_diff" not in x:
                    cells += ["--", "--"]
                    continue
                cells += [sf(x["median_diff"], nd), f"{x['a_better_in']}/{x['n']}"]
                short = {"Br_rmse": "Rmse", "flux_err": "Flux", "peak_err": "Peak", "div_corr_s2": "Div"}[m]
                mac(f"{key}{mask.capitalize()}{short}", f"{x['a_better_in']}/{x['n']}")
                mac(f"{key}{mask.capitalize()}{short}P", fp(x["p_wilcoxon"]))
            out.write(" & ".join(cells) + " \\\\\n")
        out.write("\\addlinespace[2pt]\n" if key == "SfMSE" else "")
    out.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- training runs (results/full/*/meta.json)
metas = []
for fam in P.FAMILIES:
    for seed in P.SEEDS:
        m = json.load(open(os.path.join(P.RES, "full", f"{fam}_s{seed}", "meta.json")))
        metas.append({"family": fam, "seed": seed, "best": m["best_epoch"], "stop": m["stopping_epoch"],
                      "cap": bool(m.get("hit_epoch_cap")), "hours": m["total_seconds"] / 3600,
                      "gpu": m["gpu_name"], "params": m["params_active"], "commit": m["git_commit"][:7],
                      "dirty": m["git_dirty"]})
metas = pd.DataFrame(metas)
metas.to_csv(os.path.join(FIGDATA, "training_runs.csv"), index=False)
mac("nRuns", str(len(metas)))
mac("nRunsCap", str(int(metas.cap.sum())))
mac("nRunsLateBest", str(int((metas.best >= 91).sum())))
mac("gpuHours", f"{metas.hours.sum():.0f}")
mac("gpuName", metas.gpu.iloc[0].replace("NVIDIA ", ""))
assert metas.gpu.nunique() == 1
mac("paramsUNet", ints(int(metas[metas.family.str.startswith("F_U")].params.iloc[0])))
mac("paramsHRN", ints(int(metas[~metas.family.str.startswith("F_U")].params.iloc[0])))
mac("codeCommit", metas.commit.iloc[0])
with open(os.path.join(TAB, "training.tex"), "w") as out:
    out.write("% GENERATED by scripts/make_tables.py from results/full/*/meta.json -- do not edit\n")
    out.write("\\begin{tabular}{ll ccc ccc}\n\\toprule\n")
    out.write(" & & \\multicolumn{3}{c}{best epoch (seed 0/1/2)} & \\multicolumn{3}{c}{hours} \\\\\n")
    out.write("\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nNetwork & Loss & s0 & s1 & s2 & s0 & s1 & s2 \\\\\n\\midrule\n")
    for fam in P.FAMILIES:
        x = metas[metas.family == fam].set_index("seed")
        net, loss = P.FAMILIES[fam]
        cells = [net, loss] + [f"{int(x.loc[s].best)}" + ("$^\\ast$" if x.loc[s].cap else "") for s in P.SEEDS]
        cells += [f"{x.loc[s].hours:.1f}" for s in P.SEEDS]
        out.write(" & ".join(cells) + " \\\\\n")
    out.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- divergence term in the training loss (metrics.csv, last epoch)
shares, finals = [], []
for fam in ("F_U_E3", "F_E3"):
    for seed in P.SEEDS:
        m = pd.read_csv(os.path.join(P.RES, "full", f"{fam}_s{seed}", "metrics.csv")).iloc[-1]
        shares.append(0.01 * m.train_div / m.train_total)
        if fam == "F_U_E3":
            finals.append(m.train_div)
mac("divShareLo", pct(min(shares), 1))
mac("divShareHi", pct(max(shares), 1))
mac("divLossEndUNet", f(float(np.mean(finals)), 2))
rho_sel = snr[(snr.pixels == "all") & snr.sigma_px.isin([1.0, 2.0, 4.0])]["corr"]
mac("divNoiseFloor", f(float(1 - rho_sel.mean()), 2))

# ---------------------------------------------------------------- loss weights and training settings, read from the run configs
import yaml  # noqa: E402

cfgs = {fam: yaml.safe_load(open(os.path.join(P.ROOT, "configs", f"full_{fam}_s0.yaml"))) for fam in P.FAMILIES}
for fam, c in cfgs.items():  # the three seeds differ only in the seed
    for seed in P.SEEDS[1:]:
        c2 = yaml.safe_load(open(os.path.join(P.ROOT, "configs", f"full_{fam}_s{seed}.yaml")))
        assert c2["loss"] == c["loss"] and {k: v for k, v in c2["train"].items() if k != "seed"} == \
               {k: v for k, v in c["train"].items() if k != "seed"}, fam


def wfmt(w):
    if w == 0:
        return "--"
    if w >= 0.1:
        return f"{w:g}"
    e = int(math.floor(math.log10(w)))
    m = w / 10 ** e
    return f"$10^{{{e}}}$" if abs(m - 1) < 1e-9 else f"${m:g}{{\\times}}10^{{{e}}}$"


with open(os.path.join(TAB, "weights.tex"), "w") as out:
    out.write("% GENERATED by scripts/make_tables.py from configs/full_*_s0.yaml -- do not edit\n")
    out.write("\\begin{tabular}{ll ccccc}\n\\toprule\n")
    out.write("Network & Loss & MSE & grad & hist & SSIM & $\\Ldiv$ \\\\\n\\midrule\n")
    for fam in P.FAMILIES:
        w = cfgs[fam]["loss"]["weights"]
        net, loss = P.FAMILIES[fam]
        out.write(" & ".join([net, loss] + [wfmt(float(w.get(k, 0))) for k in ("mse", "grad", "hist", "ssim", "div")])
                  + " \\\\\n")
    out.write("\\bottomrule\n\\end{tabular}\n")
c = cfgs["F_U_E3"]
assert c["loss"]["div_mode"] == "match_ms" and c["loss"]["div_scales"] == [1.0, 2.0, 4.0]
tr = c["train"]
mac("cfgLr", wfmt(float(tr["lr"])))
mac("cfgBatch", str(tr["batch_size"]))
mac("cfgEpochs", str(tr["max_epochs"]))
mac("cfgPatience", str(tr["patience"]))
mac("cfgSamples", ints(tr["samples_per_epoch"]))
mac("cfgMaxShare", f"{100 * tr['max_share']:g}")
mac("cfgValEvery", str(c["data"]["val_every"]))
for s_, v in zip(("One", "Two", "Four"), c["loss"]["div_scale_norm"]):
    mac(f"mNorm{s_}", fp(v))

# ---------------------------------------------------------------- shared-noise correction of the reliability (appendix)
C_SHARED = 0.1
mac("sharedNoiseC", f"{C_SHARED:.1f}")
for _, r in snr[snr.pixels == "all"].iterrows():
    rho = (r["corr"] - C_SHARED) / (1 - C_SHARED)
    mac(f"snrCorr{WORD[int(r.sigma_px)]}", f(math.sqrt(rho / (1 - rho)), 2 if rho < 0.5 else 1))

# ---------------------------------------------------------------- p-value summary used in the abstract
worst = max(RES[(s, k, "div_all")]["p_wilcoxon"] for s in ("val", "test")
            for k in ("UOursMSE", "HOursMSE"))
mac("pWorstOursMSE", fp(worst))
mac("pWorstPrereg", fp(max(RES[(s, k, "div_all")]["p_wilcoxon"] for s in ("val", "test")
                           for k in ("UOursMSE", "UOursMZ", "HOursMSE", "HOursMZ"))))
worst_abl = max(RES[(s, "AblDivOwn", "div_all")]["p_wilcoxon"] for s in ("val", "test"))
mac("pWorstAblDivOwn", fp(worst_abl))

# ---------------------------------------------------------------- write macros
with open(os.path.join(TAB, "numbers.tex"), "w") as out:
    out.write("% GENERATED by scripts/make_tables.py -- do not edit. Every number quoted in the text.\n")
    for k in sorted(MACROS):
        out.write(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}\n")
print(f"wrote {len(MACROS)} macros and tables to {TAB}")
