"""Phase 14 / figure: violin plot of the bootstrap TSS distributions for the mixing arms
(professor request: violins in place of the bar chart). Recomputes the H1 arms with the frozen
machinery (c23_hybrid_benchmark), shared 2,000 region resamples, seed 0, and renders:

    figures/mixing_violin.pdf   (Okabe-Ito, grayscale-safe)
    results/full/hybrid/mixing_violin_deck.png  (deck palette)
    results/full/hybrid/mixing_boot_tss.csv     (provenance: every draw)

    python scripts/c23_mixing_figure.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from c23_hybrid_benchmark import (                               # noqa: E402
    C23_FEATURES, PRIMARY_BOUNDARY, SHARP_ERA, VEC_B, boot_all, cols, fit_predict, prep,
    split_real, windows)
from c23_benchmark import load_flares, sharp_era_frames, vec_frames_c23  # noqa: E402

OUT_HY = os.path.join(ROOT, "results", "full", "hybrid")
OUT_PDF = os.path.join(ROOT, "figures", "mixing_violin.pdf")
OUT_PNG = os.path.join(OUT_HY, "mixing_violin_deck.png")

ARMS = [("real_only", "real only", 0.0),
        ("w0.1", "+ record\nw = 0.1", 0.1),
        ("w0.25", "+ record\nw = 0.25", 0.25),
        ("w0.5", "+ record\nw = 0.5", 0.5),
        ("w1", "+ record\nequal weight", 1.0),
        ("record_only", "record only", None)]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def compute():
    flares = load_flares(years=range(1996, 2017))
    era = sharp_era_frames(SHARP_ERA)
    wins = windows(era, VEC_B, flares)
    train_r, test, _ = split_real(wins, PRIMARY_BOUNDARY)
    syn = windows(vec_frames_c23(C23_FEATURES), VEC_B, flares)
    syn = syn[syn.t_end < pd.Timestamp("2010-05-01")]
    TR, SY, TE = prep(train_r, "real", "H"), prep(syn, "synC23", "T"), prep(test, "real", "H")
    fc = cols(VEC_B)
    preds, points = {}, {}
    for key, _, w in ARMS:
        if key == "real_only":
            tr = TR
        elif key == "record_only":
            tr = SY
        else:
            tr = pd.concat([TR, SY], ignore_index=True)
        point, pte, thr = fit_predict(tr, TE, fc, w_syn=(w if w else 1.0))
        preds[key], points[key] = (pte, thr), point
    boot, skipped = boot_all(TE, preds, n_boot=2000)
    rows = []
    for key, _, _ in ARMS:
        for i, v in enumerate(boot[key]["tss"]):
            rows.append({"arm": key, "draw": i, "tss": v})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_HY, "mixing_boot_tss.csv"), index=False)
    print(f"bootstrap done (skipped {skipped}); points:",
          {k: round(points[k]["tss"], 4) for k, _, _ in ARMS for k in [k]})
    return boot, points


def draw(boot, points, palette, bg, fg, muted, path, dpi=None, figsize=(6.6, 3.3)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)
    data = [boot[k]["tss"] for k, _, _ in ARMS]
    parts = ax.violinplot(data, positions=range(len(ARMS)), widths=0.72,
                          showextrema=False, showmedians=False)
    for body, (key, _, _) in zip(parts["bodies"], ARMS):
        body.set_facecolor(palette[key])
        body.set_edgecolor(fg)
        body.set_linewidth(0.6)
        body.set_alpha(0.85)
    for i, (key, _, _) in enumerate(ARMS):
        v = np.asarray(boot[key]["tss"])
        lo, hi = np.percentile(v, [2.5, 97.5])
        ax.plot([i, i], [lo, hi], color=fg, lw=1.4, solid_capstyle="round", zorder=3)
        ax.plot(i, points[key]["tss"], marker="o", ms=4.5, mfc="white", mec=fg, mew=1.1, zorder=4)
    base = points["real_only"]["tss"]
    ax.axhline(base, color=muted, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(len(ARMS) - 0.45, base + 0.003, "real-only point estimate", color=muted,
            fontsize=7.5, ha="right", va="bottom")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([lab for _, lab, _ in ARMS], color=fg)
    ax.set_ylabel("TSS (bootstrap over test regions)", color=fg)
    ax.tick_params(colors=fg)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(fg)
    ax.set_ylim(0.55, 0.88)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    boot, points = compute()
    # paper: Okabe-Ito, white background, grayscale-safe (fills differ in lightness; the
    # highlighted arms also differ in hue for color readers)
    oi = {"real_only": "#0072B2", "w0.1": "#88BDE0", "w0.25": "#88BDE0", "w0.5": "#88BDE0",
          "w1": "#D55E00", "record_only": "#E69F00"}
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    draw(boot, points, oi, "white", "#222222", "#666666", OUT_PDF)
    # deck: house palette on off-white
    deck = {"real_only": "#14213D", "w0.1": "#5A6B8C", "w0.25": "#5A6B8C", "w0.5": "#5A6B8C",
            "w1": "#B5541A", "record_only": "#F2A65A"}
    draw(boot, points, deck, "#F7F6F2", "#14213D", "#6B7280", OUT_PNG, dpi=220,
         figsize=(7.4, 4.4))


if __name__ == "__main__":
    main()
