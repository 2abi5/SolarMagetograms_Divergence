#!/usr/bin/env python3
"""figures/data/*.csv and data/pilot/div_snr_by_scale.json -> figures/out/*.pdf.

    cd paper && ../.venv-mag/bin/python scripts/make_figures.py      (or: make figures)

Run scripts/make_tables.py first: it writes the csv files read here.
Single-column figures are 3.25 in wide (AISTATS column width). Colours are the
first two slots of a CVD-validated categorical palette; every series also has
its own marker, so the figures stay readable in greyscale.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperdata as P  # noqa: E402

OUT = os.path.join(P.PAPER, "figures", "out")
DATA = os.path.join(P.PAPER, "figures", "data")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL_W = 3.25

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": True, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.labelcolor": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


PREVIEW = os.environ.get("PREVIEW_DIR")  # optional PNG copies for a quick look


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=220)
    plt.close(fig)


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def trained_band(ax, lo, hi, y_text, label="scales used by $\\mathcal{L}_{\\mathrm{div}}$"):
    ax.axvspan(lo, hi, color="#efeeea", zorder=0, linewidth=0)
    ax.text((lo + hi) / 2, y_text, label, ha="center", va="bottom", fontsize=6.5, color=MUTED)


# ------------------------------------------------------------------ Figure: reliability of the target's divergence
def fig_reliability():
    _, snr = P.pilot_snr()
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    x = np.arange(5)
    for mask, lab, c, mk in (("all", "all pixels", BLUE, "o"), ("|B|>200G", "$|\\mathbf{B}|>200$ G", ORANGE, "s")):
        r = snr[snr.pixels == mask].sort_values("sigma_px")
        ax.plot(x, r.snr.values, color=c, marker=mk, markersize=4.5, linewidth=1.4, label=lab, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 40)
    ax.axhline(1.0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8, zorder=2)
    ax.text(4.35, 1.08, "SNR = 1", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    trained_band(ax, 0.6, 3.4, 22)
    r = snr[snr.pixels == "all"].sort_values("sigma_px")
    ax.set_xticks(x)
    ax.set_xticklabels(["pixel"] + [f"{v:.2f}" for v in r.sigma_mm.values[1:]])
    ax.set_xlim(-0.35, 4.35)
    ax.set_xlabel("Gaussian smoothing scale $\\sigma$ (Mm)")
    ax.set_ylabel("SNR of $\\mathrm{div}_h$ (test-retest)")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    style(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    save(fig, "fig_reliability")


# ------------------------------------------------------------------ Figure: per-HARP paired differences
def fig_paired():
    d = pd.read_csv(os.path.join(DATA, "paired_div.csv"))
    rows = [("UOursMSE", "U-Net: Ours $-$ MSE"), ("UOursMZ", "U-Net: Ours $-$ M24"),
            ("HOursMSE", "HRN: Ours $-$ MSE"), ("HOursMZ", "HRN: Ours $-$ M24"),
            ("UMZMSE", "U-Net: M24 $-$ MSE")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    rng = np.random.default_rng(0)
    for i, (key, _) in enumerate(rows):
        y0 = len(rows) - 1 - i
        for split, off, c, mk, lab in (("val", 0.17, BLUE, "o", "validation (40 HARPs)"),
                                       ("test", -0.17, ORANGE, "^", "test (52 HARPs)")):
            v = d[(d.comparison == key) & (d.split == split)]["diff"].values
            yy = y0 + off + rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(v, yy, s=7, color=c, marker=mk, linewidths=0.3, edgecolors="white", zorder=3,
                       label=lab if i == 0 else None, alpha=0.9)
            ax.plot([np.median(v)] * 2, [y0 + off - 0.13, y0 + off + 0.13], color=INK, linewidth=1.1, zorder=4)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows][::-1])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("$\\Delta$ divergence correlation per HARP")
    style(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.2,
              borderaxespad=0.1, markerscale=1.4, columnspacing=1.2)
    save(fig, "fig_paired")


# ------------------------------------------------------------------ Figure: gain by smoothing scale, per seed
def fig_scale():
    d = pd.read_csv(os.path.join(DATA, "scale_gain.csv"), dtype={"seed": str})
    sig = [0, 1, 2, 4, 8]
    x = np.arange(len(sig))
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    for split, c, mk, lab, off in (("val", BLUE, "o", "validation", -0.08), ("test", ORANGE, "^", "test", 0.08)):
        a = d[(d.split == split) & (d.seed == "avg")].set_index("sigma_px").loc[sig]
        ax.plot(x + off, a.median_diff.values, color=c, marker=mk, markersize=4.5, linewidth=1.4, label=lab,
                zorder=3, markeredgecolor="white", markeredgewidth=0.6)
        for s in ("0", "1", "2"):
            b = d[(d.split == split) & (d.seed == s)].set_index("sigma_px").loc[sig]
            ax.scatter(x + off, b.median_diff.values, s=6, color=c, marker=mk, alpha=0.45, linewidths=0, zorder=2)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=2)
    trained_band(ax, 0.6, 3.4, 0.0185)
    ax.set_ylim(-0.004, 0.022)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in sig])
    ax.set_xlim(-0.4, 4.4)
    ax.set_xlabel("smoothing scale $\\sigma$ of the metric (px)")
    ax.set_ylabel("median $\\Delta$ div. corr.")
    style(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.2)
    save(fig, "fig_scale")


if __name__ == "__main__":
    fig_reliability()
    fig_paired()
    fig_scale()
    print("wrote", sorted(os.listdir(OUT)))
