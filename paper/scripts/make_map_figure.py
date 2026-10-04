#!/usr/bin/env python3
"""Qualitative maps on validation regions (never test) -> figures/out/fig_maps.pdf.

    cd paper && ../.venv-mag/bin/python scripts/make_map_figure.py      (CPU is enough)

Reuses the region choice, cropping and inference of scripts/make_figures.py
(the project's own figure script): the largest validation regions with a centre
angle below 45 deg, cropped around the strongest field, seed-0 U-Nets.
Top row of each region: B_r. Bottom row: div_h after Gaussian smoothing at
sigma = 2 HR px, with the correlation to SHARP inside the crop.
"""
import importlib.util
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperdata as P  # noqa: E402

sys.path.insert(0, P.ROOT)
sys.path.insert(0, os.path.join(P.ROOT, "scripts"))
_spec = importlib.util.spec_from_file_location("project_figs", os.path.join(P.ROOT, "scripts", "make_figures.py"))
PF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PF)
from div_snr_by_scale import smoothed_div  # noqa: E402
from sr.datasets import _load_pair, split_harps  # noqa: E402
from sr.evaluate import infer_region, load_model  # noqa: E402
from sr.geometry import geometry_path  # noqa: E402

MODELS = (("F_U_E1", "U-Net, MSE"), ("F_U_E2", "U-Net, M24 loss"), ("F_U_E3", "U-Net, ours"))
N_REGIONS, SEED, SIGMA, CROP = 2, 0, 2.0, 160

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 7, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def main():
    """Writes fig_maps_a.pdf (first region, main text) and fig_maps_b.pdf (second region, appendix)."""
    run_dir, res_dir = os.path.join(P.ROOT, "data", "full"), os.path.join(P.RES, "full")
    norm = float(json.load(open(os.path.join(run_dir, "stats.json"))).get("norm", 3500.0))
    regs = pd.read_csv(os.path.join(run_dir, "regions.csv"))
    val = regs[regs.harpnum.isin(split_harps(os.path.join(P.ROOT, "splits", "full.json"), "val"))]
    chosen = PF.pick_regions(val, N_REGIONS)
    models = {}
    for e, _ in MODELS:
        name = f"{e}_s{SEED}"
        cfg = yaml.safe_load(open(os.path.join(res_dir, name, "config.yaml")))
        assert int(cfg["model"]["kwargs"]["in_channels"]) == 4
        models[e] = load_model(res_dir, name, "cpu")[0]
    geo_dir = os.path.join(run_dir, "geometry_lr")

    ncol = 2 + len(MODELS)
    for k, r in enumerate(chosen):
        fig, axes = plt.subplots(2, ncol, figsize=(6.75, 2.75), squeeze=False)
        lr, hr = _load_pair(os.path.join(run_dir, r.region_file))
        true = hr.astype(np.float64) * norm
        geom = np.load(geometry_path(geo_dir, r.region_file))
        x4 = torch.as_tensor(np.nan_to_num(np.concatenate([lr[None], geom])))[None]
        preds = {}
        with torch.no_grad():
            for e, m in models.items():
                preds[e] = infer_region(m, x4)[0].cpu().numpy().astype(np.float64) * norm
        ys, xs = PF.crop_window(true[2], CROP)
        mdi = np.kron(lr * norm, np.ones((4, 4)))[:true.shape[1], :true.shape[2]]

        # B_r row
        ax_row = axes[0]
        vmax = float(np.nanpercentile(np.abs(true[2][ys, xs]), 99.5))
        imgs = [(mdi[ys, xs], "MDI input (upsampled)"), (true[2][ys, xs], "SHARP (target)")]
        imgs += [(preds[e][2][ys, xs], t) for e, t in MODELS]
        for j, (img, t) in enumerate(imgs):
            im = ax_row[j].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower",
                                  interpolation="nearest", rasterized=True)
            ax_row[j].set_title(t, fontsize=7, pad=2)
        cb = fig.colorbar(im, ax=ax_row.tolist(), fraction=0.012, pad=0.01, shrink=0.85)
        cb.set_label("$B_r$ (G)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax_row[0].set_ylabel(f"HARP {int(r.harpnum)}, {r.mdi_t_rec[:10].replace('.', '-')}\n$B_r$", fontsize=7)

        # div_h row (sigma = 2 px)
        ax_row = axes[1]
        dt, vt = smoothed_div(true / norm, SIGMA)
        dt = np.where(vt, dt, np.nan)
        cy, cx = slice(ys.start, ys.stop - 2), slice(xs.start, xs.stop - 2)
        vmax = float(np.nanpercentile(np.abs(dt[cy, cx]), 99))
        ax_row[0].axis("off")
        panels = [(dt[cy, cx], "SHARP (target)")]
        for e, t in MODELS:
            dp, vp = smoothed_div(np.where(np.isfinite(true), preds[e], np.nan) / norm, SIGMA)
            dp = np.where(vp, dp, np.nan)
            ok = np.isfinite(dp[cy, cx]) & np.isfinite(dt[cy, cx])
            c = np.corrcoef(dp[cy, cx][ok], dt[cy, cx][ok])[0, 1]
            panels.append((dp[cy, cx], f"$r = {c:.2f}$"))
        for j, (img, t) in enumerate(panels, start=1):
            im = ax_row[j].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower",
                                  interpolation="nearest", rasterized=True)
            ax_row[j].set_title(t, fontsize=7, pad=2)
        cb = fig.colorbar(im, ax=ax_row.tolist(), fraction=0.012, pad=0.01, shrink=0.85)
        cb.set_label("$\\mathrm{div}_h$ (G Mm$^{-1}$)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        ax_row[1].set_ylabel("$\\mathrm{div}_h$, $\\sigma = 2$ px", fontsize=7)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        name = f"fig_maps_{'ab'[k]}"
        out = os.path.join(P.PAPER, "figures", "out", name + ".pdf")
        fig.savefig(out, dpi=300)
        prev = os.environ.get("PREVIEW_DIR")
        if prev:
            fig.savefig(os.path.join(prev, name + ".png"), dpi=150)
        plt.close(fig)
        print("wrote", out, int(r.harpnum))


if __name__ == "__main__":
    main()
