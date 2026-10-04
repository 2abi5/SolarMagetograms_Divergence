"""
Evaluation figures (training_plan.md Phase 7 and 2c), written to <out_dir>/figures/:

  panels_<region>.png        MDI input | truth and E1 / E2 / E3 for Bp, Bt, Br (3 regions)
  div_error_<region>.png     |div_h(pred) - div_h(true)| maps for E1, E2, E3 (G/Mm)
  histograms.png             per-channel histograms, truth vs every model (pooled)
  scatter.png                prediction vs truth per channel for E1, E2, E3 (pooled)
  metrics_bar.png            main metrics with 95% bootstrap CIs over HARPs
  power_spectra_<region>.png radially averaged power, prediction vs truth (smoothing check)

The 3 regions are the largest regions of 3 different HARPs (deterministic).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sr import metrics as M  # noqa: E402
from sr.datasets import _load_pair  # noqa: E402

CH = M.CHANNELS


def _key_models(models):
    names = list(models)
    e3 = [n for n in names if n.startswith("E3")]
    keep = [n for n in ("E1", "E2") if n in names] + e3[:1]
    return keep or names[:3]


def _pick_regions(ev_regs, run_dir, k=3):
    sizes = []
    for r in ev_regs.itertuples():
        lr, _ = _load_pair(os.path.join(run_dir, r.region_file))
        sizes.append((lr.size, r.harpnum, r.Index))
    picked, seen = [], set()
    for _, h, i in sorted(sizes, reverse=True):
        if h not in seen:
            picked.append(i)
            seen.add(h)
        if len(picked) == k:
            break
    return picked


def _predict(models, lr, norm, device):
    from sr.evaluate import infer_region
    t = torch.as_tensor(np.nan_to_num(lr))[None, None].to(device)
    return {e: infer_region(m, t)[0].cpu().numpy() * norm for e, (m, _) in models.items()}


def _vlim(a):
    v = np.nanpercentile(np.abs(a), 99)
    return v if v > 0 else 1.0


def make_all(run_dir, ev_regs, models, out_dir, norm, means, calib, per, summary, device):
    from sr.evaluate import baseline_prediction, div_error_map
    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    key = _key_models(models)

    # ---- per-region figures for 3 regions
    for i in _pick_regions(ev_regs, run_dir):
        r = ev_regs.loc[i]
        name = os.path.splitext(os.path.basename(r.region_file))[0]
        lr, hr = _load_pair(os.path.join(run_dir, r.region_file))
        true = hr * norm
        mask = np.isfinite(true).all(axis=0)
        preds = _predict({e: models[e] for e in key}, lr, norm, device)

        rows = [("truth", true)] + [(e, preds[e]) for e in key]
        fig, ax = plt.subplots(len(rows), 4, figsize=(16, 3.2 * len(rows)), squeeze=False)
        vl = [_vlim(true[c]) for c in range(3)]
        for j, (label, img) in enumerate(rows):
            if j == 0:
                v = _vlim(lr * norm)
                im = ax[0, 0].imshow(lr * norm, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
                ax[0, 0].set_title("MDI LOS input")
                plt.colorbar(im, ax=ax[0, 0], fraction=0.046)
            else:
                ax[j, 0].axis("off")
            for c in range(3):
                im = ax[j, c + 1].imshow(np.where(mask, img[c], np.nan), origin="lower", cmap="RdBu_r",
                                         vmin=-vl[c], vmax=vl[c])
                ax[j, c + 1].set_title(f"{label} {CH[c]}")
                plt.colorbar(im, ax=ax[j, c + 1], fraction=0.046, label="G")
        fig.suptitle(f"HARP {r.harpnum}  {r.mdi_t_rec}")
        plt.tight_layout()
        fig.savefig(os.path.join(figs, f"panels_{name}.png"), dpi=90)
        plt.close(fig)

        fig, ax = plt.subplots(1, len(key), figsize=(5 * len(key), 4), squeeze=False)
        emaps = {e: div_error_map(preds[e], true, mask) for e in key}
        vmax = max(np.nanpercentile(m, 99) for m in emaps.values())
        for j, e in enumerate(key):
            im = ax[0, j].imshow(emaps[e], origin="lower", cmap="magma", vmin=0, vmax=vmax)
            ax[0, j].set_title(f"{e}: mean {np.nanmean(emaps[e]):.0f} G/Mm")
            plt.colorbar(im, ax=ax[0, j], fraction=0.046, label="|div error| (G/Mm)")
        plt.tight_layout()
        fig.savefig(os.path.join(figs, f"div_error_{name}.png"), dpi=90)
        plt.close(fig)

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for c in range(3):
            k, pt = M.radial_power_spectrum(true[c], mask)
            ax[c].loglog(k, pt, "k", lw=2, label="truth")
            for e in key:
                k, pp = M.radial_power_spectrum(preds[e][c], mask)
                ax[c].loglog(k, pp, label=e)
            ax[c].set_title(f"{CH[c]} power spectrum")
            ax[c].set_xlabel("wavenumber (cycles per image)")
        ax[0].legend()
        plt.tight_layout()
        fig.savefig(os.path.join(figs, f"power_spectra_{name}.png"), dpi=90)
        plt.close(fig)

    # ---- pooled histograms and scatter (all evaluated regions; scatter subsampled)
    rng = np.random.default_rng(0)
    pooled = {"truth": [[], [], []]}
    for e in list(models) + ["E0c"]:
        pooled[e] = [[], [], []]
    for r in ev_regs.itertuples():
        lr, hr = _load_pair(os.path.join(run_dir, r.region_file))
        true = hr * norm
        mask = np.isfinite(true).all(axis=0)
        preds = _predict(models, lr, norm, device)
        preds["E0c"] = baseline_prediction("E0c", lr * norm, means, calib)
        idx = np.flatnonzero(mask)
        idx = rng.choice(idx, size=min(idx.size, 3000), replace=False) if idx.size else idx
        for c in range(3):
            pooled["truth"][c].append(true[c].ravel()[idx])
            for e, p in preds.items():
                pooled[e][c].append(p[c].ravel()[idx])
    pooled = {e: [np.concatenate(v) if v else np.array([]) for v in chans] for e, chans in pooled.items()}

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for c in range(3):
        lim = np.percentile(np.abs(pooled["truth"][c]), 99.9) if pooled["truth"][c].size else 1
        bins = np.linspace(-lim, lim, 81)
        ax[c].hist(pooled["truth"][c], bins, histtype="stepfilled", color="0.8", label="truth")
        for e in pooled:
            if e != "truth":
                ax[c].hist(pooled[e][c], bins, histtype="step", label=e)
        ax[c].set_yscale("log")
        ax[c].set_title(f"{CH[c]} (G)")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(figs, "histograms.png"), dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(len(key), 3, figsize=(13, 4 * len(key)), squeeze=False)
    for j, e in enumerate(key):
        for c in range(3):
            t, p = pooled["truth"][c], pooled[e][c]
            lim = np.percentile(np.abs(t), 99.5) if t.size else 1
            ax[j, c].hexbin(t, p, gridsize=60, extent=(-lim, lim, -lim, lim), bins="log", cmap="viridis")
            ax[j, c].plot([-lim, lim], [-lim, lim], "r--", lw=1)
            ax[j, c].set_title(f"{e} {CH[c]}")
            ax[j, c].set_xlabel("truth (G)")
            ax[j, c].set_ylabel("prediction (G)")
    plt.tight_layout()
    fig.savefig(os.path.join(figs, "scatter.png"), dpi=90)
    plt.close(fig)

    # ---- bar chart with bootstrap CIs over HARPs
    mets = ["Br_rmse", "Bp_rmse", "Bt_rmse", "div_err_g_per_mm", "jz_err_ma_per_m2"]
    s = summary[(summary["mask"] == "strong") & summary.metric.isin(mets)]
    names = [n for n in ["E0_raw", "E0_cal", "E0b", "E0c"] + list(models) if n in set(s.model)]
    fig, ax = plt.subplots(1, len(mets), figsize=(4 * len(mets), 4.2))
    for j, met in enumerate(mets):
        d = s[s.metric == met].set_index("model").reindex(names)
        y = d.mean_over_harps.to_numpy()
        err = np.vstack([y - d.ci_lo.to_numpy(), d.ci_hi.to_numpy() - y])
        ax[j].bar(range(len(names)), y, yerr=np.nan_to_num(err), capsize=3)
        ax[j].set_xticks(range(len(names)))
        ax[j].set_xticklabels(names, rotation=60, fontsize=7)
        ax[j].set_title(f"{met} (|B|>200 G)", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(figs, "metrics_bar.png"), dpi=90)
    plt.close(fig)
