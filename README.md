# Noise-Aware Divergence Losses for Super-Resolution of Solar Vector Magnetograms
 
Code for the AISTATS 2027 submission of the same title.

The code learns to map SOHO/MDI line-of-sight magnetograms to SDO/HMI SHARP vector fields
($B_p$, $B_t$, $B_r$) at 4× resolution. It contains:

- the data pipeline;
- the test–retest reliability analysis of the target's horizontal divergence (Section 3.3);
- the multi-scale divergence-matching loss (Section 3.4);
- training and evaluation code for U-Net and HighRes-net;
- the pre-registered paired tests;
- the scripts that regenerate every table, number and figure of the paper.

## Contents

| Path | What it does |
|---|---|
| `data_active_region_created.py` | Downloads MDI and SHARP (`hmi.sharp_cea_720s`) records from JSOC, reprojects MDI onto each SHARP's CEA grid, aligns the pair, and saves regions and patches |
| `pipeline/build_dataset.py` | Resumable driver for the above, one day or one week at a time |
| `pipeline/split_dataset.py` | Train/validation/test split by active-region group (never by frame), optionally stratified by month |
| `sr/physics.py` | SHARP CEA conventions and the horizontal divergence $\mathrm{div}_h$ |
| `sr/losses.py` | Masked MSE, Sobel gradient, histogram, SSIM and divergence losses; `div_mode: match_ms` is the multi-scale divergence-matching loss |
| `sr/geometry.py`, `scripts/make_geometry.py` | Per-pixel line-of-sight projection coefficients (viewing-geometry input channels) |
| `sr/datasets.py` | Patch and region datasets, vector-consistent flips |
| `sr/models/` | U-Net, single-frame HighRes-net (built from the converter blocks in `source/models/`), EDSR |
| `sr/train.py` | Config-driven training (Adam, lr 1e-4, batch 64, early stopping on validation RMSE) |
| `sr/evaluate.py`, `sr/metrics.py`, `scripts/eval_v2.py` | Per-region evaluation: pixel errors, divergence correlation by scale, flux and extreme-value metrics |
| `scripts/noise_floor_temporal.py`, `scripts/div_snr_by_scale.py` | Downloads SHARP frames 12 min apart and measures the reliability and SNR of $\mathrm{div}_h$ by smoothing scale |
| `scripts/phase4_stats.py`, `scripts/div_ms_norm.py` | Training-split statistics: field normalisation and the per-scale normalisation of the divergence loss |
| `scripts/make_configs.py` | Writes the run configurations |
| `scripts/full_hypotheses.py`, `scripts/ablation_hypotheses.py` | Paired Wilcoxon tests over HARPs on seed-averaged values |
| `scripts/eval_small_flux.py` | Evaluation on small confined-flux regions |
| `scripts/run_pilot.sh`, `scripts/run_full.sh` | The complete pilot and full-year pipelines (download → train → evaluate → test) |
| `configs/` | Every configuration used: `full_*` (24 full-year runs) and `pilot_*` |
| `splits/` | The splits used in the paper (HARP numbers per split) |
| `paper/scripts/` | Regenerate the paper's tables, numbers and figures from `results/` |
| `tests/` | Unit tests (physics sign conventions, units, masks, loss gradients, splits, pipeline) |

Run names:

- `F_U_*` are the full-year U-Net runs, `F_*` the full-year HighRes-net runs, and `_s0/_s1/_s2` the seed.
- `E1` = MSE.
- `E2` = the loss of Muñoz-Jaramillo et al. (2024), called "M24" in the paper.
- `E3` = ours (MSE + gradient + SSIM + multi-scale divergence).
- `E4` = MSE + divergence.
- `E5` = MSE + gradient + SSIM.
- Pilot names are explained in `scripts/run_pilot.sh`.

## Installation

Python 3.12 and a CUDA GPU are needed for training. Evaluation and the paper scripts also run on CPU.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # pick the build for your CUDA
pip install -r requirements.txt
python -m pytest -q          # unit tests; no data, network access or GPU needed (two geometry tests skip until data/pilot exists)
```

`source env.sh` is optional. It keeps library caches inside the project folder and activates `./.venv`.

## Data

All data are public and come from the Joint Science Operations Center (JSOC). Downloading needs an
email address registered with JSOC (<http://jsoc.stanford.edu/ajax/register_email.html>). JSOC allows
one pending export per email, so downloads run one after another.

Disk space:

- the pilot (February 2011) needs about 16 GB;
- the full year (2010-05-01 to 2011-04-11) needs about 25 GB, with raw FITS files deleted after processing.

JSOC reprocesses data occasionally, so a new download can differ slightly from ours. The splits we used
are in `splits/`.

## Reproducing the paper

```bash
JSOC_EMAIL=you@example.com bash scripts/run_pilot.sh   # pilot: Figure 1, Tables 4-5, pixel-level loss sweep
JSOC_EMAIL=you@example.com bash scripts/run_full.sh    # full year: 24 runs (~60 GPU-hours), val + test evaluation, tests

python paper/scripts/make_tables.py                     # every number and table -> paper/tables/
python paper/scripts/make_figures.py                    # Figures 1, 3, 4 -> paper/figures/out/
CUDA_VISIBLE_DEVICES="" python paper/scripts/make_map_figure.py   # Figures 2, 5 (seed-0 models, CPU)
```

Both shell scripts can be stopped and re-run. They skip finished days, weeks and training runs.
`make_tables.py` recomputes every statistic from the per-region evaluation outputs and checks it
against the hypothesis-test outputs of step 5 of `run_full.sh`. It stops if any value disagrees.

| Paper item | Produced by | Reads |
|---|---|---|
| Figure 1, Table 4 (reliability of $\mathrm{div}_h$) | `scripts/div_snr_by_scale.py` | `data/pilot/div_snr_by_scale.json` |
| Pilot pixel-level divergence loss | `sr/evaluate.py` | `results/pilot/eval_val/` |
| Table 5 (input setup) | `scripts/eval_v2.py` | `results/pilot/eval_val_v2_ablation/` |
| Tables 1, 6 (test, validation) | `scripts/eval_v2.py` | `results/full/eval_{test,val}_s*/` |
| Table 2, Figure 3 (paired tests, ablation) | `scripts/eval_v2.py`, `scripts/{full,ablation}_hypotheses.py` | `results/full/eval_*_s*/`, `results/full/hypotheses_*.json` |
| Figure 4, Table 7 (gain by scale) | `scripts/eval_v2.py` | `div_by_scale_per_harp.csv` in `results/full/eval_*_s*/` |
| Table 3 (loss weights) | `configs/` | — |
| Table 8 (small confined-flux regions) | `scripts/eval_small_flux.py` | `results/full/eval_small_flux/` |
| Table 9 (training) | `sr/train.py` | `results/full/*/meta.json` |
| Figures 2, 5 (maps) | `paper/scripts/make_map_figure.py` | `results/full/F_U_E{1,2,3}_s0/best.pt`, `data/full/` |

## Notes

- Trained weights and evaluation outputs are not included because of their size.
- `sr/train.py` and `sr/evaluate.py` record the git commit of the code in each run's `meta.json`.
- Some code comments cite internal design notes (`training_plan.md`, `reports/phase_*.md`). These notes are
  not part of this archive; the paper and its appendix describe the choices they record.
- The test split is read only when `--allow_test` is passed.

## License

GPL-3.0 (see `LICENSE`). The HighRes-net building blocks in `source/models/` are copied unchanged from
the public MDI/GONG-to-HMI converter of Muñoz-Jaramillo et al. (2024), which is released under GPL-3.0.
This code reuses those blocks and is distributed under the same license.
