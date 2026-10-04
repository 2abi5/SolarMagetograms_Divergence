#!/usr/bin/env bash
# Full-year experiment (Section 4 and the appendix): 2010-05-01 .. 2011-04-11,
# U-Net and HighRes-net x {MSE (E1), M24 loss (E2), ours (E3)} plus the U-Net ablations
# MSE + divergence (E4) and MSE + gradient + SSIM (E5), three seeds each (24 runs, about 60 GPU-hours
# on RTX A4000 cards), evaluated on validation and test, then the paired tests across seeds.
#
#   JSOC_EMAIL=you@example.com bash scripts/run_full.sh         (optional: DEVICE=cuda|cpu)
#
# Every step can be re-run: the download skips finished weeks and training skips finished runs.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${JSOC_EMAIL:?set JSOC_EMAIL to an email address registered with JSOC (http://jsoc.stanford.edu/ajax/register_email.html)}"
DEVICE=${DEVICE:-cuda}
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
SEEDS="0 1 2"

train() {  # python -m sr.train, skipping runs whose meta.json says "finished"
  local out
  out=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['out_dir'])" "$1")
  if python -c "import json, sys; sys.exit(json.load(open(sys.argv[1])).get('status') != 'finished')" "$out/meta.json" 2>/dev/null; then
    echo "skip $1 (finished)"; return 0
  fi
  python -m sr.train --config "$1" --device "$DEVICE"
}

# 1. MDI / SHARP region pairs, one week at a time -> data/full/
#    (several registered emails can share the range: see --claims_dir and --skip_done_in in
#    pipeline/build_dataset.py, and scripts/merge_runs.py to merge their run folders)
python -u pipeline/build_dataset.py --run full --email "$JSOC_EMAIL" --start 2010-05-01 --end 2011-04-12 --window_days 7 \
  --save_mode regions --max_center_angle 60 --min_align_corr 0.5 --lr_grid_fix --align_refine xcorr \
  --mdi_quality_mask 0x80000000 --delete_raw --max_disk_gb 50 --max_pending_retries 12

# 2. Per-pixel viewing geometry and training-split statistics.
#    The split (splits/full.json) and the configs (configs/full_*.yaml, with the per-scale divergence
#    normalisation constants) used in the paper are included. To regenerate them:
#      python pipeline/split_dataset.py --run full --stratify_months --min_groups 5
#      python scripts/div_ms_norm.py --run full --mode regions --n 20000
#      python scripts/make_configs.py --stage full --run full --seeds 0 1 2 --samples_per_epoch 131072 \
#        --full_v2 data/full/div_ms_stats.json --full_controls
python scripts/make_geometry.py --run full
python scripts/phase4_stats.py --run full

# 3. Training: F_U_* = U-Net, F_* = HighRes-net; E1 = MSE, E2 = M24 loss, E3 = ours,
#    E4 = MSE + divergence, E5 = MSE + gradient + SSIM
for s in $SEEDS; do
  for e in F_U_E1 F_U_E2 F_U_E3 F_U_E4 F_U_E5 F_E1 F_E2 F_E3; do
    train "configs/full_${e}_s${s}.yaml"
  done
done

# 4. Evaluation on every third frame of each HARP (consecutive 96-min frames are near duplicates).
#    The test split is read only with --allow_test.
for s in $SEEDS; do
  CORE="F_U_E1_s$s F_U_E2_s$s F_U_E3_s$s F_E1_s$s F_E2_s$s F_E3_s$s"
  CTRL="F_U_E4_s$s F_U_E5_s$s"
  python scripts/eval_v2.py --run full --device "$DEVICE" --every 3 --n_boot 1000 --exps $CORE --out_dir results/full/eval_val_s$s
  python scripts/eval_v2.py --run full --device "$DEVICE" --every 3 --n_boot 1000 --exps $CTRL --out_dir results/full/eval_controls_val_s$s
  python scripts/eval_v2.py --run full --device "$DEVICE" --every 3 --n_boot 1000 --split test --allow_test \
    --exps $CORE --out_dir results/full/eval_test_s$s
  python scripts/eval_v2.py --run full --device "$DEVICE" --every 3 --n_boot 1000 --split test --allow_test \
    --exps $CTRL --out_dir results/full/eval_controls_test_s$s
done

# 5. Pre-registered paired Wilcoxon tests over HARPs on seed-averaged values
python scripts/full_hypotheses.py --evals results/full/eval_val_s0 results/full/eval_val_s1 results/full/eval_val_s2 \
  --out results/full/hypotheses_full.json
python scripts/full_hypotheses.py --evals results/full/eval_test_s0 results/full/eval_test_s1 results/full/eval_test_s2 \
  --out results/full/hypotheses_test.json
for split in val test; do
  python scripts/ablation_hypotheses.py \
    --evals results/full/eval_${split}_s0 results/full/eval_${split}_s1 results/full/eval_${split}_s2 \
            results/full/eval_controls_${split}_s0 results/full/eval_controls_${split}_s1 results/full/eval_controls_${split}_s2 \
    --out results/full/hypotheses_ablation_${split}.json
done

# 6. Small confined-flux regions (validation; appendix table)
EXPS=$(for e in F_U_E1 F_U_E2 F_U_E3 F_E1 F_E2 F_E3; do for s in $SEEDS; do printf "%s_s%s " $e $s; done; done)
python scripts/eval_small_flux.py --run full --device "$DEVICE" --every 3 --exps $EXPS \
  --pairs F_U_E3:F_U_E2 F_U_E3:F_U_E1 F_E3:F_E2 F_E3:F_E1

echo "full-year experiment done: results/full/"
