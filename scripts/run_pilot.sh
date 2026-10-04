#!/usr/bin/env bash
# One-month pilot (February 2011, validation split only):
#   - reliability of the target's horizontal divergence by scale (Figure 1, Table 4),
#   - the pixel-level divergence-loss sweep (lambda 0.01 .. 100) and the other pilot losses,
#   - the input-setup ablation: viewing geometry and vector-consistent flips (Table 5).
#
#   JSOC_EMAIL=you@example.com bash scripts/run_pilot.sh        (optional: DEVICE=cuda|cpu)
#
# JSOC allows one pending export per registered email, so the download steps run one after another.
# Every step can be re-run: the download skips finished days and training skips finished runs.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${JSOC_EMAIL:?set JSOC_EMAIL to an email address registered with JSOC (http://jsoc.stanford.edu/ajax/register_email.html)}"
DEVICE=${DEVICE:-cuda}
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

train() {  # python -m sr.train, skipping runs whose meta.json says "finished"
  local out
  out=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['out_dir'])" "$1")
  if python -c "import json, sys; sys.exit(json.load(open(sys.argv[1])).get('status') != 'finished')" "$out/meta.json" 2>/dev/null; then
    echo "skip $1 (finished)"; return 0
  fi
  python -m sr.train --config "$1" --device "$DEVICE"
}

# 1. MDI / SHARP pairs (regions and 16x16 LR patches) -> data/pilot/
python -u pipeline/build_dataset.py --run pilot --email "$JSOC_EMAIL" --start 2011-02-01 --end 2011-03-01 \
  --save_mode both --max_center_angle 60 --min_align_corr 0.5 --lr_grid_fix --align_refine xcorr \
  --mdi_quality_mask 0x80000000

# 2. The split used in the paper is splits/pilot.json (region groups, never frames). To regenerate it:
#    python pipeline/split_dataset.py --run pilot

# 3. Training-split statistics; SHARP frames 12 min after a sample of training frames;
#    signal-to-noise of div_h by smoothing scale (Figure 1, Table 4)
python scripts/phase4_stats.py --run pilot
python scripts/noise_floor_temporal.py --run pilot --n_pairs 30 --email "$JSOC_EMAIL"
python scripts/div_snr_by_scale.py --run pilot

# 4. Inputs of the other pilot configs: 32x32 LR patch set, per-pixel viewing geometry
python scripts/make_patch_set.py --src_run pilot --dst_run pilot_p32 --patch_size_lr 32 --stride_lr 16
python scripts/make_geometry.py --run pilot

# 5. Training. E1 = MSE, E2 = M24 loss, E3_l* = pixel-level divergence loss at weight *,
#    E4-E8 = further pilot losses and the U-Net (E7), *_p32 = 32x32 patches,
#    V_E1 / V_E1_geo / V_E1_aug = MSE with geometry + flips / geometry only / flips only.
for c in E1 E2 E3_l0.01 E3_l0.1 E3_l1 E3_l10 E3_l100 E1_p32 E3_p32 E3_p32_l100 E4 E5 E6 E7 E8 \
         V_E1 V_E1_geo V_E1_aug; do
  train "configs/pilot_$c.yaml"
done

# 6. Validation evaluation: pixel-level metrics (results/pilot/eval_val) and the setup ablation
#    (results/pilot/eval_val_v2_ablation)
python -m sr.evaluate --run pilot --split val --device "$DEVICE" --figures \
  --exps E1 E1_p32 E2 E3_l0.01 E3_l0.1 E3_l1 E3_l10 E3_l100 E3_p32 E3_p32_l100 E4 E5 E6 E7 E8
python scripts/eval_v2.py --run pilot --device "$DEVICE" --n_boot 1000 \
  --exps V_E1_geo V_E1_aug V_E1 E1 --out_dir results/pilot/eval_val_v2_ablation

echo "pilot done: data/pilot/div_snr_by_scale.json, results/pilot/eval_val, results/pilot/eval_val_v2_ablation"
