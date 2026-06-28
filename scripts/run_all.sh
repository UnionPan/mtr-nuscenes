#!/usr/bin/env bash
# Full experiment sweep on nuScenes Mini: baseline + headline + ablations + report.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
FILTER='Loading weights|UNEXPECTED|LOAD REPORT|^Key|^vocab|^Notes|can be ignored|nested_tensor|warnings.warn|FutureWarning|GradScaler|UserWarning'

FEATURE_CFGS="full no_mlm contrastive_only motion_only ctx_t4 ctx_t8"
IMAGE_CFGS="frozen_img adapt_img"

echo "===== no-training frozen baseline ====="
python3 scripts/baseline_frozen.py 2>&1 | grep -vE "$FILTER" | tail -20

for cfg in $FEATURE_CFGS $IMAGE_CFGS; do
  echo ""
  echo "===== TRAIN $cfg ====="
  python3 scripts/train.py --config configs/ablations/$cfg.yaml 2>&1 | grep -vE "$FILTER" | grep -E "model\]|eval ep|final|done in"
  echo "===== EVAL $cfg ====="
  python3 scripts/evaluate.py --config configs/ablations/$cfg.yaml 2>&1 | grep -vE "$FILTER" | grep -E "loaded|R@1|ade|clips/s|linear_probe|saved"
done

echo ""
echo "===== COLLECT RESULTS ====="
python3 scripts/collect_results.py 2>&1 | tail -60
echo "ALL_DONE"
