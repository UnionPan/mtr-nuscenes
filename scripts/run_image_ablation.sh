#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
F='Loading|UNEXPECTED|LOAD REPORT|^Key|^vocab|^Notes|can be ignored|nested_tensor|warnings.warn|FutureWarning|GradScaler|UserWarning'
for cfg in frozen_img adapt_img; do
  echo "===== TRAIN $cfg ====="
  python3 scripts/train.py --config configs/ablations/$cfg.yaml 2>&1 | grep -vE "$F" | grep -E "model\]|eval ep|final|done in"
  echo "===== EVAL $cfg ====="
  python3 scripts/evaluate.py --config configs/ablations/$cfg.yaml 2>&1 | grep -vE "$F" | grep -E "loaded|R@1|ade|clips/s|linear_probe|saved"
done
echo "===== COLLECT ====="
python3 scripts/collect_results.py > /dev/null 2>&1
echo "IMAGE_ABLATION_DONE"
