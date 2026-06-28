#!/usr/bin/env bash
# Front-camera scale comparison: Mini (10 scenes) vs full trainval (850 scenes).
# Robustness is skipped here (image-mode, hang-prone at scale; characterized on
# the Mini headline model already); main metrics + efficiency are kept.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1
F='Loading|UNEXPECTED|LOAD REPORT|^Key|^vocab|^Notes|can be ignored|nested_tensor|warnings.warn|FutureWarning|GradScaler|UserWarning'
for cfg in mini_front scale_trainval; do
  echo "===== TRAIN $cfg ====="
  python3 scripts/train.py --config configs/ablations/$cfg.yaml 2>&1 | grep -vE "$F" | grep -E "model\]|eval ep|final|done in"
  echo "===== EVAL $cfg ====="
  python3 scripts/evaluate.py --config configs/ablations/$cfg.yaml --skip robustness 2>&1 | grep -vE "$F" | grep -E "loaded|R@1=|ade=|clips/s|linear_probe|saved"
done
echo "SCALE_DONE"
