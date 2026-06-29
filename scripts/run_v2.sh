#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1
F='Loading|UNEXPECTED|LOAD REPORT|^Key|^vocab|^Notes|can be ignored|nested_tensor|warnings.warn|FutureWarning|GradScaler|UserWarning'
echo "=== TRAIN scale_6cam_v2 $(date +%H:%M:%S) ==="
python3 scripts/train.py --config configs/ablations/scale_6cam_v2.yaml 2>&1 | grep -vE "$F" | grep -E "model\]|eval ep|final"
echo "=== EVAL scale_6cam_v2 $(date +%H:%M:%S) ==="
python3 scripts/evaluate.py --config configs/ablations/scale_6cam_v2.yaml --skip robustness 2>&1 | grep -vE "$F" | grep -E "loaded|R@1=|ade=|linear_probe|saved"
echo "V2_DONE"
