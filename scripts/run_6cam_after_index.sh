#!/usr/bin/env bash
# 6-cam trainval pipeline AFTER the index is already built: cache features
# (num_workers=8, the value proven stable on the front-cam cache) -> train -> eval.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1
F='Loading|UNEXPECTED|LOAD REPORT|^Key|^vocab|^Notes|can be ignored|nested_tensor|warnings.warn|FutureWarning|GradScaler|UserWarning'

echo "=== cache 6-cam features (nw=8) $(date +%H:%M:%S) ==="
python3 scripts/cache_features.py --index data/index/trainval_6cam_T6_s2.json \
  --dataroot data/nuscenes_full --encoder dinov2_vitb14 --batch-size 256 --num-workers 8 \
  --out data/cache/dinov2_vitb14_trainval_6cam | grep -E "unique|img/s|saved cache" | tail -20

echo "=== TRAIN scale_trainval_6cam $(date +%H:%M:%S) ==="
python3 scripts/train.py --config configs/ablations/scale_trainval_6cam.yaml 2>&1 \
  | grep -vE "$F" | grep -E "model\]|eval ep|final"

echo "=== EVAL scale_trainval_6cam $(date +%H:%M:%S) ==="
python3 scripts/evaluate.py --config configs/ablations/scale_trainval_6cam.yaml --skip robustness 2>&1 \
  | grep -vE "$F" | grep -E "loaded|R@1=|ade=|linear_probe|saved"

echo "SCALE6CAM_DONE"
