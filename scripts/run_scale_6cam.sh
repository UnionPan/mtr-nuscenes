#!/usr/bin/env bash
# Waits for the 6-camera blob download to finish, then runs the full pipeline:
# build index -> cache features -> train -> evaluate. Designed to run unattended.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1
F='Loading|UNEXPECTED|LOAD REPORT|^Key|^vocab|^Notes|can be ignored|nested_tensor|warnings.warn|FutureWarning|GradScaler|UserWarning'

echo "waiting for ALL_BLOBS_DONE..."
until grep -q "ALL_BLOBS_DONE" runs_dl6cam.log 2>/dev/null; do sleep 30; done
echo "=== blobs ready: $(date +%H:%M:%S) ==="

echo "=== build 6-cam trainval index ==="
python3 scripts/build_index.py --dataroot data/nuscenes_full --version v1.0-trainval \
  --clip-len 6 --stride 2 --future-horizon 6 --cameras 6 --require-images \
  --out data/index/trainval_6cam_T6_s2.json

echo "=== cache 6-cam features (~200k images) ==="
python3 scripts/cache_features.py --index data/index/trainval_6cam_T6_s2.json \
  --dataroot data/nuscenes_full --encoder dinov2_vitb14 --batch-size 256 --num-workers 12 \
  --out data/cache/dinov2_vitb14_trainval_6cam | tail -3

echo "=== TRAIN scale_trainval_6cam ==="
python3 scripts/train.py --config configs/ablations/scale_trainval_6cam.yaml 2>&1 \
  | grep -vE "$F" | grep -E "model\]|eval ep|final|done in"

echo "=== EVAL scale_trainval_6cam ==="
python3 scripts/evaluate.py --config configs/ablations/scale_trainval_6cam.yaml --skip robustness 2>&1 \
  | grep -vE "$F" | grep -E "loaded|R@1=|ade=|clips/s|linear_probe|saved"

echo "SCALE6CAM_DONE"
