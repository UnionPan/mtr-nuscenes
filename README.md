# MTR — Multimodal Temporal Representation Pretraining for Autonomous Driving

Self-supervised/multimodal pretraining of temporal scene representations from
synchronized multi-camera driving clips, structured scene text, and ego-motion,
on **nuScenes Mini**, with a clean path to full nuScenes.

The model fuses a **frozen pretrained frame encoder** (DINOv2 ViT-B/14) with a
**temporal transformer**, a **text encoder** (DistilBERT), and an **ego-motion
head**, trained jointly with three objectives:

1. **InfoNCE video↔text alignment** (CLIP-style, symmetric).
2. **Masked temporal modeling** (reconstruct masked frame embeddings, data2vec-style).
3. **Ego-motion prediction** (future waypoints, anchored to the dataset-mean trajectory).

## Why these design choices

- **Frozen DINOv2 + cached embeddings.** On an RTX 8000 (Turing, 48 GB) the MVP
  trains the temporal model + heads on cached frame embeddings — ~270 clips/s vs
  ~30 clips/s for the full image pipeline. Unfreezing the backbone is a one-flag
  ablation (`freeze_encoder: false`, gradient checkpointing on).
- **fp16 autocast, fp32 optimizer states.** Turing has no native bf16 tensor
  cores; precision is configurable (`fp16`/`bf16`/`fp32`) and defaults to fp16.
- **Anchored motion head.** Future waypoints are predicted as a residual over the
  mean training trajectory with an explicit shrinkage penalty — essential
  regularization given Mini's 8 train / 2 val scenes (see the results report).
- **Multi-camera masked-mean aggregation** with per-camera embeddings, so camera
  dropout is handled natively at eval time.

## Layout

```
configs/            base.yaml + ablation configs (inherit via _base_)
src/mtr/
  data/             clip index builder, dataset, transforms, feature cache
  models/           frame/text encoders, temporal transformer, heads, losses, model
  train/            trainer (AMP, grad-accum/ckpt, DDP/FSDP), scheduler, distributed
  eval/             core metrics + full suite (robustness, efficiency)
  utils.py          config, seeding, checkpointing
scripts/            build_index, cache_features, train, evaluate, baseline, run_all
tests/              data / model / loss / metric interface tests (pytest)
reports/            generated results tables + written report
```

## Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# nuScenes Mini (~4 GB)
mkdir -p data/nuscenes && curl -L https://www.nuscenes.org/data/v1.0-mini.tgz \
  | tar -xz -C data/nuscenes
```

## Pipeline

```bash
# 1. Build clip index (synchronized multi-cam clips + ego-motion targets + captions)
python scripts/build_index.py --clip-len 6 --stride 2 --out data/index/mini_T6_s2.json

# 2. Cache frozen DINOv2 frame embeddings (one-time, reused by every feature-mode run)
python scripts/cache_features.py --index data/index/mini_T6_s2.json \
  --encoder dinov2_vitb14 --out data/cache/dinov2_vitb14

# 3. Train (single GPU)
python scripts/train.py --config configs/ablations/full.yaml
#    Multi-GPU (DDP):
torchrun --nproc_per_node=4 scripts/train.py --config configs/ablations/full.yaml

# 4. Evaluate (retrieval, linear probe, motion, robustness, efficiency)
python scripts/evaluate.py --config configs/ablations/full.yaml

# 5. Reproduce the whole study (baseline + headline + ablations + report)
bash scripts/run_all.sh
```

## Data pipeline

A *clip* is `T` consecutive 2 Hz keyframes from one scene. Each frame carries
`V` synchronized cameras. For every clip the index records:

- per-(frame, camera) image paths,
- ego pose (global x, y, yaw) per frame from the `LIDAR_TOP` reference,
- **motion target**: the next `H` ego waypoints in the last frame's ego frame,
- a **structured caption** from scene metadata: location, motion state
  (stopped / straight / turning), speed, and the top object categories.

Train/val use the official nuScenes-mini scene split (8 train / 2 val scenes).

## Model

| component | default | trained in MVP |
|---|---|---|
| frame encoder | DINOv2 ViT-B/14 (frozen) | no (cached) |
| view aggregator | per-camera embed + masked mean | yes |
| temporal transformer | 4 layers, dim 512, CLS pool | yes |
| text encoder | DistilBERT (frozen) | no |
| projections | MLP → 256-d, L2-norm | yes |
| motion head | anchored residual MLP → H×2 | yes |

~82 M total parameters, ~15.5 M trainable.

## Evaluation

`scripts/evaluate.py` reports:

- **Retrieval**: video↔text Recall@{1,5,10} and median rank.
- **Linear probe**: logistic-regression accuracy on frozen clip features
  (motion-state, pedestrian presence) with majority baselines.
- **Motion**: ADE / FDE (m) vs the mean-trajectory prior and a constant-velocity
  ceiling.
- **Robustness**: retrieval / motion under visual corruption (noise, blur,
  brightness), frame dropout, and camera dropout.
- **Efficiency**: throughput, latency, and peak memory for the full image
  pipeline vs the cached-feature pipeline.

See `reports/results.md` for measured numbers and discussion.

## Scaling to full nuScenes (measured)

The pipeline runs unchanged on full **v1.0-trainval** (official 700/150 split).
A login-free front-camera scale-up (850 scenes, 12.5k clips, ~5.5 GB from public
HF mirrors) is included and **measured** — see `reports/results.md` §5. At scale,
motion prediction crosses from below- to above the mean-trajectory prior,
retrieval reaches the top ~5 % (median rank 131/2682), and linear probes beat
their majority baselines.

```bash
# Front-camera trainval (no nuScenes login): metadata + CAM_FRONT keyframes
mkdir -p data/nuscenes_full && cd data/nuscenes_full
curl -L -o meta.tgz   https://huggingface.co/datasets/Xiaodong/Nuscenes-v1.0-trainval-CAM_FRONT/resolve/main/v1.0-trainval_meta.tgz
curl -L -o samples.zip https://huggingface.co/datasets/Xiaodong/Nuscenes-v1.0-trainval-CAM_FRONT/resolve/main/samples.zip
tar -xzf meta.tgz && unzip -q samples.zip && cd ../..

python scripts/build_index.py --dataroot data/nuscenes_full --version v1.0-trainval \
  --cameras 1 --require-images --out data/index/trainval_front_T6_s2.json
python scripts/cache_features.py --index data/index/trainval_front_T6_s2.json \
  --dataroot data/nuscenes_full --out data/cache/dinov2_vitb14_trainval_front
python scripts/train.py    --config configs/ablations/scale_trainval.yaml
python scripts/evaluate.py --config configs/ablations/scale_trainval.yaml --skip robustness
```

For the full 6-camera trainval set, register at nuscenes.org and download the
`v1.0-trainval*_blobs.tgz` (auth-gated, ~300 GB); the cached-feature design and
DDP/FSDP paths are built for that scale.
