#!/usr/bin/env python3
"""Precompute frozen frame-encoder embeddings for every image referenced by an
index, so temporal-model experiments can run in fast 'feature' mode.

Image loading/preprocessing is parallelized with a *thread pool* (PIL releases
the GIL during decode), not a multiprocessing DataLoader: the latter deadlocks
in this environment on large image sets. Threads keep the GPU fed reliably."""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mtr.data.feature_cache import FeatureCache
from mtr.data.index import load_index
from mtr.data.transforms import load_and_preprocess
from mtr.models.frame_encoder import FrameEncoder

torch.set_num_threads(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--encoder", default="dinov2_vitb14")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    index = load_index(args.index)
    paths = set()
    for split in ("train", "val"):
        for clip in index.get(split, []):
            for frame in clip["image_paths"]:
                paths.update(frame)
    paths = sorted(paths)
    print(f"unique images: {len(paths)}", flush=True)

    enc = FrameEncoder(args.encoder, args.image_size, frozen=True).to(args.device).eval()
    dim = enc.embed_dim
    feats = np.zeros((len(paths), dim), dtype=np.float16)
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]

    def load_one(p):
        return load_and_preprocess(os.path.join(args.dataroot, p), args.image_size)

    t0 = time.time()
    bs = args.batch_size
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        for i in range(0, len(paths), bs):
            chunk = paths[i:i + bs]
            imgs = torch.stack(list(pool.map(load_one, chunk))).to(args.device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=dt, enabled=(args.precision != "fp32")):
                emb = enc(imgs)
            feats[i:i + len(chunk)] = emb.float().cpu().numpy().astype(np.float16)
            if (i // bs) % 20 == 0:
                done = i + len(chunk)
                print(f"  {done}/{len(paths)}  ({done/max(time.time()-t0,1e-6):.0f} img/s)", flush=True)

    FeatureCache.save(args.out, paths, feats)
    print(f"saved cache to {args.out}  dim={dim}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
