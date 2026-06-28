#!/usr/bin/env python3
"""Precompute frozen frame-encoder embeddings for every image referenced by an
index, so temporal-model experiments can run in fast 'feature' mode.

Image loading/preprocessing is parallelized with a DataLoader (the bottleneck is
disk + PIL, not the GPU encode), which keeps the GPU fed and avoids BLAS thread
thrashing in the main process."""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mtr.data.feature_cache import FeatureCache
from mtr.data.index import load_index
from mtr.data.transforms import load_and_preprocess
from mtr.models.frame_encoder import FrameEncoder

torch.set_num_threads(4)  # avoid oversubscription with DataLoader workers


class ImagePathDataset(Dataset):
    def __init__(self, paths, dataroot, size):
        self.paths, self.dataroot, self.size = paths, dataroot, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_and_preprocess(os.path.join(self.dataroot, self.paths[i]), self.size), i


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

    loader = DataLoader(ImagePathDataset(paths, args.dataroot, args.image_size),
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=True)

    t0 = time.time()
    done = 0
    for imgs, idxs in loader:
        imgs = imgs.to(args.device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=dt, enabled=(args.precision != "fp32")):
            emb = enc(imgs)
        feats[idxs.numpy()] = emb.float().cpu().numpy().astype(np.float16)
        done += len(idxs)
        print(f"  {done}/{len(paths)}  ({done/max(time.time()-t0,1e-6):.0f} img/s)", flush=True)

    FeatureCache.save(args.out, paths, feats)
    print(f"saved cache to {args.out}  dim={dim}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
