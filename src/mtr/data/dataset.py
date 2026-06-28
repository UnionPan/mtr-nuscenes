"""Clip dataset and collate function.

Two operating modes:
  * ``image``   : returns pixel tensors ``[T, V, 3, H, W]`` (full pipeline).
  * ``feature`` : returns cached frozen-encoder embeddings ``[T, V, D]``
                  (fast temporal-model experiments; see ``feature_cache``).

Optional corruption / dropout knobs are provided for the robustness evaluation.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .index import load_index
from .transforms import corrupt_image, load_and_preprocess


class NuScenesClipDataset(Dataset):
    def __init__(
        self,
        index_path: str,
        split: str = "train",
        image_size: int = 224,
        mode: str = "image",                 # "image" | "feature"
        feature_cache: Optional["object"] = None,
        dataroot: Optional[str] = None,
        # robustness knobs (used at eval time):
        corruption: Optional[str] = None,    # kind passed to corrupt_image
        corruption_severity: float = 0.0,
        frame_dropout: float = 0.0,          # prob. of dropping a non-anchor frame
        camera_dropout: float = 0.0,         # prob. of dropping a camera view
        deterministic: bool = True,          # True: fixed per-clip (eval); False: per-epoch random
        seed: int = 0,
    ):
        index = load_index(index_path)
        self.meta = index["meta"]
        self.clips: List[Dict] = index[split]
        self.split = split
        self.image_size = image_size
        self.mode = mode
        self.feature_cache = feature_cache
        self.dataroot = dataroot or self.meta["dataroot"]
        self.corruption = corruption
        self.corruption_severity = corruption_severity
        self.frame_dropout = frame_dropout
        self.camera_dropout = camera_dropout
        self.deterministic = deterministic
        self.base_seed = seed
        if mode == "feature" and feature_cache is None:
            raise ValueError("mode='feature' requires a feature_cache")

    def __len__(self) -> int:
        return len(self.clips)

    def _abspath(self, rel: str) -> str:
        return os.path.join(self.dataroot, rel)

    def __getitem__(self, i: int) -> Dict:
        clip = self.clips[i]
        paths = clip["image_paths"]                  # [T][V] relative paths
        T, V = len(paths), len(paths[0])
        rng = np.random.default_rng(self.base_seed + i) if self.deterministic \
            else np.random.default_rng()

        # Masks. Anchor (last) frame is never dropped so motion target stays defined.
        frame_mask = torch.ones(T, dtype=torch.bool)
        cam_mask = torch.ones(T, V, dtype=torch.bool)
        if self.frame_dropout > 0:
            for t in range(T - 1):
                if rng.random() < self.frame_dropout:
                    frame_mask[t] = False
        if self.camera_dropout > 0:
            for t in range(T):
                for v in range(V):
                    if rng.random() < self.camera_dropout:
                        cam_mask[t, v] = False

        out: Dict = {
            "frame_mask": frame_mask,
            "cam_mask": cam_mask,
            "motion_target": torch.tensor(clip["motion_target"], dtype=torch.float32),
            "motion_valid": torch.tensor(clip["motion_valid"], dtype=torch.bool),
            "caption": clip["caption"],
            "speed": torch.tensor(clip["speed"], dtype=torch.float32),
            "meta": {"scene": clip["scene"], "index": i},
        }

        if self.mode == "feature":
            feats = np.stack([
                np.stack([self.feature_cache.get(paths[t][v]) for v in range(V)])
                for t in range(T)
            ])  # [T, V, D]
            out["features"] = torch.from_numpy(feats).float()
        else:
            imgs = torch.empty(T, V, 3, self.image_size, self.image_size)
            for t in range(T):
                for v in range(V):
                    p = self._abspath(paths[t][v])
                    if self.corruption and self.corruption != "none" and self.corruption_severity > 0:
                        imgs[t, v] = corrupt_image(p, self.image_size, self.corruption,
                                                   self.corruption_severity, rng)
                    else:
                        imgs[t, v] = load_and_preprocess(p, self.image_size)
            out["images"] = imgs
        return out


def collate_clips(batch: List[Dict]) -> Dict:
    """Stack a list of clip dicts into a batch. Captions stay a list of strings."""
    out: Dict = {
        "frame_mask": torch.stack([b["frame_mask"] for b in batch]),
        "cam_mask": torch.stack([b["cam_mask"] for b in batch]),
        "motion_target": torch.stack([b["motion_target"] for b in batch]),
        "motion_valid": torch.stack([b["motion_valid"] for b in batch]),
        "speed": torch.stack([b["speed"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }
    if "images" in batch[0]:
        out["images"] = torch.stack([b["images"] for b in batch])
    if "features" in batch[0]:
        out["features"] = torch.stack([b["features"] for b in batch])
    return out
