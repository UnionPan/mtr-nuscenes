"""Frozen frame-embedding cache.

Stores one embedding per (image file) so temporal-model experiments avoid
re-running the frozen ViT backbone.  Layout on disk::

    <dir>/feats.npy     float16 [N, D]
    <dir>/keys.json     {relative_image_path: row_index, "_dim": D}

Build it once with ``cache_features.py``; load read-only during training.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np


class FeatureCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, "keys.json")) as f:
            self.keys: Dict[str, int] = json.load(f)
        self.dim = int(self.keys.pop("_dim"))
        self.feats = np.load(os.path.join(cache_dir, "feats.npy"), mmap_mode="r")

    def get(self, rel_path: str) -> np.ndarray:
        return np.asarray(self.feats[self.keys[rel_path]], dtype=np.float32)

    def __contains__(self, rel_path: str) -> bool:
        return rel_path in self.keys

    @staticmethod
    def save(cache_dir: str, keys: List[str], feats: np.ndarray) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, "feats.npy"), feats.astype(np.float16))
        mapping = {k: i for i, k in enumerate(keys)}
        mapping["_dim"] = int(feats.shape[1])
        with open(os.path.join(cache_dir, "keys.json"), "w") as f:
            json.dump(mapping, f)
