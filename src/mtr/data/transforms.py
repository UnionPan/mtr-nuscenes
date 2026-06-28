"""Image preprocessing and corruption transforms.

Normalization uses ImageNet statistics, matching the DINOv2 / ViT backbones.
Corruptions are used by the robustness evaluation (visual corruption, plus
frame and camera dropout handled at the tensor/mask level elsewhere).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_and_preprocess(path: str, size: int = 224) -> torch.Tensor:
    """Load an image, resize to ``size`` x ``size``, normalize -> CHW float tensor."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def corrupt_image(path: str, size: int, kind: str, severity: float,
                  rng: np.random.Generator) -> torch.Tensor:
    """Apply a visual corruption then preprocess. Returns CHW normalized tensor.

    kinds: ``gaussian_noise``, ``blur``, ``brightness``, ``jpeg`` (via downscale),
    ``none``. ``severity`` in [0, 1].
    """
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    if kind == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5 + 4.0 * severity))
    elif kind == "brightness":
        arr = np.asarray(img, dtype=np.float32) * (1.0 - 0.7 * severity)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif kind == "jpeg":
        d = max(2, int(size / (1 + 6 * severity)))
        img = img.resize((d, d), Image.BILINEAR).resize((size, size), Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if kind == "gaussian_noise":
        arr = arr + rng.normal(0, 0.05 + 0.25 * severity, arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 1)
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
