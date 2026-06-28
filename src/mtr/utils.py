"""Shared utilities: config loading, seeding, logging, checkpointing."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict:
    """Load a YAML config, optionally merging a base config and CLI overrides.

    Supports a top-level ``_base_`` key pointing to a parent YAML (relative to
    the child), enabling small ablation configs that inherit from ``base.yaml``.
    Overrides use dotted keys, e.g. ``{"train.lr": 1e-3}``.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("_base_"))
        base = load_config(base_path)
        cfg = deep_merge(base, cfg)
    if overrides:
        for k, v in overrides.items():
            set_by_path(cfg, k, v)
    return cfg


def deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_by_path(cfg: Dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def amp_dtype(precision: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def save_checkpoint(path: str, model, optimizer, scaler, scheduler,
                    step: int, epoch: int, cfg: Dict, best: Optional[float] = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scaler": scaler.state_dict() if scaler else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step, "epoch": epoch, "cfg": cfg, "best": best,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scaler=None, scheduler=None,
                    map_location="cpu") -> Dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
