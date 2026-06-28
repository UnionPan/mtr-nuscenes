"""Cosine learning-rate schedule with linear warmup."""
from __future__ import annotations

import math

import torch


def cosine_warmup(optimizer, total_steps: int, warmup_steps: int, min_lr_frac: float = 0.01):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
