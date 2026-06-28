"""Distributed helpers for multi-GPU scaling.

DDP is the default and the right tool for this model (only the temporal model +
heads train, ~16M params).  An FSDP wrapper is provided for the documented path
where the visual backbone is unfrozen and full-model sharding helps; select it
with ``train.distributed: fsdp``.  Launch with ``torchrun``.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize the process group from torchrun env vars. Returns
    (rank, world_size, local_rank). Falls back to single-process (0, 1, 0)."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def wrap_model(model, local_rank: int, mode: str = "ddp"):
    if mode == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return FSDP(model, device_id=local_rank)
    from torch.nn.parallel import DistributedDataParallel as DDP
    return DDP(model, device_ids=[local_rank], find_unused_parameters=True)
