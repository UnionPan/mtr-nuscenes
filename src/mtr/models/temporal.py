"""Temporal transformer over per-frame embeddings.

Consumes view-aggregated, projected frame tokens ``[B, T, D]`` and produces
per-frame contextual tokens plus a pooled clip embedding (a prepended CLS
token).  Supports masked temporal modeling: selected frame inputs are replaced
by a learned mask token and the model is asked to reconstruct the original
frame embeddings at those positions.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(self, dim: int = 512, depth: int = 4, heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 max_frames: int = 16, grad_checkpointing: bool = False):
        super().__init__()
        self.dim = dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.grad_checkpointing = grad_checkpointing
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, frame_mask: Optional[torch.Tensor] = None,
                mlm_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: ``[B, T, D]``. frame_mask/mlm_mask: ``[B, T]`` bool (True = valid /
        True = masked-for-prediction). Returns ``(cls [B, D], tokens [B, T, D])``."""
        B, T, _ = x.shape
        if mlm_mask is not None:
            x = torch.where(mlm_mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, x], dim=1)                       # [B, T+1, D]
        h = h + self.pos_embed[:, : T + 1]

        if frame_mask is not None:
            cls_valid = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            key_padding = ~torch.cat([cls_valid, frame_mask], dim=1)  # True = pad
        else:
            key_padding = None

        if self.grad_checkpointing and self.training:
            h = torch.utils.checkpoint.checkpoint_sequential(
                self.encoder.layers, len(self.encoder.layers), h, use_reentrant=False)
        else:
            h = self.encoder(h, src_key_padding_mask=key_padding)
        h = self.norm(h)
        return h[:, 0], h[:, 1:]
