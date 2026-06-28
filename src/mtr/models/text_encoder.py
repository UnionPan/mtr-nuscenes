"""Pretrained text encoder (DistilBERT by default) with masked mean pooling.

Tokenizes raw caption strings internally so the dataset stays framework-free.
Frozen by default; only the downstream projection (in ``MTRModel``) trains,
which avoids overfitting on the tiny nuScenes-mini caption vocabulary.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, name: str = "distilbert-base-uncased", frozen: bool = True,
                 max_length: int = 64):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.embed_dim = self.model.config.hidden_size
        self.max_length = max_length
        self.frozen = frozen
        if frozen:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, captions: List[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        tok = self.tokenizer(captions, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.model(**tok).last_hidden_state            # [B, L, H]
        mask = tok["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return pooled
