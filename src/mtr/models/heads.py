"""Projection, motion-prediction, and masked-frame-reconstruction heads."""
from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """MLP projection to the shared contrastive embedding space (L2-normalized)."""

    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 1024,
                 normalize: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1)
        return x


class MotionHead(nn.Module):
    """Predict ``H`` future ego waypoints (x, y) from the pooled clip embedding.

    Uses *residual / anchor* prediction: the output is the dataset-mean future
    trajectory (a registered buffer, set from the training set) plus a learned
    residual whose final layer is zero-initialized.  At init the head therefore
    reproduces the constant-mean baseline and only learns generalizable
    deviations — crucial regularization in the small-scene nuScenes-mini regime.
    Dropout further limits memorization."""

    def __init__(self, in_dim: int, horizon: int = 6, hidden: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.horizon = horizon
        # Low-capacity head; the residual is further constrained by an explicit
        # shrinkage penalty toward the anchor (see losses.motion_reg).
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, horizon * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("anchor", torch.zeros(horizon, 2))

    def set_anchor(self, mean_traj: torch.Tensor) -> None:
        self.anchor.copy_(mean_traj.to(self.anchor.device, self.anchor.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x).view(x.shape[0], self.horizon, 2)
        return self.anchor.unsqueeze(0) + residual


class MaskedFramePredictor(nn.Module):
    """Reconstruct the original frame embedding at masked temporal positions."""

    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)
