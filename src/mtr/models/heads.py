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


class KinematicMotionHead(nn.Module):
    """Predict future ego waypoints as a residual over a causal constant-velocity
    anchor, conditioned on the *temporal sequence* of frame tokens (not a single
    pooled vector) and the observed kinematics.

    Combines four ideas: (1) the per-clip CV trajectory is the anchor, so the
    head only learns deviations (turns / accel) from physics; (2) observed
    kinematics ``kin`` are fed in directly; (3) it consumes the anchor-frame
    token plus a masked mean over frames, preserving instantaneous-motion
    information that mean-pooled CLS discards; the residual head is zero-init so
    the model starts exactly at the CV baseline."""

    def __init__(self, in_dim: int, horizon: int = 6, kin_dim: int = 3,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.horizon = horizon
        self.kin_dim = kin_dim
        self.kin_norm = nn.LayerNorm(kin_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2 + kin_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, horizon * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens: torch.Tensor, frame_mask, kin: torch.Tensor,
                cv_anchor: torch.Tensor) -> torch.Tensor:
        # tokens [B,T,D], frame_mask [B,T] bool|None, kin [B,kin_dim], cv_anchor [B,H,2]
        last = tokens[:, -1]                                # anchor-frame temporal token
        if frame_mask is not None:
            m = frame_mask.unsqueeze(-1).to(tokens.dtype)
            mean = (tokens * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            mean = tokens.mean(1)
        feat = torch.cat([last, mean, self.kin_norm(kin)], dim=-1)
        residual = self.net(feat).view(tokens.shape[0], self.horizon, 2)
        return cv_anchor + residual


class MaskedFramePredictor(nn.Module):
    """Reconstruct the original frame embedding at masked temporal positions."""

    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)
