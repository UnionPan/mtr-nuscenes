"""Full multimodal temporal model.

Wires the frame encoder, multi-camera view aggregation, temporal transformer,
text encoder, and the projection / motion / masked-reconstruction heads.  The
forward pass returns the representations and predictions consumed by the loss
module; ``encode_video`` / ``encode_text`` are convenience paths for evaluation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frame_encoder import FrameEncoder
from .heads import MaskedFramePredictor, MotionHead, ProjectionHead
from .temporal import TemporalTransformer
from .text_encoder import TextEncoder


class MTRModel(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        proj_dim: int = 256,
        frame_encoder_name: str = "dinov2_vitb14",
        image_size: int = 224,
        freeze_encoder: bool = True,
        grad_checkpointing: bool = False,
        temporal_depth: int = 4,
        temporal_heads: int = 8,
        max_frames: int = 16,
        max_views: int = 6,
        text_encoder_name: str = "distilbert-base-uncased",
        freeze_text: bool = True,
        motion_horizon: int = 6,
        mlm_ratio: float = 0.5,
        input_mode: str = "image",          # "image" | "feature"
        frame_feat_dim: Optional[int] = None,
        use_text: bool = True,
    ):
        super().__init__()
        self.input_mode = input_mode
        self.mlm_ratio = mlm_ratio
        self.max_views = max_views
        self.use_text = use_text

        if input_mode == "image":
            self.frame_encoder = FrameEncoder(
                frame_encoder_name, image_size, frozen=freeze_encoder,
                grad_checkpointing=grad_checkpointing)
            frame_feat_dim = self.frame_encoder.embed_dim
        else:
            self.frame_encoder = None
            assert frame_feat_dim is not None, "feature mode needs frame_feat_dim"
        self.frame_feat_dim = frame_feat_dim

        self.input_proj = nn.Linear(frame_feat_dim, dim)
        self.camera_embed = nn.Parameter(torch.zeros(1, 1, max_views, dim))
        nn.init.trunc_normal_(self.camera_embed, std=0.02)

        self.temporal = TemporalTransformer(
            dim=dim, depth=temporal_depth, heads=temporal_heads,
            max_frames=max_frames, grad_checkpointing=grad_checkpointing)

        self.video_proj = ProjectionHead(dim, proj_dim)
        self.motion_head = MotionHead(dim, horizon=motion_horizon)
        self.masked_predictor = MaskedFramePredictor(dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        if use_text:
            self.text_encoder = TextEncoder(text_encoder_name, frozen=freeze_text)
            self.text_proj = ProjectionHead(self.text_encoder.embed_dim, proj_dim)

    # ---- video encoding -------------------------------------------------
    def _frame_view_embeddings(self, batch: Dict) -> torch.Tensor:
        """Return per-(frame, view) embeddings ``[B, T, V, D_frame]``.

        Precomputed ``features`` take priority over ``images`` whenever present,
        so an image-mode model can also be driven by externally-encoded features
        (used by the robustness / efficiency evaluation)."""
        if "features" in batch:
            return batch["features"]
        imgs = batch["images"]                              # [B, T, V, 3, H, W]
        B, T, V, C, H, W = imgs.shape
        flat = imgs.view(B * T * V, C, H, W)
        feats = self.frame_encoder(flat)
        return feats.view(B, T, V, -1)

    def _aggregate_views(self, fv: torch.Tensor, cam_mask: torch.Tensor) -> torch.Tensor:
        """Project + camera-embed + masked-mean over views -> ``[B, T, D]``."""
        B, T, V, _ = fv.shape
        x = self.input_proj(fv) + self.camera_embed[:, :, :V]
        m = cam_mask.unsqueeze(-1).float()                 # [B, T, V, 1]
        return (x * m).sum(2) / m.sum(2).clamp(min=1e-6)

    def encode_video(self, batch: Dict, mlm_mask: Optional[torch.Tensor] = None):
        fv = self._frame_view_embeddings(batch)
        frame_tokens = self._aggregate_views(fv, batch["cam_mask"])
        cls, tokens = self.temporal(frame_tokens, batch.get("frame_mask"), mlm_mask)
        return cls, tokens, frame_tokens

    def encode_text(self, captions: List[str]) -> torch.Tensor:
        return self.text_proj(self.text_encoder(captions))

    # ---- full forward for training -------------------------------------
    def forward(self, batch: Dict, objectives=("contrastive", "mlm", "motion")) -> Dict:
        device = self.logit_scale.device
        frame_mask = batch.get("frame_mask")
        B = batch["cam_mask"].shape[0]
        T = batch["cam_mask"].shape[1]

        # Sample masked temporal positions (only among valid frames).
        mlm_mask = None
        if "mlm" in objectives and self.training and self.mlm_ratio > 0:
            valid = frame_mask if frame_mask is not None else torch.ones(B, T, dtype=torch.bool, device=device)
            rand = torch.rand(B, T, device=device).masked_fill(~valid, -1.0)
            k = max(1, int(round(self.mlm_ratio * T)))
            thresh = rand.topk(k, dim=1).values[:, -1:].clamp(min=0)
            mlm_mask = (rand >= thresh) & valid

        cls, tokens, frame_tokens = self.encode_video(batch, mlm_mask)

        out: Dict = {"clip_cls": cls}
        if "contrastive" in objectives and self.use_text:
            out["video_emb"] = self.video_proj(cls)
            out["text_emb"] = self.encode_text(batch["caption"])
            out["logit_scale"] = self.logit_scale.clamp(max=math.log(100.0)).exp()
        if "motion" in objectives:
            out["motion_pred"] = self.motion_head(cls)
            out["motion_anchor"] = self.motion_head.anchor
        if "mlm" in objectives and mlm_mask is not None:
            out["masked_pred"] = self.masked_predictor(tokens)
            out["masked_target"] = frame_tokens.detach()
            out["mlm_mask"] = mlm_mask
        return out


def build_model(cfg: Dict) -> MTRModel:
    m = cfg["model"]
    return MTRModel(
        dim=m.get("dim", 512),
        proj_dim=m.get("proj_dim", 256),
        frame_encoder_name=m.get("frame_encoder", "dinov2_vitb14"),
        image_size=m.get("image_size", 224),
        freeze_encoder=m.get("freeze_encoder", True),
        grad_checkpointing=m.get("grad_checkpointing", False),
        temporal_depth=m.get("temporal_depth", 4),
        temporal_heads=m.get("temporal_heads", 8),
        max_frames=m.get("max_frames", 16),
        max_views=m.get("max_views", 6),
        text_encoder_name=m.get("text_encoder", "distilbert-base-uncased"),
        freeze_text=m.get("freeze_text", True),
        motion_horizon=m.get("motion_horizon", 6),
        mlm_ratio=m.get("mlm_ratio", 0.5),
        input_mode=m.get("input_mode", "image"),
        frame_feat_dim=m.get("frame_feat_dim"),
        use_text=m.get("use_text", True),
    )
