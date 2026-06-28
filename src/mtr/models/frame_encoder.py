"""Pretrained per-frame image encoder (DINOv2 / ViT) via ``timm``.

Frozen by default (the recommended MVP path on RTX 8000): the temporal model,
projection heads, and motion head are trained on top of cached or on-the-fly
frame embeddings.  When unfrozen, gradient checkpointing can be enabled to fit
the backbone in memory.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Friendly aliases -> timm model names.
BACKBONES = {
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "vitb16": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
}


class FrameEncoder(nn.Module):
    def __init__(self, name: str = "dinov2_vitb14", image_size: int = 224,
                 frozen: bool = True, grad_checkpointing: bool = False,
                 pretrained: bool = True):
        super().__init__()
        import timm
        timm_name = BACKBONES.get(name, name)
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, img_size=image_size,
        )
        self.embed_dim = self.backbone.num_features
        self.frozen = frozen
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        if frozen:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:                # keep BN/dropout-free backbone in eval
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: ``[N, 3, H, W]`` -> embeddings ``[N, D]`` (CLS / pooled token)."""
        if self.frozen:
            with torch.no_grad():
                return self.backbone(images)
        return self.backbone(images)
