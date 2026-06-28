"""Training objectives: InfoNCE video-text alignment, masked temporal modeling,
and ego-motion prediction. ``compute_losses`` combines whichever are present in
the model output with configurable weights."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def info_nce(video_emb: torch.Tensor, text_emb: torch.Tensor,
             logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE. Embeddings are assumed L2-normalized."""
    logits = logit_scale * video_emb @ text_emb.t()        # [B, B]
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def masked_frame_loss(pred: torch.Tensor, target: torch.Tensor,
                      mlm_mask: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 between predicted and original frame embeddings at masked spots.
    Targets are per-token standardized for scale-invariance (data2vec-style)."""
    t = target
    t = (t - t.mean(-1, keepdim=True)) / (t.std(-1, keepdim=True) + 1e-6)
    p = pred
    loss = F.smooth_l1_loss(p, t, reduction="none").mean(-1)   # [B, T]
    m = mlm_mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def motion_loss(pred: torch.Tensor, target: torch.Tensor,
                valid: torch.Tensor) -> Dict[str, torch.Tensor]:
    """L2 waypoint loss + ADE/FDE metrics over valid clips. pred/target [B, H, 2]."""
    v = valid.float().view(-1, 1, 1)
    denom = valid.float().sum().clamp(min=1.0)
    dist = torch.linalg.norm(pred - target, dim=-1)            # [B, H]
    loss = (dist * v.squeeze(-1)).sum() / (denom * pred.shape[1])
    ade = (dist.mean(1) * valid.float()).sum() / denom
    fde = (dist[:, -1] * valid.float()).sum() / denom
    return {"motion": loss, "ade": ade.detach(), "fde": fde.detach()}


def compute_losses(out: Dict, batch: Dict, weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
    """Aggregate available objectives into a weighted total + per-term logs."""
    logs: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=next(iter(out.values())).device
                        if out else batch["motion_target"].device)

    if "video_emb" in out:
        l = info_nce(out["video_emb"], out["text_emb"], out["logit_scale"])
        logs["contrastive"] = l.detach()
        total = total + weights.get("contrastive", 1.0) * l
    if "masked_pred" in out:
        l = masked_frame_loss(out["masked_pred"], out["masked_target"], out["mlm_mask"])
        logs["mlm"] = l.detach()
        total = total + weights.get("mlm", 1.0) * l
    if "motion_pred" in out:
        md = motion_loss(out["motion_pred"], batch["motion_target"], batch["motion_valid"])
        logs["motion"] = md["motion"].detach()
        logs["ade"] = md["ade"]
        logs["fde"] = md["fde"]
        total = total + weights.get("motion", 1.0) * md["motion"]
        # Shrinkage of the residual toward the mean-trajectory anchor (combats
        # memorization in the small-scene regime; pulls predictions to the prior).
        if "motion_anchor" in out and weights.get("motion_reg", 0.0) > 0:
            resid = out["motion_pred"] - out["motion_anchor"].unsqueeze(0)
            reg = resid.pow(2).mean()
            logs["motion_reg"] = reg.detach()
            total = total + weights["motion_reg"] * reg

    logs["total"] = total.detach()
    return {"total": total, "logs": logs}
