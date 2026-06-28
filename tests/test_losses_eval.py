"""Loss and evaluation-metric correctness tests."""
import numpy as np
import torch
import torch.nn.functional as F

from mtr.eval.core import motion_metrics, recall_at_k
from mtr.models.losses import (compute_losses, info_nce, masked_frame_loss,
                               motion_loss)


def test_info_nce_perfect_alignment_low():
    e = F.normalize(torch.randn(8, 16), dim=-1)
    scale = torch.tensor(20.0)
    aligned = info_nce(e, e.clone(), scale)
    misaligned = info_nce(e, F.normalize(torch.randn(8, 16), dim=-1), scale)
    assert aligned < misaligned
    assert aligned < 0.1


def test_motion_loss_zero_when_exact():
    t = torch.randn(5, 6, 2)
    valid = torch.ones(5, dtype=torch.bool)
    md = motion_loss(t.clone(), t, valid)
    assert md["motion"].item() < 1e-6
    assert md["ade"].item() < 1e-6 and md["fde"].item() < 1e-6


def test_motion_loss_respects_validity():
    pred = torch.zeros(4, 6, 2)
    target = torch.ones(4, 6, 2)
    valid = torch.tensor([True, False, False, False])
    md = motion_loss(pred, target, valid)
    # only 1 valid clip contributes; ade = mean L2 of (1,1) vector = sqrt(2)
    assert abs(md["ade"].item() - np.sqrt(2)) < 1e-4


def test_masked_frame_loss_masking():
    pred = torch.randn(3, 6, 8)
    target = torch.randn(3, 6, 8)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    mask[:, 0] = True
    loss = masked_frame_loss(pred, target, mask)
    assert loss.item() >= 0 and torch.isfinite(loss)


def test_compute_losses_weighting():
    out = {
        "video_emb": F.normalize(torch.randn(4, 8), dim=-1),
        "text_emb": F.normalize(torch.randn(4, 8), dim=-1),
        "logit_scale": torch.tensor(14.0),
        "motion_pred": torch.zeros(4, 6, 2),
    }
    batch = {"motion_target": torch.ones(4, 6, 2),
             "motion_valid": torch.ones(4, dtype=torch.bool)}
    res = compute_losses(out, batch, {"contrastive": 1.0, "motion": 0.0})
    # motion weight 0 -> total equals contrastive term
    assert torch.allclose(res["total"], res["logs"]["contrastive"])


def test_recall_at_k_perfect():
    e = F.normalize(torch.randn(20, 32), dim=-1).numpy()
    r = recall_at_k(e, e, ks=(1, 5))
    assert r["v2t_R@1"] == 1.0 and r["t2v_R@1"] == 1.0
    assert r["v2t_medr"] == 1.0


def test_recall_at_k_monotonic():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((30, 16)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    t = rng.standard_normal((30, 16)); t /= np.linalg.norm(t, axis=1, keepdims=True)
    r = recall_at_k(v, t, ks=(1, 5, 10))
    assert r["v2t_R@1"] <= r["v2t_R@5"] <= r["v2t_R@10"]


def test_motion_metrics_no_valid():
    m = motion_metrics(np.zeros((3, 6, 2)), np.zeros((3, 6, 2)), np.zeros(3, dtype=bool))
    assert m["n"] == 0 and np.isnan(m["ade"])
