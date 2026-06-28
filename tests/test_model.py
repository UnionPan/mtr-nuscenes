"""Model-interface tests (feature mode, synthetic inputs — no image backbone)."""
import torch

from mtr.models import MTRModel
from mtr.models.temporal import TemporalTransformer


def _batch(B=4, T=6, V=6, D=768):
    return {
        "features": torch.randn(B, T, V, D),
        "cam_mask": torch.ones(B, T, V, dtype=torch.bool),
        "frame_mask": torch.ones(B, T, dtype=torch.bool),
        "caption": [f"scene {i}" for i in range(B)],
        "motion_target": torch.randn(B, 6, 2),
        "motion_valid": torch.ones(B, dtype=torch.bool),
    }


def _model(**kw):
    return MTRModel(input_mode="feature", frame_feat_dim=768, dim=128,
                    temporal_depth=2, temporal_heads=4, max_views=6,
                    proj_dim=64, motion_horizon=6, use_text=True, **kw)


def test_forward_all_objectives_shapes():
    m = _model().train()
    out = m(_batch(), objectives=("contrastive", "mlm", "motion"))
    assert out["video_emb"].shape == (4, 64)
    assert out["text_emb"].shape == (4, 64)
    assert out["motion_pred"].shape == (4, 6, 2)
    assert out["masked_pred"].shape[1] == 6           # [B, T, D]
    assert out["mlm_mask"].shape == (4, 6)
    # contrastive embeddings are L2-normalized
    assert torch.allclose(out["video_emb"].norm(dim=-1), torch.ones(4), atol=1e-4)


def test_objective_subset():
    m = _model().train()
    out = m(_batch(), objectives=("motion",))
    assert "motion_pred" in out and "video_emb" not in out and "masked_pred" not in out


def test_encode_video_eval_no_mlm():
    m = _model().eval()
    cls, tokens, frame_tokens = m.encode_video(_batch())
    assert cls.shape == (4, 128) and tokens.shape == (4, 6, 128)


def test_mlm_mask_only_valid_frames():
    m = _model().train()
    b = _batch()
    b["frame_mask"][:, 3:] = False                    # last 3 frames invalid
    out = m(b, objectives=("mlm",))
    # masked positions must be a subset of valid frames
    assert not (out["mlm_mask"] & ~b["frame_mask"]).any()


def test_temporal_padding_invariance():
    """Padded (invalid) frames should not change the CLS output."""
    tt = TemporalTransformer(dim=32, depth=2, heads=4, max_frames=8).eval()
    x = torch.randn(2, 6, 32)
    fm = torch.ones(2, 6, dtype=torch.bool); fm[:, 4:] = False
    cls_a, _ = tt(x, fm)
    x2 = x.clone(); x2[:, 4:] = torch.randn(2, 2, 32)  # change padded frames
    cls_b, _ = tt(x2, fm)
    assert torch.allclose(cls_a, cls_b, atol=1e-5)


def test_camera_dropout_aggregation():
    """Masked-mean view aggregation ignores dropped cameras."""
    m = _model().eval()
    b = _batch()
    fv = b["features"]
    full = m._aggregate_views(fv, torch.ones(4, 6, 6, dtype=torch.bool))
    cm = torch.ones(4, 6, 6, dtype=torch.bool); cm[:, :, 3:] = False
    half = m._aggregate_views(fv, cm)
    # equals mean over first 3 views only
    ref = m._aggregate_views(fv[:, :, :3], torch.ones(4, 6, 3, dtype=torch.bool))
    assert torch.allclose(half, ref, atol=1e-5)
    assert not torch.allclose(half, full)


def test_trainable_params_frozen_encoder():
    m = _model()
    # feature mode: no image backbone; text frozen by default
    assert m.frame_encoder is None
    text_grad = any(p.requires_grad for p in m.text_encoder.model.parameters())
    assert not text_grad
