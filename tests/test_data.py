"""Data-interface tests: index geometry, dataset shapes, masks, collate, cache."""
import os

import numpy as np
import pytest
import torch

from conftest import INDEX
from mtr.data import NuScenesClipDataset, collate_clips, load_index
from mtr.data.feature_cache import FeatureCache
from mtr.data.index import _motion_state, _quat_to_yaw

pytestmark = pytest.mark.skipif(not os.path.exists(INDEX),
                                reason="clip index not built")


def test_index_structure():
    idx = load_index(INDEX)
    assert idx["meta"]["clip_len"] == 6
    assert len(idx["train"]) > 0 and len(idx["val"]) > 0
    c = idx["train"][0]
    assert len(c["image_paths"]) == 6                 # T frames
    assert len(c["image_paths"][0]) == len(c["cameras"])
    assert len(c["motion_target"]) == idx["meta"]["future_horizon"]
    assert all(len(w) == 2 for w in c["motion_target"])
    assert isinstance(c["caption"], str) and len(c["caption"]) > 10


def test_quat_yaw_identity():
    assert abs(_quat_to_yaw([1, 0, 0, 0])) < 1e-6     # identity -> 0 heading


def test_motion_state_labels():
    assert _motion_state(0.1, 0.0) == "stopped"
    assert _motion_state(5.0, 0.0) == "driving straight"
    assert _motion_state(5.0, 0.5) == "turning left"
    assert _motion_state(5.0, -0.5) == "turning right"


def test_forward_motion_sign():
    """A moving-straight clip should have monotonically increasing forward (x)
    waypoints and small lateral (y)."""
    idx = load_index(INDEX)
    moving = [c for c in idx["train"] if c["speed"] > 3 and abs(c["yaw_rate"]) < 0.05
              and c["motion_valid"]]
    assert moving, "expected some straight-driving clips"
    wp = np.array(moving[0]["motion_target"])
    assert wp[-1, 0] > wp[0, 0] > 0                    # forward distance grows
    assert np.abs(wp[:, 1]).max() < np.abs(wp[:, 0]).max()


def test_dataset_image_mode_shapes():
    ds = NuScenesClipDataset(INDEX, split="val", mode="image", image_size=224)
    b = ds[0]
    T, V = 6, len(load_index(INDEX)["meta"]["cameras"])
    assert b["images"].shape == (T, V, 3, 224, 224)
    assert b["frame_mask"].shape == (T,) and b["cam_mask"].shape == (T, V)
    assert b["motion_target"].shape == (6, 2)
    assert b["frame_mask"].all() and b["cam_mask"].all()


def test_dropout_masks():
    ds = NuScenesClipDataset(INDEX, split="val", mode="image",
                             frame_dropout=0.99, camera_dropout=0.99, seed=0)
    b = ds[0]
    assert b["frame_mask"][-1].item()                 # anchor never dropped
    assert not b["frame_mask"][:-1].all()             # some frames dropped


def test_collate_batches():
    ds = NuScenesClipDataset(INDEX, split="val", mode="image")
    batch = collate_clips([ds[0], ds[1]])
    assert batch["images"].shape[0] == 2
    assert len(batch["caption"]) == 2
    assert batch["motion_target"].shape == (2, 6, 2)


def test_feature_cache_roundtrip(tmp_path):
    keys = ["a/x.jpg", "b/y.jpg"]
    feats = np.random.randn(2, 16).astype(np.float32)
    FeatureCache.save(str(tmp_path), keys, feats)
    fc = FeatureCache(str(tmp_path))
    assert fc.dim == 16
    assert np.allclose(fc.get("a/x.jpg"), feats[0], atol=1e-3)
    assert "b/y.jpg" in fc
