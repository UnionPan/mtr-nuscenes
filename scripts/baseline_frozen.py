#!/usr/bin/env python3
"""Frozen-encoder, no-pretraining baseline.

Mean-pools cached frozen DINOv2 embeddings over (frames x cameras) to a single
clip vector, then measures:
  * linear-probe accuracy (motion-state, pedestrian presence),
  * motion ADE/FDE from a Ridge regressor fit on train clips.
This is the reference the pretrained temporal model must beat."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mtr.data.feature_cache import FeatureCache
from mtr.data.index import load_index
from mtr.eval.core import clip_labels, linear_probe, motion_metrics
from mtr.utils import save_json


def pooled(clips, fc):
    X = []
    for c in clips:
        vecs = [fc.get(p) for frame in c["image_paths"] for p in frame]
        X.append(np.mean(vecs, axis=0))
    return np.stack(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index/mini_T6_s2.json")
    ap.add_argument("--cache", default="data/cache/dinov2_vitb14")
    ap.add_argument("--out", default="runs/baseline_frozen/eval_metrics.json")
    args = ap.parse_args()

    idx = load_index(args.index)
    fc = FeatureCache(args.cache)
    Xtr, Xva = pooled(idx["train"], fc), pooled(idx["val"], fc)
    ltr, lva = clip_labels(idx["train"]), clip_labels(idx["val"])

    probes = {k: linear_probe(Xtr, ltr[k], Xva, lva[k]) for k in ltr}

    # Ridge motion regressor on pooled frozen features.
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    H = len(idx["train"][0]["motion_target"])
    ytr = np.array([np.array(c["motion_target"]).reshape(-1) for c in idx["train"]])
    mtr = np.array([c["motion_valid"] for c in idx["train"]], dtype=bool)
    sc = StandardScaler().fit(Xtr)
    reg = Ridge(alpha=10.0).fit(sc.transform(Xtr)[mtr], ytr[mtr])
    pred = reg.predict(sc.transform(Xva)).reshape(-1, H, 2)
    tgt = np.array([np.array(c["motion_target"]) for c in idx["val"]])
    val = np.array([c["motion_valid"] for c in idx["val"]], dtype=bool)
    motion = motion_metrics(pred, tgt, val)

    res = {"name": "baseline_frozen_meanpool",
           "linear_probe": probes, "motion": motion,
           "note": "no temporal training; retrieval N/A (no video-text alignment)"}
    save_json(args.out, res)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
