"""Core evaluation primitives shared by periodic (in-training) and full eval.

Provides: clip embedding extraction, video<->text Recall@K, motion ADE/FDE, and
linear-probe accuracy on frozen clip features.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..data.index import _motion_state


@torch.no_grad()
def embed_clips(model, loader, device, precision: str = "fp16") -> Dict[str, np.ndarray]:
    """Run the model over a loader; collect projected embeddings, pooled clip
    features, and motion predictions/targets (no objective sampling / no MLM)."""
    from ..utils import amp_dtype
    core = model.module if hasattr(model, "module") else model
    core.eval()
    dt = amp_dtype(precision)
    V, Txt, CLS, MP, MT, MV = [], [], [], [], [], []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=dt, enabled=(precision != "fp32")):
            cls, tokens, _ = core.encode_video(b)
            if core.use_text:
                txt = core.encode_text(b["caption"])
                vemb = core.video_proj(cls)
            mp = core.predict_motion(cls, tokens, b)
        CLS.append(cls.float().cpu().numpy())
        MP.append(mp.float().cpu().numpy())
        MT.append(b["motion_target"].cpu().numpy())
        MV.append(b["motion_valid"].cpu().numpy())
        if core.use_text:
            V.append(vemb.float().cpu().numpy())
            Txt.append(txt.float().cpu().numpy())
    out = {
        "clip_cls": np.concatenate(CLS),
        "motion_pred": np.concatenate(MP),
        "motion_target": np.concatenate(MT),
        "motion_valid": np.concatenate(MV),
    }
    if V:
        out["video_emb"] = np.concatenate(V)
        out["text_emb"] = np.concatenate(Txt)
    return out


def recall_at_k(video_emb: np.ndarray, text_emb: np.ndarray, ks=(1, 5, 10)) -> Dict:
    """Symmetric video<->text retrieval Recall@K and median rank.
    Embeddings are assumed L2-normalized; diagonal is the positive pair."""
    sim = video_emb @ text_emb.T                      # [N, N]
    n = sim.shape[0]
    res = {}
    for name, S in [("v2t", sim), ("t2v", sim.T)]:
        ranks = np.empty(n, dtype=np.int64)
        for i in range(n):
            order = np.argsort(-S[i])
            ranks[i] = int(np.where(order == i)[0][0])
        for k in ks:
            res[f"{name}_R@{k}"] = float((ranks < k).mean())
        res[f"{name}_medr"] = float(np.median(ranks) + 1)
    res["mean_R@1"] = 0.5 * (res["v2t_R@1"] + res["t2v_R@1"])
    return res


def motion_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> Dict:
    """ADE/FDE in metres over valid clips."""
    v = valid.astype(bool)
    if v.sum() == 0:
        return {"ade": float("nan"), "fde": float("nan"), "n": 0}
    d = np.linalg.norm(pred[v] - target[v], axis=-1)   # [n, H]
    return {"ade": float(d.mean()), "fde": float(d[:, -1].mean()), "n": int(v.sum())}


def clip_labels(clips: List[Dict]) -> Dict[str, np.ndarray]:
    """Derive linear-probe labels from clip metadata."""
    states = ["stopped", "driving straight", "turning left", "turning right"]
    s2i = {s: i for i, s in enumerate(states)}
    motion = np.array([s2i[_motion_state(c["speed"], c["yaw_rate"])] for c in clips])
    ped = np.array([int("pedestrian" in c.get("objects", [])) for c in clips])
    return {"motion_state": motion, "has_pedestrian": ped}


def linear_probe(feat_tr: np.ndarray, y_tr: np.ndarray,
                 feat_va: np.ndarray, y_va: np.ndarray) -> Dict:
    """Fit a logistic-regression linear probe; report val accuracy + majority baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y_tr)) < 2:
        return {"acc": float("nan"), "majority": float("nan"), "n_classes": int(len(np.unique(y_tr)))}
    sc = StandardScaler().fit(feat_tr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(feat_tr), y_tr)
    acc = float((clf.predict(sc.transform(feat_va)) == y_va).mean())
    vals, cnts = np.unique(y_tr, return_counts=True)
    majority = float((y_va == vals[cnts.argmax()]).mean())
    return {"acc": acc, "majority": majority, "n_classes": int(len(vals))}
