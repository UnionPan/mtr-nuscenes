"""Full evaluation suite: retrieval, linear probe, motion, robustness, efficiency.

Robustness and efficiency need to run the (otherwise cached) frozen frame
encoder on possibly-corrupted images, so they encode images on the fly and feed
the resulting features into the trained temporal model.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import NuScenesClipDataset, collate_clips
from ..data.feature_cache import FeatureCache
from ..data.index import load_index
from ..models.frame_encoder import FrameEncoder
from ..utils import amp_dtype
from .core import (clip_labels, embed_clips, linear_probe, motion_metrics,
                   recall_at_k)


def _features_from_images(frame_encoder, images, dt, precision):
    B, T, V, C, H, W = images.shape
    flat = images.view(B * T * V, C, H, W)
    with torch.autocast("cuda", dtype=dt, enabled=(precision != "fp32")):
        feats = frame_encoder(flat)
    return feats.view(B, T, V, -1)


@torch.no_grad()
def embed_via_encoder(core, frame_encoder, loader, device, precision) -> Dict:
    """Like core.embed_clips but encodes images (possibly corrupted) on the fly."""
    core.eval()
    dt = amp_dtype(precision)
    V, Txt, CLS, MP, MT, MV = [], [], [], [], [], []
    for batch in loader:
        imgs = batch["images"].to(device, non_blocking=True)
        feats = _features_from_images(frame_encoder, imgs, dt, precision)
        b = {"features": feats,
             "cam_mask": batch["cam_mask"].to(device),
             "frame_mask": batch["frame_mask"].to(device),
             "caption": batch["caption"]}
        if "cv_anchor" in batch:
            b["cv_anchor"] = batch["cv_anchor"].to(device)
            b["kin"] = batch["kin"].to(device)
        with torch.autocast("cuda", dtype=dt, enabled=(precision != "fp32")):
            cls, tokens, _ = core.encode_video(b)
            mp = core.predict_motion(cls, tokens, b)
            if core.use_text:
                vemb, txt = core.video_proj(cls), core.encode_text(b["caption"])
        CLS.append(cls.float().cpu().numpy()); MP.append(mp.float().cpu().numpy())
        MT.append(batch["motion_target"].numpy()); MV.append(batch["motion_valid"].numpy())
        if core.use_text:
            V.append(vemb.float().cpu().numpy()); Txt.append(txt.float().cpu().numpy())
    out = {"clip_cls": np.concatenate(CLS), "motion_pred": np.concatenate(MP),
           "motion_target": np.concatenate(MT), "motion_valid": np.concatenate(MV)}
    if V:
        out["video_emb"] = np.concatenate(V); out["text_emb"] = np.concatenate(Txt)
    return out


def _summary(emb: Dict, ks) -> Dict:
    m = {}
    if "video_emb" in emb:
        m.update(recall_at_k(emb["video_emb"], emb["text_emb"], ks))
    m.update(motion_metrics(emb["motion_pred"], emb["motion_target"], emb["motion_valid"]))
    return m


def run_retrieval_probe_motion(core, cfg, device) -> Dict:
    """Clean-data retrieval, motion, and linear probes (cached feature mode)."""
    d = cfg["data"]
    fc = FeatureCache(d["feature_cache"]) if d.get("input_mode") == "feature" else None
    mode = d.get("input_mode", "feature")
    ks = cfg["eval"]["recall_ks"]
    bs = cfg["eval"]["batch_size"]

    def loader(split):
        ds = NuScenesClipDataset(d["index_path"], split=split, image_size=d.get("image_size", 224),
                                 mode=mode, feature_cache=fc, dataroot=d.get("dataroot"),
                                 anchor_model=d.get("anchor_model", "cv"))
        return DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_clips,
                          num_workers=d.get("num_workers", 4))

    emb_tr = embed_clips(core, loader("train"), device, cfg["train"]["precision"])
    emb_va = embed_clips(core, loader("val"), device, cfg["train"]["precision"])
    res = {"val": _summary(emb_va, ks)}

    index = load_index(d["index_path"])
    lab_tr, lab_va = clip_labels(index["train"]), clip_labels(index["val"])
    probes = {}
    for name in lab_tr:
        probes[name] = linear_probe(emb_tr["clip_cls"], lab_tr[name],
                                    emb_va["clip_cls"], lab_va[name])
    res["linear_probe"] = probes
    return res


def run_robustness(core, cfg, device) -> Dict:
    """Recall / motion under visual corruption and frame/camera dropout (image mode)."""
    d = cfg["data"]
    ks, bs = cfg["eval"]["recall_ks"], cfg["eval"]["batch_size"]
    prec = cfg["train"]["precision"]
    frame_encoder = FrameEncoder(cfg["model"]["frame_encoder"], cfg["model"].get("image_size", 224),
                                 frozen=True).to(device).eval()

    def img_loader(**kw):
        ds = NuScenesClipDataset(d["index_path"], split="val", image_size=d.get("image_size", 224),
                                 mode="image", dataroot=d.get("dataroot"),
                                 anchor_model=d.get("anchor_model", "cv"), **kw)
        return DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_clips,
                          num_workers=d.get("num_workers", 4))

    clean = _summary(embed_via_encoder(core, frame_encoder, img_loader(), device, prec), ks)
    out = {"clean": clean}
    rob = cfg["eval"]["robustness"]

    for kind in rob["corruptions"]:
        for sev in rob["severities"]:
            e = embed_via_encoder(core, frame_encoder,
                                  img_loader(corruption=kind, corruption_severity=sev),
                                  device, prec)
            out[f"corrupt::{kind}::s{sev}"] = _summary(e, ks)
    for fd in rob["frame_dropout"]:
        e = embed_via_encoder(core, frame_encoder, img_loader(frame_dropout=fd), device, prec)
        out[f"frame_dropout::{fd}"] = _summary(e, ks)
    for cd in rob["camera_dropout"]:
        e = embed_via_encoder(core, frame_encoder, img_loader(camera_dropout=cd), device, prec)
        out[f"camera_dropout::{cd}"] = _summary(e, ks)
    return out


def run_efficiency(core, cfg, device, n_batches: int = 8) -> Dict:
    """Throughput (clips/s), peak memory (GB), latency (ms/clip) for the full
    image->temporal pipeline and the cached feature-only temporal pipeline."""
    d = cfg["data"]
    bs = cfg["eval"]["batch_size"]
    prec = cfg["train"]["precision"]
    dt = amp_dtype(prec)
    T = cfg["data"]["clip_len"]
    V = cfg["model"].get("max_views", 6)
    S = cfg["model"].get("image_size", 224)
    Df = core.frame_feat_dim
    frame_encoder = FrameEncoder(cfg["model"]["frame_encoder"], S, frozen=True).to(device).eval()

    def bench(fn, warmup=2, iters=n_batches):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        dt_s = (time.time() - t0) / iters
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        return {"latency_ms_per_clip": 1000 * dt_s / bs, "throughput_clips_s": bs / dt_s,
                "peak_mem_gb": peak}

    cam_mask = torch.ones(bs, T, V, dtype=torch.bool, device=device)
    frame_mask = torch.ones(bs, T, dtype=torch.bool, device=device)
    caps = ["a driving scene"] * bs

    @torch.no_grad()
    def full_pipeline():
        imgs = torch.randn(bs, T, V, 3, S, S, device=device)
        feats = _features_from_images(frame_encoder, imgs, dt, prec)
        b = {"features": feats, "cam_mask": cam_mask, "frame_mask": frame_mask, "caption": caps}
        with torch.autocast("cuda", dtype=dt, enabled=(prec != "fp32")):
            cls, tokens, _ = core.encode_video(b)
            core.predict_motion(cls, tokens, b)

    @torch.no_grad()
    def feature_pipeline():
        feats = torch.randn(bs, T, V, Df, device=device)
        b = {"features": feats, "cam_mask": cam_mask, "frame_mask": frame_mask, "caption": caps}
        with torch.autocast("cuda", dtype=dt, enabled=(prec != "fp32")):
            cls, tokens, _ = core.encode_video(b)
            core.predict_motion(cls, tokens, b)

    return {"full_image_to_temporal": bench(full_pipeline),
            "cached_feature_temporal": bench(feature_pipeline),
            "config": {"batch_size": bs, "clip_len": T, "views": V, "image_size": S,
                       "precision": prec}}
