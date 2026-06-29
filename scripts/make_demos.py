#!/usr/bin/env python3
"""Generate visual demos from a trained MTR checkpoint (feature mode).

Produces, in reports/figures/:
  * demo_forecast_bev.png  -- bird's-eye trajectory forecasting on varied clips
                              (observed / ground-truth / CV / CTRV / model),
  * demo_surround.png      -- the 6-camera surround input + caption + BEV forecast,
  * demo_retrieval.png     -- front-camera image + top-k retrieved captions.

Usage:
  python scripts/make_demos.py --config configs/ablations/scale_6cam_ctrv.yaml \
         --ckpt runs/scale_6cam_ctrv/best.pt
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mtr.data.dataset import NuScenesClipDataset, _causal_kinematics, collate_clips
from mtr.data.feature_cache import FeatureCache
from mtr.data.index import load_index
from mtr.models.model import build_model
from mtr.utils import amp_dtype, load_config
from torch.utils.data import DataLoader

FIGDIR = "reports/figures"


def anchor_frame_obs(ego_poses):
    """Observed ego positions in the anchor (last-frame) ego frame."""
    P = np.asarray(ego_poses, float)
    ayaw = P[-1, 2]; ca, sa = np.cos(ayaw), np.sin(ayaw)
    R = np.array([[ca, sa], [-sa, ca]])
    return (P[:, :2] - P[-1, :2]) @ R.T            # [T,2], x fwd / y left


@torch.no_grad()
def run_model(cfg, ckpt, device):
    d = cfg["data"]
    fc = FeatureCache(d["feature_cache"])
    ds = NuScenesClipDataset(d["index_path"], split="val", mode="feature",
                             feature_cache=fc, dataroot=d.get("dataroot"),
                             anchor_model=d.get("anchor_model", "cv"))
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_clips,
                        num_workers=4)
    model = build_model(cfg).to(device).eval()
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd.get("model", sd), strict=False)
        print(f"[loaded] {ckpt}")
    core = model
    dt = amp_dtype(cfg["train"]["precision"])
    MP, VE, TE = [], [], []
    for b in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=dt, enabled=cfg["train"]["precision"] != "fp32"):
            cls, tokens, _ = core.encode_video(b)
            MP.append(core.predict_motion(cls, tokens, b).float().cpu().numpy())
            if core.use_text:
                VE.append(core.video_proj(cls).float().cpu().numpy())
                TE.append(core.encode_text(b["caption"]).float().cpu().numpy())
    out = {"motion_pred": np.concatenate(MP)}
    if VE:
        out["video_emb"] = np.concatenate(VE); out["text_emb"] = np.concatenate(TE)
    return ds.clips, out


def pick_varied(clips, n=6):
    """Pick clips spanning stop / straight / gentle & sharp turns."""
    idx_turn, idx_str, idx_stop = [], [], []
    for i, c in enumerate(clips):
        if not c.get("motion_valid"):
            continue
        t = np.asarray(c["motion_target"])
        total = np.linalg.norm(t[-1])
        lateral = abs(t[-1, 1])
        if total < 2.0:
            idx_stop.append(i)
        elif lateral > 3.0:
            idx_turn.append((lateral, i))
        else:
            idx_str.append((total, i))
    idx_turn.sort(reverse=True); idx_str.sort(reverse=True)
    sel = [i for _, i in idx_turn[:3]] + idx_stop[:1] + [i for _, i in idx_str[:2]]
    return sel[:n]


def bev(ax, clip, pred, show_legend=False):
    obs = anchor_frame_obs(clip["ego_poses"])
    gt = np.vstack([[0, 0], np.asarray(clip["motion_target"])])
    cv, _ = _causal_kinematics(clip["ego_poses"], len(clip["motion_target"]), "cv")
    ctrv, _ = _causal_kinematics(clip["ego_poses"], len(clip["motion_target"]), "ctrv")
    cv = np.vstack([[0, 0], cv]); ctrv = np.vstack([[0, 0], ctrv])
    pr = np.vstack([[0, 0], pred])

    def xy(a):                      # forward up, left left
        return -a[:, 1], a[:, 0]
    curves = [obs, cv, ctrv, gt, pr]
    ax.plot(*xy(obs), "-", color="0.6", lw=2, label="observed")
    ax.plot(*xy(cv), "--", color="tab:gray", lw=1.5, label="CV (straight)")
    ax.plot(*xy(ctrv), ":", color="tab:orange", lw=2.5, label="CTRV (turn)")
    ax.plot(*xy(gt), "-", color="tab:green", lw=2.5, marker="o", ms=3, label="ground truth")
    ax.plot(*xy(pr), "-", color="tab:blue", lw=2, marker="x", ms=4, label="model")
    ax.plot(0, 0, "k^", ms=11)                       # ego now
    # Limits: symmetric lateral with a floor so straight clips aren't slivers.
    allx = np.concatenate([-c[:, 1] for c in curves]); ally = np.concatenate([c[:, 0] for c in curves])
    L = max(6.0, 1.2 * np.abs(allx).max())
    ax.set_xlim(-L, L); ax.set_ylim(ally.min() - 3, ally.max() + 3)
    ade = np.linalg.norm(np.asarray(clip["motion_target"]) - pred, axis=-1).mean()
    state = clip["caption"].split("ego vehicle is ")[-1].split(" at ")[0]
    ax.set_title(f"{state} | model ADE {ade:.2f} m", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("left (m)"); ax.set_ylabel("forward (m)")
    if show_legend:
        ax.legend(fontsize=7, loc="lower left", framealpha=0.9)


def fig_forecast(clips, mp, sel):
    n = len(sel); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows))
    for k, i in enumerate(sel):
        bev(axes.flat[k], clips[i], mp[i], show_legend=(k == 0))
    for k in range(n, rows * cols):
        axes.flat[k].axis("off")
    fig.suptitle("Ego-motion forecasting (bird's-eye, anchor frame) — model vs ground truth vs physics priors",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(FIGDIR, "demo_forecast_bev.png"); fig.savefig(p, dpi=120); plt.close(fig)
    print("[saved]", p)


def load_img(dataroot, rel, size=320):
    im = Image.open(os.path.join(dataroot, rel)).convert("RGB")
    return im.resize((size, int(size * im.height / im.width)))


def fig_surround(clips, mp, sel, dataroot):
    i = sel[0]; clip = clips[i]
    order = clip["cameras"]
    grid = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    fig = plt.figure(figsize=(14, 7))
    for j, cam in enumerate(grid):
        ax = fig.add_subplot(2, 4, j + 1 if j < 3 else j + 2)
        if cam in order:
            v = order.index(cam)
            ax.imshow(load_img(dataroot, clip["image_paths"][-1][v]))
        ax.set_title(cam, fontsize=8); ax.axis("off")
    axb = fig.add_subplot(1, 4, 4)
    bev(axb, clip, mp[i], show_legend=True)
    fig.suptitle("Surround-camera input (6 views, anchor frame) + forecast\n\"" +
                 clip["caption"][:110] + "...\"", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIGDIR, "demo_surround.png"); fig.savefig(p, dpi=120); plt.close(fig)
    print("[saved]", p)


def fig_retrieval(clips, out, dataroot, n=4):
    ve, te = out["video_emb"], out["text_emb"]
    ve = ve / (np.linalg.norm(ve, axis=1, keepdims=True) + 1e-9)
    te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-9)
    sims = ve @ te.T
    rng = np.random.default_rng(0)
    sel = rng.choice(len(clips), size=n, replace=False)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n))
    for k, i in enumerate(sel):
        ax = axes[k] if n > 1 else axes
        v = clips[i]["cameras"].index("CAM_FRONT") if "CAM_FRONT" in clips[i]["cameras"] else 0
        ax.imshow(load_img(dataroot, clips[i]["image_paths"][-1][v], size=300))
        ax.axis("off")
        top = np.argsort(-sims[i])[:3]
        txt = "Top-3 retrieved captions:\n"
        for r, j in enumerate(top):
            mark = "✓" if j == i else " "
            txt += f"{mark} {r+1}. {clips[j]['caption'][:95]}\n"
        rank = int(np.where(np.argsort(-sims[i]) == i)[0][0]) + 1
        ax.text(1.02, 0.5, txt + f"\n(true caption rank: {rank}/{len(clips)})",
                transform=ax.transAxes, va="center", ha="left", fontsize=8, family="monospace")
    fig.suptitle("Video → text retrieval: front camera and the captions the model ranks highest", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(FIGDIR, "demo_retrieval.png"); fig.savefig(p, dpi=110); plt.close(fig)
    print("[saved]", p)


def _scene_front_sequences(clips):
    """Reconstruct each scene's ordered CAM_FRONT keyframe paths from the
    (overlapping, scene-tiling) clips, so future frames can be looked up."""
    from collections import defaultdict
    sf = defaultdict(dict)
    for c in clips:
        v = c["cameras"].index("CAM_FRONT") if "CAM_FRONT" in c["cameras"] else 0
        for t, ts in enumerate(c["timestamps"]):
            sf[c["scene"]][round(ts, 3)] = c["image_paths"][t][v]
    return {s: sorted(d.items()) for s, d in sf.items()}


def fig_animation(clips, mp, sel, dataroot):
    """A short GIF telling the story: for a turn / straight / stop clip, play the
    front-camera observed frames, reveal the forecast (model vs CV) in BEV, then
    play the REAL future frames so the prediction visibly comes true."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    # one turn, one straight, one stop
    pick = [sel[0]]
    for i in sel[1:]:
        if "straight" in clips[i]["caption"] and len(pick) < 2: pick.append(i)
    for i in sel[1:]:
        if np.linalg.norm(np.asarray(clips[i]["motion_target"])[-1]) < 2.0: pick.append(i); break
    pick = pick[:3]

    seqs = _scene_front_sequences(clips)
    fronts, futures = {}, {}
    for ci in pick:
        c = clips[ci]; v = c["cameras"].index("CAM_FRONT") if "CAM_FRONT" in c["cameras"] else 0
        fronts[ci] = v
        seq = seqs[c["scene"]]; times = [t for t, _ in seq]
        ats = round(c["timestamps"][-1], 3)
        ai = times.index(ats) if ats in times else len(seq) - 1
        H = len(c["motion_target"])
        futures[ci] = [p for _, p in seq[ai + 1:ai + 1 + H]]   # real future front frames

    plan = []                                   # (clip_idx, kind, k)
    for ci in pick:
        T = len(clips[ci]["image_paths"]); H = len(clips[ci]["motion_target"])
        for t in range(T):    plan.append((ci, "observe", t))
        for h in range(1, H + 1): plan.append((ci, "predict", h))
        plan += [(ci, "predict", H)] * 3        # hold

    fig, (axi, axb) = plt.subplots(1, 2, figsize=(11, 4.4))

    def draw(idx):
        ci, kind, k = plan[idx]; clip = clips[ci]
        axi.clear(); axb.clear()
        v = fronts[ci]
        if kind == "observe":
            path = clip["image_paths"][k][v]
            phase = f"observed frame {k+1}/{len(clip['image_paths'])}"
        else:
            fut = futures[ci]
            path = fut[k - 1] if k - 1 < len(fut) else clip["image_paths"][-1][v]
            phase = f"FORECAST — actual future frame +{k} ({k*0.5:.1f}s)"
        axi.imshow(load_img(dataroot, path, size=420)); axi.axis("off")
        axi.set_title(f"front camera — {phase}", fontsize=10)

        obs = anchor_frame_obs(clip["ego_poses"])
        gt = np.vstack([[0, 0], np.asarray(clip["motion_target"])])
        cv, _ = _causal_kinematics(clip["ego_poses"], len(clip["motion_target"]), "cv"); cv = np.vstack([[0, 0], cv])
        pr = np.vstack([[0, 0], mp[ci]])
        xy = lambda a: (-a[:, 1], a[:, 0])
        axb.plot(*xy(obs), "-", color="0.6", lw=2, label="observed")
        if kind == "predict":
            axb.plot(*xy(cv[:k + 1]), "--", color="tab:gray", lw=1.5, label="CV (straight)")
            axb.plot(*xy(gt[:k + 1]), "-", color="tab:green", lw=2.5, marker="o", ms=3, label="ground truth")
            axb.plot(*xy(pr[:k + 1]), "-", color="tab:blue", lw=2, marker="x", ms=4, label="model")
            axb.legend(fontsize=7, loc="lower left", framealpha=0.9)
        axb.plot(0, 0, "k^", ms=11)
        allx = np.concatenate([-c[:, 1] for c in (obs, cv, gt, pr)]); ally = np.concatenate([c[:, 0] for c in (obs, cv, gt, pr)])
        L = max(6.0, 1.2 * np.abs(allx).max())
        axb.set_xlim(-L, L); axb.set_ylim(ally.min() - 3, ally.max() + 3)
        st = clip["caption"].split("ego vehicle is ")[-1].split(" at ")[0]
        axb.set_title(f"bird's-eye — {st}", fontsize=10)
        axb.set_aspect("equal"); axb.grid(alpha=0.3); axb.set_xlabel("left (m)"); axb.set_ylabel("forward (m)")
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=len(plan), interval=450)
    p = os.path.join(FIGDIR, "demo_forecast.gif")
    anim.save(p, writer=PillowWriter(fps=2.5)); plt.close(fig)
    print("[saved]", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    cfg = load_config(args.config)
    clips, out = run_model(cfg, args.ckpt, args.device)
    mp = out["motion_pred"]
    sel = pick_varied(clips, 6)
    dataroot = cfg["data"].get("dataroot") or load_index(cfg["data"]["index_path"])["meta"]["dataroot"]
    fig_forecast(clips, mp, sel)
    fig_surround(clips, mp, sel, dataroot)
    fig_animation(clips, mp, sel, dataroot)
    if "video_emb" in out:
        fig_retrieval(clips, out, dataroot)


if __name__ == "__main__":
    main()
