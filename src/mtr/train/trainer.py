"""Configurable single- and multi-GPU trainer.

Features: AMP (fp16/bf16) autocast with fp32 optimizer states, gradient
accumulation, optional gradient checkpointing (via model config), cosine LR with
warmup, TensorBoard + CSV logging, checkpoint save/resume, deterministic seeding,
periodic evaluation, and DDP/FSDP multi-GPU support.
"""
from __future__ import annotations

import csv
import os
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..data import NuScenesClipDataset, collate_clips
from ..data.feature_cache import FeatureCache
from ..eval.core import embed_clips, recall_at_k, motion_metrics
from ..models import build_model
from ..models.losses import compute_losses
from ..utils import (AverageMeter, amp_dtype, count_params, load_checkpoint,
                     save_checkpoint, save_json, set_seed)
from .distributed import (cleanup_distributed, is_distributed, setup_distributed,
                          wrap_model)
from .scheduler import cosine_warmup


def build_dataset(cfg: Dict, split: str, **overrides) -> NuScenesClipDataset:
    d = cfg["data"]
    mode = d.get("input_mode", "image")
    fc = None
    if mode == "feature":
        fc = FeatureCache(d["feature_cache"])
    if split == "train":
        aug = d.get("train_augment", {})
        overrides.setdefault("frame_dropout", aug.get("frame_dropout", 0.0))
        overrides.setdefault("camera_dropout", aug.get("camera_dropout", 0.0))
        overrides.setdefault("deterministic", False)
    return NuScenesClipDataset(
        d["index_path"], split=split, image_size=d.get("image_size", 224),
        mode=mode, feature_cache=fc, dataroot=d.get("dataroot"),
        anchor_model=d.get("anchor_model", "cv"),
        **overrides,
    )


class Trainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.is_main = self.rank == 0
        self.distributed = is_distributed()
        set_seed(cfg["seed"] + self.rank)

        self.device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        self.precision = cfg["train"]["precision"]
        self.amp_dt = amp_dtype(self.precision)
        self.out_dir = cfg["output_dir"]
        if self.is_main:
            os.makedirs(self.out_dir, exist_ok=True)

        # Data
        self.train_ds = build_dataset(cfg, "train")
        self.val_ds = build_dataset(cfg, "val")
        tcfg = cfg["train"]
        if self.distributed:
            self.train_sampler = DistributedSampler(self.train_ds, shuffle=True,
                                                    drop_last=True)
        else:
            self.train_sampler = None
        self.train_loader = DataLoader(
            self.train_ds, batch_size=tcfg["batch_size"],
            shuffle=(self.train_sampler is None), sampler=self.train_sampler,
            num_workers=cfg["data"].get("num_workers", 4), collate_fn=collate_clips,
            pin_memory=True, drop_last=self.distributed)
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
            num_workers=cfg["data"].get("num_workers", 4), collate_fn=collate_clips,
            pin_memory=True)

        # Model
        model = build_model(cfg).to(self.device)
        self.core = model
        # Initialize the motion head's residual anchor to the mean training
        # trajectory (so the head starts at the constant-mean baseline).
        valid = [c["motion_target"] for c in self.train_ds.clips if c["motion_valid"]]
        if valid and hasattr(self.core.motion_head, "set_anchor"):
            import numpy as _np
            mean_traj = torch.tensor(_np.mean(valid, axis=0), dtype=torch.float32)
            self.core.motion_head.set_anchor(mean_traj)
        if self.distributed:
            mode = tcfg.get("distributed", "ddp")
            model = wrap_model(model, self.local_rank, mode)
            self.core = model.module
        self.model = model

        # Optimizer / schedule / scaler. The visual backbone (when unfrozen) gets
        # a reduced LR so the pretrained features adapt gently.
        bb_mult = tcfg.get("backbone_lr_mult", 0.1)
        backbone_ids = set()
        if getattr(self.core, "frame_encoder", None) is not None:
            backbone_ids |= {id(p) for p in self.core.frame_encoder.parameters()}
        # A trainable (unfrozen) pretrained text encoder also gets the reduced LR.
        if getattr(self.core, "use_text", False) and getattr(self.core, "text_encoder", None) is not None:
            backbone_ids |= {id(p) for p in self.core.text_encoder.model.parameters()}
        head, backbone = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (backbone if id(p) in backbone_ids else head).append(p)
        groups = [{"params": head, "lr": tcfg["lr"]}]
        if backbone:
            groups.append({"params": backbone, "lr": tcfg["lr"] * bb_mult})
        self.optimizer = torch.optim.AdamW(groups, lr=tcfg["lr"],
                                           weight_decay=tcfg["weight_decay"])
        self.grad_accum = tcfg.get("grad_accum", 1)
        steps_per_epoch = max(1, len(self.train_loader) // self.grad_accum)
        self.total_steps = steps_per_epoch * tcfg["epochs"]
        self.scheduler = cosine_warmup(
            self.optimizer, self.total_steps,
            int(self.total_steps * tcfg.get("warmup_frac", 0.1)))
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.precision == "fp16"))

        self.objectives = tuple(cfg["objectives"])
        self.loss_weights = cfg["loss_weights"]
        self.feature_noise = cfg["data"].get("train_augment", {}).get("feature_noise", 0.0)
        self.step = 0
        self.epoch = 0
        self.best = float("-inf")   # so the first eval always writes best.pt
                                    # (selection key can be negative, e.g. -ADE)

        if self.is_main:
            total, trainable = count_params(self.core)
            print(f"[model] total {total/1e6:.1f}M trainable {trainable/1e6:.2f}M "
                  f"| world_size={self.world_size} precision={self.precision} "
                  f"steps/epoch={steps_per_epoch} total_steps={self.total_steps}")
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(self.out_dir, "tb"))
            self.csv_path = os.path.join(self.out_dir, "train_log.csv")
            self._csv_header = False

    # ---- checkpointing --------------------------------------------------
    def maybe_resume(self):
        path = os.path.join(self.out_dir, "last.pt")
        if os.path.exists(path):
            ckpt = load_checkpoint(path, self.core, self.optimizer, self.scaler,
                                   self.scheduler, map_location=self.device)
            self.step = ckpt.get("step", 0)
            self.epoch = ckpt.get("epoch", 0)
            self.best = ckpt.get("best") if ckpt.get("best") is not None else float("-inf")
            if self.is_main:
                print(f"[resume] from {path} at epoch {self.epoch} step {self.step}")

    def _save(self, name: str):
        if not self.is_main:
            return
        save_checkpoint(os.path.join(self.out_dir, name), self.core, self.optimizer,
                        self.scaler, self.scheduler, self.step, self.epoch,
                        self.cfg, self.best)

    def _log_csv(self, row: Dict):
        if not self.is_main:
            return
        write_header = not self._csv_header and not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        self._csv_header = True

    # ---- training -------------------------------------------------------
    def train_epoch(self):
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        meters = {}
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        n_batches = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            batch = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            if self.feature_noise > 0 and "features" in batch:
                f = batch["features"]
                batch["features"] = f + torch.randn_like(f) * (self.feature_noise * f.std())
            with torch.autocast(device_type="cuda", dtype=self.amp_dt,
                                enabled=(self.precision != "fp32")):
                out = self.model(batch, objectives=self.objectives)
                res = compute_losses(out, batch, self.loss_weights)
                loss = res["total"] / self.grad_accum
            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accum == 0:
                if self.cfg["train"].get("grad_clip"):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.cfg["train"]["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.step += 1

                for k, v in res["logs"].items():
                    meters.setdefault(k, AverageMeter()).update(float(v))
                if self.is_main and self.step % self.cfg["train"]["log_every"] == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    msg = " ".join(f"{k}={meters[k].avg:.3f}" for k in
                                   ["total", "contrastive", "mlm", "motion"] if k in meters)
                    print(f"[ep {self.epoch} {i+1}/{n_batches} step {self.step}] {msg} lr={lr:.2e}")
                    for k, m in meters.items():
                        self.tb.add_scalar(f"train/{k}", m.avg, self.step)
                    self.tb.add_scalar("train/lr", lr, self.step)
        if self.is_main:
            dt = time.time() - t0
            print(f"[ep {self.epoch}] done in {dt:.1f}s "
                  f"({len(self.train_ds)/max(dt,1e-6):.1f} clips/s)")

    @torch.no_grad()
    def evaluate(self) -> Dict:
        if not self.is_main:
            return {}
        emb = embed_clips(self.core, self.val_loader, self.device, self.precision)
        metrics = {}
        if "video_emb" in emb:
            metrics.update(recall_at_k(emb["video_emb"], emb["text_emb"],
                                       self.cfg["eval"]["recall_ks"]))
        metrics.update(motion_metrics(emb["motion_pred"], emb["motion_target"],
                                      emb["motion_valid"]))
        return metrics

    def fit(self):
        self.maybe_resume()
        tcfg = self.cfg["train"]
        start_epoch = self.epoch
        for ep in range(start_epoch, tcfg["epochs"]):
            self.epoch = ep
            self.train_epoch()
            self._save("last.pt")
            do_eval = ((ep + 1) % tcfg["eval_every_epochs"] == 0) or (ep + 1 == tcfg["epochs"])
            if do_eval and self.is_main:
                m = self.evaluate()
                key = m.get("mean_R@1", -m.get("ade", 1e9))
                print(f"[eval ep {ep}] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
                for k, v in m.items():
                    self.tb.add_scalar(f"val/{k}", v, self.step)
                self._log_csv({"epoch": ep, "step": self.step, **m})
                if key > self.best:
                    self.best = key
                    self._save("best.pt")
        if self.is_main:
            final = self.evaluate()
            save_json(os.path.join(self.out_dir, "final_metrics.json"),
                      {"config": self.cfg, "metrics": final, "best": self.best})
            print("[final]", final)
        cleanup_distributed()
        return self.best
