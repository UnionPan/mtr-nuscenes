#!/usr/bin/env python3
"""Evaluate a trained checkpoint: retrieval, linear probe, motion, robustness,
efficiency. Writes <output_dir>/eval_metrics.json."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mtr.eval.full import (run_efficiency, run_retrieval_probe_motion,
                           run_robustness)
from mtr.models import build_model
from mtr.utils import load_checkpoint, load_config, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="defaults to <output_dir>/best.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip", nargs="*", default=[], help="any of: robustness efficiency")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = args.ckpt or os.path.join(cfg["output_dir"], "best.pt")
    if not os.path.exists(ckpt):                       # fall back to last.pt
        ckpt = os.path.join(cfg["output_dir"], "last.pt")
    model = build_model(cfg).to(args.device)
    if os.path.exists(ckpt):
        load_checkpoint(ckpt, model, map_location=args.device)
        print(f"[loaded] {ckpt}")
    else:
        print(f"[warn] no checkpoint at {ckpt}; evaluating randomly-initialized model")
    model.eval()

    results = {"checkpoint": ckpt, "config_name": cfg.get("name")}
    print("== retrieval / probe / motion ==")
    results["main"] = run_retrieval_probe_motion(model, cfg, args.device)
    print(results["main"]["val"])
    print("linear_probe:", results["main"]["linear_probe"])
    if "robustness" not in args.skip:
        print("== robustness ==")
        results["robustness"] = run_robustness(model, cfg, args.device)
        for k, v in results["robustness"].items():
            print(f"  {k}: mean_R@1={v.get('mean_R@1', float('nan')):.3f} ade={v['ade']:.2f}")
    if "efficiency" not in args.skip:
        print("== efficiency ==")
        results["efficiency"] = run_efficiency(model, cfg, args.device)
        for k in ("full_image_to_temporal", "cached_feature_temporal"):
            e = results["efficiency"][k]
            print(f"  {k}: {e['throughput_clips_s']:.0f} clips/s, "
                  f"{e['latency_ms_per_clip']:.1f} ms/clip, {e['peak_mem_gb']:.2f} GB")

    out = args.out or os.path.join(cfg["output_dir"], "eval_metrics.json")
    save_json(out, results)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
