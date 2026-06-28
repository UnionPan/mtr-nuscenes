#!/usr/bin/env python3
"""Train entry point. Single-GPU:  python scripts/train.py --config configs/base.yaml
Multi-GPU:  torchrun --nproc_per_node=N scripts/train.py --config configs/base.yaml"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mtr.train.trainer import Trainer
from mtr.utils import load_config


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[],
                    help="dotted overrides, e.g. train.epochs=5 output_dir=runs/x")
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = _coerce(v)
    cfg = load_config(args.config, overrides)
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
