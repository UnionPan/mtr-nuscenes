#!/usr/bin/env python3
"""Build a clip index JSON from the nuScenes devkit."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mtr.data.index import CAMERAS_6, CAMERAS_FRONT, build_clip_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--clip-len", type=int, default=6)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--future-horizon", type=int, default=6)
    ap.add_argument("--cameras", choices=["6", "1"], default="6")
    ap.add_argument("--require-images", action="store_true",
                    help="skip clips whose image files are not present on disk")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cams = CAMERAS_6 if args.cameras == "6" else CAMERAS_FRONT
    idx = build_clip_index(args.dataroot, args.version, args.clip_len, args.stride,
                           args.future_horizon, cams, out_path=args.out,
                           require_images=args.require_images)
    print(f"wrote {args.out}: {idx['meta']['n_train']} train / {idx['meta']['n_val']} val clips")


if __name__ == "__main__":
    main()
