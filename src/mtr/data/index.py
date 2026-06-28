"""Build a serializable clip index from the nuScenes devkit.

A *clip* is a window of ``T`` consecutive keyframe samples within a single
scene.  For each clip we record:
  * per-frame, per-camera image file paths (synchronized keyframes),
  * the ego pose (global x, y, yaw) at every frame,
  * a future ego-waypoint motion target expressed in the last frame's ego frame,
  * a structured natural-language caption derived from scene metadata.

The index is plain JSON so it is dataset-agnostic, cheap to cache, and lets the
training pipeline avoid re-parsing the devkit on every run.  The same builder
works on ``v1.0-mini`` and the full ``v1.0-trainval`` (clean path to full data).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

CAMERAS_6 = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]
CAMERAS_FRONT = ["CAM_FRONT"]

# Official nuScenes mini scene split (Motional devkit `nuscenes.utils.splits`).
# Mirrored here so the index can be built without importing the splits module.
MINI_TRAIN = [
    "scene-0061", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-1077", "scene-1094", "scene-1100",
]
MINI_VAL = ["scene-0103", "scene-0916"]


def _quat_to_yaw(q: List[float]) -> float:
    """nuScenes stores rotation as [w, x, y, z]; return heading (yaw) in radians."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _readable_category(name: str) -> str:
    """Map nuScenes category strings (e.g. ``vehicle.car``) to readable nouns."""
    leaf = name.split(".")[-1]
    return {"adult": "pedestrian", "child": "pedestrian", "construction": "construction vehicle",
            "trafficcone": "traffic cone", "barrier": "barrier"}.get(leaf, leaf)


def _motion_state(speed: float, yaw_rate: float) -> str:
    if speed < 0.5:
        return "stopped"
    if abs(yaw_rate) > 0.15:
        return "turning left" if yaw_rate > 0 else "turning right"
    return "driving straight"


def _make_caption(location: str, description: str, speed: float,
                  yaw_rate: float, objects: List[str]) -> str:
    state = _motion_state(speed, yaw_rate)
    loc = location.replace("-", " ")
    obj_str = ", ".join(objects) if objects else "no salient objects"
    desc = description.strip().rstrip(".") if description else "urban driving"
    return (f"In {loc}, the ego vehicle is {state} at {speed:.1f} meters per second. "
            f"Scene: {desc}. Nearby objects: {obj_str}.")


def build_clip_index(
    dataroot: str,
    version: str = "v1.0-mini",
    clip_len: int = 6,
    stride: int = 3,
    future_horizon: int = 6,
    cameras: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    require_images: bool = False,
) -> Dict:
    """Build the clip index. Returns a dict with ``train``/``val`` clip lists.

    Args:
        dataroot: path containing the nuScenes ``samples/`` and json folders.
        version: devkit version string.
        clip_len: number of keyframes T per clip.
        stride: hop between successive clip start frames within a scene.
        future_horizon: number of future keyframes H used for the motion target.
        cameras: list of camera channels (defaults to the 6 surround cameras).
        out_path: if given, write the index as JSON here.
    """
    from nuscenes.nuscenes import NuScenes

    cameras = cameras or CAMERAS_6
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    # Resolve the train/val scene split.
    if version == "v1.0-mini":
        split_train, split_val = set(MINI_TRAIN), set(MINI_VAL)
    else:
        from nuscenes.utils.splits import create_splits_scenes
        sc = create_splits_scenes()
        split_train, split_val = set(sc["train"]), set(sc["val"])

    clips = {"train": [], "val": []}

    for scene in nusc.scene:
        scene_name = scene["name"]
        split = "train" if scene_name in split_train else ("val" if scene_name in split_val else None)
        if split is None:
            continue
        location = nusc.get("log", scene["log_token"])["location"]
        description = scene["description"]

        # Ordered list of sample tokens in the scene.
        sample_tokens = []
        tok = scene["first_sample_token"]
        while tok:
            sample_tokens.append(tok)
            tok = nusc.get("sample", tok)["next"]
        n = len(sample_tokens)

        # Precompute per-sample ego pose (x, y, yaw), timestamp, and image paths.
        poses, times, frames_paths, frames_objs = [], [], [], []
        for st in sample_tokens:
            sample = nusc.get("sample", st)
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose", sd["ego_pose_token"])
            poses.append((ego["translation"][0], ego["translation"][1], _quat_to_yaw(ego["rotation"])))
            times.append(sample["timestamp"] * 1e-6)  # microseconds -> seconds
            paths = {}
            for cam in cameras:
                cam_sd = nusc.get("sample_data", sample["data"][cam])
                paths[cam] = cam_sd["filename"]
            frames_paths.append(paths)
            objs = [_readable_category(nusc.get("sample_annotation", a)["category_name"])
                    for a in sample["anns"]]
            frames_objs.append(objs)

        poses = np.asarray(poses, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)

        for start in range(0, n - clip_len + 1, stride):
            idxs = list(range(start, start + clip_len))
            anchor = idxs[-1]
            ax, ay, ayaw = poses[anchor]
            ca, sa = np.cos(ayaw), np.sin(ayaw)
            # Rotation global->anchor-ego (yaw only).
            R = np.array([[ca, sa], [-sa, ca]])

            # Speed and yaw-rate at anchor (forward finite difference, fall back to backward).
            if anchor + 1 < n:
                j, dt = anchor + 1, times[anchor + 1] - times[anchor]
            else:
                j, dt = anchor - 1, times[anchor] - times[anchor - 1]
            dt = max(abs(dt), 1e-3)
            d = poses[j, :2] - poses[anchor, :2]
            speed = float(np.linalg.norm(d) / dt)
            dyaw = float(np.arctan2(np.sin(poses[j, 2] - poses[anchor, 2]),
                                    np.cos(poses[j, 2] - poses[anchor, 2])))
            yaw_rate = dyaw / dt * (1.0 if j > anchor else -1.0)

            # Future waypoints in anchor ego frame (x forward, y left).
            waypoints, valid = [], True
            for h in range(1, future_horizon + 1):
                fi = anchor + h
                if fi >= n:
                    waypoints.append([0.0, 0.0])
                    valid = False
                else:
                    rel = R @ (poses[fi, :2] - np.array([ax, ay]))
                    waypoints.append([float(rel[0]), float(rel[1])])

            # Top-3 object categories across the clip frames.
            obj_counter: Counter = Counter()
            for fi in idxs:
                obj_counter.update(frames_objs[fi])
            top_objs = [o for o, _ in obj_counter.most_common(3)]

            caption = _make_caption(location, description, speed, yaw_rate, top_objs)

            clip_image_paths = [[frames_paths[i][c] for c in cameras] for i in idxs]
            if require_images and not all(
                os.path.exists(os.path.join(dataroot, p))
                for frame in clip_image_paths for p in frame):
                continue  # skip clips whose images aren't present on disk

            clips[split].append({
                "scene": scene_name,
                "location": location,
                "sample_tokens": [sample_tokens[i] for i in idxs],
                "cameras": cameras,
                "image_paths": clip_image_paths,
                "ego_poses": [[float(poses[i, 0]), float(poses[i, 1]), float(poses[i, 2])] for i in idxs],
                "timestamps": [float(times[i]) for i in idxs],
                "speed": speed,
                "yaw_rate": yaw_rate,
                "motion_target": waypoints,       # [H, 2]
                "motion_valid": bool(valid),
                "objects": top_objs,
                "caption": caption,
            })

    meta = {
        "version": version, "dataroot": os.path.abspath(dataroot),
        "clip_len": clip_len, "stride": stride, "future_horizon": future_horizon,
        "cameras": cameras,
        "n_train": len(clips["train"]), "n_val": len(clips["val"]),
    }
    index = {"meta": meta, **clips}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(index, f)
    return index


def load_index(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)
