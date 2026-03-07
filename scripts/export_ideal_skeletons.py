#!/usr/bin/env python3
"""Export ideal_skeletons.pkl → viz/public/data/ideal_kinematics.json"""

import json
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKL_PATH = ROOT / "data" / "ideal_skeletons.pkl"
OUT_PATH = ROOT / "viz" / "public" / "data" / "ideal_kinematics.json"

# Downsample to 60 frames for smooth playback
MAX_FRAMES = 60


def export():
    with open(PKL_PATH, "rb") as f:
        shots = pickle.load(f)

    out = []
    for s in shots:
        t = s["t"]
        original_pts = s["original_pts"]  # (N, 21, 3)
        ideal_pts = s["ideal_pts"]        # (N, 21, 3)
        n_frames = len(t)

        # Downsample if needed
        if n_frames > MAX_FRAMES:
            idx = np.linspace(0, n_frames - 1, MAX_FRAMES, dtype=int)
            t = t[idx]
            original_pts = original_pts[idx]
            ideal_pts = ideal_pts[idx]

        # Round for compact JSON
        t_list = [round(float(v), 3) for v in t]
        original_frames = [
            [[round(float(c), 3) for c in joint] for joint in frame]
            for frame in original_pts
        ]
        ideal_frames = [
            [[round(float(c), 3) for c in joint] for joint in frame]
            for frame in ideal_pts
        ]

        # Determine which joints changed significantly
        diff = np.abs(original_pts - ideal_pts).mean(axis=0).mean(axis=1)  # (21,)
        changed_joints = [int(j) for j in np.where(diff > 0.005)[0]]

        out.append({
            "event_id": s["event_id"],
            "match_time": s["match_time"],
            "shooter_jersey": s["shooter_jersey"],
            "kicking_side": s["kicking_side"],
            "wcs_original": s["wcs_original"],
            "wcs_ideal": s["wcs_ideal"],
            "t": t_list,
            "original_frames": original_frames,
            "ideal_frames": ideal_frames,
            "ideal_peak_times": {
                k: round(float(v), 3)
                for k, v in s["ideal_peak_times"].items()
            },
            "modification_flags": s["modification_flags"],
            "changed_joints": changed_joints,
        })

    payload = {"shots": out}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Exported {len(out)} ideal shots → {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    export()
