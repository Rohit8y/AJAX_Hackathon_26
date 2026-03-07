#!/usr/bin/env python3
"""Export shot_kinematics.pkl → viz/public/data/kinematics.json"""

import json
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKL_PATH = ROOT / "data" / "shot_kinematics.pkl"
OUT_PATH = ROOT / "viz" / "public" / "data" / "kinematics.json"

# Downsample time series to window [-2.5, 0.5] and reduce points
T_MIN, T_MAX = -2.5, 0.5
MAX_POINTS = 60  # enough for smooth curves


def downsample(t: np.ndarray, *arrays: np.ndarray):
    """Keep only points in [T_MIN, T_MAX] and downsample."""
    mask = (t >= T_MIN) & (t <= T_MAX)
    t_win = t[mask]
    arrs = [a[mask] for a in arrays]
    if len(t_win) > MAX_POINTS:
        idx = np.linspace(0, len(t_win) - 1, MAX_POINTS, dtype=int)
        t_win = t_win[idx]
        arrs = [a[idx] for a in arrs]
    return t_win, *arrs


def export():
    with open(PKL_PATH, "rb") as f:
        shots = pickle.load(f)

    out = []
    for i, s in enumerate(shots):
        t_ds, pelvis, hip, knee, foot = downsample(
            s["t"], s["omega_pelvis"], s["omega_hip"], s["omega_knee"], s["omega_foot"]
        )

        # skeleton_at_contact: dict {part_id: [x, y, z]}
        skel = s["skeleton_at_contact"]
        skeleton = {str(k): [round(v, 3) for v in pos] for k, pos in skel.items()}

        # Determine if goal
        is_goal = "DOELPUNT" in s["event_code"]

        # Determine team from event_code
        team = "AJAX" if "AJAX" in s["event_code"] else "FORTUNA SITTARD"

        out.append({
            "id": i,
            "event_code": s["event_code"],
            "team": team,
            "is_goal": is_goal,
            "match_time": s["match_time"],
            "shooter_jersey": s["shooter_jersey"],
            "kicking_side": s["kicking_side"],
            "ball_speed": round(s["ball_speed_after_mps"], 1),
            "whipchain_score": s["whipchain_score"],
            "quality": s["quality_flag"],
            # Time series (downsampled)
            "t": [round(v, 3) for v in t_ds],
            "omega_pelvis": [round(v, 2) for v in pelvis],
            "omega_hip": [round(v, 2) for v in hip],
            "omega_knee": [round(v, 2) for v in knee],
            "omega_foot": [round(v, 2) for v in foot],
            # Peaks
            "peak_t_pelvis": round(s["peak_t_pelvis"], 3),
            "peak_t_hip": round(s["peak_t_hip"], 3),
            "peak_t_knee": round(s["peak_t_knee"], 3),
            "peak_t_foot": round(s["peak_t_foot"], 3),
            "peak_omega_pelvis": round(s["peak_omega_pelvis"], 2),
            "peak_omega_hip": round(s["peak_omega_hip"], 2),
            "peak_omega_knee": round(s["peak_omega_knee"], 2),
            "peak_omega_foot": round(s["peak_omega_foot"], 2),
            # Skeleton at contact frame
            "skeleton": skeleton,
        })

    # Sort by whipchain_score descending
    out.sort(key=lambda x: -x["whipchain_score"])

    payload = {"shots": out}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Exported {len(out)} shots → {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    export()
