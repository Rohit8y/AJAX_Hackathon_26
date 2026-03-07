"""
Phase 2: Flatten shots_trimmed.parquet → SI units + velocity + acceleration.

Outputs:
  data/shots_parts.parquet  — one row per (frame, player, body_part)
                              with x/y/z in metres, vx/vy/vz in m/s, ax/ay/az in m/s²
  data/shots_ball.parquet   — one row per frame
                              with ball position (m), velocity (m/s), acceleration (m/s²)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from utils.skeleton_data import SkeletonData

# ── Config ────────────────────────────────────────────────────────────────────

PARQUET_IN  = "data/shots_trimmed.parquet"
PARTS_OUT   = "data/shots_parts.parquet"
BALL_OUT    = "data/shots_ball.parquet"

SG_WINDOW_S = 0.25   # Savitzky-Golay window in seconds
SG_POLY     = 3      # polynomial order

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading trimmed parquet...")
sd        = SkeletonData(PARQUET_IN)
framerate = sd.metadata.framerate
dt        = 1.0 / framerate
print(f"  framerate : {framerate} Hz  (dt = {dt:.4f} s)")
print(f"  frames    : {sd.frame_count:,}")

# ── Flatten to tidy DataFrame ─────────────────────────────────────────────────

print("\nFlattening to tidy DataFrame...")
flat = sd.to_flat_dataframe(include_ball=True)
print(f"  rows (player × part × frame): {len(flat):,}")

# ── Split: parts + ball ───────────────────────────────────────────────────────

# --- Parts ---
parts_raw = flat[["frame_number", "jersey_number", "team", "team_name",
                   "body_part", "body_part_name",
                   "pos_x", "pos_y", "pos_z"]].copy()

# cm → m, frame_number → t
parts_raw["x"] = parts_raw["pos_x"] / 100.0
parts_raw["y"] = parts_raw["pos_y"] / 100.0
parts_raw["z"] = parts_raw["pos_z"] / 100.0
parts_raw["t"] = parts_raw["frame_number"] / framerate
parts_raw.drop(columns=["pos_x", "pos_y", "pos_z"], inplace=True)

# --- Ball ---
ball_raw = (
    flat[["frame_number", "ball_x", "ball_y", "ball_z",
          "ball_vx", "ball_vy", "ball_vz"]]
    .drop_duplicates("frame_number")
    .copy()
)
ball_raw["bx"] = ball_raw["ball_x"] / 100.0   # cm → m
ball_raw["by"] = ball_raw["ball_y"] / 100.0
ball_raw["bz"] = ball_raw["ball_z"] / 100.0
ball_raw["bvx"] = ball_raw["ball_vx"]          # already m/s
ball_raw["bvy"] = ball_raw["ball_vy"]
ball_raw["bvz"] = ball_raw["ball_vz"]
ball_raw["t"]   = ball_raw["frame_number"] / framerate
ball_raw.drop(columns=["ball_x","ball_y","ball_z","ball_vx","ball_vy","ball_vz"], inplace=True)
ball_raw.sort_values("frame_number", inplace=True)
ball_raw.reset_index(drop=True, inplace=True)

print(f"  unique players : {parts_raw[['team','jersey_number']].drop_duplicates().shape[0]}")
print(f"  unique frames  : {parts_raw['frame_number'].nunique()}")
print(f"  ball frames    : {len(ball_raw)}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def central_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """Central difference derivative. Keeps shape (N, 3)."""
    out = np.empty_like(arr)
    out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
    out[0]    = (arr[1]  - arr[0])   / dt
    out[-1]   = (arr[-1] - arr[-2])  / dt
    return out


def sg_vel_acc(pos: np.ndarray, fs: float,
               window_s: float = SG_WINDOW_S,
               poly: int = SG_POLY) -> tuple[np.ndarray, np.ndarray]:
    """
    Savitzky-Golay velocity + acceleration from position.
    pos: (N, 3) in metres.
    Returns vel (N, 3) m/s, acc (N, 3) m/s².
    Falls back to central diff when group is too short for SG window.
    """
    w = int(round(window_s * fs))
    if w < 5:
        w = 5
    if w % 2 == 0:
        w += 1

    n = pos.shape[0]
    if n < w:
        # Group too short for SG — use central diff
        vel = central_diff(pos, 1.0 / fs)
        acc = central_diff(vel, 1.0 / fs)
        return vel, acc

    vel = savgol_filter(pos, window_length=w, polyorder=poly,
                        deriv=1, delta=1.0/fs, axis=0, mode="interp")
    acc = savgol_filter(pos, window_length=w, polyorder=poly,
                        deriv=2, delta=1.0/fs, axis=0, mode="interp")
    return vel, acc

# ── Player kinematics: SG per (team, jersey, body_part) group ─────────────────

print("\nComputing player velocity + acceleration (Savitzky-Golay)...")

result_chunks = []
groups = parts_raw.groupby(["team", "jersey_number", "body_part"], sort=False)

for (team, jersey, part), grp in groups:
    grp = grp.sort_values("frame_number").copy()
    pos = grp[["x", "y", "z"]].to_numpy(dtype=float)

    vel, acc = sg_vel_acc(pos, framerate)

    grp[["vx", "vy", "vz"]] = vel
    grp[["ax", "ay", "az"]] = acc
    grp["speed"]   = np.linalg.norm(vel, axis=1)
    grp["acc_mag"] = np.linalg.norm(acc, axis=1)
    result_chunks.append(grp)

parts_df = (
    pd.concat(result_chunks, ignore_index=True)
    .sort_values(["frame_number", "team", "jersey_number", "body_part"])
    .reset_index(drop=True)
)

print(f"  done — {len(parts_df):,} rows")

# ── Ball kinematics: central diff on velocity ─────────────────────────────────

print("\nComputing ball acceleration (central diff on velocity)...")

bv = ball_raw[["bvx", "bvy", "bvz"]].to_numpy(dtype=float)
ba = central_diff(bv, dt)

ball_df = ball_raw.copy()
ball_df[["bax", "bay", "baz"]] = ba
ball_df["ball_speed"]   = np.linalg.norm(bv, axis=1)
ball_df["ball_acc_mag"] = np.linalg.norm(ba, axis=1)

print(f"  done — {len(ball_df):,} frames")

# ── Sanity checks ─────────────────────────────────────────────────────────────

print("\nSanity checks:")
pelvis_speed = parts_df[parts_df["body_part"] == 12]["speed"]
print(f"  pelvis speed  max  : {pelvis_speed.max():.2f} m/s  (expect 8–12)")
print(f"  pelvis speed  mean : {pelvis_speed.mean():.2f} m/s")
print(f"  ball speed    max  : {ball_df['ball_speed'].max():.2f} m/s  (expect 5–35)")
print(f"  ball acc max       : {ball_df['ball_acc_mag'].max():.1f} m/s²  (spikes > 100 = shot contact)")

# ── Write outputs ─────────────────────────────────────────────────────────────

print(f"\nWriting {PARTS_OUT} ...")
parts_df.to_parquet(PARTS_OUT, index=False)

print(f"Writing {BALL_OUT} ...")
ball_df.to_parquet(BALL_OUT, index=False)

print("\nDone.")
print(f"  shots_parts.parquet columns: {parts_df.columns.tolist()}")
print(f"  shots_ball.parquet  columns: {ball_df.columns.tolist()}")
