"""
Phase 3: Detect exact shot frame + shooter per event window.

Inputs:
  data/shots_ball.parquet   — ball kinematics (one row per frame)
  data/shots_parts.parquet  — player kinematics (one row per frame×player×part)
  data/XML_anonymized.xml   — events with frame mappings
  data/shots_trimmed.parquet — used only for EventParser calibration

Output:
  data/shot_detections.csv  — one row per shot event
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from skeleton_data import SkeletonData
from event_data import EventParser

# ── Config ────────────────────────────────────────────────────────────────────

TRIMMED_PARQUET = "data/shots_trimmed.parquet"
XML             = "data/XML_anonymized.xml"
BALL_IN         = "data/shots_ball.parquet"
PARTS_IN        = "data/shots_parts.parquet"
CSV_OUT         = "data/shot_detections.csv"

EVENT_CODES = [
    "AJAX | SCHOT",
    "FORTUNA SITTARD | SCHOT",
    "AJAX | DOELPUNT",
    "FORTUNA SITTARD | DOELPUNT",
]

FOOT_PARTS      = [16, 17, 19, 21]   # left_ankle, right_ankle, left_toe, right_toe
MIN_ACC_MPS2    = 50.0               # below this → no clear contact detected
MAX_FOOT_DIST_M = 2.0                # above this → shooter ID is unreliable

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading data...")
ball_df  = pd.read_parquet(BALL_IN)
parts_df = pd.read_parquet(PARTS_IN)

# EventParser needs a SkeletonData for time calibration
sd     = SkeletonData(TRIMMED_PARQUET)
parser = EventParser(XML, sd)
events = parser.get_events(EVENT_CODES, pad_before_sec=0, pad_after_sec=0)

print(f"  ball frames   : {len(ball_df):,}")
print(f"  parts rows    : {len(parts_df):,}")
print(f"  shot events   : {len(events)}")

# Pre-filter parts to foot parts only (speeds up per-event lookups)
feet_df = parts_df[parts_df["body_part"].isin(FOOT_PARTS)].copy()

# ── Per-event detection ───────────────────────────────────────────────────────

results = []

for ev in events:
    # --- Slice ball data for this event window ---
    window_ball = ball_df[
        (ball_df["frame_number"] >= ev.start_frame) &
        (ball_df["frame_number"] <= ev.end_frame)
    ].copy()

    if window_ball.empty or window_ball["ball_acc_mag"].isna().all():
        results.append({
            "event_id": ev.id, "event_code": ev.code,
            "event_start_frame": ev.start_frame, "event_end_frame": ev.end_frame,
            "shot_frame": None, "shot_time_s": None,
            "ball_acc_spike_mps2": None,
            "ball_speed_before_mps": None, "ball_speed_after_mps": None,
            "shooter_jersey": None, "shooter_team": None, "shooter_body_part": None,
            "foot_ball_dist_m": None, "low_confidence": True,
            "xml_players_tagged": "|".join(ev.players_tagged),
        })
        continue

    window_ball = window_ball.reset_index(drop=True)
    acc_mag = window_ball["ball_acc_mag"].to_numpy()

    # --- Find peak acceleration frame ---
    spike_idx   = int(np.nanargmax(acc_mag))
    spike_acc   = float(acc_mag[spike_idx])
    shot_frame  = int(window_ball.loc[spike_idx, "frame_number"])
    shot_time_s = float(window_ball.loc[spike_idx, "t"])

    # Ball speed 3 frames before and after spike
    before = window_ball.loc[max(0, spike_idx - 3) : spike_idx - 1, "ball_speed"]
    after  = window_ball.loc[spike_idx + 1 : spike_idx + 3, "ball_speed"]
    speed_before = float(before.mean()) if not before.empty else np.nan
    speed_after  = float(after.mean())  if not after.empty  else np.nan

    # Ball position at spike frame (metres)
    bx = float(window_ball.loc[spike_idx, "bx"])
    by = float(window_ball.loc[spike_idx, "by"])
    bz = float(window_ball.loc[spike_idx, "bz"])

    # --- Find closest foot at spike frame ---
    frame_feet = feet_df[feet_df["frame_number"] == shot_frame].copy()

    shooter_jersey = None
    shooter_team   = None
    shooter_part   = None
    foot_dist      = np.nan

    if not frame_feet.empty:
        frame_feet = frame_feet.copy()
        frame_feet["dist_m"] = np.sqrt(
            (frame_feet["x"] - bx) ** 2 +
            (frame_feet["y"] - by) ** 2 +
            (frame_feet["z"] - bz) ** 2
        )
        closest = frame_feet.loc[frame_feet["dist_m"].idxmin()]
        shooter_jersey = int(closest["jersey_number"])
        shooter_team   = str(closest["team_name"])
        shooter_part   = str(closest["body_part_name"])
        foot_dist      = float(closest["dist_m"])

    low_confidence = (spike_acc < MIN_ACC_MPS2) or (np.isnan(foot_dist)) or (foot_dist > MAX_FOOT_DIST_M)

    results.append({
        "event_id":             ev.id,
        "event_code":           ev.code,
        "event_start_frame":    ev.start_frame,
        "event_end_frame":      ev.end_frame,
        "shot_frame":           shot_frame,
        "shot_time_s":          round(shot_time_s, 3),
        "ball_acc_spike_mps2":  round(spike_acc, 1),
        "ball_speed_before_mps": round(speed_before, 2) if not np.isnan(speed_before) else None,
        "ball_speed_after_mps":  round(speed_after,  2) if not np.isnan(speed_after)  else None,
        "shooter_jersey":       shooter_jersey,
        "shooter_team":         shooter_team,
        "shooter_body_part":    shooter_part,
        "foot_ball_dist_m":     round(foot_dist, 3) if not np.isnan(foot_dist) else None,
        "low_confidence":       low_confidence,
        "xml_players_tagged":   "|".join(ev.players_tagged),
    })

# ── Output ────────────────────────────────────────────────────────────────────

out = pd.DataFrame(results)
out.to_csv(CSV_OUT, index=False)

print(f"\nResults written to {CSV_OUT}")
print(f"  total events    : {len(out)}")
print(f"  high confidence : {(~out['low_confidence']).sum()}")
print(f"  low confidence  : {out['low_confidence'].sum()}")
print()
print(out[[
    "event_code", "shot_frame", "shot_time_s",
    "ball_acc_spike_mps2", "ball_speed_after_mps",
    "shooter_jersey", "shooter_team", "foot_ball_dist_m", "low_confidence"
]].to_string(index=False))
