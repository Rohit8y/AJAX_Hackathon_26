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

from skeleton_data import SkeletonData, Phase
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
LATER_CONTACT_LOOKAHEAD_FRAMES    = 25
LATER_CONTACT_MIN_ACC_RATIO       = 0.75
LATER_CONTACT_MIN_FOOT_GAIN_M     = 0.15

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading data...")
ball_df  = pd.read_parquet(BALL_IN)
parts_df = pd.read_parquet(PARTS_IN)

# EventParser needs a SkeletonData for time calibration
sd     = SkeletonData(TRIMMED_PARQUET)
parser = EventParser(XML, sd)
events = parser.get_events(EVENT_CODES, pad_before_sec=0, pad_after_sec=0)

# Phase boundaries for match-relative time
meta = sd.metadata
p1   = meta.frames_for_phase(Phase.FIRST_HALF)
p2   = meta.frames_for_phase(Phase.SECOND_HALF)
p1_start = p1.start_frame if p1 else 0
p2_start = p2.start_frame if p2 else None
framerate = meta.framerate

def frame_to_match_time(frame: int) -> tuple[int, float]:
    """Returns (half, match_time_s) where match_time_s is seconds from kickoff.
    Half 1: 0s = kickoff. Half 2: 0s = second half kickoff (reported as 45:00+)."""
    if p2_start is not None and frame >= p2_start:
        return 2, 45 * 60 + (frame - p2_start) / framerate
    return 1, (frame - p1_start) / framerate

FAR_FROM_GOAL_M = 35.0   # if spike is farther than this from either goal → use position-based goal_cos

# Goal positions in metres (same coordinate system as ball bx/by)
goal_x = (meta.pitch_long - meta.pitch_padding_left - meta.pitch_padding_right) / 10 / 2
GOALS  = np.array([[goal_x, 0.0], [-goal_x, 0.0]])

def goal_cos_xonly(bvx_arr, spd_arr):
    """
    |bvx| / speed — fraction of ball speed in the X direction.
    Goals lie on the X axis so this filters lateral passes (low score) from
    shots toward goal (high score) without depending on ball position.
    """
    safe_spd = np.where(spd_arr > 0.5, spd_arr, np.nan)
    return np.abs(np.nan_to_num(bvx_arr / safe_spd, nan=0.0))

def goal_cos_pos(bx_arr, by_arr, bvx_arr, bvy_arr, spd_arr):
    """
    cos(angle between ball velocity and direction to nearest goal), computed
    from actual ball position. Correctly handles wide-angle shots from near
    the goal line that have large bvy components.
    """
    n = len(bx_arr)
    best = np.zeros(n)
    safe_spd = np.where(spd_arr > 0.5, spd_arr, np.nan)
    for gx, gy in GOALS:
        to_gx = gx - bx_arr; to_gy = gy - by_arr
        dist  = np.sqrt(to_gx**2 + to_gy**2)
        to_gx_n = to_gx / np.where(dist > 0.1, dist, np.nan)
        to_gy_n = to_gy / np.where(dist > 0.1, dist, np.nan)
        dx = bvx_arr / safe_spd; dy = bvy_arr / safe_spd
        cos_a = dx * to_gx_n + dy * to_gy_n
        best = np.maximum(best, np.nan_to_num(cos_a, nan=0.0))
    return best

print(f"  ball frames   : {len(ball_df):,}")
print(f"  parts rows    : {len(parts_df):,}")
print(f"  shot events   : {len(events)}")
print(f"  phase 1 start : frame {p1_start}")
print(f"  phase 2 start : frame {p2_start}")

# Pre-filter parts to foot parts only (speeds up per-event lookups)
feet_df = parts_df[parts_df["body_part"].isin(FOOT_PARTS)].copy()
ball_by_frame = ball_df.set_index("frame_number", drop=False).sort_index()
feet_by_frame = feet_df.set_index("frame_number", drop=False).sort_index()

def nearest_foot_contact(frame: int) -> tuple[float, int | None, str | None, str | None]:
    """Closest foot-to-ball contact at exactly one frame."""
    try:
        ball_row = ball_by_frame.loc[frame]
    except KeyError:
        return np.nan, None, None, None

    try:
        ff = feet_by_frame.loc[frame]
    except KeyError:
        return np.nan, None, None, None

    if isinstance(ff, pd.Series):
        ff = ff.to_frame().T

    dist = np.sqrt(
        (ff["x"] - float(ball_row["bx"])) ** 2 +
        (ff["y"] - float(ball_row["by"])) ** 2 +
        (ff["z"] - float(ball_row["bz"])) ** 2
    )
    best_pos = int(np.argmin(dist.to_numpy()))
    best_row = ff.iloc[best_pos]
    return (
        float(dist.iloc[best_pos]),
        int(best_row["jersey_number"]),
        str(best_row["team_name"]),
        str(best_row["body_part_name"]),
    )

def best_contact_in_lookback(frame: int, lookback_frames: int) -> tuple[float, int | None, str | None, str | None]:
    """Best contact in [frame-lookback_frames, frame]. Used for shooter ID."""
    best_dist = np.nan
    best_jersey = None
    best_team = None
    best_part = None

    for check_frame in range(frame - lookback_frames, frame + 1):
        dist, jersey, team, part = nearest_foot_contact(check_frame)
        if np.isnan(dist):
            continue
        if np.isnan(best_dist) or dist < best_dist:
            best_dist = dist
            best_jersey = jersey
            best_team = team
            best_part = part

    return best_dist, best_jersey, best_team, best_part

def is_local_peak(arr: np.ndarray, idx: int) -> bool:
    return 0 < idx < len(arr) - 1 and arr[idx] >= arr[idx - 1] and arr[idx] >= arr[idx + 1]

def maybe_promote_later_contact(window_ball: pd.DataFrame, acc_mag: np.ndarray, spike_idx: int) -> int:
    """
    Promote a later contact only when the current winner looks like a setup touch:
    a later local-peak contact must keep most of the acceleration while improving
    exact-frame foot contact materially.
    """
    current_frame = int(window_ball.loc[spike_idx, "frame_number"])
    current_acc = float(acc_mag[spike_idx])
    current_exact_dist, _, _, _ = nearest_foot_contact(current_frame)
    if np.isnan(current_exact_dist):
        return spike_idx

    best_idx = spike_idx
    best_score = -np.inf
    end_idx = min(len(window_ball) - 2, spike_idx + LATER_CONTACT_LOOKAHEAD_FRAMES)

    for cand_idx in range(spike_idx + 1, end_idx + 1):
        cand_acc = float(acc_mag[cand_idx])
        if cand_acc < current_acc * LATER_CONTACT_MIN_ACC_RATIO:
            continue
        if not is_local_peak(acc_mag, cand_idx):
            continue

        cand_frame = int(window_ball.loc[cand_idx, "frame_number"])
        cand_exact_dist, _, _, _ = nearest_foot_contact(cand_frame)
        if np.isnan(cand_exact_dist):
            continue
        if cand_exact_dist > current_exact_dist - LATER_CONTACT_MIN_FOOT_GAIN_M:
            continue

        score = cand_acc / (1.0 + cand_exact_dist)
        if score > best_score:
            best_idx = cand_idx
            best_score = score

    return best_idx

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
            "half": None, "shot_frame": None, "match_time_s": None, "match_time_mmss": None,
            "ball_acc_spike_mps2": None,
            "ball_speed_before_mps": None, "ball_speed_after_mps": None,
            "shooter_jersey": None, "shooter_team": None, "shooter_body_part": None,
            "foot_ball_dist_m": None, "low_confidence": True,
            "xml_players_tagged": "|".join(ev.players_tagged),
        })
        continue

    window_ball = window_ball.reset_index(drop=True)
    acc_mag    = window_ball["ball_acc_mag"].to_numpy()
    ball_speed = window_ball["ball_speed"].to_numpy()
    n          = len(ball_speed)

    # --- Find shot contact frame ---
    # Score = delta_speed × goal_cos:
    #   delta_speed: ball speed after - before contact (real kick sends slow ball fast)
    #   goal_cos:    cos(angle between ball velocity and direction to nearest goal)
    #                → passes go sideways (low score), shots go toward goal (high score)
    LOOK = 3
    delta_speed = np.full(n, np.nan)
    for i in range(n):
        after  = ball_speed[i+1 : min(n, i+1+LOOK)]
        before = ball_speed[max(0, i-LOOK) : i]
        if len(after) > 0 and len(before) > 0:
            delta_speed[i] = np.mean(after) - np.mean(before)

    bx_a   = window_ball["bx"].to_numpy()
    by_a   = window_ball["by"].to_numpy()
    bvx_a  = window_ball["bvx"].to_numpy()
    bvy_a  = window_ball["bvy"].to_numpy()

    # Phase 1: |bvx|/speed — filters lateral passes, robust for most shots
    gc1       = goal_cos_xonly(bvx_a, ball_speed)
    shot_score = np.nan_to_num(delta_speed, nan=0.0) * gc1
    spike_idx  = int(np.argmax(shot_score))

    # Phase 2: if winning frame is far from both goals (>35 m) the xonly
    # winner is likely a long pass that set up the real shot. Re-score using
    # position-based goal_cos which correctly handles wide-angle near-goal
    # shots that have large bvy (e.g. ball coming in from the flank).
    spike_bx = bx_a[spike_idx]; spike_by = by_a[spike_idx]
    dist_to_goal = min(np.sqrt((gx-spike_bx)**2+(gy-spike_by)**2) for gx,gy in GOALS)
    if dist_to_goal > FAR_FROM_GOAL_M:
        gc2        = goal_cos_pos(bx_a, by_a, bvx_a, bvy_a, ball_speed)
        shot_score = np.nan_to_num(delta_speed, nan=0.0) * gc2
        spike_idx  = int(np.argmax(shot_score))

    # Narrow multi-touch override: if the selected frame is a setup touch, promote
    # a later local acceleration peak only when its exact-frame foot contact is
    # clearly tighter. This keeps long shots stable while fixing pass-then-shot
    # sequences like event 640.
    spike_idx = maybe_promote_later_contact(window_ball, acc_mag, spike_idx)

    spike_acc  = float(acc_mag[spike_idx])
    shot_frame = int(window_ball.loc[spike_idx, "frame_number"])
    half, match_time_s = frame_to_match_time(shot_frame)
    mins, secs = divmod(int(match_time_s), 60)
    match_time_mmss = f"{mins:02d}:{secs:02d}"

    # Ball speed 3 frames before and after spike
    before = window_ball.loc[max(0, spike_idx - 3) : spike_idx - 1, "ball_speed"]
    after  = window_ball.loc[spike_idx + 1 : spike_idx + 3, "ball_speed"]
    speed_before = float(before.mean()) if not before.empty else np.nan
    speed_after  = float(after.mean())  if not after.empty  else np.nan

    # --- Find shooter by minimum foot-ball distance in [shot_frame-3, shot_frame] ---
    # At shot_frame the ball has already left the foot (ball moves ~1m/frame at 25 m/s).
    # The actual contact is the frame just before the ball speed jumps — where the foot
    # is closest to the ball. Scan backwards to find it.
    CONTACT_LOOKBACK = 3

    foot_dist, shooter_jersey, shooter_team, shooter_part = best_contact_in_lookback(
        shot_frame,
        CONTACT_LOOKBACK,
    )

    low_confidence = (spike_acc < MIN_ACC_MPS2) or (np.isnan(foot_dist)) or (foot_dist > MAX_FOOT_DIST_M)

    results.append({
        "event_id":             ev.id,
        "event_code":           ev.code,
        "event_start_frame":    ev.start_frame,
        "event_end_frame":      ev.end_frame,
        "half":                 half,
        "shot_frame":           shot_frame,
        "match_time_s":         round(match_time_s, 1),
        "match_time_mmss":      match_time_mmss,
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
    "event_code", "half", "match_time_mmss",
    "ball_acc_spike_mps2", "ball_speed_after_mps",
    "shooter_jersey", "shooter_team", "foot_ball_dist_m", "low_confidence"
]].to_string(index=False))
