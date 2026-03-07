# Shot Frame Detection — Phase 2: Ball Spike Detection + Shooter ID

## Goal
For each shot/goal event window in `data/shots_trimmed.parquet`, find:
1. The **exact shot frame** (frame of ball-foot contact) via ball acceleration spike
2. The **shooter** (player whose foot is closest to ball at that frame)

Output: `data/shot_detections.csv` — one row per detected shot.

---

## Key Facts (from Primer + codebase)

| Property | Value |
|---|---|
| Ball position in parquet | cm (→ divide by 100 for metres) |
| Ball velocity in parquet | m/s (already SI — no conversion needed) |
| Skeleton positions | cm (→ divide by 100 for metres) |
| Framerate | 25 Hz → dt = 0.04 s |
| Acceleration from velocity | central diff on `[vx, vy, vz]` → m/s² |
| Foot body parts for shooter | RIGHT_TOE=21, LEFT_TOE=19, RIGHT_ANKLE=17, LEFT_ANKLE=16 |

---

## Algorithm

### Step 1 — Load trimmed parquet + events
```
sd = SkeletonData("data/shots_trimmed.parquet")
ep = EventParser("data/XML_anonymized.xml", sd)
events = ep.get_events(EVENT_CODES, pad_before_sec=0, pad_after_sec=0)
# No extra padding — frames already padded in Phase 1
```

Note: `EventParser` uses `sd.metadata` for time calibration. Trimmed parquet preserves
the same schema metadata as the original (PyArrow write preserves it), so calibration
is identical.

### Step 2 — For each event window: extract ball time series

Use `SkeletonData.frames(event.start_frame, event.end_frame)` to iterate Frame objects.

Build a ball DataFrame per event:
```
columns: frame_number, t, ball_x_m, ball_y_m, ball_z_m, ball_vx, ball_vy, ball_vz
```
- `ball_x/y/z_m = ball_pos / 100`  (cm → m)
- `ball_vx/vy/vz` = raw velocity (already m/s)
- Skip frames where `frame.ball is None`

### Step 3 — Compute ball acceleration & find spike frame

```python
from scipy.signal import savgol_filter

# Smooth velocity then differentiate (Savitzky-Golay, window=0.25s, poly=3)
# OR: central diff on velocity directly (simpler, acceptable at 25Hz)

def central_diff(arr, dt):
    out = np.empty_like(arr)
    out[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
    out[0]  = (arr[1] - arr[0]) / dt
    out[-1] = (arr[-1] - arr[-2]) / dt
    return out

vel = ball_df[["ball_vx", "ball_vy", "ball_vz"]].to_numpy()
acc = central_diff(vel, dt=1/framerate)         # shape (N, 3)
acc_mag = np.linalg.norm(acc, axis=1)           # shape (N,)
ball_speed = np.linalg.norm(vel, axis=1)        # shape (N,)
```

**Spike frame selection:**
- Take the frame index with `argmax(acc_mag)` within the event window
- Guard: if `max(acc_mag) < 50 m/s²`, flag as "low confidence" (no clear contact)
- `shot_frame = ball_df.iloc[spike_idx]["frame_number"]`
- `ball_speed_before = ball_speed[max(0, spike_idx - 3):spike_idx].mean()`
- `ball_speed_after  = ball_speed[spike_idx+1:spike_idx+4].mean()`

### Step 4 — Identify shooter

At `shot_frame`, get all players' foot positions:

```python
flat = sd.to_flat_dataframe(start_frame=shot_frame, end_frame=shot_frame)
foot_parts = [16, 17, 19, 21]   # left_ankle, right_ankle, left_toe, right_toe
feet = flat[flat["body_part"].isin(foot_parts)].copy()

# Convert positions to metres
feet["x_m"] = feet["pos_x"] / 100
feet["y_m"] = feet["pos_y"] / 100
feet["z_m"] = feet["pos_z"] / 100

# Ball position at shot_frame (metres)
bx, by, bz = ball_at_spike_frame  # from ball_df

feet["dist_m"] = np.sqrt(
    (feet["x_m"] - bx)**2 +
    (feet["y_m"] - by)**2 +
    (feet["z_m"] - bz)**2
)

closest = feet.loc[feet["dist_m"].idxmin()]
shooter_jersey = closest["jersey_number"]
shooter_team   = closest["team_name"]
shooter_part   = closest["body_part_name"]
foot_ball_dist = closest["dist_m"]
```

Threshold: if `foot_ball_dist > 2.0 m`, shooter detection is unreliable — flag it.

### Step 5 — Collect results & write CSV

One row per event:

| Column | Description |
|---|---|
| `event_id` | XML event id |
| `event_code` | e.g. "AJAX \| SCHOT" |
| `event_start_frame` | padded window start |
| `event_end_frame` | padded window end |
| `shot_frame` | detected contact frame |
| `shot_time_s` | `shot_frame / framerate` |
| `ball_acc_spike_mps2` | peak acceleration magnitude |
| `ball_speed_before_mps` | mean speed 3 frames before spike |
| `ball_speed_after_mps` | mean speed 3 frames after spike |
| `shooter_jersey` | jersey number of closest player |
| `shooter_team` | team name |
| `shooter_body_part` | which foot part was closest |
| `foot_ball_dist_m` | foot-to-ball distance at contact |
| `low_confidence` | True if acc < 50 or dist > 2.0 m |
| `xml_players_tagged` | players tagged in event (from XML labels) |

---

## New File: `detect_shot_frames.py`

```
Constants (top of file, configurable):
  PARQUET_IN   = "data/shots_trimmed.parquet"
  XML          = "data/XML_anonymized.xml"
  CSV_OUT      = "data/shot_detections.csv"
  EVENT_CODES  = ["AJAX | SCHOT", "FORTUNA SITTARD | SCHOT",
                  "AJAX | DOELPUNT", "FORTUNA SITTARD | DOELPUNT"]
  MIN_ACC_MPS2 = 50.0     # threshold for valid spike
  MAX_FOOT_DIST_M = 2.0   # threshold for valid shooter
  FOOT_PARTS   = [16, 17, 19, 21]  # ankle + toe (both sides)

Steps:
  1. Load SkeletonData + EventParser
  2. get_events(EVENT_CODES, pad_before_sec=0, pad_after_sec=0)
  3. For each event:
     a. frames = sd.frames(event.start_frame, event.end_frame)
     b. Build ball_df (skip frames with no ball)
     c. central_diff on velocity → acc_mag
     d. spike_idx = argmax(acc_mag)
     e. flat = sd.to_flat_dataframe(shot_frame, shot_frame)
     f. Find closest foot → shooter
     g. Append row to results list
  4. pd.DataFrame(results).to_csv(CSV_OUT, index=False)
  5. Print summary: N events, N detected, N low-confidence
```

**Reuses:** `SkeletonData` (`skeleton_data.py:301`), `EventParser` (`event_data.py`)

---

## Verification

```bash
.venv/bin/python detect_shot_frames.py
# Expected output:
#   Events loaded: ~10
#   Shots detected: ~10
#   Low confidence: ~1-2
#   Wrote data/shot_detections.csv

.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('data/shot_detections.csv')
print(df[['event_code','shot_frame','shooter_jersey','ball_speed_after_mps','low_confidence']])
"
```

Sanity checks:
- `ball_speed_after_mps` should be 5–30 m/s for a real shot (not 0.1 or 150)
- `foot_ball_dist_m` should be < 0.5 m for a good detection
- `shooter_jersey` should match `xml_players_tagged` where tagged

---

## Next Phase
**Phase 3**: Per-shot kinematics — knee angular velocity, foot speed, pelvis orientation
at `shot_frame ± window` using the segment frame machinery from the primer (Sections 5–8).
