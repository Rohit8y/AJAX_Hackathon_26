# Shot Window Processing — Phase 2: Flatten + SI Units + Kinematics

## Goal
Take `data/shots_trimmed.parquet` and produce two tidy, analysis-ready DataFrames:
- **parts_df**: one row per `(frame, player, body_part)` with x/y/z in metres, plus velocity and acceleration columns
- **ball_df**: one row per frame with ball position (m) and velocity (m/s), plus ball acceleration

Saved as `data/shots_parts.parquet` and `data/shots_ball.parquet`.

---

## Primer Sections Covered

| Step | Primer Section |
|---|---|
| Flatten nested parquet to tidy rows | §2.1 |
| cm → m, frame_number → t (seconds) | §1.3 |
| Savitzky-Golay smooth + velocity + acceleration | §3.2 |
| Sprint metrics per player | §3.3 |

---

## Output Schemas

### parts_df — one row per `(frame_number, jersey_number, team, body_part)`

| Column | Type | Notes |
|---|---|---|
| `frame_number` | int | raw frame counter |
| `t` | float | `frame_number / framerate` in seconds |
| `team` | int | team id |
| `team_name` | str | e.g. "TEAM_A" |
| `jersey_number` | int | |
| `body_part` | int | 1–21 per PARTS dict |
| `body_part_name` | str | e.g. "RIGHT_TOE" |
| `x, y, z` | float | position in **metres** |
| `vx, vy, vz` | float | velocity in **m/s** (SG deriv 1) |
| `ax, ay, az` | float | acceleration in **m/s²** (SG deriv 2) |
| `speed` | float | `‖v‖` in m/s |
| `acc_mag` | float | `‖a‖` in m/s² |

### ball_df — one row per `frame_number`

| Column | Type | Notes |
|---|---|---|
| `frame_number` | int | |
| `t` | float | seconds |
| `bx, by, bz` | float | position in **metres** |
| `bvx, bvy, bvz` | float | velocity in **m/s** (raw from parquet — already SI) |
| `bax, bay, baz` | float | acceleration in **m/s²** (central diff on velocity) |
| `ball_speed` | float | `‖bv‖` m/s |
| `ball_acc_mag` | float | `‖ba‖` m/s² |

---

## Algorithm

### Step 1 — Load trimmed parquet

```python
sd = SkeletonData("data/shots_trimmed.parquet")
framerate = sd.metadata.framerate   # 25.0
dt = 1.0 / framerate
```

### Step 2 — Flatten to tidy parts_df + ball_df

Use `sd.to_flat_dataframe()` for the full trimmed file (no frame range restriction —
the file is already small). Then:

```python
flat = sd.to_flat_dataframe()   # existing method, returns ball columns too

# Split into parts and ball
parts_raw = flat[["frame_number", "jersey_number", "team", "team_name",
                   "body_part", "body_part_name",
                   "pos_x", "pos_y", "pos_z"]].copy()

# SI conversion: cm → m
parts_raw["x"] = parts_raw["pos_x"] / 100
parts_raw["y"] = parts_raw["pos_y"] / 100
parts_raw["z"] = parts_raw["pos_z"] / 100
parts_raw["t"] = parts_raw["frame_number"] / framerate

# Ball: deduplicate (ball cols repeat per player-part row)
ball_raw = flat[["frame_number", "ball_x", "ball_y", "ball_z",
                  "ball_vx", "ball_vy", "ball_vz"]].drop_duplicates("frame_number").copy()
ball_raw["bx"] = ball_raw["ball_x"] / 100
ball_raw["by"] = ball_raw["ball_y"] / 100
ball_raw["bz"] = ball_raw["ball_z"] / 100
ball_raw["t"]  = ball_raw["frame_number"] / framerate
```

### Step 3 — Savitzky-Golay velocity + acceleration per player per body part

For each `(team, jersey_number, body_part)` group, apply SG filter:

```python
from scipy.signal import savgol_filter

SG_WINDOW_S = 0.25   # seconds
SG_POLY     = 3

def sg_vel_acc(pos: np.ndarray, fs: float, window_s=0.25, poly=3):
    """pos shape (N,3). Returns vel (N,3), acc (N,3) in m/s and m/s²."""
    w = int(round(window_s * fs))
    if w < 5: w = 5
    if w % 2 == 0: w += 1
    vel = savgol_filter(pos, window_length=w, polyorder=poly,
                        deriv=1, delta=1/fs, axis=0, mode="interp")
    acc = savgol_filter(pos, window_length=w, polyorder=poly,
                        deriv=2, delta=1/fs, axis=0, mode="interp")
    return vel, acc

# Group by player + body part, compute derivatives, write back
records = []
for (team, jersey, part), grp in parts_raw.groupby(["team","jersey_number","body_part"]):
    grp = grp.sort_values("frame_number")
    pos = grp[["x","y","z"]].to_numpy()
    vel, acc = sg_vel_acc(pos, framerate)
    grp = grp.copy()
    grp[["vx","vy","vz"]] = vel
    grp[["ax","ay","az"]] = acc
    grp["speed"]   = np.linalg.norm(vel, axis=1)
    grp["acc_mag"] = np.linalg.norm(acc, axis=1)
    records.append(grp)

parts_df = pd.concat(records).sort_values(["frame_number","team","jersey_number","body_part"])
```

### Step 4 — Ball acceleration (central diff on velocity)

Ball velocity is already m/s from the parquet — no SG needed, just diff:

```python
def central_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    out = np.empty_like(arr)
    out[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
    out[0]  = (arr[1] - arr[0]) / dt
    out[-1] = (arr[-1] - arr[-2]) / dt
    return out

ball_df = ball_raw.sort_values("frame_number").copy()
bv = ball_df[["bvx","bvy","bvz"]].to_numpy()
ba = central_diff(bv, dt)
ball_df[["bax","bay","baz"]] = ba
ball_df["ball_speed"]   = np.linalg.norm(bv, axis=1)
ball_df["ball_acc_mag"] = np.linalg.norm(ba, axis=1)
```

### Step 5 — Save outputs

```python
parts_df.to_parquet("data/shots_parts.parquet", index=False)
ball_df.to_parquet("data/shots_ball.parquet", index=False)
```

---

## New File: `process_shot_windows.py`

```
Constants:
  PARQUET_IN   = "data/shots_trimmed.parquet"
  PARTS_OUT    = "data/shots_parts.parquet"
  BALL_OUT     = "data/shots_ball.parquet"
  SG_WINDOW_S  = 0.25
  SG_POLY      = 3

Steps:
  1. SkeletonData(PARQUET_IN) → flat = to_flat_dataframe()
  2. Split flat → parts_raw (cm → m, add t) + ball_raw (cm → m, add t)
  3. Groupby (team, jersey, body_part) → sg_vel_acc → parts_df with vx/vy/vz/ax/ay/az/speed/acc_mag
  4. central_diff on ball velocity → ball_df with bax/bay/baz/ball_speed/ball_acc_mag
  5. Write PARTS_OUT + BALL_OUT
  6. Print: N players, N body parts, frame range, max player speed sanity check
```

---

## Verification

```bash
.venv/bin/python process_shot_windows.py
# Expected:
#   N players: ~22, N frames: ~4500
#   Max pelvis speed: 8–11 m/s  (if you see 500+ → unit error, still in cm)
#   Ball max speed: 5–35 m/s    (if you see 0.05–0.35 → still in cm/s)

.venv/bin/python -c "
import pandas as pd
parts = pd.read_parquet('data/shots_parts.parquet')
ball  = pd.read_parquet('data/shots_ball.parquet')
print(parts.columns.tolist())
print('Max speed (pelvis):', parts[parts.body_part==12]['speed'].max())
print('Ball acc max:',       ball['ball_acc_mag'].max())
"
```

---

## Next Phase
**Phase 3**: Shot detection — ball acceleration spike → exact shot frame + shooter ID
(using `ball_df["ball_acc_mag"]` and foot proximity from `parts_df`).
