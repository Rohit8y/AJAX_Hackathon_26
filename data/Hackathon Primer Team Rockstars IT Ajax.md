**Written by Niek Bloks - ML engineer at Team Rockstars IT**

# Hack the Game — 3D Kinematics Primer (Ajax Tracking Data)

> **Goal**: turn raw 3D tracking data (players + ball) into **reliable metrics** and **creative features** within a hackathon timeframe.  
> **Scope**: positions → velocities → accelerations → 3D angles → **3D angular velocity/acceleration**, including explicit matrix multiplications and practical Python patterns.

**Audience**: developers, data engineers, analysts, and 3D/graphics folks — no biomechanics or physics background required.

**How this primer is structured (levels)**
- **Level 0** — *3D points as time series*: speed, acceleration, distances, proximity events.
- **Level 1** — *angles from vectors*: knee bend proxy, head-to-ball viewing angle, “open body” proxies.
- **Level 2** — *segment frames + transforms*: positions/velocities expressed *relative to the body* (e.g., ball-in-foot coordinates).
- **Level 3** — *3D angular velocity and joint angular velocity*: how fast segments/joints rotate in full 3D.

### What each level gives you in practice

- Level 0 gives you “tracking KPI building blocks”: speed, acceleration, sprint profiles, ball speed changes, distances, and event timing.
- Level 1 gives you fast, robust angles that are good enough for many football questions (knee bend proxy, head-to-ball, body open/closed, etc.).
- Level 2 makes your metrics **relative** and therefore more interpretable (ball in foot coordinates; teammate in torso coordinates).
- Level 3 unlocks full rotational kinematics: segment/joint angular velocities, sequencing, and “turning” features.

**What “kinematics” means (and what it doesn’t)**
Kinematics describes **motion**: position, orientation, velocity, and acceleration.  
It does **not** model forces or torques (that’s dynamics). For a 14‑hour hackathon, kinematics already unlocks a huge space: event detection, technique features, scanning/perception proxies, and 3D visualizations.

**A single mental model**
Think of the feed as a function of time:

$$
p(t)=\begin{bmatrix}x(t)\\y(t)\\z(t)\end{bmatrix}
\quad\Rightarrow\quad
v(t)=\frac{dp(t)}{dt},\; a(t)=\frac{d^2p(t)}{dt^2}
$$

If you can build a **rotation matrix** $R(t)$ for a body segment (pelvis, thigh, shank, foot, trunk), you can also compute **angular velocity** $\omega(t)$ (rad/s). That’s covered in Sections 5–8.

---

## 0. What you get from the tracking feed (TF15)

The match feed provides, per frame:

- **Players**: a skeleton with *(to date)* **21 body points** per player: ears, nose, shoulders, hips, knees, ankles, heels, toes, …  
- **Ball**: 3D position and **3D velocity**.
- **Sampling rate**: `framerate` = **25 / 30 / 50 / 60 Hz** (per match file metadata).
- **Units**:
  - Skeleton point positions: **centimeters** (cm) from the pitch center  
  - Ball position: **centimeters** (cm) from the pitch center  
  - Ball velocity: **meters per second** (m/s)

**Recommendation**: convert everything to **meters** early, and keep time in **seconds**.

### 0.1 Coordinate system: what X, Y, Z mean on the pitch

TF15 positions are expressed in the **TRACAB pitch coordinate system** (see the coordinate system diagram in the TF15 spec):

- The origin **(0, 0, 0)** is at the **center point** of the field.
- **X** points along the **pitch length** (goal-to-goal): values range from $-\tfrac{1}{2}\,\text{pitch length}$ to $+\tfrac{1}{2}\,\text{pitch length}$.
- **Y** points along the **pitch width** (touchline-to-touchline): values range from $-\tfrac{1}{2}\,\text{pitch width}$ to $+\tfrac{1}{2}\,\text{pitch width}$.
- **Z** is **vertical height** above the pitch ($z=0$ is ground contact).

This is intended as a **right-handed** coordinate system: with a consistent axis choice, the right-hand rule applies (conceptually $\mathbf{x}\times\mathbf{y}=\mathbf{z}$).

> **Football gotcha**: the coordinate system is fixed to the stadium, *not* to a team’s attacking direction.  
> If you want “towards opponent goal” features, normalize direction per half (e.g., flip X for the team that attacks to negative X in that half), so that “attacking” always points to $+X$.

### 0.2 Frame numbers, timestamps, and halves

The feed is sampled at a fixed `framerate` (Hz). Each row/frame has a `frame_number` which is essentially a counter. If you compute:

$$
t = \frac{\text{frame\\_number}}{\text{framerate}}
$$

you get a continuous time axis in seconds.

The TF15 header also provides start/end frame numbers for match phases (first half, second half, extra time, …). Those are useful when you want to slice analyses per half without relying on wall-clock time.

---

## 1. Quick-start for everyone (Level 0)

This section is intentionally simple: treat each body point as a 3D time series.  
You can already build sprint metrics, distances, accelerations, turning, and view angles.

A helpful mindset for newcomers:

- A **body part** (e.g., right ankle) is just a moving point in 3D.
- A **time series** is the same measurement repeated over time: $x(t), y(t), z(t)$.
- If you can compute $p(t)\rightarrow v(t)\rightarrow a(t)$, you already have “football physics”: who is accelerating, who is braking, who is closing down space, and what happens right before a touch.

You do **not** need to understand anatomy to start. Think “robot arm points” rather than “muscles”.

### 1.1 Minimal environment

You can do almost everything in this primer with four core libraries:

- **NumPy**: fast arrays; think “vectorized math”.
- **Pandas**: tables and joins; think “SQL in Python”.
- **PyArrow**: reads **Parquet**, which is a compressed columnar file format (great for huge match files).
- **SciPy**: signal processing tools (filters, smoothing, derivatives).

If you prefer notebooks: Jupyter + these packages is a solid setup for a hackathon.

```bash
pip install numpy pandas pyarrow scipy
```

- `numpy`: vectors/matrices
- `pandas`: data wrangling
- `pyarrow`: Parquet reading (nested schema-friendly)
- `scipy`: smoothing + derivatives (Savitzky–Golay)

### 1.2 Part-ID mapping (TF15)

TF15 stores each skeleton part as an integer `part_id`.  
Mapping these IDs to readable names early makes debugging much easier (“why is part 17 flying away?”).

A few practical notes:

- The skeleton is not a “medical-grade marker set”; it is an **estimated pose** from multi-camera tracking.
- Some parts may be noisier than others (e.g., toes/heels during occlusion).
- For many features, you will intentionally **reduce** complexity by choosing 1–2 representative points (pelvis, hips, or the ball).

The full list below matches the TF15 spec.

```python
PARTS = {
    1: "left_ear",
    2: "nose",
    3: "right_ear",
    4: "left_shoulder",
    5: "neck",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10:"right_wrist",
    11:"left_hip",
    12:"pelvis",
    13:"right_hip",
    14:"left_knee",
    15:"right_knee",
    16:"left_ankle",
    17:"right_ankle",
    18:"left_heel",
    19:"left_toe",
    20:"right_heel",
    21:"right_toe",
}
```

### 1.3 Units and time step

Kinematics is basically “**distance divided by time**” (velocity) and “**velocity divided by time**” (acceleration).  
So two things must be consistent:

1) **Units** (cm vs m)  
2) **Time step** $\Delta t$ (from the frame rate)

A very common mistake is mixing cm and m. If you do that, your speed and acceleration will be off by a factor of 100.

**Rule of thumb:** convert to SI once (meters, seconds) and never think about it again.

```python
def cm_to_m(x_cm: float) -> float:
    return 0.01 * x_cm

def cm3_to_m3(p_cm: "np.ndarray") -> "np.ndarray":
    # p_cm shape (..., 3)
    return 0.01 * p_cm

def dt_from_framerate(framerate_hz: float) -> float:
    return 1.0 / float(framerate_hz)
```

---

## 2. Reading TF15 Parquet into a “tidy” table (Level 0 → 1)

The Parquet schema contains nested arrays (frames → skeletons → parts).  
For hackathon speed, a simple Python loop is acceptable and easiest to debug.

**Why flatten?**  
Most analysis libraries (and your own brain at 03:00 in the hackathon) work best with “tidy” data:

- each row is one observation (here: a body part at a time frame)
- each column is one variable (x, y, z, jersey, …)

Once the data is flat, everything becomes straightforward: group by player, pivot to wide format, compute derivatives, join with ball data, etc.

**Performance note:** looping through all frames is not the fastest way to read Parquet, but it is the most transparent.  
If you later need speed, you can switch to DuckDB / PyArrow dataset filters — after you have a correct baseline.

### 2.1 Parse frames into a flat DataFrame

**Output table**: one row per `(frame, player, body_part)` with columns:

- `frame`, `t`
- `team`, `jersey_number`
- `part_id`, `part_name`
- `x, y, z` in **meters**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass(frozen=True)
class MatchMeta:
    framerate: float


def read_tf15_parquet_flat(path: str) -> Tuple[MatchMeta, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      meta: MatchMeta (contains framerate)
      parts_df: rows = (frame, jersey, part_id) with x,y,z in meters
      ball_df:  rows = (frame) with ball position (m) and velocity (m/s) if available
    """

    table = pq.read_table(path)
    frames = table.to_pylist()  # list[dict] where each dict is one frame

    # Metadata is embedded at file-level in TF15; depending on exporter it may appear in every row
    # or as separate parquet metadata. Below is a robust approach: take first frame fields if present.
    first = frames[0] if frames else {}
    framerate = first.get("framerate") or first.get("header", {}).get("framerate")
    if framerate is None:
        raise ValueError("Could not find 'framerate' in parquet. Inspect the schema / metadata.")

    meta = MatchMeta(framerate=float(framerate))
    dt = 1.0 / meta.framerate

    part_rows: List[Dict[str, Any]] = []
    ball_rows: List[Dict[str, Any]] = []

    for fr in frames:
        frame_number = int(fr.get("frame_number", fr.get("frame", fr.get("framecount", 0))))
        t = frame_number * dt

        # ---- Ball ----
        ball_exists = int(fr.get("ball_exists", 0))
        if ball_exists and "ball" in fr and fr["ball"] is not None:
            b = fr["ball"]
            ball_rows.append({
                "frame": frame_number,
                "t": t,
                "bx": 0.01 * float(b["position_x"]),
                "by": 0.01 * float(b["position_y"]),
                "bz": 0.01 * float(b["position_z"]),
                "bvx": float(b.get("velocity_x", np.nan)),  # already m/s in TF15
                "bvy": float(b.get("velocity_y", np.nan)),
                "bvz": float(b.get("velocity_z", np.nan)),
            })

        # ---- Skeletons ----
        for sk in fr.get("skeletons", []) or []:
            jersey = int(sk.get("jersey_number", -1))
            team = int(sk.get("team", -1))
            for part in sk.get("parts", []) or []:
                part_id = int(part.get("name"))
                part_rows.append({
                    "frame": frame_number,
                    "t": t,
                    "team": team,
                    "jersey": jersey,
                    "part_id": part_id,
                    "part": PARTS.get(part_id, f"part_{part_id}"),
                    "x": 0.01 * float(part["position_x"]),
                    "y": 0.01 * float(part["position_y"]),
                    "z": 0.01 * float(part["position_z"]),
                })

    parts_df = pd.DataFrame(part_rows).sort_values(["frame", "team", "jersey", "part_id"])
    ball_df = pd.DataFrame(ball_rows).sort_values(["frame"])

    return meta, parts_df, ball_df
```

**Notes**
- The field names (`frame_number`, `skeletons`, `parts`, …) follow the TF15 spec.
- If your Parquet exporter stores metadata differently, adjust how `framerate` and `frame_number` are read.

---

## 3. From position to velocity and acceleration (Level 0 → 1)

You do **not** get player speed or acceleration directly (except ball velocity). You derive them from position.

Conceptually (continuous time):

$$
\mathbf{v}(t)=\frac{d\mathbf{p}(t)}{dt} \quad [\text{m/s}], \qquad
\mathbf{a}(t)=\frac{d^2\mathbf{p}(t)}{dt^2} \quad [\text{m/s}^2]
$$

In the match file, you have samples at fixed intervals $\Delta t=1/\text{framerate}$.  
So in code you approximate derivatives using finite differences or filters.

**Why care?**  
Most interesting football moments are “changes”:

- acceleration bursts when pressing
- deceleration when braking for a cut
- foot speed just before ball contact
- ball acceleration spikes at touch events

All of those live in $\mathbf{v}(t)$ and $\mathbf{a}(t)$, not in raw $\mathbf{p}(t)$.

### 3.1 Choosing a representative point per player
Most football metrics use a single “player position” proxy:
- `pelvis` (part 12), or
- midpoint of `left_hip` and `right_hip` (11 and 13)

```python
def get_part_series(parts_df: pd.DataFrame, team: int, jersey: int, part_id: int) -> pd.DataFrame:
    df = parts_df[(parts_df.team == team) & (parts_df.jersey == jersey) & (parts_df.part_id == part_id)].copy()
    return df.sort_values("frame")

def to_numpy_xyz(df: pd.DataFrame) -> np.ndarray:
    return df[["x", "y", "z"]].to_numpy(dtype=float)
```

### 3.2 Smoothing + derivatives (recommended)
Numerical differentiation amplifies noise. A safe default is a Savitzky–Golay filter.

If you only remember one signal-processing fact:

> **Differentiation is a noise amplifier.**

Small frame-to-frame jitter in position becomes large spikes in velocity, and even larger spikes in acceleration.

A **Savitzky–Golay (SG)** filter is a good hackathon default because it does two things at once:

- fits a low-degree polynomial in a moving window (smoothing)
- analytically differentiates that polynomial (derivatives)

So you get smooth $p(t)$, $v(t)$, and $a(t)$ with one consistent method.

**Parameter intuition**
- `window_s`: how much time the filter “looks at” (e.g., 0.25 s). Larger = smoother but can blur sharp events.
- `poly`: polynomial order. 3 is a common, safe choice.

For high-speed events like ball contact you may want a shorter window (e.g., 0.10–0.15 s). For running speed you can go longer.

```python
from scipy.signal import savgol_filter

def savgol_pos_vel_acc(pos: np.ndarray, fs: float, window_s: float = 0.25, poly: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pos shape: (N, 3)
    returns: (pos_smooth, vel, acc)
    """
    dt = 1.0 / fs
    w = int(round(window_s * fs))
    if w < 5:
        w = 5
    if w % 2 == 0:
        w += 1  # SG needs odd window length

    pos_s = savgol_filter(pos, window_length=w, polyorder=poly, deriv=0, delta=dt, axis=0, mode="interp")
    vel   = savgol_filter(pos, window_length=w, polyorder=poly, deriv=1, delta=dt, axis=0, mode="interp")
    acc   = savgol_filter(pos, window_length=w, polyorder=poly, deriv=2, delta=dt, axis=0, mode="interp")
    return pos_s, vel, acc
```

### 3.3 Standard sprint metrics

From the 3D velocity vector $\mathbf{v}=[v_x,v_y,v_z]^T$, the **speed** is just its length:

$$
s(t)=\lVert\mathbf{v}(t)\rVert = \sqrt{v_x^2+v_y^2+v_z^2}
$$

Same idea for acceleration magnitude.

In football, you often also care about *horizontal* speed (on the grass) vs vertical motion:

$$
s_{xy}(t)=\sqrt{v_x^2+v_y^2}
$$

This primer keeps everything 3D, but you can always drop $z$ if a feature is about “running on the pitch”.

```python
def norm(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(v, axis=axis)

def sprint_metrics(vel: np.ndarray, acc: np.ndarray) -> dict:
    speed = norm(vel)
    acc_mag = norm(acc)
    return {
        "max_speed_mps": float(np.nanmax(speed)),
        "max_acc_mps2": float(np.nanmax(acc_mag)),
        "max_dec_mps2": float(np.nanmin(np.sum(acc * (vel / (speed[:, None] + 1e-12)), axis=1))),  # tangential decel approx
    }
```

### 3.4 Express velocity and acceleration in a player/segment frame (local coordinates)

A frequent feature request is:  
**“How fast is the foot moving *relative to the pelvis*?”** or  
**“What is the ball velocity in the foot’s local axes?”**

Before jumping into formulas, one key idea:

#### Global vs local coordinates (the “player-centric” view)

- **Global / pitch coordinates**: the raw TF15 $x,y,z$ values. Great for “where on the pitch?” questions.
- **Local / segment coordinates**: coordinates attached to the player (pelvis, foot, trunk). Great for “relative to the body” questions.

Why bother with local coordinates?
- A right-foot kick looks similar whether it happens near the left corner flag or the center circle.
- A “ball approaching the foot” feature should not change if you rotate the whole scene.

So we build a segment coordinate system $S$ and express vectors in that system.

If $R_{G\_S}$ is the rotation matrix whose **columns are the segment axes expressed in global coordinates**, then the component form of the same vector in the segment frame is:

$$
\mathbf{v}_S = R_{G\_S}^T\,\mathbf{v}_G
$$

The transpose appears because we are doing a **change of basis** (global → local) and $R^{-1}=R^T$ for rotation matrices.

There are two pragmatic approaches:

**Approach A — rotate global vectors into the local frame (fast)**  
If you have a segment frame orientation `R_G_S(t)` (columns = segment axes in global coordinates):

- a global vector `v_G` becomes local components  
$$
  v\_S = R\_{G\_S}^T \, v\_G
$$

This is the right operation when `v` is already a *difference vector* (e.g., velocity, acceleration, relative position).

```python
def to_local_vectors(R_G_S: np.ndarray, v_G: np.ndarray) -> np.ndarray:
    """
    R_G_S: (N,3,3)  segment axes expressed in global coords
    v_G:   (N,3)    vectors in global coords
    returns v_S: (N,3) vectors expressed in segment coords
    """
    return np.einsum("nij,nj->ni", np.transpose(R_G_S, (0,2,1)), v_G)
```

**Approach B — compute coordinates in the moving frame, then differentiate (robust)**  
If you want the **ball velocity relative to the foot**, you can:

1) compute ball coordinates in the foot frame each frame: `p_ball^foot(t)`  
2) differentiate that signal over time to get `v_ball^foot(t)` and `a_ball^foot(t)`

This automatically accounts for the fact that the foot frame is translating and rotating.

```python
def kinematics_in_moving_frame(
    P_G: np.ndarray,
    R_G_S: np.ndarray,
    o_G: np.ndarray,
    fs: float,
    window_s: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    P_G:   (N,3) global positions of some point (e.g., ball)
    R_G_S: (N,3,3) segment orientation
    o_G:   (N,3) segment origin in global coords (e.g., ankle)
    returns: (P_S, V_S, A_S) in segment coordinates
    """
    # Transform point into segment frame
    T_GS = make_T(R_G_S, o_G)      # global -> segment pose
    T_SG = invert_T(T_GS)          # segment -> global inverse
    P_S = apply_T(T_SG, P_G)       # (N,3) ball in segment coords

    # Differentiate in that coordinate system
    P_Ss, V_S, A_S = savgol_pos_vel_acc(P_S, fs=fs, window_s=window_s)
    return P_Ss, V_S, A_S
```

### 3.5 Relative kinematics between two points (or two players)

Given two tracked points `A(t)` and `B(t)`:

- relative position: `r_AB = p_B - p_A`
- relative velocity: `v_AB = v_B - v_A`
- relative acceleration: `a_AB = a_B - a_A`

```python
def relative_kinematics(pA: np.ndarray, pB: np.ndarray, vA: np.ndarray, vB: np.ndarray, aA: np.ndarray, aB: np.ndarray):
    r = pB - pA
    v = vB - vA
    a = aB - aA
    return r, v, a
```

Typical use cases:
- distance and closing speed between defender and attacker
- ball-to-foot relative speed
- relative acceleration between pelvis and ball (reaction)

### 3.6 Advanced note: time derivative in a rotating coordinate system

If a vector is expressed in a rotating frame, its time derivative has an extra term:

$$
\left(\frac{d r}{dt}\right)_{S} =
R_{G\_S}^T \left(\frac{d r}{dt}\right)_{G} - \omega_{S} \times r_{S}
$$

This matters if you want physics-correct derivatives inside rotating frames.  
In many hackathon features, **Approach B** (transform then differentiate) is the easiest way to stay consistent.

---

## 4. 3D angles without full segment frames (Level 1)

“Angle” sounds simple, but in 3D there are different meanings:

- **Angle between two vectors** (one number): fast and robust  
- **Angle in a chosen plane** (one signed number): still fast, more interpretable  
- **Full 3D joint orientation** (three numbers): requires segment frames (Level 2/3)

Level 1 focuses on the first two options because they work well even when you don’t have perfect anatomical marker sets.

### A tiny refresher: dot product = angle

For two 3D vectors $\mathbf{a}$ and $\mathbf{b}$:

$$
\mathbf{a}\cdot\mathbf{b} = |\mathbf{a}|\,|\mathbf{b}|\cos\theta
$$

Rearranging gives the classic 3D angle formula used throughout this section.

> Tip: computers sometimes produce values like 1.00000002 due to floating point rounding.  
> That’s why the code clips the cosine into [-1, 1] before calling `arccos`.

This is the fastest way to get angles.  
It uses only vectors (no rotation matrices yet).

At this level you treat bones/segments as **straight lines** between two tracked points.

Example: “thigh direction” ≈ hip → knee vector.

An angle between two 3D vectors is computed via the dot product:

$$
\mathbf{u}\cdot\mathbf{v} = \lVert\mathbf{u}\rVert\,\lVert\mathbf{v}\rVert\cos(\theta)
\quad\Rightarrow\quad
\theta = \arccos\left(\frac{\mathbf{u}\cdot\mathbf{v}}{\lVert\mathbf{u}\rVert\,\lVert\mathbf{v}\rVert}\right)
$$

**Radians vs degrees**
- The math functions in NumPy return angles in **radians**.
- If you want degrees: $\theta_{deg}=\theta_{rad}\cdot 180/\pi$.

We keep radians in most computations (because angular velocity naturally ends up in rad/s), and convert to degrees only for plotting or interpretation.

### 4.1 Knee angle (simple 3D hinge proxy)
At the knee, define:
- thigh vector: `hip - knee`
- shank vector: `ankle - knee`

Angle between those vectors:

In practice, due to floating-point rounding, the cosine value can end up slightly above 1 or below -1.  
That would make `arccos` return `nan`. The `np.clip(..., -1, 1)` prevents that.

```python
def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # u, v shape (N,3)
    uu = u / (norm(u)[:, None] + 1e-12)
    vv = v / (norm(v)[:, None] + 1e-12)
    cosang = np.clip(np.sum(uu * vv, axis=1), -1.0, 1.0)
    return np.arccos(cosang)  # radians

def knee_angle_series(parts_df: pd.DataFrame, team: int, jersey: int, side: str = "right") -> pd.Series:
    hip_id   = 13 if side == "right" else 11
    knee_id  = 15 if side == "right" else 14
    ankle_id = 17 if side == "right" else 16

    hip   = to_numpy_xyz(get_part_series(parts_df, team, jersey, hip_id))
    knee  = to_numpy_xyz(get_part_series(parts_df, team, jersey, knee_id))
    ankle = to_numpy_xyz(get_part_series(parts_df, team, jersey, ankle_id))

    u = hip - knee
    v = ankle - knee
    theta = angle_between(u, v)
    return pd.Series(theta, name=f"{side}_knee_angle_rad")
```

**What it gives**: one scalar angle (bend/extend magnitude).  
**What it does not give**: full 3-DOF knee orientation (flexion vs ab/adduction vs rotation).

### 4.2 Viewing angle to the ball (head proxy)
A practical head “forward” vector is:
- `nose - mid(ears)`

This is a simple geometric proxy:

- midpoint(ears) is a rough “head center”
- nose points roughly forward

So `nose - ear_mid` approximates the direction the head is facing.

It won’t perfectly match gaze (eyes can move independently), but it is often good enough to:
- detect quick head turns (“scans”)
- estimate whether the head is oriented towards the ball, a teammate, or space

```python
def viewing_angle_to_ball(parts_df: pd.DataFrame, ball_df: pd.DataFrame, team: int, jersey: int) -> pd.Series:
    nose = to_numpy_xyz(get_part_series(parts_df, team, jersey, 2))
    le   = to_numpy_xyz(get_part_series(parts_df, team, jersey, 1))
    re   = to_numpy_xyz(get_part_series(parts_df, team, jersey, 3))

    ear_mid = 0.5 * (le + re)
    head_fwd = nose - ear_mid

    # align ball rows by frame index (simplest: assume same frames; otherwise merge on frame)
    ball = ball_df[["bx", "by", "bz"]].to_numpy(float)
    to_ball = ball - ear_mid

    theta = angle_between(head_fwd, to_ball)
    return pd.Series(theta, name="view_angle_to_ball_rad")
```

---

## 5. Segment frames + transformation matrices (Level 2)

Level 2 is the point where you stop thinking of points as “just coordinates on the pitch” and start thinking of **moving coordinate systems**.

### What is a segment frame?

A **frame** (or “local coordinate system”) is just three perpendicular unit vectors plus an origin:

- origin $\mathbf{o}(t)$ (a 3D point)
- axes $\mathbf{a}_x(t), \mathbf{a}_y(t), \mathbf{a}_z(t)$ (3D unit vectors)

Together, these define a **pose**: position + orientation.

Why do we want this?

- Because “ball relative to foot” is often more meaningful than “ball in global pitch coordinates”.
- Because joint motion (knee, hip) is defined by how one segment rotates relative to another.

### Rotation matrix intuition

The rotation matrix

$$
R(t) = [\mathbf{a}_x(t)\ \mathbf{a}_y(t)\ \mathbf{a}_z(t)]
$$

stores those axes as **columns**. If the columns are orthonormal, then:

- $R^T R = I$
- $R^{-1} = R^T$

That last property is what makes local↔global transforms so convenient.

### Translation + rotation together: why 4×4?

A 3×3 rotation can’t represent translation. A 4×4 homogeneous transform can:

$$
T =
\begin{bmatrix}
R & \mathbf{o} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$

This lets you apply rotation and translation in one matrix multiplication.


To get **3D joint rotations** and **angular velocity**, you need segment orientation matrices.

### 5.0 From skeleton points to segments (the missing step)

Up to Level 1, it’s totally fine to treat each TF15 landmark as “just a moving 3D point”.

But the moment you want *segment axes*, *segment angular velocity*, or *joint angular velocity*, you need one extra layer of structure:

> **Segments** are rigid links between landmarks (e.g., thigh = hip→knee).  
> TF15 gives you points — **you must infer segments** from those points in a consistent way.

If you’re new to movement science, think of this like a simple robotics model:

- a **point** is a position in 3D
- a **segment** is a rigid stick between two points
- a **joint** is where two sticks meet (e.g., knee = thigh ↔ shank)

#### Two points vs three points (why this matters)

With **two points** you can compute a segment’s direction and length:

- direction vector (not unit-length yet):  
  $$
  \mathbf{u}(t)=\mathbf{p}_\text{distal}(t)-\mathbf{p}_\text{proximal}(t)
  $$
- segment length:  
  $$
  L(t)=\|\mathbf{u}(t)\|
  $$

That’s enough for:
- distances
- “knee bend proxy” angles
- simple technique features

But it is **not enough** for a full 3D rotation matrix.

Why? Because a line in 3D can still “twist” around itself.  
To define a full 3D segment **frame** you need a **plane**, which means at least **three non-colinear points**.

So in practice we will define each segment frame using:

- a **primary axis** from two points (proximal→distal)
- a **third point** to define a plane (resolves the twist around the long axis)

This is why the next subsection (5.1) uses three points `(p0, p1, p2)`.

---

### 5.0.1 The TF15 skeleton as a graph (connectivity)

It helps to visualize the skeleton as a graph:

- nodes = TF15 points (hips, knees, ankles, …)
- edges = “bones” / segment connections

A minimal connectivity list (IDs from TF15) looks like:

```python
# Node ids refer to PARTS dict defined earlier
SKELETON_EDGES = [
    # pelvis / trunk
    (12, 11), (12, 13), (12, 5),      # pelvis -> left_hip, right_hip, neck
    (5, 4), (5, 6),                   # neck -> left_shoulder, right_shoulder

    # left leg
    (11, 14), (14, 16), (16, 18), (16, 19),  # left_hip -> left_knee -> left_ankle -> (heel, toe)

    # right leg
    (13, 15), (15, 17), (17, 20), (17, 21),  # right_hip -> right_knee -> right_ankle -> (heel, toe)

    # left arm
    (4, 7), (7, 9),                    # left_shoulder -> left_elbow -> left_wrist

    # right arm
    (6, 8), (8, 10),                   # right_shoulder -> right_elbow -> right_wrist

    # head
    (5, 1), (5, 2), (5, 3),            # neck -> (left_ear, nose, right_ear)
]
```

You do *not* need this graph to compute segment frames, but it helps with:

- sanity checks (“is a knee suddenly 5 meters away from the hip?”)
- visualizations (“draw bones as lines between points”)
- defining a consistent segment list

---

### 5.0.2 Align points over time (wide table)

When you call `get_part_series(...)` for different points, you *might* get:
- different numbers of frames per part (occlusion / dropouts)
- slightly different frame indices per part

If you subtract arrays that aren’t aligned by frame, you silently create wrong vectors.

A robust pattern is:

1) filter to one player  
2) pivot into a **wide** table indexed by `frame`  
3) optionally interpolate *small* gaps (not long gaps)

```python
def player_parts_wide(parts_df: pd.DataFrame, team: int, jersey: int, part_ids: list[int] | None = None) -> pd.DataFrame:
    """
    Returns a wide table indexed by frame, with MultiIndex columns (coord, part_id).

    Example column keys:
      ('x', 12), ('y', 12), ('z', 12)   -> pelvis
      ('x', 15), ('y', 15), ('z', 15)   -> right_knee
    """
    df = parts_df[(parts_df.team == team) & (parts_df.jersey == jersey)].copy()
    if part_ids is not None:
        df = df[df.part_id.isin(part_ids)]

    wide = df.pivot(index="frame", columns="part_id", values=["x", "y", "z"]).sort_index()

    # Ensure all requested part_ids exist as columns (even if entirely NaN)
    if part_ids is None:
        part_ids = sorted(df.part_id.unique().tolist())
    full_cols = pd.MultiIndex.from_product([["x", "y", "z"], part_ids])
    wide = wide.reindex(columns=full_cols)

    return wide

def xyz_from_wide(wide: pd.DataFrame, part_id: int) -> np.ndarray:
    """Return Nx3 array aligned by frame index (may contain NaNs)."""
    cols = [("x", part_id), ("y", part_id), ("z", part_id)]
    return wide.loc[:, cols].to_numpy(dtype=float)

def interpolate_small_gaps(wide: pd.DataFrame, limit_frames: int = 3) -> pd.DataFrame:
    """
    Fill short NaN gaps only (limit in frames). For long gaps, leave NaNs.
    """
    return wide.interpolate(limit=limit_frames, limit_direction="both")
```

**Why interpolation at all?**  
Because building frames and taking derivatives both behave badly if a single frame is missing.

**Hackathon rule of thumb**
- interpolate gaps of 1–3 frames (40–120 ms at 25 Hz)
- do *not* interpolate large gaps — better to mask those windows

---

### 5.0.3 Helper points (midpoints) that make segments more stable

Some useful “virtual points” are not given directly by TF15, but are easy to compute and often more stable than a single landmark.

Examples:

- hip midpoint:  
  $$
  \mathbf{p}_{hip\_mid}=\frac{\mathbf{p}_{Lhip}+\mathbf{p}_{Rhip}}{2}
  $$
- shoulder midpoint:  
  $$
  \mathbf{p}_{shoulder\_mid}=\frac{\mathbf{p}_{Lsh}+ \mathbf{p}_{Rsh}}{2}
  $$
- ear midpoint:  
  $$
  \mathbf{p}_{ear\_mid}=\frac{\mathbf{p}_{Lear}+\mathbf{p}_{Rear}}{2}
  $$

In code, use `nanmean` so the midpoint still exists when one side is missing:

```python
def nan_midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a,b: (N,3) possibly with NaNs
    returns: (N,3) midpoint where possible
    """
    return np.nanmean(np.stack([a, b], axis=0), axis=0)

def build_point_cache(wide: pd.DataFrame) -> dict:
    """
    Collect base TF15 points + derived helper points in one dict.
    Keys are:
      - int part_id for raw points
      - str for derived points
    """
    pts: dict = {}

    # raw TF15 points by id (will contain NaNs if point missing)
    for pid in PARTS.keys():
        if ("x", pid) in wide.columns:
            pts[pid] = xyz_from_wide(wide, pid)

    # derived midpoints (safe if one side missing)
    pts["hip_mid"] = nan_midpoint(pts[11], pts[13])
    pts["shoulder_mid"] = nan_midpoint(pts[4], pts[6])
    pts["ear_mid"] = nan_midpoint(pts[1], pts[3])

    return pts
```

---

### 5.0.4 Segment recipes: which points define which segment?

Now we can explicitly define how each segment is inferred.

For limb segments (thigh, shank, upper arm, forearm), we use the same pattern:

- **p0** = proximal point
- **p1** = distal point  → defines the long axis
- **p2** = a third point  → defines a plane to resolve twist
- **origin** = where you want the segment frame located (often proximal joint)

Below is a pragmatic “technical frame” recipe set that works well with TF15.

> These are **engineering frames**: consistent and useful, not clinical anatomical frames.

| Segment | origin | p0 → p1 (long axis) | p2 (plane reference candidates) |
|---|---:|---|---|
| right_thigh | right_hip (13) | right_hip (13) → right_knee (15) | pelvis (12), left_hip (11) |
| right_shank | right_knee (15) | right_knee (15) → right_ankle (17) | right_toe (21), right_heel (20), right_hip (13) |
| right_foot  | right_ankle (17) | right_heel (20) → right_toe (21) | right_ankle (17), right_knee (15) |
| left_thigh  | left_hip (11) | left_hip (11) → left_knee (14) | pelvis (12), right_hip (13) |
| left_shank  | left_knee (14) | left_knee (14) → left_ankle (16) | left_toe (19), left_heel (18), left_hip (11) |
| left_foot   | left_ankle (16) | left_heel (18) → left_toe (19) | left_ankle (16), left_knee (14) |
| right_upperarm | right_shoulder (6) | right_shoulder (6) → right_elbow (8) | neck (5), right_wrist (10) |
| right_forearm  | right_elbow (8) | right_elbow (8) → right_wrist (10) | right_shoulder (6), neck (5) |
| left_upperarm  | left_shoulder (4) | left_shoulder (4) → left_elbow (7) | neck (5), left_wrist (9) |
| left_forearm   | left_elbow (7) | left_elbow (7) → left_wrist (9) | left_shoulder (4), neck (5) |

For **pelvis, trunk, and head**, we typically build frames from *two* meaningful directions (e.g., left-right and up) and then compute the third axis by a cross product. Those are implemented a bit differently (see 5.1.2).

To make segment definitions executable (not just a table), store them as a small dictionary:

```python
from dataclasses import dataclass
from typing import Tuple, Union

PointKey = Union[int, str]  # int = TF15 part_id, str = derived point (e.g., "hip_mid")

@dataclass(frozen=True)
class SegmentRecipe:
    name: str
    origin: PointKey
    p0: PointKey
    p1: PointKey
    p2_candidates: Tuple[PointKey, ...]


SEGMENT_RECIPES = {
    # Right leg
    "right_thigh": SegmentRecipe("right_thigh", origin=13, p0=13, p1=15, p2_candidates=(12, 11)),
    "right_shank": SegmentRecipe("right_shank", origin=15, p0=15, p1=17, p2_candidates=(21, 20, 13)),
    "right_foot":  SegmentRecipe("right_foot",  origin=17, p0=20, p1=21, p2_candidates=(17, 15)),

    # Left leg
    "left_thigh":  SegmentRecipe("left_thigh",  origin=11, p0=11, p1=14, p2_candidates=(12, 13)),
    "left_shank":  SegmentRecipe("left_shank",  origin=14, p0=14, p1=16, p2_candidates=(19, 18, 11)),
    "left_foot":   SegmentRecipe("left_foot",   origin=16, p0=18, p1=19, p2_candidates=(16, 14)),

    # Right arm
    "right_upperarm": SegmentRecipe("right_upperarm", origin=6, p0=6, p1=8,  p2_candidates=(5, 10)),
    "right_forearm":  SegmentRecipe("right_forearm",  origin=8, p0=8, p1=10, p2_candidates=(6, 5)),

    # Left arm
    "left_upperarm":  SegmentRecipe("left_upperarm",  origin=4, p0=4, p1=7, p2_candidates=(5, 9)),
    "left_forearm":   SegmentRecipe("left_forearm",   origin=7, p0=7, p1=9, p2_candidates=(4, 5)),
}
```

At this point we have a concrete answer to:

> “What *is* the thigh segment in this dataset?”  
> → it’s the hip→knee line, plus a plane reference point to define a consistent 3D frame.

Next, in 5.1, we’ll build the actual rotation matrices from these recipes.

### 5.1 Construct a local frame from 3 points (Gram–Schmidt)

Given three points `p0, p1, p2` on/near a rigid segment:

1. x-axis: from p0 to p1  
2. z-axis: perpendicular to the plane spanned by x-axis and (p2 - p0)  
3. y-axis: completes a right-handed orthonormal frame

```python
def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

def build_frame_from_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    \"\"\"
    p0,p1,p2 shape: (N,3)
    returns R shape: (N,3,3)
      columns are the unit axes [ax, ay, az] expressed in global coordinates
    \"\"\"
    ax = normalize(p1 - p0)

    # temporary vector defining the plane
    v = (p2 - p0)

    az = normalize(np.cross(ax, v))
    ay = np.cross(az, ax)  # already orthogonal by construction

    R = np.stack([ax, ay, az], axis=-1)  # (N,3,3) with columns
    return R
```


### 5.1.1 Make frames robust (fallback plane points + continuity)

In ideal lab data you might always have three clean, non-colinear markers per segment.  
In match tracking you should assume the opposite:

- some points jitter (toes/heels can be noisy)
- sometimes points are missing for a few frames
- sometimes your chosen “plane” point becomes nearly colinear with the segment axis

If you build segment frames naïvely, you may see **180° flips** of an axis from one frame to the next.  
Those flips make angular velocity explode (because a flip looks like a huge “rotation per second”).

So we add three practical safeguards:

1) **Choose the best plane reference per frame** (from a list of candidates)  
2) **Mark invalid frames** (degenerate geometry) and fill with last valid  
3) **Enforce temporal continuity** (prevent sign flips)

The code below does all three.

```python
def build_frame_from_points_candidates(
    p0: np.ndarray,
    p1: np.ndarray,
    p2_candidates: list[np.ndarray],
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized frame construction with per-frame plane-point selection.

    p0, p1: (N,3) define the long axis (p1-p0)
    p2_candidates: list of (N,3) points; we will pick the candidate that gives the
                   largest cross-product magnitude (most stable plane) per frame.

    Returns:
      R_raw: (N,3,3) rotation matrices (may contain NaNs where invalid)
      valid: (N,) boolean mask (True where a valid plane could be formed)
      choice: (N,) int indices into p2_candidates (which plane point was used per frame)
    """
    ax = normalize(p1 - p0)  # (N,3)

    # v_k = p2_k - p0  for each candidate k
    v_stack = np.stack([(p2 - p0) for p2 in p2_candidates], axis=0)  # (K,N,3)

    # cross_k = ax x v_k  (broadcast ax over K)
    cross_stack = np.cross(ax[None, :, :], v_stack)                 # (K,N,3)
    cross_norm = np.linalg.norm(cross_stack, axis=-1)               # (K,N)

    # Handle NaNs: treat them as -inf so they never win argmax
    score = np.where(np.isfinite(cross_norm), cross_norm, -np.inf)

    choice = np.argmax(score, axis=0)  # (N,)
    idx = np.arange(ax.shape[0])
    best_norm = score[choice, idx]     # (N,)
    best_cross = cross_stack[choice, idx, :]  # (N,3)

    valid = best_norm > eps

    az = np.full_like(ax, np.nan)
    az[valid] = best_cross[valid] / (best_norm[valid, None] + 1e-12)

    ay = np.cross(az, ax)
    ay = normalize(ay)

    R_raw = np.stack([ax, ay, az], axis=-1)
    return R_raw, valid, choice

def forward_fill_rotations(R_raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Replace invalid frames with the last valid rotation.
    If the series starts invalid, use identity.
    """
    R = R_raw.copy()
    last = np.eye(3)
    for i in range(R.shape[0]):
        good = bool(valid[i]) and np.isfinite(R[i]).all()
        if good:
            last = R[i]
        else:
            R[i] = last
    return R

def enforce_z_continuity(R: np.ndarray) -> np.ndarray:
    """
    Prevent 180° flips of the z-axis over time by forcing successive z-axes
    to have a positive dot product. To keep det(R)=+1, we flip BOTH y and z.
    """
    R = R.copy()
    for i in range(1, R.shape[0]):
        z_prev = R[i - 1, :, 2]
        z_curr = R[i, :, 2]
        if float(np.dot(z_prev, z_curr)) < 0.0:
            R[i, :, 1] *= -1.0  # flip y
            R[i, :, 2] *= -1.0  # flip z
    return R
```

> If you later see small numerical drift (i.e., $R^T R$ not exactly identity), you can optionally project each frame back onto $SO(3)$ with the SVD method shown in Section 10.1.

---

### 5.1.2 Turn recipes into segment poses (R(t), origin, length)

Now we can *apply* the segment recipes from 5.0.4 to build actual segment poses.

A segment “pose” is:

- origin $\mathbf{o}(t)$ (where we attach the frame)
- rotation matrix $R(t)$ (segment axes in global coordinates)
- optional: endpoints and length for QC

#### Limb segments (3-point frames)

For recipes like thigh/shank/foot/arms, we:

1) resolve `p0, p1` (long axis)  
2) resolve a list of `p2_candidates` (plane refs)  
3) build a robust $R(t)$ using the helper functions above  
4) define origin at the chosen joint point

```python
def resolve_point(pts: dict, key: PointKey) -> np.ndarray:
    if key not in pts:
        raise KeyError(f"PointKey {key!r} not found. Available keys: {list(pts.keys())[:10]} ...")
    return pts[key]

def segment_pose_from_recipe(
    pts: dict,
    recipe: SegmentRecipe,
    eps: float = 1e-6,
) -> dict:
    p0 = resolve_point(pts, recipe.p0)
    p1 = resolve_point(pts, recipe.p1)
    p2s = [resolve_point(pts, k) for k in recipe.p2_candidates]

    R_raw, valid, choice = build_frame_from_points_candidates(p0, p1, p2s, eps=eps)
    R = forward_fill_rotations(R_raw, valid)
    R = enforce_z_continuity(R)

    o = resolve_point(pts, recipe.origin)
    length = np.linalg.norm(p1 - p0, axis=1)

    return {
        "R": R,               # (N,3,3)
        "o": o,               # (N,3)
        "p0": p0, "p1": p1,   # endpoints used for the long axis
        "length": length,     # (N,)
        "valid": valid,       # (N,)
        "p2_choice": choice,  # (N,) which candidate was used per frame
    }
```

#### Pelvis / trunk / head frames (2-vector frames)

For pelvis, trunk, and head you often want axes that are easier to interpret:

- **x** = left-right  
- **z** = up  
- **y** = forward (computed by cross product so the frame is right-handed)

So we build frames from two meaningful directions and then orthonormalize:

```python
def build_frame_from_xz(x_dir: np.ndarray, z_dir: np.ndarray) -> np.ndarray:
    """
    Build a right-handed frame with:
      x = normalize(x_dir)
      z = normalize(z_dir)
      y = normalize(z x x)   (so y is roughly 'forward')
    Returns R with columns [x, y, z].
    """
    x = normalize(x_dir)
    z = normalize(z_dir)
    y = normalize(np.cross(z, x))
    x = np.cross(y, z)  # re-orthonormalize
    return np.stack([x, y, z], axis=-1)

def build_frame_from_xy(x_dir: np.ndarray, y_dir: np.ndarray) -> np.ndarray:
    """
    Build a right-handed frame with:
      x = normalize(x_dir)
      y = normalize(y_dir)
      z = normalize(x x y)
    Returns R with columns [x, y, z].
    """
    x = normalize(x_dir)
    y = normalize(y_dir)
    z = normalize(np.cross(x, y))
    y = np.cross(z, x)  # re-orthonormalize
    return np.stack([x, y, z], axis=-1)

def fix_forward_sign(R: np.ndarray, ref_fwd: np.ndarray) -> np.ndarray:
    """
    Ensure the forward axis (column 1) points roughly in the same direction
    as a reference forward vector ref_fwd (e.g., nose - hip_mid).

    If dot(forward, ref_fwd) < 0, flip x and y (a 180° yaw flip), keeping det(R)=+1.
    """
    R = R.copy()
    fwd = R[:, :, 1]
    s = np.sum(fwd * ref_fwd, axis=1)
    flip = s < 0
    R[flip, :, 0] *= -1.0  # flip x
    R[flip, :, 1] *= -1.0  # flip y
    return R

def fix_up_sign(R: np.ndarray, up_global: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """
    Ensure the up axis (column 2) points roughly upward in global coordinates.
    If not, flip x and z to keep det(R)=+1.
    """
    R = R.copy()
    up = R[:, :, 2]
    s = np.sum(up * up_global[None, :], axis=1)
    flip = s < 0
    R[flip, :, 0] *= -1.0  # flip x
    R[flip, :, 2] *= -1.0  # flip z
    return R

def pelvis_pose(pts: dict) -> dict:
    left_hip = pts[11]
    right_hip = pts[13]
    pelvis = pts[12]
    neck = pts[5]
    nose = pts[2]

    x_dir = right_hip - left_hip
    z_dir = neck - pelvis
    R_raw = build_frame_from_xz(x_dir, z_dir)

    ref_fwd = nose - pts["hip_mid"]
    R_raw = fix_forward_sign(R_raw, ref_fwd)

    valid = np.isfinite(R_raw).reshape(R_raw.shape[0], -1).all(axis=1)
    R = forward_fill_rotations(R_raw, valid)
    R = enforce_z_continuity(R)

    return {"R": R, "o": pelvis, "valid": valid}

def trunk_pose(pts: dict) -> dict:
    left_sh = pts[4]
    right_sh = pts[6]
    pelvis = pts[12]
    neck = pts[5]
    nose = pts[2]

    x_dir = right_sh - left_sh
    z_dir = neck - pelvis
    R_raw = build_frame_from_xz(x_dir, z_dir)

    ref_fwd = nose - pts["shoulder_mid"]
    R_raw = fix_forward_sign(R_raw, ref_fwd)

    valid = np.isfinite(R_raw).reshape(R_raw.shape[0], -1).all(axis=1)
    R = forward_fill_rotations(R_raw, valid)
    R = enforce_z_continuity(R)

    return {"R": R, "o": pts["shoulder_mid"], "valid": valid}

def head_pose(pts: dict) -> dict:
    left_ear = pts[1]
    right_ear = pts[3]
    nose = pts[2]

    x_dir = right_ear - left_ear
    y_dir = nose - pts["ear_mid"]  # forward
    R_raw = build_frame_from_xy(x_dir, y_dir)
    R_raw = fix_up_sign(R_raw)

    valid = np.isfinite(R_raw).reshape(R_raw.shape[0], -1).all(axis=1)
    R = forward_fill_rotations(R_raw, valid)
    R = enforce_z_continuity(R)

    return {"R": R, "o": pts["ear_mid"], "valid": valid}
```

#### One convenience function: build a full segment model for one player

This wrapper gives you a consistent starting point for Level 2 and Level 3 analyses:

```python
def build_segment_model_for_player(
    parts_df: pd.DataFrame,
    team: int,
    jersey: int,
    interpolate_limit_frames: int = 3,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict, dict]:
    """
    Returns:
      frames: (N,) frame numbers
      seg: dict mapping segment name -> pose dict with R,o,valid,(length,...)
      pts: dict of raw+derived points (each Nx3)
    """
    part_ids = sorted(PARTS.keys())
    wide = player_parts_wide(parts_df, team, jersey, part_ids=part_ids)
    wide = interpolate_small_gaps(wide, limit_frames=interpolate_limit_frames)

    pts = build_point_cache(wide)

    seg: dict = {}
    seg["pelvis"] = pelvis_pose(pts)
    seg["trunk"]  = trunk_pose(pts)
    seg["head"]   = head_pose(pts)

    for name, recipe in SEGMENT_RECIPES.items():
        seg[name] = segment_pose_from_recipe(pts, recipe, eps=eps)

    frames = wide.index.to_numpy()
    return frames, seg, pts
```

**How you use this later**

- Segment angular velocity:  
  `omega = omega_from_R(seg["right_shank"]["R"], dt)`  
- Joint rotation:  
  `R_knee = relative_rotation(seg["right_thigh"]["R"], seg["right_shank"]["R"])`  
- Joint angular velocity:  
  `omega_knee = omega_from_R(R_knee, dt)`  

This makes the “points → segments → frames → angular velocity” pathway explicit and repeatable for any team.

### 5.2 Homogeneous transforms (4×4)

A pose transform combines rotation + translation:

- `R`: 3×3 rotation matrix
- `o`: 3×1 origin/translation

```python
def make_T(R: np.ndarray, o: np.ndarray) -> np.ndarray:
    """
    R: (N,3,3) or (3,3)
    o: (N,3)   or (3,)
    returns T: (N,4,4) or (4,4)
    """
    R = np.asarray(R)
    o = np.asarray(o)
    if R.ndim == 2:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = o
        return T
    else:
        N = R.shape[0]
        T = np.repeat(np.eye(4)[None, :, :], N, axis=0)
        T[:, :3, :3] = R
        T[:, :3,  3] = o
        return T

def invert_T(T: np.ndarray) -> np.ndarray:
    """Inverse of rigid transform."""
    T = np.asarray(T)
    if T.ndim == 2:
        R = T[:3, :3]
        o = T[:3, 3]
        Ti = np.eye(4)
        Ti[:3, :3] = R.T
        Ti[:3, 3] = -R.T @ o
        return Ti
    else:
        R = T[:, :3, :3]
        o = T[:, :3, 3]
        Ti = np.repeat(np.eye(4)[None, :, :], T.shape[0], axis=0)
        Ti[:, :3, :3] = np.transpose(R, (0,2,1))
        Ti[:, :3, 3] = -np.einsum("nij,nj->ni", np.transpose(R, (0,2,1)), o)
        return Ti

def apply_T(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Apply transform to points.
    T: (4,4) or (N,4,4)
    P: (M,3) or (N,3)  (broadcast rules apply if you expand dims)
    """
    T = np.asarray(T)
    P = np.asarray(P)
    if T.ndim == 2:
        Ph = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)  # (M,4)
        out = (T @ Ph.T).T
        return out[:, :3]
    else:
        # assume P is (N,3)
        Ph = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)  # (N,4)
        out = np.einsum("nij,nj->ni", T, Ph)  # (N,4)
        return out[:, :3]
```

### 5.3 “Relative position” in another segment’s coordinates
Example: ball position expressed in the foot frame:

```python
def ball_in_foot_frame(ball_pos: np.ndarray, R_foot: np.ndarray, o_foot: np.ndarray) -> np.ndarray:
    T_G_foot = make_T(R_foot, o_foot)
    T_foot_G = invert_T(T_G_foot)
    return apply_T(T_foot_G, ball_pos)
```

This is extremely useful for:
- kick/impact modeling
- “ball contact point” features
- pass reception in the player’s own frame

---

## 6. Joint rotation matrices (Level 2 → 3)
Once you have segment frames, you can describe joint motion without ambiguous “clinical” terminology.

### “Relative rotation” explained in plain words

- $R_{G,P}(t)$: orientation of the **proximal** segment (e.g., thigh) in global coordinates  
- $R_{G,D}(t)$: orientation of the **distal** segment (e.g., shank) in global coordinates  

To get “how the shank is oriented relative to the thigh”, you *remove* the thigh orientation:

$$
R_{P\rightarrow D}(t) = R_{G,P}(t)^T\,R_{G,D}(t)
$$

That transpose is not a random trick — it’s the inverse of a rotation matrix.

### Why you may not need Euler angles at all

Euler/Cardan angles are popular because they give three interpretable numbers (flexion, ab/adduction, rotation).  
But:

- they depend on a chosen axis order
- they can suffer from gimbal lock
- differentiating them can be noisy

In a hackathon, it’s often better to compute **angular velocity directly from the rotation matrix** (next section) and only compute Euler angles if you really need the three-angle breakdown.
Given:
- proximal segment orientation `R_G_P`
- distal segment orientation `R_G_D`

The relative rotation (joint) is:

### What “relative rotation” means

If $R_{G\_P}$ tells you how the proximal segment is oriented in global space, and $R_{G\_D}$ does the same for the distal segment, then:

- $R_{G\_P}^T$ converts global vectors into proximal-segment coordinates
- multiplying by $R_{G\_D}$ then expresses distal axes in proximal coordinates

So:

$$
R_{P\rightarrow D} = R_{G\_P}^T\,R_{G\_D}
$$

This is a core move in 3D kinematics: it turns “two absolute orientations” into “one joint orientation”.

> **Order matters**: rotations do not commute.  
> $R_{G\_P}^T R_{G\_D} \neq R_{G\_D} R_{G\_P}^T$.

```python
def relative_rotation(R_G_P: np.ndarray, R_G_D: np.ndarray) -> np.ndarray:
    # R_P_D = R_G_P^T @ R_G_D
    return np.einsum("nij,njk->nik", np.transpose(R_G_P, (0,2,1)), R_G_D)
```

Example: knee (thigh → shank):

- `R_G_thigh`: built from hip, knee, pelvis (or hip/knee + another point)
- `R_G_shank`: built from knee, ankle, toe (or knee/ankle + another point)

Then:
- `R_knee = R_G_thigh^T @ R_G_shank`

### 6.1 Joint angles (3 angles) from a rotation matrix

A joint rotation matrix `R_joint(t)` contains the full 3D orientation relationship.  
Sometimes you want **three interpretable angles** (e.g., flex/extend, abd/add, int/ext rotation).

Behind the scenes, this means **decomposing** a rotation matrix into three sequential rotations (Euler/Cardan/Tait–Bryan angles).

That gives you numbers that humans like (e.g., “knee flexion 60°”), but it comes with trade-offs:

- the same matrix can be decomposed in **multiple valid ways** (different axis orders)
- some poses cause **gimbal lock** (one angle becomes ambiguous)
- “flexion axis” depends on how you defined the segment frames in the first place

So treat Euler angles as:
- great for visualization and descriptive features
- something you should keep convention-consistent, not “absolute truth”

A practical approach in Python is to use SciPy’s `Rotation` class. You must choose:

- the **axis sequence** (e.g., `"xyz"`, `"zyx"`, …)
- whether the sequence is **intrinsic** (rotations about moving axes) or **extrinsic** (about fixed axes)

For hackathon prototypes, be explicit and consistent; do not mix conventions.

```python
from scipy.spatial.transform import Rotation as SciRot

def rotation_matrices_to_euler(R: np.ndarray, seq: str = "xyz", degrees: bool = True) -> np.ndarray:
    """
    R: (N,3,3)
    returns angles: (N,3) in chosen sequence
    """
    rot = SciRot.from_matrix(R)
    ang = rot.as_euler(seq, degrees=degrees)
    return ang
```

Example: knee angles using a chosen sequence:

```python
def knee_euler_angles(parts_df: pd.DataFrame, meta: MatchMeta, team: int, jersey: int, seq: str = "xyz") -> pd.DataFrame:
    dt = 1.0 / meta.framerate

    R_thigh, _, R_shank, _ = segment_frames_right_leg(parts_df, team, jersey)
    R_knee = relative_rotation(R_thigh, R_shank)

    ang = rotation_matrices_to_euler(R_knee, seq=seq, degrees=True)
    return pd.DataFrame({
        f"{seq}_a_deg": ang[:, 0],
        f"{seq}_b_deg": ang[:, 1],
        f"{seq}_c_deg": ang[:, 2],
    })
```

**Important**  
Euler/Cardan angles can suffer from **gimbal lock** near certain poses.  
If you need a robust “single-number” orientation change, prefer:

- axis–angle / rotation vector (`as_rotvec`), or  
- angular velocity `ω(t)`.

---

## 7. 3D angular velocity from rotation matrices (Level 3)

Angular velocity is a **3D vector** (rad/s), not “degrees per second”.

A beginner-friendly interpretation:

- The direction of $\boldsymbol{\omega}$ is the **instantaneous rotation axis**.
- The magnitude $\lVert\boldsymbol{\omega}\rVert$ is the **rotation speed** in **radians per second**.

If you know angular velocity from everyday life:
- a door rotates about the hinge axis
- a spinning ball rotates about its spin axis

In 3D, joints and segments can rotate about an axis that changes over time. That is why $\boldsymbol{\omega}(t)$ is a vector.

**Why not just “differentiate joint angles”?**  
Because 3D rotations do not behave like ordinary numbers. Differentiating Euler angles can work for small motions, but it is sensitive to axis conventions and gimbal lock. Computing $\boldsymbol{\omega}$ directly from $R(t)$ avoids those issues.

### 7.1 The core formula

Given a rotation matrix time series `R(t)`:

1. Compute the time derivative `Rdot(t)`  
2. Compute the skew-symmetric matrix

$$
\Omega(t) = \dot{R}(t) R(t)^{T}
$$

3. Extract angular velocity vector `ω` from `Ω`:

$$
\Omega = \begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}
\Rightarrow
\omega = \begin{bmatrix}
\Omega_{32} \\
\Omega_{13} \\
\Omega_{21}
\end{bmatrix}
$$

### 7.2 Implementation

There are two numerical steps here:

1) approximate the derivative $\dot{R}(t)$ from discrete samples  
2) multiply matrices to get $\Omega(t)=\dot{R}(t)R(t)^T$

The derivative step is where noise can enter. Central differences are a good default:

$$
\dot{R}(t_k) \approx \frac{R(t_{k+1})-R(t_{k-1})}{2\Delta t}
$$

At the first/last sample you fall back to forward/backward difference.

In practice, you often get cleaner results by smoothing the **marker positions** before you build $R(t)$.

```python
def central_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """Central difference for equally spaced samples. Keeps shape."""
    arr = np.asarray(arr)
    out = np.empty_like(arr)
    out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
    out[0] = (arr[1] - arr[0]) / dt
    out[-1] = (arr[-1] - arr[-2]) / dt
    return out

def omega_from_R(R: np.ndarray, dt: float) -> np.ndarray:
    """
    R shape: (N,3,3)
    returns omega shape: (N,3) in global coordinates, rad/s
    """
    Rdot = central_diff(R, dt)  # (N,3,3)

    Rt = np.transpose(R, (0,2,1))
    Omega = np.einsum("nij,njk->nik", Rdot, Rt)  # Ω = Rdot * R^T

    omega = np.stack([
        Omega[:, 2, 1],  # Ω_32
        Omega[:, 0, 2],  # Ω_13
        Omega[:, 1, 0],  # Ω_21
    ], axis=1)

    return omega
```

### 7.3 Angular acceleration

```python
def alpha_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return central_diff(omega, dt)
```

### 7.4 Discrete-time alternative (often more stable): incremental rotation vectors

For sampled data, a numerically stable way to approximate angular velocity is:

Instead of differentiating $R(t)$ directly, you look at **how much the segment rotated from one frame to the next**.

This is often more stable when:
- the sampling rate is modest (25–30 Hz)
- your $R(t)$ has small frame-to-frame jitter
- you care about short windows around impacts

The idea: the relative rotation between consecutive frames is a small rotation, which you can represent as an axis–angle “rotation vector”.

1) Compute the incremental relative rotation between frames:

$$
R_{\Delta}(k) = R(k)^{T} R(k+1)
$$

2) Convert that small rotation to a **rotation vector** `rotvec` (axis × angle):

$$
\text{rotvec}(k) = \theta(k) \; \hat{n}(k)
$$

3) Angular velocity is then approximately:

$$
\omega(k) \approx \frac{\text{rotvec}(k)}{\Delta t}
$$

SciPy can do the matrix → rotvec conversion safely:

```python
from scipy.spatial.transform import Rotation as SciRot

def omega_from_R_incremental(R: np.ndarray, dt: float) -> np.ndarray:
    """
    R: (N,3,3)
    returns omega: (N,3) rad/s (last sample repeated)
    """
    R_rel = np.einsum("nij,njk->nik", np.transpose(R[:-1], (0,2,1)), R[1:])  # (N-1,3,3)
    rotvec = SciRot.from_matrix(R_rel).as_rotvec()  # (N-1,3), rad
    omega = rotvec / dt
    omega = np.vstack([omega, omega[-1]])  # pad to length N
    return omega
```

If you run into noisy `Rdot`, try this method.

---

## 8. Knee angular velocity before a kick (worked recipe)

This section ties everything together:

1) detect a kick time `t_contact`  
2) build thigh and shank frames  
3) compute knee relative rotation `R_knee(t)`  
4) compute knee angular velocity `ω_knee(t)`  
5) summarize in a pre-contact window

**What are we actually measuring?**

The knee does not “have” an orientation by itself — it is a relationship between two segments:

- **thigh** (proximal segment)
- **shank** (distal segment)

So “knee angular velocity” is:

> how fast the shank frame is rotating relative to the thigh frame

In this primer we compute that by:
1) building $R_{G\_\text{thigh}}(t)$ and $R_{G\_\text{shank}}(t)$  
2) forming the relative rotation $R_{\text{knee}}(t)=R_{G\_\text{thigh}}(t)^T R_{G\_\text{shank}}(t)$  
3) extracting $\omega_{\text{knee}}(t)$ from $R_{\text{knee}}(t)$

Finally we summarize $\omega_{\text{knee}}(t)$ in a window just **before** contact, because that is where technique differences show up.

### 8.1 Detecting a kick moment (ball + foot)

A robust hackathon detector uses two signals:

- **foot–ball distance** becomes minimal
- **ball acceleration** spikes (ball velocity already in m/s)

Intuition:

- The ball can travel close to a foot without contact (e.g., a near miss).
- A real touch usually causes a **change in ball velocity**, which appears as an acceleration spike.
- Combining both reduces false detections.

This is not perfect (deflections, headers, bounces, multiple contacts), but it is a strong baseline for hackathon prototyping.

```python
def detect_kick_frame(
    parts_df: pd.DataFrame,
    ball_df: pd.DataFrame,
    meta: MatchMeta,
    team: int,
    jersey: int,
    side: str = "right",
    search_radius_m: float = 0.5,
) -> Optional[int]:
    """
    Candidate kick/contact frame from:
      - toe/foot proximity to ball (distance minimum)
      - ball acceleration spike (from provided velocity if available)

    Returns: frame number or None
    """
    fs = meta.framerate
    dt = 1.0 / fs

    toe_id = 21 if side == "right" else 19
    toe_df = get_part_series(parts_df, team, jersey, toe_id)

    merged = pd.merge(
        toe_df[["frame", "x", "y", "z"]],
        ball_df[["frame", "bx", "by", "bz", "bvx", "bvy", "bvz"]],
        on="frame",
        how="inner",
    )
    if merged.empty:
        return None

    toe = merged[["x", "y", "z"]].to_numpy(float)
    ball = merged[["bx", "by", "bz"]].to_numpy(float)
    d = np.linalg.norm(toe - ball, axis=1)

    # Ball acceleration magnitude from provided velocity (preferred)…
    vball = merged[["bvx", "bvy", "bvz"]].to_numpy(float)
    if np.isfinite(vball).all():
        aball = central_diff(vball, dt)
    else:
        # …or fall back to numerical differentiation of ball positions
        _, v_est, aball = savgol_pos_vel_acc(ball, fs=fs, window_s=0.15, poly=3)

    a_mag = np.linalg.norm(aball, axis=1)

    # Candidates: close enough to ball
    cand = np.where(d <= search_radius_m)[0]
    if cand.size == 0:
        return None

    # Pick the candidate with the largest acceleration spike, break ties by minimal distance
    best = cand[np.argmax(a_mag[cand])]
    # (optional) if several frames have similar a_mag, refine by minimal distance in a local window
    return int(merged.iloc[int(best)]["frame"])
```

Practical improvements (if you have time):
- evaluate both feet and keep the best candidate
- keep the **top-k** candidates (crosses may include multiple touches)
- require ball speed increase after the event (for real kicks)

### 8.2 Build thigh and shank frames (right leg example)

**Thigh frame**: use hip (13), knee (15), pelvis (12)  
**Shank frame**: use knee (15), ankle (17), toe (21)

These choices are pragmatic:

- hip→knee gives a strong axis along the thigh
- pelvis provides a third point to define a plane (helps stabilize the thigh frame)
- knee→ankle gives a strong axis along the shank
- toe helps define shank/foot plane direction

If you swap the third point, you may rotate the frame about the long axis. That will change the *components* of $\omega$ but usually not the magnitude $\lVert\omega\rVert$. Document your choice so your team is consistent.

```python
def segment_frames_right_leg(parts_df: pd.DataFrame, team: int, jersey: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hip   = to_numpy_xyz(get_part_series(parts_df, team, jersey, 13))
    knee  = to_numpy_xyz(get_part_series(parts_df, team, jersey, 15))
    pelvis= to_numpy_xyz(get_part_series(parts_df, team, jersey, 12))
    ankle = to_numpy_xyz(get_part_series(parts_df, team, jersey, 17))
    toe   = to_numpy_xyz(get_part_series(parts_df, team, jersey, 21))

    R_thigh = build_frame_from_points(p0=hip,  p1=knee,  p2=pelvis)
    R_shank = build_frame_from_points(p0=knee, p1=ankle, p2=toe)

    # choose origins (for transforms, if needed)
    o_thigh = hip
    o_shank = knee

    return R_thigh, o_thigh, R_shank, o_shank
```

### 8.3 Knee angular velocity (relative rotation method)

```python
def knee_angular_velocity(parts_df: pd.DataFrame, meta: MatchMeta, team: int, jersey: int, side: str = "right") -> pd.DataFrame:
    if side != "right":
        raise NotImplementedError("Left side variant: change part IDs (11/14/16/19/18).")

    dt = 1.0 / meta.framerate
    R_thigh, _, R_shank, _ = segment_frames_right_leg(parts_df, team, jersey)

    # relative knee rotation
    R_knee = relative_rotation(R_thigh, R_shank)  # (N,3,3)

    # angular velocity of the relative rotation
    omega_knee_global = omega_from_R(R_knee, dt)  # (N,3)

    # express ω in thigh coordinates (often more interpretable)
    omega_knee_thigh = np.einsum("nij,nj->ni", np.transpose(R_thigh, (0,2,1)), omega_knee_global)

    df = pd.DataFrame({
        "omega_x_thigh": omega_knee_thigh[:, 0],
        "omega_y_thigh": omega_knee_thigh[:, 1],
        "omega_z_thigh": omega_knee_thigh[:, 2],
        "omega_mag": np.linalg.norm(omega_knee_thigh, axis=1),
    })
    return df
```

### 8.4 Summarize “pre-kick” knee angular velocity

Once you have a time series (speed, angle, $\omega$, …) you usually want a **compact feature** per event.

A typical pre-contact window is 200–400 ms, because:
- it is long enough to capture the swing phase dynamics
- it is short enough to remain “about the kick”, not the run-up

Common summaries:
- mean (overall level)
- max (peak intensity)
- 95th percentile (robust “near-peak”)

```python
def summarize_pre_event(signal: np.ndarray, frames: np.ndarray, fs: float, event_frame: int, window_s: float = 0.3) -> dict:
    """Summarize a signal in the window [event - window, event)."""
    dt = 1.0 / fs
    w = int(round(window_s / dt))
    idx = np.where(frames == event_frame)[0]
    if len(idx) == 0:
        return {}
    i0 = int(idx[0])
    a = max(0, i0 - w)
    b = i0  # exclude event frame
    seg = signal[a:b]
    if seg.size == 0:
        return {}
    return {
        "mean": float(np.nanmean(seg)),
        "max": float(np.nanmax(seg)),
        "p95": float(np.nanpercentile(seg, 95)),
    }
```

---

## 9. More football-relevant 3D angle + angular velocity analyses (Level 1 → 3)

Once you can compute `R(t)` and `ω(t)`, the feature space explodes.  
Below are ideas that are realistic within a hackathon build.

A practical way to make these ideas implementable quickly is to standardize a small set of reusable primitives:

- `p_part(t)`: position of a body part
- `v_part(t), a_part(t)`: derived from smoothing + differentiation
- `R_segment(t)`: segment frame from three points
- `omega_segment(t)`: angular velocity from `R_segment`
- `R_joint(t) = R_prox^T R_dist`, `omega_joint(t)`: relative rotations and joint angular velocity
- “event windows”: align everything to a detected moment (touch, cut, receive)

Once you have those, most features become simple combinations, thresholds, or time-to-peak metrics.

### 9.1 Kicking technique (ball striking, crossing, shooting)

**(A) Proximal-to-distal sequencing**
Compute peak angular velocities and timing for:
- pelvis yaw rate
- hip/thigh angular speed
- knee relative angular speed
- ankle/foot angular speed

Then compare:
- time lag between peaks
- relation to ball exit speed

**(B) Foot orientation at impact**
At `t_contact`, take the foot “forward” axis (e.g., toe direction) and compute:
- angle between foot forward axis and ball velocity vector
- angle between foot plane normal and ball velocity (slice vs chip)

**(C) “Ankle lock” proxy**
Track angular velocity magnitude of the foot segment just before impact:
- low foot angular velocity + high ball speed may indicate a “locked” strike
- high foot angular velocity could indicate flicks / chips / toe pokes

### 9.2 Receiving and first touch

- torso facing angle to incoming ball trajectory
- head angular velocity (scan) in the 1–2 s before receiving
- pelvis–trunk dissociation: relative angular velocity between pelvis frame and trunk frame

### 9.3 Turning, feints, and 1v1 actions

- pelvis yaw rate + lateral acceleration (cut severity)
- trunk vs pelvis yaw (deception)
- head yaw rate vs pelvis yaw rate (is the player “selling” a feint with head/shoulders?)

### 9.4 Pressing and defensive actions

- acceleration bursts toward ball carrier
- braking intensity (negative tangential acceleration)
- “re-orientation speed”: time for pelvis facing direction to align with run direction

### 9.5 Heading / aerial duels (if ball Z and body Z are informative)

- neck/head angular velocity spikes around ball height peaks
- torso pitch angular velocity during jump take-off and landing

---

## 10. Quality control and debugging (do this early)

In kinematics pipelines, most “weird results” come from a small set of issues:

- unit mismatch (cm vs m)
- missing frames / misaligned merges (ball vs player)
- noisy points → unstable frames → exploding derivatives
- accidental axis flips (right-handed vs left-handed frames)

The checks below are quick to implement and can save hours of debugging.

### 10.1 Check that your rotation matrices stay valid
A rotation matrix must satisfy:
- `R.T @ R ≈ I`
- `det(R) ≈ +1`

If noise and differentiation produce drift, project back onto SO(3) using SVD:

```python
def project_to_so3(R: np.ndarray) -> np.ndarray:
    """Project a noisy 3x3 matrix onto the nearest proper rotation matrix."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn
```

For a time series `R[t]`, apply per frame (it is not free, but it stabilizes ω).

### 10.2 Unit sanity checks
- Ball speed from provided `(bvx,bvy,bvz)` should be in a human-ball range (0–40 m/s typical).
- Player pelvis speed should rarely exceed ~10–12 m/s.
- If you see 50–100 m/s for players: unit mismatch (cm vs m).

---

## 11. Appendix: segment frame recipes (pragmatic)

The TRACAB skeleton does not provide full marker clusters.  
Frames below are “technical frames”: consistent, usable, but not clinical-grade anatomical frames.

In biomechanics research you often see “anatomical coordinate systems” defined by bony landmarks and international standards.  
In a match-tracking skeleton, we rarely have enough points for that.

So we use **technical frames** instead:

- they are defined directly from the available points (hips, shoulders, nose, …)
- they are repeatable and computationally cheap
- they may not correspond 1:1 to anatomical flexion/extension axes — but they are still excellent for *relative* features and pattern detection

The most important rule is: **pick one definition and stick to it** across players and frames.

### 11.1 Pelvis frame (example)
- origin: pelvis (12)
- left-right axis: `right_hip - left_hip`
- up axis: `neck - pelvis`
- forward axis: `left-right × up` (choose sign convention and keep it consistent)

### 11.2 Trunk frame (example)
- origin: neck (5) or midpoint shoulders
- left-right: `right_shoulder - left_shoulder`
- up: `neck - pelvis`
- forward: `left-right × up`

### 11.3 Head frame (example)
- origin: midpoint ears
- left-right: `right_ear - left_ear`
- forward: `nose - midpoint ears`
- up: `left-right × forward`

---

## 12. Appendix: glossary

- **R (rotation matrix)**: a 3×3 matrix; columns are the local unit axes expressed in global coordinates.
- **T (homogeneous transform)**: a 4×4 matrix combining rotation + translation.
- **Relative rotation**: `R_rel = R_prox^T @ R_dist`.
- **Angular velocity (ω)**: 3D vector (rad/s) describing instantaneous rotation axis and rate.
- **Ω (skew matrix)**: `Ω = Rdot @ R^T`, contains ω in a cross-product form.

---

## 13. Suggested “starter tasks” for teams

1) Load match → flatten to `parts_df`, `ball_df`  
2) Pick 2–3 players → compute pelvis speed/acceleration curves  
3) Detect 20 “ball contact candidates” (distance minima + ball accel)  
4) For each candidate:
   - compute foot speed and foot orientation  
   - compute knee angular velocity pre-contact  
5) Visualize in 2D+3D:
   - ball trajectory  
   - foot and pelvis facing arrows  
   - ω-magnitude time series aligned to contact

---

