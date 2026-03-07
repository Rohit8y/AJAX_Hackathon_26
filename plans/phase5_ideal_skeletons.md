# Phase 5: Ideal WCS=100 Skeleton Generation

## Context

The kinematics pipeline (Phases 1–4) is complete. We have `data/shot_kinematics.pkl`
with real angular velocity time series and WhipChain scores for all 25 shots.

**Goal:** For each shot, generate a `(N_frames × 21 body_parts × 3 xyz)` ideal position
array where the kicking leg's rotation matrices have been morphed so that:
- The cascade ordering is correct: `pelvis → hip → knee → foot`
- The foot/pelvis amplitude ratio ≥ 3

This gives coaches a "what would a perfect kick look like" reference for each player.

---

## WCS=100 Requirements

From `whipchain_score()` in `utils/kinematics.py`:
- `sequence_score = 1.0` → all 3 cascade gaps positive: `t_pelvis < t_hip < t_knee < t_foot`
- `amp_score = 1.0` → `peak_omega_foot / peak_omega_pelvis ≥ 3.0`
- Combined: `round((0.6×1.0 + 0.4×1.0)×100) = 100`

---

## Inputs / Outputs

| File | Direction | Description |
|---|---|---|
| `data/shot_kinematics.pkl` | Read | Real angular velocities + peak times per shot |
| `data/shots_parts.parquet` | Read | Real 3D skeleton positions (frame × player × body_part) |
| `data/shot_detections.csv` | Read | Shot metadata: shooter jersey, kicking side, shot_frame |
| `data/ideal_skeletons.pkl` | **Write** | List of 25 dicts (one per shot) |
| `data/ideal_skeletons_summary.csv` | **Write** | Scalar summary for quick inspection |

No existing files are modified.

---

## Algorithm

### Step A: Ideal Peak Time Assignment

For each shot with real peak times `[tp, th, tk, tf]`:
1. Sort all 4 times ascending: `[s0, s1, s2, s3]`
2. Assign in correct cascade order: `tp_ideal=s0, th_ideal=s1, tk_ideal=s2, tf_ideal=s3`
3. Enforce minimum 1-frame gap (0.04 s) between consecutive ideal times

This reassigns **which rotational event belongs to which segment** without changing
when the overall kick motion happens. If already ordered, no change needed.

### Step B: Ideal Amplitude Targets

- Keep `peak_omega_pelvis` unchanged (pelvis is the root; its amplitude is real)
- Require: `peak_omega_foot_ideal ≥ 3.0 × peak_omega_pelvis`
- Scale factor for foot: `k_foot = max(1.0, 3.0 × peak_omega_pelvis / peak_omega_foot)`
- Scale factor for shank: `k_shank = 1.0 + (k_foot - 1.0) × 0.5` (interpolated)
- Pelvis and thigh: only time-warped, not amplitude-scaled

### Step C: Rotation Matrix Time Warping (SLERP)

For each segment (pelvis, thigh, shank, foot):
1. Build `Slerp` interpolant from the real `(N, 3, 3)` rotation matrices over the real time axis `t`
2. Define piecewise-linear time warp `f(t)` mapping output time → source time, anchored at the ideal peak time:
   - `f(t) = t` for `t < PRE_T_START` or `t > PRE_T_END` (identity outside window)
   - Before peak: `f(t) = t_start + (t - t_start) × (t_peak_real - t_start) / (t_peak_ideal - t_start)`
   - After peak: `f(t) = t_peak_real + (t - t_peak_ideal) × (t_end - t_peak_real) / (t_end - t_peak_ideal)`
3. Clamp `f(t)` to `[t[0], t[-1]]` and evaluate `Slerp(f(t))`

### Step D: Amplitude Scaling (foot and shank)

For foot (and shank by `k_shank`):
1. `R_mean = Rotation.from_matrix(R_warped).mean().as_matrix()`
2. `R_rel[i] = R_mean.T @ R_warped[i]` (deviation from mean)
3. `R_rel_scaled[i] = Rotation.from_rotvec(Rotation.from_matrix(R_rel[i]).as_rotvec() × k).as_matrix()`
4. `R_scaled[i] = R_mean @ R_rel_scaled[i]`

### Step E: Forward Kinematics Reconstruction (kicking leg only)

Using modified `R_thigh_ideal`, `R_shank_ideal`, `R_foot_ideal`:

```
# Bone lengths (mean across frames from original data)
L_thigh = mean(||knee_orig - hip_orig||)
L_shank = mean(||ankle_orig - knee_orig||)
L_foot  = mean(||toe_orig - heel_orig||)

# Heel offset in shank body frame (approx. constant)
heel_in_shank = mean(R_shank_orig[i].T @ (heel_orig[i] - ankle_orig[i]))

# Forward kinematics chain (HIP unchanged from original)
KNEE  = HIP + R_thigh_ideal[:, :, 1] * L_thigh
ANKLE = KNEE + R_shank_ideal[:, :, 1] * L_shank
HEEL  = ANKLE + einsum("nij,j->ni", R_shank_ideal, heel_in_shank)
TOE   = HEEL + R_foot_ideal[:, :, 1] * L_foot
```

Only 4 body parts change (kicking KNEE, ANKLE, HEEL, TOE). All other 17 stay real.

### Step F: Smooth Reconstructed Positions

Apply `savgol_filter(window=5, poly=2)` to the 4 reconstructed positions to remove
numerical artifacts at warp boundaries.

### Step G: Verification

Re-run the full kinematics pipeline on `ideal_pts` to compute `wcs_ideal`.
- Accept if `wcs_ideal ≥ 90` (minor numerical tolerances expected)
- Log any shots where verification fails

### Shortcut for WCS=100 shots

If `wcs_original == 100`: copy `original_pts` → `ideal_pts` directly.
No modification needed; `time_warped=False, amplitude_scaled=False, scale_factor_foot=1.0`.

---

## Output Schema

### `ideal_skeletons.pkl` — list of 25 dicts:

```python
{
    # Identity
    "event_id":       int,
    "match_time":     str,
    "shooter_jersey": int,
    "kicking_side":   str,

    # Scores
    "wcs_original":   int,
    "wcs_ideal":      int,   # target ≥ 90

    # Time axis
    "t":              np.array (N,),       # seconds, t=0 at contact

    # Full 3D skeleton time series
    "original_pts":   np.array (N, 21, 3), # real data, body_part order 1–21 (index = id-1)
    "ideal_pts":      np.array (N, 21, 3), # reconstructed ideal, same order

    # Ideal peak times (seconds, t=0 at contact)
    "ideal_peak_times": {
        "pelvis": float, "hip": float, "knee": float, "foot": float
    },

    # Modification summary
    "modification_flags": {
        "time_warped":       bool,
        "amplitude_scaled":  bool,
        "scale_factor_foot": float,
    },
}
```

### `ideal_skeletons_summary.csv` — 25 rows:

```
event_id, match_time, shooter_jersey, kicking_side,
wcs_original, wcs_ideal,
original_cascade_ok, ideal_cascade_ok,
original_amp_ratio, ideal_amp_ratio,
time_warped, amplitude_scaled, scale_factor_foot
```

---

## Sanity Report (printed at end)

```
Shot | WCS_orig → WCS_ideal | TimeWarp | AmpScale | Verified
-----+---------------------+----------+----------+---------
 640 |     100 →       100 |       No |       No |      OK
 318 |      80 →        98 |      Yes |       No |      OK
 ...
Final: N / 25 shots achieved wcs_ideal ≥ 90
```

---

## Key Design Decisions

1. **Sorted-time reassignment** preserves the actual timing distribution of the real kick
   — just assigns peaks to the correct segments.

2. **SLERP** via `scipy.spatial.transform.Slerp`: numerically stable SO(3) interpolation.
   Piecewise-linear time warp is applied only within `[PRE_T_START, PRE_T_END]` window.

3. **Forward kinematics from HIP** (not pelvis): hip position is tracked directly in the
   parquet; avoids compound errors from re-integrating from pelvis.

4. **4 body parts change** per shot (knee, ankle, heel, toe on kicking side). All 17
   others stay real. More natural-looking result.

5. **Foot-only amplitude scaling**: pelvis amplitude is the anchor. Only foot (and
   minimally shank) needs scaling to achieve `amp_ratio ≥ 3`.

---

## Implementation Plan

Single runnable script: `utils/generate_ideal_skeletons.py` (~320 lines).

**Functions:**
- `_build_pts(wide)` → `{part_id: (N,3)}` dict from wide pivot
- `_pts_to_array(pts, N)` → `(N, 21, 3)` with NaN for missing parts
- `time_warp(t, t_peak_real, t_peak_ideal)` → warped source times
- `warp_rotation_sequence(R, t, t_peak_real, t_peak_ideal)` → warped `(N, 3, 3)`
- `scale_rotation_amplitude(R, k)` → amplitude-scaled `(N, 3, 3)`
- `forward_kinematics_kicking_leg(pts_orig, R_thigh, R_shank, R_foot, side)` → updated pts dict
- `smooth_positions(pos, window=5, poly=2)` → SG-filtered `(N, 3)`
- `verify_wcs(pts, t, side)` → `(wcs_ideal, peak_info_dict)`

**Reused from `utils/kinematics.py`:**
- `build_pelvis_frame`, `build_thigh_frame`, `build_shank_frame`, `build_foot_frame`
- `relative_rotation`, `omega_from_R_incremental`, `whipchain_score`
- `SIDE_IDS`, `NECK`, `L_HIP`, `PELVIS`, `R_HIP`

**Execution:**
```bash
.venv/bin/python3 utils/generate_ideal_skeletons.py
```

---

## Constants

```python
FRAMERATE   = 25.0
DT          = 1.0 / FRAMERATE
PRE_FRAMES  = 75    # 3 s before contact
POST_FRAMES = 25    # 1 s after contact
PRE_T_START = -2.0  # peak extraction window start (s)
PRE_T_END   = -0.05 # peak extraction window end (s)
MIN_GAP     = 0.04  # minimum gap between consecutive ideal peak times (1 frame)
```
