"""
Comprehensive tests for utils/kinematics.py (Phase 2).
Run with:  .venv/bin/python3 utils/test_kinematics.py
"""

import sys
import traceback
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, ".")
from utils.kinematics import (
    normalize,
    build_frame_from_points,
    build_frame_from_points_candidates,
    build_frame_from_xz,
    forward_fill_rotations,
    enforce_z_continuity,
    fix_forward_sign,
    build_pelvis_frame,
    build_thigh_frame,
    build_shank_frame,
    build_foot_frame,
    relative_rotation,
    omega_from_R,
    omega_from_R_incremental,
    detect_kicking_side,
    whipchain_score,
)


PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"
_failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"{PASS}  {name}")
    else:
        print(f"{FAIL}  {name}  {detail}")
        _failures.append(name)


def almost_equal(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)


def is_rotation(R: np.ndarray, tol=1e-5) -> np.ndarray:
    """Return bool array: True where R[i] is a valid SO(3) rotation matrix."""
    N = R.shape[0]
    ok = np.ones(N, dtype=bool)
    for i in range(N):
        Ri = R[i]
        orthonormal = np.allclose(Ri.T @ Ri, np.eye(3), atol=tol)
        det_one     = abs(np.linalg.det(Ri) - 1.0) < tol
        ok[i] = orthonormal and det_one
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 2A: normalize
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2A: normalize ──")

v = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
u = normalize(v)
check("unit length for non-zero row",  abs(np.linalg.norm(u[0]) - 1.0) < 1e-9)
check("zero-vector stays finite",      np.all(np.isfinite(u[1])))
check("already unit → unchanged",      almost_equal(u[2], [1, 0, 0]))

# batch shape
v_batch = np.random.randn(50, 3)
u_batch = normalize(v_batch)
norms   = np.linalg.norm(u_batch, axis=-1)
check("batch normalize: all norms ≈ 1", np.allclose(norms, 1.0, atol=1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# 2A: build_frame_from_points
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2A: build_frame_from_points ──")

N = 20
p0 = np.zeros((N, 3))
p1 = np.tile([0, 1, 0], (N, 1))          # y-axis along (0,1,0)
p2 = np.tile([1, 0, 0], (N, 1))          # plane reference along x

R, valid = build_frame_from_points(p0, p1, p2)

check("all valid",              valid.all())
check("output shape (N,3,3)",   R.shape == (N, 3, 3))
check("all SO(3)",              is_rotation(R).all())
check("y-axis along [0,1,0]",  almost_equal(R[:, :, 1], np.tile([0,1,0], (N,1))))

# degenerate: p2 collinear with p1-p0
p2_bad = np.tile([0, 2, 0], (N, 1))
_, valid_bad = build_frame_from_points(p0, p1, p2_bad)
check("collinear p2 → invalid",  (~valid_bad).all())


# ─────────────────────────────────────────────────────────────────────────────
# 2A: build_frame_from_points_candidates
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2A: build_frame_from_points_candidates ──")

N = 30
p0 = np.zeros((N, 3))
p1 = np.tile([0, 1, 0], (N, 1))
# First candidate is bad (collinear), second is good
p2_bad  = np.tile([0, 2, 0], (N, 1))
p2_good = np.tile([1, 0, 0], (N, 1))

R_cand, valid_cand = build_frame_from_points_candidates(p0, p1, [p2_bad, p2_good])
check("candidates: all valid when at least one good", valid_cand.all())
check("candidates: output is SO(3)",                  is_rotation(R_cand).all())

# Both bad
R_both_bad, valid_both_bad = build_frame_from_points_candidates(
    p0, p1, [p2_bad, np.tile([0, 3, 0], (N, 1))]
)
check("candidates: all invalid when all bad", (~valid_both_bad).all())

# Pick the better candidate (larger cross-product norm)
p2_weak  = np.tile([0.001, 0, 1], (N, 1))   # small x component
p2_strong = np.tile([1, 0, 0], (N, 1))       # full x component
R_sel, _ = build_frame_from_points_candidates(p0, p1, [p2_weak, p2_strong])
# z-axis should be close to [0,0,-1] or [0,0,1] (y×p2 direction)
check("candidates: selects strongest plane",  is_rotation(R_sel).all())


# ─────────────────────────────────────────────────────────────────────────────
# 2A: build_frame_from_xz
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2A: build_frame_from_xz ──")

N = 15
x_dir = np.tile([1, 0, 0], (N, 1))          # right = x
z_dir = np.tile([0, 0, 1], (N, 1))          # up = z

R_xz, valid_xz = build_frame_from_xz(x_dir, z_dir)
check("build_frame_from_xz: all valid",         valid_xz.all())
check("build_frame_from_xz: SO(3)",             is_rotation(R_xz).all())
check("x col ≈ [1,0,0]",                        almost_equal(R_xz[:, :, 0], np.tile([1,0,0], (N,1))))
check("z col ≈ [0,0,1]",                        almost_equal(R_xz[:, :, 2], np.tile([0,0,1], (N,1))))
check("y col ≈ [0,1,0] (forward = z×x)",        almost_equal(R_xz[:, :, 1], np.tile([0,1,0], (N,1))))

# Tilted z
x_dir2 = np.tile([1, 0, 0], (N, 1))
z_dir2 = np.tile([0.1, 0, 1], (N, 1))  # slightly tilted
R_xz2, v2 = build_frame_from_xz(x_dir2, z_dir2)
check("tilted z: still valid SO(3)", v2.all() and is_rotation(R_xz2).all())


# ─────────────────────────────────────────────────────────────────────────────
# 2B: forward_fill_rotations
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2B: forward_fill_rotations ──")

N = 10
R_eye = np.eye(3)
R_rot = Rotation.from_euler("y", 45, degrees=True).as_matrix()

R_raw = np.stack([R_rot] * N)
valid  = np.ones(N, dtype=bool)
valid[3] = False
valid[4] = False

R_filled = forward_fill_rotations(R_raw, valid)
check("filled frames equal last good", almost_equal(R_filled[3], R_rot))
check("filled frame 4 equal last good", almost_equal(R_filled[4], R_rot))

# Leading invalid frames
R_raw2 = np.stack([R_rot] * N)
valid2  = np.zeros(N, dtype=bool)
valid2[5:] = True
R_filled2 = forward_fill_rotations(R_raw2, valid2)
check("leading invalid filled with first valid", almost_equal(R_filled2[0], R_rot))

# All invalid → no crash
R_all_bad = np.stack([R_rot] * N)
valid_all_bad = np.zeros(N, dtype=bool)
R_fb = forward_fill_rotations(R_all_bad, valid_all_bad)
check("all invalid → returns original without crash", R_fb.shape == (N, 3, 3))


# ─────────────────────────────────────────────────────────────────────────────
# 2B: enforce_z_continuity
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2B: enforce_z_continuity ──")

N = 5
# Build a valid frame then inject a flip
R_base, _ = build_frame_from_points(
    np.zeros((N, 3)),
    np.tile([0, 1, 0], (N, 1)),
    np.tile([1, 0, 0], (N, 1)),
)
# Flip z and x at frame 2
R_flipped = R_base.copy()
R_flipped[2, :, 2] = -R_flipped[2, :, 2]
R_flipped[2, :, 0] = np.cross(R_flipped[2, :, 1], R_flipped[2, :, 2])

R_fixed = enforce_z_continuity(R_flipped)
check("enforce_z: z continuity restored",
      np.dot(R_fixed[2, :, 2], R_fixed[1, :, 2]) > 0)
check("enforce_z: still SO(3) after fix", is_rotation(R_fixed).all())


# ─────────────────────────────────────────────────────────────────────────────
# 2B: fix_forward_sign
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2B: fix_forward_sign ──")

N = 4
R_base2, _ = build_frame_from_xz(
    np.tile([1, 0, 0], (N, 1)),
    np.tile([0, 0, 1], (N, 1)),
)
# Flip y and x at frame 1
R_flipped2 = R_base2.copy()
R_flipped2[1, :, 0] = -R_flipped2[1, :, 0]
R_flipped2[1, :, 1] = -R_flipped2[1, :, 1]

R_fixed2 = fix_forward_sign(R_flipped2, np.array([0.0, 1.0, 0.0]))
check("fix_forward: y faces forward after fix",
      np.dot(R_fixed2[1, :, 1], [0, 1, 0]) > 0)
check("fix_forward: still SO(3)", is_rotation(R_fixed2).all())


# ─────────────────────────────────────────────────────────────────────────────
# 2C: Segment builders with synthetic pts
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2C: Segment builders (synthetic) ──")

# Build a minimal pts dict for a player in a standard stance
# Positions roughly like a standing person (x=medial, y=fwd, z=up)
N = 50
t = np.linspace(0, 2*np.pi, N)

def _col(v): return np.tile(v, (N, 1)).astype(float)

# Slight oscillation to simulate motion
pts = {
    1:  _col([0.1,  0.0, 1.7]),   # L_EAR
    2:  _col([0.0,  0.1, 1.7]),   # NOSE
    3:  _col([-0.1, 0.0, 1.7]),   # R_EAR
    4:  _col([0.2,  0.0, 1.4]),   # L_SHOULDER
    5:  _col([0.0,  0.0, 1.5]),   # NECK
    6:  _col([-0.2, 0.0, 1.4]),   # R_SHOULDER
    7:  _col([0.3,  0.0, 1.1]),   # L_ELBOW
    8:  _col([-0.3, 0.0, 1.1]),   # R_ELBOW
    9:  _col([0.3,  0.0, 0.9]),   # L_WRIST
    10: _col([-0.3, 0.0, 0.9]),   # R_WRIST
    11: _col([0.1,  0.0, 0.9]),   # L_HIP
    12: _col([0.0,  0.0, 0.95]),  # PELVIS
    13: _col([-0.1, 0.0, 0.9]),   # R_HIP
    14: _col([0.1,  0.0, 0.5]),   # L_KNEE
    15: _col([-0.1, 0.0, 0.5]),   # R_KNEE
    16: _col([0.1,  0.0, 0.08]),  # L_ANKLE
    17: _col([-0.1, 0.0, 0.08]),  # R_ANKLE
    18: _col([0.05, -0.05, 0.03]),# L_HEEL
    19: _col([0.05,  0.15, 0.03]),# L_TOE
    20: _col([-0.05,-0.05, 0.03]),# R_HEEL
    21: _col([-0.05, 0.15, 0.03]),# R_TOE
}
# Add small perturbation to avoid perfectly static data
rng = np.random.default_rng(42)
for k in pts:
    pts[k] = pts[k] + rng.normal(0, 0.002, (N, 3))

R_pelvis, vp = build_pelvis_frame(pts)
check("pelvis: SO(3)",   is_rotation(R_pelvis).all())
check("pelvis: y≈forward (positive y component)",
      np.mean(R_pelvis[:, 1, 1]) > 0)  # y-col, y-component

for side in ("left", "right"):
    R_th, _ = build_thigh_frame(pts, side)
    R_sh, _ = build_shank_frame(pts, side)
    R_ft, _ = build_foot_frame(pts, side)
    check(f"thigh {side}: SO(3)",  is_rotation(R_th).all())
    check(f"shank {side}: SO(3)",  is_rotation(R_sh).all())
    check(f"foot  {side}: SO(3)",  is_rotation(R_ft).all())


# ─────────────────────────────────────────────────────────────────────────────
# 2D: relative_rotation
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2D: relative_rotation ──")

N = 10
R_prox = np.stack([Rotation.from_euler("y", a, degrees=True).as_matrix()
                   for a in np.linspace(0, 30, N)])
R_dist = np.stack([Rotation.from_euler("y", a, degrees=True).as_matrix()
                   for a in np.linspace(10, 50, N)])

R_rel = relative_rotation(R_prox, R_dist)
check("relative_rotation: shape",   R_rel.shape == (N, 3, 3))
check("relative_rotation: SO(3)",   is_rotation(R_rel).all())

# When prox == dist, relative = identity
R_same = np.stack([np.eye(3)] * N)
R_rel_id = relative_rotation(R_same, R_same)
check("relative_rotation of same → identity", almost_equal(R_rel_id, np.stack([np.eye(3)]*N)))

# Verify: R_rel[i] == R_prox[i].T @ R_dist[i]
for i in range(N):
    expected = R_prox[i].T @ R_dist[i]
    if not almost_equal(R_rel[i], expected):
        check("relative_rotation: manual spot-check", False, f"failed at i={i}")
        break
else:
    check("relative_rotation: manual spot-check all frames", True)


# ─────────────────────────────────────────────────────────────────────────────
# 2E: omega_from_R  (fallback)
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2E: omega_from_R ──")

# Constant rotation about z at 2 rad/s
dt   = 1.0 / 25.0
N_om = 40
angles = np.linspace(0, 2 * dt * N_om, N_om)
R_spin = np.stack([Rotation.from_euler("z", a).as_matrix() for a in angles])

omega_fd = omega_from_R(R_spin, dt)
# Interior frames should have |omega_z| ≈ 2 rad/s, others close
omega_z_mid = omega_fd[2:-2, 2]
check("omega_from_R: z-component ≈ 2 rad/s (interior)",
      np.allclose(omega_z_mid, 2.0, atol=0.1),
      f"got mean={omega_z_mid.mean():.3f}")
check("omega_from_R: x,y components near 0",
      np.allclose(omega_fd[2:-2, :2], 0.0, atol=0.1))


# ─────────────────────────────────────────────────────────────────────────────
# 2E: omega_from_R_incremental (primary)
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2E: omega_from_R_incremental ──")

omega_inc = omega_from_R_incremental(R_spin, dt)
omega_z_mid_inc = omega_inc[2:-2, 2]
check("incremental: z-component ≈ 2 rad/s (interior)",
      np.allclose(omega_z_mid_inc, 2.0, atol=0.15),
      f"got mean={omega_z_mid_inc.mean():.3f}")
check("incremental: x,y components near 0",
      np.allclose(omega_inc[2:-2, :2], 0.0, atol=0.15))
check("incremental shape", omega_inc.shape == (N_om, 3))

# Multi-axis rotation
angles_xy = np.linspace(0, 1.0, 50)
R_multi = np.stack([Rotation.from_euler("xyz", [a, 0.5*a, 0.3*a]).as_matrix()
                    for a in angles_xy])
omega_multi = omega_from_R_incremental(R_multi, dt)
check("multi-axis: finite values", np.all(np.isfinite(omega_multi)))
check("multi-axis: reasonable magnitudes (<50 rad/s)",
      np.all(np.linalg.norm(omega_multi, axis=1) < 50.0))

# Cross-check: incremental vs finite-diff agree in magnitude (interior)
omega_fd2  = omega_from_R(R_spin, dt)
mag_inc = np.linalg.norm(omega_inc[5:-5], axis=1)
mag_fd  = np.linalg.norm(omega_fd2[5:-5], axis=1)
check("incremental ≈ fd magnitude (interior, tol=0.2)",
      np.allclose(mag_inc, mag_fd, atol=0.2),
      f"max diff={np.max(np.abs(mag_inc - mag_fd)):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2F: detect_kicking_side
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2F: detect_kicking_side ──")

# Build synthetic wide pivot with left toe much closer to ball
frames = np.arange(100, 110)
data = {}
for bp in range(1, 22):
    for coord in ("x", "y", "z"):
        data[(coord, bp)] = np.random.randn(len(frames)) * 0.05

# Left toe (19) positioned near ball at (1, 0, 0)
data[("x", 19)] = np.ones(len(frames)) * 1.0
data[("y", 19)] = np.zeros(len(frames))
data[("z", 19)] = np.zeros(len(frames))

# Right toe (21) far from ball
data[("x", 21)] = np.ones(len(frames)) * 5.0
data[("y", 21)] = np.zeros(len(frames))
data[("z", 21)] = np.zeros(len(frames))

wide_df = pd.DataFrame(data, index=frames)
wide_df.columns = pd.MultiIndex.from_tuples(wide_df.columns)

ball_df = pd.DataFrame({"bx": np.ones(len(frames)), "by": np.zeros(len(frames)),
                        "bz": np.zeros(len(frames))}, index=frames)

side = detect_kicking_side(wide_df, shot_frame=109, ball_df=ball_df)
check("detect_kicking_side with ball: left detected", side == "left")

# Right toe closer
data2 = {k: v.copy() for k, v in data.items()}
data2[("x", 19)] = np.ones(len(frames)) * 5.0   # left far
data2[("x", 21)] = np.ones(len(frames)) * 1.0   # right near
wide_df2 = pd.DataFrame(data2, index=frames)
wide_df2.columns = pd.MultiIndex.from_tuples(wide_df2.columns)

side2 = detect_kicking_side(wide_df2, shot_frame=109, ball_df=ball_df)
check("detect_kicking_side with ball: right detected", side2 == "right")

# Fallback (no ball)
# Left toe moving more
data3 = {k: np.zeros(len(frames)) for k, _ in data.items()}
data3[("x", 19)] = np.linspace(-0.5, 0.5, len(frames))  # large movement left
data3[("x", 21)] = np.zeros(len(frames))                 # right stationary
wide_df3 = pd.DataFrame(data3, index=frames)
wide_df3.columns = pd.MultiIndex.from_tuples(wide_df3.columns)

side3 = detect_kicking_side(wide_df3, shot_frame=109, ball_df=None)
check("detect_kicking_side fallback: left detected (moving more)", side3 == "left")


# ─────────────────────────────────────────────────────────────────────────────
# 2G: whipchain_score
# ─────────────────────────────────────────────────────────────────────────────

print("\n── 2G: whipchain_score ──")

# Perfect cascade: each peaks later than previous, foot >> pelvis
score_perfect = whipchain_score(
    peak_t_pelvis=-1.5, peak_t_hip=-1.2, peak_t_knee=-0.9, peak_t_foot=-0.5,
    peak_omega_pelvis=5.0, peak_omega_foot=18.0
)
check("perfect cascade: sequence=1.0, amp high → score > 80",
      score_perfect > 80,
      f"got {score_perfect}")

# Wrong order: foot peaks before pelvis
score_reversed = whipchain_score(
    peak_t_pelvis=-0.5, peak_t_hip=-0.8, peak_t_knee=-1.1, peak_t_foot=-1.4,
    peak_omega_pelvis=5.0, peak_omega_foot=18.0
)
# sequence=0 → 0.6*0 = 0; amp_ratio=18/5=3.6→cap→amp_score=1.0 → 0.4*100=40
check("reversed cascade: sequence=0.0 → score ≤ 40",
      score_reversed <= 40,
      f"got {score_reversed}")

# Low amp ratio
score_lowamp = whipchain_score(
    peak_t_pelvis=-1.5, peak_t_hip=-1.2, peak_t_knee=-0.9, peak_t_foot=-0.5,
    peak_omega_pelvis=10.0, peak_omega_foot=10.0  # ratio=1 → amp_score=0.33
)
check("low amp ratio: score < 80", score_lowamp < 80)

# Score is always int 0–100
for _ in range(50):
    s = whipchain_score(
        peak_t_pelvis=np.random.uniform(-2, -1),
        peak_t_hip=np.random.uniform(-2, -0.5),
        peak_t_knee=np.random.uniform(-1.5, -0.2),
        peak_t_foot=np.random.uniform(-1, 0),
        peak_omega_pelvis=np.random.uniform(0, 15),
        peak_omega_foot=np.random.uniform(0, 40),
    )
    if not (0 <= s <= 100 and isinstance(s, int)):
        check("score always int 0–100", False, f"got {s}")
        break
else:
    check("score always int 0–100 (50 random trials)", True)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full pipeline on real shot data
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Integration: real shot data ──")

try:
    import pandas as pd

    parts = pd.read_parquet("data/shots_parts.parquet")
    detections = pd.read_csv("data/shot_detections.csv")

    # Take first low-confidence=False shot
    row = detections[detections["low_confidence"] == False].iloc[0]
    shot_frame    = int(row["shot_frame"])
    shooter       = int(row["shooter_jersey"])
    side_from_csv = "left" if "LEFT" in str(row["shooter_body_part"]) else "right"

    # Filter window
    window_parts = parts[
        (parts["jersey_number"] == shooter) &
        (parts["frame_number"] >= shot_frame - 75) &
        (parts["frame_number"] <= shot_frame + 25)
    ]
    check("integration: window not empty", len(window_parts) > 0)

    # Pivot
    wide = window_parts.pivot(index="frame_number", columns="body_part",
                              values=["x", "y", "z"])
    wide = wide.interpolate(limit=3, limit_direction="both")

    # Build pts dict
    frame_idx = wide.index.values
    N_real = len(frame_idx)
    pts_real = {bp: wide[["x","y","z"]].xs(bp, axis=1, level=1).values
                for bp in range(1, 22) if bp in wide["x"].columns}

    # Make sure all required parts exist
    required = [5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    missing = [p for p in required if p not in pts_real]
    check("integration: all required body parts present",
          len(missing) == 0, f"missing: {missing}")

    if not missing:
        R_p, vp = build_pelvis_frame(pts_real)
        R_t, vt = build_thigh_frame(pts_real, side_from_csv)
        R_s, vs = build_shank_frame(pts_real, side_from_csv)
        R_f, vf = build_foot_frame(pts_real, side_from_csv)

        check("integration: pelvis frame SO(3)",  is_rotation(R_p).all(),
              f"valid={vp.sum()}/{N_real}")
        check("integration: thigh  frame SO(3)",  is_rotation(R_t).all(),
              f"valid={vt.sum()}/{N_real}")
        check("integration: shank  frame SO(3)",  is_rotation(R_s).all(),
              f"valid={vs.sum()}/{N_real}")
        check("integration: foot   frame SO(3)",  is_rotation(R_f).all(),
              f"valid={vf.sum()}/{N_real}")

        # Relative rotations
        R_hip_joint  = relative_rotation(R_p, R_t)
        R_knee_joint = relative_rotation(R_t, R_s)
        check("integration: hip joint SO(3)",  is_rotation(R_hip_joint).all())
        check("integration: knee joint SO(3)", is_rotation(R_knee_joint).all())

        # Angular velocities
        dt = 1.0 / 25.0
        om_p = omega_from_R_incremental(R_p, dt)
        om_h = omega_from_R_incremental(R_hip_joint, dt)
        om_k = omega_from_R_incremental(R_knee_joint, dt)
        om_f = omega_from_R_incremental(R_f, dt)

        mag_p = np.linalg.norm(om_p, axis=1)
        mag_h = np.linalg.norm(om_h, axis=1)
        mag_k = np.linalg.norm(om_k, axis=1)
        mag_f = np.linalg.norm(om_f, axis=1)

        check("integration: omega finite (pelvis)", np.all(np.isfinite(mag_p)))
        check("integration: omega finite (hip)",    np.all(np.isfinite(mag_h)))
        check("integration: omega finite (knee)",   np.all(np.isfinite(mag_k)))
        check("integration: omega finite (foot)",   np.all(np.isfinite(mag_f)))

        # Sanity ranges (pre-contact window)
        t = (frame_idx - shot_frame) / 25.0
        pre = (t >= -2.0) & (t <= -0.05)
        if pre.sum() > 0:
            pk_p = float(mag_p[pre].max())
            pk_k = float(mag_k[pre].max())
            pk_f = float(mag_f[pre].max())
            check("integration: peak pelvis omega in [0.5, 100]",
                  0.5 < pk_p < 100.0, f"got {pk_p:.2f}")
            check("integration: peak knee   omega in [0.5, 80]",
                  0.5 < pk_k < 80.0, f"got {pk_k:.2f}")
            check("integration: peak foot   omega in [0.5, 80]",
                  0.5 < pk_f < 80.0, f"got {pk_f:.2f}")
            check("integration: no exploding omega (>100)",
                  max(pk_p, pk_k, pk_f) < 100.0,
                  f"max={max(pk_p, pk_k, pk_f):.1f}")

            # WhipChain
            t_pre = t[pre]
            pt_p = float(t_pre[np.argmax(mag_p[pre])])
            pt_h = float(t_pre[np.argmax(mag_h[pre])])
            pt_k = float(t_pre[np.argmax(mag_k[pre])])
            pt_f = float(t_pre[np.argmax(mag_f[pre])])

            wcs = whipchain_score(pt_p, pt_h, pt_k, pt_f, pk_p, pk_f)
            check("integration: whipchain score 0–100", 0 <= wcs <= 100,
                  f"got {wcs}")

            print(f"\n  Shot {int(row['event_id'])} ({row['match_time_mmss']}), "
                  f"shooter #{shooter} ({side_from_csv})")
            print(f"  peak ω: pelvis={pk_p:.1f}  knee={pk_k:.1f}  foot={pk_f:.1f}  rad/s")
            print(f"  peak t: pelvis={pt_p:.2f}  hip={pt_h:.2f}  "
                  f"knee={pt_k:.2f}  foot={pt_f:.2f}  s")
            print(f"  WhipChain score: {wcs}")

except Exception as exc:
    print(f"\n  [integration test crashed] {exc}")
    traceback.print_exc()
    _failures.append("integration: crashed")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
total  = sum(1 for line in open(__file__) if line.strip().startswith("check("))
passed = total - len(_failures)
print(f"Passed: {passed}  |  Failed: {len(_failures)}")
if _failures:
    print("Failed tests:")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All tests passed ✓")
