"""
Core kinematics math primitives for the WhipChain analysis pipeline.

All functions are vectorised over N frames.

Frame convention (applies to ALL rotation matrices produced here):
    R is (N, 3, 3); columns are body-frame unit axes:
        col 0  x  →  lateral / derived
        col 1  y  →  primary axis (segment distal direction, or pelvis-forward)
        col 2  z  →  secondary axis (plane normal for segments, superior for pelvis)
    Right-hand rule: x = y × z,  y = z × x,  z = x × y
    det(R) = +1 for all valid frames.

Typical body-part IDs (1-indexed, as in the parquet):
    1=L_EAR, 2=NOSE, 3=R_EAR, 4=L_SHOULDER, 5=NECK, 6=R_SHOULDER,
    7=L_ELBOW, 8=R_ELBOW, 9=L_WRIST, 10=R_WRIST,
    11=L_HIP, 12=PELVIS, 13=R_HIP,
    14=L_KNEE, 15=R_KNEE, 16=L_ANKLE, 17=R_ANKLE,
    18=L_HEEL, 19=L_TOE, 20=R_HEEL, 21=R_TOE
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ── Body-part ID constants ────────────────────────────────────────────────────

NECK    = 5
L_HIP   = 11
PELVIS  = 12
R_HIP   = 13

# Side-specific IDs: side in {"left", "right"}
SIDE_IDS = {
    "right": dict(hip=13, knee=15, ankle=17, heel=20, toe=21),
    "left":  dict(hip=11, knee=14, ankle=16, heel=18, toe=19),
}


# ── 2A: Rotation Matrix Builders ─────────────────────────────────────────────

def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalise rows of (..., 3) array to unit length.

    Parameters
    ----------
    v   : (..., 3) float array
    eps : minimum norm before clamping (avoids division by zero)

    Returns
    -------
    (..., 3) unit vectors; zero-vectors become zero-vectors (not NaN).
    """
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < eps, eps, n)


def build_frame_from_points(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build rotation matrices from three point-clouds via Gram-Schmidt.

    y-axis: p0 → p1 (segment long axis, proximal to distal)
    z-axis: Gram-Schmidt component of (p2 − p0) perpendicular to y
    x-axis: y × z  (right-hand derived)

    Parameters
    ----------
    p0, p1, p2 : (N, 3) position arrays in metres

    Returns
    -------
    R     : (N, 3, 3)  rotation matrices (columns = x, y, z body axes)
    valid : (N,)  bool mask — True where cross-product norm > 1e-4
    """
    y = normalize(p1 - p0)                                    # (N, 3)

    temp   = p2 - p0                                           # (N, 3)
    z_raw  = temp - np.einsum("ni,ni->n", temp, y)[:, None] * y  # Gram-Schmidt
    norm_z = np.linalg.norm(z_raw, axis=-1)                   # (N,)
    valid  = norm_z > 1e-4

    z = np.where(valid[:, None], z_raw / np.maximum(norm_z[:, None], 1e-9), 0.0)
    x = np.cross(y, z)                                         # y × z = x

    R = np.stack([x, y, z], axis=-1)                          # (N, 3, 3) column-wise
    return R, valid


def build_frame_from_points_candidates(
    p0: np.ndarray,
    p1: np.ndarray,
    p2_list: list[np.ndarray],
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Like build_frame_from_points but tries multiple p2 candidates per frame.

    Selects the candidate that gives the largest Gram-Schmidt norm (most stable
    plane definition) on a per-frame basis.

    Parameters
    ----------
    p0, p1   : (N, 3)
    p2_list  : list of (N, 3) candidate plane-reference arrays
    eps      : minimum acceptable Gram-Schmidt norm

    Returns
    -------
    R     : (N, 3, 3)
    valid : (N,) bool — True where best candidate norm > eps
    """
    N = p0.shape[0]
    y = normalize(p1 - p0)                                    # (N, 3)

    best_norm  = np.full(N, -1.0)
    best_z_raw = np.zeros((N, 3))

    for p2 in p2_list:
        temp  = p2 - p0
        z_raw = temp - np.einsum("ni,ni->n", temp, y)[:, None] * y
        n     = np.linalg.norm(z_raw, axis=-1)
        better = n > best_norm
        best_norm  = np.where(better, n, best_norm)
        best_z_raw = np.where(better[:, None], z_raw, best_z_raw)

    valid = best_norm > eps
    z = np.where(
        valid[:, None],
        best_z_raw / np.maximum(best_norm[:, None], 1e-9),
        0.0,
    )
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=-1)
    return R, valid


def build_frame_from_xz(
    x_dir: np.ndarray,
    z_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build rotation matrix from a lateral direction and a superior direction.

    z-axis: normalize(z_dir)             — superior / up
    x-axis: Gram-Schmidt(x_dir ⊥ z)    — lateral / right
    y-axis: z × x                        — anterior / forward

    (Used for pelvis frame per Primer §5.1.2.)

    Parameters
    ----------
    x_dir : (N, 3) right-lateral direction  (R_hip − L_hip)
    z_dir : (N, 3) superior direction        (neck − pelvis)

    Returns
    -------
    R     : (N, 3, 3)
    valid : (N,) bool
    """
    z = normalize(z_dir)                                      # (N, 3) superior

    x_raw  = x_dir - np.einsum("ni,ni->n", x_dir, z)[:, None] * z
    norm_x = np.linalg.norm(x_raw, axis=-1)
    valid  = norm_x > 1e-4

    x = np.where(valid[:, None], x_raw / np.maximum(norm_x[:, None], 1e-9), 0.0)
    y = np.cross(z, x)                                        # z × x = forward

    R = np.stack([x, y, z], axis=-1)                         # (N, 3, 3)
    return R, valid


# ── 2B: Continuity Fixes ─────────────────────────────────────────────────────

def forward_fill_rotations(
    R_raw: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Replace invalid frames with the last-good rotation matrix.

    Frames before the first valid one are filled with the first valid rotation
    (backward fill for the leading edge).

    Parameters
    ----------
    R_raw : (N, 3, 3)
    valid : (N,) bool

    Returns
    -------
    (N, 3, 3) with no invalid frames (unless all are invalid).
    """
    R = R_raw.copy()
    N = len(valid)

    if not valid.any():
        return R

    first_valid = int(np.argmax(valid))

    # Forward pass: carry last good forward
    last_good = R[first_valid].copy()
    for i in range(N):
        if valid[i]:
            last_good = R[i].copy()
        else:
            R[i] = last_good

    # Back-fill leading invalid frames
    if first_valid > 0:
        R[:first_valid] = R[first_valid]

    return R


def enforce_z_continuity(R: np.ndarray) -> np.ndarray:
    """Prevent 180° flips of the z-axis (col 2) between consecutive frames.

    When a flip is detected (z[i] · z[i-1] < 0):
      - z is negated
      - x is recomputed as y × new_z  (maintains det = +1, y unchanged)

    Parameters
    ----------
    R : (N, 3, 3) rotation matrices

    Returns
    -------
    (N, 3, 3) with continuous z-axis.
    """
    R = R.copy()
    for i in range(1, len(R)):
        if np.dot(R[i, :, 2], R[i - 1, :, 2]) < 0:
            R[i, :, 2] = -R[i, :, 2]
            R[i, :, 0] = np.cross(R[i, :, 1], R[i, :, 2])   # x = y × new_z
    return R


def fix_forward_sign(R: np.ndarray, ref_fwd: np.ndarray) -> np.ndarray:
    """Ensure the y-axis (col 1) faces consistently forward across all frames.

    Two-pass approach to avoid per-frame discontinuities:
      Pass 1: align the first frame's y-axis to ref_fwd.
      Pass 2: propagate continuity — if y[i] flips relative to y[i-1], flip
              both x and y at frame i (keeps z unchanged and det = +1).

    This prevents spurious 180° flips at frames where y · ref_fwd crosses zero.

    Parameters
    ----------
    R       : (N, 3, 3)
    ref_fwd : (3,) reference forward direction (global frame)

    Returns
    -------
    (N, 3, 3)
    """
    R = R.copy()
    if len(R) == 0:
        return R

    # Pass 1: set sign of first frame against the global reference
    if np.dot(R[0, :, 1], ref_fwd) < 0:
        R[0, :, 0] = -R[0, :, 0]
        R[0, :, 1] = -R[0, :, 1]

    # Pass 2: propagate — flip x+y if y flips relative to previous frame
    for i in range(1, len(R)):
        if np.dot(R[i, :, 1], R[i - 1, :, 1]) < 0:
            R[i, :, 0] = -R[i, :, 0]
            R[i, :, 1] = -R[i, :, 1]

    return R


# ── 2C: Segment-specific Frame Builders ──────────────────────────────────────

def build_pelvis_frame(pts: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build pelvis rotation matrix using build_frame_from_xz.

    x_dir = R_hip(13) − L_hip(11)   → lateral right
    z_dir = neck(5)   − pelvis(12)  → superior

    Applies forward_fill → enforce_z_continuity → fix_forward_sign.

    Parameters
    ----------
    pts : dict mapping body_part_id → (N, 3) position array

    Returns
    -------
    R     : (N, 3, 3)
    valid : (N,) bool
    """
    x_dir = pts[R_HIP] - pts[L_HIP]      # lateral
    z_dir = pts[NECK]  - pts[PELVIS]      # superior

    R, valid = build_frame_from_xz(x_dir, z_dir)
    R = forward_fill_rotations(R, valid)
    R = enforce_z_continuity(R)
    R = fix_forward_sign(R, np.array([0.0, 1.0, 0.0]))
    return R, valid


def build_thigh_frame(pts: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Build thigh frame: hip → knee, plane from pelvis / opposite hip.

    Parameters
    ----------
    pts  : dict body_part_id → (N, 3)
    side : "left" or "right"

    Returns
    -------
    R, valid
    """
    ids     = SIDE_IDS[side]
    opp_hip = SIDE_IDS["left"]["hip"] if side == "right" else SIDE_IDS["right"]["hip"]

    R, valid = build_frame_from_points_candidates(
        pts[ids["hip"]],
        pts[ids["knee"]],
        [pts[PELVIS], pts[opp_hip]],
    )
    R = forward_fill_rotations(R, valid)
    R = enforce_z_continuity(R)
    return R, valid


def build_shank_frame(pts: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Build shank frame: knee → ankle, plane candidates: toe, heel, opp-hip.

    Parameters
    ----------
    pts  : dict body_part_id → (N, 3)
    side : "left" or "right"

    Returns
    -------
    R, valid
    """
    ids     = SIDE_IDS[side]
    opp_hip = SIDE_IDS["left"]["hip"] if side == "right" else SIDE_IDS["right"]["hip"]

    R, valid = build_frame_from_points_candidates(
        pts[ids["knee"]],
        pts[ids["ankle"]],
        [pts[ids["toe"]], pts[ids["heel"]], pts[opp_hip]],
    )
    R = forward_fill_rotations(R, valid)
    R = enforce_z_continuity(R)
    return R, valid


def build_foot_frame(pts: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Build foot frame: heel → toe, plane candidates: ankle, knee.

    Parameters
    ----------
    pts  : dict body_part_id → (N, 3)
    side : "left" or "right"

    Returns
    -------
    R, valid
    """
    ids = SIDE_IDS[side]

    R, valid = build_frame_from_points_candidates(
        pts[ids["heel"]],
        pts[ids["toe"]],
        [pts[ids["ankle"]], pts[ids["knee"]]],
    )
    R = forward_fill_rotations(R, valid)
    R = enforce_z_continuity(R)
    return R, valid


# ── 2D: Joint Rotations ───────────────────────────────────────────────────────

def relative_rotation(R_prox: np.ndarray, R_dist: np.ndarray) -> np.ndarray:
    """Relative rotation of distal segment w.r.t. proximal (joint rotation).

    R_rel[i] = R_prox[i].T  @  R_dist[i]

    Usage::

        R_hip_joint  = relative_rotation(R_pelvis, R_thigh)
        R_knee_joint = relative_rotation(R_thigh,  R_shank)

    Parameters
    ----------
    R_prox, R_dist : (N, 3, 3)

    Returns
    -------
    (N, 3, 3)
    """
    return np.einsum("nij,nik->njk", R_prox, R_dist)


# ── 2E: Angular Velocity ──────────────────────────────────────────────────────

def _skew_to_omega(S: np.ndarray) -> np.ndarray:
    """Extract angular velocity (N, 3) from skew-symmetric matrices (N, 3, 3)."""
    return np.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], axis=-1)


def omega_from_R(R: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity via finite-difference Rdot @ R.T (fallback / validation).

    More susceptible to numerical noise at low frame rates than the incremental
    method.  Use omega_from_R_incremental as primary.

    Parameters
    ----------
    R  : (N, 3, 3) sequence of rotation matrices
    dt : float  time step in seconds (1/framerate)

    Returns
    -------
    (N, 3) rad/s in the world frame
    """
    Rdot = np.empty_like(R)
    Rdot[1:-1] = (R[2:] - R[:-2]) / (2.0 * dt)
    Rdot[0]    = (R[1]  - R[0])   / dt
    Rdot[-1]   = (R[-1] - R[-2])  / dt

    S = np.einsum("nij,nkj->nik", Rdot, R)         # Rdot @ R.T
    S = 0.5 * (S - np.transpose(S, (0, 2, 1)))     # enforce skew-symmetry
    return _skew_to_omega(S)


def omega_from_R_incremental(R: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity via SciPy Rotation.as_rotvec() — PRIMARY method.

    Computes incremental rotation dR[i] = R[i-1].T @ R[i] per step, converts
    to rotation vector, and divides by dt.  More stable than Rdot at 25 Hz.

    Parameters
    ----------
    R  : (N, 3, 3) sequence of rotation matrices
    dt : float  time step in seconds

    Returns
    -------
    (N, 3) rad/s (expressed in the body frame at each step)
    """
    N = len(R)
    omega = np.zeros((N, 3))

    # Incremental rotations: R[i-1].T @ R[i]  for i in 1..N-1
    dR = np.einsum("nij,nik->njk", R[:-1], R[1:])   # (N-1, 3, 3)

    rot_vec   = Rotation.from_matrix(dR).as_rotvec() # (N-1, 3)
    omega_mid = rot_vec / dt                          # rad/s

    omega[1:] = omega_mid
    omega[0]  = omega[1]   # replicate first sample

    return omega


# ── 2F: Kicking Side Detection ────────────────────────────────────────────────

def detect_kicking_side(
    wide,
    shot_frame: int,
    ball_df=None,
    framerate: float = 25.0,
) -> str:
    """Determine which foot struck the ball.

    Primary method: compare mean toe-to-ball distance in [shot_frame-5,
    shot_frame].  The toe closer to the ball is the kicking foot.

    Fallback (when ball_df is None): the foot whose toe has larger position
    variance in the same window is the kicking foot.

    Parameters
    ----------
    wide      : pivot DataFrame with MultiIndex columns (coord, body_part_id)
                  coords are "x", "y", "z"; body_part_id ints 1–21.
                  Index = frame_number.
    shot_frame: int, contact frame number
    ball_df   : optional DataFrame indexed by frame_number with columns
                  bx, by, bz (ball position in metres). Pass None to use
                  fallback.
    framerate : float (unused but kept for API consistency)

    Returns
    -------
    "left" or "right"
    """
    window = wide.loc[
        (wide.index >= shot_frame - 5) & (wide.index <= shot_frame)
    ]
    if len(window) == 0:
        return "right"

    L_TOE_ID = 19
    R_TOE_ID = 21

    if ball_df is not None:
        ball_win = ball_df.loc[ball_df.index.isin(window.index)]

        def dist_to_ball(toe_id: int) -> float:
            try:
                tx = window[("x", toe_id)].values
                ty = window[("y", toe_id)].values
                tz = window[("z", toe_id)].values
                bx = ball_win["bx"].values if len(ball_win) == len(window) else np.zeros_like(tx)
                by = ball_win["by"].values if len(ball_win) == len(window) else np.zeros_like(ty)
                bz = ball_win["bz"].values if len(ball_win) == len(window) else np.zeros_like(tz)
                return float(np.nanmean(np.sqrt((tx - bx)**2 + (ty - by)**2 + (tz - bz)**2)))
            except KeyError:
                return np.inf

        d_left  = dist_to_ball(L_TOE_ID)
        d_right = dist_to_ball(R_TOE_ID)
        return "left" if d_left < d_right else "right"

    # Fallback: toe with higher positional variance = swinging (kicking) foot
    def toe_variance(toe_id: int) -> float:
        try:
            tx = window[("x", toe_id)].values
            ty = window[("y", toe_id)].values
            return float(np.nanvar(tx) + np.nanvar(ty))
        except KeyError:
            return 0.0

    v_left  = toe_variance(L_TOE_ID)
    v_right = toe_variance(R_TOE_ID)
    return "left" if v_left > v_right else "right"


# ── 2G: WhipChain Score ───────────────────────────────────────────────────────

def whipchain_score(
    peak_t_pelvis: float,
    peak_t_hip: float,
    peak_t_knee: float,
    peak_t_foot: float,
    peak_omega_pelvis: float,
    peak_omega_foot: float,
) -> int:
    """Compute WhipChain score (0–100) for a single shot.

    Combines two sub-scores:
      sequence_score : fraction of the 3 cascade gaps that are positive
                       (each distal segment peaks AFTER its proximal neighbor).
      amp_score      : foot / pelvis peak angular-velocity ratio, normalised
                       to [0, 1] with cap at ratio = 3.

    Formula::

        gaps           = [t_hip − t_pelvis, t_knee − t_hip, t_foot − t_knee]
        sequence_score = sum(g > 0 for g in gaps) / 3          → [0, 1]
        amp_ratio      = peak_omega_foot / max(peak_omega_pelvis, 1e-3)
        amp_score      = min(amp_ratio / 3.0, 1.0)             → [0, 1]
        score          = round((0.6 × sequence_score + 0.4 × amp_score) × 100)

    Parameters
    ----------
    peak_t_*     : float, time (s) of peak angular-velocity in pre-contact window
    peak_omega_* : float, peak angular-velocity magnitude (rad/s)

    Returns
    -------
    int  0–100
    """
    gaps = [
        peak_t_hip   - peak_t_pelvis,
        peak_t_knee  - peak_t_hip,
        peak_t_foot  - peak_t_knee,
    ]
    sequence_score = sum(g > 0 for g in gaps) / 3.0
    amp_ratio      = peak_omega_foot / max(peak_omega_pelvis, 1e-3)
    amp_score      = min(amp_ratio / 3.0, 1.0)
    return round((0.6 * sequence_score + 0.4 * amp_score) * 100)
