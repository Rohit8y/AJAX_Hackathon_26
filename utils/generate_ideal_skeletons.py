"""
Phase 5: Ideal WCS=100 Skeleton Generation
------------------------------------------
Reads  : data/shot_kinematics.pkl
         data/shots_parts.parquet
         data/shot_detections.csv
Writes : data/ideal_skeletons.pkl          (list of 25 dicts)
         data/ideal_skeletons_summary.csv  (scalar summary)

Run:
    .venv/bin/python3 utils/generate_ideal_skeletons.py
"""

import pickle
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, ".")
from utils.kinematics import (
    build_pelvis_frame,
    build_thigh_frame,
    build_shank_frame,
    build_foot_frame,
    relative_rotation,
    omega_from_R_incremental,
    whipchain_score,
    SIDE_IDS,
)

# ── Constants ─────────────────────────────────────────────────────────────────

FRAMERATE   = 25.0
DT          = 1.0 / FRAMERATE
PRE_FRAMES  = 75    # 3 s before contact
POST_FRAMES = 25    # 1 s after contact
PRE_T_START = -2.0  # peak extraction window start (s)
PRE_T_END   = -0.05 # peak extraction window end (s)
MIN_GAP     = 0.04  # minimum gap between consecutive ideal peak times (1 frame)

PARTS_PATH      = "data/shots_parts.parquet"
KINEMATICS_PKL  = "data/shot_kinematics.pkl"
DETECTIONS_PATH = "data/shot_detections.csv"
OUT_PKL         = "data/ideal_skeletons.pkl"
OUT_CSV         = "data/ideal_skeletons_summary.csv"

REQUIRED_PARTS = list(range(1, 22))  # body-part IDs 1–21


# ── Helper: build pts dict from wide pivot ────────────────────────────────────

def _build_pts(wide: pd.DataFrame) -> dict:
    """Return {part_id: (N, 3) float array} for all body parts that exist."""
    pts = {}
    for bp in REQUIRED_PARTS:
        if bp in wide["x"].columns:
            pts[bp] = wide[["x", "y", "z"]].xs(bp, axis=1, level=1).values.astype(float)
    return pts


def _pts_to_array(pts: dict, N: int) -> np.ndarray:
    """Convert pts dict to (N, 21, 3) array; NaN for missing parts."""
    arr = np.full((N, 21, 3), np.nan)
    for bp_id, pos in pts.items():
        if 1 <= bp_id <= 21:
            arr[:, bp_id - 1, :] = pos
    return arr


# ── Time warp ─────────────────────────────────────────────────────────────────

def time_warp(
    t: np.ndarray,
    t_peak_real: float,
    t_peak_ideal: float,
) -> np.ndarray:
    """Piecewise-linear time warp: maps output time → source time.

    Only modifies times within [PRE_T_START, PRE_T_END]; identity outside.
    Anchored so that:
      - source time at t_peak_ideal = t_peak_real
      - linear interpolation before and after the peak within the window
    """
    t_start = PRE_T_START
    t_end   = PRE_T_END

    warped = t.copy()
    in_window   = (t >= t_start) & (t <= t_end)
    before_peak = in_window & (t <= t_peak_ideal)
    after_peak  = in_window & (t  > t_peak_ideal)

    # Before peak: [t_start, t_peak_ideal] → [t_start, t_peak_real]
    if t_peak_ideal > t_start and before_peak.any():
        scale = (t_peak_real - t_start) / (t_peak_ideal - t_start)
        warped[before_peak] = t_start + (t[before_peak] - t_start) * scale

    # After peak: [t_peak_ideal, t_end] → [t_peak_real, t_end]
    if t_end > t_peak_ideal and after_peak.any():
        scale = (t_end - t_peak_real) / (t_end - t_peak_ideal)
        warped[after_peak] = t_peak_real + (t[after_peak] - t_peak_ideal) * scale

    # Clamp to valid source range
    warped = np.clip(warped, t[0], t[-1])
    return warped


# ── Rotation warp (SLERP) ─────────────────────────────────────────────────────

def warp_rotation_sequence(
    R: np.ndarray,
    t: np.ndarray,
    t_peak_real: float,
    t_peak_ideal: float,
) -> np.ndarray:
    """Time-warp a (N, 3, 3) rotation sequence via SLERP.

    Builds a SLERP interpolant on the real time axis, then evaluates it at
    the warped source times so the peak shifts from t_peak_real to t_peak_ideal.
    """
    if abs(t_peak_ideal - t_peak_real) < 1e-6:
        return R.copy()

    rots  = Rotation.from_matrix(R)
    slerp = Slerp(t, rots)

    t_source = time_warp(t, t_peak_real, t_peak_ideal)
    return slerp(t_source).as_matrix()


# ── Amplitude scaling ─────────────────────────────────────────────────────────

def scale_rotation_amplitude(R: np.ndarray, k: float) -> np.ndarray:
    """Scale rotation amplitudes by factor k around the mean rotation.

    Steps:
      R_mean     = mean rotation of the sequence
      R_rel[i]   = R_mean.T @ R[i]          (deviation from mean)
      rotvec_scaled = rotvec(R_rel) * k
      R_scaled[i] = R_mean @ R_rel_scaled[i]
    """
    if abs(k - 1.0) < 1e-6:
        return R.copy()

    rots       = Rotation.from_matrix(R)
    R_mean_mat = rots.mean().as_matrix()        # (3, 3)
    R_mean_inv = R_mean_mat.T

    R_rel      = np.einsum("ij,njk->nik", R_mean_inv, R)          # (N, 3, 3)
    rv_rel     = Rotation.from_matrix(R_rel).as_rotvec()           # (N, 3)
    R_rel_sc   = Rotation.from_rotvec(rv_rel * k).as_matrix()      # (N, 3, 3)
    R_scaled   = np.einsum("ij,njk->nik", R_mean_mat, R_rel_sc)    # (N, 3, 3)
    return R_scaled


# ── Forward kinematics ────────────────────────────────────────────────────────

def forward_kinematics_kicking_leg(
    pts_orig: dict,
    R_thigh_ideal: np.ndarray,
    R_shank_ideal: np.ndarray,
    R_foot_ideal: np.ndarray,
    side: str,
) -> dict:
    """Reconstruct kicking leg positions via forward kinematics from HIP.

    HIP stays at original position. KNEE, ANKLE, HEEL, TOE are recomputed
    using the ideal rotation matrices and mean bone lengths.

    Returns a copy of pts_orig with the 4 kicking-leg parts updated.
    """
    ids = SIDE_IDS[side]
    pts = {k: v.copy() for k, v in pts_orig.items()}

    hip_pos    = pts_orig[ids["hip"]]     # (N, 3) — unchanged
    knee_orig  = pts_orig[ids["knee"]]
    ankle_orig = pts_orig[ids["ankle"]]
    heel_orig  = pts_orig[ids["heel"]]
    toe_orig   = pts_orig[ids["toe"]]

    # Mean bone lengths (metres)
    L_thigh = float(np.nanmean(np.linalg.norm(knee_orig  - hip_pos,    axis=1)))
    L_shank = float(np.nanmean(np.linalg.norm(ankle_orig - knee_orig,  axis=1)))
    L_foot  = float(np.nanmean(np.linalg.norm(toe_orig   - heel_orig,  axis=1)))

    # Heel offset expressed in original shank body frame (mean across frames)
    R_shank_orig, _ = build_shank_frame(pts_orig, side)
    heel_world_off  = heel_orig - ankle_orig                                  # (N, 3)
    heel_in_shank   = np.nanmean(
        np.einsum("nij,nj->ni", R_shank_orig.transpose(0, 2, 1), heel_world_off),
        axis=0,
    )  # (3,) constant offset in shank body frame

    # Forward kinematics
    # R[:, :, 1] = y-axis = segment primary (proximal → distal) direction in world frame
    KNEE  = hip_pos   + R_thigh_ideal[:, :, 1] * L_thigh
    ANKLE = KNEE      + R_shank_ideal[:, :, 1] * L_shank
    HEEL  = ANKLE     + np.einsum("nij,j->ni", R_shank_ideal, heel_in_shank)
    TOE   = HEEL      + R_foot_ideal[:, :, 1]  * L_foot

    pts[ids["knee"]]  = KNEE
    pts[ids["ankle"]] = ANKLE
    pts[ids["heel"]]  = HEEL
    pts[ids["toe"]]   = TOE
    return pts


# ── Smooth positions ──────────────────────────────────────────────────────────

def smooth_positions(pos: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay filter to (N, 3) position array."""
    if len(pos) < window:
        return pos.copy()
    result = pos.copy()
    for i in range(3):
        result[:, i] = savgol_filter(pos[:, i], window, poly)
    return result


# ── Verify WCS ────────────────────────────────────────────────────────────────

def verify_wcs(
    pts: dict,
    t: np.ndarray,
    side: str,
) -> tuple[int, dict]:
    """Re-run the full kinematics pipeline on pts to compute WCS.

    Returns
    -------
    wcs_ideal : int
    peak_info : dict
    """
    R_pelvis, _ = build_pelvis_frame(pts)
    R_thigh,  _ = build_thigh_frame(pts, side)
    R_shank,  _ = build_shank_frame(pts, side)
    R_foot,   _ = build_foot_frame(pts, side)

    R_hip_joint  = relative_rotation(R_pelvis, R_thigh)
    R_knee_joint = relative_rotation(R_thigh,  R_shank)

    om_pelvis = omega_from_R_incremental(R_pelvis,     DT)
    om_hip    = omega_from_R_incremental(R_hip_joint,  DT)
    om_knee   = omega_from_R_incremental(R_knee_joint, DT)
    om_foot   = omega_from_R_incremental(R_foot,       DT)

    mag_pelvis = np.linalg.norm(om_pelvis, axis=1)
    mag_foot   = np.linalg.norm(om_foot,   axis=1)

    pre = (t >= PRE_T_START) & (t <= PRE_T_END)

    def _peak(mag):
        if pre.sum() == 0:
            return np.nan, np.nan
        idx = int(np.argmax(mag[pre]))
        return float(t[pre][idx]), float(mag[pre][idx])

    pt_pelvis, po_pelvis = _peak(mag_pelvis)
    pt_hip,    _         = _peak(np.linalg.norm(om_hip,  axis=1))
    pt_knee,   _         = _peak(np.linalg.norm(om_knee, axis=1))
    pt_foot,   po_foot   = _peak(mag_foot)

    if any(np.isnan(v) for v in [pt_pelvis, pt_hip, pt_knee, pt_foot]):
        wcs = -1
    else:
        wcs = whipchain_score(pt_pelvis, pt_hip, pt_knee, pt_foot, po_pelvis, po_foot)

    amp_ratio = po_foot / max(po_pelvis, 1e-3)
    peak_info = {
        "peak_t_pelvis":     pt_pelvis,
        "peak_t_hip":        pt_hip,
        "peak_t_knee":       pt_knee,
        "peak_t_foot":       pt_foot,
        "peak_omega_pelvis": po_pelvis,
        "peak_omega_foot":   po_foot,
        "amp_ratio":         amp_ratio,
        "cascade_ok":        (
            sum([
                (pt_hip   - pt_pelvis) > 0,
                (pt_knee  - pt_hip)   > 0,
                (pt_foot  - pt_knee)  > 0,
            ]) if not any(np.isnan(v) for v in [pt_pelvis, pt_hip, pt_knee, pt_foot])
            else 0
        ),
    }
    return wcs, peak_info


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data …")
with open(KINEMATICS_PKL, "rb") as f:
    kinematics = pickle.load(f)

parts      = pd.read_parquet(PARTS_PATH)
detections = pd.read_csv(DETECTIONS_PATH)

print(f"  shot_kinematics : {len(kinematics)} shots")
print(f"  shots_parts     : {len(parts):,} rows")
print(f"  detections      : {len(detections)} shots")

# Index detections by event_id for fast lookup
det_by_event = detections.set_index("event_id")


# ── Main processing loop ──────────────────────────────────────────────────────

results   = []
summaries = []

for kin in kinematics:
    event_id      = int(kin["event_id"])
    shot_frame    = int(kin["shot_frame"])
    shooter       = int(kin["shooter_jersey"])
    kicking_side  = str(kin["kicking_side"])
    match_time    = str(kin["match_time"])
    wcs_orig      = int(kin["whipchain_score"])

    peak_t_pelvis = float(kin["peak_t_pelvis"])
    peak_t_hip    = float(kin["peak_t_hip"])
    peak_t_knee   = float(kin["peak_t_knee"])
    peak_t_foot   = float(kin["peak_t_foot"])
    peak_om_pelvis = float(kin["peak_omega_pelvis"])
    peak_om_foot   = float(kin["peak_omega_foot"])

    amp_ratio_orig = peak_om_foot / max(peak_om_pelvis, 1e-3)
    cascade_orig   = sum([
        (peak_t_hip   - peak_t_pelvis) > 0,
        (peak_t_knee  - peak_t_hip)   > 0,
        (peak_t_foot  - peak_t_knee)  > 0,
    ])

    try:
        # ── Reconstruct wide pivot for this shot ──────────────────────────────
        window = parts[
            (parts["jersey_number"] == shooter) &
            (parts["frame_number"]  >= shot_frame - PRE_FRAMES) &
            (parts["frame_number"]  <= shot_frame + POST_FRAMES)
        ].copy()

        window = (
            window.groupby(["frame_number", "body_part"], as_index=False)
            .agg({"x": "mean", "y": "mean", "z": "mean"})
        )
        wide = window.pivot(
            index="frame_number", columns="body_part", values=["x", "y", "z"]
        )
        wide = wide.interpolate(limit=3, limit_direction="both")
        frame_idx = wide.index.values        # (N,)
        N = len(frame_idx)
        t = (frame_idx - shot_frame) / FRAMERATE  # t=0 at contact

        pts_orig = _build_pts(wide)
        original_pts = _pts_to_array(pts_orig, N)

        # ── Shortcut: WCS already 100 ─────────────────────────────────────────
        if wcs_orig == 100:
            results.append({
                "event_id":       event_id,
                "match_time":     match_time,
                "shooter_jersey": shooter,
                "kicking_side":   kicking_side,
                "wcs_original":   wcs_orig,
                "wcs_ideal":      100,
                "t":              t,
                "original_pts":   original_pts,
                "ideal_pts":      original_pts.copy(),
                "ideal_peak_times": {
                    "pelvis": peak_t_pelvis,
                    "hip":    peak_t_hip,
                    "knee":   peak_t_knee,
                    "foot":   peak_t_foot,
                },
                "modification_flags": {
                    "time_warped":       False,
                    "amplitude_scaled":  False,
                    "scale_factor_foot": 1.0,
                },
            })
            summaries.append({
                "event_id": event_id, "match_time": match_time,
                "shooter_jersey": shooter, "kicking_side": kicking_side,
                "wcs_original": wcs_orig, "wcs_ideal": 100,
                "original_cascade_ok": cascade_orig == 3,
                "ideal_cascade_ok": True,
                "original_amp_ratio": round(amp_ratio_orig, 2),
                "ideal_amp_ratio": round(amp_ratio_orig, 2),
                "time_warped": False, "amplitude_scaled": False,
                "scale_factor_foot": 1.0,
            })
            print(f"  [{event_id:>4}] WCS={wcs_orig:>3} → 100  (already perfect, no change)")
            continue

        # ── Step A: Ideal peak time assignment ────────────────────────────────
        real_times = [peak_t_pelvis, peak_t_hip, peak_t_knee, peak_t_foot]
        sorted_times = sorted(real_times)

        # Enforce minimum 1-frame gap between consecutive ideal peak times
        for i in range(1, 4):
            if sorted_times[i] - sorted_times[i - 1] < MIN_GAP:
                sorted_times[i] = sorted_times[i - 1] + MIN_GAP

        tp_ideal, th_ideal, tk_ideal, tf_ideal = sorted_times

        needs_time_warp = (
            [tp_ideal, th_ideal, tk_ideal, tf_ideal] !=
            [peak_t_pelvis, peak_t_hip, peak_t_knee, peak_t_foot]
        )

        # ── Step B: Amplitude targets ─────────────────────────────────────────
        k_foot  = max(1.0, 3.0 * peak_om_pelvis / max(peak_om_foot, 1e-3))
        k_shank = 1.0 + (k_foot - 1.0) * 0.5
        needs_amp_scale = k_foot > 1.0 + 1e-6

        # ── Step C: Build original segment rotation matrices ──────────────────
        R_pelvis, _ = build_pelvis_frame(pts_orig)
        R_thigh,  _ = build_thigh_frame(pts_orig, kicking_side)
        R_shank,  _ = build_shank_frame(pts_orig, kicking_side)
        R_foot,   _ = build_foot_frame(pts_orig, kicking_side)

        # Time-warp each segment to its ideal peak time
        R_pelvis_w = warp_rotation_sequence(R_pelvis, t, peak_t_pelvis, tp_ideal)
        R_thigh_w  = warp_rotation_sequence(R_thigh,  t, peak_t_hip,    th_ideal)
        R_shank_w  = warp_rotation_sequence(R_shank,  t, peak_t_knee,   tk_ideal)
        R_foot_w   = warp_rotation_sequence(R_foot,   t, peak_t_foot,   tf_ideal)

        # ── Step D: Amplitude scaling (shank and foot only) ───────────────────
        R_shank_ideal = scale_rotation_amplitude(R_shank_w, k_shank)
        R_foot_ideal  = scale_rotation_amplitude(R_foot_w,  k_foot)
        # Pelvis and thigh: only time-warped
        R_thigh_ideal = R_thigh_w

        # ── Step E: Forward kinematics reconstruction ─────────────────────────
        pts_ideal = forward_kinematics_kicking_leg(
            pts_orig, R_thigh_ideal, R_shank_ideal, R_foot_ideal, kicking_side
        )

        # ── Step F: Smooth the 4 reconstructed positions ─────────────────────
        ids = SIDE_IDS[kicking_side]
        for part_id in [ids["knee"], ids["ankle"], ids["heel"], ids["toe"]]:
            pts_ideal[part_id] = smooth_positions(pts_ideal[part_id])

        ideal_arr = _pts_to_array(pts_ideal, N)

        # ── Step G: Verification ──────────────────────────────────────────────
        wcs_ideal, peak_info = verify_wcs(pts_ideal, t, kicking_side)

        # Regression safeguard: if the warp made things worse, revert to original
        if wcs_ideal < wcs_orig:
            pts_ideal  = pts_orig
            ideal_arr  = original_pts.copy()
            wcs_ideal  = wcs_orig
            peak_info["amp_ratio"]   = amp_ratio_orig
            peak_info["cascade_ok"]  = cascade_orig
            needs_time_warp    = False
            needs_amp_scale    = False
            k_foot             = 1.0
            k_shank            = 1.0
            tp_ideal, th_ideal, tk_ideal, tf_ideal = (
                peak_t_pelvis, peak_t_hip, peak_t_knee, peak_t_foot
            )

        verified = wcs_ideal >= 90

        results.append({
            "event_id":       event_id,
            "match_time":     match_time,
            "shooter_jersey": shooter,
            "kicking_side":   kicking_side,
            "wcs_original":   wcs_orig,
            "wcs_ideal":      wcs_ideal,
            "t":              t,
            "original_pts":   original_pts,
            "ideal_pts":      ideal_arr,
            "ideal_peak_times": {
                "pelvis": tp_ideal,
                "hip":    th_ideal,
                "knee":   tk_ideal,
                "foot":   tf_ideal,
            },
            "modification_flags": {
                "time_warped":       needs_time_warp,
                "amplitude_scaled":  needs_amp_scale,
                "scale_factor_foot": round(k_foot, 3),
            },
        })

        amp_ratio_ideal = peak_info["amp_ratio"]
        cascade_ideal   = peak_info["cascade_ok"]

        summaries.append({
            "event_id": event_id, "match_time": match_time,
            "shooter_jersey": shooter, "kicking_side": kicking_side,
            "wcs_original": wcs_orig, "wcs_ideal": wcs_ideal,
            "original_cascade_ok": cascade_orig == 3,
            "ideal_cascade_ok": cascade_ideal == 3,
            "original_amp_ratio": round(amp_ratio_orig, 2),
            "ideal_amp_ratio": round(amp_ratio_ideal, 2),
            "time_warped": needs_time_warp,
            "amplitude_scaled": needs_amp_scale,
            "scale_factor_foot": round(k_foot, 3),
        })

        status = "OK" if verified else "WARN"
        print(
            f"  [{event_id:>4}] WCS={wcs_orig:>3} → {wcs_ideal:>3}  "
            f"tw={'Y' if needs_time_warp else 'N'}  "
            f"as={'Y' if needs_amp_scale else 'N'}  "
            f"k_foot={k_foot:.2f}  "
            f"amp_ratio {amp_ratio_orig:.2f}→{amp_ratio_ideal:.2f}  "
            f"cascade {cascade_orig}/3→{cascade_ideal}/3  "
            f"[{status}]"
        )

    except Exception as exc:
        print(f"  [{event_id:>4}] FAIL  {exc}")
        traceback.print_exc()


# ── Write outputs ─────────────────────────────────────────────────────────────

print(f"\nProcessed: {len(results)} / {len(kinematics)} shots")

print(f"Writing {OUT_PKL} …")
with open(OUT_PKL, "wb") as f:
    pickle.dump(results, f)

summary_df = pd.DataFrame(summaries)
print(f"Writing {OUT_CSV} …")
summary_df.to_csv(OUT_CSV, index=False)


# ── Sanity report ─────────────────────────────────────────────────────────────

print("\n" + "═" * 72)
print("Phase 5 Sanity Report")
print("═" * 72)

if not summary_df.empty:
    verified_count = (summary_df["wcs_ideal"] >= 90).sum()
    print(f"\n  Shots processed     : {len(summary_df)} / {len(kinematics)}")
    print(f"  wcs_ideal ≥ 90      : {verified_count} / {len(summary_df)}")
    print(f"  Time-warped         : {summary_df['time_warped'].sum()}")
    print(f"  Amplitude-scaled    : {summary_df['amplitude_scaled'].sum()}")
    print()

    print(
        f"  {'Shot':>5}  {'Time':>6}  {'#':>3}  {'Side':<6}  "
        f"{'WCS_orig':>8}  {'WCS_ideal':>9}  "
        f"{'TimeWarp':>8}  {'AmpScale':>8}  {'Verified':>8}"
    )
    print("  " + "-" * 68)
    for _, row in summary_df.iterrows():
        ok = "OK" if row["wcs_ideal"] >= 90 else "WARN"
        print(
            f"  {row['event_id']:>5}  {row['match_time']:>6}  "
            f"#{row['shooter_jersey']:>2}  {row['kicking_side']:<6}  "
            f"{row['wcs_original']:>8}  {row['wcs_ideal']:>9}  "
            f"{'Yes' if row['time_warped'] else 'No':>8}  "
            f"{'Yes' if row['amplitude_scaled'] else 'No':>8}  "
            f"{ok:>8}"
        )

    print(f"\n  Final: {verified_count} / {len(summary_df)} shots achieved wcs_ideal ≥ 90")

print("\nDone.")
