"""
Phase 3: Shot Kinematics Pipeline
----------------------------------
Reads  : data/shots_parts.parquet  + data/shots_ball.parquet
         data/shot_detections.csv
Writes : data/shot_kinematics.pkl           (list of per-shot dicts, inc. np.arrays)
         data/shot_kinematics_summary.csv   (scalar columns only, for quick analysis)

Run:
    .venv/bin/python3 utils/compute_shot_kinematics.py
"""

import pickle
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from utils.kinematics import (
    build_pelvis_frame,
    build_thigh_frame,
    build_shank_frame,
    build_foot_frame,
    relative_rotation,
    omega_from_R_incremental,
    detect_kicking_side,
    whipchain_score,
)

# ── Config ────────────────────────────────────────────────────────────────────

PARTS_PATH      = "data/shots_parts.parquet"
BALL_PATH       = "data/shots_ball.parquet"
DETECTIONS_PATH = "data/shot_detections.csv"
PKL_OUT         = "data/shot_kinematics.pkl"
CSV_OUT         = "data/shot_kinematics_summary.csv"

FRAMERATE   = 25.0
DT          = 1.0 / FRAMERATE
PRE_FRAMES  = 75   # frames before shot_frame  (3 s)
POST_FRAMES = 25   # frames after  shot_frame  (1 s)
PRE_T_START = -2.0   # pre-contact window for peak extraction (s)
PRE_T_END   = -0.05
EXPLODE_THR = 100.0  # rad/s — flag as "exploding" if any segment exceeds this


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data …")
parts      = pd.read_parquet(PARTS_PATH)
ball_df    = pd.read_parquet(BALL_PATH)
detections = pd.read_csv(DETECTIONS_PATH)

print(f"  shots_parts  : {len(parts):,} rows")
print(f"  shots_ball   : {len(ball_df):,} rows")
print(f"  detections   : {len(detections)} shots")

# Index ball by frame_number for fast lookup
ball_df = ball_df.set_index("frame_number").sort_index()


# ── Helper: build pts dict from wide pivot ────────────────────────────────────

REQUIRED_PARTS = list(range(1, 22))   # body-part IDs 1–21


def _build_pts(wide: pd.DataFrame) -> dict:
    """Return {part_id: (N, 3) float array} for all body parts that exist."""
    pts = {}
    for bp in REQUIRED_PARTS:
        if bp in wide["x"].columns:
            pts[bp] = wide[["x", "y", "z"]].xs(bp, axis=1, level=1).values.astype(float)
    return pts


def _missing_required(pts: dict, side: str) -> list[int]:
    from utils.kinematics import SIDE_IDS, NECK, L_HIP, PELVIS, R_HIP
    ids = SIDE_IDS[side]
    needed = [NECK, L_HIP, PELVIS, R_HIP,
              ids["hip"], ids["knee"], ids["ankle"],
              ids["heel"], ids["toe"]]
    return [p for p in needed if p not in pts]


# ── Per-shot processor ────────────────────────────────────────────────────────

results   = []
summaries = []

skip_count = 0
fail_count = 0

for _, row in detections.iterrows():
    event_id     = int(row["event_id"])
    shot_frame   = int(row["shot_frame"])
    shooter      = row["shooter_jersey"]
    low_conf     = bool(row["low_confidence"])
    match_time   = str(row["match_time_mmss"])
    ball_speed   = float(row["ball_speed_after_mps"])
    event_code   = str(row["event_code"])

    # ── Step 1: Skip low-confidence or null shooter ───────────────────────────
    if low_conf or pd.isna(shooter):
        print(f"  [{event_id:>4}] SKIP  low_confidence={low_conf} shooter={shooter}")
        skip_count += 1
        continue

    shooter = int(shooter)

    try:
        # ── Step 2: Filter to shooter window ─────────────────────────────────
        window = parts[
            (parts["jersey_number"] == shooter) &
            (parts["frame_number"] >= shot_frame - PRE_FRAMES) &
            (parts["frame_number"] <= shot_frame + POST_FRAMES)
        ].copy()

        if len(window) == 0:
            print(f"  [{event_id:>4}] SKIP  no rows in window for shooter #{shooter}")
            skip_count += 1
            continue

        # ── Step 3: Wide pivot ────────────────────────────────────────────────
        # Overlapping shot windows can produce duplicate (frame, body_part) pairs;
        # collapse by averaging positions (negligible error for adjacent frames).
        window = (
            window.groupby(["frame_number", "body_part"], as_index=False)
            .agg({"x": "mean", "y": "mean", "z": "mean"})
        )
        wide = window.pivot(
            index="frame_number", columns="body_part", values=["x", "y", "z"]
        )
        wide = wide.interpolate(limit=3, limit_direction="both")
        frame_idx = wide.index.values        # (N,) frame numbers
        N = len(frame_idx)

        # ── Step 4: Detect kicking side ───────────────────────────────────────
        # Primary: parse shooter_body_part from CSV (reliable)
        body_part_str = str(row.get("shooter_body_part", ""))
        if "LEFT" in body_part_str.upper():
            kicking_side = "left"
        elif "RIGHT" in body_part_str.upper():
            kicking_side = "right"
        else:
            # Fallback: use ball-distance heuristic
            ball_win = ball_df.loc[ball_df.index.isin(frame_idx)]
            kicking_side = detect_kicking_side(wide, shot_frame,
                                               ball_df=ball_win if len(ball_win) else None)

        # ── Step 5: Build pts dict ────────────────────────────────────────────
        pts = _build_pts(wide)
        missing = _missing_required(pts, kicking_side)
        if missing:
            print(f"  [{event_id:>4}] SKIP  missing body parts {missing} for {kicking_side} side")
            skip_count += 1
            continue

        # ── Step 6: Build 4 segment frames ───────────────────────────────────
        R_pelvis, _ = build_pelvis_frame(pts)
        R_thigh,  _ = build_thigh_frame(pts, kicking_side)
        R_shank,  _ = build_shank_frame(pts, kicking_side)
        R_foot,   _ = build_foot_frame(pts, kicking_side)

        # ── Step 7: Relative joint rotations ─────────────────────────────────
        R_hip_joint  = relative_rotation(R_pelvis, R_thigh)
        R_knee_joint = relative_rotation(R_thigh,  R_shank)

        # ── Step 8: Angular velocities (incremental method, primary) ──────────
        om_pelvis = omega_from_R_incremental(R_pelvis,     DT)
        om_hip    = omega_from_R_incremental(R_hip_joint,  DT)
        om_knee   = omega_from_R_incremental(R_knee_joint, DT)
        om_foot   = omega_from_R_incremental(R_foot,       DT)

        mag_pelvis = np.linalg.norm(om_pelvis, axis=1)
        mag_hip    = np.linalg.norm(om_hip,    axis=1)
        mag_knee   = np.linalg.norm(om_knee,   axis=1)
        mag_foot   = np.linalg.norm(om_foot,   axis=1)

        # ── Step 9: Time axis & quality flag ─────────────────────────────────
        t = (frame_idx - shot_frame) / FRAMERATE      # t=0 at contact

        max_any = max(mag_pelvis.max(), mag_hip.max(),
                      mag_knee.max(), mag_foot.max())
        quality_flag = "exploding" if max_any > EXPLODE_THR else "ok"

        # ── Step 10: Peak extraction (pre-contact window) ─────────────────────
        pre = (t >= PRE_T_START) & (t <= PRE_T_END)

        def _peak(mag):
            if pre.sum() == 0:
                return np.nan, np.nan
            idx = int(np.argmax(mag[pre]))
            return float(t[pre][idx]), float(mag[pre][idx])

        peak_t_pelvis, peak_om_pelvis = _peak(mag_pelvis)
        peak_t_hip,    peak_om_hip    = _peak(mag_hip)
        peak_t_knee,   peak_om_knee   = _peak(mag_knee)
        peak_t_foot,   peak_om_foot   = _peak(mag_foot)

        # ── Step 11: WhipChain score ──────────────────────────────────────────
        if any(np.isnan(v) for v in
               [peak_t_pelvis, peak_t_hip, peak_t_knee, peak_t_foot]):
            wcs = -1
        else:
            wcs = whipchain_score(
                peak_t_pelvis, peak_t_hip, peak_t_knee, peak_t_foot,
                peak_om_pelvis, peak_om_foot,
            )

        # ── Step 12: Skeleton at contact ──────────────────────────────────────
        contact_mask = frame_idx == shot_frame
        skeleton_at_contact = {}
        if contact_mask.any():
            ci = int(np.argmax(contact_mask))
            for bp, arr in pts.items():
                skeleton_at_contact[bp] = arr[ci].tolist()
        else:
            # Nearest frame
            ci = int(np.argmin(np.abs(frame_idx - shot_frame)))
            for bp, arr in pts.items():
                skeleton_at_contact[bp] = arr[ci].tolist()

        # ── Step 13: Assemble output dict ────────────────────────────────────
        out = {
            # Identity
            "event_id":             event_id,
            "event_code":           event_code,
            "shot_frame":           shot_frame,
            "match_time":           match_time,
            "ball_speed_after_mps": ball_speed,
            "shooter_jersey":       shooter,
            "kicking_side":         kicking_side,
            "quality_flag":         quality_flag,

            # Time axis (t=0 at contact)
            "t":            t,

            # Angular velocity magnitudes (rad/s)
            "omega_pelvis": mag_pelvis,
            "omega_hip":    mag_hip,
            "omega_knee":   mag_knee,
            "omega_foot":   mag_foot,

            # Peaks in pre-contact window
            "peak_t_pelvis":     peak_t_pelvis,
            "peak_t_hip":        peak_t_hip,
            "peak_t_knee":       peak_t_knee,
            "peak_t_foot":       peak_t_foot,
            "peak_omega_pelvis": peak_om_pelvis,
            "peak_omega_hip":    peak_om_hip,
            "peak_omega_knee":   peak_om_knee,
            "peak_omega_foot":   peak_om_foot,

            # Score
            "whipchain_score": wcs,

            # 3D viz
            "skeleton_at_contact": skeleton_at_contact,
        }
        results.append(out)

        # ── Scalar summary row ────────────────────────────────────────────────
        gaps = [
            peak_t_hip   - peak_t_pelvis if not np.isnan(peak_t_hip)    else np.nan,
            peak_t_knee  - peak_t_hip    if not np.isnan(peak_t_knee)   else np.nan,
            peak_t_foot  - peak_t_knee   if not np.isnan(peak_t_foot)   else np.nan,
        ]
        summaries.append({
            "event_id":             event_id,
            "event_code":           event_code,
            "shot_frame":           shot_frame,
            "match_time":           match_time,
            "ball_speed_after_mps": ball_speed,
            "shooter_jersey":       shooter,
            "kicking_side":         kicking_side,
            "quality_flag":         quality_flag,
            "peak_t_pelvis":        peak_t_pelvis,
            "peak_t_hip":           peak_t_hip,
            "peak_t_knee":          peak_t_knee,
            "peak_t_foot":          peak_t_foot,
            "peak_omega_pelvis":    peak_om_pelvis,
            "peak_omega_hip":       peak_om_hip,
            "peak_omega_knee":      peak_om_knee,
            "peak_omega_foot":      peak_om_foot,
            "gap_pelvis_to_hip":    gaps[0],
            "gap_hip_to_knee":      gaps[1],
            "gap_knee_to_foot":     gaps[2],
            "whipchain_score":      wcs,
        })

        cascade_ok = sum(g > 0 for g in gaps if not np.isnan(g))
        print(
            f"  [{event_id:>4}] OK   #{shooter:>2} {kicking_side:<5}  "
            f"ω_pelvis={peak_om_pelvis:5.1f}  ω_knee={peak_om_knee:5.1f}  "
            f"ω_foot={peak_om_foot:5.1f}  cascade={cascade_ok}/3  "
            f"WCS={wcs:>3}  {quality_flag}"
        )

    except Exception as exc:
        print(f"  [{event_id:>4}] FAIL  {exc}")
        traceback.print_exc()
        fail_count += 1


# ── Write outputs ─────────────────────────────────────────────────────────────

print(f"\nProcessed: {len(results)} OK  |  {skip_count} skipped  |  {fail_count} failed")

print(f"\nWriting {PKL_OUT} …")
with open(PKL_OUT, "wb") as f:
    pickle.dump(results, f)

summary_df = pd.DataFrame(summaries)
print(f"Writing {CSV_OUT} …")
summary_df.to_csv(CSV_OUT, index=False)

# ── Phase 4 sanity checks ─────────────────────────────────────────────────────

print("\n" + "═" * 65)
print("Phase 4 Sanity Report")
print("═" * 65)

if not summary_df.empty:
    valid = summary_df[summary_df["quality_flag"] == "ok"]

    def _rng(col):
        return f"{col.min():.1f} – {col.max():.1f}  (mean {col.mean():.1f})"

    print(f"\nShots processed   : {len(summary_df)} / {len(detections)}")
    print(f"Quality ok        : {len(valid)}")
    print(f"Exploding         : {(summary_df['quality_flag']=='exploding').sum()}")

    print(f"\n{'Metric':<25} {'Range (ok shots)':<35} {'Expected'}")
    print("-" * 80)

    checks = [
        ("peak_omega_pelvis", "2–15 rad/s"),
        ("peak_omega_knee",   "8–30 rad/s"),
        ("peak_omega_foot",   "10–40 rad/s"),
        ("whipchain_score",   "0–100, mean > 30"),
    ]
    for col, expected in checks:
        if col in valid.columns and valid[col].notna().any():
            print(f"  {col:<23} {_rng(valid[col]):<35} {expected}")

    # Cascade direction check
    if "gap_pelvis_to_hip" in valid.columns:
        gaps_df = valid[["gap_pelvis_to_hip", "gap_hip_to_knee", "gap_knee_to_foot"]].dropna()
        frac_pos = (gaps_df > 0).mean()
        print(f"\n  Cascade positive-gap fractions (closer to 1.0 = better technique):")
        print(f"    pelvis→hip  : {frac_pos['gap_pelvis_to_hip']:.2f}")
        print(f"    hip→knee    : {frac_pos['gap_hip_to_knee']:.2f}")
        print(f"    knee→foot   : {frac_pos['gap_knee_to_foot']:.2f}")

    # Valid kinematics count (no NaN peaks, no exploding)
    valid_kin = valid[valid[["peak_omega_pelvis","peak_omega_knee","peak_omega_foot"]]
                      .notna().all(axis=1)]
    print(f"\n  Shots with valid kinematics : {len(valid_kin)} / {len(detections)}"
          f"  (target ≥ 15)")

    print(f"\n  mean WhipChain score : {summary_df['whipchain_score'].mean():.1f}"
          f"  (target mean > 30 for quality kicks)")

    print("\n── Per-shot summary ──")
    cols_show = ["event_id", "match_time", "shooter_jersey", "kicking_side",
                 "peak_omega_pelvis", "peak_omega_knee", "peak_omega_foot",
                 "whipchain_score", "quality_flag"]
    print(summary_df[cols_show].to_string(index=False))

print("\nDone.")
