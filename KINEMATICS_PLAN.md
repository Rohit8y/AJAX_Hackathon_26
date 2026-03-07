---                                                                                                                    
  Part 1: The Kinematics Pipeline (Primer → Code mapping)                                                                
                                                                                                          
  Your data state: shots_parts.parquet has all 21 body parts in metres for every frame. shot_detections.csv has the      
  contact frame + shooter for 25 shots. You need to go from this to angular velocity time series.                      
                                                                                                                         
  ---                                                                                                                    
  Step 1 — Wide pivot per shot (Primer §5.0.2)                                                                           
                                                                                                                         
  For each row in shot_detections.csv, filter shots_parts.parquet to [shot_frame - 75, shot_frame + 25] (3s before, 1s   
  after at 25Hz), then pivot:                                                                                            
                                                                                                                         
  # Filter to shooter, window around shot_frame                                                                          
  window = parts_df[
      (parts_df.jersey_number == shooter_jersey) &
      (parts_df.frame_number >= shot_frame - 75) &
      (parts_df.frame_number <= shot_frame + 25)
  ]
  wide = window.pivot(index="frame_number", columns="body_part", values=["x","y","z"])
  wide = wide.interpolate(limit=3, limit_direction="both")  # fill short gaps

  ---
  Step 2 — Build rotation matrices (Primer §5.1, §8.2, §11.1)

  Four segments using build_frame_from_points(p0, p1, p2) → each returns (N, 3, 3):

  ┌─────────┬───────────┬────────────┬───────────────────────────────────────────────────┐
  │ Segment │    p0     │     p1     │                  p2 (plane ref)                   │
  ├─────────┼───────────┼────────────┼───────────────────────────────────────────────────┤
  │ Pelvis  │ —         │ —          │ build_frame_from_xz(R_hip - L_hip, neck - pelvis) │
  ├─────────┼───────────┼────────────┼───────────────────────────────────────────────────┤
  │ Thigh   │ hip (13)  │ knee (15)  │ pelvis (12)                                       │
  ├─────────┼───────────┼────────────┼───────────────────────────────────────────────────┤
  │ Shank   │ knee (15) │ ankle (17) │ toe (21)                                          │
  ├─────────┼───────────┼────────────┼───────────────────────────────────────────────────┤
  │ Foot    │ heel (20) │ toe (21)   │ ankle (17)                                        │
  └─────────┴───────────┴────────────┴───────────────────────────────────────────────────┘

  After building each R_raw, apply forward_fill_rotations then enforce_z_continuity to kill any 180° flips.

  ---
  Step 3 — Relative joint rotations (Primer §6)

  R_knee_joint = R_thigh.T @ R_shank   # shape (N,3,3) via einsum
  R_hip_joint  = R_pelvis.T @ R_thigh

  This gives you "how the shank moves relative to the thigh" — exactly what Section 8.3 calls R_knee.

  ---
  Step 4 — Angular velocities (Primer §7.2 / §8.3)

  Use omega_from_R(R, dt) on each:

  dt = 1/25  # 25 Hz
  omega_pelvis = omega_from_R(R_pelvis, dt)   # (N,3) rad/s global
  omega_hip    = omega_from_R(R_hip_joint, dt)
  omega_knee   = omega_from_R(R_knee_joint, dt)
  omega_foot   = omega_from_R(R_foot, dt)

  # Magnitudes — the "heartbeat" signal
  mag_pelvis = np.linalg.norm(omega_pelvis, axis=1)
  mag_hip    = np.linalg.norm(omega_hip,    axis=1)
  mag_knee   = np.linalg.norm(omega_knee,   axis=1)
  mag_foot   = np.linalg.norm(omega_foot,   axis=1)

  If you see exploding values, switch to omega_from_R_incremental (§7.4) — more stable at 25Hz.

  ---
  Step 5 — Normalize time, extract peaks

  t = (frames - shot_frame) / 25.0  # t=0 is contact

  # For each segment, find peak in pre-contact window [-2s, -0.05s]
  pre = (t >= -2.0) & (t <= -0.05)
  peak_t_pelvis = t[pre][np.argmax(mag_pelvis[pre])]
  # ... repeat for hip, knee, foot

  ---
  Part 2: Output Schema + Fancy Demo Plan

  Kinematics module output — one dict per shot

  {
      # Identity
      "event_id":    int,
      "shot_frame":  int,
      "match_time":  "06:12",
      "ball_speed_after_mps": 24.5,  # already in shot_detections.csv
      "shooter_jersey": int,
      "kicking_side": "right",  # detect from which foot was closer

      # Time axis (aligned to contact)
      "t":           np.array,  # seconds, t=0 = contact frame

      # The four cascade signals (rad/s magnitudes)
      "omega_pelvis": np.array,
      "omega_hip":    np.array,
      "omega_knee":   np.array,
      "omega_foot":   np.array,

      # Peak values (in pre-contact window)
      "peak_t_pelvis":    float,  # time of peak (negative = before contact)
      "peak_t_hip":       float,
      "peak_t_knee":      float,
      "peak_t_foot":      float,
      "peak_omega_pelvis": float, # rad/s
      "peak_omega_hip":    float,
      "peak_omega_knee":   float,
      "peak_omega_foot":   float,

      # WhipChain score (0–100)
      "whipchain_score": float,
      # Skeleton at contact (for 3D viz)
      "skeleton_at_contact": dict,  # {part_id: [x,y,z]}
  }

  WhipChain Score formula:
  # Correct cascade = each segment peaks AFTER the previous (all gaps positive)
  gaps = [t_hip - t_pelvis, t_knee - t_hip, t_foot - t_knee]
  sequence_score = sum(g > 0 for g in gaps) / 3   # 0→1

  # Distal amplification: foot should spin faster than pelvis
  amp_ratio = peak_omega_foot / max(peak_omega_pelvis, 1e-3)
  amp_score = min(amp_ratio / 3.0, 1.0)  # normalised, cap at 1

  whipchain_score = round((0.6 * sequence_score + 0.4 * amp_score) * 100)