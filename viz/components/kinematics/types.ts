export interface KinematicsShot {
  id: number;
  event_code: string;
  team: string;
  is_goal: boolean;
  match_time: string;
  shooter_jersey: number;
  kicking_side: string;
  ball_speed: number;
  whipchain_score: number;
  quality: string;
  // Time series
  t: number[];
  omega_pelvis: number[];
  omega_hip: number[];
  omega_knee: number[];
  omega_foot: number[];
  // Peaks
  peak_t_pelvis: number;
  peak_t_hip: number;
  peak_t_knee: number;
  peak_t_foot: number;
  peak_omega_pelvis: number;
  peak_omega_hip: number;
  peak_omega_knee: number;
  peak_omega_foot: number;
  // Skeleton at contact: part_id string -> [x,y,z]
  skeleton: Record<string, [number, number, number]>;
}

export interface KinematicsData {
  shots: KinematicsShot[];
}

// Body part mapping for skeleton rendering
export const BODY_PART_NAMES: Record<number, string> = {
  1: "LEFT_EAR", 2: "NOSE", 3: "RIGHT_EAR",
  4: "LEFT_SHOULDER", 5: "NECK", 6: "RIGHT_SHOULDER",
  7: "LEFT_ELBOW", 8: "RIGHT_ELBOW",
  9: "LEFT_WRIST", 10: "RIGHT_WRIST",
  11: "LEFT_HIP", 12: "PELVIS", 13: "RIGHT_HIP",
  14: "LEFT_KNEE", 15: "RIGHT_KNEE",
  16: "LEFT_ANKLE", 17: "RIGHT_ANKLE",
  18: "LEFT_HEEL", 19: "LEFT_TOE",
  20: "RIGHT_HEEL", 21: "RIGHT_TOE",
};

export const BONE_PAIRS: [number, number][] = [
  [2, 1], [2, 3],       // Head
  [2, 5], [5, 12],      // Spine
  [5, 4], [4, 7], [7, 9],   // Left arm
  [5, 6], [6, 8], [8, 10],  // Right arm
  [12, 11], [11, 14], [14, 16], [16, 18], [18, 19], // Left leg
  [12, 13], [13, 15], [15, 17], [17, 20], [20, 21], // Right leg
];

// Kicking leg part IDs
export const KICKING_LEG_PARTS = {
  left: {
    thigh: [11, 14],   // LEFT_HIP -> LEFT_KNEE
    shank: [14, 16],   // LEFT_KNEE -> LEFT_ANKLE
    foot: [16, 18, 19], // LEFT_ANKLE, LEFT_HEEL, LEFT_TOE
  },
  right: {
    thigh: [13, 15],   // RIGHT_HIP -> RIGHT_KNEE
    shank: [15, 17],   // RIGHT_KNEE -> RIGHT_ANKLE
    foot: [17, 20, 21], // RIGHT_ANKLE, RIGHT_HEEL, RIGHT_TOE
  },
};

export const CASCADE_COLORS = {
  pelvis: "#60A5FA",
  hip: "#34D399",
  knee: "#FB923C",
  foot: "#F87171",
} as const;

export type CascadeSegment = "pelvis" | "hip" | "knee" | "foot";

// Ideal skeleton types
export interface IdealShotData {
  event_id: number;
  match_time: string;
  shooter_jersey: number;
  kicking_side: string;
  wcs_original: number;
  wcs_ideal: number;
  t: number[];
  original_frames: [number, number, number][][]; // frames x 21 joints x [x,y,z]
  ideal_frames: [number, number, number][][];
  ideal_peak_times: Record<CascadeSegment, number>;
  modification_flags: {
    time_warped: boolean;
    amplitude_scaled: boolean;
    scale_factor_foot: number;
  };
  changed_joints: number[];
}

export interface IdealKinematicsData {
  shots: IdealShotData[];
}
