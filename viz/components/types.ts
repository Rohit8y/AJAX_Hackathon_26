export interface ShotFrame {
  f: number; // frame_number
  ball: [number, number, number] | null;
  players: PlayerFrame[];
}

export interface PlayerFrame {
  jersey: number;
  team: number; // 1=HOME, 2=TEAM_A, 3=TEAM_B, 4=REFEREE
  pos: ([number, number, number] | null)[]; // 21 joints in joint_names order
}

export interface ShotData {
  fps: number;
  joint_names: string[];
  frames: ShotFrame[];
}
