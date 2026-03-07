"use client";

import { useMemo } from "react";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import { PlayerFrame, ShotData } from "./types";
import GazeOverlay from "./GazeOverlay";

// Team colors
const TEAM_COLORS: Record<number, string> = {
  1: "#D2001A", // HOME - Ajax red
  2: "#4488FF", // TEAM_A - blue
  3: "#FFFFFF", // TEAM_B - white
  4: "#FFD700", // REFEREE - yellow
};

// Bone connections as pairs of joint names
const BONE_PAIRS: [string, string][] = [
  // Head
  ["NOSE", "LEFT_EAR"],
  ["NOSE", "RIGHT_EAR"],
  // Spine
  ["NOSE", "NECK"],
  ["NECK", "PELVIS"],
  // Left arm
  ["NECK", "LEFT_SHOULDER"],
  ["LEFT_SHOULDER", "LEFT_ELBOW"],
  ["LEFT_ELBOW", "LEFT_WRIST"],
  // Right arm
  ["NECK", "RIGHT_SHOULDER"],
  ["RIGHT_SHOULDER", "RIGHT_ELBOW"],
  ["RIGHT_ELBOW", "RIGHT_WRIST"],
  // Left leg
  ["PELVIS", "LEFT_HIP"],
  ["LEFT_HIP", "LEFT_KNEE"],
  ["LEFT_KNEE", "LEFT_ANKLE"],
  ["LEFT_ANKLE", "LEFT_HEEL"],
  ["LEFT_HEEL", "LEFT_TOE"],
  // Right leg
  ["PELVIS", "RIGHT_HIP"],
  ["RIGHT_HIP", "RIGHT_KNEE"],
  ["RIGHT_KNEE", "RIGHT_ANKLE"],
  ["RIGHT_ANKLE", "RIGHT_HEEL"],
  ["RIGHT_HEEL", "RIGHT_TOE"],
];

interface Props {
  player: PlayerFrame;
  jointNames: ShotData["joint_names"];
  showGaze?: boolean;
  ballPos?: THREE.Vector3 | null;
}

export default function PlayerSkeleton({ player, jointNames, showGaze, ballPos }: Props) {
  const color = TEAM_COLORS[player.team] ?? "#AAAAAA";

  // Build name -> [x, y, z] lookup, mapping data Y->Three.js Z (field XY -> XZ plane)
  const jointMap = useMemo(() => {
    const map: Record<string, THREE.Vector3> = {};
    player.pos.forEach((p, i) => {
      if (p) {
        const name = jointNames[i];
        // Data: x=along field, y=across field, z=height
        // Three.js: x=right, y=up, z=depth → map data x->x, data z->y (up), data y->-z
        map[name] = new THREE.Vector3(p[0], p[2], -p[1]);
      }
    });
    return map;
  }, [player.pos, jointNames]);

  // Build bone line segments
  const bones = useMemo(() => {
    const result: [THREE.Vector3, THREE.Vector3][] = [];
    for (const [a, b] of BONE_PAIRS) {
      const va = jointMap[a];
      const vb = jointMap[b];
      if (va && vb) {
        result.push([va, vb]);
      }
    }
    return result;
  }, [jointMap]);

  const nosePos = jointMap["NOSE"];
  const jointPositions = Object.values(jointMap);

  return (
    <group>
      {/* Bone lines */}
      {bones.map(([a, b], i) => (
        <Line
          key={i}
          points={[a, b]}
          color={color}
          lineWidth={1.5}
        />
      ))}

      {/* Joint dots */}
      {jointPositions.map((pos, i) => (
        <mesh key={i} position={pos}>
          <sphereGeometry args={[0.06, 6, 6]} />
          <meshBasicMaterial color={color} />
        </mesh>
      ))}

      {/* Head sphere at NOSE */}
      {nosePos && (
        <mesh position={nosePos}>
          <sphereGeometry args={[0.15, 8, 8]} />
          <meshBasicMaterial color={color} transparent opacity={0.7} />
        </mesh>
      )}

      {/* Gaze overlay */}
      {showGaze && ballPos && (
        <GazeOverlay jointMap={jointMap} ballPos={ballPos} />
      )}
    </group>
  );
}
