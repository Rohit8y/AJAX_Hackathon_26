"use client";

import { useMemo, useRef } from "react";
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

// Emissive colors (subtler version)
const TEAM_EMISSIVE: Record<number, string> = {
  1: "#D2001A",
  2: "#4488FF",
  3: "#888888",
  4: "#FFD700",
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

const _tempVec = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _quat = new THREE.Quaternion();

function CylinderBone({ start, end, color, emissive }: { start: THREE.Vector3; end: THREE.Vector3; color: string; emissive: string }) {
  const length = start.distanceTo(end);
  if (length < 0.001) return null;

  const mid = _tempVec.copy(start).add(end).multiplyScalar(0.5);
  const dir = new THREE.Vector3().subVectors(end, start).normalize();
  _quat.setFromUnitVectors(_up, dir);

  return (
    <mesh position={[mid.x, mid.y, mid.z]} quaternion={_quat.clone()}>
      <cylinderGeometry args={[0.025, 0.025, length, 6]} />
      <meshStandardMaterial
        color={color}
        emissive={emissive}
        emissiveIntensity={0.3}
        roughness={0.6}
        metalness={0.2}
      />
    </mesh>
  );
}

export default function PlayerSkeleton({ player, jointNames, showGaze, ballPos }: Props) {
  const color = TEAM_COLORS[player.team] ?? "#AAAAAA";
  const emissive = TEAM_EMISSIVE[player.team] ?? "#444444";

  // Build name -> [x, y, z] lookup
  const jointMap = useMemo(() => {
    const map: Record<string, THREE.Vector3> = {};
    player.pos.forEach((p, i) => {
      if (p) {
        const name = jointNames[i];
        map[name] = new THREE.Vector3(p[0], p[2], -p[1]);
      }
    });
    return map;
  }, [player.pos, jointNames]);

  // Build bone pairs
  const bones = useMemo(() => {
    const result: { start: THREE.Vector3; end: THREE.Vector3 }[] = [];
    for (const [a, b] of BONE_PAIRS) {
      const va = jointMap[a];
      const vb = jointMap[b];
      if (va && vb) {
        result.push({ start: va, end: vb });
      }
    }
    return result;
  }, [jointMap]);

  const nosePos = jointMap["NOSE"];
  const jointPositions = Object.values(jointMap);

  return (
    <group>
      {/* 3D cylinder bones */}
      {bones.map((bone, i) => (
        <CylinderBone
          key={i}
          start={bone.start}
          end={bone.end}
          color={color}
          emissive={emissive}
        />
      ))}

      {/* Joint spheres */}
      {jointPositions.map((pos, i) => (
        <mesh key={i} position={pos}>
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshStandardMaterial
            color={color}
            emissive={emissive}
            emissiveIntensity={0.2}
            roughness={0.5}
          />
        </mesh>
      ))}

      {/* Head sphere at NOSE */}
      {nosePos && (
        <mesh position={nosePos}>
          <sphereGeometry args={[0.15, 10, 10]} />
          <meshStandardMaterial
            color={color}
            emissive={emissive}
            emissiveIntensity={0.3}
            transparent
            opacity={0.8}
            roughness={0.4}
          />
        </mesh>
      )}

      {/* Gaze overlay */}
      {showGaze && ballPos && (
        <GazeOverlay jointMap={jointMap} ballPos={ballPos} />
      )}
    </group>
  );
}
