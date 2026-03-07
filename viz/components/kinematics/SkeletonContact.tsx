"use client";

import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";
import { KinematicsShot, BONE_PAIRS, KICKING_LEG_PARTS } from "./types";

interface Props {
  shot: KinematicsShot;
}

function SkeletonMesh({ shot }: Props) {
  const side = shot.kicking_side as "left" | "right";
  const kickParts = KICKING_LEG_PARTS[side];
  const kickPartIds = new Set([...kickParts.thigh, ...kickParts.shank, ...kickParts.foot]);

  // Build joint positions: data [x,y,z] -> Three.js [x, z, -y]
  const joints = useMemo(() => {
    const map: Record<number, THREE.Vector3> = {};
    for (const [id, pos] of Object.entries(shot.skeleton)) {
      map[Number(id)] = new THREE.Vector3(pos[0], pos[2], -pos[1]);
    }
    return map;
  }, [shot.skeleton]);

  // Center skeleton around origin
  const center = useMemo(() => {
    const pelvis = joints[12]; // PELVIS
    return pelvis ? new THREE.Vector3(pelvis.x, 0, pelvis.z) : new THREE.Vector3();
  }, [joints]);

  // Bone colors based on kicking leg
  const boneSegments = useMemo(() => {
    return BONE_PAIRS.map(([a, b]) => {
      const va = joints[a];
      const vb = joints[b];
      if (!va || !vb) return null;

      const aKick = kickPartIds.has(a);
      const bKick = kickPartIds.has(b);
      let color = "#9ca3af"; // default gray

      if (aKick && bKick) {
        // Part of kicking leg
        if (kickParts.foot.includes(a) || kickParts.foot.includes(b)) {
          color = "#ef4444"; // red for foot
        } else if (kickParts.shank.includes(a) && kickParts.shank.includes(b)) {
          color = "#f97316"; // orange for shank
        } else {
          color = "#fb923c"; // light orange for thigh
        }
      }

      return {
        points: [
          new THREE.Vector3(va.x - center.x, va.y, va.z - center.z),
          new THREE.Vector3(vb.x - center.x, vb.y, vb.z - center.z),
        ] as [THREE.Vector3, THREE.Vector3],
        color,
        emissive: aKick && bKick,
      };
    }).filter(Boolean) as { points: [THREE.Vector3, THREE.Vector3]; color: string; emissive: boolean }[];
  }, [joints, center, kickPartIds, kickParts]);

  const jointPositions = useMemo(() => {
    return Object.entries(joints).map(([id, pos]) => ({
      id: Number(id),
      pos: new THREE.Vector3(pos.x - center.x, pos.y, pos.z - center.z),
      isKick: kickPartIds.has(Number(id)),
    }));
  }, [joints, center, kickPartIds]);

  const nosePos = joints[2];
  const noseCentered = nosePos ? new THREE.Vector3(nosePos.x - center.x, nosePos.y, nosePos.z - center.z) : null;

  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} />
      <pointLight position={[0, 2, 0]} intensity={0.4} color="#f97316" />

      {/* Grid floor */}
      <gridHelper args={[4, 10, "#1f2937", "#111827"]} position={[0, 0, 0]} />

      {/* Bone lines */}
      {boneSegments.map((seg, i) => (
        <Line
          key={i}
          points={seg.points}
          color={seg.color}
          lineWidth={seg.emissive ? 3 : 1.5}
        />
      ))}

      {/* Joint dots */}
      {jointPositions.map(({ id, pos, isKick }) => (
        <mesh key={id} position={pos}>
          <sphereGeometry args={[0.04, 6, 6]} />
          <meshStandardMaterial
            color={isKick ? "#f97316" : "#9ca3af"}
            emissive={isKick ? "#f97316" : "#000000"}
            emissiveIntensity={isKick ? 0.5 : 0}
          />
        </mesh>
      ))}

      {/* Head */}
      {noseCentered && (
        <mesh position={noseCentered}>
          <sphereGeometry args={[0.1, 8, 8]} />
          <meshStandardMaterial color="#d1d5db" transparent opacity={0.6} />
        </mesh>
      )}

      <OrbitControls
        enablePan={false}
        minDistance={1.5}
        maxDistance={5}
        autoRotate
        autoRotateSpeed={1}
        target={[0, 0.8, 0]}
      />
    </>
  );
}

export default function SkeletonContact({ shot }: Props) {
  return (
    <div className="w-full h-full min-h-[300px] rounded-lg overflow-hidden border border-gray-800">
      <Canvas
        camera={{ position: [1.5, 1.5, 2], fov: 45, near: 0.1, far: 50 }}
        style={{ background: "#0a0a14" }}
      >
        <SkeletonMesh shot={shot} />
      </Canvas>
    </div>
  );
}
