"use client";

import { useMemo } from "react";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";
import { ShotFrame, ShotData } from "./types";
import PlayerSkeleton from "./PlayerSkeleton";

interface Props {
  frame: ShotFrame | null;
  jointNames: ShotData["joint_names"];
}

// Field dimensions: x ∈ [-56,56], y ∈ [-39,38]
// Map to Three.js: data x->x, data y->-z, height->y
const FIELD_W = 112; // meters
const FIELD_H = 77;

function PitchMarkings() {
  const lines = useMemo(() => {
    const segs: [THREE.Vector3, THREE.Vector3][] = [];

    const fx = (x: number) => x;
    const fy = (y: number) => -y; // data y -> Three.js -z

    // Boundary
    const corners = [
      [-56, -39],
      [56, -39],
      [56, 38],
      [-56, 38],
    ] as const;
    for (let i = 0; i < 4; i++) {
      const [x1, y1] = corners[i];
      const [x2, y2] = corners[(i + 1) % 4];
      segs.push([
        new THREE.Vector3(fx(x1), 0.01, fy(y1)),
        new THREE.Vector3(fx(x2), 0.01, fy(y2)),
      ]);
    }

    // Halfway line
    segs.push([
      new THREE.Vector3(0, 0.01, fy(-39)),
      new THREE.Vector3(0, 0.01, fy(38)),
    ]);

    // Center circle (radius ~9.15m)
    const R = 9.15;
    const N = 48;
    for (let i = 0; i < N; i++) {
      const a1 = (i / N) * Math.PI * 2;
      const a2 = ((i + 1) / N) * Math.PI * 2;
      segs.push([
        new THREE.Vector3(R * Math.cos(a1), 0.01, R * Math.sin(a1)),
        new THREE.Vector3(R * Math.cos(a2), 0.01, R * Math.sin(a2)),
      ]);
    }

    // Goal boxes (left and right, 18.32m wide, 5.5m deep)
    const goalBoxHalf = 20.15 / 2;
    const goalBoxDepth = 5.5;
    for (const side of [-1, 1]) {
      const xEdge = side * 56;
      const xInner = side * (56 - goalBoxDepth);
      segs.push([
        new THREE.Vector3(fx(xEdge), 0.01, fy(-goalBoxHalf)),
        new THREE.Vector3(fx(xInner), 0.01, fy(-goalBoxHalf)),
      ]);
      segs.push([
        new THREE.Vector3(fx(xInner), 0.01, fy(-goalBoxHalf)),
        new THREE.Vector3(fx(xInner), 0.01, fy(goalBoxHalf)),
      ]);
      segs.push([
        new THREE.Vector3(fx(xInner), 0.01, fy(goalBoxHalf)),
        new THREE.Vector3(fx(xEdge), 0.01, fy(goalBoxHalf)),
      ]);
    }

    // Penalty spots
    // (rendered as small cross lines)
    for (const side of [-1, 1]) {
      const px = side * 45;
      segs.push([
        new THREE.Vector3(px - 0.4, 0.01, 0),
        new THREE.Vector3(px + 0.4, 0.01, 0),
      ]);
      segs.push([
        new THREE.Vector3(px, 0.01, -0.4),
        new THREE.Vector3(px, 0.01, 0.4),
      ]);
    }

    return segs;
  }, []);

  return (
    <group>
      {/* Grass */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
        <planeGeometry args={[FIELD_W + 10, FIELD_H + 10]} />
        <meshLambertMaterial color="#1a5c1a" />
      </mesh>

      {/* White markings */}
      {lines.map(([a, b], i) => (
        <Line key={i} points={[a, b]} color="white" lineWidth={1} />
      ))}
    </group>
  );
}

export default function ShotScene({ frame, jointNames }: Props) {
  if (!frame) return null;

  const ballPos =
    frame.ball
      ? new THREE.Vector3(frame.ball[0], frame.ball[2], -frame.ball[1])
      : null;

  return (
    <>
      <ambientLight intensity={0.8} />
      <directionalLight position={[20, 40, 10]} intensity={0.6} />

      <OrbitControls
        enablePan
        maxPolarAngle={Math.PI / 2.2}
        target={[0, 0, 0]}
      />

      <PitchMarkings />

      {/* Players */}
      {frame.players.map((player, i) => (
        <PlayerSkeleton
          key={`${player.jersey}-${player.team}-${i}`}
          player={player}
          jointNames={jointNames}
        />
      ))}

      {/* Ball */}
      {ballPos && (
        <mesh position={ballPos}>
          <sphereGeometry args={[0.11, 12, 12]} />
          <meshStandardMaterial color="white" emissive="#ffff88" emissiveIntensity={0.3} />
        </mesh>
      )}
    </>
  );
}
