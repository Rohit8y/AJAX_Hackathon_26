"use client";

import { useMemo, useRef } from "react";
import { OrbitControls, Line, Sparkles, Trail } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";
import * as THREE from "three";
import { ShotFrame, ShotData } from "./types";
import PlayerSkeleton from "./PlayerSkeleton";

interface Props {
  frame: ShotFrame | null;
  jointNames: ShotData["joint_names"];
  showGaze: boolean;
}

// Field dimensions: x in [-56,56], y in [-39,38]
const FIELD_W = 112;
const FIELD_H = 77;

function PitchMarkings() {
  const lines = useMemo(() => {
    const segs: [THREE.Vector3, THREE.Vector3][] = [];

    const fx = (x: number) => x;
    const fy = (y: number) => -y;

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

    // Goal boxes (18.32m wide, 5.5m deep)
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
        <meshStandardMaterial color="#1a5c1a" roughness={0.9} metalness={0.0} />
      </mesh>

      {/* White markings */}
      {lines.map(([a, b], i) => (
        <Line key={i} points={[a, b]} color="white" lineWidth={1.5} />
      ))}
    </group>
  );
}

function Ball({ position }: { position: THREE.Vector3 }) {
  const meshRef = useRef<THREE.Mesh>(null!);

  return (
    <group>
      <Trail
        width={0.6}
        length={8}
        color="#ffaa44"
        attenuation={(t) => t * t}
      >
        <mesh ref={meshRef} position={position}>
          <sphereGeometry args={[0.11, 16, 16]} />
          <meshStandardMaterial
            color="white"
            emissive="#ffaa44"
            emissiveIntensity={1.5}
          />
        </mesh>
      </Trail>
      {/* Pool of light on pitch */}
      <pointLight
        position={[position.x, position.y + 0.5, position.z]}
        color="#ffdd88"
        intensity={3}
        distance={8}
        decay={2}
      />
    </group>
  );
}

export default function ShotScene({ frame, jointNames, showGaze }: Props) {
  if (!frame) return null;

  const ballPos =
    frame.ball
      ? new THREE.Vector3(frame.ball[0], frame.ball[2], -frame.ball[1])
      : null;

  return (
    <>
      {/* Fog for depth */}
      <fog attach="fog" args={["#0a0a14", 40, 120]} />

      {/* Lighting — dramatic stadium feel */}
      <ambientLight intensity={0.3} />
      <directionalLight position={[20, 40, 10]} intensity={0.4} />

      {/* Stadium floodlights at corners */}
      <spotLight position={[-56, 35, -39]} angle={0.6} penumbra={0.8} intensity={40} color="#eee8d5" distance={140} castShadow={false} />
      <spotLight position={[56, 35, -39]} angle={0.6} penumbra={0.8} intensity={40} color="#eee8d5" distance={140} castShadow={false} />
      <spotLight position={[-56, 35, 39]} angle={0.6} penumbra={0.8} intensity={40} color="#eee8d5" distance={140} castShadow={false} />
      <spotLight position={[56, 35, 39]} angle={0.6} penumbra={0.8} intensity={40} color="#eee8d5" distance={140} castShadow={false} />

      <OrbitControls
        enablePan
        enableDamping
        dampingFactor={0.05}
        maxPolarAngle={Math.PI / 2.2}
        target={[0, 0, 0]}
      />

      <PitchMarkings />

      {/* Floating dust particles */}
      <Sparkles count={80} scale={[100, 20, 70]} size={1.5} speed={0.3} opacity={0.15} color="#ffffff" />

      {/* Players */}
      {frame.players.map((player, i) => (
        <PlayerSkeleton
          key={`${player.jersey}-${player.team}-${i}`}
          player={player}
          jointNames={jointNames}
          showGaze={showGaze}
          ballPos={ballPos}
        />
      ))}

      {/* Ball with trail and glow */}
      {ballPos && <Ball position={ballPos} />}

      {/* Postprocessing */}
      <EffectComposer>
        <Bloom
          intensity={0.4}
          luminanceThreshold={0.6}
          luminanceSmoothing={0.9}
          mipmapBlur
        />
        <Vignette eskil={false} offset={0.1} darkness={0.7} />
      </EffectComposer>
    </>
  );
}
