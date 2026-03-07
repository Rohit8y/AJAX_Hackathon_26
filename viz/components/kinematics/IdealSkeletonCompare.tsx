"use client";

import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import { IdealShotData, BONE_PAIRS, BODY_PART_NAMES } from "./types";

interface Props {
  idealShot: IdealShotData;
}

const _up = new THREE.Vector3(0, 1, 0);

function CylinderBone({ start, end, color, emissive, emissiveIntensity }: {
  start: THREE.Vector3; end: THREE.Vector3; color: string; emissive: string; emissiveIntensity: number;
}) {
  const length = start.distanceTo(end);
  if (length < 0.001) return null;
  const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
  const dir = new THREE.Vector3().subVectors(end, start).normalize();
  const quat = new THREE.Quaternion().setFromUnitVectors(_up, dir);

  return (
    <mesh position={[mid.x, mid.y, mid.z]} quaternion={quat}>
      <cylinderGeometry args={[0.02, 0.02, length, 6]} />
      <meshStandardMaterial color={color} emissive={emissive} emissiveIntensity={emissiveIntensity} roughness={0.5} />
    </mesh>
  );
}

function SkeletonGroup({ joints, color, emissive, changedJoints, label, offsetX }: {
  joints: Record<number, THREE.Vector3>;
  color: string;
  emissive: string;
  changedJoints: Set<number>;
  label: string;
  offsetX: number;
}) {
  const center = joints[12] // PELVIS
    ? new THREE.Vector3(joints[12].x, 0, joints[12].z)
    : new THREE.Vector3();

  const bones = useMemo(() => {
    return BONE_PAIRS.map(([a, b]) => {
      const va = joints[a];
      const vb = joints[b];
      if (!va || !vb) return null;
      return {
        start: new THREE.Vector3(va.x - center.x + offsetX, va.y, va.z - center.z),
        end: new THREE.Vector3(vb.x - center.x + offsetX, vb.y, vb.z - center.z),
        highlighted: changedJoints.has(a) || changedJoints.has(b),
      };
    }).filter(Boolean) as { start: THREE.Vector3; end: THREE.Vector3; highlighted: boolean }[];
  }, [joints, center, offsetX, changedJoints]);

  const jointPositions = useMemo(() => {
    return Object.entries(joints).map(([id, pos]) => ({
      id: Number(id),
      pos: new THREE.Vector3(pos.x - center.x + offsetX, pos.y, pos.z - center.z),
      highlighted: changedJoints.has(Number(id)),
    }));
  }, [joints, center, offsetX, changedJoints]);

  const nosePos = joints[2];
  const noseCentered = nosePos
    ? new THREE.Vector3(nosePos.x - center.x + offsetX, nosePos.y, nosePos.z - center.z)
    : null;

  return (
    <group>
      {bones.map((bone, i) => (
        <CylinderBone
          key={i}
          start={bone.start}
          end={bone.end}
          color={bone.highlighted ? "#fb923c" : color}
          emissive={bone.highlighted ? "#fb923c" : emissive}
          emissiveIntensity={bone.highlighted ? 0.5 : 0.2}
        />
      ))}
      {jointPositions.map(({ id, pos, highlighted }) => (
        <mesh key={id} position={pos}>
          <sphereGeometry args={[0.04, 8, 8]} />
          <meshStandardMaterial
            color={highlighted ? "#fb923c" : color}
            emissive={highlighted ? "#fb923c" : emissive}
            emissiveIntensity={highlighted ? 0.5 : 0.15}
          />
        </mesh>
      ))}
      {noseCentered && (
        <mesh position={noseCentered}>
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshStandardMaterial color={color} emissive={emissive} emissiveIntensity={0.2} transparent opacity={0.7} />
        </mesh>
      )}
      {/* Label */}
      {noseCentered && (
        <Html position={[offsetX, 2.2, 0]} center>
          <div className="text-xs font-semibold px-2 py-0.5 rounded-full whitespace-nowrap"
            style={{ background: "rgba(0,0,0,0.7)", color, border: `1px solid ${color}40` }}>
            {label}
          </div>
        </Html>
      )}
    </group>
  );
}

function AnimatedScene({ idealShot, frameIndex }: { idealShot: IdealShotData; frameIndex: number }) {
  const changedSet = useMemo(() => new Set(idealShot.changed_joints), [idealShot.changed_joints]);

  const buildJoints = useCallback((frames: [number, number, number][][], fi: number) => {
    const map: Record<number, THREE.Vector3> = {};
    const frame = frames[fi];
    if (!frame) return map;
    frame.forEach((pos, i) => {
      // data [x,y,z] -> Three.js [x, z, -y]
      map[i + 1] = new THREE.Vector3(pos[0], pos[2], -pos[1]);
    });
    return map;
  }, []);

  const originalJoints = useMemo(() => buildJoints(idealShot.original_frames, frameIndex), [idealShot.original_frames, frameIndex, buildJoints]);
  const idealJoints = useMemo(() => buildJoints(idealShot.ideal_frames, frameIndex), [idealShot.ideal_frames, frameIndex, buildJoints]);

  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 10, 5]} intensity={0.7} />
      <pointLight position={[-1.2, 2, 0]} intensity={0.3} color="#D2001A" />
      <pointLight position={[1.2, 2, 0]} intensity={0.3} color="#22d3ee" />

      <gridHelper args={[6, 12, "#1f2937", "#111827"]} position={[0, 0, 0]} />

      <SkeletonGroup
        joints={originalJoints}
        color="#D2001A"
        emissive="#D2001A"
        changedJoints={changedSet}
        label="Original"
        offsetX={-1.2}
      />
      <SkeletonGroup
        joints={idealJoints}
        color="#22d3ee"
        emissive="#22d3ee"
        changedJoints={changedSet}
        label="Ideal"
        offsetX={1.2}
      />

      <OrbitControls
        enablePan={false}
        minDistance={2}
        maxDistance={6}
        target={[0, 0.8, 0]}
      />
    </>
  );
}

export default function IdealSkeletonCompare({ idealShot }: Props) {
  const totalFrames = idealShot.original_frames.length;
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setFrameIndex(0);
    setPlaying(true);
  }, [idealShot]);

  useEffect(() => {
    if (!playing) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    // Calculate fps from time array
    const dt = totalFrames > 1
      ? (idealShot.t[idealShot.t.length - 1] - idealShot.t[0]) / (totalFrames - 1)
      : 0.05;
    const ms = Math.max(dt * 1000, 33);

    intervalRef.current = setInterval(() => {
      setFrameIndex(prev => {
        if (prev >= totalFrames - 1) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, ms);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, totalFrames, idealShot.t]);

  const currentT = idealShot.t[frameIndex] ?? 0;
  const delta = idealShot.wcs_ideal - idealShot.wcs_original;

  return (
    <div className="rounded-lg border border-gray-800 overflow-hidden bg-gray-900/30">
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800/50">
        <div className="text-xs uppercase tracking-widest text-gray-500 font-semibold">
          Original vs Ideal Motion
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs">
            <span className="w-2 h-2 rounded-full bg-red-600" />
            <span className="text-gray-400">Original</span>
            <span className="font-bold text-white ml-1">{idealShot.wcs_original}</span>
          </div>
          <span className="text-gray-600">→</span>
          <div className="flex items-center gap-1.5 text-xs">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            <span className="text-gray-400">Ideal</span>
            <span className="font-bold text-white ml-1">{idealShot.wcs_ideal}</span>
          </div>
          <span className={`text-xs font-bold ${delta > 0 ? "text-green-400" : "text-gray-500"}`}>
            {delta > 0 ? "+" : ""}{delta}
          </span>
        </div>
      </div>

      <div className="h-[340px]">
        <Canvas
          camera={{ position: [0, 1.8, 4], fov: 40, near: 0.1, far: 50 }}
          style={{ background: "#0a0a14" }}
        >
          <AnimatedScene idealShot={idealShot} frameIndex={frameIndex} />
        </Canvas>
      </div>

      {/* Playback controls */}
      <div className="flex items-center gap-3 px-4 py-2 border-t border-gray-800/50">
        <button
          onClick={() => { setFrameIndex(0); setPlaying(true); }}
          className="text-gray-400 hover:text-white text-xs px-1 transition-colors"
        >
          ⏮
        </button>
        <button
          onClick={() => setPlaying(p => !p)}
          className="bg-cyan-700/50 hover:bg-cyan-600/50 text-cyan-200 rounded px-2.5 py-0.5 text-xs font-medium transition-colors"
        >
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <input
          type="range"
          min={0}
          max={totalFrames - 1}
          value={frameIndex}
          onChange={e => { setFrameIndex(Number(e.target.value)); setPlaying(false); }}
          className="flex-1 accent-cyan-500 h-1"
        />
        <span className="text-gray-500 text-xs font-mono w-14 text-right">
          t={currentT.toFixed(2)}s
        </span>
      </div>
    </div>
  );
}
