"use client";
import { useMemo } from "react";
import { Line, Html } from "@react-three/drei";
import * as THREE from "three";

const GAZE_RAY_LENGTH = 1.5; // meters, head-forward ray
const ARC_RADIUS = 0.65;
const ARC_SEGMENTS = 24;

interface Props {
  jointMap: Record<string, THREE.Vector3>;
  ballPos: THREE.Vector3;
}

function angleColor(deg: number) {
  if (deg < 30) return "#22c55e";
  if (deg < 60) return "#eab308";
  return "#ef4444";
}

export default function GazeOverlay({ jointMap, ballPos }: Props) {
  const leftEar = jointMap["LEFT_EAR"];
  const rightEar = jointMap["RIGHT_EAR"];
  const nose = jointMap["NOSE"];

  const overlay = useMemo(() => {
    if (!leftEar || !rightEar || !nose) return null;

    const earMid = leftEar.clone().add(rightEar).multiplyScalar(0.5);
    const headFwd = nose.clone().sub(earMid);
    if (headFwd.length() < 0.001) return null;
    headFwd.normalize();

    const toBallVec = ballPos.clone().sub(earMid);
    if (toBallVec.length() < 0.001) return null;
    const toBallNorm = toBallVec.clone().normalize();

    const angleDeg = THREE.MathUtils.radToDeg(headFwd.angleTo(toBallNorm));
    const color = angleColor(angleDeg);

    // Head-forward ray endpoint
    const rayEnd = earMid.clone().addScaledVector(headFwd, GAZE_RAY_LENGTH);

    // Arc from headFwd to ball direction
    const arcPoints: THREE.Vector3[] = [];
    if (angleDeg > 1 && angleDeg < 175) {
      const q = new THREE.Quaternion().setFromUnitVectors(headFwd, toBallNorm);
      const identity = new THREE.Quaternion();
      for (let i = 0; i <= ARC_SEGMENTS; i++) {
        const t = i / ARC_SEGMENTS;
        const qt = identity.clone().slerp(q, t);
        const dir = headFwd.clone().applyQuaternion(qt).normalize();
        arcPoints.push(earMid.clone().addScaledVector(dir, ARC_RADIUS));
      }
    }

    // Place label at the bisector direction, just outside the arc
    const bisector = headFwd.clone().add(toBallNorm).normalize();
    const labelPos = earMid.clone().addScaledVector(bisector, ARC_RADIUS + 0.2);

    return { earMid, rayEnd, arcPoints, color, angleDeg, labelPos, ballPos };
  }, [leftEar, rightEar, nose, ballPos]);

  if (!overlay) return null;
  const { earMid, rayEnd, arcPoints, color, angleDeg, labelPos } = overlay;

  return (
    <group>
      {/* Thick gaze ray: head forward direction */}
      <Line points={[earMid, rayEnd]} color={color} lineWidth={3} />

      {/* Thin line from head to ball */}
      <Line points={[earMid, ballPos]} color={color} lineWidth={1} />

      {/* Arc showing the angle gap */}
      {arcPoints.length > 1 && (
        <Line points={arcPoints} color={color} lineWidth={2} />
      )}

      {/* Degree label at the bisector */}
      <Html position={labelPos} center distanceFactor={15}>
        <div style={{
          color,
          fontSize: "10px",
          fontFamily: "monospace",
          fontWeight: "bold",
          background: "rgba(0,0,0,0.65)",
          padding: "1px 4px",
          borderRadius: "3px",
          pointerEvents: "none",
          whiteSpace: "nowrap",
        }}>
          {angleDeg.toFixed(0)}°
        </div>
      </Html>
    </group>
  );
}
