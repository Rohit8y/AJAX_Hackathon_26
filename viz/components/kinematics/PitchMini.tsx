"use client";

import { KinematicsShot } from "./types";

interface Props {
  shot: KinematicsShot;
}

export default function PitchMini({ shot }: Props) {
  // Field dimensions: x ∈ [-56,56], y ∈ [-39,38]
  // SVG viewbox: map field coords to SVG space
  const fieldW = 112;
  const fieldH = 77;
  const svgW = 320;
  const svgH = 220;
  const margin = 10;

  // Scale field coords to SVG
  const sx = (x: number) => margin + ((x + 56) / fieldW) * (svgW - 2 * margin);
  const sy = (y: number) => margin + ((39 - y) / fieldH) * (svgH - 2 * margin);

  // Shooter position from skeleton pelvis (part 12)
  const pelvis = shot.skeleton["12"];
  const shooterX = pelvis ? sx(pelvis[0]) : svgW / 2;
  const shooterY = pelvis ? sy(pelvis[1]) : svgH / 2;

  // Ball speed arrow: use kicking side to estimate direction toward nearest goal
  // Determine which goal the shot goes toward based on x position
  const shooterFieldX = pelvis ? pelvis[0] : 0;
  const goalX = shooterFieldX > 0 ? 56 : -56;
  const goalSvgX = sx(goalX);

  // Arrow direction (normalized, scaled)
  const dx = goalSvgX - shooterX;
  const dy = 0; // roughly toward goal line
  const len = Math.sqrt(dx * dx + dy * dy);
  const arrowLen = Math.min(len * 0.4, 40);
  const nx = dx / len;
  const ny = dy / len;

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full max-w-[320px]">
        {/* Pitch background */}
        <rect x={margin} y={margin} width={svgW - 2 * margin} height={svgH - 2 * margin} fill="#1a5c1a" rx={2} />

        {/* Field lines */}
        <rect x={margin} y={margin} width={svgW - 2 * margin} height={svgH - 2 * margin} fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth={1} />

        {/* Halfway line */}
        <line x1={sx(0)} y1={margin} x2={sx(0)} y2={svgH - margin} stroke="rgba(255,255,255,0.2)" strokeWidth={0.8} />

        {/* Center circle */}
        <circle cx={sx(0)} cy={sy(0)} r={((9.15 / fieldW) * (svgW - 2 * margin))} fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth={0.8} />

        {/* Goal boxes - left */}
        <rect x={margin} y={sy(20.15)} width={((16.5 / fieldW) * (svgW - 2 * margin))} height={((40.3 / fieldH) * (svgH - 2 * margin))} fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth={0.8} />

        {/* Goal boxes - right */}
        <rect x={sx(56 - 16.5)} y={sy(20.15)} width={((16.5 / fieldW) * (svgW - 2 * margin))} height={((40.3 / fieldH) * (svgH - 2 * margin))} fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth={0.8} />

        {/* Goals */}
        <line x1={margin} y1={sy(3.66)} x2={margin} y2={sy(-3.66)} stroke="#fbbf24" strokeWidth={2} />
        <line x1={svgW - margin} y1={sy(3.66)} x2={svgW - margin} y2={sy(-3.66)} stroke="#fbbf24" strokeWidth={2} />

        {/* Shooter position */}
        <circle cx={shooterX} cy={shooterY} r={5} fill="#ef4444" stroke="#fca5a5" strokeWidth={1.5}>
          <animate attributeName="r" values="5;7;5" dur="2s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="1;0.7;1" dur="2s" repeatCount="indefinite" />
        </circle>

        {/* Ball trajectory arrow */}
        <line
          x1={shooterX}
          y1={shooterY}
          x2={shooterX + nx * arrowLen}
          y2={shooterY + ny * arrowLen}
          stroke="#fbbf24"
          strokeWidth={2}
          markerEnd="url(#arrowhead)"
        />
        <defs>
          <marker id="arrowhead" markerWidth={8} markerHeight={6} refX={8} refY={3} orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#fbbf24" />
          </marker>
        </defs>

        {/* Shooter label */}
        <text x={shooterX} y={shooterY - 10} fill="white" fontSize={10} textAnchor="middle" fontWeight="bold">
          #{shot.shooter_jersey}
        </text>
      </svg>
    </div>
  );
}
