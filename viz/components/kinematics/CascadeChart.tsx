"use client";

import { useMemo, useState, useRef, useCallback } from "react";
import { KinematicsShot, CASCADE_COLORS, CascadeSegment } from "./types";

interface Props {
  shot: KinematicsShot;
  compareShot?: KinematicsShot | null;
  idealPeakTimes?: Record<CascadeSegment, number> | null;
  height?: number;
}

const SEGMENTS: { key: CascadeSegment; dataKey: string; peakTKey: string; peakOmegaKey: string }[] = [
  { key: "pelvis", dataKey: "omega_pelvis", peakTKey: "peak_t_pelvis", peakOmegaKey: "peak_omega_pelvis" },
  { key: "hip", dataKey: "omega_hip", peakTKey: "peak_t_hip", peakOmegaKey: "peak_omega_hip" },
  { key: "knee", dataKey: "omega_knee", peakTKey: "peak_t_knee", peakOmegaKey: "peak_omega_knee" },
  { key: "foot", dataKey: "omega_foot", peakTKey: "peak_t_foot", peakOmegaKey: "peak_omega_foot" },
];

const PADDING = { top: 20, right: 20, bottom: 40, left: 50 };

function cardinalSpline(points: [number, number][], tension = 0.4): string {
  if (points.length < 2) return "";
  const n = points.length;
  let d = `M ${points[0][0]} ${points[0][1]}`;
  for (let i = 0; i < n - 1; i++) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(n - 1, i + 2)];
    const cp1x = p1[0] + (p2[0] - p0[0]) * tension / 3;
    const cp1y = p1[1] + (p2[1] - p0[1]) * tension / 3;
    const cp2x = p2[0] - (p3[0] - p1[0]) * tension / 3;
    const cp2y = p2[1] - (p3[1] - p1[1]) * tension / 3;
    d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2[0]} ${p2[1]}`;
  }
  return d;
}

export default function CascadeChart({ shot, compareShot, idealPeakTimes, height = 300 }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<{ x: number; t: number; values: Record<CascadeSegment, number> } | null>(null);
  const [mounted, setMounted] = useState(false);

  // Trigger line animation on mount
  useState(() => {
    setTimeout(() => setMounted(true), 50);
  });

  const { tMin, tMax, omegaMax, xScale, yScale, width } = useMemo(() => {
    const tArr = shot.t;
    const tMin = tArr[0];
    const tMax = tArr[tArr.length - 1];
    const allOmega = [...shot.omega_pelvis, ...shot.omega_hip, ...shot.omega_knee, ...shot.omega_foot];
    if (compareShot) {
      allOmega.push(...compareShot.omega_pelvis, ...compareShot.omega_hip, ...compareShot.omega_knee, ...compareShot.omega_foot);
    }
    const omegaMax = Math.ceil(Math.max(...allOmega) / 5) * 5;
    const width = 800;
    const plotW = width - PADDING.left - PADDING.right;
    const plotH = height - PADDING.top - PADDING.bottom;
    return {
      tMin, tMax, omegaMax, width,
      xScale: (t: number) => PADDING.left + ((t - tMin) / (tMax - tMin)) * plotW,
      yScale: (omega: number) => PADDING.top + plotH - (omega / omegaMax) * plotH,
    };
  }, [shot, compareShot, height]);

  const buildPaths = useCallback((s: KinematicsShot) => {
    return SEGMENTS.map(seg => {
      const data = s[seg.dataKey as keyof KinematicsShot] as number[];
      const points: [number, number][] = s.t.map((t, i) => [xScale(t), yScale(data[i])]);
      const path = cardinalSpline(points);
      const peakT = s[seg.peakTKey as keyof KinematicsShot] as number;
      const peakOmega = s[seg.peakOmegaKey as keyof KinematicsShot] as number;
      return { key: seg.key, path, peakX: xScale(peakT), peakY: yScale(peakOmega), peakT, peakOmega };
    });
  }, [xScale, yScale]);

  const paths = useMemo(() => buildPaths(shot), [buildPaths, shot]);
  const comparePaths = useMemo(() => compareShot ? buildPaths(compareShot) : null, [buildPaths, compareShot]);

  const plotH = height - PADDING.top - PADDING.bottom;

  // Grid lines
  const gridLines = useMemo(() => {
    const lines: { y: number; label: string }[] = [];
    const step = omegaMax <= 20 ? 5 : omegaMax <= 50 ? 10 : 20;
    for (let v = 0; v <= omegaMax; v += step) {
      lines.push({ y: yScale(v), label: String(v) });
    }
    return lines;
  }, [omegaMax, yScale]);

  // Time axis labels
  const timeLabels = useMemo(() => {
    const labels: { x: number; label: string }[] = [];
    for (let t = Math.ceil(tMin * 2) / 2; t <= tMax; t += 0.5) {
      labels.push({ x: xScale(t), label: t.toFixed(1) });
    }
    return labels;
  }, [tMin, tMax, xScale]);

  // Contact line at t=0
  const contactX = xScale(0);

  // Hover handler
  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const svgX = ((e.clientX - rect.left) / rect.width) * 800;
    const t = tMin + ((svgX - PADDING.left) / (800 - PADDING.left - PADDING.right)) * (tMax - tMin);
    if (t < tMin || t > tMax) { setHover(null); return; }

    // Find closest index
    let closestIdx = 0;
    let closestDist = Infinity;
    for (let i = 0; i < shot.t.length; i++) {
      const d = Math.abs(shot.t[i] - t);
      if (d < closestDist) { closestDist = d; closestIdx = i; }
    }

    setHover({
      x: xScale(shot.t[closestIdx]),
      t: shot.t[closestIdx],
      values: {
        pelvis: shot.omega_pelvis[closestIdx],
        hip: shot.omega_hip[closestIdx],
        knee: shot.omega_knee[closestIdx],
        foot: shot.omega_foot[closestIdx],
      },
    });
  }, [shot, tMin, tMax, xScale]);

  // Analysis window shading (t=-1.0 to t=0)
  const windowLeft = xScale(Math.max(tMin, -1.0));
  const windowRight = contactX;

  return (
    <div className="w-full">
      <div className="text-xs uppercase tracking-widest text-gray-500 font-semibold mb-2 px-1">
        Angular Velocity Cascade
        {compareShot && <span className="text-gray-600 ml-2">(dashed = comparison)</span>}
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 800 ${height}`}
        className="w-full"
        style={{ maxHeight: height }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHover(null)}
      >
        {/* Analysis window shading */}
        <rect
          x={windowLeft}
          y={PADDING.top}
          width={Math.max(0, windowRight - windowLeft)}
          height={plotH}
          fill="url(#analysis-gradient)"
          opacity={0.3}
        />
        <defs>
          <linearGradient id="analysis-gradient" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#6366f1" stopOpacity="0" />
            <stop offset="100%" stopColor="#6366f1" stopOpacity="0.3" />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {gridLines.map((g, i) => (
          <g key={i}>
            <line x1={PADDING.left} y1={g.y} x2={800 - PADDING.right} y2={g.y} stroke="#1f2937" strokeWidth={1} />
            <text x={PADDING.left - 8} y={g.y + 4} fill="#6b7280" fontSize={11} textAnchor="end">{g.label}</text>
          </g>
        ))}

        {/* Time axis */}
        {timeLabels.map((l, i) => (
          <text key={i} x={l.x} y={height - 8} fill="#6b7280" fontSize={11} textAnchor="middle">{l.label}s</text>
        ))}

        {/* Y-axis label */}
        <text x={14} y={PADDING.top + plotH / 2} fill="#6b7280" fontSize={11} textAnchor="middle" transform={`rotate(-90, 14, ${PADDING.top + plotH / 2})`}>
          ω (rad/s)
        </text>

        {/* Contact line */}
        <line x1={contactX} y1={PADDING.top} x2={contactX} y2={PADDING.top + plotH} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="6 3" opacity={0.7} />
        <text x={contactX} y={height - 22} fill="#ef4444" fontSize={10} textAnchor="middle" fontWeight="bold">CONTACT</text>

        {/* Compare paths (dashed) */}
        {comparePaths?.map(p => (
          <path key={`cmp-${p.key}`} d={p.path} fill="none" stroke={CASCADE_COLORS[p.key]} strokeWidth={1.5} strokeDasharray="6 4" opacity={0.5} />
        ))}

        {/* Main curves */}
        {paths.map(p => {
          const pathLen = 2000;
          return (
            <path
              key={p.key}
              d={p.path}
              fill="none"
              stroke={CASCADE_COLORS[p.key]}
              strokeWidth={2.5}
              strokeLinecap="round"
              strokeDasharray={mounted ? "none" : `${pathLen}`}
              strokeDashoffset={mounted ? 0 : pathLen}
              style={{ transition: "stroke-dashoffset 1.2s ease-out" }}
            />
          );
        })}

        {/* Peak markers */}
        {paths.map(p => (
          <g key={`peak-${p.key}`}>
            <line x1={p.peakX} y1={p.peakY} x2={p.peakX} y2={PADDING.top + plotH} stroke={CASCADE_COLORS[p.key]} strokeWidth={1} strokeDasharray="3 3" opacity={0.4} />
            <polygon
              points={`${p.peakX},${p.peakY - 6} ${p.peakX + 5},${p.peakY} ${p.peakX},${p.peakY + 6} ${p.peakX - 5},${p.peakY}`}
              fill={CASCADE_COLORS[p.key]}
              stroke="#111827"
              strokeWidth={1}
            />
          </g>
        ))}

        {/* Ideal peak time lines */}
        {idealPeakTimes && SEGMENTS.map(seg => {
          const t = idealPeakTimes[seg.key];
          if (t === undefined || t < tMin || t > tMax) return null;
          const ix = xScale(t);
          return (
            <g key={`ideal-${seg.key}`}>
              <line
                x1={ix} y1={PADDING.top} x2={ix} y2={PADDING.top + plotH}
                stroke={CASCADE_COLORS[seg.key]}
                strokeWidth={1.5}
                strokeDasharray="4 4"
                opacity={0.5}
              />
              <text
                x={ix} y={PADDING.top - 4}
                fill={CASCADE_COLORS[seg.key]}
                fontSize={9}
                textAnchor="middle"
                opacity={0.7}
              >
                ideal
              </text>
            </g>
          );
        })}

        {/* Legend */}
        {SEGMENTS.map((seg, i) => {
          const lx = 800 - PADDING.right - 100;
          const ly = PADDING.top + 14 + i * 18;
          return (
            <g key={`legend-${seg.key}`}>
              <line x1={lx} y1={ly} x2={lx + 16} y2={ly} stroke={CASCADE_COLORS[seg.key]} strokeWidth={2.5} />
              <text x={lx + 22} y={ly + 4} fill="#d1d5db" fontSize={11} className="capitalize">{seg.key}</text>
            </g>
          );
        })}

        {/* Hover crosshair */}
        {hover && (
          <g>
            <line x1={hover.x} y1={PADDING.top} x2={hover.x} y2={PADDING.top + plotH} stroke="#9ca3af" strokeWidth={1} strokeDasharray="4 2" />
            {SEGMENTS.map(seg => {
              const val = hover.values[seg.key];
              return (
                <circle key={seg.key} cx={hover.x} cy={yScale(val)} r={4} fill={CASCADE_COLORS[seg.key]} stroke="#111827" strokeWidth={1.5} />
              );
            })}
            {/* Tooltip */}
            <foreignObject x={Math.min(hover.x + 10, 650)} y={PADDING.top} width={140} height={100}>
              <div className="bg-gray-900/95 border border-gray-700 rounded px-2.5 py-2 text-xs">
                <div className="text-gray-400 mb-1">t = {hover.t.toFixed(2)}s</div>
                {SEGMENTS.map(seg => (
                  <div key={seg.key} className="flex justify-between gap-3">
                    <span style={{ color: CASCADE_COLORS[seg.key] }} className="capitalize">{seg.key}</span>
                    <span className="text-gray-200 tabular-nums">{hover.values[seg.key].toFixed(1)}</span>
                  </div>
                ))}
              </div>
            </foreignObject>
          </g>
        )}
      </svg>
    </div>
  );
}
