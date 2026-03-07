"use client";

import { useEffect, useState } from "react";
import { KinematicsShot, CASCADE_COLORS } from "./types";

interface Props {
  shot: KinematicsShot;
}

function scoreColor(score: number): string {
  if (score >= 80) return "#22c55e";
  if (score >= 60) return "#eab308";
  if (score >= 40) return "#f97316";
  return "#ef4444";
}

function scoreGradientId(score: number): string {
  if (score >= 80) return "gauge-green";
  if (score >= 60) return "gauge-yellow";
  if (score >= 40) return "gauge-orange";
  return "gauge-red";
}

export default function WhipChainGauge({ shot }: Props) {
  const [animatedScore, setAnimatedScore] = useState(0);
  const [arcProgress, setArcProgress] = useState(0);

  useEffect(() => {
    setAnimatedScore(0);
    setArcProgress(0);
    const target = shot.whipchain_score;
    const duration = 800;
    const start = performance.now();

    function animate(now: number) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      setAnimatedScore(Math.round(eased * target));
      setArcProgress(eased * target);
      if (progress < 1) requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
  }, [shot]);

  const radius = 80;
  const strokeWidth = 10;
  const circumference = Math.PI * radius; // half circle
  const arcLength = (arcProgress / 100) * circumference;
  const color = scoreColor(shot.whipchain_score);

  // Sequence checks: proper order is pelvis peaks first, then hip, knee, foot
  const peaks = [
    { label: "Pelvis → Hip", ok: shot.peak_t_pelvis <= shot.peak_t_hip, gap: ((shot.peak_t_hip - shot.peak_t_pelvis) * 1000).toFixed(0), color: CASCADE_COLORS.pelvis },
    { label: "Hip → Knee", ok: shot.peak_t_hip <= shot.peak_t_knee, gap: ((shot.peak_t_knee - shot.peak_t_hip) * 1000).toFixed(0), color: CASCADE_COLORS.hip },
    { label: "Knee → Foot", ok: shot.peak_t_knee <= shot.peak_t_foot, gap: ((shot.peak_t_foot - shot.peak_t_knee) * 1000).toFixed(0), color: CASCADE_COLORS.knee },
  ];

  return (
    <div className="flex flex-col items-center gap-4 py-6">
      {/* Gauge */}
      <div className="relative">
        <svg width="200" height="120" viewBox="-100 -100 200 120">
          <defs>
            <linearGradient id="gauge-green" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#f97316" />
              <stop offset="50%" stopColor="#eab308" />
              <stop offset="100%" stopColor="#22c55e" />
            </linearGradient>
            <linearGradient id="gauge-yellow" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="50%" stopColor="#f97316" />
              <stop offset="100%" stopColor="#eab308" />
            </linearGradient>
            <linearGradient id="gauge-orange" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="100%" stopColor="#f97316" />
            </linearGradient>
            <linearGradient id="gauge-red" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#991b1b" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {/* Background arc */}
          <path
            d={`M -${radius} 0 A ${radius} ${radius} 0 0 1 ${radius} 0`}
            fill="none"
            stroke="#1f2937"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          {/* Score arc */}
          <path
            d={`M -${radius} 0 A ${radius} ${radius} 0 0 1 ${radius} 0`}
            fill="none"
            stroke={`url(#${scoreGradientId(shot.whipchain_score)})`}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${arcLength} ${circumference}`}
            filter="url(#glow)"
            style={{ transition: "stroke-dasharray 0.1s" }}
          />
        </svg>
        {/* Score number */}
        <div className="absolute inset-0 flex items-end justify-center pb-2">
          <span className="text-5xl font-bold tabular-nums" style={{ color }}>
            {animatedScore}
          </span>
        </div>
      </div>

      {/* Label */}
      <div className="text-center">
        <div className="text-xs uppercase tracking-widest text-gray-500 font-semibold">WhipChain Score</div>
        <div className="text-sm text-gray-400 mt-1">
          #{shot.shooter_jersey} · {shot.match_time} · {shot.ball_speed} m/s · {shot.kicking_side} foot
        </div>
      </div>

      {/* Sequence indicators */}
      <div className="flex gap-4 mt-1">
        {peaks.map((p, i) => (
          <div key={i} className="flex flex-col items-center gap-1">
            <div className="flex items-center gap-1.5">
              <span className={`text-sm ${p.ok ? "text-green-400" : "text-red-400"}`}>
                {p.ok ? "✓" : "✗"}
              </span>
              <span className="text-xs text-gray-400">{p.label}</span>
            </div>
            <span className="text-xs text-gray-500">{p.gap}ms</span>
          </div>
        ))}
      </div>
    </div>
  );
}
