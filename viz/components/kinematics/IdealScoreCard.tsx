"use client";

import { IdealShotData } from "./types";

interface Props {
  idealShot: IdealShotData;
}

export default function IdealScoreCard({ idealShot }: Props) {
  const delta = idealShot.wcs_ideal - idealShot.wcs_original;
  const flags = idealShot.modification_flags;

  const badges: string[] = [];
  if (flags.time_warped) badges.push("Time Warped");
  if (flags.amplitude_scaled) {
    const factor = flags.scale_factor_foot;
    badges.push(`Amplitude Scaled (foot ×${factor.toFixed(2)})`);
  }

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4 flex items-center gap-6 flex-wrap">
      {/* Original score */}
      <div className="text-center">
        <div className="text-xs text-gray-500 mb-1">Original WCS</div>
        <div className="text-2xl font-bold text-white tabular-nums">{idealShot.wcs_original}</div>
      </div>

      {/* Arrow */}
      <div className="flex flex-col items-center">
        <svg width="40" height="20" viewBox="0 0 40 20" className="text-green-400">
          <line x1="2" y1="10" x2="32" y2="10" stroke="currentColor" strokeWidth="2" />
          <polygon points="30,4 38,10 30,16" fill="currentColor" />
        </svg>
        <span className={`text-sm font-bold mt-0.5 ${delta > 0 ? "text-green-400" : delta < 0 ? "text-red-400" : "text-gray-500"}`}>
          {delta > 0 ? "+" : ""}{delta}
        </span>
      </div>

      {/* Ideal score */}
      <div className="text-center">
        <div className="text-xs text-cyan-400 mb-1">Ideal WCS</div>
        <div className="text-2xl font-bold text-cyan-300 tabular-nums">{idealShot.wcs_ideal}</div>
      </div>

      {/* Modification badges */}
      {badges.length > 0 && (
        <div className="flex gap-2 ml-auto flex-wrap">
          {badges.map(b => (
            <span key={b} className="text-xs px-2.5 py-1 rounded-full bg-gray-800 text-gray-300 border border-gray-700">
              {b}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
