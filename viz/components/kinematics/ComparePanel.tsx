"use client";

import { useState } from "react";
import { KinematicsShot } from "./types";
import CascadeChart from "./CascadeChart";
import WhipChainGauge from "./WhipChainGauge";

interface Props {
  shots: KinematicsShot[];
  currentShot: KinematicsShot;
}

export default function ComparePanel({ shots, currentShot }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [compareId, setCompareId] = useState<number | null>(null);

  const compareShot = compareId !== null ? shots.find(s => s.id === compareId) ?? null : null;

  return (
    <div className="border-t border-gray-800">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between text-sm text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 transition-colors"
      >
        <span className="uppercase tracking-widest text-xs font-semibold">Compare Mode</span>
        <span className="text-lg">{expanded ? "−" : "+"}</span>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500">Compare with:</span>
            <select
              value={compareId ?? ""}
              onChange={(e) => setCompareId(e.target.value ? Number(e.target.value) : null)}
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300 focus:outline-none focus:border-red-500"
            >
              <option value="">Select a shot…</option>
              {shots.filter(s => s.id !== currentShot.id).map(s => (
                <option key={s.id} value={s.id}>
                  #{s.shooter_jersey} · {s.match_time} · Score {s.whipchain_score} {s.is_goal ? "⚽" : ""}
                </option>
              ))}
            </select>
          </div>

          {compareShot && (
            <>
              {/* Side-by-side gauges */}
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-xs text-gray-500 mb-1">Shot A</div>
                  <WhipChainGauge shot={currentShot} />
                </div>
                <div className="text-center">
                  <div className="text-xs text-gray-500 mb-1">Shot B</div>
                  <WhipChainGauge shot={compareShot} />
                </div>
              </div>

              {/* Overlaid cascade */}
              <CascadeChart shot={currentShot} compareShot={compareShot} />

              {/* Delta indicators */}
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: "Score", a: currentShot.whipchain_score, b: compareShot.whipchain_score, unit: "" },
                  { label: "Ball Speed", a: currentShot.ball_speed, b: compareShot.ball_speed, unit: " m/s" },
                  { label: "Peak Foot ω", a: currentShot.peak_omega_foot, b: compareShot.peak_omega_foot, unit: " rad/s" },
                ].map(({ label, a, b, unit }) => {
                  const delta = a - b;
                  return (
                    <div key={label} className="bg-gray-800/50 rounded-lg p-3 text-center">
                      <div className="text-xs text-gray-500 mb-1">{label}</div>
                      <div className={`text-sm font-bold ${delta > 0 ? "text-green-400" : delta < 0 ? "text-red-400" : "text-gray-400"}`}>
                        {delta > 0 ? "+" : ""}{delta.toFixed(1)}{unit}
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
