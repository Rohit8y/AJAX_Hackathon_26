"use client";

import { useRef, useEffect } from "react";
import { KinematicsShot } from "./types";

interface Props {
  shots: KinematicsShot[];
  activeId: number;
  onSelect: (id: number) => void;
}

function scoreColor(score: number): string {
  if (score >= 80) return "#22c55e";
  if (score >= 60) return "#eab308";
  if (score >= 40) return "#f97316";
  return "#ef4444";
}

export default function ShotStrip({ shots, activeId, onSelect }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    }
  }, [activeId]);

  return (
    <div className="border-t border-gray-800 bg-gray-900/50">
      <div className="px-4 pt-3 pb-1 text-xs uppercase tracking-widest text-gray-500 font-semibold">
        All Shots (sorted by score)
      </div>
      <div
        ref={scrollRef}
        className="flex gap-2 px-4 pb-3 pt-1 overflow-x-auto"
        style={{ scrollbarWidth: "thin", scrollbarColor: "#374151 transparent" }}
      >
        {shots.map((shot) => {
          const active = shot.id === activeId;
          return (
            <button
              key={shot.id}
              ref={active ? activeRef : undefined}
              onClick={() => onSelect(shot.id)}
              className={`flex-shrink-0 w-24 rounded-lg px-2.5 py-2 text-left transition-all ${
                active
                  ? "bg-red-900/30 border-red-500 ring-1 ring-red-500/30 scale-105"
                  : "bg-gray-800/50 border-gray-700 hover:bg-gray-800 hover:border-gray-600"
              } border`}
            >
              <div className="flex items-center gap-1.5 mb-1">
                {shot.is_goal && <span className="text-xs">⚽</span>}
                <span
                  className="text-lg font-bold tabular-nums leading-none"
                  style={{ color: scoreColor(shot.whipchain_score) }}
                >
                  {shot.whipchain_score}
                </span>
              </div>
              <div className="text-xs text-gray-300 font-medium">#{shot.shooter_jersey}</div>
              <div className="text-xs text-gray-500">{shot.match_time}</div>
              <div className={`text-xs mt-0.5 font-semibold ${shot.is_goal ? "text-green-400" : "text-gray-500"}`}>
                {shot.is_goal ? "GOAL" : "SHOT"}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
