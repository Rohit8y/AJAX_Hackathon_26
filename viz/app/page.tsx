"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import ShotScene from "@/components/ShotScene";
import Playback from "@/components/Playback";
import { ShotData } from "@/components/types";

interface ShotMeta {
  idx: number;
  frames: number;
  duration: number;
  first_frame: number;
  last_frame: number;
}

const TEAM_LABELS: Record<number, string> = {
  1: "Home",
  2: "Team A",
  3: "Team B",
  4: "Referee",
};

const TEAM_COLORS: Record<number, string> = {
  1: "#D2001A",
  2: "#4488FF",
  3: "#FFFFFF",
  4: "#FFD700",
};

export default function Home() {
  const [shotList, setShotList] = useState<ShotMeta[]>([]);
  const [activeShotIdx, setActiveShotIdx] = useState(0);
  const [shotData, setShotData] = useState<ShotData | null>(null);
  const [loading, setLoading] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showGaze, setShowGaze] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load shot index on mount
  useEffect(() => {
    fetch("/data/shots_index.json")
      .then((r) => r.json())
      .then((data: ShotMeta[]) => setShotList(data))
      .catch(() => {});
  }, []);

  // Load shot JSON on selection
  useEffect(() => {
    setLoading(true);
    setIsPlaying(false);
    setFrameIndex(0);
    if (intervalRef.current) clearInterval(intervalRef.current);

    fetch(`/data/shot_${activeShotIdx}.json`)
      .then((r) => r.json())
      .then((data: ShotData) => {
        setShotData(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [activeShotIdx]);

  // Playback interval
  useEffect(() => {
    if (!shotData || !isPlaying) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }
    const ms = 1000 / (shotData.fps ?? 25);
    intervalRef.current = setInterval(() => {
      setFrameIndex((prev) => {
        if (prev >= shotData.frames.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, ms);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, shotData]);

  const handlePlay = useCallback(() => setIsPlaying(true), []);
  const handlePause = useCallback(() => setIsPlaying(false), []);
  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setFrameIndex(0);
  }, []);
  const handleSeek = useCallback((idx: number) => {
    setFrameIndex(idx);
  }, []);

  const currentFrame = shotData?.frames[frameIndex] ?? null;

  // Compute closest player to ball for stats overlay
  const closestInfo = (() => {
    if (!currentFrame?.ball || !shotData) return null;
    const [bx, by] = currentFrame.ball;
    let minDist = Infinity;
    let closest: { jersey: number; team: number; dist: number } | null = null;
    for (const player of currentFrame.players) {
      const pelvisIdx = shotData.joint_names.indexOf("PELVIS");
      const pos = player.pos[pelvisIdx];
      if (!pos) continue;
      const dx = pos[0] - bx;
      const dy = pos[1] - by;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minDist) {
        minDist = dist;
        closest = { jersey: player.jersey, team: player.team, dist };
      }
    }
    return closest;
  })();

  const shotDuration = shotData
    ? (shotData.frames.length / shotData.fps).toFixed(1)
    : "—";

  const playerCount = currentFrame?.players.length ?? 0;

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="flex items-center gap-4 px-4 py-2 bg-gray-900 border-b border-gray-800 shrink-0">
        <div className="text-red-600 font-bold text-lg tracking-wide">⚽ AJAX 3D</div>
        <div className="text-gray-400 text-sm">Shot Skeleton Visualizer</div>
        {loading && (
          <div className="ml-auto text-yellow-400 text-sm animate-pulse">Loading…</div>
        )}
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-52 bg-gray-900 border-r border-gray-800 flex flex-col overflow-y-auto shrink-0">
          <div className="px-3 pt-3 pb-1 text-xs text-gray-500 uppercase tracking-widest font-semibold">
            Shot Clips
          </div>
          {shotList.map((meta) => (
            <button
              key={meta.idx}
              onClick={() => setActiveShotIdx(meta.idx)}
              className={`text-left px-3 py-2.5 text-sm border-b border-gray-800 transition-colors ${
                activeShotIdx === meta.idx
                  ? "bg-red-900/40 text-red-400 border-l-2 border-l-red-600"
                  : "text-gray-300 hover:bg-gray-800"
              }`}
            >
              <div className="font-medium">Shot {meta.idx + 1}</div>
              <div className="text-xs text-gray-500 mt-0.5">
                {meta.frames} frames · {meta.duration}s
              </div>
            </button>
          ))}

          {/* Metadata panel */}
          {shotData && currentFrame && (
            <div className="mt-auto px-3 py-3 border-t border-gray-800 space-y-2">
              <div className="pb-1 border-b border-gray-800">
                <div className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-1.5">Overlays</div>
                <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer select-none">
                  <input type="checkbox" checked={showGaze} onChange={(e) => setShowGaze(e.target.checked)} className="accent-red-600" />
                  Gaze to ball
                </label>
              </div>
              <div className="text-xs text-gray-500 uppercase tracking-widest font-semibold">
                Stats
              </div>
              <div className="text-xs text-gray-400">
                <span className="text-gray-200">{playerCount}</span> players
              </div>
              <div className="text-xs text-gray-400">
                Duration:{" "}
                <span className="text-gray-200">{shotDuration}s</span>
              </div>
              {closestInfo && (
                <div className="text-xs text-gray-400">
                  Nearest to ball:{" "}
                  <span
                    style={{ color: TEAM_COLORS[closestInfo.team] }}
                    className="font-semibold"
                  >
                    #{closestInfo.jersey}
                  </span>{" "}
                  <span className="text-gray-500">
                    ({TEAM_LABELS[closestInfo.team]})
                  </span>
                  <div className="text-gray-500">
                    {closestInfo.dist.toFixed(1)}m away
                  </div>
                </div>
              )}

              {/* Team legend */}
              <div className="pt-1 space-y-1">
                {Object.entries(TEAM_LABELS).map(([t, label]) => (
                  <div key={t} className="flex items-center gap-1.5 text-xs">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full border border-gray-600"
                      style={{ background: TEAM_COLORS[Number(t)] }}
                    />
                    <span className="text-gray-400">{label}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>

        {/* 3D Canvas */}
        <main className="flex-1 relative">
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500">
              Loading shot data…
            </div>
          ) : !shotData ? (
            <div className="absolute inset-0 flex items-center justify-center text-gray-500">
              No data available
            </div>
          ) : (
            <Canvas
              camera={{
                position: [0, 40, 55],
                fov: 55,
                near: 0.1,
                far: 500,
              }}
              gl={{ antialias: true }}
              style={{ background: "#0a0a14" }}
            >
              <ShotScene
                frame={currentFrame}
                jointNames={shotData.joint_names}
                showGaze={showGaze}
              />
            </Canvas>
          )}
        </main>
      </div>

      {/* Playback bar */}
      {shotData && (
        <Playback
          fps={shotData.fps}
          totalFrames={shotData.frames.length}
          currentIndex={frameIndex}
          isPlaying={isPlaying}
          onPlay={handlePlay}
          onPause={handlePause}
          onSeek={handleSeek}
          onReset={handleReset}
          frameNumber={currentFrame?.f ?? 0}
        />
      )}
    </div>
  );
}
