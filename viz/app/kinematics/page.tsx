"use client";

import { useState, useEffect } from "react";
import NavBar from "@/components/NavBar";
import WhipChainGauge from "@/components/kinematics/WhipChainGauge";
import CascadeChart from "@/components/kinematics/CascadeChart";
import SkeletonContact from "@/components/kinematics/SkeletonContact";
import PitchMini from "@/components/kinematics/PitchMini";
import ShotStrip from "@/components/kinematics/ShotStrip";
import ComparePanel from "@/components/kinematics/ComparePanel";
import IdealSkeletonCompare from "@/components/kinematics/IdealSkeletonCompare";
import IdealScoreCard from "@/components/kinematics/IdealScoreCard";
import { KinematicsData, KinematicsShot, IdealKinematicsData, IdealShotData, CASCADE_COLORS } from "@/components/kinematics/types";

export default function KinematicsPage() {
  const [data, setData] = useState<KinematicsData | null>(null);
  const [idealData, setIdealData] = useState<IdealKinematicsData | null>(null);
  const [activeId, setActiveId] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/data/kinematics.json")
      .then((r) => r.json())
      .then((d: KinematicsData) => {
        setData(d);
        if (d.shots.length > 0) setActiveId(d.shots[0].id);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetch("/data/ideal_kinematics.json")
      .then((r) => r.json())
      .then((d: IdealKinematicsData) => setIdealData(d))
      .catch(() => {});
  }, []);

  const shot = data?.shots.find((s) => s.id === activeId) ?? null;

  // Match ideal shot to current shot by shooter_jersey + match_time
  const idealShot: IdealShotData | null = (() => {
    if (!shot || !idealData) return null;
    return idealData.shots.find(
      (s) => s.shooter_jersey === shot.shooter_jersey && s.match_time === shot.match_time
    ) ?? null;
  })();

  if (loading) {
    return (
      <div className="flex flex-col h-screen bg-gray-950 text-white">
        <NavBar />
        <div className="flex-1 flex flex-col items-center justify-center text-gray-500 gap-3">
          <div className="w-8 h-8 border-2 border-red-600 border-t-transparent rounded-full animate-spin-slow" />
          <span className="text-sm">Loading kinematics data...</span>
        </div>
      </div>
    );
  }

  if (!data || !shot) {
    return (
      <div className="flex flex-col h-screen bg-gray-950 text-white">
        <NavBar />
        <div className="flex-1 flex items-center justify-center text-gray-500">
          No kinematics data available
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white">
      <NavBar />

      <div className="flex-1 overflow-y-auto">
        {/* Hero: Gauge */}
        <section className="max-w-4xl mx-auto px-4 pt-6 animate-fade-in">
          <WhipChainGauge shot={shot} />
        </section>

        {/* Ideal Score Card */}
        {idealShot && (
          <section className="max-w-5xl mx-auto px-4 pt-4 animate-fade-in">
            <div className="text-xs uppercase tracking-widest text-gray-500 font-semibold mb-2 px-1">
              Ideal Motion Optimization
            </div>
            <IdealScoreCard idealShot={idealShot} />
          </section>
        )}

        {/* Cascade Chart */}
        <section className="max-w-5xl mx-auto px-4 pt-4 pb-2 animate-fade-in">
          <CascadeChart
            shot={shot}
            idealPeakTimes={idealShot?.ideal_peak_times ?? null}
          />
        </section>

        {/* Bottom panels: 3D Skeleton + Shot Context */}
        <section className="max-w-5xl mx-auto px-4 py-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 3D Skeleton */}
            <div className="animate-fade-in">
              <div className="text-xs uppercase tracking-widest text-gray-500 font-semibold mb-2 px-1">
                Skeleton at Contact
              </div>
              <div className="h-[340px]">
                <SkeletonContact shot={shot} />
              </div>
            </div>

            {/* Shot Context */}
            <div className="animate-fade-in">
              <div className="text-xs uppercase tracking-widest text-gray-500 font-semibold mb-2 px-1">
                Shot Context
              </div>
              <div className="flex flex-col gap-3">
                <PitchMini shot={shot} />

                {/* Stats card */}
                <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-4 space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <div className="text-xs text-gray-500">Ball Speed</div>
                      <div className="text-lg font-bold text-white">{shot.ball_speed} m/s</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Kicking Side</div>
                      <div className="text-lg font-bold text-white capitalize">{shot.kicking_side}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Event</div>
                      <div className="text-sm font-medium text-white">{shot.team}</div>
                      <div className={`text-xs font-semibold ${shot.is_goal ? "text-green-400" : "text-gray-500"}`}>
                        {shot.is_goal ? "GOAL" : "SHOT"}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Match Time</div>
                      <div className="text-lg font-bold text-white">{shot.match_time}</div>
                    </div>
                  </div>

                  {/* Peak angular velocities */}
                  <div className="border-t border-gray-800 pt-3">
                    <div className="text-xs text-gray-500 mb-2">Peak Angular Velocities</div>
                    <div className="space-y-1.5">
                      {([
                        { label: "Pelvis", value: shot.peak_omega_pelvis, t: shot.peak_t_pelvis, color: CASCADE_COLORS.pelvis },
                        { label: "Hip", value: shot.peak_omega_hip, t: shot.peak_t_hip, color: CASCADE_COLORS.hip },
                        { label: "Knee", value: shot.peak_omega_knee, t: shot.peak_t_knee, color: CASCADE_COLORS.knee },
                        { label: "Foot", value: shot.peak_omega_foot, t: shot.peak_t_foot, color: CASCADE_COLORS.foot },
                      ]).map(({ label, value, t, color }) => (
                        <div key={label} className="flex items-center gap-2 text-sm">
                          <span className="w-2 h-2 rounded-full" style={{ background: color }} />
                          <span className="text-gray-400 w-14">{label}</span>
                          <span className="text-white font-medium tabular-nums">{value.toFixed(1)} rad/s</span>
                          <span className="text-gray-600 text-xs tabular-nums ml-auto">t={t.toFixed(2)}s</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Ideal Skeleton Compare */}
        {idealShot && (
          <section className="max-w-5xl mx-auto px-4 pb-4 animate-fade-in">
            <IdealSkeletonCompare idealShot={idealShot} />
          </section>
        )}

        {/* Compare Panel */}
        <section className="max-w-5xl mx-auto px-4 pb-2">
          <ComparePanel shots={data.shots} currentShot={shot} />
        </section>
      </div>

      {/* Shot selector strip (fixed at bottom) */}
      <ShotStrip shots={data.shots} activeId={activeId} onSelect={setActiveId} />
    </div>
  );
}
