"use client";

interface Props {
  fps: number;
  totalFrames: number;
  currentIndex: number;
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onSeek: (index: number) => void;
  onReset: () => void;
  frameNumber: number;
}

export default function Playback({
  fps,
  totalFrames,
  currentIndex,
  isPlaying,
  onPlay,
  onPause,
  onSeek,
  onReset,
  frameNumber,
}: Props) {
  const duration = totalFrames / fps;
  const currentTime = currentIndex / fps;

  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1).padStart(4, "0");
    return `${m}:${sec}`;
  };

  return (
    <div className="flex items-center gap-3 bg-gray-900/60 backdrop-blur-xl border-t border-gray-700/50 px-4 py-2">
      {/* Buttons */}
      <button
        onClick={onReset}
        className="text-gray-400 hover:text-white transition-colors text-sm px-2"
        title="Reset"
      >
        ⏮
      </button>
      <button
        onClick={isPlaying ? onPause : onPlay}
        className="bg-red-700 hover:bg-red-600 text-white rounded px-3 py-1 text-sm font-medium transition-colors min-w-[60px]"
      >
        {isPlaying ? "⏸ Pause" : "▶ Play"}
      </button>

      {/* Slider */}
      <div className="flex-1 flex items-center gap-2">
        <span className="text-gray-400 text-xs font-mono w-12 text-right">
          {fmt(currentTime)}
        </span>
        <input
          type="range"
          min={0}
          max={totalFrames - 1}
          value={currentIndex}
          onChange={(e) => onSeek(Number(e.target.value))}
          className="flex-1 slider-red h-1"
        />
        <span className="text-gray-400 text-xs font-mono w-12">
          {fmt(duration)}
        </span>
      </div>

      {/* Frame info */}
      <div className="text-gray-500 text-xs font-mono whitespace-nowrap">
        frame {frameNumber} ({currentIndex + 1}/{totalFrames})
      </div>
    </div>
  );
}
