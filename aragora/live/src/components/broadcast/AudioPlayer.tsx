'use client';

import { useRef, useState } from 'react';
import type { AudioPlayerProps } from './types';

/**
 * Terminal-styled audio player with play/pause, progress bar, and download
 */
export function AudioPlayer({ url }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    audioRef.current.currentTime = percent * duration;
  };

  const formatTime = (seconds: number): string => {
    if (!isFinite(seconds) || isNaN(seconds)) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-2">
      <audio
        ref={audioRef}
        src={url}
        onTimeUpdate={() => setProgress(audioRef.current?.currentTime || 0)}
        onLoadedMetadata={() => setDuration(audioRef.current?.duration || 0)}
        onEnded={() => setIsPlaying(false)}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      />

      <div className="flex items-center gap-3">
        <button
          onClick={togglePlay}
          className="px-2 py-1 text-xs font-theme-data border border-[var(--accent)]/40 hover:bg-[var(--accent)]/10 transition-colors"
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? '[PAUSE]' : '[PLAY]'}
        </button>

        <div
          className="flex-1 h-2 bg-border rounded overflow-hidden cursor-pointer"
          onClick={handleSeek}
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={duration}
        >
          <div
            className="h-full bg-[var(--accent)] transition-all duration-100"
            style={{ width: `${duration ? (progress / duration) * 100 : 0}%` }}
          />
        </div>

        <span className="text-xs font-theme-data text-text-muted whitespace-nowrap">
          {formatTime(progress)} / {formatTime(duration)}
        </span>
      </div>

      <a
        href={url}
        download
        className="inline-block text-xs font-theme-data text-[var(--acid-cyan)] hover:underline"
      >
        [DOWNLOAD MP3]
      </a>
    </div>
  );
}
