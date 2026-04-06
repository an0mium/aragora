'use client';

import { useState } from 'react';
import { logger } from '@/utils/logger';

interface TranscriptionSegment {
  text: string;
  start: number;
  end: number;
  confidence?: number;
}

interface TranscriptionResult {
  text: string;
  language: string;
  duration: number;
  segments: TranscriptionSegment[];
  provider?: string;
  model?: string;
}

interface TranscriptionViewerProps {
  result: TranscriptionResult;
  onCreateDebate?: (text: string) => void;
  className?: string;
}

export function TranscriptionViewer({
  result,
  onCreateDebate,
  className = '',
}: TranscriptionViewerProps) {
  const [showTimestamps, setShowTimestamps] = useState(true);
  const [selectedSegment, setSelectedSegment] = useState<number | null>(null);
  const [copySuccess, setCopySuccess] = useState(false);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(result.text);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      logger.error('Failed to copy:', err);
    }
  };

  const wordCount = result.text.split(/\s+/).filter(Boolean).length;

  return (
    <div className={`border border-[var(--accent)]/30 bg-surface/30 ${className}`}>
      {/* Header */}
      <div className="p-3 border-b border-[var(--accent)]/20 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-[var(--accent)] font-theme-data text-xs">[TRANSCRIPTION]</span>
          <span className="text-text-muted font-theme-data text-[10px]">
            {wordCount} words | {formatTime(result.duration)}
          </span>
          {result.language && (
            <span className="text-[var(--acid-cyan)] font-theme-data text-[10px] uppercase">
              {result.language}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTimestamps(!showTimestamps)}
            className={`px-2 py-1 font-theme-data text-[10px] border transition-colors ${
              showTimestamps
                ? 'border-[var(--accent)]/50 text-[var(--accent)] bg-[var(--accent)]/10'
                : 'border-[var(--accent)]/20 text-text-muted hover:text-[var(--accent)]'
            }`}
          >
            [TIMESTAMPS]
          </button>
          <button
            onClick={handleCopy}
            className="px-2 py-1 font-theme-data text-[10px] border border-[var(--accent)]/20 text-text-muted hover:text-[var(--accent)] transition-colors"
          >
            {copySuccess ? '[COPIED]' : '[COPY]'}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 max-h-96 overflow-y-auto">
        {showTimestamps && result.segments.length > 0 ? (
          <div className="space-y-2">
            {result.segments.map((segment, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedSegment(selectedSegment === idx ? null : idx)}
                className={`flex gap-3 p-2 cursor-pointer transition-colors rounded ${
                  selectedSegment === idx
                    ? 'bg-[var(--accent)]/10 border border-[var(--accent)]/30'
                    : 'hover:bg-surface/50'
                }`}
              >
                <span className="text-[var(--acid-cyan)] font-theme-data text-[10px] shrink-0 w-16">
                  [{formatTime(segment.start)}]
                </span>
                <span className="text-text font-theme-data text-sm leading-relaxed">
                  {segment.text}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-text font-theme-data text-sm leading-relaxed whitespace-pre-wrap">
            {result.text}
          </p>
        )}
      </div>

      {/* Footer with actions */}
      <div className="p-3 border-t border-[var(--accent)]/20 flex items-center justify-between">
        <div className="flex items-center gap-2 text-[10px] font-theme-data text-text-muted/50">
          {result.provider && <span>Provider: {result.provider}</span>}
          {result.model && <span>| Model: {result.model}</span>}
        </div>
        {onCreateDebate && (
          <button
            onClick={() => onCreateDebate(result.text)}
            className="px-3 py-1.5 bg-[var(--accent)]/20 border border-[var(--accent)]/50 text-[var(--accent)] font-theme-data text-xs hover:bg-[var(--accent)]/30 transition-colors"
          >
            CREATE DEBATE FROM TEXT
          </button>
        )}
      </div>
    </div>
  );
}

export type { TranscriptionResult, TranscriptionSegment };
