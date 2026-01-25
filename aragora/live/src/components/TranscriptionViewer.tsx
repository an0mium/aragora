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
    <div className={`border border-acid-green/30 bg-surface/30 ${className}`}>
      {/* Header */}
      <div className="p-3 border-b border-acid-green/20 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-acid-green font-mono text-xs">[TRANSCRIPTION]</span>
          <span className="text-text-muted font-mono text-[10px]">
            {wordCount} words | {formatTime(result.duration)}
          </span>
          {result.language && (
            <span className="text-acid-cyan font-mono text-[10px] uppercase">
              {result.language}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTimestamps(!showTimestamps)}
            className={`px-2 py-1 font-mono text-[10px] border transition-colors ${
              showTimestamps
                ? 'border-acid-green/50 text-acid-green bg-acid-green/10'
                : 'border-acid-green/20 text-text-muted hover:text-acid-green'
            }`}
          >
            [TIMESTAMPS]
          </button>
          <button
            onClick={handleCopy}
            className="px-2 py-1 font-mono text-[10px] border border-acid-green/20 text-text-muted hover:text-acid-green transition-colors"
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
                    ? 'bg-acid-green/10 border border-acid-green/30'
                    : 'hover:bg-surface/50'
                }`}
              >
                <span className="text-acid-cyan font-mono text-[10px] shrink-0 w-16">
                  [{formatTime(segment.start)}]
                </span>
                <span className="text-text font-mono text-sm leading-relaxed">
                  {segment.text}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-text font-mono text-sm leading-relaxed whitespace-pre-wrap">
            {result.text}
          </p>
        )}
      </div>

      {/* Footer with actions */}
      <div className="p-3 border-t border-acid-green/20 flex items-center justify-between">
        <div className="flex items-center gap-2 text-[10px] font-mono text-text-muted/50">
          {result.provider && <span>Provider: {result.provider}</span>}
          {result.model && <span>| Model: {result.model}</span>}
        </div>
        {onCreateDebate && (
          <button
            onClick={() => onCreateDebate(result.text)}
            className="px-3 py-1.5 bg-acid-green/20 border border-acid-green/50 text-acid-green font-mono text-xs hover:bg-acid-green/30 transition-colors"
          >
            CREATE DEBATE FROM TEXT
          </button>
        )}
      </div>
    </div>
  );
}

export type { TranscriptionResult, TranscriptionSegment };
