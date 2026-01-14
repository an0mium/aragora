'use client';

import { useMemo } from 'react';
import { getAgentColors } from '@/utils/agentColors';
import type { TranscriptMessageCardProps, CruxClaim } from './types';

/**
 * Find fuzzy matches of crux statements in content.
 * Returns array of {start, end, crux} for each match.
 */
function findCruxMatches(
  content: string,
  cruxes: CruxClaim[]
): Array<{ start: number; end: number; crux: CruxClaim }> {
  const matches: Array<{ start: number; end: number; crux: CruxClaim }> = [];
  const contentLower = content.toLowerCase();

  for (const crux of cruxes) {
    // Look for substantial substring matches (at least 30 chars or full statement)
    const statement = crux.statement;
    const minLen = Math.min(30, statement.length);

    // Try to find exact match first
    const statementLower = statement.toLowerCase();
    let idx = contentLower.indexOf(statementLower);
    if (idx !== -1) {
      matches.push({ start: idx, end: idx + statement.length, crux });
      continue;
    }

    // Try matching first N characters (for truncated matches)
    const prefix = statementLower.slice(0, minLen);
    idx = contentLower.indexOf(prefix);
    if (idx !== -1 && minLen >= 30) {
      // Find where the match might end (look for sentence end or 200 chars)
      let endIdx = idx + minLen;
      const remaining = contentLower.slice(endIdx);
      const sentenceEnd = remaining.search(/[.!?\n]/);
      if (sentenceEnd !== -1 && sentenceEnd < 200) {
        endIdx += sentenceEnd + 1;
      } else {
        endIdx = Math.min(idx + 200, content.length);
      }
      matches.push({ start: idx, end: endIdx, crux });
    }
  }

  // Sort by start position and remove overlaps
  matches.sort((a, b) => a.start - b.start);
  const filtered: Array<{ start: number; end: number; crux: CruxClaim }> = [];
  for (const m of matches) {
    if (filtered.length === 0 || m.start >= filtered[filtered.length - 1].end) {
      filtered.push(m);
    }
  }

  return filtered;
}

/**
 * Render content with crux highlighting.
 */
function HighlightedContent({
  content,
  cruxes,
}: {
  content: string;
  cruxes?: CruxClaim[];
}) {
  const parts = useMemo(() => {
    if (!cruxes || cruxes.length === 0) {
      return [{ text: content, isHighlight: false, crux: undefined }];
    }

    const matches = findCruxMatches(content, cruxes);
    if (matches.length === 0) {
      return [{ text: content, isHighlight: false, crux: undefined }];
    }

    const result: Array<{ text: string; isHighlight: boolean; crux?: CruxClaim }> = [];
    let lastEnd = 0;

    for (const match of matches) {
      // Add text before the match
      if (match.start > lastEnd) {
        result.push({
          text: content.slice(lastEnd, match.start),
          isHighlight: false,
        });
      }
      // Add the highlighted match
      result.push({
        text: content.slice(match.start, match.end),
        isHighlight: true,
        crux: match.crux,
      });
      lastEnd = match.end;
    }

    // Add remaining text
    if (lastEnd < content.length) {
      result.push({
        text: content.slice(lastEnd),
        isHighlight: false,
      });
    }

    return result;
  }, [content, cruxes]);

  return (
    <>
      {parts.map((part, i) =>
        part.isHighlight ? (
          <span
            key={i}
            className="bg-acid-yellow/20 border-b-2 border-acid-yellow text-acid-yellow relative group cursor-help"
            title={`Crux: ${part.crux?.statement?.slice(0, 100)}${(part.crux?.statement?.length || 0) > 100 ? '...' : ''}`}
          >
            {part.text}
            <span className="absolute -top-1 -right-1 text-[8px] bg-acid-yellow text-bg-dark px-0.5 rounded font-mono opacity-0 group-hover:opacity-100 transition-opacity">
              CRUX
            </span>
          </span>
        ) : (
          <span key={i}>{part.text}</span>
        )
      )}
    </>
  );
}

export function TranscriptMessageCard({ message, cruxes }: TranscriptMessageCardProps) {
  const colors = getAgentColors(message.agent || 'system');
  // Detect synthesis messages by role or agent name
  const isSynthesis =
    message.role === 'synthesis' ||
    message.agent === 'synthesis-agent' ||
    message.agent === 'consensus';

  // Special rendering for synthesis messages - highly visible final conclusion
  if (isSynthesis) {
    return (
      <div className="relative my-6" id="synthesis-message">
        {/* Glowing border effect */}
        <div className="absolute inset-0 bg-acid-green/10 blur-xl rounded-lg" />
        {/* Synthesis header bar */}
        <div className="relative bg-acid-green/20 border-l-4 border-acid-green px-4 py-3 flex items-center justify-between rounded-t-lg">
          <div className="flex items-center gap-3">
            <span className="text-2xl">{'🎯'}</span>
            <span className="text-acid-green font-bold text-base tracking-wider">
              FINAL SYNTHESIS
            </span>
          </div>
          {message.timestamp && (
            <span className="text-[10px] text-acid-green/70 font-mono">
              {new Date(message.timestamp * 1000).toLocaleTimeString()}
            </span>
          )}
        </div>
        {/* Synthesis content */}
        <div className="relative bg-bg-secondary/90 border-2 border-acid-green/40 border-t-0 p-6 rounded-b-lg">
          <div className="text-sm text-text-primary font-medium leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>
          <div className="mt-4 pt-4 border-t border-acid-green/20 flex items-center justify-between">
            <span className="text-xs text-acid-green/70 font-mono">
              Generated by Claude Opus 4.5
            </span>
            <span className="text-xs text-acid-green/50 font-mono">
              DEBATE CONCLUSION
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Standard rendering for non-synthesis messages
  return (
    <div className={`${colors.bg} border ${colors.border} p-4`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`font-mono font-bold text-sm ${colors.text}`}>
            {(message.agent || 'SYSTEM').toUpperCase()}
          </span>
          {message.role && (
            <span className="text-xs text-text-muted border border-text-muted/30 px-1">{message.role}</span>
          )}
          {message.round !== undefined && message.round > 0 && (
            <span className="text-xs text-text-muted">R{message.round}</span>
          )}
        </div>
        {message.timestamp && (
          <span className="text-[10px] text-text-muted font-mono">
            {new Date(message.timestamp * 1000).toLocaleTimeString()}
          </span>
        )}
      </div>
      <div className="text-sm text-text whitespace-pre-wrap">
        <HighlightedContent content={message.content} cruxes={cruxes} />
      </div>
    </div>
  );
}
