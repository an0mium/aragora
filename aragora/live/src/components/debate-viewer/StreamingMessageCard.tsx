'use client';

import { useState } from 'react';
import { getAgentColors } from '@/utils/agentColors';
import type { StreamingMessageCardProps } from './types';

export function StreamingMessageCard({ message }: StreamingMessageCardProps) {
  const colors = getAgentColors(message.agent);
  const [showReasoning, setShowReasoning] = useState(false);

  const hasReasoning = (message.reasoning && message.reasoning.length > 0) ||
    (message.evidence && message.evidence.length > 0) ||
    (message.confidence !== null && message.confidence !== undefined);

  return (
    <div className={`${colors.bg} border-2 ${colors.border} p-4 animate-pulse min-h-[120px]`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`font-mono font-bold text-sm ${colors.text}`}>{message.agent.toUpperCase()}</span>
          <span className="text-xs text-acid-cyan border border-acid-cyan/30 px-1 animate-pulse">STREAMING</span>
          {message.confidence !== null && message.confidence !== undefined && (
            <span className="text-xs text-acid-yellow border border-acid-yellow/30 px-1">
              {Math.round(message.confidence * 100)}% conf
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {hasReasoning && (
            <button
              onClick={() => setShowReasoning(!showReasoning)}
              className="text-[10px] font-mono text-text-muted hover:text-acid-green transition-colors border border-border px-1"
            >
              {showReasoning ? '[HIDE REASONING]' : '[SHOW REASONING]'}
            </button>
          )}
          <span className="text-[10px] text-text-muted font-mono">
            {Math.round((Date.now() - message.startTime) / 1000)}s
          </span>
        </div>
      </div>

      {/* Collapsible reasoning panel */}
      {showReasoning && hasReasoning && (
        <div className="mb-3 border border-acid-green/20 bg-bg/50 p-2 space-y-2">
          {message.reasoning && message.reasoning.length > 0 && (
            <div>
              <div className="text-[10px] font-mono text-acid-cyan uppercase mb-1">Reasoning Chain</div>
              <div className="space-y-1">
                {message.reasoning.map((step, idx) => (
                  <div key={idx} className="text-xs text-text-muted font-mono pl-2 border-l border-acid-cyan/30">
                    {step.step !== undefined && (
                      <span className="text-acid-cyan mr-1">#{step.step}</span>
                    )}
                    {step.thinking}
                  </div>
                ))}
              </div>
            </div>
          )}

          {message.evidence && message.evidence.length > 0 && (
            <div>
              <div className="text-[10px] font-mono text-acid-yellow uppercase mb-1">Evidence Sources</div>
              <div className="space-y-1">
                {message.evidence.map((source, idx) => (
                  <div key={idx} className="text-xs text-text-muted font-mono pl-2 border-l border-acid-yellow/30">
                    {source.title}
                    {source.relevance !== undefined && (
                      <span className="text-acid-yellow ml-2">({Math.round(source.relevance * 100)}%)</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {message.confidence !== null && message.confidence !== undefined && (
            <div>
              <div className="text-[10px] font-mono text-acid-green uppercase mb-1">Confidence</div>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-1.5 bg-bg border border-border rounded-full overflow-hidden">
                  <div
                    className="h-full bg-acid-green transition-all duration-300"
                    style={{ width: `${Math.round(message.confidence * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-acid-green">{Math.round(message.confidence * 100)}%</span>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="text-sm text-text whitespace-pre-wrap">
        {message.content}
        <span className="inline-block w-2 h-4 bg-acid-cyan ml-1 animate-pulse">|</span>
      </div>
    </div>
  );
}
