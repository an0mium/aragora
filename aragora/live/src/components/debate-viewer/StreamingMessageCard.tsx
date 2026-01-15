'use client';

import { getAgentColors } from '@/utils/agentColors';
import type { StreamingMessageCardProps } from './types';

export function StreamingMessageCard({ message }: StreamingMessageCardProps) {
  const colors = getAgentColors(message.agent);
  return (
    <div className={`${colors.bg} border-2 ${colors.border} p-4 animate-pulse min-h-[120px]`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`font-mono font-bold text-sm ${colors.text}`}>{message.agent.toUpperCase()}</span>
          <span className="text-xs text-acid-cyan border border-acid-cyan/30 px-1 animate-pulse">STREAMING</span>
        </div>
        <span className="text-[10px] text-text-muted font-mono">
          {Math.round((Date.now() - message.startTime) / 1000)}s
        </span>
      </div>
      <div className="text-sm text-text whitespace-pre-wrap">
        {message.content}
        <span className="inline-block w-2 h-4 bg-acid-cyan ml-1 animate-pulse">▌</span>
      </div>
    </div>
  );
}
