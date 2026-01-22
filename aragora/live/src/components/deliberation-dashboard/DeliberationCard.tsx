'use client';

import React from 'react';
import Link from 'next/link';
import type { Deliberation } from './types';
import { getAgentColors } from '@/utils/agentColors';

interface DeliberationCardProps {
  deliberation: Deliberation;
}

const STATUS_CONFIG = {
  initializing: { color: 'bg-yellow-400', label: 'INIT', textColor: 'text-yellow-400' },
  active: { color: 'bg-acid-green animate-pulse', label: 'LIVE', textColor: 'text-acid-green' },
  consensus_forming: { color: 'bg-acid-cyan animate-pulse', label: 'CONSENSUS', textColor: 'text-acid-cyan' },
  complete: { color: 'bg-blue-400', label: 'DONE', textColor: 'text-blue-400' },
  failed: { color: 'bg-crimson', label: 'FAILED', textColor: 'text-crimson' },
} as const;

export function DeliberationCard({ deliberation }: DeliberationCardProps) {
  const status = STATUS_CONFIG[deliberation.status];
  const progress = (deliberation.current_round / deliberation.total_rounds) * 100;

  return (
    <Link
      href={`/debate/${deliberation.id}`}
      className="block bg-surface border border-acid-green/20 hover:border-acid-green/40 transition-all p-4"
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${status.color}`} />
          <span className={`text-xs font-mono uppercase ${status.textColor}`}>
            {status.label}
          </span>
        </div>
        <span className="text-xs font-mono text-text-muted">
          R{deliberation.current_round}/{deliberation.total_rounds}
        </span>
      </div>

      {/* Task */}
      <p className="text-sm font-mono text-text line-clamp-2 mb-3 min-h-[2.5rem]">
        {deliberation.task}
      </p>

      {/* Progress bar */}
      <div className="h-1 bg-bg rounded-full overflow-hidden mb-3">
        <div
          className={`h-full transition-all ${
            deliberation.status === 'consensus_forming' ? 'bg-acid-cyan' : 'bg-acid-green'
          }`}
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Consensus score */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono text-text-muted">Consensus</span>
        <span className={`text-sm font-mono ${
          deliberation.consensus_score >= 0.8 ? 'text-success' :
          deliberation.consensus_score >= 0.5 ? 'text-acid-yellow' :
          'text-text-muted'
        }`}>
          {Math.round(deliberation.consensus_score * 100)}%
        </span>
      </div>

      {/* Agents */}
      <div className="flex flex-wrap gap-1">
        {deliberation.agents.slice(0, 4).map((agent) => {
          const colors = getAgentColors(agent);
          return (
            <span
              key={agent}
              className={`px-1.5 py-0.5 text-xs font-mono ${colors.bg} ${colors.text} ${colors.border} border`}
            >
              {agent.split('-')[0]}
            </span>
          );
        })}
        {deliberation.agents.length > 4 && (
          <span className="px-1.5 py-0.5 text-xs font-mono text-text-muted">
            +{deliberation.agents.length - 4}
          </span>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-acid-green/10">
        <span className="text-xs font-mono text-text-muted">
          {deliberation.message_count} msgs
        </span>
        <span className="text-xs font-mono text-text-muted">
          {new Date(deliberation.updated_at).toLocaleTimeString()}
        </span>
      </div>
    </Link>
  );
}
