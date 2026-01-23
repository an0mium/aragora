'use client';

import React, { useState } from 'react';

export interface AgentRankingEntry {
  rank: number;
  agent_name: string;
  elo: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  games_played: number;
  calibration_score?: number;
  response_time_ms?: number;
}

interface AgentLeaderboardProps {
  /** Leaderboard data */
  agents: AgentRankingEntry[];
  /** Loading state */
  loading?: boolean;
  /** Title */
  title?: string;
  /** Maximum agents to display */
  limit?: number;
  /** Callback when agent is selected */
  onAgentClick?: (agent: AgentRankingEntry) => void;
  /** Show ELO change indicators (reserved for future use) */
  showTrends?: boolean;
  /** Additional CSS classes */
  className?: string;
}

type SortField = 'rank' | 'elo' | 'win_rate' | 'games_played';
type SortDirection = 'asc' | 'desc';

export function AgentLeaderboard({
  agents,
  loading = false,
  title = 'AGENT LEADERBOARD',
  limit = 10,
  onAgentClick,
  showTrends: _showTrends = false,
  className = '',
}: AgentLeaderboardProps) {
  const [sortField, setSortField] = useState<SortField>('rank');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection(field === 'rank' ? 'asc' : 'desc');
    }
  };

  const sortedAgents = [...agents].sort((a, b) => {
    const multiplier = sortDirection === 'asc' ? 1 : -1;
    switch (sortField) {
      case 'rank':
        return (a.rank - b.rank) * multiplier;
      case 'elo':
        return (a.elo - b.elo) * multiplier;
      case 'win_rate':
        return (a.win_rate - b.win_rate) * multiplier;
      case 'games_played':
        return (a.games_played - b.games_played) * multiplier;
      default:
        return 0;
    }
  }).slice(0, limit);

  const getRankStyle = (rank: number): string => {
    switch (rank) {
      case 1:
        return 'text-acid-yellow';
      case 2:
        return 'text-gray-300';
      case 3:
        return 'text-orange-400';
      default:
        return 'text-text-muted';
    }
  };

  const getRankIcon = (rank: number): string => {
    switch (rank) {
      case 1:
        return '[1ST]';
      case 2:
        return '[2ND]';
      case 3:
        return '[3RD]';
      default:
        return `[${rank}]`;
    }
  };

  const SortHeader = ({ field, label }: { field: SortField; label: string }) => (
    <th
      className="text-right p-3 text-text-muted font-mono text-xs cursor-pointer hover:text-acid-green transition-colors"
      onClick={() => handleSort(field)}
    >
      <span className="flex items-center justify-end gap-1">
        {label}
        {sortField === field && (
          <span className="text-acid-green">{sortDirection === 'asc' ? '^' : 'v'}</span>
        )}
      </span>
    </th>
  );

  if (loading) {
    return (
      <div className={`card p-4 ${className}`}>
        <h3 className="font-mono text-sm text-acid-green mb-4">{'>'} {title}</h3>
        <div className="animate-pulse space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-10 bg-surface rounded" />
          ))}
        </div>
      </div>
    );
  }

  if (agents.length === 0) {
    return (
      <div className={`card p-4 ${className}`}>
        <h3 className="font-mono text-sm text-acid-green mb-4">{'>'} {title}</h3>
        <div className="text-center text-text-muted font-mono text-sm py-8">
          No agent data available
        </div>
      </div>
    );
  }

  return (
    <div className={`card overflow-hidden ${className}`}>
      <div className="p-4 border-b border-acid-green/20">
        <h3 className="font-mono text-sm text-acid-green">{'>'} {title}</h3>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-acid-green/20 bg-acid-green/5">
              <th
                className="text-left p-3 text-text-muted font-mono cursor-pointer hover:text-acid-green transition-colors"
                onClick={() => handleSort('rank')}
              >
                <span className="flex items-center gap-1">
                  Rank
                  {sortField === 'rank' && (
                    <span className="text-acid-green">{sortDirection === 'asc' ? '^' : 'v'}</span>
                  )}
                </span>
              </th>
              <th className="text-left p-3 text-text-muted font-mono">Agent</th>
              <SortHeader field="elo" label="ELO" />
              <SortHeader field="win_rate" label="Win Rate" />
              <th className="text-right p-3 text-text-muted font-mono">W/L/D</th>
              <SortHeader field="games_played" label="Games" />
              {agents.some(a => a.calibration_score !== undefined) && (
                <th className="text-right p-3 text-text-muted font-mono">Calibration</th>
              )}
            </tr>
          </thead>
          <tbody>
            {sortedAgents.map((agent) => (
              <tr
                key={agent.agent_name}
                className={`border-b border-acid-green/10 hover:bg-acid-green/5 transition-colors ${
                  onAgentClick ? 'cursor-pointer' : ''
                }`}
                onClick={() => onAgentClick?.(agent)}
              >
                <td className={`p-3 font-mono font-bold ${getRankStyle(agent.rank)}`}>
                  {getRankIcon(agent.rank)}
                </td>
                <td className="p-3">
                  <span className="font-mono text-acid-cyan">{agent.agent_name}</span>
                </td>
                <td className="p-3 text-right">
                  <span className="font-mono text-purple-400">{Math.round(agent.elo)}</span>
                </td>
                <td className="p-3 text-right">
                  <span className={`font-mono ${agent.win_rate >= 50 ? 'text-acid-green' : 'text-crimson'}`}>
                    {agent.win_rate.toFixed(1)}%
                  </span>
                </td>
                <td className="p-3 text-right font-mono">
                  <span className="text-acid-green">{agent.wins}</span>
                  <span className="text-text-muted">/</span>
                  <span className="text-crimson">{agent.losses}</span>
                  <span className="text-text-muted">/</span>
                  <span className="text-acid-yellow">{agent.draws}</span>
                </td>
                <td className="p-3 text-right font-mono text-text">
                  {agent.games_played}
                </td>
                {agents.some(a => a.calibration_score !== undefined) && (
                  <td className="p-3 text-right">
                    {agent.calibration_score !== undefined ? (
                      <span className="font-mono text-acid-cyan">
                        {(agent.calibration_score * 100).toFixed(0)}%
                      </span>
                    ) : (
                      <span className="text-text-muted">-</span>
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary */}
      <div className="p-4 border-t border-acid-green/20 bg-surface/50">
        <div className="flex justify-between text-xs font-mono text-text-muted">
          <span>Total agents: <span className="text-acid-green">{agents.length}</span></span>
          <span>
            Total games: <span className="text-acid-cyan">{agents.reduce((a, b) => a + b.games_played, 0)}</span>
          </span>
          <span>
            Avg ELO: <span className="text-purple-400">
              {Math.round(agents.reduce((a, b) => a + b.elo, 0) / agents.length)}
            </span>
          </span>
        </div>
      </div>
    </div>
  );
}

export default AgentLeaderboard;
