'use client';

import { useEffect, useState } from 'react';

interface Team {
  team_id: string;
  team_name: string;
  member_count: number;
  cost_usd: string;
  cost_percent_of_total: number;
  debates_completed: number;
  consensus_rate_percent: number;
  avg_quality_score: number;
  most_used_agent: string;
  trend: string;
}

interface TeamMetricsData {
  period: string;
  total_teams: number;
  teams: Team[];
  summary: {
    total_cost_usd: string;
    total_debates: number;
    avg_consensus_rate_percent: number;
    top_performing_team: string;
    most_efficient_team: string;
  };
}

interface TeamMetricsProps {
  backendUrl: string;
}

export function TeamMetrics({ backendUrl }: TeamMetricsProps) {
  const [data, setData] = useState<TeamMetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`${backendUrl}/api/v1/intelligence/team-metrics`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [backendUrl]);

  if (loading) {
    return (
      <div className="card p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-surface rounded w-1/3 mb-4"></div>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 bg-surface rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-6 border-red-500/50">
        <p className="text-red-500 font-mono text-sm">Error: {error}</p>
      </div>
    );
  }

  if (!data) return null;

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing':
        return <span className="text-green-400">^</span>;
      case 'decreasing':
        return <span className="text-red-400">v</span>;
      default:
        return <span className="text-yellow-400">-</span>;
    }
  };

  return (
    <div className="card p-6">
      <h2 className="text-lg font-mono text-acid-green mb-4">Team Performance</h2>

      {/* Team list */}
      <div className="space-y-3">
        {data.teams.map((team) => (
          <div
            key={team.team_id}
            className="bg-surface/50 rounded-lg p-4 border border-acid-green/20"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="font-mono font-medium">{team.team_name}</span>
                <span className="text-xs text-text-muted">({team.member_count} members)</span>
                {getTrendIcon(team.trend)}
              </div>
              <span className="text-sm font-mono text-acid-green">${team.cost_usd}</span>
            </div>

            <div className="grid grid-cols-4 gap-4 text-xs font-mono">
              <div>
                <p className="text-text-muted">Debates</p>
                <p>{team.debates_completed}</p>
              </div>
              <div>
                <p className="text-text-muted">Consensus</p>
                <p>{team.consensus_rate_percent}%</p>
              </div>
              <div>
                <p className="text-text-muted">Quality</p>
                <p>{(team.avg_quality_score * 100).toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-text-muted">Top Agent</p>
                <p className="truncate">{team.most_used_agent}</p>
              </div>
            </div>

            {/* Cost bar */}
            <div className="mt-2 h-1 bg-bg rounded-full overflow-hidden">
              <div
                className="h-full bg-acid-green"
                style={{ width: `${team.cost_percent_of_total}%` }}
              ></div>
            </div>
            <p className="text-xs text-text-muted mt-1">
              {team.cost_percent_of_total.toFixed(1)}% of total cost
            </p>
          </div>
        ))}
      </div>

      {/* Summary footer */}
      <div className="mt-4 pt-4 border-t border-acid-green/20 text-xs font-mono">
        <div className="flex justify-between text-text-muted">
          <span>Top performer: {data.summary.top_performing_team}</span>
          <span>Most efficient: {data.summary.most_efficient_team}</span>
        </div>
      </div>
    </div>
  );
}
