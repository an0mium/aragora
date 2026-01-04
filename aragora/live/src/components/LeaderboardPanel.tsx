'use client';

import { useState, useEffect, useCallback } from 'react';

interface AgentRanking {
  name: string;
  elo: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  games: number;
}

interface Match {
  debate_id: string;
  winner: string;
  participants: string[];
  domain: string;
  elo_changes: Record<string, number>;
  created_at: string;
}

interface LeaderboardPanelProps {
  wsMessages?: any[];
}

export function LeaderboardPanel({ wsMessages = [] }: LeaderboardPanelProps) {
  const [agents, setAgents] = useState<AgentRanking[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'rankings' | 'matches'>('rankings');
  const [lastEventId, setLastEventId] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const [leaderboardRes, matchesRes] = await Promise.all([
        fetch('/api/leaderboard?limit=10'),
        fetch('/api/matches/recent?limit=5'),
      ]);

      if (leaderboardRes.ok) {
        const data = await leaderboardRes.json();
        setAgents(data.agents || []);
      }

      if (matchesRes.ok) {
        const data = await matchesRes.json();
        setMatches(data.matches || []);
      }

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch leaderboard');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Listen for match_recorded WebSocket events for real-time updates (debate consensus feature)
  useEffect(() => {
    const matchEvents = wsMessages.filter((msg) => msg.type === 'match_recorded');
    if (matchEvents.length > 0) {
      const latestEvent = matchEvents[matchEvents.length - 1];
      const eventId = latestEvent.data?.debate_id;

      // Only refresh if this is a new match event
      if (eventId && eventId !== lastEventId) {
        setLastEventId(eventId);
        fetchData(); // Refresh leaderboard when a match is recorded
      }
    }
  }, [wsMessages, lastEventId, fetchData]);

  const getEloColor = (elo: number): string => {
    if (elo >= 1600) return 'text-green-400';
    if (elo >= 1500) return 'text-yellow-400';
    if (elo >= 1400) return 'text-orange-400';
    return 'text-red-400';
  };

  const getRankBadge = (rank: number): string => {
    if (rank === 1) return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    if (rank === 2) return 'bg-gray-400/20 text-gray-300 border-gray-400/30';
    if (rank === 3) return 'bg-amber-600/20 text-amber-500 border-amber-600/30';
    return 'bg-surface text-text-muted border-border';
  };

  const formatEloChange = (change: number): string => {
    if (change > 0) return `+${change}`;
    return String(change);
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-text">Agent Leaderboard</h3>
        <button
          onClick={fetchData}
          className="px-2 py-1 bg-surface border border-border rounded text-sm text-text hover:bg-surface-hover"
        >
          Refresh
        </button>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-bg border border-border rounded p-1 mb-4">
        <button
          onClick={() => setActiveTab('rankings')}
          className={`px-3 py-1 rounded text-sm transition-colors flex-1 ${
            activeTab === 'rankings'
              ? 'bg-accent text-bg font-medium'
              : 'text-text-muted hover:text-text'
          }`}
        >
          Rankings
        </button>
        <button
          onClick={() => setActiveTab('matches')}
          className={`px-3 py-1 rounded text-sm transition-colors flex-1 ${
            activeTab === 'matches'
              ? 'bg-accent text-bg font-medium'
              : 'text-text-muted hover:text-text'
          }`}
        >
          Recent Matches
        </button>
      </div>

      {/* Rankings Tab */}
      {activeTab === 'rankings' && (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {loading && (
            <div className="text-center text-text-muted py-4">Loading rankings...</div>
          )}

          {error && (
            <div className="text-center text-red-400 py-4 text-sm">{error}</div>
          )}

          {!loading && !error && agents.length === 0 && (
            <div className="text-center text-text-muted py-4">
              No rankings yet. Run debate cycles to generate rankings.
            </div>
          )}

          {agents.map((agent, index) => (
            <div
              key={agent.name}
              className="flex items-center gap-3 p-2 bg-bg border border-border rounded-lg hover:border-accent/50 transition-colors"
            >
              {/* Rank Badge */}
              <div
                className={`w-7 h-7 flex items-center justify-center rounded-full text-xs font-bold border ${getRankBadge(index + 1)}`}
              >
                {index + 1}
              </div>

              {/* Agent Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-text">
                    {agent.name}
                  </span>
                  <span className={`text-sm font-mono font-bold ${getEloColor(agent.elo)}`}>
                    {agent.elo}
                  </span>
                </div>
                <div className="text-xs text-text-muted">
                  {agent.wins}W-{agent.losses}L-{agent.draws}D ({agent.win_rate}%)
                </div>
              </div>

              {/* Games Played */}
              <div className="text-xs text-text-muted">
                {agent.games} games
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Recent Matches Tab */}
      {activeTab === 'matches' && (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {loading && (
            <div className="text-center text-text-muted py-4">Loading matches...</div>
          )}

          {!loading && matches.length === 0 && (
            <div className="text-center text-text-muted py-4">
              No matches yet. Run debate cycles to see match history.
            </div>
          )}

          {matches.map((match) => (
            <div
              key={match.debate_id}
              className="p-2 bg-bg border border-border rounded-lg"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-green-400">
                  {match.winner} wins
                </span>
                {match.domain && (
                  <span className="px-1.5 py-0.5 text-xs bg-surface rounded text-text-muted">
                    {match.domain}
                  </span>
                )}
              </div>

              <div className="flex flex-wrap gap-2 text-xs">
                {Object.entries(match.elo_changes).map(([agent, change]) => (
                  <span
                    key={agent}
                    className={`${change >= 0 ? 'text-green-400' : 'text-red-400'}`}
                  >
                    {agent}: {formatEloChange(change)}
                  </span>
                ))}
              </div>

              <div className="text-xs text-text-muted mt-1">
                {new Date(match.created_at).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
