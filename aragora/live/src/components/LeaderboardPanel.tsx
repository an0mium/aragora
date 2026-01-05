'use client';

import { useState, useEffect, useCallback } from 'react';
import { AgentMomentsModal } from './AgentMomentsModal';

interface AgentRanking {
  name: string;
  elo: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  games: number;
  consistency?: number;  // 0-1 consistency score from FlipDetector
  consistency_class?: string;  // "high" | "medium" | "low"
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
  loopId?: string | null;
  apiBase?: string;
}

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

export function LeaderboardPanel({ wsMessages = [], loopId, apiBase = DEFAULT_API_BASE }: LeaderboardPanelProps) {
  const [agents, setAgents] = useState<AgentRanking[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'rankings' | 'matches'>('rankings');
  const [lastEventId, setLastEventId] = useState<string | null>(null);
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);
  const [availableDomains, setAvailableDomains] = useState<string[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);

      // Build query params with loop_id and domain filtering
      const leaderboardParams = new URLSearchParams({ limit: '10' });
      const matchesParams = new URLSearchParams({ limit: '5' });

      if (loopId) {
        leaderboardParams.set('loop_id', loopId);
        matchesParams.set('loop_id', loopId);
      }
      if (selectedDomain) {
        leaderboardParams.set('domain', selectedDomain);
        matchesParams.set('domain', selectedDomain);
      }

      const [leaderboardRes, matchesRes] = await Promise.all([
        fetch(`${apiBase}/api/leaderboard?${leaderboardParams}`),
        fetch(`${apiBase}/api/matches/recent?${matchesParams}`),
      ]);

      if (leaderboardRes.ok) {
        const data = await leaderboardRes.json();
        const agentsList: AgentRanking[] = data.agents || [];

        // Fetch consistency scores for each agent in parallel
        const consistencyPromises = agentsList.map(agent =>
          fetch(`${apiBase}/api/agent/${agent.name}/consistency`)
            .then(res => res.ok ? res.json() : null)
            .catch(() => null)
        );
        const consistencies = await Promise.all(consistencyPromises);

        // Merge consistency data into agents
        const agentsWithConsistency = agentsList.map((agent, idx) => {
          const consistencyData = consistencies[idx];
          if (consistencyData && typeof consistencyData.consistency === 'number') {
            return {
              ...agent,
              consistency: consistencyData.consistency,
              consistency_class: consistencyData.consistency_class,
            };
          }
          return agent;
        });

        setAgents(agentsWithConsistency);
        // Extract unique domains from matches for the domain filter
        if (data.domains) {
          setAvailableDomains(data.domains);
        }
      }

      if (matchesRes.ok) {
        const data = await matchesRes.json();
        setMatches(data.matches || []);
        // Also extract domains from recent matches
        const domainSet = new Set<string>(data.matches?.map((m: Match) => m.domain).filter(Boolean) || []);
        const matchDomains = Array.from(domainSet);
        if (matchDomains.length > 0) {
          setAvailableDomains(prev => {
            const combined = new Set<string>([...prev, ...matchDomains]);
            return Array.from(combined);
          });
        }
      }

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch leaderboard');
    } finally {
      setLoading(false);
    }
  }, [apiBase, loopId, selectedDomain]);

  useEffect(() => {
    fetchData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Listen for match_recorded WebSocket events for real-time updates (debate consensus feature)
  useEffect(() => {
    const matchEvents = wsMessages.filter((msg) => {
      if (msg.type !== 'match_recorded') return false;
      // Filter by loopId if specified (multi-loop support)
      if (loopId && msg.data?.loop_id && msg.data.loop_id !== loopId) return false;
      return true;
    });

    if (matchEvents.length > 0) {
      const latestEvent = matchEvents[matchEvents.length - 1];
      const eventId = latestEvent.data?.debate_id;

      // Only refresh if this is a new match event
      if (eventId && eventId !== lastEventId) {
        setLastEventId(eventId);
        fetchData(); // Refresh leaderboard when a match is recorded

        // Track new domains from match events
        const eventDomain = latestEvent.data?.domain;
        if (eventDomain && !availableDomains.includes(eventDomain)) {
          setAvailableDomains(prev => [...prev, eventDomain]);
        }
      }
    }
  }, [wsMessages, lastEventId, fetchData, loopId, availableDomains]);

  const getEloColor = (elo: number): string => {
    if (elo >= 1600) return 'text-green-400';
    if (elo >= 1500) return 'text-yellow-400';
    if (elo >= 1400) return 'text-orange-400';
    return 'text-red-400';
  };

  const getConsistencyColor = (consistency: number): string => {
    if (consistency >= 0.8) return 'text-green-400';
    if (consistency >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConsistencyClass = (consistency: number): string => {
    if (consistency >= 0.8) return 'high';
    if (consistency >= 0.6) return 'medium';
    return 'low';
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

      {/* Domain Filter */}
      {availableDomains.length > 0 && (
        <div className="flex items-center gap-2 mb-3">
          <span className="text-xs text-text-muted">Domain:</span>
          <select
            value={selectedDomain || ''}
            onChange={(e) => setSelectedDomain(e.target.value || null)}
            className="flex-1 bg-bg border border-border rounded px-2 py-1 text-sm text-text"
          >
            <option value="">All Domains</option>
            {availableDomains.map((domain) => (
              <option key={domain} value={domain}>
                {domain}
              </option>
            ))}
          </select>
        </div>
      )}

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
                  <button
                    onClick={() => setSelectedAgent(agent.name)}
                    className="text-sm font-medium text-text hover:text-accent transition-colors cursor-pointer"
                    title="View agent timeline"
                  >
                    {agent.name}
                  </button>
                  <span className={`text-sm font-mono font-bold ${getEloColor(agent.elo)}`}>
                    {agent.elo}
                  </span>
                  {agent.consistency !== undefined && (
                    <span
                      className={`text-xs px-1.5 py-0.5 rounded ${getConsistencyColor(agent.consistency)} bg-surface`}
                      title={`Consistency: ${(agent.consistency * 100).toFixed(0)}%`}
                    >
                      {(agent.consistency * 100).toFixed(0)}%
                    </span>
                  )}
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

      {/* Agent Moments Modal */}
      {selectedAgent && (
        <AgentMomentsModal
          agentName={selectedAgent}
          onClose={() => setSelectedAgent(null)}
          apiBase={apiBase}
        />
      )}
    </div>
  );
}
