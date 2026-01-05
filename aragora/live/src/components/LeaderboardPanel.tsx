'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
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

interface AgentReputation {
  agent: string;
  score: number;
  vote_weight: number;
  proposal_acceptance_rate: number;
  critique_value: number;
  debates_participated: number;
}

interface TeamCombination {
  agents: string[];
  success_rate: number;
  total_debates: number;
  wins: number;
}

interface RankingStats {
  mean_elo: number;
  median_elo: number;
  total_agents: number;
  total_matches: number;
  rating_distribution: Record<string, number>;
  trending_up: string[];
  trending_down: string[];
}

interface AgentIntrospection {
  agent: string;
  self_model: {
    strengths: string[];
    weaknesses: string[];
    biases: string[];
  };
  confidence_calibration: number;
  recent_performance_assessment: string;
  improvement_focus: string[];
  last_updated: string;
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
  const [reputations, setReputations] = useState<AgentReputation[]>([]);
  const [teams, setTeams] = useState<TeamCombination[]>([]);
  const [stats, setStats] = useState<RankingStats | null>(null);
  const [introspections, setIntrospections] = useState<AgentIntrospection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'rankings' | 'matches' | 'reputation' | 'teams' | 'stats' | 'minds'>('rankings');
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

      const [leaderboardRes, matchesRes, reputationRes, teamsRes, statsRes, introspectionRes] = await Promise.all([
        fetch(`${apiBase}/api/leaderboard?${leaderboardParams}`),
        fetch(`${apiBase}/api/matches/recent?${matchesParams}`),
        fetch(`${apiBase}/api/reputation/all`),
        fetch(`${apiBase}/api/routing/best-teams?min_debates=3&limit=10`),
        fetch(`${apiBase}/api/ranking/stats`),
        fetch(`${apiBase}/api/introspection/all`),
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

      if (reputationRes.ok) {
        const data = await reputationRes.json();
        setReputations(data.reputations || []);
      }

      if (teamsRes.ok) {
        const data = await teamsRes.json();
        setTeams(data.combinations || []);
      }

      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data);
      }

      if (introspectionRes.ok) {
        const data = await introspectionRes.json();
        setIntrospections(data.agents || []);
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
          Matches
        </button>
        <button
          onClick={() => setActiveTab('reputation')}
          className={`px-3 py-1 rounded text-sm transition-colors flex-1 ${
            activeTab === 'reputation'
              ? 'bg-accent text-bg font-medium'
              : 'text-text-muted hover:text-text'
          }`}
        >
          Reputation
        </button>
        <button
          onClick={() => setActiveTab('teams')}
          className={`px-3 py-1 rounded text-sm transition-colors flex-1 ${
            activeTab === 'teams'
              ? 'bg-accent text-bg font-medium'
              : 'text-text-muted hover:text-text'
          }`}
        >
          Teams
        </button>
        <button
          onClick={() => setActiveTab('stats')}
          className={`px-3 py-1 rounded text-sm transition-colors flex-1 ${
            activeTab === 'stats'
              ? 'bg-accent text-bg font-medium'
              : 'text-text-muted hover:text-text'
          }`}
        >
          Stats
        </button>
        <button
          onClick={() => setActiveTab('minds')}
          className={`px-3 py-1 rounded text-sm transition-colors flex-1 ${
            activeTab === 'minds'
              ? 'bg-accent text-bg font-medium'
              : 'text-text-muted hover:text-text'
          }`}
        >
          Minds
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
                  <Link
                    href={`/agent/${encodeURIComponent(agent.name)}/`}
                    className="text-sm font-medium text-text hover:text-accent transition-colors cursor-pointer"
                    title="View agent profile"
                  >
                    {agent.name}
                  </Link>
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

      {/* Reputation Tab */}
      {activeTab === 'reputation' && (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {loading && (
            <div className="text-center text-text-muted py-4">Loading reputation data...</div>
          )}

          {!loading && reputations.length === 0 && (
            <div className="text-center text-text-muted py-4">
              No reputation data yet. Run debate cycles to build agent reputations.
            </div>
          )}

          {reputations.map((rep, index) => (
            <div
              key={rep.agent}
              className="flex items-center gap-3 p-2 bg-bg border border-border rounded-lg hover:border-accent/50 transition-colors"
            >
              {/* Rank */}
              <div className="w-6 h-6 flex items-center justify-center rounded-full text-xs font-bold bg-surface text-text-muted">
                {index + 1}
              </div>

              {/* Agent Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <Link
                    href={`/agent/${encodeURIComponent(rep.agent)}/`}
                    className="text-sm font-medium text-text hover:text-accent transition-colors cursor-pointer"
                    title="View agent profile"
                  >
                    {rep.agent}
                  </Link>
                  <span className={`text-sm font-mono font-bold ${rep.score >= 0.7 ? 'text-green-400' : rep.score >= 0.4 ? 'text-yellow-400' : 'text-red-400'}`}>
                    {(rep.score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex gap-3 text-xs text-text-muted">
                  <span title="Vote weight in consensus">
                    Vote: <span className="text-text">{rep.vote_weight.toFixed(2)}x</span>
                  </span>
                  <span title="Proposal acceptance rate">
                    Accept: <span className="text-text">{(rep.proposal_acceptance_rate * 100).toFixed(0)}%</span>
                  </span>
                  <span title="Critique value score">
                    Critique: <span className="text-text">{rep.critique_value.toFixed(2)}</span>
                  </span>
                </div>
              </div>

              {/* Debates count */}
              <div className="text-xs text-text-muted">
                {rep.debates_participated} debates
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Teams Tab */}
      {activeTab === 'teams' && (
        <div className="space-y-2 max-h-80 overflow-y-auto">
          {loading && (
            <div className="text-center text-text-muted py-4">Loading team data...</div>
          )}

          {!loading && teams.length === 0 && (
            <div className="text-center text-text-muted py-4">
              No team combinations yet. Run more debates to find winning teams.
            </div>
          )}

          {teams.map((team, index) => (
            <div
              key={team.agents.join('-')}
              className="flex items-center gap-3 p-2 bg-bg border border-border rounded-lg hover:border-accent/50 transition-colors"
            >
              {/* Rank Badge */}
              <div
                className={`w-7 h-7 flex items-center justify-center rounded-full text-xs font-bold border ${getRankBadge(index + 1)}`}
              >
                {index + 1}
              </div>

              {/* Team Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  {team.agents.map((agent, i) => (
                    <Link
                      key={agent}
                      href={`/agent/${encodeURIComponent(agent)}/`}
                      className="text-sm font-medium text-text hover:text-accent transition-colors cursor-pointer"
                      title="View agent profile"
                    >
                      {agent}{i < team.agents.length - 1 && <span className="text-text-muted ml-1">+</span>}
                    </Link>
                  ))}
                </div>
                <div className="text-xs text-text-muted">
                  {team.wins}W / {team.total_debates} debates
                </div>
              </div>

              {/* Success Rate */}
              <div className={`text-sm font-mono font-bold ${team.success_rate >= 0.7 ? 'text-green-400' : team.success_rate >= 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>
                {(team.success_rate * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Stats Tab */}
      {activeTab === 'stats' && (
        <div className="space-y-3 max-h-80 overflow-y-auto">
          {loading && (
            <div className="text-center text-text-muted py-4">Loading stats...</div>
          )}

          {!loading && !stats && (
            <div className="text-center text-text-muted py-4">
              No ranking stats yet. Run debates to generate statistics.
            </div>
          )}

          {stats && (
            <>
              {/* Key Metrics */}
              <div className="grid grid-cols-2 gap-2">
                <div className="p-3 bg-bg border border-border rounded-lg">
                  <div className="text-xs text-text-muted">Mean ELO</div>
                  <div className="text-xl font-bold text-accent">{stats.mean_elo?.toFixed(0) || 'N/A'}</div>
                </div>
                <div className="p-3 bg-bg border border-border rounded-lg">
                  <div className="text-xs text-text-muted">Median ELO</div>
                  <div className="text-xl font-bold text-text">{stats.median_elo?.toFixed(0) || 'N/A'}</div>
                </div>
                <div className="p-3 bg-bg border border-border rounded-lg">
                  <div className="text-xs text-text-muted">Total Agents</div>
                  <div className="text-xl font-bold text-text">{stats.total_agents || 0}</div>
                </div>
                <div className="p-3 bg-bg border border-border rounded-lg">
                  <div className="text-xs text-text-muted">Total Matches</div>
                  <div className="text-xl font-bold text-text">{stats.total_matches || 0}</div>
                </div>
              </div>

              {/* Rating Distribution */}
              {stats.rating_distribution && Object.keys(stats.rating_distribution).length > 0 && (
                <div className="p-3 bg-bg border border-border rounded-lg">
                  <div className="text-xs text-text-muted mb-2">Rating Distribution</div>
                  <div className="space-y-1">
                    {Object.entries(stats.rating_distribution)
                      .sort((a, b) => parseInt(b[0]) - parseInt(a[0]))
                      .map(([tier, count]) => (
                        <div key={tier} className="flex items-center gap-2">
                          <span className="text-xs text-text-muted w-16">{tier}+</span>
                          <div className="flex-1 h-2 bg-surface rounded">
                            <div
                              className="h-full bg-accent rounded"
                              style={{ width: `${Math.min((count / stats.total_agents) * 100, 100)}%` }}
                            />
                          </div>
                          <span className="text-xs text-text w-8 text-right">{count}</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Trending Agents */}
              {((stats.trending_up && stats.trending_up.length > 0) ||
                (stats.trending_down && stats.trending_down.length > 0)) && (
                <div className="grid grid-cols-2 gap-2">
                  {stats.trending_up && stats.trending_up.length > 0 && (
                    <div className="p-3 bg-bg border border-border rounded-lg">
                      <div className="text-xs text-green-400 mb-1">Trending Up</div>
                      {stats.trending_up.slice(0, 3).map((agent) => (
                        <div key={agent} className="text-sm text-text truncate">{agent}</div>
                      ))}
                    </div>
                  )}
                  {stats.trending_down && stats.trending_down.length > 0 && (
                    <div className="p-3 bg-bg border border-border rounded-lg">
                      <div className="text-xs text-red-400 mb-1">Trending Down</div>
                      {stats.trending_down.slice(0, 3).map((agent) => (
                        <div key={agent} className="text-sm text-text truncate">{agent}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Minds (Introspection) Tab */}
      {activeTab === 'minds' && (
        <div className="space-y-3 max-h-80 overflow-y-auto">
          {loading && (
            <div className="text-center text-text-muted py-4">Loading introspection data...</div>
          )}

          {!loading && introspections.length === 0 && (
            <div className="text-center text-text-muted py-4">
              No introspection data yet. Agents build self-models through debate participation.
            </div>
          )}

          {introspections.map((intro) => (
            <div
              key={intro.agent}
              className="p-3 bg-bg border border-border rounded-lg hover:border-purple-500/30 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <Link
                  href={`/agent/${encodeURIComponent(intro.agent)}/`}
                  className="font-medium text-text hover:text-accent transition-colors"
                >
                  {intro.agent}
                </Link>
                <span className={`text-xs px-2 py-0.5 rounded ${
                  intro.confidence_calibration >= 0.8 ? 'bg-green-500/20 text-green-400' :
                  intro.confidence_calibration >= 0.5 ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-red-500/20 text-red-400'
                }`}>
                  {Math.round(intro.confidence_calibration * 100)}% calibrated
                </span>
              </div>

              {/* Performance Assessment */}
              {intro.recent_performance_assessment && (
                <p className="text-sm text-text-muted mb-2 italic">
                  &quot;{intro.recent_performance_assessment}&quot;
                </p>
              )}

              {/* Strengths / Weaknesses / Biases */}
              <div className="grid grid-cols-3 gap-2 text-xs mb-2">
                {intro.self_model?.strengths?.length > 0 && (
                  <div>
                    <div className="text-green-400 mb-1">Strengths</div>
                    {intro.self_model.strengths.slice(0, 2).map((s, i) => (
                      <div key={i} className="text-text-muted truncate">{s}</div>
                    ))}
                  </div>
                )}
                {intro.self_model?.weaknesses?.length > 0 && (
                  <div>
                    <div className="text-yellow-400 mb-1">Weaknesses</div>
                    {intro.self_model.weaknesses.slice(0, 2).map((w, i) => (
                      <div key={i} className="text-text-muted truncate">{w}</div>
                    ))}
                  </div>
                )}
                {intro.self_model?.biases?.length > 0 && (
                  <div>
                    <div className="text-red-400 mb-1">Known Biases</div>
                    {intro.self_model.biases.slice(0, 2).map((b, i) => (
                      <div key={i} className="text-text-muted truncate">{b}</div>
                    ))}
                  </div>
                )}
              </div>

              {/* Improvement Focus */}
              {intro.improvement_focus?.length > 0 && (
                <div className="text-xs">
                  <span className="text-purple-400">Focus:</span>{' '}
                  <span className="text-text-muted">{intro.improvement_focus.slice(0, 2).join(', ')}</span>
                </div>
              )}

              <div className="text-xs text-text-muted mt-2">
                Updated: {intro.last_updated ? new Date(intro.last_updated).toLocaleDateString() : 'N/A'}
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
