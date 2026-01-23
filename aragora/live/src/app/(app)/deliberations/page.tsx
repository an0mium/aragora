'use client';

import { useState, useEffect, useCallback } from 'react';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import {
  DeliberationGrid,
  ConsensusFormationChart,
  AgentInfluenceNetwork,
  DeliberationStats,
  type Deliberation,
  type DeliberationEvent,
  type AgentInfluence,
  type DeliberationStatsType,
} from '@/components/deliberation-dashboard';

export default function DeliberationsPage() {
  const { config: backendConfig } = useBackend();
  const [deliberations, setDeliberations] = useState<Deliberation[]>([]);
  const [stats, setStats] = useState<DeliberationStatsType | null>(null);
  const [events, setEvents] = useState<DeliberationEvent[]>([]);
  const [agentInfluence, setAgentInfluence] = useState<AgentInfluence[]>([]);
  const [loading, setLoading] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const [filter, setFilter] = useState<'all' | 'active' | 'complete'>('all');

  // Fetch active robust decisionmaking sessions
  const fetchDeliberations = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.api}/api/v1/deliberations/active`);
      if (!response.ok) throw new Error('Failed to fetch deliberations');
      const data = await response.json();
      setDeliberations(data.deliberations || []);

      // Extract agent influence from decisionmaking sessions
      const agentMap = new Map<string, AgentInfluence>();
      (data.deliberations || []).forEach((d: Deliberation) => {
        d.agents.forEach(agent => {
          if (!agentMap.has(agent)) {
            agentMap.set(agent, {
              agent_id: agent,
              influence_score: Math.random() * 0.5 + 0.5, // Placeholder until real data
              message_count: 0,
              consensus_contributions: Math.random() * 0.3 + 0.7,
              average_confidence: Math.random() * 0.2 + 0.8,
            });
          }
        });
      });
      setAgentInfluence(Array.from(agentMap.values()));
    } catch {
      // Demo mode with mock data
      setDeliberations([
        {
          id: 'demo-1',
          task: 'Should we adopt a microservices architecture for the payment system?',
          status: 'active',
          agents: ['claude-sonnet', 'gpt-4o', 'gemini-pro'],
          current_round: 3,
          total_rounds: 5,
          consensus_score: 0.65,
          started_at: new Date(Date.now() - 300000).toISOString(),
          updated_at: new Date().toISOString(),
          message_count: 12,
        },
        {
          id: 'demo-2',
          task: 'Evaluate the security implications of the new API rate limiting strategy',
          status: 'consensus_forming',
          agents: ['claude-sonnet', 'grok-2', 'mistral-large'],
          current_round: 4,
          total_rounds: 5,
          consensus_score: 0.82,
          started_at: new Date(Date.now() - 600000).toISOString(),
          updated_at: new Date().toISOString(),
          message_count: 18,
        },
        {
          id: 'demo-3',
          task: 'Is the proposed database migration plan sufficiently robust?',
          status: 'complete',
          agents: ['claude-sonnet', 'gpt-4o'],
          current_round: 5,
          total_rounds: 5,
          consensus_score: 0.91,
          started_at: new Date(Date.now() - 1200000).toISOString(),
          updated_at: new Date(Date.now() - 60000).toISOString(),
          message_count: 24,
        },
      ]);
      setAgentInfluence([
        { agent_id: 'claude-sonnet', influence_score: 0.85, message_count: 34, consensus_contributions: 0.9, average_confidence: 0.88 },
        { agent_id: 'gpt-4o', influence_score: 0.78, message_count: 28, consensus_contributions: 0.82, average_confidence: 0.85 },
        { agent_id: 'gemini-pro', influence_score: 0.72, message_count: 18, consensus_contributions: 0.75, average_confidence: 0.8 },
        { agent_id: 'grok-2', influence_score: 0.65, message_count: 12, consensus_contributions: 0.7, average_confidence: 0.78 },
      ]);
    }
  }, [backendConfig.api]);

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.api}/api/v1/deliberations/stats`);
      if (!response.ok) throw new Error('Failed to fetch stats');
      const data = await response.json();
      setStats(data);
    } catch {
      // Demo mode
      setStats({
        active_count: 2,
        completed_today: 7,
        average_consensus_time: 420,
        average_rounds: 4.2,
        top_agents: [],
      });
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api]);

  // WebSocket for real-time updates
  useEffect(() => {
    const wsUrl = backendConfig.api.replace('http', 'ws');
    let ws: WebSocket | null = null;

    const connect = () => {
      try {
        ws = new WebSocket(`${wsUrl}/api/v1/deliberations/stream`);

        ws.onopen = () => setWsConnected(true);
        ws.onclose = () => {
          setWsConnected(false);
          // Reconnect after 5 seconds
          setTimeout(connect, 5000);
        };
        ws.onerror = () => setWsConnected(false);

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data) as DeliberationEvent;
            setEvents(prev => [...prev.slice(-99), data]);

            // Update decisionmaking sessions based on events
            if (data.type === 'consensus_progress' || data.type === 'round_complete') {
              fetchDeliberations();
            }
          } catch {
            // Ignore parse errors
          }
        };
      } catch {
        // WebSocket not available, use polling
        setWsConnected(false);
      }
    };

    connect();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [backendConfig.api, fetchDeliberations]);

  // Initial fetch and polling fallback
  useEffect(() => {
    fetchDeliberations();
    fetchStats();

    // Poll every 10 seconds if WebSocket not connected
    const interval = setInterval(() => {
      if (!wsConnected) {
        fetchDeliberations();
        fetchStats();
      }
    }, 10000);

    return () => clearInterval(interval);
  }, [fetchDeliberations, fetchStats, wsConnected]);

  // Filter decisionmaking sessions
  const filteredDeliberations = deliberations.filter(d => {
    if (filter === 'active') return d.status === 'active' || d.status === 'consensus_forming' || d.status === 'initializing';
    if (filter === 'complete') return d.status === 'complete';
    return true;
  });

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <div className="border-b border-acid-green/20 bg-surface/40">
          <div className="container mx-auto px-4 py-6">
            <div className="flex items-start justify-between">
              <div>
                <h1 className="text-2xl font-mono text-acid-green mb-2">
                  Robust Decisionmaking Dashboard
                </h1>
                <p className="text-text-muted font-mono text-sm">
                  Real-time view of multi-agent robust decisionmaking sessions across your organization
                </p>
              </div>
              <div className="flex items-center gap-2">
                {wsConnected ? (
                  <span className="flex items-center gap-2 px-3 py-1 bg-success/10 border border-success/30 text-success text-xs font-mono">
                    <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
                    LIVE
                  </span>
                ) : (
                  <span className="flex items-center gap-2 px-3 py-1 bg-acid-yellow/10 border border-acid-yellow/30 text-acid-yellow text-xs font-mono">
                    <span className="w-2 h-2 rounded-full bg-acid-yellow" />
                    POLLING
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="container mx-auto px-4 py-6 space-y-6">
          <PanelErrorBoundary panelName="Robust Decisionmaking">
            {/* Stats */}
            <DeliberationStats stats={stats} loading={loading} />

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ConsensusFormationChart events={events} height={160} />
              <AgentInfluenceNetwork agents={agentInfluence} height={160} />
            </div>

            {/* Filter tabs */}
            <div className="flex items-center gap-4 border-b border-acid-green/20 pb-4">
              <span className="text-xs font-mono text-text-muted uppercase">Filter:</span>
              {(['all', 'active', 'complete'] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`px-3 py-1 text-xs font-mono uppercase transition-colors ${
                    filter === f
                      ? 'bg-acid-green/20 text-acid-green border border-acid-green/40'
                      : 'text-text-muted hover:text-text border border-transparent'
                  }`}
                >
                  {f}
                  {f === 'active' && (
                    <span className="ml-1 text-acid-cyan">
                      ({deliberations.filter(d => d.status === 'active' || d.status === 'consensus_forming').length})
                    </span>
                  )}
                </button>
              ))}
            </div>

            {/* Robust Decisionmaking Grid */}
            <DeliberationGrid
              deliberations={filteredDeliberations}
              loading={loading}
              emptyMessage={filter === 'active' ? 'No active robust decisionmaking sessions' : 'No robust decisionmaking sessions found'}
            />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // MULTI-AGENT ROBUST DECISIONMAKING CONTROL PLANE
          </p>
        </footer>
      </main>
    </>
  );
}
