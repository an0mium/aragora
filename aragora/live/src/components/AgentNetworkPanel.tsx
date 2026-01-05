'use client';

import { useState, useEffect, useCallback } from 'react';

interface RelationshipEntry {
  agent: string;
  score: number;
  debate_count?: number;
}

interface AgentNetwork {
  agent: string;
  influences: RelationshipEntry[];
  influenced_by: RelationshipEntry[];
  rivals: RelationshipEntry[];
  allies: RelationshipEntry[];
}

interface AgentNetworkPanelProps {
  selectedAgent?: string;
  apiBase?: string;
  onAgentSelect?: (agent: string) => void;
}

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

export function AgentNetworkPanel({
  selectedAgent,
  apiBase = DEFAULT_API_BASE,
  onAgentSelect,
}: AgentNetworkPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [network, setNetwork] = useState<AgentNetwork | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentInput, setAgentInput] = useState(selectedAgent || '');
  const [availableAgents, setAvailableAgents] = useState<string[]>([]);

  // Fetch available agents from leaderboard
  useEffect(() => {
    fetch(`${apiBase}/api/leaderboard?limit=20`)
      .then((res) => res.json())
      .then((data) => {
        const agents = (data.agents || []).map((a: any) => a.name);
        setAvailableAgents(agents);
        if (!agentInput && agents.length > 0) {
          setAgentInput(agents[0]);
        }
      })
      .catch(() => {});
  }, [apiBase, agentInput]);

  const fetchNetwork = useCallback(async (agent: string) => {
    if (!agent) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiBase}/api/agent/${encodeURIComponent(agent)}/network`);
      if (!response.ok) {
        throw new Error(`Failed to fetch network: ${response.statusText}`);
      }

      const data = await response.json();
      setNetwork(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load network');
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    if (selectedAgent) {
      setAgentInput(selectedAgent);
      fetchNetwork(selectedAgent);
    }
  }, [selectedAgent, fetchNetwork]);

  const handleFetch = () => {
    if (agentInput) {
      fetchNetwork(agentInput);
    }
  };

  const renderRelationshipList = (
    title: string,
    items: RelationshipEntry[],
    icon: string,
    colorClass: string
  ) => {
    if (!items || items.length === 0) {
      return (
        <div className="text-zinc-500 text-sm">No {title.toLowerCase()} data</div>
      );
    }

    return (
      <div>
        <h4 className="text-sm font-medium text-zinc-400 mb-2 flex items-center gap-2">
          <span>{icon}</span> {title}
        </h4>
        <div className="space-y-1">
          {items.map((item) => (
            <div
              key={item.agent}
              className={`flex items-center justify-between p-2 rounded ${colorClass} cursor-pointer hover:opacity-80`}
              onClick={() => {
                setAgentInput(item.agent);
                fetchNetwork(item.agent);
                onAgentSelect?.(item.agent);
              }}
            >
              <span className="font-medium">{item.agent}</span>
              <div className="flex items-center gap-2 text-xs">
                <span className="opacity-75">
                  Score: {(item.score * 100).toFixed(0)}%
                </span>
                {item.debate_count !== undefined && (
                  <span className="opacity-50">
                    ({item.debate_count} debates)
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Collapsed view
  if (!isExpanded) {
    return (
      <div
        className="border border-blue-500/30 bg-surface/50 p-3 cursor-pointer hover:border-blue-500/50 transition-colors"
        onClick={() => setIsExpanded(true)}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-mono text-blue-400">
            {'>'} AGENT_NETWORK {network ? `[${network.agent}]` : ''}
          </h3>
          <div className="flex items-center gap-2">
            {network && (
              <span className="text-xs font-mono text-text-muted">
                {network.rivals?.length || 0} rivals, {network.allies?.length || 0} allies
              </span>
            )}
            <span className="text-xs text-text-muted">[EXPAND]</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="border border-blue-500/30 bg-surface/50 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-mono text-blue-400">
          {'>'} AGENT_NETWORK
        </h3>
        <button
          onClick={() => setIsExpanded(false)}
          className="text-xs text-text-muted hover:text-blue-400"
        >
          [COLLAPSE]
        </button>
      </div>

      {/* Agent Selector */}
      <div className="flex gap-2 mb-4">
        <select
          value={agentInput}
          onChange={(e) => setAgentInput(e.target.value)}
          className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-zinc-300"
        >
          <option value="">Select an agent...</option>
          {availableAgents.map((agent) => (
            <option key={agent} value={agent}>
              {agent}
            </option>
          ))}
        </select>
        <button
          onClick={handleFetch}
          disabled={!agentInput || loading}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded"
        >
          {loading ? 'Loading...' : 'View Network'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-900/20 border border-red-800 rounded text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Network Display */}
      {network && (
        <div className="space-y-4">
          {/* Agent Header */}
          <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
            <h4 className="text-lg font-medium text-white mb-2">
              {network.agent}&apos;s Relationship Network
            </h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="text-zinc-400">
                <span className="text-white font-medium">{network.rivals?.length || 0}</span> rivals
              </div>
              <div className="text-zinc-400">
                <span className="text-white font-medium">{network.allies?.length || 0}</span> allies
              </div>
              <div className="text-zinc-400">
                <span className="text-white font-medium">{network.influences?.length || 0}</span> influenced
              </div>
              <div className="text-zinc-400">
                <span className="text-white font-medium">{network.influenced_by?.length || 0}</span> influencers
              </div>
            </div>
          </div>

          {/* Relationship Sections */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Rivals */}
            <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
              {renderRelationshipList(
                'Rivals',
                network.rivals,
                '⚔️',
                'bg-red-900/20 border border-red-800/30 text-red-400'
              )}
            </div>

            {/* Allies */}
            <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
              {renderRelationshipList(
                'Allies',
                network.allies,
                '🤝',
                'bg-green-900/20 border border-green-800/30 text-green-400'
              )}
            </div>

            {/* Influences */}
            <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
              {renderRelationshipList(
                'Influences',
                network.influences,
                '📤',
                'bg-blue-900/20 border border-blue-800/30 text-blue-400'
              )}
            </div>

            {/* Influenced By */}
            <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
              {renderRelationshipList(
                'Influenced By',
                network.influenced_by,
                '📥',
                'bg-purple-900/20 border border-purple-800/30 text-purple-400'
              )}
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!network && !loading && !error && (
        <div className="text-center py-8 text-zinc-500">
          Select an agent to view their relationship network
        </div>
      )}

      <div className="mt-3 text-[10px] text-text-muted font-mono">
        Agent rivalry and alliance relationship visualization
      </div>
    </div>
  );
}

export default AgentNetworkPanel;
