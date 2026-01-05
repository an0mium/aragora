'use client';

import { useState, useMemo } from 'react';
import type { StreamEvent } from '@/types/events';
import { isAgentMessage } from '@/types/events';
import { RoleBadge } from './RoleBadge';
import { getAgentColors, AGENT_COLORS } from '@/utils/agentColors';

interface CompareViewProps {
  events: StreamEvent[];
  onClose: () => void;
}

interface AgentData {
  name: string;
  content: string;
  role: string;
  cognitiveRole?: string;
  round: number;
  confidence?: number;
  timestamp: number;
}

export function CompareView({ events, onClose }: CompareViewProps) {
  const [selectedAgents, setSelectedAgents] = useState<[string | null, string | null]>([null, null]);
  const [selectedRound, setSelectedRound] = useState<number | 'latest'>('latest');

  // Extract all agents and their messages
  const { agents, rounds } = useMemo(() => {
    const agentMap: Record<string, AgentData[]> = {};
    const roundSet = new Set<number>();

    events.forEach((event) => {
      if (event.type === 'agent_message' && event.agent) {
        const round = event.round || 0;
        roundSet.add(round);

        if (!agentMap[event.agent]) {
          agentMap[event.agent] = [];
        }

        agentMap[event.agent].push({
          name: event.agent,
          content: event.data?.content as string || '',
          role: event.data?.role as string || 'proposer',
          cognitiveRole: event.data?.cognitive_role as string,
          round,
          confidence: event.data?.confidence as number,
          timestamp: event.timestamp,
        });
      }
    });

    return {
      agents: Object.keys(agentMap).sort(),
      rounds: Array.from(roundSet).sort((a, b) => a - b),
      agentMap,
    };
  }, [events]);

  // Get agent data for comparison
  const getAgentResponse = (agentName: string | null): AgentData | null => {
    if (!agentName) return null;

    const messages = events
      .filter(isAgentMessage)
      .filter((e) => e.agent === agentName)
      .map((e) => ({
        name: agentName,
        content: e.data.content || '',
        role: e.data.role || 'proposer',
        cognitiveRole: e.data.cognitive_role,
        round: e.round || 0,
        confidence: e.data.confidence,
        timestamp: e.timestamp,
      }));

    if (messages.length === 0) return null;

    if (selectedRound === 'latest') {
      return messages[messages.length - 1];
    }

    return messages.find((m) => m.round === selectedRound) || messages[messages.length - 1];
  };

  const leftAgent = getAgentResponse(selectedAgents[0]);
  const rightAgent = getAgentResponse(selectedAgents[1]);

  // Auto-select first two agents if none selected
  if (!selectedAgents[0] && !selectedAgents[1] && agents.length >= 2) {
    setSelectedAgents([agents[0], agents[1]]);
  }

  return (
    <div className="fixed inset-0 bg-bg/95 z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-border">
        <div className="flex items-center gap-4">
          <h2 className="text-lg font-semibold">Compare Agents</h2>

          {/* Round Selector */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-text-muted">Round:</span>
            <select
              value={selectedRound}
              onChange={(e) => setSelectedRound(e.target.value === 'latest' ? 'latest' : parseInt(e.target.value))}
              className="bg-surface border border-border rounded px-2 py-1 text-sm"
            >
              <option value="latest">Latest</option>
              {rounds.map((r) => (
                <option key={r} value={r}>Round {r}</option>
              ))}
            </select>
          </div>
        </div>

        <button
          onClick={onClose}
          className="px-4 py-2 text-sm bg-surface border border-border rounded hover:bg-surface-hover"
        >
          Close
        </button>
      </div>

      {/* Comparison Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel */}
        <ComparisonPane
          agents={agents}
          selectedAgent={selectedAgents[0]}
          onSelectAgent={(name) => setSelectedAgents([name, selectedAgents[1]])}
          agentData={leftAgent}
          position="left"
        />

        {/* Divider */}
        <div className="w-px bg-border" />

        {/* Right Panel */}
        <ComparisonPane
          agents={agents}
          selectedAgent={selectedAgents[1]}
          onSelectAgent={(name) => setSelectedAgents([selectedAgents[0], name])}
          agentData={rightAgent}
          position="right"
        />
      </div>
    </div>
  );
}

interface ComparisonPaneProps {
  agents: string[];
  selectedAgent: string | null;
  onSelectAgent: (name: string) => void;
  agentData: AgentData | null;
  position: 'left' | 'right';
}

function ComparisonPane({ agents, selectedAgent, onSelectAgent, agentData, position }: ComparisonPaneProps) {
  const colors = selectedAgent ? getAgentColors(selectedAgent) : AGENT_COLORS.default;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Agent Selector */}
      <div className="p-4 border-b border-border">
        <select
          value={selectedAgent || ''}
          onChange={(e) => onSelectAgent(e.target.value)}
          className={`w-full bg-surface border rounded px-3 py-2 text-sm font-medium ${colors.border} ${colors.text}`}
        >
          <option value="" disabled>Select Agent</option>
          {agents.map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </div>

      {/* Agent Response */}
      <div className="flex-1 overflow-y-auto p-4">
        {agentData ? (
          <div>
            {/* Meta */}
            <div className="flex items-center gap-3 mb-4">
              <RoleBadge role={agentData.role} cognitiveRole={agentData.cognitiveRole} size="sm" />
              {agentData.round > 0 && (
                <span className="px-2 py-0.5 text-xs bg-surface rounded border border-border">
                  Round {agentData.round}
                </span>
              )}
              {agentData.confidence !== undefined && (
                <span className={`text-xs font-mono ${
                  agentData.confidence >= 0.8 ? 'text-green-400' :
                  agentData.confidence >= 0.6 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {Math.round(agentData.confidence * 100)}%
                </span>
              )}
            </div>

            {/* Content */}
            <div className={`p-4 rounded-lg ${colors.bg} border ${colors.border}`}>
              <div className="agent-output whitespace-pre-wrap break-words">
                {agentData.content}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-text-muted">
            Select an agent to view their response
          </div>
        )}
      </div>
    </div>
  );
}

// Button to open comparison mode
export function CompareButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 text-sm bg-surface border border-border rounded hover:bg-surface-hover flex items-center gap-2"
    >
      <span>⚡</span>
      <span>Compare</span>
    </button>
  );
}
