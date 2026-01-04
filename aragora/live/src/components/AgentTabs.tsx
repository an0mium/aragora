'use client';

import { useState, useMemo, useRef, useEffect } from 'react';
import type { StreamEvent } from '@/types/events';
import { RoleBadge } from './RoleBadge';

interface AgentTabsProps {
  events: StreamEvent[];
}

// Special tab ID for unified "All Agents" view
const ALL_AGENTS_TAB = '__all__';

// Agent color schemes by model family
const MODEL_COLORS: Record<string, { bg: string; text: string; border: string; tab: string }> = {
  gemini: { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/30', tab: 'bg-purple-500' },
  codex: { bg: 'bg-yellow-500/10', text: 'text-yellow-400', border: 'border-yellow-500/30', tab: 'bg-yellow-500' },
  claude: { bg: 'bg-indigo-500/10', text: 'text-indigo-400', border: 'border-indigo-500/30', tab: 'bg-indigo-500' },
  grok: { bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/30', tab: 'bg-red-500' },
  default: { bg: 'bg-surface', text: 'text-text', border: 'border-border', tab: 'bg-gray-500' },
};

function getAgentColors(agentName: string) {
  const name = agentName.toLowerCase();
  if (name.startsWith('gemini')) return MODEL_COLORS.gemini;
  if (name.startsWith('codex')) return MODEL_COLORS.codex;
  if (name.startsWith('claude')) return MODEL_COLORS.claude;
  if (name.startsWith('grok')) return MODEL_COLORS.grok;
  return MODEL_COLORS.default;
}

// Terminal-style role indicators
const ROLE_ICONS: Record<string, string> = {
  proposer: '💡',
  critic: '🔍',
  synthesizer: '🔄',
  judge: '⚖️',
  reviewer: '📋',
  implementer: '🛠️',
  default: '▶',
};

interface AgentData {
  name: string;
  latestContent: string;
  role: string;
  cognitiveRole?: string;
  round: number;
  confidence?: number;
  citations?: string[];
  timestamp: number;
  allMessages: Array<{ content: string; round: number; role: string; timestamp: number }>;
}

export function AgentTabs({ events }: AgentTabsProps) {
  // Default to "All Agents" unified timeline view
  const [selectedAgent, setSelectedAgent] = useState<string>(ALL_AGENTS_TAB);
  const [showHistory, setShowHistory] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Extract agent data from events
  const agentData = useMemo(() => {
    const agents: Record<string, AgentData> = {};

    events.forEach((event) => {
      if (event.type === 'agent_message' && event.agent) {
        const agentName = event.agent;
        const content = event.data?.content as string || '';
        const role = event.data?.role as string || 'proposer';
        const cognitiveRole = event.data?.cognitive_role as string;
        const round = event.round || 0;
        const confidence = event.data?.confidence as number;
        const citations = event.data?.citations as string[];

        if (!agents[agentName]) {
          agents[agentName] = {
            name: agentName,
            latestContent: content,
            role,
            cognitiveRole,
            round,
            confidence,
            citations,
            timestamp: event.timestamp,
            allMessages: [],
          };
        }

        agents[agentName].allMessages.push({
          content,
          round,
          role,
          timestamp: event.timestamp,
        });

        // Update to latest message
        if (event.timestamp >= agents[agentName].timestamp) {
          agents[agentName].latestContent = content;
          agents[agentName].role = role;
          agents[agentName].cognitiveRole = cognitiveRole;
          agents[agentName].round = round;
          agents[agentName].confidence = confidence;
          agents[agentName].citations = citations;
          agents[agentName].timestamp = event.timestamp;
        }
      }
    });

    return Object.values(agents).sort((a, b) => a.name.localeCompare(b.name));
  }, [events]);

  // Extract unified timeline of all agent messages
  const unifiedTimeline = useMemo(() => {
    return events
      .filter((e) => e.type === 'agent_message' && e.agent)
      .map((e) => ({
        agent: e.agent || '',
        content: (e.data?.content as string) || '',
        role: (e.data?.role as string) || 'proposer',
        cognitiveRole: e.data?.cognitive_role as string | undefined,
        round: e.round || 0,
        timestamp: e.timestamp,
      }))
      .sort((a, b) => a.timestamp - b.timestamp);
  }, [events]);

  // Auto-scroll when new messages arrive in unified view
  useEffect(() => {
    if (autoScroll && scrollRef.current && selectedAgent === ALL_AGENTS_TAB) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [unifiedTimeline.length, autoScroll, selectedAgent]);

  const handleScroll = () => {
    if (scrollRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
      setAutoScroll(isAtBottom);
    }
  };

  const activeAgent = selectedAgent;
  const currentAgent = selectedAgent !== ALL_AGENTS_TAB
    ? agentData.find((a) => a.name === activeAgent)
    : null;

  if (agentData.length === 0) {
    return (
      <div className="card flex flex-col h-full">
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h2 className="text-sm font-medium text-text-muted uppercase tracking-wider">
            Agent Responses
          </h2>
        </div>
        <div className="flex-1 flex items-center justify-center text-text-muted">
          Waiting for agent responses...
        </div>
      </div>
    );
  }

  return (
    <div className="card flex flex-col h-full">
      {/* Tab Bar */}
      <div className="flex items-center border-b border-border overflow-x-auto">
        {/* All Agents Tab (default) */}
        <button
          onClick={() => setSelectedAgent(ALL_AGENTS_TAB)}
          className={`
            relative px-4 py-3 text-sm font-medium whitespace-nowrap transition-all
            ${activeAgent === ALL_AGENTS_TAB ? 'text-acid-green' : 'text-text-muted hover:text-text'}
          `}
        >
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-acid-green" />
            All Agents
            <span className="text-xs opacity-60">{unifiedTimeline.length}</span>
          </span>
          {activeAgent === ALL_AGENTS_TAB && (
            <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-acid-green" />
          )}
        </button>
        {/* Individual Agent Tabs */}
        {agentData.map((agent) => {
          const colors = getAgentColors(agent.name);
          const isActive = agent.name === activeAgent;

          return (
            <button
              key={agent.name}
              onClick={() => setSelectedAgent(agent.name)}
              className={`
                relative px-4 py-3 text-sm font-medium whitespace-nowrap transition-all
                ${isActive ? colors.text : 'text-text-muted hover:text-text'}
              `}
            >
              <span className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${colors.tab}`}
                />
                {agent.name}
                {agent.round > 0 && (
                  <span className="text-xs opacity-60">R{agent.round}</span>
                )}
              </span>
              {isActive && (
                <span
                  className={`absolute bottom-0 left-0 right-0 h-0.5 ${colors.tab}`}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Content Area */}
      {currentAgent && (
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Agent Header */}
          <div className="p-4 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-3">
              <RoleBadge role={currentAgent.role} cognitiveRole={currentAgent.cognitiveRole} />
              {currentAgent.round > 0 && (
                <span className="px-2 py-0.5 text-xs bg-surface rounded border border-border">
                  Round {currentAgent.round}
                </span>
              )}
            </div>
            <div className="flex items-center gap-3">
              {currentAgent.confidence !== undefined && (
                <span className="text-sm">
                  <span className="text-text-muted">Confidence:</span>{' '}
                  <span className={`font-mono font-medium ${
                    currentAgent.confidence >= 0.8 ? 'text-green-400' :
                    currentAgent.confidence >= 0.6 ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {Math.round(currentAgent.confidence * 100)}%
                  </span>
                </span>
              )}
              {currentAgent.citations && currentAgent.citations.length > 0 && (
                <span className="text-sm text-text-muted">
                  Citations: {currentAgent.citations.length}
                </span>
              )}
              <button
                onClick={() => setShowHistory(!showHistory)}
                className={`px-2 py-1 text-xs rounded border transition-colors ${
                  showHistory
                    ? 'bg-accent text-white border-accent'
                    : 'bg-surface text-text-muted border-border hover:text-text'
                }`}
              >
                {showHistory ? 'Latest' : 'History'}
              </button>
            </div>
          </div>

          {/* Response Content */}
          <div className="flex-1 overflow-y-auto p-4">
            {showHistory ? (
              <div className="space-y-4">
                {currentAgent.allMessages
                  .sort((a, b) => b.timestamp - a.timestamp)
                  .map((msg, idx) => (
                    <div key={idx} className="border-l-2 border-border pl-4">
                      <div className="flex items-center gap-2 mb-2 text-xs text-text-muted">
                        <span>Round {msg.round}</span>
                        <span>•</span>
                        <span>{new Date(msg.timestamp * 1000).toLocaleTimeString()}</span>
                      </div>
                      <div className="agent-output whitespace-pre-wrap break-words">
                        {msg.content}
                      </div>
                    </div>
                  ))}
              </div>
            ) : (
              <div className="agent-output whitespace-pre-wrap break-words">
                {currentAgent.latestContent}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
