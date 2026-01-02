'use client';

import { useState, useEffect, useRef } from 'react';
import type { StreamEvent } from '@/types/events';

interface AgentPanelProps {
  events: StreamEvent[];
}

// Agent colors for different models
const AGENT_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  // Gemini - purple (Google AI)
  'gemini-visionary': { bg: 'bg-purple/10', text: 'text-purple', border: 'border-purple/30' },
  'gemini': { bg: 'bg-purple/10', text: 'text-purple', border: 'border-purple/30' },
  'gemini-explorer': { bg: 'bg-purple/10', text: 'text-purple', border: 'border-purple/30' },
  // Codex - gold (OpenAI)
  'codex-engineer': { bg: 'bg-gold/10', text: 'text-gold', border: 'border-gold/30' },
  'codex': { bg: 'bg-gold/10', text: 'text-gold', border: 'border-gold/30' },
  // Claude - accent/indigo (Anthropic)
  'claude-synthesizer': { bg: 'bg-accent/10', text: 'text-accent', border: 'border-accent/30' },
  'claude-visionary': { bg: 'bg-accent/10', text: 'text-accent', border: 'border-accent/30' },
  'claude': { bg: 'bg-accent/10', text: 'text-accent', border: 'border-accent/30' },
  // Grok - crimson red (xAI)
  'grok-explorer': { bg: 'bg-crimson/10', text: 'text-crimson', border: 'border-crimson/30' },
  'grok': { bg: 'bg-crimson/10', text: 'text-crimson', border: 'border-crimson/30' },
  'grok-pragmatist': { bg: 'bg-crimson/10', text: 'text-crimson', border: 'border-crimson/30' },
  // Default
  default: { bg: 'bg-surface', text: 'text-text', border: 'border-border' },
};

const ROLE_ICONS: Record<string, string> = {
  proposer: '💡',
  critic: '🔍',
  synthesizer: '🔮',
  judge: '⚖️',
  reviewer: '📋',
  implementer: '🔧',
  default: '🤖',
};

export function AgentPanel({ events }: AgentPanelProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Filter to agent-related events, preferring agent_message over log_message
  // to avoid duplicated content (log_message has truncated versions)
  const agentEvents = events.filter((e, index, arr) => {
    // Always include these primary event types
    if (
      e.type === 'agent_message' ||
      e.type === 'critique' ||
      e.type === 'consensus' ||
      e.type === 'vote'
    ) {
      return true;
    }
    // For log_message, only include if it's not a duplicate of an agent_message
    if (e.type === 'log_message') {
      const msg = (e.data?.message as string) || '';
      // Skip log messages that look like agent message summaries (they have agent_message equivalents)
      if (msg.match(/^\s*\[proposer\]|\[critique\]/)) {
        return false;
      }
      return true;
    }
    return false;
  });

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [agentEvents.length, autoScroll]);

  // Handle scroll to detect if user has scrolled up
  const handleScroll = () => {
    if (scrollRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
      setAutoScroll(isAtBottom);
    }
  };

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const expandAll = () => {
    const allIds = agentEvents.map((_, i) => `event-${i}`);
    setExpandedIds(new Set(allIds));
  };

  const collapseAll = () => {
    setExpandedIds(new Set());
  };

  return (
    <div className="card flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b border-border">
        <h2 className="text-sm font-medium text-text-muted uppercase tracking-wider">
          Agent Activity
        </h2>
        <div className="flex gap-2">
          <button
            onClick={expandAll}
            className="text-xs text-text-muted hover:text-text px-2 py-1 rounded hover:bg-surface"
          >
            Expand All
          </button>
          <button
            onClick={collapseAll}
            className="text-xs text-text-muted hover:text-text px-2 py-1 rounded hover:bg-surface"
          >
            Collapse All
          </button>
        </div>
      </div>
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 space-y-3"
      >
        {agentEvents.length === 0 ? (
          <div className="text-center text-text-muted py-8">
            Waiting for agent activity...
          </div>
        ) : (
          agentEvents.map((event, index) => (
            <EventCard
              key={`event-${index}`}
              id={`event-${index}`}
              event={event}
              isExpanded={expandedIds.has(`event-${index}`)}
              onToggle={toggleExpand}
            />
          ))
        )}
      </div>
      {!autoScroll && agentEvents.length > 0 && (
        <button
          onClick={() => {
            setAutoScroll(true);
            if (scrollRef.current) {
              scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
            }
          }}
          className="absolute bottom-4 right-4 bg-accent text-white px-3 py-1 rounded-full text-sm shadow-lg"
        >
          ↓ New messages
        </button>
      )}
    </div>
  );
}

interface EventCardProps {
  id: string;
  event: StreamEvent;
  isExpanded: boolean;
  onToggle: (id: string) => void;
}

function EventCard({ id, event, isExpanded, onToggle }: EventCardProps) {
  const agentName = event.agent || 'system';
  const colors = AGENT_COLORS[agentName] || AGENT_COLORS.default;
  const timestamp = new Date(event.timestamp * 1000).toLocaleTimeString();

  // Get content based on event type
  let content = '';
  let preview = '';
  let role = '';
  let icon = ROLE_ICONS.default;

  switch (event.type) {
    case 'agent_message':
      content = event.data.content as string;
      role = event.data.role as string;
      icon = ROLE_ICONS[role] || ROLE_ICONS.default;
      preview = content.slice(0, 150) + (content.length > 150 ? '...' : '');
      break;
    case 'critique':
      const issues = event.data.issues as string[];
      const severity = event.data.severity as number;
      const target = event.data.target as string;
      const critiqueContent = event.data.content as string;
      // Use full content if available, otherwise format issues
      content = critiqueContent || `Issues with ${target}:\n${issues.map((i) => `• ${i}`).join('\n')}\n\nSeverity: ${severity.toFixed(1)}`;
      preview = `→ ${target}: ${issues.length} issues (severity ${severity.toFixed(1)})`;
      icon = '🔍';
      role = 'critic';
      break;
    case 'consensus':
      const reached = event.data.reached as boolean;
      const confidence = event.data.confidence as number;
      const answer = event.data.answer as string;
      content = answer;
      preview = `${reached ? '✓' : '✗'} Consensus ${reached ? 'reached' : 'not reached'} (${Math.round(confidence * 100)}%)`;
      icon = '⚖️';
      break;
    case 'vote':
      content = `Vote: ${event.data.vote} (confidence: ${event.data.confidence})`;
      preview = content;
      icon = '🗳️';
      break;
    case 'log_message':
      content = event.data.message as string;
      preview = content.slice(0, 100);
      icon = '📝';
      break;
    default:
      content = JSON.stringify(event.data, null, 2);
      preview = event.type;
  }

  return (
    <div
      className={`${colors.bg} border ${colors.border} rounded-lg overflow-hidden transition-all`}
    >
      <button
        onClick={() => onToggle(id)}
        className="w-full text-left p-3 flex items-start gap-3 hover:bg-white/5 transition-colors"
      >
        <span className="text-lg flex-shrink-0">{icon}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={`font-medium ${colors.text}`}>{agentName}</span>
            {event.round !== undefined && event.round > 0 && (
              <span className="text-xs text-text-muted bg-surface px-1.5 py-0.5 rounded">
                Round {event.round}
              </span>
            )}
            <span className="text-xs text-text-muted ml-auto">{timestamp}</span>
          </div>
          <p className="text-sm text-text-muted truncate">{preview}</p>
        </div>
        <span className="text-text-muted flex-shrink-0">
          {isExpanded ? '▼' : '▶'}
        </span>
      </button>
      {isExpanded && (
        <div className="px-3 pb-3 pt-0">
          <div className="bg-bg/50 rounded-lg p-3 text-sm whitespace-pre-wrap break-words overflow-y-auto">
            {content}
          </div>
        </div>
      )}
    </div>
  );
}
