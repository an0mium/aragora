'use client';

import { useState, useEffect, useRef } from 'react';
import type { StreamEvent } from '@/types/events';

interface AgentPanelProps {
  events: StreamEvent[];
}

// Agent color schemes by model family
const MODEL_COLORS = {
  // Gemini - purple (Google AI)
  gemini: { bg: 'bg-purple/10', text: 'text-purple', border: 'border-purple/30' },
  // Codex - gold (OpenAI)
  codex: { bg: 'bg-gold/10', text: 'text-gold', border: 'border-gold/30' },
  // Claude - accent/indigo (Anthropic)
  claude: { bg: 'bg-accent/10', text: 'text-accent', border: 'border-accent/30' },
  // Grok - crimson red (xAI)
  grok: { bg: 'bg-crimson/10', text: 'text-crimson', border: 'border-crimson/30' },
  // Default - neutral
  default: { bg: 'bg-surface', text: 'text-text', border: 'border-border' },
};

/**
 * Get colors for an agent by name, using prefix matching.
 * This ensures any variant (e.g., "grok-explorer", "grok-pragmatist") gets the right color.
 */
function getAgentColors(agentName: string): { bg: string; text: string; border: string } {
  const name = agentName.toLowerCase();
  // Match by prefix (order matters - check specific models first)
  if (name.startsWith('gemini')) return MODEL_COLORS.gemini;
  if (name.startsWith('codex')) return MODEL_COLORS.codex;
  if (name.startsWith('claude')) return MODEL_COLORS.claude;
  if (name.startsWith('grok')) return MODEL_COLORS.grok;
  return MODEL_COLORS.default;
}

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

  // Collect all agent_message content for deduplication
  const agentMessageContents = new Set(
    events
      .filter((e) => e.type === 'agent_message')
      .map((e) => {
        const content = (e.data?.content as string) || '';
        // Normalize: first 100 chars, lowercase, trimmed
        return content.slice(0, 100).toLowerCase().trim();
      })
  );

  // Filter to agent-related events, preferring agent_message over log_message
  // to avoid duplicated content (log_message often echoes agent_message)
  const agentEvents = events.filter((e) => {
    // Always include these primary event types
    if (
      e.type === 'agent_message' ||
      e.type === 'critique' ||
      e.type === 'consensus' ||
      e.type === 'vote'
    ) {
      return true;
    }
    // For log_message, filter out duplicates more aggressively
    if (e.type === 'log_message') {
      const msg = (e.data?.message as string) || '';

      // Skip log messages that look like arena message summaries (they have agent_message equivalents)
      // These have format: "    [role] agent (round N): content"
      if (msg.match(/^\s*\[(proposer|critic|synthesizer|judge|reviewer|implementer)\]/i)) {
        return false;
      }
      // Skip vote summaries (have vote events)
      if (msg.match(/^\s*\[vote\]/i)) {
        return false;
      }
      // Skip critique summaries (have critique events)
      if (msg.match(/^\s*\[critique\]/i)) {
        return false;
      }
      // Skip consensus summaries (have consensus events)
      if (msg.match(/^\s*\[consensus\]/i)) {
        return false;
      }
      // Skip round markers (redundant with agent_message)
      if (msg.match(/^\s*Round \d+:/i)) {
        return false;
      }
      // Skip agent-attributed log messages that duplicate agent_message content
      // Check if content matches any agent_message (first 100 chars)
      const normalizedMsg = msg.slice(0, 100).toLowerCase().trim();
      if (agentMessageContents.has(normalizedMsg)) {
        return false;
      }
      // Skip if log message starts with agent name (likely duplicate)
      if (msg.match(/^\s*(gemini|claude|codex|grok)[-\w]*\s*:/i)) {
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
  const colors = getAgentColors(agentName);
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
      // Show full content in preview (no truncation for better visibility)
      preview = content;
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
      // Show full content in preview (no truncation for better visibility)
      preview = content;
      // Use agent-specific icons for attributed log messages
      icon = agentName !== 'system' ? ROLE_ICONS.default : '📝';
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
          <p className="text-sm text-text-muted whitespace-pre-wrap break-words">{preview}</p>
        </div>
        <span className="text-text-muted flex-shrink-0">
          {isExpanded ? '▼' : '▶'}
        </span>
      </button>
      {isExpanded && (
        <div className="px-3 pb-3 pt-0">
          <div className="bg-bg/50 rounded-lg p-3 text-sm whitespace-pre-wrap break-words overflow-y-auto max-h-[500px]">
            {content}
          </div>
        </div>
      )}
    </div>
  );
}
