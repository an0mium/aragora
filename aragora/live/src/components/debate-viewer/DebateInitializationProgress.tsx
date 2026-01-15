'use client';

import type { StreamEvent } from '@/types/events';

interface Props {
  task: string;
  agents: string[];
  streamEvents: StreamEvent[];
}

/**
 * Shows initialization progress while waiting for agent messages.
 * Displays task, agents, and phase progress from streamEvents.
 */
export function DebateInitializationProgress({ task, agents, streamEvents }: Props) {
  // Get latest phase_progress event for status message
  const latestProgress = streamEvents
    .filter(e => e.type === 'phase_progress')
    .pop();

  const progressMessage = (latestProgress?.data as { message?: string })?.message;
  const message = progressMessage
    || (task ? 'Initializing debate...' : 'Connecting...');

  return (
    <div className="text-center py-8 space-y-4">
      {/* Task/Question */}
      {task && (
        <div className="text-sm text-text-muted font-mono max-w-2xl mx-auto">
          {task}
        </div>
      )}

      {/* Agent badges */}
      {agents.length > 0 && (
        <div className="flex flex-wrap justify-center gap-2">
          {agents.map(agent => (
            <span
              key={agent}
              className="px-2 py-1 bg-surface/50 border border-acid-green/20 rounded text-xs font-mono"
            >
              {agent}
            </span>
          ))}
        </div>
      )}

      {/* Progress message */}
      <div className="text-acid-cyan text-sm font-mono animate-pulse">
        {message}
      </div>

      {/* Progress bar */}
      <div className="w-48 h-1 bg-surface/30 rounded mx-auto overflow-hidden">
        <div
          className="h-full bg-acid-green/50 animate-pulse transition-all duration-1000"
          style={{ width: task ? '33%' : '10%' }}
        />
      </div>
    </div>
  );
}
