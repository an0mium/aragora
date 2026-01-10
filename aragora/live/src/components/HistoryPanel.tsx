'use client';

import { useSupabaseHistory } from '@/hooks/useSupabaseHistory';
import { useLocalHistory } from '@/hooks/useLocalHistory';
import { PanelHeader, StatsGrid, RefreshButton } from './shared';

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

function formatDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatLoopId(loopId: string): string {
  // nomic-20260102-091500 -> Jan 2, 09:15
  const match = loopId.match(/nomic-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/);
  if (match) {
    const [, year, month, day, hour, minute] = match;
    const date = new Date(
      parseInt(year),
      parseInt(month) - 1,
      parseInt(day),
      parseInt(hour),
      parseInt(minute)
    );
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }
  return loopId;
}

export function HistoryPanel() {
  const supabaseHistory = useSupabaseHistory();
  const localHistory = useLocalHistory(DEFAULT_API_BASE);

  // Use Supabase if configured, otherwise fall back to local API
  const useSupabase = supabaseHistory.isConfigured;

  const isLoading = useSupabase ? supabaseHistory.isLoading : localHistory.isLoading;
  const error = useSupabase ? supabaseHistory.error : localHistory.error;
  const cycles = useSupabase ? supabaseHistory.cycles : localHistory.cycles;
  const events = useSupabase ? supabaseHistory.events : localHistory.events;
  const debates = useSupabase ? supabaseHistory.debates : localHistory.debates;
  const refresh = useSupabase ? supabaseHistory.refresh : localHistory.refresh;

  // Only for Supabase mode
  const { recentLoops, selectedLoopId, selectLoop } = supabaseHistory;

  const stats = [
    { value: cycles.length, label: 'Cycles', color: 'text-acid-cyan' },
    { value: events.length, label: 'Events', color: 'text-acid-green' },
    { value: debates.length, label: 'Debates', color: 'text-purple' },
  ];

  return (
    <div className="panel">
      <PanelHeader title="History" loading={isLoading} onRefresh={refresh} />

      {error && (
        <div className="mb-4 p-2 bg-crimson/10 border border-crimson text-crimson text-xs font-mono">
          {error}
        </div>
      )}

      {/* Loop selector - only for Supabase mode */}
      {useSupabase && (
        <div className="mb-4">
          <label className="text-xs text-text-muted block mb-1 font-mono">SELECT_LOOP</label>
          <select
            value={selectedLoopId || ''}
            onChange={(e) => selectLoop(e.target.value)}
            className="w-full bg-bg border border-border px-2 py-1 text-sm font-mono text-text focus:border-acid-green focus:outline-none"
          >
            {recentLoops.length === 0 && (
              <option value="">No loops found</option>
            )}
            {recentLoops.map((loopId) => (
              <option key={loopId} value={loopId}>
                {formatLoopId(loopId)}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Local API summary */}
      {!useSupabase && localHistory.summary && (
        <div className="mb-4 p-2 bg-bg border border-border text-xs font-mono text-text-muted">
          <span className="text-acid-green">&gt;</span> Using local API
          {localHistory.summary.recent_loop_id && (
            <span className="ml-2 text-text">• {formatLoopId(localHistory.summary.recent_loop_id)}</span>
          )}
        </div>
      )}

      {/* Stats */}
      {(useSupabase ? selectedLoopId : true) && (
        <StatsGrid stats={stats} columns={3} className="mb-4" />
      )}

      {/* Cycles list */}
      {cycles.length > 0 && (
        <div className="mb-4">
          <h4 className="text-xs font-mono text-text-muted mb-2">PHASES</h4>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {cycles.map((cycle) => (
              <div
                key={cycle.id}
                className="flex items-center justify-between text-xs font-mono bg-bg border border-border px-2 py-1"
              >
                <span className="text-text">
                  C{cycle.cycle_number}: {cycle.phase}
                </span>
                <span
                  className={
                    cycle.success === true
                      ? 'text-acid-green'
                      : cycle.success === false
                      ? 'text-crimson'
                      : 'text-warning'
                  }
                >
                  {cycle.success === true
                    ? '[OK]'
                    : cycle.success === false
                    ? '[FAIL]'
                    : '[...]'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent events */}
      {events.length > 0 && (
        <div>
          <h4 className="text-xs font-mono text-text-muted mb-2">
            EVENTS ({events.length})
          </h4>
          <div className="space-y-1 max-h-40 overflow-y-auto text-xs font-mono">
            {events.slice(-100).map((event) => (
              <div
                key={event.id}
                className="text-text-muted truncate"
                title={JSON.stringify(event.event_data)}
              >
                <span className="text-text-muted opacity-50">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>{' '}
                <span className="text-acid-cyan">{event.event_type}</span>
                {event.agent && (
                  <span className="text-purple"> [{event.agent}]</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Debates preview */}
      {debates.length > 0 && (
        <div className="mt-4">
          <h4 className="text-xs font-mono text-text-muted mb-2">DEBATES</h4>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {debates.map((debate) => (
              <div
                key={debate.id}
                className="bg-bg border border-border p-2 text-xs font-mono"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-text">
                    {debate.phase} (C{debate.cycle_number})
                  </span>
                  <span
                    className={
                      debate.consensus_reached
                        ? 'text-acid-green'
                        : 'text-warning'
                    }
                  >
                    {debate.consensus_reached
                      ? `[${(debate.confidence * 100).toFixed(0)}%]`
                      : '[NO_CONSENSUS]'}
                  </span>
                </div>
                <div className="text-text-muted truncate" title={debate.task}>
                  {debate.task}
                </div>
                <div className="text-text-muted opacity-50 mt-1">
                  agents: {debate.agents.join(', ')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
