'use client';

import { useSupabaseHistory } from '@/hooks/useSupabaseHistory';

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
  const {
    isConfigured,
    isLoading,
    error,
    recentLoops,
    selectedLoopId,
    cycles,
    events,
    debates,
    selectLoop,
    refresh,
  } = useSupabaseHistory();

  if (!isConfigured) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-semibold mb-2 text-gray-300">History</h3>
        <p className="text-gray-500 text-sm">
          Supabase not configured. Set NEXT_PUBLIC_SUPABASE_URL and
          NEXT_PUBLIC_SUPABASE_ANON_KEY environment variables.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-300">History</h3>
        <button
          onClick={refresh}
          disabled={isLoading}
          className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50"
        >
          {isLoading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-2 bg-red-900/50 border border-red-700 rounded text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Loop selector */}
      <div className="mb-4">
        <label className="text-xs text-gray-500 block mb-1">Select Loop</label>
        <select
          value={selectedLoopId || ''}
          onChange={(e) => selectLoop(e.target.value)}
          className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
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

      {/* Stats */}
      {selectedLoopId && (
        <div className="grid grid-cols-3 gap-2 mb-4">
          <div className="bg-gray-700/50 rounded p-2 text-center">
            <div className="text-xl font-bold text-blue-400">{cycles.length}</div>
            <div className="text-xs text-gray-500">Cycles</div>
          </div>
          <div className="bg-gray-700/50 rounded p-2 text-center">
            <div className="text-xl font-bold text-green-400">{events.length}</div>
            <div className="text-xs text-gray-500">Events</div>
          </div>
          <div className="bg-gray-700/50 rounded p-2 text-center">
            <div className="text-xl font-bold text-purple-400">{debates.length}</div>
            <div className="text-xs text-gray-500">Debates</div>
          </div>
        </div>
      )}

      {/* Cycles list */}
      {cycles.length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Phases</h4>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {cycles.map((cycle) => (
              <div
                key={cycle.id}
                className="flex items-center justify-between text-xs bg-gray-700/30 rounded px-2 py-1"
              >
                <span className="text-gray-300">
                  C{cycle.cycle_number}: {cycle.phase}
                </span>
                <span
                  className={
                    cycle.success === true
                      ? 'text-green-400'
                      : cycle.success === false
                      ? 'text-red-400'
                      : 'text-yellow-400'
                  }
                >
                  {cycle.success === true
                    ? '✓'
                    : cycle.success === false
                    ? '✗'
                    : '...'}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent events */}
      {events.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-400 mb-2">
            Recent Events ({events.length})
          </h4>
          <div className="space-y-1 max-h-40 overflow-y-auto text-xs font-mono">
            {events.slice(-100).map((event) => (
              <div
                key={event.id}
                className="text-gray-400 truncate"
                title={JSON.stringify(event.event_data)}
              >
                <span className="text-gray-600">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>{' '}
                <span className="text-blue-400">{event.event_type}</span>
                {event.agent && (
                  <span className="text-purple-400"> [{event.agent}]</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Debates preview */}
      {debates.length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Debates</h4>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {debates.map((debate) => (
              <div
                key={debate.id}
                className="bg-gray-700/30 rounded p-2 text-xs"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-gray-300 font-medium">
                    {debate.phase} (C{debate.cycle_number})
                  </span>
                  <span
                    className={
                      debate.consensus_reached
                        ? 'text-green-400'
                        : 'text-yellow-400'
                    }
                  >
                    {debate.consensus_reached
                      ? `✓ ${(debate.confidence * 100).toFixed(0)}%`
                      : 'No consensus'}
                  </span>
                </div>
                <div className="text-gray-500" title={debate.task}>
                  {debate.task}
                </div>
                <div className="text-gray-600 mt-1">
                  Agents: {debate.agents.join(', ')}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
