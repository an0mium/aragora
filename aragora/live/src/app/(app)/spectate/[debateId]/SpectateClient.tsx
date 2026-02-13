'use client';

import { useEffect, useRef, useCallback } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useRightSidebar } from '@/context/RightSidebarContext';
import {
  useSpectateStore,
  EVENT_STYLES,
  selectFilteredEvents,
  type SpectatorEvent,
  type SpectatorEventType,
} from '@/store/spectateStore';

export default function SpectateClient() {
  const params = useParams();
  const debateId = params.debateId as string;

  const eventListRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Store state
  const {
    connectionStatus,
    error,
    task,
    agents,
    currentRound,
    events,
    autoScroll,
    showTimestamps,
    connect,
    setConnectionStatus,
    setError,
    addEvent,
    setTask,
    setAutoScroll,
    setShowTimestamps,
    reset,
  } = useSpectateStore();

  const filteredEvents = useSpectateStore(selectFilteredEvents);

  const { setContext, clearContext } = useRightSidebar();

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    connect(debateId);

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8765';
    const ws = new WebSocket(`${wsUrl}/spectate/${debateId}`);

    ws.onopen = () => {
      setConnectionStatus('connected');
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Handle different message types
        if (data.type === 'metadata') {
          if (data.task) setTask(data.task);
        } else {
          // It's a spectator event
          addEvent({
            type: data.type as SpectatorEventType,
            timestamp: data.timestamp || Date.now() / 1000,
            agent: data.agent || null,
            details: data.details || null,
            metric: data.metric ?? null,
            round: data.round ?? null,
          });
        }
      } catch (err) {
        console.error('Failed to parse spectator event:', err);
      }
    };

    ws.onerror = () => {
      setError('Connection error. The debate may have ended.');
      setConnectionStatus('error');
    };

    ws.onclose = (event) => {
      if (event.code === 1000) {
        // Normal closure - debate ended
        setConnectionStatus('disconnected');
      } else {
        setError('Connection lost. Attempting to reconnect...');
        setConnectionStatus('error');
        // Try to reconnect after 3 seconds
        setTimeout(() => {
          if (wsRef.current?.readyState !== WebSocket.OPEN) {
            connectWebSocket();
          }
        }, 3000);
      }
    };

    wsRef.current = ws;
  }, [debateId, connect, setConnectionStatus, setError, setTask, addEvent]);

  // Initialize connection
  useEffect(() => {
    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Leaving spectate');
        wsRef.current = null;
      }
      reset();
    };
  }, [debateId, connectWebSocket, reset]);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScroll && eventListRef.current) {
      eventListRef.current.scrollTop = eventListRef.current.scrollHeight;
    }
  }, [events, autoScroll]);

  // Set up right sidebar
  useEffect(() => {
    setContext({
      title: 'Spectating',
      subtitle: task ? task.slice(0, 30) + (task.length > 30 ? '...' : '') : 'Loading...',
      statsContent: (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Round</span>
            <span className="text-sm font-mono text-[var(--acid-green)]">{currentRound}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Events</span>
            <span className="text-sm font-mono text-[var(--acid-cyan)]">{events.length}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Agents</span>
            <span className="text-sm font-mono text-[var(--acid-cyan)]">{agents.length}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Status</span>
            <span
              className={`text-sm font-mono ${
                connectionStatus === 'connected'
                  ? 'text-green-400'
                  : connectionStatus === 'error'
                    ? 'text-red-400'
                    : 'text-yellow-400'
              }`}
            >
              {connectionStatus.toUpperCase()}
            </span>
          </div>
        </div>
      ),
      actionsContent: (
        <div className="space-y-2">
          <Link
            href="/spectate"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
          >
            ← BACK TO LIST
          </Link>
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`block w-full px-3 py-2 text-xs font-mono text-center border transition-colors ${
              autoScroll
                ? 'bg-[var(--acid-green)]/10 text-[var(--acid-green)] border-[var(--acid-green)]/30'
                : 'bg-[var(--surface)] text-[var(--text-muted)] border-[var(--border)]'
            }`}
          >
            AUTO-SCROLL {autoScroll ? 'ON' : 'OFF'}
          </button>
          <button
            onClick={() => setShowTimestamps(!showTimestamps)}
            className={`block w-full px-3 py-2 text-xs font-mono text-center border transition-colors ${
              showTimestamps
                ? 'bg-[var(--acid-cyan)]/10 text-[var(--acid-cyan)] border-[var(--acid-cyan)]/30'
                : 'bg-[var(--surface)] text-[var(--text-muted)] border-[var(--border)]'
            }`}
          >
            TIMESTAMPS {showTimestamps ? 'ON' : 'OFF'}
          </button>
        </div>
      ),
    });

    return () => clearContext();
  }, [
    task,
    currentRound,
    events.length,
    agents.length,
    connectionStatus,
    autoScroll,
    showTimestamps,
    setContext,
    clearContext,
    setAutoScroll,
    setShowTimestamps,
  ]);

  // Format timestamp
  const formatTimestamp = (ts: number) => {
    const date = new Date(ts * 1000);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <div className="max-w-6xl mx-auto px-4 py-8">
          {/* Header */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                <Link href="/spectate" className="text-text-muted hover:text-acid-green transition-colors">
                  ← Back
                </Link>
                <div className="flex items-center gap-2">
                  {connectionStatus === 'connected' && (
                    <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
                  )}
                  <h1 className="text-xl font-mono text-acid-green">SPECTATING</h1>
                </div>
              </div>

              {/* Connection Status Badge */}
              <div
                className={`px-3 py-1 text-xs font-mono border ${
                  connectionStatus === 'connected'
                    ? 'bg-green-500/10 text-green-400 border-green-500/30'
                    : connectionStatus === 'connecting'
                      ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30'
                      : 'bg-red-500/10 text-red-400 border-red-500/30'
                }`}
              >
                {connectionStatus.toUpperCase()}
              </div>
            </div>

            {/* Task */}
            {task && <p className="text-text-muted text-sm font-mono">{task}</p>}
          </div>

          {/* Error Banner */}
          {error && (
            <div className="mb-4 border border-warning/30 bg-warning/10 p-3">
              <p className="text-warning text-sm font-mono">{error}</p>
            </div>
          )}

          {/* Agents Bar */}
          {agents.length > 0 && (
            <div className="mb-4 flex flex-wrap gap-2">
              {agents.map((agent) => (
                <span
                  key={agent}
                  className="px-3 py-1 text-xs font-mono bg-acid-cyan/10 text-acid-cyan border border-acid-cyan/30"
                >
                  {agent}
                </span>
              ))}
              <span className="px-3 py-1 text-xs font-mono bg-surface text-text-muted border border-border">
                Round {currentRound}
              </span>
            </div>
          )}

          {/* Event Feed */}
          <div className="border border-acid-green/30 bg-surface/50">
            {/* Feed Header */}
            <div className="flex items-center justify-between px-4 py-2 border-b border-acid-green/20 bg-surface/80">
              <span className="text-xs font-mono text-acid-green">
                EVENT FEED ({filteredEvents.length} events)
              </span>
              <span className="text-xs font-mono text-text-muted">
                {autoScroll ? 'Auto-scrolling' : 'Scroll paused'}
              </span>
            </div>

            {/* Event List */}
            <div
              ref={eventListRef}
              className="h-[500px] overflow-y-auto p-4 space-y-2 font-mono text-sm"
              onScroll={(e) => {
                const target = e.target as HTMLDivElement;
                const atBottom = target.scrollHeight - target.scrollTop <= target.clientHeight + 50;
                if (!atBottom && autoScroll) {
                  setAutoScroll(false);
                }
              }}
            >
              {filteredEvents.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center text-text-muted">
                    <div className="w-8 h-8 border-2 border-acid-green/30 border-t-acid-green rounded-full animate-spin mx-auto mb-4" />
                    <p>Waiting for events...</p>
                  </div>
                </div>
              ) : (
                filteredEvents.map((event, index) => (
                  <EventLine
                    key={`${event.timestamp}-${index}`}
                    event={event}
                    showTimestamp={showTimestamps}
                    formatTimestamp={formatTimestamp}
                  />
                ))
              )}
            </div>
          </div>

          {/* Legend */}
          <div className="mt-4 border border-acid-green/20 bg-surface/30 p-4">
            <h3 className="text-xs font-mono text-acid-cyan mb-3">EVENT TYPES</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2">
              {Object.entries(EVENT_STYLES).map(([type, style]) => (
                <div key={type} className="flex items-center gap-2 text-xs font-mono">
                  <span>{style.icon}</span>
                  <span className={style.color}>{style.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

// Event Line Component
function EventLine({
  event,
  showTimestamp,
  formatTimestamp,
}: {
  event: SpectatorEvent;
  showTimestamp: boolean;
  formatTimestamp: (ts: number) => string;
}) {
  const style = EVENT_STYLES[event.type] || { icon: '•', color: 'text-gray-400', label: 'UNKNOWN' };

  return (
    <div className="flex items-start gap-2 py-1 hover:bg-acid-green/5 px-2 -mx-2 rounded">
      {/* Timestamp */}
      {showTimestamp && (
        <span className="text-text-muted/50 text-xs shrink-0 w-20">[{formatTimestamp(event.timestamp)}]</span>
      )}

      {/* Round */}
      {event.round !== null && <span className="text-acid-cyan/70 text-xs shrink-0">R{event.round}</span>}

      {/* Icon */}
      <span className="shrink-0">{style.icon}</span>

      {/* Agent */}
      {event.agent && <span className="text-acid-green font-bold shrink-0">{event.agent}</span>}

      {/* Details */}
      {event.details && <span className="text-text truncate">{event.details}</span>}

      {/* Metric */}
      {event.metric !== null && (
        <span className="text-text-muted text-xs shrink-0">({event.metric.toFixed(2)})</span>
      )}
    </div>
  );
}
