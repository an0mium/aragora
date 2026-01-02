'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import type { StreamEvent, NomicState, LoopInstance } from '@/types/events';

const DEFAULT_WS_URL = 'ws://localhost:8765';
const RECONNECT_INTERVAL = 3000;
const MAX_EVENTS = 1000;

export function useNomicStream(wsUrl: string = DEFAULT_WS_URL) {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [nomicState, setNomicState] = useState<NomicState | null>(null);
  const [activeLoops, setActiveLoops] = useState<LoopInstance[]>([]);
  const [selectedLoopId, setSelectedLoopId] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        wsRef.current = null;

        // Reconnect after delay
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, RECONNECT_INTERVAL);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as StreamEvent;

          // Handle sync event (initial state)
          if (data.type === 'sync') {
            setNomicState(data.data as NomicState);
            return;
          }

          // Handle loop list event (sent on connect and on request)
          if (data.type === 'loop_list') {
            const loopData = data.data as { loops: LoopInstance[]; count: number };
            setActiveLoops(loopData.loops || []);
            // Auto-select first loop if none selected
            if (!selectedLoopId && loopData.loops?.length > 0) {
              setSelectedLoopId(loopData.loops[0].loop_id);
            }
            return;
          }

          // Handle loop registration
          if (data.type === 'loop_register') {
            const newLoop: LoopInstance = {
              loop_id: data.data.loop_id as string,
              name: data.data.name as string,
              started_at: data.data.started_at as number,
              cycle: 0,
              phase: 'starting',
              path: data.data.path as string,
            };
            setActiveLoops((prev) => [...prev, newLoop]);
            // Auto-select if this is the first loop
            if (!selectedLoopId) {
              setSelectedLoopId(newLoop.loop_id);
            }
            return;
          }

          // Handle loop unregistration
          if (data.type === 'loop_unregister') {
            const loopId = data.data.loop_id as string;
            setActiveLoops((prev) => prev.filter((l) => l.loop_id !== loopId));
            // If this was the selected loop, switch to another
            if (selectedLoopId === loopId) {
              setActiveLoops((prev) => {
                if (prev.length > 0) {
                  setSelectedLoopId(prev[0].loop_id);
                } else {
                  setSelectedLoopId(null);
                }
                return prev;
              });
            }
            return;
          }

          // Add event to list, keeping only last MAX_EVENTS
          setEvents((prev) => {
            const newEvents = [...prev, data];
            if (newEvents.length > MAX_EVENTS) {
              return newEvents.slice(-MAX_EVENTS);
            }
            return newEvents;
          });

          // Update state based on event type
          updateStateFromEvent(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
      // Retry connection
      reconnectTimeoutRef.current = setTimeout(connect, RECONNECT_INTERVAL);
    }
  }, [wsUrl]);

  const updateStateFromEvent = useCallback((event: StreamEvent) => {
    switch (event.type) {
      case 'cycle_start':
        setNomicState((prev) => ({
          ...prev,
          phase: 'starting',
          cycle: event.data.cycle as number,
        }));
        break;
      case 'phase_start':
        setNomicState((prev) => ({
          ...prev,
          phase: event.data.phase as string,
          stage: 'running',
        }));
        break;
      case 'phase_end':
        setNomicState((prev) => ({
          ...prev,
          stage: event.data.success ? 'complete' : 'failed',
        }));
        break;
      case 'task_complete':
        setNomicState((prev) => ({
          ...prev,
          completed_tasks: (prev?.completed_tasks || 0) + 1,
          last_task: event.data.task_id as string,
          last_success: event.data.success as boolean,
        }));
        break;
      case 'cycle_end':
        setNomicState((prev) => ({
          ...prev,
          phase: 'complete',
          stage: event.data.outcome as string,
        }));
        break;
    }
  }, []);

  useEffect(() => {
    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect]);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  const selectLoop = useCallback((loopId: string) => {
    setSelectedLoopId(loopId);
    // Clear events when switching loops (optional - remove if you want to keep history)
    setEvents([]);
  }, []);

  const requestLoopList = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'get_loops' }));
    }
  }, []);

  return {
    events,
    connected,
    nomicState,
    clearEvents,
    // Multi-loop support
    activeLoops,
    selectedLoopId,
    selectLoop,
    requestLoopList,
  };
}

// Fetch nomic state from REST API
export async function fetchNomicState(apiUrl: string = 'http://localhost:8080'): Promise<NomicState> {
  const response = await fetch(`${apiUrl}/api/nomic/state`);
  if (!response.ok) {
    throw new Error(`Failed to fetch state: ${response.status}`);
  }
  return response.json();
}

// Fetch nomic log lines from REST API
export async function fetchNomicLog(
  apiUrl: string = 'http://localhost:8080',
  lines: number = 100
): Promise<string[]> {
  const response = await fetch(`${apiUrl}/api/nomic/log?lines=${lines}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch log: ${response.status}`);
  }
  const data = await response.json();
  return data.lines || [];
}
