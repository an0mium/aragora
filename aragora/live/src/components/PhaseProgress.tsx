'use client';

import { useState, useEffect, useCallback } from 'react';
import type { StreamEvent } from '@/types/events';
import { logger } from '@/utils/logger';
import { API_BASE_URL } from '@/config';

interface NomicState {
  phase: string;
  stage: string;
  cycle: number;
  saved_at: string;
  loop_id?: string;
  loop_name?: string;
}

interface PhaseProgressProps {
  events: StreamEvent[];
  currentPhase: string;
  apiBase?: string;
}

const DEFAULT_API_BASE = API_BASE_URL;
const PHASES = ['debate', 'design', 'implement', 'verify', 'commit'];

export function PhaseProgress({ events, currentPhase, apiBase = DEFAULT_API_BASE }: PhaseProgressProps) {
  const [nomicState, setNomicState] = useState<NomicState | null>(null);

  const fetchNomicState = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/api/nomic/state`);
      if (response.ok) {
        const data = await response.json();
        setNomicState(data);
      }
    } catch (err) {
      logger.error('Failed to fetch nomic state:', err);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchNomicState();
    // Poll every 10 seconds for updates
    const interval = setInterval(fetchNomicState, 10000);
    return () => clearInterval(interval);
  }, [fetchNomicState]);
  type PhaseStatusType = 'pending' | 'active' | 'complete' | 'failed';

  // Get phase statuses from events
  const phaseStatuses: { phase: string; status: PhaseStatusType }[] = PHASES.map((phase) => {
    const startEvent = events.find((e) => {
      if (e.type !== 'phase_start') return false;
      const eventData = e.data as Record<string, unknown>;
      return eventData.phase === phase;
    });
    const endEvent = events.find((e) => {
      if (e.type !== 'phase_end') return false;
      const eventData = e.data as Record<string, unknown>;
      return eventData.phase === phase;
    });

    if (endEvent) {
      const endEventData = endEvent.data as Record<string, unknown>;
      return {
        phase,
        status: (endEventData.success ? 'complete' : 'failed') as PhaseStatusType,
      };
    }
    if (startEvent || currentPhase === phase) {
      return { phase, status: 'active' as PhaseStatusType };
    }
    return { phase, status: 'pending' as PhaseStatusType };
  });

  // Use API state if available, otherwise fall back to events
  const effectivePhase = nomicState?.phase || currentPhase;

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-text-muted uppercase tracking-wider">
          Phase Progress
        </h2>
        {nomicState && (
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="text-acid-green">Cycle {nomicState.cycle}</span>
            {nomicState.stage && (
              <span className="text-text-muted">• {nomicState.stage.replace(/_/g, ' ')}</span>
            )}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2">
        {phaseStatuses.map(({ phase, status }, index) => {
          // Override status based on API state
          let effectiveStatus = status;
          if (nomicState) {
            const phaseIndex = PHASES.indexOf(phase);
            const currentIndex = PHASES.indexOf(effectivePhase);
            if (phaseIndex < currentIndex) {
              effectiveStatus = 'complete';
            } else if (phaseIndex === currentIndex) {
              effectiveStatus = 'active';
            } else {
              effectiveStatus = 'pending';
            }
          }
          return (
            <div key={phase} className="flex items-center">
              <PhaseBlock phase={phase} status={effectiveStatus} />
              {index < phaseStatuses.length - 1 && (
                <div className="w-4 h-0.5 bg-border" />
              )}
            </div>
          );
        })}
      </div>
      {nomicState?.saved_at && (
        <div className="mt-2 text-[10px] text-text-muted font-mono">
          Last update: {new Date(nomicState.saved_at).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}

interface PhaseBlockProps {
  phase: string;
  status: 'pending' | 'active' | 'complete' | 'failed';
}

function PhaseBlock({ phase, status }: PhaseBlockProps) {
  const baseClasses = 'px-3 py-2 rounded-lg text-sm font-medium transition-all';

  const statusClasses: Record<string, string> = {
    pending: 'bg-surface border border-border text-text-muted',
    active: 'bg-accent/20 border border-accent text-accent animate-pulse',
    complete: 'bg-success/20 border border-success text-success',
    failed: 'bg-warning/20 border border-warning text-warning',
  };

  return (
    <div className={`${baseClasses} ${statusClasses[status]}`}>
      {phase.charAt(0).toUpperCase() + phase.slice(1)}
    </div>
  );
}
