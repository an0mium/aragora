'use client';

/**
 * Control Plane WebSocket hook for real-time agent orchestration updates.
 *
 * Provides:
 * - Agent status updates (idle/working/error/rate_limited)
 * - Job progress tracking
 * - Queue updates
 * - Finding detection events
 * - System metrics streaming
 */

import { useState, useCallback, useMemo } from 'react';
import { useWebSocketBase, WebSocketConnectionStatus } from './useWebSocketBase';
import { useBackend } from '@/components/BackendSelector';

// Event types from the control plane stream
export type ControlPlaneEventType =
  | 'agent_status'
  | 'job_progress'
  | 'job_started'
  | 'job_completed'
  | 'job_failed'
  | 'queue_update'
  | 'finding_detected'
  | 'metrics_update'
  | 'system_health';

export interface AgentState {
  id: string;
  name: string;
  model: string;
  status: 'idle' | 'working' | 'error' | 'rate_limited';
  current_task?: string;
  current_job_id?: string;
  requests_today: number;
  tokens_used: number;
  last_active?: string;
  error_message?: string;
}

export interface JobState {
  id: string;
  type: 'document_processing' | 'audit' | 'debate' | 'batch_upload';
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'paused' | 'cancelled';
  progress: number;
  phase?: string;
  started_at?: string;
  completed_at?: string;
  document_count?: number;
  documents_processed?: number;
  findings_count?: number;
  agents_assigned: string[];
  error_message?: string;
}

export interface SystemMetrics {
  active_jobs: number;
  queued_jobs: number;
  agents_available: number;
  agents_busy: number;
  agents_error: number;
  documents_processed_today: number;
  audits_completed_today: number;
  tokens_used_today: number;
  findings_today: number;
  cpu_usage?: number;
  memory_usage?: number;
}

export interface FindingEvent {
  id: string;
  session_id: string;
  document_id: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  category: string;
  title: string;
  found_by: string;
  timestamp: string;
}

export interface ControlPlaneEvent {
  type: ControlPlaneEventType;
  timestamp: string;
  seq?: number;
  data: AgentState | JobState | SystemMetrics | FindingEvent | AgentState[] | JobState[];
}

export interface UseControlPlaneWebSocketOptions {
  /** Whether the connection is enabled */
  enabled?: boolean;
  /** Whether to automatically reconnect on disconnection */
  autoReconnect?: boolean;
  /** Callback when agent status changes */
  onAgentStatusChange?: (agent: AgentState) => void;
  /** Callback when job progress updates */
  onJobProgress?: (job: JobState) => void;
  /** Callback when a finding is detected */
  onFindingDetected?: (finding: FindingEvent) => void;
  /** Callback when metrics update */
  onMetricsUpdate?: (metrics: SystemMetrics) => void;
}

export interface UseControlPlaneWebSocketReturn {
  /** Current WebSocket connection status */
  status: WebSocketConnectionStatus;
  /** Whether connected to the control plane stream */
  isConnected: boolean;
  /** Connection error message if any */
  error: string | null;
  /** Current reconnection attempt */
  reconnectAttempt: number;
  /** Current state of all agents */
  agents: AgentState[];
  /** Current state of all jobs */
  jobs: JobState[];
  /** Current system metrics */
  metrics: SystemMetrics | null;
  /** Recent findings (last 50) */
  recentFindings: FindingEvent[];
  /** Manually reconnect */
  reconnect: () => void;
  /** Manually disconnect */
  disconnect: () => void;
  /** Request a specific job's status */
  requestJobStatus: (jobId: string) => void;
  /** Request agent refresh */
  requestAgentRefresh: () => void;
}

const DEFAULT_METRICS: SystemMetrics = {
  active_jobs: 0,
  queued_jobs: 0,
  agents_available: 0,
  agents_busy: 0,
  agents_error: 0,
  documents_processed_today: 0,
  audits_completed_today: 0,
  tokens_used_today: 0,
  findings_today: 0,
};

/**
 * Hook for connecting to the Control Plane WebSocket stream.
 *
 * @example
 * ```tsx
 * const {
 *   status,
 *   isConnected,
 *   agents,
 *   jobs,
 *   metrics,
 *   recentFindings,
 * } = useControlPlaneWebSocket({
 *   enabled: true,
 *   onFindingDetected: (finding) => {
 *     toast(`New finding: ${finding.title}`);
 *   },
 * });
 * ```
 */
export function useControlPlaneWebSocket({
  enabled = true,
  autoReconnect = true,
  onAgentStatusChange,
  onJobProgress,
  onFindingDetected,
  onMetricsUpdate,
}: UseControlPlaneWebSocketOptions = {}): UseControlPlaneWebSocketReturn {
  const { config: backendConfig } = useBackend();

  // State for agents, jobs, metrics, findings
  const [agents, setAgents] = useState<AgentState[]>([]);
  const [jobs, setJobs] = useState<JobState[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [recentFindings, setRecentFindings] = useState<FindingEvent[]>([]);

  // Build WebSocket URL
  const wsUrl = useMemo(() => {
    if (!backendConfig?.api) return '';
    const baseUrl = backendConfig.api.replace(/^http/, 'ws');
    return `${baseUrl}/api/control-plane/stream`;
  }, [backendConfig?.api]);

  // Handle incoming events
  const handleEvent = useCallback(
    (event: ControlPlaneEvent) => {
      switch (event.type) {
        case 'agent_status': {
          const agent = event.data as AgentState;
          setAgents((prev) => {
            const idx = prev.findIndex((a) => a.id === agent.id);
            if (idx >= 0) {
              const updated = [...prev];
              updated[idx] = agent;
              return updated;
            }
            return [...prev, agent];
          });
          onAgentStatusChange?.(agent);
          break;
        }

        case 'job_progress':
        case 'job_started':
        case 'job_completed':
        case 'job_failed': {
          const job = event.data as JobState;
          setJobs((prev) => {
            const idx = prev.findIndex((j) => j.id === job.id);
            if (idx >= 0) {
              const updated = [...prev];
              updated[idx] = job;
              return updated;
            }
            return [...prev, job];
          });
          onJobProgress?.(job);
          break;
        }

        case 'queue_update': {
          // Full queue state update
          const jobList = event.data as JobState[];
          if (Array.isArray(jobList)) {
            setJobs(jobList);
          }
          break;
        }

        case 'finding_detected': {
          const finding = event.data as FindingEvent;
          setRecentFindings((prev) => {
            const updated = [finding, ...prev].slice(0, 50); // Keep last 50
            return updated;
          });
          onFindingDetected?.(finding);
          break;
        }

        case 'metrics_update':
        case 'system_health': {
          const newMetrics = event.data as SystemMetrics;
          setMetrics(newMetrics);
          onMetricsUpdate?.(newMetrics);
          break;
        }

        default:
          // Unknown event type - ignore
          break;
      }
    },
    [onAgentStatusChange, onJobProgress, onFindingDetected, onMetricsUpdate]
  );

  // Use base WebSocket hook
  const { status, error, isConnected, reconnectAttempt, send, reconnect, disconnect } =
    useWebSocketBase<ControlPlaneEvent>({
      wsUrl,
      enabled: enabled && !!wsUrl,
      autoReconnect,
      subscribeMessage: { type: 'subscribe', channels: ['agents', 'jobs', 'metrics', 'findings'] },
      onEvent: handleEvent,
      onConnect: () => {
        // Request initial state on connect
        send({ type: 'get_state' });
      },
      logPrefix: '[ControlPlane]',
    });

  // Request specific job status
  const requestJobStatus = useCallback(
    (jobId: string) => {
      send({ type: 'get_job', job_id: jobId });
    },
    [send]
  );

  // Request agent refresh
  const requestAgentRefresh = useCallback(() => {
    send({ type: 'get_agents' });
  }, [send]);

  return {
    status,
    isConnected,
    error,
    reconnectAttempt,
    agents,
    jobs,
    metrics: metrics || DEFAULT_METRICS,
    recentFindings,
    reconnect,
    disconnect,
    requestJobStatus,
    requestAgentRefresh,
  };
}

export default useControlPlaneWebSocket;
