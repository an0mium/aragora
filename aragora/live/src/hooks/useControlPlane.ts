'use client';

/**
 * Control Plane hook for managing agents and tasks.
 *
 * Provides:
 * - Agent listing and status tracking
 * - Task management and filtering
 * - System health monitoring
 * - Real-time updates via WebSocket integration
 */

import { useCallback, useEffect, useRef } from 'react';
import { useAragoraClient } from './useAragoraClient';
import {
  useControlPlaneStore,
  selectAgents,
  selectAgentsLoading,
  selectAgentsError,
  selectTasks,
  selectTasksLoading,
  selectTasksError,
  selectHealth,
  selectHealthLoading,
  selectStats,
  selectStatsLoading,
  selectIsConnected,
  selectFilteredTasks,
  selectAgentCount,
  selectTaskCount,
  selectIsHealthy,
  type ControlPlaneAgent,
  type ControlPlaneTask,
} from '@/store/controlPlaneStore';
import { useControlPlaneWebSocket } from './useControlPlaneWebSocket';

export interface UseControlPlaneOptions {
  /** Whether to enable real-time WebSocket updates */
  enableRealtime?: boolean;
  /** Auto-refresh interval in ms (0 to disable) */
  refreshInterval?: number;
  /** Whether to load data on mount */
  loadOnMount?: boolean;
}

export interface UseControlPlaneReturn {
  // Data
  agents: ControlPlaneAgent[];
  tasks: ControlPlaneTask[];
  filteredTasks: ControlPlaneTask[];
  health: ReturnType<typeof selectHealth>;
  stats: ReturnType<typeof selectStats>;

  // Counts
  agentCount: ReturnType<typeof selectAgentCount>;
  taskCount: ReturnType<typeof selectTaskCount>;

  // Status
  isHealthy: boolean;
  isConnected: boolean;
  agentsLoading: boolean;
  tasksLoading: boolean;
  healthLoading: boolean;
  statsLoading: boolean;
  agentsError: string | null;
  tasksError: string | null;

  // Actions
  loadAgents: () => Promise<void>;
  loadTasks: () => Promise<void>;
  loadHealth: () => Promise<void>;
  loadStats: () => Promise<void>;
  loadAll: () => Promise<void>;

  createTask: (data: { name: string; agent_id?: string }) => Promise<ControlPlaneTask>;
  cancelTask: (taskId: string) => Promise<void>;

  setTaskFilters: (filters: {
    status?: ControlPlaneTask['status'];
    agentId?: string;
  }) => void;

  selectAgent: (agentId: string | null) => void;
  selectTask: (taskId: string | null) => void;
}

/**
 * Hook for interacting with the Control Plane.
 *
 * @example
 * ```tsx
 * const {
 *   agents,
 *   tasks,
 *   isHealthy,
 *   loadAll,
 *   createTask,
 *   cancelTask,
 * } = useControlPlane({ enableRealtime: true });
 *
 * // Create a new task
 * const task = await createTask({ name: 'Document Analysis' });
 * ```
 */
export function useControlPlane({
  enableRealtime = true,
  refreshInterval = 0,
  loadOnMount = true,
}: UseControlPlaneOptions = {}): UseControlPlaneReturn {
  const client = useAragoraClient();
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Store selectors
  const agents = useControlPlaneStore(selectAgents);
  const agentsLoading = useControlPlaneStore(selectAgentsLoading);
  const agentsError = useControlPlaneStore(selectAgentsError);
  const tasks = useControlPlaneStore(selectTasks);
  const tasksLoading = useControlPlaneStore(selectTasksLoading);
  const tasksError = useControlPlaneStore(selectTasksError);
  const health = useControlPlaneStore(selectHealth);
  const healthLoading = useControlPlaneStore(selectHealthLoading);
  const stats = useControlPlaneStore(selectStats);
  const statsLoading = useControlPlaneStore(selectStatsLoading);
  const isConnected = useControlPlaneStore(selectIsConnected);
  const filteredTasks = useControlPlaneStore(selectFilteredTasks);
  const agentCount = useControlPlaneStore(selectAgentCount);
  const taskCount = useControlPlaneStore(selectTaskCount);
  const isHealthy = useControlPlaneStore(selectIsHealthy);

  // Store actions
  const setAgents = useControlPlaneStore((s) => s.setAgents);
  const setAgentsLoading = useControlPlaneStore((s) => s.setAgentsLoading);
  const setAgentsError = useControlPlaneStore((s) => s.setAgentsError);
  const updateAgent = useControlPlaneStore((s) => s.updateAgent);
  const setTasks = useControlPlaneStore((s) => s.setTasks);
  const setTasksLoading = useControlPlaneStore((s) => s.setTasksLoading);
  const setTasksError = useControlPlaneStore((s) => s.setTasksError);
  const updateTask = useControlPlaneStore((s) => s.updateTask);
  const setTaskFiltersAction = useControlPlaneStore((s) => s.setTaskFilters);
  const setHealth = useControlPlaneStore((s) => s.setHealth);
  const setHealthLoading = useControlPlaneStore((s) => s.setHealthLoading);
  const setStats = useControlPlaneStore((s) => s.setStats);
  const setStatsLoading = useControlPlaneStore((s) => s.setStatsLoading);
  const setIsConnected = useControlPlaneStore((s) => s.setIsConnected);
  const setLastUpdate = useControlPlaneStore((s) => s.setLastUpdate);
  const setSelectedAgentId = useControlPlaneStore((s) => s.setSelectedAgentId);
  const setSelectedTaskId = useControlPlaneStore((s) => s.setSelectedTaskId);

  // Real-time updates via WebSocket
  const { isConnected: wsConnected } = useControlPlaneWebSocket({
    enabled: enableRealtime,
    onAgentStatusChange: (agent) => {
      // Map WebSocket agent format to store format
      updateAgent({
        agent_id: agent.id,
        name: agent.name,
        provider: '',
        model: agent.model,
        capabilities: [],
        status: agent.status === 'idle' ? 'available' : agent.status === 'working' ? 'busy' : agent.status === 'rate_limited' ? 'draining' : 'offline',
      });
      setLastUpdate(Date.now());
    },
    onJobProgress: (job) => {
      // Map job to task format
      updateTask({
        id: job.id,
        name: job.name,
        status: job.status === 'queued' ? 'pending' : job.status as ControlPlaneTask['status'],
        assigned_agent: job.agents_assigned[0],
        created_at: job.started_at || new Date().toISOString(),
        started_at: job.started_at,
        completed_at: job.completed_at,
        error: job.error_message,
      });
      setLastUpdate(Date.now());
    },
    onMetricsUpdate: (metrics) => {
      setStats({
        total_agents: metrics.agents_available + metrics.agents_busy + metrics.agents_error,
        available_agents: metrics.agents_available,
        busy_agents: metrics.agents_busy,
        offline_agents: metrics.agents_error,
        pending_tasks: metrics.queued_jobs,
        running_tasks: metrics.active_jobs,
        completed_tasks_24h: metrics.audits_completed_today,
        failed_tasks_24h: 0,
        queue_size: metrics.queued_jobs,
      });
    },
  });

  // Update connection status
  useEffect(() => {
    setIsConnected(wsConnected);
  }, [wsConnected, setIsConnected]);

  // Load agents from SDK
  const loadAgents = useCallback(async () => {
    if (!client) return;
    setAgentsLoading(true);
    try {
      const response = await client.controlPlane.listAgents();
      setAgents(response.agents);
    } catch (err) {
      setAgentsError(err instanceof Error ? err.message : 'Failed to load agents');
    }
  }, [client, setAgents, setAgentsLoading, setAgentsError]);

  // Load tasks from SDK
  const loadTasks = useCallback(async () => {
    if (!client) return;
    setTasksLoading(true);
    try {
      const response = await client.controlPlane.listTasks();
      setTasks(response.tasks);
    } catch (err) {
      setTasksError(err instanceof Error ? err.message : 'Failed to load tasks');
    }
  }, [client, setTasks, setTasksLoading, setTasksError]);

  // Load health from SDK
  const loadHealth = useCallback(async () => {
    if (!client) return;
    setHealthLoading(true);
    try {
      const response = await client.controlPlane.health();
      setHealth(response.health);
    } catch (err) {
      setHealth(null);
    }
  }, [client, setHealth, setHealthLoading]);

  // Load stats - derived from health for now
  const loadStats = useCallback(async () => {
    if (!client) return;
    setStatsLoading(true);
    try {
      const response = await client.controlPlane.health();
      const health = response.health;
      const totalAgents = health.agents_total ?? Object.keys(health.agents).length;
      const availableAgents = health.agents_available ?? Object.values(health.agents).filter(a => a.status === 'healthy').length;
      // Map health to stats format
      setStats({
        total_agents: totalAgents,
        available_agents: availableAgents,
        busy_agents: totalAgents - availableAgents,
        offline_agents: 0,
        pending_tasks: 0,
        running_tasks: health.active_tasks ?? 0,
        completed_tasks_24h: 0,
        failed_tasks_24h: 0,
        queue_size: 0,
        uptime_seconds: health.uptime_seconds,
      });
    } catch (err) {
      setStats(null);
    }
  }, [client, setStats, setStatsLoading]);

  // Load all data
  const loadAll = useCallback(async () => {
    await Promise.all([loadAgents(), loadTasks(), loadHealth(), loadStats()]);
  }, [loadAgents, loadTasks, loadHealth, loadStats]);

  // Create task
  const createTask = useCallback(
    async (data: { name: string; agent_id?: string }): Promise<ControlPlaneTask> => {
      if (!client) throw new Error('Client not available');
      const response = await client.controlPlane.createTask(data);
      updateTask(response.task);
      return response.task;
    },
    [client, updateTask]
  );

  // Cancel task
  const cancelTask = useCallback(
    async (taskId: string): Promise<void> => {
      if (!client) throw new Error('Client not available');
      await client.controlPlane.cancelTask(taskId);
      // Refresh tasks after cancel
      loadTasks();
    },
    [client, loadTasks]
  );

  // Set task filters
  const setTaskFilters = useCallback(
    (filters: { status?: ControlPlaneTask['status']; agentId?: string }) => {
      setTaskFiltersAction(filters);
    },
    [setTaskFiltersAction]
  );

  // Selection actions
  const selectAgent = useCallback(
    (agentId: string | null) => {
      setSelectedAgentId(agentId);
    },
    [setSelectedAgentId]
  );

  const selectTask = useCallback(
    (taskId: string | null) => {
      setSelectedTaskId(taskId);
    },
    [setSelectedTaskId]
  );

  // Load on mount
  useEffect(() => {
    if (loadOnMount && client) {
      loadAll();
    }
  }, [loadOnMount, client, loadAll]);

  // Auto-refresh
  useEffect(() => {
    if (refreshInterval > 0) {
      intervalRef.current = setInterval(() => {
        loadAll();
      }, refreshInterval);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [refreshInterval, loadAll]);

  return {
    // Data
    agents,
    tasks,
    filteredTasks,
    health,
    stats,

    // Counts
    agentCount,
    taskCount,

    // Status
    isHealthy,
    isConnected,
    agentsLoading,
    tasksLoading,
    healthLoading,
    statsLoading,
    agentsError,
    tasksError,

    // Actions
    loadAgents,
    loadTasks,
    loadHealth,
    loadStats,
    loadAll,
    createTask,
    cancelTask,
    setTaskFilters,
    selectAgent,
    selectTask,
  };
}

export default useControlPlane;
