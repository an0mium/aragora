/**
 * Tests for controlPlaneStore
 *
 * Tests cover:
 * - Agent registry management (setAgents, updateAgent, removeAgent)
 * - Task queue management (setTasks, updateTask, removeTask, filters)
 * - Health and stats tracking
 * - Connection state management
 * - Selection state (selectedAgentId, selectedTaskId)
 * - Derived selectors
 * - Reset functionality
 */

import { act } from '@testing-library/react';
import {
  useControlPlaneStore,
  selectAgents,
  selectTasks,
  selectHealth,
  selectStats,
  selectIsConnected,
  selectAgentById,
  selectTaskById,
  selectAgentsByStatus,
  selectTasksByStatus,
  selectFilteredTasks,
  selectIsHealthy,
  selectAgentCount,
  selectTaskCount,
  ControlPlaneAgent,
  ControlPlaneTask,
  ControlPlaneHealth,
  ControlPlaneStats,
} from '../controlPlaneStore';

// Sample test data
const mockAgent: ControlPlaneAgent = {
  agent_id: 'agent-1',
  name: 'Claude API',
  provider: 'anthropic',
  status: 'available',
  model: 'claude-3',
  capabilities: ['reasoning', 'coding'],
  elo_rating: 1500,
  tasks_completed: 10,
  tasks_failed: 1,
  avg_latency_ms: 250,
  last_heartbeat: '2026-01-25T10:00:00Z',
  registered_at: '2026-01-01T00:00:00Z',
};

const mockAgent2: ControlPlaneAgent = {
  agent_id: 'agent-2',
  name: 'GPT-4 API',
  provider: 'openai',
  status: 'busy',
  model: 'gpt-4',
  capabilities: ['reasoning'],
  elo_rating: 1450,
};

const mockTask: ControlPlaneTask = {
  id: 'task-1',
  name: 'Analyze document',
  status: 'running',
  assigned_agent: 'agent-1',
  created_at: '2026-01-25T09:00:00Z',
  started_at: '2026-01-25T09:01:00Z',
};

const mockTask2: ControlPlaneTask = {
  id: 'task-2',
  name: 'Generate report',
  status: 'pending',
  created_at: '2026-01-25T09:30:00Z',
};

const mockHealth: ControlPlaneHealth = {
  status: 'healthy',
  agents: {
    'agent-1': {
      agent_id: 'agent-1',
      status: 'healthy',
      last_heartbeat: '2026-01-25T10:00:00Z',
      latency_ms: 50,
      error_rate: 0.01,
    },
  },
  agents_available: 2,
  agents_total: 3,
  active_tasks: 1,
  uptime_seconds: 86400,
};

const mockStats: ControlPlaneStats = {
  total_agents: 3,
  available_agents: 2,
  busy_agents: 1,
  offline_agents: 0,
  pending_tasks: 5,
  running_tasks: 2,
  completed_tasks_24h: 50,
  failed_tasks_24h: 2,
  avg_task_duration_ms: 5000,
  queue_size: 10,
  uptime_seconds: 86400,
};

describe('controlPlaneStore', () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    act(() => {
      useControlPlaneStore.getState().resetAll();
    });
  });

  describe('Agent Management', () => {
    it('setAgents replaces all agents', () => {
      act(() => {
        useControlPlaneStore.getState().setAgents([mockAgent, mockAgent2]);
      });

      const state = useControlPlaneStore.getState();
      expect(state.agents).toHaveLength(2);
      expect(state.agentsLoading).toBe(false);
      expect(state.agentsError).toBeNull();
    });

    it('updateAgent updates existing agent', () => {
      act(() => {
        useControlPlaneStore.getState().setAgents([mockAgent]);
      });

      const updatedAgent = { ...mockAgent, status: 'busy' as const };
      act(() => {
        useControlPlaneStore.getState().updateAgent(updatedAgent);
      });

      const state = useControlPlaneStore.getState();
      expect(state.agents).toHaveLength(1);
      expect(state.agents[0].status).toBe('busy');
    });

    it('updateAgent adds new agent if not found', () => {
      act(() => {
        useControlPlaneStore.getState().setAgents([mockAgent]);
        useControlPlaneStore.getState().updateAgent(mockAgent2);
      });

      const state = useControlPlaneStore.getState();
      expect(state.agents).toHaveLength(2);
    });

    it('removeAgent removes agent by ID', () => {
      act(() => {
        useControlPlaneStore.getState().setAgents([mockAgent, mockAgent2]);
        useControlPlaneStore.getState().removeAgent('agent-1');
      });

      const state = useControlPlaneStore.getState();
      expect(state.agents).toHaveLength(1);
      expect(state.agents[0].agent_id).toBe('agent-2');
    });

    it('setAgentsLoading updates loading state', () => {
      act(() => {
        useControlPlaneStore.getState().setAgentsLoading(true);
      });

      expect(useControlPlaneStore.getState().agentsLoading).toBe(true);
    });

    it('setAgentsError sets error and clears loading', () => {
      act(() => {
        useControlPlaneStore.getState().setAgentsLoading(true);
        useControlPlaneStore.getState().setAgentsError('Failed to fetch agents');
      });

      const state = useControlPlaneStore.getState();
      expect(state.agentsError).toBe('Failed to fetch agents');
      expect(state.agentsLoading).toBe(false);
    });
  });

  describe('Task Management', () => {
    it('setTasks replaces all tasks', () => {
      act(() => {
        useControlPlaneStore.getState().setTasks([mockTask, mockTask2]);
      });

      const state = useControlPlaneStore.getState();
      expect(state.tasks).toHaveLength(2);
      expect(state.tasksLoading).toBe(false);
      expect(state.tasksError).toBeNull();
    });

    it('updateTask updates existing task', () => {
      act(() => {
        useControlPlaneStore.getState().setTasks([mockTask]);
      });

      const updatedTask = { ...mockTask, status: 'completed' as const };
      act(() => {
        useControlPlaneStore.getState().updateTask(updatedTask);
      });

      const state = useControlPlaneStore.getState();
      expect(state.tasks[0].status).toBe('completed');
    });

    it('updateTask adds new task if not found', () => {
      act(() => {
        useControlPlaneStore.getState().setTasks([mockTask]);
        useControlPlaneStore.getState().updateTask(mockTask2);
      });

      expect(useControlPlaneStore.getState().tasks).toHaveLength(2);
    });

    it('removeTask removes task by ID', () => {
      act(() => {
        useControlPlaneStore.getState().setTasks([mockTask, mockTask2]);
        useControlPlaneStore.getState().removeTask('task-1');
      });

      const state = useControlPlaneStore.getState();
      expect(state.tasks).toHaveLength(1);
      expect(state.tasks[0].id).toBe('task-2');
    });

    it('setTaskFilters updates filters', () => {
      act(() => {
        useControlPlaneStore.getState().setTaskFilters({ status: 'pending' });
      });

      expect(useControlPlaneStore.getState().taskFilters.status).toBe('pending');
    });

    it('setTaskFilters merges with existing filters', () => {
      act(() => {
        useControlPlaneStore.getState().setTaskFilters({ status: 'running' });
        useControlPlaneStore.getState().setTaskFilters({ agentId: 'agent-1' });
      });

      const state = useControlPlaneStore.getState();
      expect(state.taskFilters.status).toBe('running');
      expect(state.taskFilters.agentId).toBe('agent-1');
    });
  });

  describe('Health and Stats', () => {
    it('setHealth updates health data', () => {
      act(() => {
        useControlPlaneStore.getState().setHealth(mockHealth);
      });

      const state = useControlPlaneStore.getState();
      expect(state.health).toEqual(mockHealth);
      expect(state.healthLoading).toBe(false);
    });

    it('setHealthLoading updates loading state', () => {
      act(() => {
        useControlPlaneStore.getState().setHealthLoading(true);
      });

      expect(useControlPlaneStore.getState().healthLoading).toBe(true);
    });

    it('setStats updates stats data', () => {
      act(() => {
        useControlPlaneStore.getState().setStats(mockStats);
      });

      const state = useControlPlaneStore.getState();
      expect(state.stats).toEqual(mockStats);
      expect(state.statsLoading).toBe(false);
    });
  });

  describe('Connection State', () => {
    it('setIsConnected updates connection status', () => {
      act(() => {
        useControlPlaneStore.getState().setIsConnected(true);
      });

      expect(useControlPlaneStore.getState().isConnected).toBe(true);
    });

    it('setLastUpdate updates timestamp', () => {
      const timestamp = Date.now();
      act(() => {
        useControlPlaneStore.getState().setLastUpdate(timestamp);
      });

      expect(useControlPlaneStore.getState().lastUpdate).toBe(timestamp);
    });
  });

  describe('Selection State', () => {
    it('setSelectedAgentId updates selected agent', () => {
      act(() => {
        useControlPlaneStore.getState().setSelectedAgentId('agent-1');
      });

      expect(useControlPlaneStore.getState().selectedAgentId).toBe('agent-1');
    });

    it('setSelectedTaskId updates selected task', () => {
      act(() => {
        useControlPlaneStore.getState().setSelectedTaskId('task-1');
      });

      expect(useControlPlaneStore.getState().selectedTaskId).toBe('task-1');
    });
  });

  describe('resetAll', () => {
    it('resets all state to initial values', () => {
      // Set up some state
      act(() => {
        useControlPlaneStore.getState().setAgents([mockAgent]);
        useControlPlaneStore.getState().setTasks([mockTask]);
        useControlPlaneStore.getState().setHealth(mockHealth);
        useControlPlaneStore.getState().setStats(mockStats);
        useControlPlaneStore.getState().setIsConnected(true);
        useControlPlaneStore.getState().setSelectedAgentId('agent-1');
      });

      // Reset
      act(() => {
        useControlPlaneStore.getState().resetAll();
      });

      const state = useControlPlaneStore.getState();
      expect(state.agents).toHaveLength(0);
      expect(state.tasks).toHaveLength(0);
      expect(state.health).toBeNull();
      expect(state.stats).toBeNull();
      expect(state.isConnected).toBe(false);
      expect(state.selectedAgentId).toBeNull();
    });
  });

  describe('Selectors', () => {
    beforeEach(() => {
      act(() => {
        useControlPlaneStore.getState().setAgents([mockAgent, mockAgent2]);
        useControlPlaneStore.getState().setTasks([mockTask, mockTask2]);
        useControlPlaneStore.getState().setHealth(mockHealth);
        useControlPlaneStore.getState().setStats(mockStats);
        useControlPlaneStore.getState().setIsConnected(true);
      });
    });

    it('selectAgents returns agents', () => {
      const agents = selectAgents(useControlPlaneStore.getState());
      expect(agents).toHaveLength(2);
    });

    it('selectTasks returns tasks', () => {
      const tasks = selectTasks(useControlPlaneStore.getState());
      expect(tasks).toHaveLength(2);
    });

    it('selectHealth returns health', () => {
      const health = selectHealth(useControlPlaneStore.getState());
      expect(health).toEqual(mockHealth);
    });

    it('selectStats returns stats', () => {
      const stats = selectStats(useControlPlaneStore.getState());
      expect(stats).toEqual(mockStats);
    });

    it('selectIsConnected returns connection status', () => {
      const connected = selectIsConnected(useControlPlaneStore.getState());
      expect(connected).toBe(true);
    });

    it('selectAgentById returns specific agent', () => {
      const agent = selectAgentById('agent-1')(useControlPlaneStore.getState());
      expect(agent?.name).toBe('Claude API');
    });

    it('selectAgentById returns undefined for non-existent agent', () => {
      const agent = selectAgentById('non-existent')(useControlPlaneStore.getState());
      expect(agent).toBeUndefined();
    });

    it('selectTaskById returns specific task', () => {
      const task = selectTaskById('task-1')(useControlPlaneStore.getState());
      expect(task?.name).toBe('Analyze document');
    });

    it('selectAgentsByStatus filters by status', () => {
      const availableAgents = selectAgentsByStatus('available')(useControlPlaneStore.getState());
      expect(availableAgents).toHaveLength(1);
      expect(availableAgents[0].agent_id).toBe('agent-1');
    });

    it('selectTasksByStatus filters by status', () => {
      const pendingTasks = selectTasksByStatus('pending')(useControlPlaneStore.getState());
      expect(pendingTasks).toHaveLength(1);
      expect(pendingTasks[0].id).toBe('task-2');
    });

    it('selectFilteredTasks applies status filter', () => {
      act(() => {
        useControlPlaneStore.getState().setTaskFilters({ status: 'running' });
      });

      const filtered = selectFilteredTasks(useControlPlaneStore.getState());
      expect(filtered).toHaveLength(1);
      expect(filtered[0].id).toBe('task-1');
    });

    it('selectFilteredTasks applies agent filter', () => {
      act(() => {
        useControlPlaneStore.getState().setTaskFilters({ agentId: 'agent-1' });
      });

      const filtered = selectFilteredTasks(useControlPlaneStore.getState());
      expect(filtered).toHaveLength(1);
      expect(filtered[0].id).toBe('task-1');
    });

    it('selectIsHealthy returns true for healthy status', () => {
      const isHealthy = selectIsHealthy(useControlPlaneStore.getState());
      expect(isHealthy).toBe(true);
    });

    it('selectIsHealthy returns false for degraded status', () => {
      act(() => {
        useControlPlaneStore.getState().setHealth({ ...mockHealth, status: 'degraded' });
      });

      const isHealthy = selectIsHealthy(useControlPlaneStore.getState());
      expect(isHealthy).toBe(false);
    });

    it('selectAgentCount returns count by status', () => {
      const counts = selectAgentCount(useControlPlaneStore.getState());
      expect(counts.total).toBe(2);
      expect(counts.available).toBe(1);
      expect(counts.busy).toBe(1);
      expect(counts.offline).toBe(0);
    });

    it('selectTaskCount returns count by status', () => {
      const counts = selectTaskCount(useControlPlaneStore.getState());
      expect(counts.total).toBe(2);
      expect(counts.pending).toBe(1);
      expect(counts.running).toBe(1);
      expect(counts.completed).toBe(0);
      expect(counts.failed).toBe(0);
    });
  });
});
