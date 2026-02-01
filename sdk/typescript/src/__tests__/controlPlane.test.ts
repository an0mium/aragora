/**
 * Control Plane Namespace Test Suite
 *
 * Tests for the control plane namespace including:
 * - Agent management (register, unregister, heartbeat, status)
 * - Task management (submit, claim, complete, fail, cancel)
 * - Policy management (create, update, delete, list)
 * - Schedule management (create, get, list, cancel)
 * - Deliberations (create, vote, close)
 * - Audit logs and violations
 * - Metrics
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AragoraClient, createClient } from '../client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Control Plane Namespace', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-key',
      retryEnabled: false,
    });
  });

  // ===========================================================================
  // Health Check
  // ===========================================================================

  describe('Health', () => {
    it('should get control plane health status', async () => {
      const mockHealth = {
        status: 'healthy',
        agents_total: 15,
        agents_active: 12,
        tasks_pending: 5,
        tasks_running: 8,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockHealth)),
      });

      const result = await client.controlPlane.getHealth();

      expect(result.status).toBe('healthy');
      expect(result.agents_total).toBe(15);
      expect(result.agents_active).toBe(12);
      expect(result.tasks_pending).toBe(5);
      expect(result.tasks_running).toBe(8);
    });

    it('should handle degraded status', async () => {
      const mockHealth = {
        status: 'degraded',
        agents_total: 10,
        agents_active: 3,
        tasks_pending: 50,
        tasks_running: 2,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockHealth)),
      });

      const result = await client.controlPlane.getHealth();

      expect(result.status).toBe('degraded');
      expect(result.agents_active).toBeLessThan(result.agents_total);
    });
  });

  // ===========================================================================
  // Agents Sub-API
  // ===========================================================================

  describe('Agents', () => {
    it('should register an agent', async () => {
      const mockResponse = {
        registered: true,
        agent_id: 'worker-001',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.controlPlane.agents.register({
        agent_id: 'worker-001',
        name: 'Code Review Worker',
        capabilities: ['code-review', 'security-scan', 'lint'],
        metadata: { version: '1.0.0' },
      });

      expect(result.registered).toBe(true);
      expect(result.agent_id).toBe('worker-001');
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/control-plane/agents/register'),
        expect.objectContaining({ method: 'POST' })
      );
    });

    it('should unregister an agent', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ unregistered: true })),
      });

      const result = await client.controlPlane.agents.unregister('worker-001');

      expect(result.unregistered).toBe(true);
    });

    it('should send heartbeat', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ acknowledged: true })),
      });

      const result = await client.controlPlane.agents.heartbeat({
        agent_id: 'worker-001',
        status: 'busy',
        current_task: 'task-123',
        metrics: { cpu_percent: 45, memory_mb: 512 },
      });

      expect(result.acknowledged).toBe(true);
    });

    it('should get agent status', async () => {
      const mockStatus = {
        agent_id: 'worker-001',
        status: 'idle',
        last_heartbeat: '2024-01-15T10:30:00Z',
        current_task: undefined,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.controlPlane.agents.getStatus('worker-001');

      expect(result.agent_id).toBe('worker-001');
      expect(result.status).toBe('idle');
      expect(result.last_heartbeat).toBeDefined();
    });

    it('should list registered agents', async () => {
      const mockAgents = {
        agents: [
          { agent_id: 'w1', name: 'Worker 1', status: 'idle', capabilities: ['review'] },
          { agent_id: 'w2', name: 'Worker 2', status: 'busy', capabilities: ['scan'] },
          { agent_id: 'w3', name: 'Worker 3', status: 'offline' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockAgents)),
      });

      const result = await client.controlPlane.agents.list();

      expect(result.agents).toHaveLength(3);
      expect(result.agents[0].agent_id).toBe('w1');
    });

    it('should filter agents by status and capability', async () => {
      const mockAgents = {
        agents: [
          { agent_id: 'w1', name: 'Worker 1', status: 'idle', capabilities: ['code-review'] },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockAgents)),
      });

      const result = await client.controlPlane.agents.list({
        status: 'idle',
        capability: 'code-review',
        limit: 10,
      });

      expect(result.agents).toHaveLength(1);
      const calledUrl = mockFetch.mock.calls[0][0];
      expect(calledUrl).toContain('status=idle');
      expect(calledUrl).toContain('capability=code-review');
    });
  });

  // ===========================================================================
  // Tasks Sub-API
  // ===========================================================================

  describe('Tasks', () => {
    it('should submit a task', async () => {
      const mockResponse = {
        task_id: 'task-456',
        status: 'pending',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.controlPlane.tasks.submit({
        task_type: 'code-review',
        payload: { repo: 'my-repo', pr_number: 42 },
        priority: 'high',
        timeout_seconds: 300,
      });

      expect(result.task_id).toBe('task-456');
      expect(result.status).toBe('pending');
    });

    it('should get task status', async () => {
      const mockStatus = {
        task_id: 'task-456',
        status: 'running',
        assigned_agent: 'worker-001',
        submitted_at: '2024-01-15T10:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.controlPlane.tasks.getStatus('task-456');

      expect(result.task_id).toBe('task-456');
      expect(result.status).toBe('running');
      expect(result.assigned_agent).toBe('worker-001');
    });

    it('should list tasks with filters', async () => {
      const mockTasks = {
        tasks: [
          { task_id: 't1', task_type: 'review', status: 'pending', priority: 'high', submitted_at: '2024-01-15T09:00:00Z' },
          { task_id: 't2', task_type: 'scan', status: 'running', priority: 'normal', submitted_at: '2024-01-15T09:30:00Z' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockTasks)),
      });

      const result = await client.controlPlane.tasks.list({ status: 'pending' });

      expect(result.tasks).toHaveLength(2);
    });

    it('should claim a task', async () => {
      const mockClaim = {
        task_id: 'task-789',
        payload: { repo: 'test-repo', branch: 'main' },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockClaim)),
      });

      const result = await client.controlPlane.tasks.claim({
        agent_id: 'worker-001',
        task_type: 'code-review',
        capabilities: ['review', 'lint'],
      });

      expect(result).not.toBeNull();
      expect(result!.task_id).toBe('task-789');
      expect(result!.payload).toBeDefined();
    });

    it('should return null when no task available to claim', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve('null'),
      });

      const result = await client.controlPlane.tasks.claim({
        agent_id: 'worker-001',
      });

      expect(result).toBeNull();
    });

    it('should complete a task', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ completed: true })),
      });

      const result = await client.controlPlane.tasks.complete('task-456', {
        result: { findings: [], passed: true },
        metrics: { duration_ms: 1500 },
      });

      expect(result.completed).toBe(true);
    });

    it('should fail a task', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ failed: true })),
      });

      const result = await client.controlPlane.tasks.fail('task-456', {
        error: 'Repository not accessible',
        retry: true,
      });

      expect(result.failed).toBe(true);
    });

    it('should cancel a task', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ cancelled: true })),
      });

      const result = await client.controlPlane.tasks.cancel('task-456');

      expect(result.cancelled).toBe(true);
    });
  });

  // ===========================================================================
  // Policies Sub-API
  // ===========================================================================

  describe('Policies', () => {
    it('should create a policy', async () => {
      const mockResponse = {
        policy_id: 'policy-001',
        created: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.controlPlane.policies.create({
        name: 'Rate Limit Policy',
        description: 'Limits requests per minute',
        rules: [
          { condition: 'requests_per_minute > 100', action: 'deny', priority: 1 },
          { condition: 'user.tier == premium', action: 'allow', priority: 2 },
        ],
        enabled: true,
        scope: 'global',
      });

      expect(result.policy_id).toBe('policy-001');
      expect(result.created).toBe(true);
    });

    it('should get a policy by ID', async () => {
      const mockPolicy = {
        policy_id: 'policy-001',
        name: 'Rate Limit Policy',
        description: 'Limits requests per minute',
        rules: [
          { condition: 'requests_per_minute > 100', action: 'deny', priority: 1 },
        ],
        enabled: true,
        scope: 'global',
        created_at: '2024-01-10T00:00:00Z',
        updated_at: '2024-01-15T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockPolicy)),
      });

      const result = await client.controlPlane.policies.get('policy-001');

      expect(result.policy_id).toBe('policy-001');
      expect(result.name).toBe('Rate Limit Policy');
      expect(result.rules).toHaveLength(1);
    });

    it('should update a policy', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ updated: true })),
      });

      const result = await client.controlPlane.policies.update('policy-001', {
        enabled: false,
      });

      expect(result.updated).toBe(true);
    });

    it('should delete a policy', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ deleted: true })),
      });

      const result = await client.controlPlane.policies.delete('policy-001');

      expect(result.deleted).toBe(true);
    });

    it('should list policies', async () => {
      const mockPolicies = {
        policies: [
          { policy_id: 'p1', name: 'Policy 1', enabled: true, scope: 'global', rules_count: 3 },
          { policy_id: 'p2', name: 'Policy 2', enabled: false, scope: 'tenant', rules_count: 1 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockPolicies)),
      });

      const result = await client.controlPlane.policies.list({ enabled: true });

      expect(result.policies).toHaveLength(2);
    });
  });

  // ===========================================================================
  // Schedules Sub-API
  // ===========================================================================

  describe('Schedules', () => {
    it('should create a scheduled task with cron', async () => {
      const mockResponse = {
        schedule_id: 'sched-001',
        next_run_at: '2024-01-16T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.controlPlane.schedules.create({
        task_type: 'cleanup',
        payload: { target: 'logs', max_age_days: 30 },
        cron: '0 0 * * *',
        priority: 'low',
      });

      expect(result.schedule_id).toBe('sched-001');
      expect(result.next_run_at).toBeDefined();
    });

    it('should create a one-time scheduled task', async () => {
      const mockResponse = {
        schedule_id: 'sched-002',
        next_run_at: '2024-01-20T15:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.controlPlane.schedules.create({
        task_type: 'backup',
        payload: { database: 'main' },
        schedule_at: '2024-01-20T15:00:00Z',
      });

      expect(result.schedule_id).toBe('sched-002');
    });

    it('should get a scheduled task', async () => {
      const mockSchedule = {
        schedule_id: 'sched-001',
        task_type: 'cleanup',
        status: 'active',
        cron: '0 0 * * *',
        next_run_at: '2024-01-16T00:00:00Z',
        last_run_at: '2024-01-15T00:00:00Z',
        run_count: 10,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSchedule)),
      });

      const result = await client.controlPlane.schedules.get('sched-001');

      expect(result.schedule_id).toBe('sched-001');
      expect(result.status).toBe('active');
      expect(result.run_count).toBe(10);
    });

    it('should list scheduled tasks', async () => {
      const mockSchedules = {
        schedules: [
          { schedule_id: 's1', task_type: 'cleanup', status: 'active', cron: '0 0 * * *' },
          { schedule_id: 's2', task_type: 'backup', status: 'paused', next_run_at: '2024-01-20T00:00:00Z' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSchedules)),
      });

      const result = await client.controlPlane.schedules.list();

      expect(result.schedules).toHaveLength(2);
    });

    it('should cancel a scheduled task', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ cancelled: true })),
      });

      const result = await client.controlPlane.schedules.cancel('sched-001');

      expect(result.cancelled).toBe(true);
    });
  });

  // ===========================================================================
  // Deliberations Sub-API
  // ===========================================================================

  describe('Deliberations', () => {
    it('should create a deliberation', async () => {
      const mockResponse = {
        deliberation_id: 'delib-001',
        created: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResponse)),
      });

      const result = await client.controlPlane.deliberations.create({
        topic: 'Should we adopt microservices?',
        description: 'Discussing architectural changes',
        participants: ['agent-1', 'agent-2', 'agent-3'],
        deadline: '2024-01-20T00:00:00Z',
      });

      expect(result.deliberation_id).toBe('delib-001');
      expect(result.created).toBe(true);
    });

    it('should get a deliberation', async () => {
      const mockDeliberation = {
        deliberation_id: 'delib-001',
        topic: 'Should we adopt microservices?',
        description: 'Discussing architectural changes',
        status: 'voting',
        participants: ['agent-1', 'agent-2', 'agent-3'],
        votes: [
          { participant: 'agent-1', vote: 'yes', timestamp: '2024-01-15T10:00:00Z' },
        ],
        created_at: '2024-01-14T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockDeliberation)),
      });

      const result = await client.controlPlane.deliberations.get('delib-001');

      expect(result.deliberation_id).toBe('delib-001');
      expect(result.status).toBe('voting');
      expect(result.votes).toHaveLength(1);
    });

    it('should list deliberations', async () => {
      const mockDelibs = {
        deliberations: [
          { deliberation_id: 'd1', topic: 'Topic 1', status: 'open', participants_count: 3, created_at: '2024-01-14T00:00:00Z' },
          { deliberation_id: 'd2', topic: 'Topic 2', status: 'closed', participants_count: 5, created_at: '2024-01-10T00:00:00Z' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockDelibs)),
      });

      const result = await client.controlPlane.deliberations.list({ status: 'open' });

      expect(result.deliberations).toHaveLength(2);
    });

    it('should vote on a deliberation', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ voted: true })),
      });

      const result = await client.controlPlane.deliberations.vote('delib-001', {
        participant: 'agent-2',
        vote: 'yes',
        rationale: 'Better scalability',
      });

      expect(result.voted).toBe(true);
    });

    it('should close a deliberation', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ closed: true, outcome: 'approved' })),
      });

      const result = await client.controlPlane.deliberations.close('delib-001', {
        outcome: 'approved',
      });

      expect(result.closed).toBe(true);
      expect(result.outcome).toBe('approved');
    });

    it('should get deliberation transcript', async () => {
      const mockTranscript = {
        deliberation_id: 'delib-001',
        topic: 'Should we adopt microservices?',
        transcript: [
          { type: 'message', participant: 'agent-1', content: 'I think we should...', timestamp: '2024-01-15T10:00:00Z' },
          { type: 'vote', participant: 'agent-1', content: 'yes', timestamp: '2024-01-15T10:05:00Z' },
          { type: 'decision', content: 'Approved by majority', timestamp: '2024-01-15T12:00:00Z' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockTranscript)),
      });

      const result = await client.controlPlane.deliberations.getTranscript('delib-001');

      expect(result.deliberation_id).toBe('delib-001');
      expect(result.transcript).toHaveLength(3);
      expect(result.transcript[0].type).toBe('message');
    });
  });

  // ===========================================================================
  // Audit Logs Sub-API
  // ===========================================================================

  describe('Audit Logs', () => {
    it('should list audit logs', async () => {
      const mockLogs = {
        logs: [
          { log_id: 'log-1', action: 'agent.register', actor: 'user-1', timestamp: '2024-01-15T10:00:00Z' },
          { log_id: 'log-2', action: 'task.submit', actor: 'user-2', resource_type: 'task', resource_id: 'task-123', timestamp: '2024-01-15T10:05:00Z' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockLogs)),
      });

      const result = await client.controlPlane.auditLogs.list({
        action: 'agent.register',
        limit: 50,
      });

      expect(result.logs).toHaveLength(2);
    });

    it('should get a specific audit log', async () => {
      const mockLog = {
        log_id: 'log-1',
        action: 'agent.register',
        actor: 'user-1',
        details: { agent_id: 'worker-001', capabilities: ['review'] },
        ip_address: '192.168.1.1',
        user_agent: 'Mozilla/5.0',
        timestamp: '2024-01-15T10:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockLog)),
      });

      const result = await client.controlPlane.auditLogs.get('log-1');

      expect(result.log_id).toBe('log-1');
      expect(result.action).toBe('agent.register');
      expect(result.ip_address).toBeDefined();
    });
  });

  // ===========================================================================
  // Violations Sub-API
  // ===========================================================================

  describe('Violations', () => {
    it('should list policy violations', async () => {
      const mockViolations = {
        violations: [
          {
            violation_id: 'v1',
            policy_id: 'policy-001',
            policy_name: 'Rate Limit Policy',
            severity: 'high',
            actor: 'user-1',
            acknowledged: false,
            timestamp: '2024-01-15T10:00:00Z',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockViolations)),
      });

      const result = await client.controlPlane.violations.list({
        severity: 'high',
        acknowledged: false,
      });

      expect(result.violations).toHaveLength(1);
      expect(result.violations[0].severity).toBe('high');
    });

    it('should acknowledge a violation', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ acknowledged: true })),
      });

      const result = await client.controlPlane.violations.acknowledge('v1', {
        notes: 'Reviewed and approved as expected behavior',
      });

      expect(result.acknowledged).toBe(true);
    });

    it('should escalate a violation', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ escalated: true, escalation_id: 'esc-001' })),
      });

      const result = await client.controlPlane.violations.escalate('v1', {
        escalate_to: 'security-team',
        reason: 'Potential security breach',
      });

      expect(result.escalated).toBe(true);
      expect(result.escalation_id).toBe('esc-001');
    });
  });

  // ===========================================================================
  // Metrics Sub-API
  // ===========================================================================

  describe('Metrics', () => {
    it('should get agent metrics', async () => {
      const mockMetrics = {
        agent_id: 'worker-001',
        metrics: {
          tasks_completed: 150,
          tasks_failed: 5,
          avg_task_duration_ms: 2500,
          uptime_percent: 99.5,
          error_rate: 0.033,
        },
        timeseries: [
          { timestamp: '2024-01-15T10:00:00Z', tasks_completed: 10, tasks_failed: 0 },
          { timestamp: '2024-01-15T11:00:00Z', tasks_completed: 12, tasks_failed: 1 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockMetrics)),
      });

      const result = await client.controlPlane.metrics.getAgentMetrics('worker-001', {
        resolution: 'hour',
      });

      expect(result.agent_id).toBe('worker-001');
      expect(result.metrics.tasks_completed).toBe(150);
      expect(result.timeseries).toHaveLength(2);
    });

    it('should get task metrics', async () => {
      const mockMetrics = {
        metrics: {
          total_submitted: 1000,
          total_completed: 950,
          total_failed: 50,
          avg_wait_time_ms: 500,
          avg_execution_time_ms: 3000,
        },
        by_type: {
          'code-review': { submitted: 500, completed: 480, failed: 20 },
          'security-scan': { submitted: 300, completed: 290, failed: 10 },
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockMetrics)),
      });

      const result = await client.controlPlane.metrics.getTaskMetrics({
        resolution: 'day',
      });

      expect(result.metrics.total_submitted).toBe(1000);
      expect(result.by_type).toBeDefined();
    });

    it('should get system metrics', async () => {
      const mockMetrics = {
        agents: {
          total: 20,
          active: 15,
          idle: 3,
          offline: 2,
        },
        tasks: {
          pending: 25,
          running: 10,
          completed_24h: 500,
          failed_24h: 15,
        },
        policies: {
          total: 10,
          enabled: 8,
          violations_24h: 3,
        },
        deliberations: {
          open: 2,
          completed_24h: 5,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockMetrics)),
      });

      const result = await client.controlPlane.metrics.getSystemMetrics();

      expect(result.agents.total).toBe(20);
      expect(result.tasks.pending).toBe(25);
      expect(result.policies.violations_24h).toBe(3);
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================

  describe('Error Handling', () => {
    it('should handle agent not found', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({
          error: 'Agent not found',
          code: 'NOT_FOUND',
        }),
      });

      await expect(client.controlPlane.agents.getStatus('nonexistent'))
        .rejects.toThrow('Agent not found');
    });

    it('should handle task already completed', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Task already completed',
          code: 'INVALID_VALUE',
        }),
      });

      await expect(
        client.controlPlane.tasks.complete('task-done', { result: {} })
      ).rejects.toThrow('Task already completed');
    });

    it('should handle policy conflict', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: () => Promise.resolve({
          error: 'Policy name already exists',
          code: 'INVALID_VALUE',
        }),
      });

      await expect(
        client.controlPlane.policies.create({
          name: 'Existing Policy',
          rules: [],
        })
      ).rejects.toThrow('Policy name already exists');
    });

    it('should handle forbidden access', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: () => Promise.resolve({
          error: 'Insufficient permissions to access control plane',
          code: 'FORBIDDEN',
        }),
      });

      await expect(client.controlPlane.getHealth())
        .rejects.toThrow('Insufficient permissions');
    });
  });

  // ===========================================================================
  // Type Safety
  // ===========================================================================

  describe('Type Safety', () => {
    it('should return properly typed health response', async () => {
      const mockHealth = {
        status: 'healthy',
        agents_total: 10,
        agents_active: 8,
        tasks_pending: 5,
        tasks_running: 3,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockHealth)),
      });

      const result = await client.controlPlane.getHealth();

      expect(['healthy', 'degraded', 'unhealthy']).toContain(result.status);
      expect(typeof result.agents_total).toBe('number');
      expect(typeof result.agents_active).toBe('number');
      expect(typeof result.tasks_pending).toBe('number');
      expect(typeof result.tasks_running).toBe('number');
    });

    it('should return properly typed agent list', async () => {
      const mockAgents = {
        agents: [
          { agent_id: 'w1', name: 'Worker 1', status: 'idle', capabilities: ['review', 'scan'] },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockAgents)),
      });

      const result = await client.controlPlane.agents.list();

      expect(Array.isArray(result.agents)).toBe(true);
      expect(typeof result.agents[0].agent_id).toBe('string');
      expect(Array.isArray(result.agents[0].capabilities)).toBe(true);
    });

    it('should return properly typed task status', async () => {
      const mockStatus = {
        task_id: 'task-123',
        status: 'completed',
        assigned_agent: 'worker-001',
        result: { passed: true },
        submitted_at: '2024-01-15T10:00:00Z',
        completed_at: '2024-01-15T10:05:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.controlPlane.tasks.getStatus('task-123');

      expect(typeof result.task_id).toBe('string');
      expect(typeof result.status).toBe('string');
      expect(typeof result.submitted_at).toBe('string');
    });
  });
});
