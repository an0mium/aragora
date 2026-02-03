/**
 * Nomic Namespace Tests
 *
 * Comprehensive tests for the nomic namespace API including:
 * - State and monitoring
 * - Loop control
 * - Proposal management
 * - Operational modes
 * - Gas Town (witnesses, mayor)
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { NomicAPI } from '../nomic';

interface MockClient {
  get: Mock;
  post: Mock;
  request: Mock;
}

describe('NomicAPI Namespace', () => {
  let api: NomicAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
      request: vi.fn(),
    };
    api = new NomicAPI(mockClient as any);
  });

  // ===========================================================================
  // State & Monitoring
  // ===========================================================================

  describe('State & Monitoring', () => {
    it('should get nomic state', async () => {
      const mockState = {
        running: true,
        cycle: 5,
        phase: 'implement',
        paused: false,
        started_at: '2024-01-20T08:00:00Z',
        last_update: '2024-01-20T10:00:00Z',
        target_cycles: 10,
        auto_approve: false,
      };
      mockClient.get.mockResolvedValue(mockState);

      const result = await api.getState();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/nomic/state');
      expect(result.running).toBe(true);
      expect(result.phase).toBe('implement');
    });

    it('should get nomic health', async () => {
      const mockHealth = {
        status: 'healthy',
        cycle: 5,
        phase: 'implement',
        last_activity: '2024-01-20T10:00:00Z',
        stall_duration_seconds: null,
        warnings: [],
      };
      mockClient.get.mockResolvedValue(mockHealth);

      const result = await api.getHealth();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/nomic/health');
      expect(result.status).toBe('healthy');
    });

    it('should detect stalled loop', async () => {
      const mockHealth = {
        status: 'stalled',
        cycle: 5,
        phase: 'verify',
        last_activity: '2024-01-20T09:00:00Z',
        stall_duration_seconds: 3600,
        warnings: ['Loop has been stalled for 1 hour'],
      };
      mockClient.get.mockResolvedValue(mockHealth);

      const result = await api.getHealth();

      expect(result.status).toBe('stalled');
      expect(result.warnings).toHaveLength(1);
    });

    it('should get nomic metrics', async () => {
      const mockMetrics = {
        summary: {
          total_cycles: 100,
          successful_cycles: 95,
          failed_cycles: 5,
        },
        stuck_detection: {
          is_stuck: false,
        },
        status: 'running',
      };
      mockClient.get.mockResolvedValue(mockMetrics);

      const result = await api.getMetrics();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/nomic/metrics');
      expect(result.stuck_detection.is_stuck).toBe(false);
    });

    it('should get nomic logs', async () => {
      const mockLogs = {
        lines: [
          '[2024-01-20 10:00:00] Starting cycle 5',
          '[2024-01-20 10:00:01] Phase: implement',
          '[2024-01-20 10:00:05] Implementation complete',
        ],
        total: 1000,
        showing: 100,
      };
      mockClient.request.mockResolvedValue(mockLogs);

      const result = await api.getLogs(100);

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/nomic/log', { params: { lines: 100 } });
      expect(result.lines).toHaveLength(3);
    });

    it('should get risk register', async () => {
      const mockRisks = {
        risks: [
          {
            id: 'risk_1',
            severity: 'high',
            description: 'Test coverage below threshold',
            category: 'quality',
            created_at: '2024-01-20T09:00:00Z',
          },
        ],
        total: 1,
        critical_count: 0,
        high_count: 1,
      };
      mockClient.request.mockResolvedValue(mockRisks);

      const result = await api.getRiskRegister({ limit: 50 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/nomic/risk-register', { params: { limit: 50 } });
      expect(result.high_count).toBe(1);
    });
  });

  // ===========================================================================
  // Loop Control
  // ===========================================================================

  describe('Loop Control', () => {
    it('should start nomic loop', async () => {
      const mockResult = { status: 'started', pid: 12345, target_cycles: 5 };
      mockClient.post.mockResolvedValue(mockResult);

      const result = await api.start({ cycles: 5, auto_approve: false });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/nomic/control/start', { cycles: 5, auto_approve: false });
      expect(result.status).toBe('started');
    });

    it('should start loop with dry run', async () => {
      const mockResult = { status: 'started', pid: 12346, target_cycles: 1 };
      mockClient.post.mockResolvedValue(mockResult);

      await api.start({ dry_run: true });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/nomic/control/start', { dry_run: true });
    });

    it('should stop nomic loop', async () => {
      const mockResult = { status: 'stopped', pid: 12345 };
      mockClient.post.mockResolvedValue(mockResult);

      const result = await api.stop({ graceful: true });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/nomic/control/stop', { graceful: true });
      expect(result.status).toBe('stopped');
    });

    it('should pause nomic loop', async () => {
      const mockResult = { status: 'paused', cycle: 5, phase: 'implement' };
      mockClient.post.mockResolvedValue(mockResult);

      const result = await api.pause();

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/nomic/control/pause');
      expect(result.status).toBe('paused');
    });

    it('should resume nomic loop', async () => {
      const mockResult = { status: 'resumed', cycle: 5, phase: 'implement' };
      mockClient.post.mockResolvedValue(mockResult);

      const result = await api.resume();

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/nomic/control/resume');
      expect(result.status).toBe('resumed');
    });

    it('should skip phase', async () => {
      const mockResult = {
        status: 'skipped',
        previous_phase: 'implement',
        next_phase: 'verify',
        cycle: 5,
      };
      mockClient.post.mockResolvedValue(mockResult);

      const result = await api.skipPhase();

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/nomic/control/skip-phase');
      expect(result.previous_phase).toBe('implement');
      expect(result.next_phase).toBe('verify');
    });
  });

  // ===========================================================================
  // Proposal Management
  // ===========================================================================

  describe('Proposal Management', () => {
    it('should get proposals', async () => {
      const mockProposals = {
        proposals: [
          {
            id: 'prop_1',
            title: 'Add caching layer',
            description: 'Implement Redis caching for API responses',
            status: 'pending',
            submitted_at: '2024-01-20T09:00:00Z',
          },
          {
            id: 'prop_2',
            title: 'Improve error handling',
            description: 'Add structured error types',
            status: 'pending',
            submitted_at: '2024-01-20T09:30:00Z',
          },
        ],
        total: 2,
        all_proposals: 50,
      };
      mockClient.get.mockResolvedValue(mockProposals);

      const result = await api.getProposals();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/nomic/proposals');
      expect(result.proposals).toHaveLength(2);
    });

    it('should approve proposal', async () => {
      const mockResult = { status: 'approved', proposal_id: 'prop_1' };
      mockClient.post.mockResolvedValue(mockResult);

      const result = await api.approveProposal({
        proposal_id: 'prop_1',
        approved_by: 'admin@example.com',
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/nomic/proposals/approve', {
        proposal_id: 'prop_1',
        approved_by: 'admin@example.com',
      });
      expect(result.status).toBe('approved');
    });

    it('should reject proposal', async () => {
      const mockResult = { status: 'rejected', proposal_id: 'prop_2' };
      mockClient.post.mockResolvedValue(mockResult);

      const result = await api.rejectProposal({
        proposal_id: 'prop_2',
        rejected_by: 'admin@example.com',
        reason: 'Not aligned with current priorities',
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/nomic/proposals/reject', {
        proposal_id: 'prop_2',
        rejected_by: 'admin@example.com',
        reason: 'Not aligned with current priorities',
      });
    });

    it('should get specific proposal', async () => {
      const mockProposals = {
        proposals: [
          { id: 'prop_1', title: 'First', status: 'pending' },
          { id: 'prop_2', title: 'Second', status: 'pending' },
        ],
        total: 2,
        all_proposals: 2,
      };
      mockClient.get.mockResolvedValue(mockProposals);

      const result = await api.getProposal('prop_1');

      expect(result?.id).toBe('prop_1');
    });

    it('should return null for non-existent proposal', async () => {
      const mockProposals = { proposals: [], total: 0, all_proposals: 0 };
      mockClient.get.mockResolvedValue(mockProposals);

      const result = await api.getProposal('non_existent');

      expect(result).toBeNull();
    });
  });

  // ===========================================================================
  // Operational Modes
  // ===========================================================================

  describe('Operational Modes', () => {
    it('should get modes', async () => {
      const mockModes = {
        modes: [
          { name: 'Development', type: 'builtin', description: 'Development mode', slug: 'dev' },
          { name: 'Production', type: 'builtin', description: 'Production mode', slug: 'prod' },
          { name: 'Custom Mode', type: 'custom', description: 'User-defined mode' },
        ],
        total: 3,
      };
      mockClient.get.mockResolvedValue(mockModes);

      const result = await api.getModes();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/modes');
      expect(result.modes).toHaveLength(3);
    });
  });

  // ===========================================================================
  // Gas Town
  // ===========================================================================

  describe('Gas Town', () => {
    it('should get witness status', async () => {
      const mockStatus = {
        active: true,
        patrol_count: 150,
        last_patrol: '2024-01-20T09:55:00Z',
        violations_detected: 5,
        current_focus: 'implementation_quality',
        witnesses: [
          { id: 'w_1', name: 'CodeQuality', status: 'active', last_report: '2024-01-20T09:55:00Z' },
          { id: 'w_2', name: 'Security', status: 'investigating', last_report: '2024-01-20T09:50:00Z' },
        ],
      };
      mockClient.get.mockResolvedValue(mockStatus);

      const result = await api.getWitnessStatus();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/nomic/witness/status');
      expect(result.active).toBe(true);
      expect(result.witnesses).toHaveLength(2);
    });

    it('should get mayor info', async () => {
      const mockMayor = {
        current_mayor: 'claude',
        elected_at: '2024-01-01T00:00:00Z',
        term_ends: '2024-04-01T00:00:00Z',
        approval_rating: 0.85,
        policies_enacted: 12,
        vetoes: 2,
        emergency_powers_active: false,
      };
      mockClient.get.mockResolvedValue(mockMayor);

      const result = await api.getMayorCurrent();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/nomic/mayor/current');
      expect(result.current_mayor).toBe('claude');
      expect(result.approval_rating).toBe(0.85);
    });
  });

  // ===========================================================================
  // Convenience Methods
  // ===========================================================================

  describe('Convenience Methods', () => {
    it('should use state alias', async () => {
      const mockState = { running: true, cycle: 5, phase: 'implement', paused: false };
      mockClient.get.mockResolvedValue(mockState);

      const result = await api.state();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/nomic/state');
      expect(result.running).toBe(true);
    });

    it('should use health alias', async () => {
      const mockHealth = { status: 'healthy', cycle: 5 };
      mockClient.get.mockResolvedValue(mockHealth);

      const result = await api.health();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/nomic/health');
      expect(result.status).toBe('healthy');
    });

    it('should check if running', async () => {
      mockClient.get.mockResolvedValue({ running: true, paused: false });

      const result = await api.isRunning();

      expect(result).toBe(true);
    });

    it('should return false when paused', async () => {
      mockClient.get.mockResolvedValue({ running: true, paused: true });

      const result = await api.isRunning();

      expect(result).toBe(false);
    });

    it('should return false when not running', async () => {
      mockClient.get.mockResolvedValue({ running: false, paused: false });

      const result = await api.isRunning();

      expect(result).toBe(false);
    });
  });
});
