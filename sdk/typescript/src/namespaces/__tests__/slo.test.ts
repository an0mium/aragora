/**
 * SLO Namespace Tests
 *
 * Comprehensive tests for the SLO namespace API including:
 * - Overall SLO status
 * - Individual SLO status
 * - Error budgets
 * - Violations
 * - Alerts
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { SLONamespace } from '../slo';

interface MockClient {
  request: Mock;
}

describe('SLONamespace', () => {
  let api: SLONamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new SLONamespace(mockClient as any);
  });

  // ===========================================================================
  // Overall Status
  // ===========================================================================

  describe('Overall Status', () => {
    it('should get overall SLO status', async () => {
      const mockStatus = {
        status: 'healthy',
        timestamp: '2024-01-20T10:00:00Z',
        slos: {
          availability: {
            name: 'availability',
            current_percent: 99.95,
            target_percent: 99.9,
            is_meeting: true,
            window_start: '2024-01-13T00:00:00Z',
            window_end: '2024-01-20T00:00:00Z',
            total_requests: 1000000,
            successful_requests: 999500,
            failed_requests: 500,
          },
          latency: {
            name: 'latency',
            current_percent: 99.5,
            target_percent: 99.0,
            is_meeting: true,
            window_start: '2024-01-13T00:00:00Z',
            window_end: '2024-01-20T00:00:00Z',
            total_requests: 1000000,
            successful_requests: 995000,
            failed_requests: 5000,
          },
        },
        alerts: [],
        summary: {
          total: 2,
          meeting: 2,
          not_meeting: 0,
          compliance_percent: 100,
        },
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/slos/status');
      expect(result.status).toBe('healthy');
      expect(result.summary.compliance_percent).toBe(100);
    });

    it('should get degraded status', async () => {
      const mockStatus = {
        status: 'degraded',
        timestamp: '2024-01-20T10:00:00Z',
        slos: {
          availability: {
            name: 'availability',
            current_percent: 99.8,
            target_percent: 99.9,
            is_meeting: false,
            window_start: '2024-01-13T00:00:00Z',
            window_end: '2024-01-20T00:00:00Z',
            total_requests: 1000000,
            successful_requests: 998000,
            failed_requests: 2000,
          },
        },
        alerts: [
          {
            slo_name: 'availability',
            severity: 'warning',
            message: 'Availability SLO not meeting target',
            triggered_at: '2024-01-20T09:30:00Z',
            acknowledged: false,
          },
        ],
        summary: {
          total: 2,
          meeting: 1,
          not_meeting: 1,
          compliance_percent: 50,
        },
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getStatus();

      expect(result.status).toBe('degraded');
      expect(result.alerts).toHaveLength(1);
    });
  });

  // ===========================================================================
  // Individual SLOs
  // ===========================================================================

  describe('Individual SLOs', () => {
    it('should get specific SLO', async () => {
      const mockSLO = {
        name: 'availability',
        current_percent: 99.95,
        target_percent: 99.9,
        is_meeting: true,
        window_start: '2024-01-13T00:00:00Z',
        window_end: '2024-01-20T00:00:00Z',
        total_requests: 1000000,
        successful_requests: 999500,
        failed_requests: 500,
      };
      mockClient.request.mockResolvedValue(mockSLO);

      const result = await api.getSLO('availability');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/slos/availability');
      expect(result.name).toBe('availability');
      expect(result.is_meeting).toBe(true);
    });

    it('should get availability SLO', async () => {
      const mockSLO = {
        name: 'availability',
        current_percent: 99.95,
        target_percent: 99.9,
        is_meeting: true,
      };
      mockClient.request.mockResolvedValue(mockSLO);

      const result = await api.getAvailability();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/slos/availability');
      expect(result.name).toBe('availability');
    });

    it('should get latency SLO', async () => {
      const mockSLO = {
        name: 'latency',
        current_percent: 99.5,
        target_percent: 99.0,
        is_meeting: true,
      };
      mockClient.request.mockResolvedValue(mockSLO);

      const result = await api.getLatency();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/slos/latency');
      expect(result.name).toBe('latency');
    });

    it('should get debate success SLO', async () => {
      const mockSLO = {
        name: 'debate-success',
        current_percent: 98.5,
        target_percent: 95.0,
        is_meeting: true,
      };
      mockClient.request.mockResolvedValue(mockSLO);

      const result = await api.getDebateSuccess();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/slos/debate-success');
      expect(result.name).toBe('debate-success');
    });
  });

  // ===========================================================================
  // Error Budgets
  // ===========================================================================

  describe('Error Budgets', () => {
    it('should get all error budgets', async () => {
      const mockBudgets = [
        {
          slo_name: 'availability',
          budget_percent: 0.1,
          consumed_percent: 0.05,
          remaining_percent: 0.05,
          is_exhausted: false,
          burn_rate: 1.2,
          window_days: 7,
        },
        {
          slo_name: 'latency',
          budget_percent: 1.0,
          consumed_percent: 0.5,
          remaining_percent: 0.5,
          is_exhausted: false,
          burn_rate: 0.8,
          window_days: 7,
        },
      ];
      mockClient.request.mockResolvedValue(mockBudgets);

      const result = await api.getErrorBudget();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/slos/error-budget');
      expect(result).toHaveLength(2);
    });

    it('should get specific error budget', async () => {
      const mockBudget = {
        slo_name: 'availability',
        budget_percent: 0.1,
        consumed_percent: 0.08,
        remaining_percent: 0.02,
        is_exhausted: false,
        burn_rate: 2.5,
        projected_exhaustion: '2024-01-22T00:00:00Z',
        window_days: 7,
      };
      mockClient.request.mockResolvedValue(mockBudget);

      const result = await api.getErrorBudget('availability');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/slos/error-budget?slo=availability');
      expect((result as any).slo_name).toBe('availability');
      expect((result as any).burn_rate).toBe(2.5);
    });

    it('should show exhausted error budget', async () => {
      const mockBudget = {
        slo_name: 'availability',
        budget_percent: 0.1,
        consumed_percent: 0.15,
        remaining_percent: 0,
        is_exhausted: true,
        burn_rate: 5.0,
        window_days: 7,
      };
      mockClient.request.mockResolvedValue(mockBudget);

      const result = await api.getErrorBudget('availability');

      expect((result as any).is_exhausted).toBe(true);
      expect((result as any).remaining_percent).toBe(0);
    });
  });

  // ===========================================================================
  // Violations
  // ===========================================================================

  describe('Violations', () => {
    it('should get violations with default limit', async () => {
      const mockViolations = [
        {
          slo_name: 'availability',
          timestamp: '2024-01-20T09:00:00Z',
          actual_percent: 99.7,
          target_percent: 99.9,
          duration_seconds: 300,
          severity: 'warning',
          resolved: true,
          resolved_at: '2024-01-20T09:05:00Z',
        },
      ];
      mockClient.request.mockResolvedValue(mockViolations);

      const result = await api.getViolations();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/slos/violations');
      expect(result).toHaveLength(1);
    });

    it('should get violations with filters', async () => {
      const mockViolations = [
        {
          slo_name: 'availability',
          timestamp: '2024-01-20T08:00:00Z',
          actual_percent: 99.5,
          target_percent: 99.9,
          duration_seconds: 600,
          severity: 'critical',
          resolved: false,
        },
      ];
      mockClient.request.mockResolvedValue(mockViolations);

      const result = await api.getViolations({
        limit: 10,
        sloName: 'availability',
        severity: 'critical',
        unresolved: true,
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/slos/violations?limit=10&slo=availability&severity=critical&unresolved=true'
      );
      expect(result[0].severity).toBe('critical');
      expect(result[0].resolved).toBe(false);
    });
  });

  // ===========================================================================
  // Targets
  // ===========================================================================

  describe('Targets', () => {
    it('should get SLO targets', async () => {
      const mockTargets = [
        {
          name: 'availability',
          target_percent: 99.9,
          window_days: 7,
          description: 'Service availability target',
        },
        {
          name: 'latency',
          target_percent: 99.0,
          window_days: 7,
          description: 'P95 latency under 500ms',
        },
        {
          name: 'debate-success',
          target_percent: 95.0,
          window_days: 30,
          description: 'Debate completion rate',
        },
      ];
      mockClient.request.mockResolvedValue(mockTargets);

      const result = await api.getTargets();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/slos/targets');
      expect(result).toHaveLength(3);
      expect(result[0].target_percent).toBe(99.9);
    });
  });

  // ===========================================================================
  // Compliance Check
  // ===========================================================================

  describe('Compliance Check', () => {
    it('should return true when all SLOs are meeting targets', async () => {
      const mockStatus = {
        status: 'healthy',
        timestamp: '2024-01-20T10:00:00Z',
        slos: {},
        alerts: [],
        summary: { total: 3, meeting: 3, not_meeting: 0, compliance_percent: 100 },
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.isCompliant();

      expect(result).toBe(true);
    });

    it('should return false when SLOs are degraded', async () => {
      const mockStatus = {
        status: 'degraded',
        timestamp: '2024-01-20T10:00:00Z',
        slos: {},
        alerts: [],
        summary: { total: 3, meeting: 2, not_meeting: 1, compliance_percent: 66.7 },
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.isCompliant();

      expect(result).toBe(false);
    });

    it('should return false when SLOs are critical', async () => {
      const mockStatus = {
        status: 'critical',
        timestamp: '2024-01-20T10:00:00Z',
        slos: {},
        alerts: [],
        summary: { total: 3, meeting: 0, not_meeting: 3, compliance_percent: 0 },
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.isCompliant();

      expect(result).toBe(false);
    });
  });

  // ===========================================================================
  // Alerts
  // ===========================================================================

  describe('Alerts', () => {
    it('should get current alerts', async () => {
      const mockStatus = {
        status: 'degraded',
        timestamp: '2024-01-20T10:00:00Z',
        slos: {},
        alerts: [
          {
            slo_name: 'availability',
            severity: 'warning',
            message: 'Availability below target',
            triggered_at: '2024-01-20T09:30:00Z',
            acknowledged: false,
          },
          {
            slo_name: 'latency',
            severity: 'critical',
            message: 'Latency significantly elevated',
            triggered_at: '2024-01-20T09:45:00Z',
            acknowledged: true,
          },
        ],
        summary: { total: 2, meeting: 0, not_meeting: 2, compliance_percent: 0 },
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getAlerts();

      expect(result).toHaveLength(2);
      expect(result[0].severity).toBe('warning');
      expect(result[1].acknowledged).toBe(true);
    });

    it('should return empty alerts when healthy', async () => {
      const mockStatus = {
        status: 'healthy',
        timestamp: '2024-01-20T10:00:00Z',
        slos: {},
        alerts: [],
        summary: { total: 2, meeting: 2, not_meeting: 0, compliance_percent: 100 },
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getAlerts();

      expect(result).toHaveLength(0);
    });
  });
});
