/**
 * Analytics Namespace Tests
 *
 * Comprehensive tests for the analytics namespace API including:
 * - Dashboard analytics
 * - Debate analytics
 * - Agent analytics
 * - Usage analytics
 * - Export functionality
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { AnalyticsNamespace } from '../analytics';

interface MockClient {
  request: Mock;
}

describe('AnalyticsNamespace', () => {
  let api: AnalyticsNamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new AnalyticsNamespace(mockClient as any);
  });

  // ===========================================================================
  // Dashboard Analytics
  // ===========================================================================

  describe('Dashboard Analytics', () => {
    it('should get dashboard overview', async () => {
      const mockDashboard = {
        total_debates: 1500,
        active_debates: 25,
        total_decisions: 1200,
        consensus_rate: 0.85,
        average_debate_duration: 180,
        top_agents: [
          { agent: 'claude', debates: 500, win_rate: 0.75 },
          { agent: 'gpt4', debates: 450, win_rate: 0.72 },
        ],
        recent_activity: [
          { date: '2024-01-20', debates: 15, decisions: 12 },
          { date: '2024-01-19', debates: 18, decisions: 15 },
        ],
      };
      mockClient.request.mockResolvedValue(mockDashboard);

      const result = await api.getDashboard();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/dashboard', {
        params: undefined,
      });
      expect(result.total_debates).toBe(1500);
      expect(result.consensus_rate).toBe(0.85);
    });

    it('should get dashboard with date range', async () => {
      const mockDashboard = { total_debates: 100 };
      mockClient.request.mockResolvedValue(mockDashboard);

      await api.getDashboard({
        start_date: '2024-01-01',
        end_date: '2024-01-31',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/dashboard', {
        params: { start_date: '2024-01-01', end_date: '2024-01-31' },
      });
    });

    it('should get dashboard with tenant filter', async () => {
      const mockDashboard = { total_debates: 50 };
      mockClient.request.mockResolvedValue(mockDashboard);

      await api.getDashboard({ tenant_id: 't123' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/dashboard', {
        params: { tenant_id: 't123' },
      });
    });
  });

  // ===========================================================================
  // Debate Analytics
  // ===========================================================================

  describe('Debate Analytics', () => {
    it('should get debate analytics', async () => {
      const mockAnalytics = {
        total_debates: 500,
        by_status: { completed: 450, in_progress: 30, failed: 20 },
        by_outcome: { consensus: 400, majority: 40, no_consensus: 10 },
        average_rounds: 3.5,
        average_duration_seconds: 240,
        average_agents_per_debate: 4.2,
        trend: [
          { date: '2024-01-20', count: 15 },
          { date: '2024-01-19', count: 18 },
        ],
      };
      mockClient.request.mockResolvedValue(mockAnalytics);

      const result = await api.getDebateAnalytics();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/debates', {
        params: undefined,
      });
      expect(result.total_debates).toBe(500);
      expect(result.by_outcome.consensus).toBe(400);
    });

    it('should get debate analytics with filters', async () => {
      const mockAnalytics = { total_debates: 100 };
      mockClient.request.mockResolvedValue(mockAnalytics);

      await api.getDebateAnalytics({
        start_date: '2024-01-01',
        end_date: '2024-01-31',
        group_by: 'day',
        domain: 'technology',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/debates', {
        params: {
          start_date: '2024-01-01',
          end_date: '2024-01-31',
          group_by: 'day',
          domain: 'technology',
        },
      });
    });

    it('should get debate trends', async () => {
      const mockTrends = {
        period: 'week',
        data: [
          { date: '2024-01-14', debates: 100, decisions: 85 },
          { date: '2024-01-21', debates: 120, decisions: 100 },
        ],
        growth_rate: 0.20,
      };
      mockClient.request.mockResolvedValue(mockTrends);

      const result = await api.getDebateTrends('week');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/debates/trends', {
        params: { period: 'week' },
      });
      expect(result.growth_rate).toBe(0.20);
    });
  });

  // ===========================================================================
  // Agent Analytics
  // ===========================================================================

  describe('Agent Analytics', () => {
    it('should get agent analytics', async () => {
      const mockAnalytics = {
        total_agents: 15,
        active_agents: 12,
        agents: [
          {
            agent: 'claude',
            total_debates: 500,
            wins: 375,
            losses: 100,
            draws: 25,
            win_rate: 0.75,
            elo: 1850,
            average_response_time: 2.5,
          },
          {
            agent: 'gpt4',
            total_debates: 450,
            wins: 324,
            losses: 90,
            draws: 36,
            win_rate: 0.72,
            elo: 1800,
            average_response_time: 3.0,
          },
        ],
      };
      mockClient.request.mockResolvedValue(mockAnalytics);

      const result = await api.getAgentAnalytics();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/agents', {
        params: undefined,
      });
      expect(result.agents).toHaveLength(2);
      expect(result.agents[0].win_rate).toBe(0.75);
    });

    it('should get specific agent analytics', async () => {
      const mockAnalytics = {
        agent: 'claude',
        total_debates: 500,
        performance_by_domain: {
          technology: { debates: 200, win_rate: 0.80 },
          science: { debates: 150, win_rate: 0.75 },
        },
        recent_debates: [
          { id: 'd1', topic: 'AI Ethics', outcome: 'win' },
        ],
      };
      mockClient.request.mockResolvedValue(mockAnalytics);

      const result = await api.getAgentAnalytics({ agent: 'claude' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/agents', {
        params: { agent: 'claude' },
      });
      expect(result.performance_by_domain.technology.win_rate).toBe(0.80);
    });

    it('should get agent comparison', async () => {
      const mockComparison = {
        agents: ['claude', 'gpt4'],
        metrics: {
          win_rate: { claude: 0.75, gpt4: 0.72 },
          elo: { claude: 1850, gpt4: 1800 },
          response_time: { claude: 2.5, gpt4: 3.0 },
        },
        head_to_head: { claude_wins: 60, gpt4_wins: 40 },
      };
      mockClient.request.mockResolvedValue(mockComparison);

      const result = await api.compareAgents(['claude', 'gpt4']);

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/analytics/agents/compare', {
        json: { agents: ['claude', 'gpt4'] },
      });
      expect(result.head_to_head.claude_wins).toBe(60);
    });
  });

  // ===========================================================================
  // Usage Analytics
  // ===========================================================================

  describe('Usage Analytics', () => {
    it('should get usage analytics', async () => {
      const mockUsage = {
        period: { start: '2024-01-01', end: '2024-01-31' },
        api_calls: 50000,
        debates_created: 500,
        tokens_used: 2500000,
        storage_used_mb: 250,
        by_endpoint: {
          '/api/v1/debates': 20000,
          '/api/v1/agents': 15000,
        },
      };
      mockClient.request.mockResolvedValue(mockUsage);

      const result = await api.getUsageAnalytics();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/usage', {
        params: undefined,
      });
      expect(result.api_calls).toBe(50000);
      expect(result.tokens_used).toBe(2500000);
    });

    it('should get usage by tenant', async () => {
      const mockUsage = { api_calls: 5000, tenant_id: 't123' };
      mockClient.request.mockResolvedValue(mockUsage);

      await api.getUsageAnalytics({ tenant_id: 't123' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/usage', {
        params: { tenant_id: 't123' },
      });
    });

    it('should get cost breakdown', async () => {
      const mockCosts = {
        total_cost: 150.00,
        by_provider: {
          anthropic: 80.00,
          openai: 50.00,
          mistral: 20.00,
        },
        by_model: {
          'claude-3-opus': 60.00,
          'gpt-4': 40.00,
        },
      };
      mockClient.request.mockResolvedValue(mockCosts);

      const result = await api.getCostBreakdown();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/costs', {
        params: undefined,
      });
      expect(result.total_cost).toBe(150.00);
    });
  });

  // ===========================================================================
  // Export Functionality
  // ===========================================================================

  describe('Export Functionality', () => {
    it('should export analytics as JSON', async () => {
      const mockExport = {
        format: 'json',
        data: { debates: [], agents: [] },
        generated_at: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockExport);

      const result = await api.exportAnalytics('json');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/export', {
        params: { format: 'json' },
      });
      expect(result.format).toBe('json');
    });

    it('should export analytics as CSV', async () => {
      const mockExport = { format: 'csv', download_url: 'https://...' };
      mockClient.request.mockResolvedValue(mockExport);

      const result = await api.exportAnalytics('csv', {
        start_date: '2024-01-01',
        end_date: '2024-01-31',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/export', {
        params: { format: 'csv', start_date: '2024-01-01', end_date: '2024-01-31' },
      });
    });

    it('should schedule report generation', async () => {
      const mockSchedule = {
        report_id: 'rpt_123',
        status: 'scheduled',
        estimated_completion: '2024-01-20T10:05:00Z',
      };
      mockClient.request.mockResolvedValue(mockSchedule);

      const result = await api.scheduleReport({
        type: 'monthly_summary',
        format: 'pdf',
        recipients: ['admin@example.com'],
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/analytics/reports/schedule', {
        json: {
          type: 'monthly_summary',
          format: 'pdf',
          recipients: ['admin@example.com'],
        },
      });
      expect(result.report_id).toBe('rpt_123');
    });
  });

  // ===========================================================================
  // Real-time Analytics
  // ===========================================================================

  describe('Real-time Analytics', () => {
    it('should get real-time metrics', async () => {
      const mockRealtime = {
        active_debates: 15,
        active_users: 42,
        requests_per_minute: 120,
        average_latency_ms: 85,
        error_rate: 0.002,
        timestamp: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockRealtime);

      const result = await api.getRealtime();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/analytics/realtime');
      expect(result.active_debates).toBe(15);
      expect(result.error_rate).toBe(0.002);
    });
  });
});
