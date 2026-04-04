import { renderHook } from '@testing-library/react';

import { useDecisionIntegrity } from '@/hooks/useDecisionIntegrity';
import { useSWRFetch } from '@/hooks/useSWRFetch';

jest.mock('@/hooks/useSWRFetch', () => ({
  useSWRFetch: jest.fn(),
}));

const mockUseSWRFetch = useSWRFetch as jest.Mock;

describe('useDecisionIntegrity', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseSWRFetch.mockImplementation((endpoint: string | null) => {
      if (endpoint === '/api/v2/compliance/status') {
        return {
          data: {
            data: {
              status: 'partially_compliant',
              compliance_score: 82,
              frameworks: {
                soc2_type2: {
                  status: 'compliant',
                  controls_assessed: 10,
                  controls_compliant: 10,
                },
                gdpr: {
                  status: 'supported',
                },
              },
              controls_summary: {
                total: 12,
                compliant: 10,
                non_compliant: 2,
              },
              last_audit: '2026-03-31T21:00:00Z',
            },
          },
          error: null,
          isLoading: false,
          isValidating: false,
          mutate: jest.fn(),
        };
      }

      if (endpoint === '/api/v2/memory/stats') {
        return {
          data: {
            total_entries: 144,
            memory_pressure: 0.42,
            status: 'normal',
            tiers: {
              fast: { count: 12, limit: 100, utilization: 0.12 },
            },
          },
          error: null,
          isLoading: false,
          isValidating: false,
          mutate: jest.fn(),
        };
      }

      if (endpoint === '/api/v2/receipts/stats') {
        return {
          data: {
            total: 12,
            verified: 9,
            by_verdict: { APPROVED: 8, REJECTED: 4 },
            by_risk_level: { LOW: 10, HIGH: 2 },
          },
          error: null,
          isLoading: false,
          isValidating: false,
          mutate: jest.fn(),
        };
      }

      if (endpoint === '/api/v2/receipts/deliveries?limit=20') {
        return {
          data: {
            deliveries: [
              {
                receiptId: 'rcpt-1',
                status: 'success',
                deliveredAt: '2026-03-31T20:00:00Z',
                channel: 'slack',
              },
              {
                receiptId: 'rcpt-2',
                status: 'failed',
                deliveredAt: '2026-03-31T19:00:00Z',
                channel: 'email',
              },
              {
                receiptId: 'rcpt-3',
                status: 'pending',
                deliveredAt: '2026-03-31T18:00:00Z',
                channel: 'teams',
              },
            ],
          },
          error: null,
          isLoading: false,
          isValidating: false,
          mutate: jest.fn(),
        };
      }

      if (endpoint === '/api/v2/compliance/audit-events?limit=20') {
        return {
          data: {
            data: {
              entries: [
                {
                  id: 'audit-1',
                  timestamp: '2026-03-31T20:30:00Z',
                  event_type: 'receipt.generated',
                  actor: 'system',
                  resource: 'rcpt-1',
                  action: 'generated',
                  outcome: 'success',
                  details: 'Receipt generated',
                },
              ],
              total: 1,
            },
          },
          error: null,
          isLoading: false,
          isValidating: false,
          mutate: jest.fn(),
        };
      }

      if (endpoint === '/api/v2/agents/leaderboard') {
        return {
          data: {
            leaderboard: [
              {
                name: 'claude',
                elo: 1735,
                wins: 5,
                losses: 2,
                matches: 7,
                win_rate: 0.714,
              },
            ],
          },
          error: null,
          isLoading: false,
          isValidating: false,
          mutate: jest.fn(),
        };
      }

      return {
        data: null,
        error: null,
        isLoading: false,
        isValidating: false,
        mutate: jest.fn(),
      };
    });
  });

  it('uses the canonical v2 routes and normalizes mixed backend shapes for the UI', () => {
    const { result } = renderHook(() => useDecisionIntegrity());

    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/debates?status=active',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/consensus/stats',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/compliance/status',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/memory/stats',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/receipts/stats',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/receipts/deliveries?limit=20',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/compliance/audit-events?limit=20',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/agents/leaderboard',
      expect.objectContaining({ enabled: true, refreshInterval: 60_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/consensus/settled?limit=10',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );

    expect(result.current.compliance).toEqual({
      status: 'partially_compliant',
      overall_score: 0.82,
      violations_count: 2,
      frameworks: [
        {
          name: 'SOC 2 Type 2',
          status: 'compliant',
          score: 1,
          last_assessed: '2026-03-31T21:00:00Z',
        },
        {
          name: 'GDPR',
          status: 'supported',
          score: undefined,
          last_assessed: '2026-03-31T21:00:00Z',
        },
      ],
      findings: [],
    });
    expect(result.current.memory).toMatchObject({
      total_entries: 144,
      memory_pressure: 0.42,
      status: 'normal',
    });
    expect(result.current.receipts).toEqual({
      total_receipts: 12,
      verified_count: 9,
      delivered: 1,
      pending: 1,
      failed: 1,
      delivery_rate: 0.5,
      by_verdict: { APPROVED: 8, REJECTED: 4 },
      by_risk_level: { LOW: 10, HIGH: 2 },
      generated_at: undefined,
      recent: [
        {
          id: 'rcpt-1',
          status: 'delivered',
          created_at: '2026-03-31T20:00:00Z',
          delivered_at: '2026-03-31T20:00:00Z',
          channel: 'slack',
        },
        {
          id: 'rcpt-2',
          status: 'failed',
          created_at: '2026-03-31T19:00:00Z',
          delivered_at: '2026-03-31T19:00:00Z',
          channel: 'email',
        },
        {
          id: 'rcpt-3',
          status: 'pending',
          created_at: '2026-03-31T18:00:00Z',
          delivered_at: '2026-03-31T18:00:00Z',
          channel: 'teams',
        },
      ],
    });
    expect(result.current.audit).toEqual({
      events: [
        {
          id: 'audit-1',
          timestamp: '2026-03-31T20:30:00Z',
          event_type: 'receipt.generated',
          actor: 'system',
          resource: 'rcpt-1',
          action: 'generated',
          details: 'Receipt generated',
          severity: 'info',
        },
      ],
      total: 1,
    });
    expect(result.current.leaderboard).toEqual({
      leaderboard: [
        {
          name: 'claude',
          elo: 1735,
          wins: 5,
          losses: 2,
          matches: 7,
          win_rate: 0.714,
          debates_participated: 7,
        },
      ],
    });
    expect(result.current.metrics).toMatchObject({
      complianceScore: 82,
      memoryPressure: 42,
      receiptDeliveryRate: 50,
    });
  });
});
