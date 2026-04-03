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
      if (endpoint === '/api/v1/consensus/stats') {
        return {
          data: {
            total_topics: 5,
            high_confidence_count: 4,
            avg_confidence: 0.83,
            total_dissents: 2,
            by_strength: { strong: 3 },
            by_domain: { product: 2, engineering: 3 },
          },
          error: null,
          isLoading: false,
          isValidating: false,
          mutate: jest.fn(),
        };
      }

      if (endpoint === '/api/v2/compliance/status') {
        return {
          data: {
            data: {
              status: 'partial',
              compliance_score: 87,
              frameworks: {
                soc2_type2: {
                  status: 'in_progress',
                  controls_assessed: 3,
                  controls_compliant: 1,
                },
                gdpr: {
                  status: 'supported',
                  data_export: true,
                  consent_tracking: true,
                  retention_policy: false,
                },
                hipaa: {
                  status: 'partial',
                  note: 'PHI handling requires additional configuration',
                },
              },
              controls_summary: {
                total: 10,
                compliant: 6,
                non_compliant: 4,
              },
              last_audit: '2026-03-30T00:00:00Z',
              next_audit_due: '2026-06-30T00:00:00Z',
              generated_at: '2026-03-31T00:00:00Z',
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

      if (endpoint === '/api/v1/receipts/deliveries?limit=20') {
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

      return {
        data: null,
        error: null,
        isLoading: false,
        isValidating: false,
        mutate: jest.fn(),
      };
    });
  });

  it('uses the live consensus/compliance routes and normalizes delivery history for the UI', () => {
    const { result } = renderHook(() => useDecisionIntegrity());

    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v1/consensus/stats',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/compliance/status',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/receipts/stats',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v1/receipts/deliveries?limit=20',
      expect.objectContaining({ enabled: true, refreshInterval: 30_000 }),
    );

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

    expect(result.current.consensus).toEqual(
      expect.objectContaining({
        avg_confidence: 0.83,
        total_topics: 5,
      }),
    );
    expect(result.current.compliance).toEqual({
      status: 'partial',
      overall_score: 0.87,
      frameworks: [
        {
          name: 'SOC 2 Type II',
          status: 'partial',
          score: 1 / 3,
          last_assessed: '2026-03-30T00:00:00Z',
        },
        {
          name: 'GDPR',
          status: 'compliant',
          score: 2 / 3,
          last_assessed: '2026-03-30T00:00:00Z',
        },
        {
          name: 'HIPAA',
          status: 'partial',
          last_assessed: '2026-03-30T00:00:00Z',
        },
      ],
      findings: [],
    });
    expect(result.current.metrics).toEqual(
      expect.objectContaining({
        consensusHealth: 83,
        complianceScore: 87,
      }),
    );
  });
});
