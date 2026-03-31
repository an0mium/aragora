import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import AuditTrailPage from '../(app)/audit-trail/page';
import { useSWRFetch } from '@/hooks/useSWRFetch';

const mockFetch = jest.fn();

global.fetch = mockFetch as typeof fetch;

jest.mock('next/link', () => {
  return function MockLink({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) {
    return <a href={href}>{children}</a>;
  };
});

jest.mock('@/components/BackendSelector', () => ({
  useBackend: () => ({
    config: {
      api: 'https://api.aragora.ai',
    },
  }),
}));

jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => <div data-testid="scanlines" />,
  CRTVignette: () => <div data-testid="crt-vignette" />,
}));

jest.mock('@/components/PanelErrorBoundary', () => ({
  PanelErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/hooks/useSWRFetch', () => ({
  useSWRFetch: jest.fn(),
}));

const mockUseSWRFetch = useSWRFetch as jest.MockedFunction<typeof useSWRFetch>;

function jsonResponse(data: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => data,
    text: async () => JSON.stringify(data),
  } as Response;
}

describe('runtime backend selection for audit trail integrity actions', () => {
  beforeEach(() => {
    mockFetch.mockReset();
    mockUseSWRFetch.mockImplementation((endpoint) => {
      if (endpoint?.startsWith('/api/v1/audit-trails?')) {
        return {
          data: {
            trails: [
              {
                trail_id: 'trail-123',
                gauntlet_id: 'gauntlet-123',
                created_at: '2026-03-31T00:00:00Z',
                verdict: 'approved',
                confidence: 0.91,
                total_findings: 2,
                duration_seconds: 12.4,
                checksum: 'sha256-trail',
              },
            ],
            total: 1,
            limit: 20,
            offset: 0,
          },
          error: null,
          isLoading: false,
        };
      }

      if (endpoint?.startsWith('/api/v1/receipts?')) {
        return {
          data: {
            receipts: [
              {
                receipt_id: 'receipt-123',
                gauntlet_id: 'gauntlet-123',
                timestamp: '2026-03-31T00:00:00Z',
                verdict: 'approved',
                confidence: 0.95,
                risk_level: 'low',
                findings_count: 1,
                checksum: 'sha256-receipt',
              },
            ],
            total: 1,
            limit: 20,
            offset: 0,
          },
          error: null,
          isLoading: false,
        };
      }

      return {
        data: null,
        error: null,
        isLoading: false,
      };
    });
    mockFetch.mockResolvedValue(
      jsonResponse({
        valid: true,
        stored_checksum: 'sha256-stored',
        computed_checksum: 'sha256-computed',
        match: true,
      }),
    );
  });

  it('uses the selected backend when verifying audit trails', async () => {
    const user = userEvent.setup();

    render(<AuditTrailPage />);

    await user.click(await screen.findByRole('button', { name: 'VERIFY' }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.aragora.ai/api/v1/audit-trails/trail-123/verify',
        { method: 'POST' },
      );
    });
  });

  it('uses the selected backend when verifying decision receipts', async () => {
    const user = userEvent.setup();

    render(<AuditTrailPage />);

    await user.click(screen.getByRole('button', { name: '[DECISION RECEIPTS]' }));
    await user.click(await screen.findByRole('button', { name: 'VERIFY' }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.aragora.ai/api/v1/receipts/receipt-123/verify',
        { method: 'POST' },
      );
    });
  });
});
