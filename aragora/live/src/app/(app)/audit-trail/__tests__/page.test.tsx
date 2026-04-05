import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import AuditTrailPage from '../page';
import { useSWRFetch } from '@/hooks/useSWRFetch';
import { apiPost } from '@/lib/api';

jest.mock('next/link', () => ({
  __esModule: true,
  default: ({ href, children }: { href: string; children: React.ReactNode }) => (
    <a href={href}>{children}</a>
  ),
}));

jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => null,
  CRTVignette: () => null,
}));

jest.mock('@/components/PanelErrorBoundary', () => ({
  PanelErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/hooks/useSWRFetch', () => ({
  useSWRFetch: jest.fn(),
}));

jest.mock('@/lib/api', () => ({
  apiPost: jest.fn(),
}));

const mockUseSWRFetch = useSWRFetch as jest.Mock;
const mockApiPost = apiPost as jest.Mock;

describe('AuditTrailPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockUseSWRFetch.mockImplementation((endpoint: string | null) => {
      if (!endpoint) {
        return { data: null, error: null, isLoading: false };
      }

      if (endpoint.startsWith('/api/v1/audit-trails?')) {
        return {
          data: {
            trails: [
              {
                trail_id: 'trail-123',
                gauntlet_id: 'gauntlet-123',
                created_at: '2026-03-25T00:00:00Z',
                verdict: 'approved',
                confidence: 0.92,
                total_findings: 3,
                duration_seconds: 12.4,
                checksum: 'trail-checksum-123',
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

      if (endpoint.startsWith('/api/v1/receipts?')) {
        return {
          data: {
            receipts: [
              {
                receipt_id: 'receipt-123',
                gauntlet_id: 'gauntlet-123',
                timestamp: '2026-03-25T00:00:00Z',
                verdict: 'approved',
                confidence: 0.88,
                risk_level: 'low',
                findings_count: 2,
                checksum: 'receipt-checksum-123',
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

      return { data: null, error: null, isLoading: false };
    });
  });

  it('verifies audit trails through apiPost', async () => {
    const user = userEvent.setup();
    mockApiPost.mockResolvedValue({
      trail_id: 'trail-123',
      valid: true,
      stored_checksum: 'trail-checksum-123',
      computed_checksum: 'trail-checksum-123',
      match: true,
    });

    render(<AuditTrailPage />);

    await user.click(screen.getByRole('button', { name: 'VERIFY' }));

    await waitFor(() => {
      expect(mockApiPost).toHaveBeenCalledWith('/api/v1/audit-trails/trail-123/verify');
    });

    expect(await screen.findByText('[VALID]')).toBeInTheDocument();
  });

  it('verifies decision receipts through apiPost', async () => {
    const user = userEvent.setup();
    mockApiPost.mockResolvedValue({
      receipt_id: 'receipt-123',
      valid: true,
      stored_checksum: 'receipt-checksum-123',
      computed_checksum: 'receipt-checksum-123',
      match: true,
    });

    render(<AuditTrailPage />);

    await user.click(screen.getByRole('button', { name: /decision receipts/i }));
    await user.click(screen.getByRole('button', { name: 'VERIFY' }));

    await waitFor(() => {
      expect(mockApiPost).toHaveBeenCalledWith('/api/v1/receipts/receipt-123/verify');
    });

    expect(await screen.findByText('[VALID]')).toBeInTheDocument();
  });
});
