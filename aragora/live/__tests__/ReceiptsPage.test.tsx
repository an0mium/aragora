/**
 * Tests for Receipts page — SWR-based receipt fetching, gauntlet fallback, filtering
 */
import { render, screen, waitFor, fireEvent } from '@testing-library/react';

// Mock next/link
jest.mock('next/link', () => {
  return ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  );
});

// Mock config
jest.mock('../src/config', () => ({
  API_BASE_URL: 'http://localhost:8080',
  WS_URL: 'ws://localhost:8765/ws',
}));

// Mock MatrixRain
jest.mock('../src/components/MatrixRain', () => ({
  Scanlines: () => null,
  CRTVignette: () => null,
}));

// Mock AsciiBanner
jest.mock('../src/components/AsciiBanner', () => ({
  AsciiBannerCompact: () => <div data-testid="ascii-banner">ARAGORA</div>,
}));

// Mock ThemeToggle
jest.mock('../src/components/ThemeToggle', () => ({
  ThemeToggle: () => <button>Theme</button>,
}));

// Mock BackendSelector
jest.mock('../src/components/BackendSelector', () => ({
  BackendSelector: () => <div>Backend</div>,
  useBackend: () => ({
    config: { api: 'http://localhost:8080', ws: 'ws://localhost:8765' },
  }),
}));

// Mock ErrorWithRetry
jest.mock('../src/components/ErrorWithRetry', () => ({
  ErrorWithRetry: ({ error, onRetry }: { error: string; onRetry: () => void }) => (
    <div data-testid="error-retry">
      <span>{error}</span>
      <button onClick={onRetry}>Retry</button>
    </div>
  ),
}));

// Mock DeliveryModal
jest.mock('../src/components/receipts', () => ({
  DeliveryModal: () => null,
}));

// Mock logger
jest.mock('../src/utils/logger', () => ({
  logger: { warn: jest.fn(), error: jest.fn(), info: jest.fn(), debug: jest.fn() },
}));

// Mock useSWRFetch
const mockUseSWRFetch = jest.fn();
jest.mock('../src/hooks/useSWRFetch', () => ({
  useSWRFetch: (...args: unknown[]) => mockUseSWRFetch(...args),
}));

import ReceiptsPage from '../src/app/(app)/receipts/page';

describe('ReceiptsPage', () => {
  const mockMutate = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    // Default: no data from either endpoint
    mockUseSWRFetch.mockReturnValue({
      data: null,
      error: null,
      isLoading: false,
      mutate: mockMutate,
    });
  });

  const sampleResult = (overrides: Record<string, unknown> = {}) => ({
    id: 'gauntlet-abc123456789',
    status: 'completed' as const,
    verdict: 'PASS' as const,
    confidence: 0.95,
    created_at: new Date().toISOString(),
    input_summary: 'Test microservices migration plan',
    risk_summary: { critical: 0, high: 1, medium: 2, low: 3 },
    vulnerabilities_found: 3,
    ...overrides,
  });

  it('renders page title', () => {
    render(<ReceiptsPage />);

    // H1 title and H2 list header both say "Decision Receipts"
    const heading = screen.getByRole('heading', { level: 1 });
    expect(heading.textContent).toBe('Decision Receipts');
    expect(screen.getByText(/Audit-ready records of every AI-debated decision/)).toBeInTheDocument();
  });

  it('shows loading state', () => {
    mockUseSWRFetch.mockReturnValue({
      data: null,
      error: null,
      isLoading: true,
      mutate: mockMutate,
    });

    render(<ReceiptsPage />);

    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('renders receipts from v2 endpoint', async () => {
    // First call: v2 receipts
    mockUseSWRFetch
      .mockReturnValueOnce({
        data: {
          receipts: [sampleResult({ input_summary: 'Migration plan review' })],
        },
        error: null,
        isLoading: false,
        mutate: mockMutate,
      })
      // Second call: gauntlet fallback (should not be fetched)
      .mockReturnValueOnce({
        data: null,
        error: null,
        isLoading: false,
        mutate: mockMutate,
      });

    render(<ReceiptsPage />);

    await waitFor(() => {
      expect(screen.getByText('Migration plan review')).toBeInTheDocument();
    });
  });

  it('calls useSWRFetch with correct v2 receipts endpoint', () => {
    render(<ReceiptsPage />);

    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v2/receipts?limit=50',
      expect.objectContaining({
        refreshInterval: 30000,
        baseUrl: 'http://localhost:8080',
      })
    );
  });

  it('shows verdict badges with correct colors', async () => {
    mockUseSWRFetch
      .mockReturnValueOnce({
        data: {
          receipts: [
            sampleResult({ id: 'r1', verdict: 'PASS' }),
            sampleResult({ id: 'r2', verdict: 'CONDITIONAL' }),
            sampleResult({ id: 'r3', verdict: 'FAIL' }),
          ],
        },
        error: null,
        isLoading: false,
        mutate: mockMutate,
      })
      .mockReturnValueOnce({
        data: null,
        error: null,
        isLoading: false,
        mutate: mockMutate,
      });

    render(<ReceiptsPage />);

    await waitFor(() => {
      // All three verdict badges should be shown (filter buttons + result badges)
      expect(screen.getAllByText('PASS').length).toBeGreaterThanOrEqual(2); // 1 filter + 1 badge
      expect(screen.getAllByText('CONDITIONAL').length).toBeGreaterThanOrEqual(2);
      expect(screen.getAllByText('FAIL').length).toBeGreaterThanOrEqual(2);
    });
  });

  it('shows risk summary counts', async () => {
    mockUseSWRFetch
      .mockReturnValueOnce({
        data: {
          receipts: [sampleResult({
            risk_summary: { critical: 2, high: 3, medium: 1, low: 0 },
          })],
        },
        error: null,
        isLoading: false,
        mutate: mockMutate,
      })
      .mockReturnValueOnce({
        data: null,
        error: null,
        isLoading: false,
        mutate: mockMutate,
      });

    render(<ReceiptsPage />);

    await waitFor(() => {
      expect(screen.getByText('C:2')).toBeInTheDocument();
      expect(screen.getByText('H:3')).toBeInTheDocument();
      expect(screen.getByText('M:1')).toBeInTheDocument();
    });
  });

  describe('filtering', () => {
    it('renders filter buttons', () => {
      render(<ReceiptsPage />);

      expect(screen.getByText('all')).toBeInTheDocument();
      expect(screen.getByText('PASS')).toBeInTheDocument();
      expect(screen.getByText('CONDITIONAL')).toBeInTheDocument();
      expect(screen.getByText('FAIL')).toBeInTheDocument();
    });

    it('filters results by verdict', async () => {
      mockUseSWRFetch
        .mockReturnValueOnce({
          data: {
            receipts: [
              sampleResult({ id: 'r1', verdict: 'PASS', input_summary: 'Pass result' }),
              sampleResult({ id: 'r2', verdict: 'FAIL', input_summary: 'Fail result' }),
            ],
          },
          error: null,
          isLoading: false,
          mutate: mockMutate,
        })
        .mockReturnValueOnce({
          data: null,
          error: null,
          isLoading: false,
          mutate: mockMutate,
        });

      render(<ReceiptsPage />);

      await waitFor(() => {
        expect(screen.getByText('Pass result')).toBeInTheDocument();
        expect(screen.getByText('Fail result')).toBeInTheDocument();
      });

      // Filter to PASS only - need the filter button, not the badge
      const passButtons = screen.getAllByText('PASS');
      // Click the filter button (one that's a standalone button, not inside a result)
      const filterButton = passButtons.find(el => el.closest('button')?.className.includes('border'));
      if (filterButton) {
        fireEvent.click(filterButton);
      }

      await waitFor(() => {
        expect(screen.getByText('Pass result')).toBeInTheDocument();
        expect(screen.queryByText('Fail result')).not.toBeInTheDocument();
      });
    });
  });

  it('shows empty state when no results', async () => {
    mockUseSWRFetch.mockReturnValue({
      data: { receipts: [] },
      error: null,
      isLoading: false,
      mutate: mockMutate,
    });

    render(<ReceiptsPage />);

    await waitFor(() => {
      expect(screen.getByText(/No decision receipts yet/)).toBeInTheDocument();
    });
  });

  it('shows error with retry button', async () => {
    mockUseSWRFetch
      .mockReturnValueOnce({
        data: null,
        error: new Error('Connection failed'),
        isLoading: false,
        mutate: mockMutate,
      })
      .mockReturnValueOnce({
        data: null,
        error: new Error('Connection failed'),
        isLoading: false,
        mutate: mockMutate,
      });

    render(<ReceiptsPage />);

    await waitFor(() => {
      expect(screen.getByTestId('error-retry')).toBeInTheDocument();
    });
  });

  it('truncates gauntlet ID display', async () => {
    mockUseSWRFetch
      .mockReturnValueOnce({
        data: {
          receipts: [sampleResult({ id: 'abcdef123456789xyz' })],
        },
        error: null,
        isLoading: false,
        mutate: mockMutate,
      })
      .mockReturnValueOnce({
        data: null,
        error: null,
        isLoading: false,
        mutate: mockMutate,
      });

    render(<ReceiptsPage />);

    await waitFor(() => {
      expect(screen.getByText('abcdef123456...')).toBeInTheDocument();
    });
  });

  it('disables click on non-completed results', async () => {
    mockUseSWRFetch
      .mockReturnValueOnce({
        data: {
          receipts: [sampleResult({ status: 'running', verdict: undefined })],
        },
        error: null,
        isLoading: false,
        mutate: mockMutate,
      })
      .mockReturnValueOnce({
        data: null,
        error: null,
        isLoading: false,
        mutate: mockMutate,
      });

    render(<ReceiptsPage />);

    await waitFor(() => {
      const resultButton = screen.getByText(/gauntlet-abc/).closest('button');
      expect(resultButton).toBeDisabled();
    });
  });

  it('handles data from results key (alternative response shape)', async () => {
    mockUseSWRFetch
      .mockReturnValueOnce({
        data: {
          results: [sampleResult({ input_summary: 'Alt shape result' })],
        },
        error: null,
        isLoading: false,
        mutate: mockMutate,
      })
      .mockReturnValueOnce({
        data: null,
        error: null,
        isLoading: false,
        mutate: mockMutate,
      });

    render(<ReceiptsPage />);

    await waitFor(() => {
      expect(screen.getByText('Alt shape result')).toBeInTheDocument();
    });
  });

  // ---------------------------------------------------------------------------
  // Empty state action links
  // ---------------------------------------------------------------------------

  it('shows Oracle and debate links in empty state', async () => {
    mockUseSWRFetch.mockReturnValue({
      data: { receipts: [] },
      error: null,
      isLoading: false,
      mutate: mockMutate,
    });

    render(<ReceiptsPage />);

    await waitFor(() => {
      expect(screen.getByText('Ask the Oracle')).toBeInTheDocument();
      expect(screen.getByText('Start a debate')).toBeInTheDocument();

      // Verify link destinations
      const oracleLink = screen.getByText('Ask the Oracle').closest('a');
      expect(oracleLink?.getAttribute('href')).toBe('/oracle');

      const debateLink = screen.getByText('Start a debate').closest('a');
      expect(debateLink?.getAttribute('href')).toBe('/debate');
    });
  });

  // ---------------------------------------------------------------------------
  // Gauntlet fallback endpoint
  // ---------------------------------------------------------------------------

  it('falls back to gauntlet endpoint when v2 receipts returns no data', async () => {
    mockUseSWRFetch
      // First call: v2 receipts returns null (no data)
      .mockReturnValueOnce({
        data: null,
        error: null,
        isLoading: false,
        mutate: mockMutate,
      })
      // Second call: gauntlet fallback returns data
      .mockReturnValueOnce({
        data: {
          results: [sampleResult({ input_summary: 'Gauntlet fallback result' })],
        },
        error: null,
        isLoading: false,
        mutate: mockMutate,
      });

    render(<ReceiptsPage />);

    await waitFor(() => {
      expect(screen.getByText('Gauntlet fallback result')).toBeInTheDocument();
    });
  });

  // ---------------------------------------------------------------------------
  // Date display
  // ---------------------------------------------------------------------------

  it('displays formatted date for each receipt', async () => {
    const fixedDate = '2026-02-15T10:30:00Z';
    mockUseSWRFetch
      .mockReturnValueOnce({
        data: {
          receipts: [sampleResult({ created_at: fixedDate })],
        },
        error: null,
        isLoading: false,
        mutate: mockMutate,
      })
      .mockReturnValueOnce({
        data: null,
        error: null,
        isLoading: false,
        mutate: mockMutate,
      });

    render(<ReceiptsPage />);

    await waitFor(() => {
      // The component calls new Date(created_at).toLocaleDateString()
      const expectedDate = new Date(fixedDate).toLocaleDateString();
      expect(screen.getByText(expectedDate)).toBeInTheDocument();
    });
  });

  // ---------------------------------------------------------------------------
  // Low risk items not shown when count is 0
  // ---------------------------------------------------------------------------

  it('does not show zero-count risk summary entries', async () => {
    mockUseSWRFetch
      .mockReturnValueOnce({
        data: {
          receipts: [sampleResult({
            risk_summary: { critical: 0, high: 0, medium: 1, low: 0 },
          })],
        },
        error: null,
        isLoading: false,
        mutate: mockMutate,
      })
      .mockReturnValueOnce({
        data: null,
        error: null,
        isLoading: false,
        mutate: mockMutate,
      });

    render(<ReceiptsPage />);

    await waitFor(() => {
      expect(screen.getByText('M:1')).toBeInTheDocument();
      // C:0, H:0, L:0 should NOT be rendered
      expect(screen.queryByText('C:0')).not.toBeInTheDocument();
      expect(screen.queryByText('H:0')).not.toBeInTheDocument();
      expect(screen.queryByText('L:0')).not.toBeInTheDocument();
    });
  });
});
