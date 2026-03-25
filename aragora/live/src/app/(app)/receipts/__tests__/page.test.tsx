import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ReceiptsPage from '../page';
import { useSWRFetch } from '@/hooks/useSWRFetch';

jest.mock('next/link', () => {
  return function MockLink({ children, href }: { children: React.ReactNode; href: string }) {
    return <a href={href}>{children}</a>;
  };
});

const mockReplace = jest.fn();
let mockPathname = '/receipts';
let mockQuery = '';

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    replace: mockReplace,
    push: jest.fn(),
    prefetch: jest.fn(),
  }),
  usePathname: () => mockPathname,
  useSearchParams: () => new URLSearchParams(mockQuery),
}));

jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => <div data-testid="scanlines" />,
  CRTVignette: () => <div data-testid="crt-vignette" />,
}));

jest.mock('@/components/BackendSelector', () => ({
  useBackend: () => ({ config: { api: 'http://localhost:8080' } }),
}));

jest.mock('@/components/ErrorWithRetry', () => ({
  ErrorWithRetry: ({ error }: { error: string }) => <div>{error}</div>,
}));

jest.mock('@/components/PanelErrorBoundary', () => ({
  PanelErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/receipts', () => ({
  DeliveryModal: () => null,
}));

jest.mock('@/hooks/useSWRFetch', () => ({
  useSWRFetch: jest.fn(),
}));

const mockUseSWRFetch = useSWRFetch as jest.Mock;
const mockFetch = jest.fn();
global.fetch = mockFetch as unknown as typeof fetch;

function setSearchQuery(query: string) {
  mockQuery = query;
}

function configureListMocks() {
  mockUseSWRFetch.mockImplementation((endpoint: string | null) => {
    if (endpoint === '/api/v1/gauntlet/receipts?limit=50') {
      return {
        data: { receipts: [] },
        error: null,
        isLoading: false,
        mutate: jest.fn(),
      };
    }

    if (endpoint === '/api/v2/receipts?limit=50') {
      return {
        data: {
          receipts: [
            {
              id: 'receipt-123',
              receipt_id: 'receipt-123',
              verdict: 'PASS',
              confidence: 0.91,
              created_at: '2026-03-25T12:34:56Z',
              input_summary: 'Receipt 123 summary',
            },
          ],
        },
        error: null,
        isLoading: false,
        mutate: jest.fn(),
      };
    }

    return {
      data: null,
      error: null,
      isLoading: false,
      mutate: jest.fn(),
    };
  });
}

function configureDetailFetch() {
  mockFetch.mockImplementation(async (input: string | URL | Request) => {
    const url = String(input);
    if (url === 'http://localhost:8080/api/v2/receipts/receipt-123') {
      return {
        ok: true,
        json: async () => ({
          receipt_id: 'receipt-123',
          gauntlet_id: 'run-123',
          timestamp: '2026-03-25T12:34:56Z',
          input_summary: 'Receipt 123 summary',
          input_hash: 'input-hash-123',
          risk_summary: { critical: 0, high: 0, medium: 1, low: 0 },
          attacks_attempted: 1,
          attacks_successful: 0,
          probes_run: 2,
          vulnerabilities_found: 1,
          verdict: 'PASS',
          confidence: 0.91,
          robustness_score: 0.82,
          vulnerability_details: [],
          verdict_reasoning: 'Looks good.',
          dissenting_views: [],
          provenance_chain: [],
          artifact_hash: 'artifact-hash-123',
        }),
      } as Response;
    }

    return {
      ok: false,
      status: 404,
      json: async () => ({}),
    } as Response;
  });
}

describe('ReceiptsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockPathname = '/receipts';
    setSearchQuery('');
    configureListMocks();
    configureDetailFetch();
  });

  it('auto-opens receipt detail when the id query param matches a loaded receipt', async () => {
    setSearchQuery('id=receipt-123');

    render(<ReceiptsPage />);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/v2/receipts/receipt-123',
        expect.objectContaining({ signal: expect.anything() })
      );
    });

    expect(await screen.findByText('Decision Receipt')).toBeInTheDocument();
    expect(screen.getByText(/ID: receipt-123/)).toBeInTheDocument();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it('updates the URL when a receipt is opened and clears it on back', async () => {
    const user = userEvent.setup();

    render(<ReceiptsPage />);

    await user.click(screen.getByRole('button', { name: /Receipt 123 summary/i }));

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith('/receipts?id=receipt-123');
    });

    await user.click(screen.getByRole('button', { name: 'Back' }));

    expect(mockReplace).toHaveBeenLastCalledWith('/receipts');
    expect(screen.queryByText('Decision Receipt')).not.toBeInTheDocument();
  });

  it('leaves the page on the list view when the id query param does not match', async () => {
    setSearchQuery('id=missing-receipt');

    render(<ReceiptsPage />);

    expect(await screen.findByRole('button', { name: /Receipt 123 summary/i })).toBeInTheDocument();

    await waitFor(() => {
      expect(mockFetch).not.toHaveBeenCalled();
    });

    expect(screen.queryByText('Decision Receipt')).not.toBeInTheDocument();
  });
});
