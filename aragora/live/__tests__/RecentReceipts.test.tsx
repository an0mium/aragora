import { render, screen, waitFor } from '@testing-library/react';
import { RecentReceipts } from '../src/components/RecentReceipts';

jest.mock('next/link', () => {
  return ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>{children}</a>
  );
});

jest.mock('../src/components/DebateThisButton', () => ({
  DebateThisButton: () => <button data-testid="debate-btn">Debate</button>,
}));

const mockApiFetch = jest.fn();
jest.mock('../src/lib/api', () => ({
  apiFetch: (...args: unknown[]) => mockApiFetch(...args),
}));

const MOCK_RECEIPTS = [
  {
    id: 'r-001',
    receipt_id: 'receipt-001',
    verdict: 'PASS' as const,
    created_at: '2026-02-25T12:00:00Z',
    artifact_hash: 'abc123',
    findings_count: 3,
    input_summary: 'Should we adopt Kubernetes?',
    confidence: 0.92,
  },
  {
    id: 'r-002',
    verdict: 'FAIL' as const,
    created_at: '2026-02-24T10:00:00Z',
    artifact_hash: 'def456',
    findings_count: 0,
    input_summary: 'Migration to microservices',
    confidence: 0.45,
  },
  {
    id: 'r-003',
    verdict: 'WARN' as const,
    created_at: '2026-02-23T08:00:00Z',
    artifact_hash: 'ghi789',
    findings_count: 1,
  },
];

describe('RecentReceipts', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('loading state', () => {
    it('shows loading indicator', () => {
      mockApiFetch.mockReturnValue(new Promise(() => {}));
      render(<RecentReceipts />);
      expect(screen.getByText('RECENT RECEIPTS')).toBeInTheDocument();
      expect(screen.getByText('loading...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message', async () => {
      mockApiFetch.mockRejectedValue(new Error('Network error'));
      render(<RecentReceipts />);
      await waitFor(() => {
        expect(screen.getByText('Could not load receipts.')).toBeInTheDocument();
      });
    });
  });

  describe('empty state', () => {
    it('shows empty message when no receipts', async () => {
      mockApiFetch.mockResolvedValue({ receipts: [] });
      render(<RecentReceipts />);
      await waitFor(() => {
        expect(screen.getByText(/No decision receipts yet/)).toBeInTheDocument();
      });
    });
  });

  describe('populated state', () => {
    beforeEach(() => {
      mockApiFetch.mockResolvedValue({ receipts: MOCK_RECEIPTS });
    });

    it('renders receipt summaries', async () => {
      render(<RecentReceipts />);
      await waitFor(() => {
        expect(screen.getByText('Should we adopt Kubernetes?')).toBeInTheDocument();
        expect(screen.getByText('Migration to microservices')).toBeInTheDocument();
      });
    });

    it('renders verdict badges', async () => {
      render(<RecentReceipts />);
      await waitFor(() => {
        expect(screen.getByText('PASS')).toBeInTheDocument();
        expect(screen.getByText('FAIL')).toBeInTheDocument();
        expect(screen.getByText('WARN')).toBeInTheDocument();
      });
    });

    it('renders confidence percentages', async () => {
      render(<RecentReceipts />);
      await waitFor(() => {
        expect(screen.getByText('92%')).toBeInTheDocument();
        expect(screen.getByText('45%')).toBeInTheDocument();
      });
    });

    it('renders findings count for non-zero', async () => {
      render(<RecentReceipts />);
      await waitFor(() => {
        expect(screen.getByText('3 findings')).toBeInTheDocument();
        expect(screen.getByText('1 finding')).toBeInTheDocument();
      });
    });

    it('renders VIEW ALL link', async () => {
      render(<RecentReceipts />);
      await waitFor(() => {
        const link = screen.getByText('[VIEW ALL]');
        expect(link.closest('a')).toHaveAttribute('href', '/receipts');
      });
    });

    it('renders receipt links', async () => {
      render(<RecentReceipts />);
      await waitFor(() => {
        const link = screen.getByText('Should we adopt Kubernetes?').closest('a');
        expect(link).toHaveAttribute('href', '/receipts?id=r-001');
      });
    });

    it('renders SHA-256 footer', async () => {
      render(<RecentReceipts />);
      await waitFor(() => {
        expect(screen.getByText('SHA-256 verified audit trail')).toBeInTheDocument();
        expect(screen.getByText('3 receipts')).toBeInTheDocument();
      });
    });

    it('renders DebateThisButton for each receipt', async () => {
      render(<RecentReceipts />);
      await waitFor(() => {
        const buttons = screen.getAllByTestId('debate-btn');
        expect(buttons).toHaveLength(3);
      });
    });

    it('falls back to truncated ID when no input_summary', async () => {
      render(<RecentReceipts />);
      await waitFor(() => {
        expect(screen.getByText(/Receipt r-003/)).toBeInTheDocument();
      });
    });
  });

  describe('limit prop', () => {
    it('passes limit to API call', async () => {
      mockApiFetch.mockResolvedValue({ receipts: [] });
      render(<RecentReceipts limit={10} />);
      await waitFor(() => {
        expect(mockApiFetch).toHaveBeenCalledWith('/api/gauntlet/receipts?limit=10');
      });
    });

    it('defaults to limit=5', async () => {
      mockApiFetch.mockResolvedValue({ receipts: [] });
      render(<RecentReceipts />);
      await waitFor(() => {
        expect(mockApiFetch).toHaveBeenCalledWith('/api/gauntlet/receipts?limit=5');
      });
    });
  });
});
