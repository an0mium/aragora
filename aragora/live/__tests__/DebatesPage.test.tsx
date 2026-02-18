/**
 * Tests for Debates page — backend API integration, pagination, filtering, and Supabase fallback
 */
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';

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

// Mock EmptyState
jest.mock('../src/components/ui/EmptyState', () => ({
  DebatesEmptyState: ({ onStart }: { onStart: () => void }) => (
    <div data-testid="empty-state">
      <button onClick={onStart}>Start debate</button>
    </div>
  ),
}));

// Mock RightSidebarContext
jest.mock('../src/context/RightSidebarContext', () => ({
  useRightSidebar: () => ({
    setContext: jest.fn(),
    clearContext: jest.fn(),
  }),
}));

// Mock logger
jest.mock('../src/utils/logger', () => ({
  logger: { warn: jest.fn(), error: jest.fn(), info: jest.fn(), debug: jest.fn() },
}));

// Mock Supabase
const mockFetchRecentDebates = jest.fn();
jest.mock('../src/utils/supabase', () => ({
  fetchRecentDebates: (...args: unknown[]) => mockFetchRecentDebates(...args),
}));

// Mock agent colors
jest.mock('../src/utils/agentColors', () => ({
  getAgentColors: () => ({ bg: 'bg-blue-500/20', text: 'text-blue-400' }),
}));

// Setup fetch mock
const mockFetch = jest.fn();

import DebatesPage from '../src/app/(app)/debates/page';

describe('DebatesPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetchRecentDebates.mockResolvedValue([]);
    global.fetch = mockFetch;
  });

  function mockBackendDebates(debates: unknown[], options?: { total?: number; has_more?: boolean }) {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        debates,
        total: options?.total ?? debates.length,
        has_more: options?.has_more ?? false,
      }),
    });
  }

  function mockBackendError() {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));
  }

  const sampleDebate = (overrides: Record<string, unknown> = {}) => ({
    id: 'debate-1',
    task: 'Should we use TypeScript?',
    agents: ['claude', 'gpt-4'],
    consensus_reached: true,
    confidence: 0.85,
    created_at: new Date().toISOString(),
    ...overrides,
  });

  it('renders page header', async () => {
    mockBackendDebates([sampleDebate()]);

    await act(async () => {
      render(<DebatesPage />);
    });

    // Header H1 contains DEBATE ARCHIVE
    const heading = screen.getByRole('heading', { level: 1 });
    expect(heading.textContent).toContain('DEBATE ARCHIVE');
  });

  it('shows loading state initially', () => {
    // Never resolve fetch
    mockFetch.mockReturnValue(new Promise(() => {}));

    render(<DebatesPage />);

    expect(screen.getByText(/LOADING DEBATES/)).toBeInTheDocument();
  });

  it('renders debates from backend API', async () => {
    mockBackendDebates([
      sampleDebate({ task: 'Microservices vs monolith' }),
      sampleDebate({ id: 'debate-2', task: 'REST vs GraphQL' }),
    ]);

    await act(async () => {
      render(<DebatesPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('Microservices vs monolith')).toBeInTheDocument();
      expect(screen.getByText('REST vs GraphQL')).toBeInTheDocument();
    });
  });

  it('shows [API] data source indicator for backend data', async () => {
    mockBackendDebates([sampleDebate()]);

    await act(async () => {
      render(<DebatesPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('[API]')).toBeInTheDocument();
    });
  });

  it('falls back to Supabase when backend fails', async () => {
    const supabaseDebates = [
      {
        id: 'supa-1',
        task: 'Supabase debate',
        agents: ['claude'],
        consensus_reached: true,
        confidence: 0.9,
        created_at: new Date().toISOString(),
        loop_id: '',
        cycle_number: 0,
        phase: 'completed',
        transcript: [],
        winning_proposal: null,
        vote_tally: null,
      },
    ];

    mockBackendError();
    mockFetchRecentDebates.mockResolvedValue(supabaseDebates);

    await act(async () => {
      render(<DebatesPage />);
    });

    await waitFor(() => {
      expect(mockFetchRecentDebates).toHaveBeenCalled();
      expect(screen.getByText('Supabase debate')).toBeInTheDocument();
    });
  });

  it('shows [DB] data source indicator for Supabase data', async () => {
    mockBackendError();
    mockFetchRecentDebates.mockResolvedValue([
      {
        id: 'supa-1',
        task: 'DB debate',
        agents: ['claude'],
        consensus_reached: true,
        confidence: 0.8,
        created_at: new Date().toISOString(),
        loop_id: '',
        cycle_number: 0,
        phase: 'completed',
        transcript: [],
        winning_proposal: null,
        vote_tally: null,
      },
    ]);

    await act(async () => {
      render(<DebatesPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('[DB]')).toBeInTheDocument();
    });
  });

  it('normalizes question field to task', async () => {
    mockBackendDebates([
      sampleDebate({ task: undefined, question: 'What should we build?' }),
    ]);

    await act(async () => {
      render(<DebatesPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('What should we build?')).toBeInTheDocument();
    });
  });

  it('normalizes debate_id field', async () => {
    mockBackendDebates([
      sampleDebate({ id: 'raw-id', debate_id: 'canonical-id', task: 'Canonical test' }),
    ]);

    await act(async () => {
      render(<DebatesPage />);
    });

    await waitFor(() => {
      const link = screen.getByText('Canonical test').closest('a');
      expect(link?.getAttribute('href')).toBe('/debates/canonical-id');
    });
  });

  it('shows empty state when no debates', async () => {
    mockBackendDebates([]);
    mockFetchRecentDebates.mockResolvedValue([]);

    await act(async () => {
      render(<DebatesPage />);
    });

    await waitFor(() => {
      expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    });
  });

  describe('filtering', () => {
    it('renders filter buttons', async () => {
      mockBackendDebates([sampleDebate()]);

      await act(async () => {
        render(<DebatesPage />);
      });

      await waitFor(() => {
        // Filter buttons are in a filter row with "Filter:" label
        expect(screen.getByText('ALL')).toBeInTheDocument();
        // "CONSENSUS" may also appear as a status badge — get all and check one is a button
        const consensusElements = screen.getAllByText('CONSENSUS');
        expect(consensusElements.some(el => el.tagName === 'BUTTON')).toBe(true);
        expect(screen.getByText('NO CONSENSUS')).toBeInTheDocument();
      });
    });

    it('filters by consensus status', async () => {
      mockBackendDebates([
        sampleDebate({ id: 'd1', task: 'Consensus debate', consensus_reached: true }),
        sampleDebate({ id: 'd2', task: 'Dissent debate', consensus_reached: false }),
      ]);

      await act(async () => {
        render(<DebatesPage />);
      });

      await waitFor(() => {
        expect(screen.getByText('Consensus debate')).toBeInTheDocument();
        expect(screen.getByText('Dissent debate')).toBeInTheDocument();
      });

      // Find the CONSENSUS filter button (not the status badge)
      const consensusButtons = screen.getAllByText('CONSENSUS');
      const filterBtn = consensusButtons.find(el => el.tagName === 'BUTTON')!;
      fireEvent.click(filterBtn);

      expect(screen.getByText('Consensus debate')).toBeInTheDocument();
      expect(screen.queryByText('Dissent debate')).not.toBeInTheDocument();

      // Filter to no consensus
      fireEvent.click(screen.getByText('NO CONSENSUS'));

      expect(screen.queryByText('Consensus debate')).not.toBeInTheDocument();
      expect(screen.getByText('Dissent debate')).toBeInTheDocument();

      // Back to all
      fireEvent.click(screen.getByText('ALL'));

      expect(screen.getByText('Consensus debate')).toBeInTheDocument();
      expect(screen.getByText('Dissent debate')).toBeInTheDocument();
    });

    it('shows filtered count', async () => {
      mockBackendDebates([
        sampleDebate({ id: 'd1', consensus_reached: true }),
        sampleDebate({ id: 'd2', consensus_reached: false }),
        sampleDebate({ id: 'd3', consensus_reached: true }),
      ]);

      await act(async () => {
        render(<DebatesPage />);
      });

      await waitFor(() => {
        expect(screen.getByText(/Showing 3 of 3 debates/)).toBeInTheDocument();
      });

      // Find the CONSENSUS filter button (not status badges)
      const consensusButtons = screen.getAllByText('CONSENSUS');
      const filterBtn = consensusButtons.find(el => el.tagName === 'BUTTON')!;
      fireEvent.click(filterBtn);

      expect(screen.getByText(/Showing 2 of 3 debates/)).toBeInTheDocument();
    });
  });

  describe('pagination', () => {
    it('shows load more button when has_more is true', async () => {
      const debates = Array.from({ length: 20 }, (_, i) =>
        sampleDebate({ id: `d${i}`, task: `Debate ${i}` })
      );
      mockBackendDebates(debates, { has_more: true });

      await act(async () => {
        render(<DebatesPage />);
      });

      await waitFor(() => {
        expect(screen.getByText('LOAD MORE DEBATES')).toBeInTheDocument();
      });
    });

    it('hides load more button when no more data', async () => {
      mockBackendDebates([sampleDebate()], { has_more: false });

      await act(async () => {
        render(<DebatesPage />);
      });

      await waitFor(() => {
        expect(screen.queryByText('LOAD MORE DEBATES')).not.toBeInTheDocument();
      });
    });

    it('loads more debates on button click', async () => {
      const firstPage = Array.from({ length: 20 }, (_, i) =>
        sampleDebate({ id: `d${i}`, task: `Debate ${i}` })
      );
      mockBackendDebates(firstPage, { has_more: true });

      await act(async () => {
        render(<DebatesPage />);
      });

      await waitFor(() => {
        expect(screen.getByText('LOAD MORE DEBATES')).toBeInTheDocument();
      });

      // Mock second page
      const secondPage = Array.from({ length: 5 }, (_, i) =>
        sampleDebate({ id: `d${20 + i}`, task: `Debate ${20 + i}` })
      );
      mockBackendDebates(secondPage);

      await act(async () => {
        fireEvent.click(screen.getByText('LOAD MORE DEBATES'));
      });

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledTimes(2);
      });
    });
  });

  it('displays confidence percentage', async () => {
    mockBackendDebates([sampleDebate({ confidence: 0.92 })]);

    await act(async () => {
      render(<DebatesPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('92% conf')).toBeInTheDocument();
    });
  });

  it('displays cycle and phase info', async () => {
    mockBackendDebates([
      sampleDebate({ cycle_number: 3, phase: 'critique' }),
    ]);

    await act(async () => {
      render(<DebatesPage />);
    });

    await waitFor(() => {
      expect(screen.getByText('C3 / critique')).toBeInTheDocument();
    });
  });

  it('sends correct API request with offset', async () => {
    mockBackendDebates([sampleDebate()]);

    await act(async () => {
      render(<DebatesPage />);
    });

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/debates?limit=20&offset=0'),
        expect.objectContaining({
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });
  });
});
