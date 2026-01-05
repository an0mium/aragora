import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { TrendingTopicsPanel } from '../src/components/TrendingTopicsPanel';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('TrendingTopicsPanel', () => {
  const mockTopics = [
    { topic: 'AI Safety Research', source: 'hackernews', score: 0.85, category: 'ai' },
    { topic: 'Quantum Computing Breakthrough', source: 'arxiv', score: 0.72, debate_count: 3 },
    { topic: 'React 19 Release', source: 'reddit', score: 0.65, last_active: new Date().toISOString() },
    { topic: 'OpenAI Update', source: 'twitter', score: 0.45 },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ topics: mockTopics }),
    });
  });

  describe('collapsed state', () => {
    it('renders header in collapsed state', () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      expect(screen.getByText('[TRENDING]')).toBeInTheDocument();
      expect(screen.getByText('[+]')).toBeInTheDocument();
    });

    it('does not fetch topics when collapsed', () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  describe('expanded state', () => {
    it('expands when header is clicked', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      const header = screen.getByRole('button');
      fireEvent.click(header);

      await waitFor(() => {
        expect(screen.getByText('[-]')).toBeInTheDocument();
      });
    });

    it('fetches topics when expanded', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      const header = screen.getByRole('button');
      fireEvent.click(header);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/pulse/trending?limit=10'
        );
      });
    });

    it('displays loading state', async () => {
      mockFetch.mockImplementation(() => new Promise(() => {})); // Never resolves

      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      const header = screen.getByRole('button');
      fireEvent.click(header);

      expect(screen.getByText('Scanning for trending topics...')).toBeInTheDocument();
    });

    it('displays topics after loading', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      const header = screen.getByRole('button');
      fireEvent.click(header);

      await waitFor(() => {
        expect(screen.getByText('AI Safety Research')).toBeInTheDocument();
        expect(screen.getByText('Quantum Computing Breakthrough')).toBeInTheDocument();
      });
    });
  });

  describe('topic display', () => {
    it('shows score as percentage', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByText('85%')).toBeInTheDocument();
        expect(screen.getByText('72%')).toBeInTheDocument();
      });
    });

    it('shows source icons', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        // Source icons are shown as emoji
        expect(screen.getByTitle('hackernews')).toHaveTextContent('🔶');
        expect(screen.getByTitle('arxiv')).toHaveTextContent('📄');
      });
    });

    it('shows debate count when available', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByText('3 debates')).toBeInTheDocument();
      });
    });

    it('shows category badge when available', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByText('ai')).toBeInTheDocument();
      });
    });
  });

  describe('error handling', () => {
    it('displays error message on fetch failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByText('Failed to fetch trending topics')).toBeInTheDocument();
      });
    });

    it('displays network error message', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByText('Network error')).toBeInTheDocument();
      });
    });
  });

  describe('empty state', () => {
    it('shows empty message when no topics', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ topics: [] }),
      });

      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByText('No trending topics detected')).toBeInTheDocument();
      });
    });
  });

  describe('refresh functionality', () => {
    it('shows refresh button when expanded', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByText('Refresh Trending')).toBeInTheDocument();
      });
    });

    it('refreshes topics when refresh button clicked', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByText('AI Safety Research')).toBeInTheDocument();
      });

      // Clear and click refresh
      mockFetch.mockClear();
      const updatedTopics = [{ topic: 'New Topic', source: 'github', score: 0.9 }];
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ topics: updatedTopics }),
      });

      fireEvent.click(screen.getByText('Refresh Trending'));

      await waitFor(() => {
        expect(screen.getByText('New Topic')).toBeInTheDocument();
      });
    });

    it('shows refreshing state', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByText('Refresh Trending')).toBeInTheDocument();
      });

      mockFetch.mockImplementation(() => new Promise(() => {}));

      fireEvent.click(screen.getByText('Refresh Trending'));

      await waitFor(() => {
        expect(screen.getByText('Refreshing...')).toBeInTheDocument();
      });
    });
  });

  describe('auto-refresh', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('auto-refreshes at specified interval', async () => {
      render(
        <TrendingTopicsPanel
          apiBase="http://localhost:8080"
          autoRefresh={true}
          refreshInterval={5000}
        />
      );

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledTimes(1);
      });

      // Advance time by refresh interval
      await act(async () => {
        jest.advanceTimersByTime(5000);
      });

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledTimes(2);
      });
    });

    it('does not auto-refresh when disabled', async () => {
      render(
        <TrendingTopicsPanel
          apiBase="http://localhost:8080"
          autoRefresh={false}
          refreshInterval={5000}
        />
      );

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledTimes(1);
      });

      await act(async () => {
        jest.advanceTimersByTime(10000);
      });

      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  describe('legend', () => {
    it('shows source legend', async () => {
      render(<TrendingTopicsPanel apiBase="http://localhost:8080" />);

      fireEvent.click(screen.getByRole('button'));

      await waitFor(() => {
        expect(screen.getByText('🔶 HN')).toBeInTheDocument();
        expect(screen.getByText('📄 arXiv')).toBeInTheDocument();
        expect(screen.getByText('🐙 GitHub')).toBeInTheDocument();
        expect(screen.getByText('💬 Debates')).toBeInTheDocument();
      });
    });
  });
});
