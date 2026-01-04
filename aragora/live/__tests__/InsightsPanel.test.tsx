/**
 * Tests for InsightsPanel component
 *
 * Note: This is a placeholder test file. To run these tests, you would need to:
 * 1. Install Jest and React Testing Library: npm install --save-dev jest @testing-library/react @testing-library/jest-dom
 * 2. Configure Jest in package.json or jest.config.js
 * 3. Add test script to package.json
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { InsightsPanel } from '../src/components/InsightsPanel';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('InsightsPanel', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe('Tab Navigation', () => {
    it('renders all three tabs', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ insights: [], flips: [], summary: {} }),
      });

      render(<InsightsPanel wsMessages={[]} />);

      expect(screen.getByText(/Insights/)).toBeInTheDocument();
      expect(screen.getByText(/Memory/)).toBeInTheDocument();
      expect(screen.getByText(/Flips/)).toBeInTheDocument();
    });

    it('defaults to Insights tab', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ insights: [], flips: [], summary: {} }),
      });

      render(<InsightsPanel wsMessages={[]} />);

      // Check that Insights tab button is active (has different styling)
      const insightsTab = screen.getByRole('button', { name: /Insights/ });
      expect(insightsTab).toHaveClass('bg-accent');
    });

    it('switches to Flips tab when clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ insights: [], flips: [], summary: {} }),
      });

      render(<InsightsPanel wsMessages={[]} />);

      const flipsTab = screen.getByRole('button', { name: /Flips/ });
      fireEvent.click(flipsTab);

      expect(flipsTab).toHaveClass('bg-accent');
    });
  });

  describe('Flips Tab Content', () => {
    const mockFlips = [
      {
        id: 'flip-1',
        agent: 'claude',
        type: 'contradiction',
        type_emoji: '🔄',
        before: { claim: 'Original position A', confidence: '85%' },
        after: { claim: 'New position B', confidence: '70%' },
        similarity: '45%',
        domain: 'architecture',
        timestamp: '2026-01-04T12:00:00Z',
      },
      {
        id: 'flip-2',
        agent: 'gemini',
        type: 'refinement',
        type_emoji: '🔧',
        before: { claim: 'Initial approach', confidence: '60%' },
        after: { claim: 'Improved approach', confidence: '80%' },
        similarity: '75%',
        domain: 'performance',
        timestamp: '2026-01-04T11:00:00Z',
      },
    ];

    const mockSummary = {
      total_flips: 5,
      by_type: { contradiction: 2, refinement: 3 },
      by_agent: { claude: 3, gemini: 2 },
      recent_24h: 2,
    };

    it('displays flips when available', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/flips/recent')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ flips: mockFlips }),
          });
        }
        if (url.includes('/api/flips/summary')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ summary: mockSummary }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ insights: [] }),
        });
      });

      render(<InsightsPanel wsMessages={[]} />);

      // Switch to Flips tab
      const flipsTab = screen.getByRole('button', { name: /Flips/ });
      fireEvent.click(flipsTab);

      await waitFor(() => {
        expect(screen.getByText('claude')).toBeInTheDocument();
        expect(screen.getByText('gemini')).toBeInTheDocument();
      });
    });

    it('displays flip type badges with correct colors', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/flips/recent')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ flips: mockFlips }),
          });
        }
        if (url.includes('/api/flips/summary')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ summary: mockSummary }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ insights: [] }),
        });
      });

      render(<InsightsPanel wsMessages={[]} />);

      const flipsTab = screen.getByRole('button', { name: /Flips/ });
      fireEvent.click(flipsTab);

      await waitFor(() => {
        const contradictionBadge = screen.getByText(/contradiction/);
        expect(contradictionBadge).toHaveClass('text-red-400');

        const refinementBadge = screen.getByText(/refinement/);
        expect(refinementBadge).toHaveClass('text-green-400');
      });
    });

    it('displays before/after claims', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/flips/recent')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ flips: mockFlips }),
          });
        }
        if (url.includes('/api/flips/summary')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ summary: mockSummary }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ insights: [] }),
        });
      });

      render(<InsightsPanel wsMessages={[]} />);

      const flipsTab = screen.getByRole('button', { name: /Flips/ });
      fireEvent.click(flipsTab);

      await waitFor(() => {
        expect(screen.getByText('Original position A')).toBeInTheDocument();
        expect(screen.getByText('New position B')).toBeInTheDocument();
        expect(screen.getByText('Before')).toBeInTheDocument();
        expect(screen.getByText('After')).toBeInTheDocument();
      });
    });

    it('displays flip summary when available', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/flips/recent')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ flips: mockFlips }),
          });
        }
        if (url.includes('/api/flips/summary')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ summary: mockSummary }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ insights: [] }),
        });
      });

      render(<InsightsPanel wsMessages={[]} />);

      const flipsTab = screen.getByRole('button', { name: /Flips/ });
      fireEvent.click(flipsTab);

      await waitFor(() => {
        expect(screen.getByText('Position Reversals')).toBeInTheDocument();
        expect(screen.getByText('2 in 24h')).toBeInTheDocument();
        expect(screen.getByText('2 contradictions')).toBeInTheDocument();
        expect(screen.getByText('3 refinements')).toBeInTheDocument();
      });
    });

    it('shows empty state when no flips', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/flips/recent')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ flips: [] }),
          });
        }
        if (url.includes('/api/flips/summary')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ summary: null }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ insights: [] }),
        });
      });

      render(<InsightsPanel wsMessages={[]} />);

      const flipsTab = screen.getByRole('button', { name: /Flips/ });
      fireEvent.click(flipsTab);

      await waitFor(() => {
        expect(
          screen.getByText(/No position flips detected yet/)
        ).toBeInTheDocument();
      });
    });

    it('displays domain tag when present', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/flips/recent')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ flips: mockFlips }),
          });
        }
        if (url.includes('/api/flips/summary')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ summary: mockSummary }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ insights: [] }),
        });
      });

      render(<InsightsPanel wsMessages={[]} />);

      const flipsTab = screen.getByRole('button', { name: /Flips/ });
      fireEvent.click(flipsTab);

      await waitFor(() => {
        expect(screen.getByText('architecture')).toBeInTheDocument();
        expect(screen.getByText('performance')).toBeInTheDocument();
      });
    });
  });

  describe('API Integration', () => {
    it('uses provided apiBase for API calls', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ insights: [], flips: [], summary: {} }),
      });

      render(<InsightsPanel wsMessages={[]} apiBase="https://custom-api.example.com" />);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('https://custom-api.example.com')
        );
      });
    });

    it('handles API errors gracefully', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
      });

      render(<InsightsPanel wsMessages={[]} />);

      await waitFor(() => {
        expect(screen.getByText(/HTTP 500/)).toBeInTheDocument();
      });
    });

    it('calls refresh when Refresh button is clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ insights: [] }),
      });

      render(<InsightsPanel wsMessages={[]} />);

      const initialCallCount = mockFetch.mock.calls.length;

      const refreshButton = screen.getByText('Refresh');
      fireEvent.click(refreshButton);

      await waitFor(() => {
        expect(mockFetch.mock.calls.length).toBeGreaterThan(initialCallCount);
      });
    });
  });

  describe('Memory Tab', () => {
    it('displays memory recalls from WebSocket messages', () => {
      const wsMessages = [
        {
          type: 'memory_recall',
          data: {
            query: 'Test query',
            hits: [
              { topic: 'Related topic 1', similarity: 0.85 },
              { topic: 'Related topic 2', similarity: 0.72 },
            ],
            count: 2,
          },
          timestamp: '2026-01-04T12:00:00Z',
        },
      ];

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ insights: [], flips: [], summary: {} }),
      });

      render(<InsightsPanel wsMessages={wsMessages} />);

      const memoryTab = screen.getByRole('button', { name: /Memory/ });
      fireEvent.click(memoryTab);

      expect(screen.getByText('Query: Test query')).toBeInTheDocument();
      expect(screen.getByText('Related topic 1')).toBeInTheDocument();
      expect(screen.getByText('85%')).toBeInTheDocument();
    });

    it('shows empty state when no memory recalls', () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ insights: [], flips: [], summary: {} }),
      });

      render(<InsightsPanel wsMessages={[]} />);

      const memoryTab = screen.getByRole('button', { name: /Memory/ });
      fireEvent.click(memoryTab);

      expect(
        screen.getByText(/No memory recalls yet/)
      ).toBeInTheDocument();
    });
  });
});
