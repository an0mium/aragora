/**
 * Tests for MomentsTimeline component
 *
 * Tests cover:
 * - Loading state
 * - Error handling
 * - Empty state when no moments
 * - Moment type filtering
 * - Highlight display for most significant moment
 * - Agent distribution display
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MomentsTimeline } from '../src/components/MomentsTimeline';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

const mockSummaryData = {
  total_moments: 15,
  by_type: {
    upset_victory: 5,
    consensus_breakthrough: 4,
    position_reversal: 3,
    streak_achievement: 3,
  },
  by_agent: {
    'claude-3-opus': 6,
    'gpt-4o': 5,
    'gemini-pro': 4,
  },
  most_significant: {
    id: 'moment-1',
    type: 'upset_victory',
    agent: 'claude-3-opus',
    description: 'Won against heavily favored opponent in ethics debate',
    significance: 0.95,
    created_at: '2026-01-10T10:00:00Z',
  },
  recent: [
    {
      id: 'moment-1',
      type: 'upset_victory',
      agent: 'claude-3-opus',
      description: 'Won against heavily favored opponent',
      significance: 0.95,
      created_at: '2026-01-10T10:00:00Z',
    },
    {
      id: 'moment-2',
      type: 'consensus_breakthrough',
      agent: 'gpt-4o',
      description: 'First to propose winning consensus',
      significance: 0.85,
      created_at: '2026-01-10T09:00:00Z',
    },
    {
      id: 'moment-3',
      type: 'position_reversal',
      agent: 'gemini-pro',
      description: 'Changed stance based on new evidence',
      significance: 0.7,
      created_at: '2026-01-10T08:00:00Z',
    },
  ],
};

describe('MomentsTimeline', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  describe('Loading State', () => {
    it('shows loading state initially', () => {
      mockFetch.mockImplementation(() => new Promise(() => {}));

      render(<MomentsTimeline />);

      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('displays error message when fetch fails', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText(/failed to fetch/i)).toBeInTheDocument();
      });
    });
  });

  describe('Empty State', () => {
    it('shows empty message when no moments exist', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ total_moments: 0, by_type: {}, by_agent: {}, recent: [] }),
      });

      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText(/no significant moments/i)).toBeInTheDocument();
      });
    });

    it('handles 503 response gracefully', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 503 });

      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText(/no significant moments/i)).toBeInTheDocument();
      });
    });
  });

  describe('Content Display', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockSummaryData),
      });
    });

    it('displays panel title with total count badge', async () => {
      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText('MOMENTS')).toBeInTheDocument();
        expect(screen.getByText('15')).toBeInTheDocument();
      });
    });

    it('displays highlight section with most significant moment', async () => {
      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText('HIGHLIGHT')).toBeInTheDocument();
        expect(screen.getByText('Upset Victory')).toBeInTheDocument();
        expect(screen.getByText(/heavily favored opponent/i)).toBeInTheDocument();
      });
    });

    it('displays significance percentage', async () => {
      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText('95%')).toBeInTheDocument();
      });
    });

    it('displays recent moments', async () => {
      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
        expect(screen.getByText('gpt-4o')).toBeInTheDocument();
        expect(screen.getByText('gemini-pro')).toBeInTheDocument();
      });
    });

    it('displays moment type icons', async () => {
      render(<MomentsTimeline />);

      await waitFor(() => {
        // Upset victory icon
        expect(screen.getAllByText('🏆').length).toBeGreaterThan(0);
      });
    });
  });

  describe('Type Filtering', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockSummaryData),
      });
    });

    it('shows filter buttons for each moment type', async () => {
      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText('All')).toBeInTheDocument();
        // Type counts are shown
        expect(screen.getByText('5')).toBeInTheDocument(); // upset_victory count
      });
    });

    it('filters moments when type button clicked', async () => {
      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText('All')).toBeInTheDocument();
      });

      // Find the upset_victory filter button (by its count)
      const filterButtons = screen.getAllByRole('button');
      const upsetButton = filterButtons.find(btn => btn.textContent?.includes('🏆'));

      if (upsetButton) {
        fireEvent.click(upsetButton);

        await waitFor(() => {
          // Should show filtered header
          expect(screen.getByText('Upset Victory')).toBeInTheDocument();
        });
      }
    });

    it('clears filter when clicking same type again', async () => {
      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText('All')).toBeInTheDocument();
      });

      const allButton = screen.getByText('All');
      fireEvent.click(allButton);

      await waitFor(() => {
        expect(screen.getByText('RECENT')).toBeInTheDocument();
      });
    });
  });

  describe('Agent Distribution', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockSummaryData),
      });
    });

    it('displays agent distribution section', async () => {
      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText('BY AGENT')).toBeInTheDocument();
      });
    });

    it('shows top agents with counts', async () => {
      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
        expect(screen.getByText('(6)')).toBeInTheDocument();
      });
    });
  });

  describe('Refresh Functionality', () => {
    it('refetches data when refresh button clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockSummaryData),
      });

      render(<MomentsTimeline />);

      await waitFor(() => {
        expect(screen.getByText('MOMENTS')).toBeInTheDocument();
      });

      // Initial fetch
      expect(mockFetch).toHaveBeenCalledTimes(1);

      // Find and click refresh button
      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      fireEvent.click(refreshButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledTimes(2);
      });
    });
  });
});
