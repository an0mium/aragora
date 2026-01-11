/**
 * Tests for MemoryAnalyticsPanel component
 *
 * Tests cover:
 * - Loading state
 * - Error handling
 * - Empty/unavailable state
 * - Tier distribution display
 * - Key metrics (promotion rate, learning velocity)
 * - Retrieval performance stats
 * - Recommendations display
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryAnalyticsPanel } from '../src/components/MemoryAnalyticsPanel';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

const mockAnalyticsData = {
  summary: {
    total_memories: 1250,
    active_memories: 890,
    tier_distribution: {
      fast: 150,
      medium: 400,
      slow: 500,
      glacial: 200,
    },
  },
  promotions: {
    fast_to_medium: 45,
    medium_to_slow: 30,
    slow_to_glacial: 15,
    promotion_rate: 0.72,
  },
  learning_velocity: {
    current: 3.5,
    trend: 'increasing',
    percentile_7d: 85,
  },
  retrieval_stats: {
    avg_latency_ms: 12.5,
    hit_rate: 0.92,
    most_retrieved_topics: ['ethics', 'technology', 'climate', 'economics', 'healthcare'],
  },
  recommendations: [
    {
      type: 'optimization',
      message: 'Consider promoting high-access slow tier memories',
      priority: 'medium',
    },
    {
      type: 'cleanup',
      message: 'Archive 50 low-access glacial memories',
      priority: 'low',
    },
  ],
};

describe('MemoryAnalyticsPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  describe('Loading State', () => {
    it('shows loading state initially', () => {
      mockFetch.mockImplementation(() => new Promise(() => {}));

      render(<MemoryAnalyticsPanel />);

      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('displays error message when fetch fails', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText(/failed to fetch/i)).toBeInTheDocument();
      });
    });

    it('displays error for non-ok response', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 500 });

      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText(/failed to fetch memory analytics/i)).toBeInTheDocument();
      });
    });
  });

  describe('Empty/Unavailable State', () => {
    it('shows unavailable message for 503 response', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 503 });

      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText(/memory analytics not available/i)).toBeInTheDocument();
      });
    });
  });

  describe('Panel Header', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockAnalyticsData),
      });
    });

    it('displays panel title', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('MEMORY ANALYTICS')).toBeInTheDocument();
      });
    });

    it('displays total memories badge', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('1,250')).toBeInTheDocument();
      });
    });

    it('displays brain icon', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('🧠')).toBeInTheDocument();
      });
    });
  });

  describe('Tier Distribution', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockAnalyticsData),
      });
    });

    it('displays tier distribution section', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('TIER DISTRIBUTION')).toBeInTheDocument();
      });
    });

    it('displays all tier names', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('fast:')).toBeInTheDocument();
        expect(screen.getByText('medium:')).toBeInTheDocument();
        expect(screen.getByText('slow:')).toBeInTheDocument();
        expect(screen.getByText('glacial:')).toBeInTheDocument();
      });
    });

    it('displays tier counts', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('150')).toBeInTheDocument();
        expect(screen.getByText('400')).toBeInTheDocument();
        expect(screen.getByText('500')).toBeInTheDocument();
        expect(screen.getByText('200')).toBeInTheDocument();
      });
    });
  });

  describe('Key Metrics', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockAnalyticsData),
      });
    });

    it('displays promotion rate', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('72.0%')).toBeInTheDocument();
        expect(screen.getByText('Promotion Rate')).toBeInTheDocument();
      });
    });

    it('displays learning velocity', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('3.5')).toBeInTheDocument();
        expect(screen.getByText('Learning Velocity')).toBeInTheDocument();
      });
    });

    it('displays trend icon for increasing velocity', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('📈')).toBeInTheDocument();
      });
    });

    it('displays correct icon for decreasing trend', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          ...mockAnalyticsData,
          learning_velocity: { ...mockAnalyticsData.learning_velocity, trend: 'decreasing' },
        }),
      });

      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('📉')).toBeInTheDocument();
      });
    });

    it('displays stable trend icon', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          ...mockAnalyticsData,
          learning_velocity: { ...mockAnalyticsData.learning_velocity, trend: 'stable' },
        }),
      });

      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('➡️')).toBeInTheDocument();
      });
    });
  });

  describe('Retrieval Performance', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockAnalyticsData),
      });
    });

    it('displays retrieval performance section', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('RETRIEVAL PERFORMANCE')).toBeInTheDocument();
      });
    });

    it('displays hit rate', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('92%')).toBeInTheDocument();
        expect(screen.getByText('hit rate')).toBeInTheDocument();
      });
    });

    it('displays average latency', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('12.5ms avg')).toBeInTheDocument();
      });
    });

    it('displays most retrieved topics', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('ethics')).toBeInTheDocument();
        expect(screen.getByText('technology')).toBeInTheDocument();
        expect(screen.getByText('climate')).toBeInTheDocument();
      });
    });

    it('limits topics display to 5', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('healthcare')).toBeInTheDocument();
      });

      // All 5 topics should be visible
      const topics = ['ethics', 'technology', 'climate', 'economics', 'healthcare'];
      topics.forEach(topic => {
        expect(screen.getByText(topic)).toBeInTheDocument();
      });
    });
  });

  describe('Recommendations', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockAnalyticsData),
      });
    });

    it('displays recommendations section', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('RECOMMENDATIONS')).toBeInTheDocument();
      });
    });

    it('displays recommendation messages', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('Consider promoting high-access slow tier memories')).toBeInTheDocument();
        expect(screen.getByText('Archive 50 low-access glacial memories')).toBeInTheDocument();
      });
    });

    it('does not show recommendations section when empty', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          ...mockAnalyticsData,
          recommendations: [],
        }),
      });

      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('TIER DISTRIBUTION')).toBeInTheDocument();
      });

      expect(screen.queryByText('RECOMMENDATIONS')).not.toBeInTheDocument();
    });

    it('limits displayed recommendations to 3', async () => {
      const manyRecommendations = [
        { type: 'a', message: 'Recommendation 1', priority: 'high' as const },
        { type: 'b', message: 'Recommendation 2', priority: 'medium' as const },
        { type: 'c', message: 'Recommendation 3', priority: 'low' as const },
        { type: 'd', message: 'Recommendation 4', priority: 'low' as const },
      ];

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          ...mockAnalyticsData,
          recommendations: manyRecommendations,
        }),
      });

      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('Recommendation 1')).toBeInTheDocument();
        expect(screen.getByText('Recommendation 2')).toBeInTheDocument();
        expect(screen.getByText('Recommendation 3')).toBeInTheDocument();
      });

      expect(screen.queryByText('Recommendation 4')).not.toBeInTheDocument();
    });
  });

  describe('Summary Footer', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockAnalyticsData),
      });
    });

    it('displays active memories count', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('890 active')).toBeInTheDocument();
      });
    });

    it('displays analysis period', async () => {
      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('30-day analysis')).toBeInTheDocument();
      });
    });
  });

  describe('Refresh Functionality', () => {
    it('refetches data when refresh button clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockAnalyticsData),
      });

      render(<MemoryAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('MEMORY ANALYTICS')).toBeInTheDocument();
      });

      expect(mockFetch).toHaveBeenCalledTimes(1);

      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      fireEvent.click(refreshButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe('Custom API Base', () => {
    it('uses custom API base when provided', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockAnalyticsData),
      });

      render(<MemoryAnalyticsPanel apiBase="https://custom-api.example.com" />);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('https://custom-api.example.com/api/memory/analytics')
        );
      });
    });
  });
});
