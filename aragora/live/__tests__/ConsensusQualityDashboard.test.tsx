/**
 * Tests for ConsensusQualityDashboard component
 *
 * Tests cover:
 * - Loading state
 * - Error handling
 * - Empty state when no debates
 * - Quality score display with color coding
 * - Consensus rate and confidence metrics
 * - Trend indicator
 * - Alert banner display
 * - Confidence history chart
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ConsensusQualityDashboard } from '../src/components/ConsensusQualityDashboard';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

const mockQualityData = {
  stats: {
    total_debates: 50,
    confidence_history: [
      { debate_id: 'd1', confidence: 0.85, consensus_reached: true, timestamp: '2026-01-10T10:00:00Z' },
      { debate_id: 'd2', confidence: 0.72, consensus_reached: true, timestamp: '2026-01-10T09:00:00Z' },
      { debate_id: 'd3', confidence: 0.45, consensus_reached: false, timestamp: '2026-01-10T08:00:00Z' },
    ],
    trend: 'improving',
    average_confidence: 0.78,
    consensus_rate: 0.84,
    consensus_reached_count: 42,
  },
  quality_score: 82,
  alert: null,
};

const mockQualityDataWithAlert = {
  ...mockQualityData,
  quality_score: 35,
  alert: {
    level: 'warning',
    message: 'Consensus rate declining over the past week',
  },
};

describe('ConsensusQualityDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  describe('Loading State', () => {
    it('shows loading state initially', () => {
      mockFetch.mockImplementation(() => new Promise(() => {}));

      render(<ConsensusQualityDashboard />);

      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('displays error message when fetch fails', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/failed to fetch/i)).toBeInTheDocument();
      });
    });

    it('displays error for non-ok response', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 500 });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/failed to fetch consensus quality/i)).toBeInTheDocument();
      });
    });
  });

  describe('Empty State', () => {
    it('shows empty message when no debates exist', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          stats: { total_debates: 0, confidence_history: [], trend: 'insufficient_data', average_confidence: 0, consensus_rate: 0, consensus_reached_count: 0 },
          quality_score: 0,
          alert: null,
        }),
      });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/no debate data/i)).toBeInTheDocument();
      });
    });
  });

  describe('Quality Score Display', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockQualityData),
      });
    });

    it('displays quality score prominently', async () => {
      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('82')).toBeInTheDocument();
        expect(screen.getByText('Quality Score')).toBeInTheDocument();
      });
    });

    it('applies green color for high scores (>=80)', async () => {
      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        const scoreElement = screen.getByText('82');
        expect(scoreElement).toHaveClass('text-green-400');
      });
    });

    it('applies yellow color for medium scores (60-79)', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ ...mockQualityData, quality_score: 65 }),
      });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        const scoreElement = screen.getByText('65');
        expect(scoreElement).toHaveClass('text-yellow-400');
      });
    });

    it('applies red color for low scores (<40)', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ ...mockQualityData, quality_score: 25 }),
      });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        const scoreElement = screen.getByText('25');
        expect(scoreElement).toHaveClass('text-red-400');
      });
    });
  });

  describe('Key Metrics', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockQualityData),
      });
    });

    it('displays consensus rate', async () => {
      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('84%')).toBeInTheDocument();
        expect(screen.getByText('Consensus Rate')).toBeInTheDocument();
      });
    });

    it('displays average confidence', async () => {
      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('78%')).toBeInTheDocument();
        expect(screen.getByText('Avg Confidence')).toBeInTheDocument();
      });
    });

    it('displays trend indicator', async () => {
      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/improving/i)).toBeInTheDocument();
        expect(screen.getByText('📈')).toBeInTheDocument();
      });
    });
  });

  describe('Trend Indicator', () => {
    it('shows green color for improving trend', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockQualityData),
      });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        const trendElement = screen.getByText(/improving/i).closest('div');
        expect(trendElement).toHaveClass('text-green-400');
      });
    });

    it('shows red color for declining trend', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          ...mockQualityData,
          stats: { ...mockQualityData.stats, trend: 'declining' },
        }),
      });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('📉')).toBeInTheDocument();
      });
    });

    it('shows stable indicator', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          ...mockQualityData,
          stats: { ...mockQualityData.stats, trend: 'stable' },
        }),
      });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('➡️')).toBeInTheDocument();
      });
    });
  });

  describe('Alert Banner', () => {
    it('displays warning alert when present', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockQualityDataWithAlert),
      });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('⚠️')).toBeInTheDocument();
        expect(screen.getByText('Consensus rate declining over the past week')).toBeInTheDocument();
      });
    });

    it('displays critical alert with appropriate styling', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          ...mockQualityData,
          alert: { level: 'critical', message: 'System error detected' },
        }),
      });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('🚨')).toBeInTheDocument();
        expect(screen.getByText('System error detected')).toBeInTheDocument();
      });
    });

    it('does not display alert section when no alert', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockQualityData),
      });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('82')).toBeInTheDocument();
      });

      expect(screen.queryByText('🚨')).not.toBeInTheDocument();
      expect(screen.queryByText('⚠️')).not.toBeInTheDocument();
    });
  });

  describe('Confidence History Chart', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockQualityData),
      });
    });

    it('displays confidence history section', async () => {
      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('CONFIDENCE HISTORY')).toBeInTheDocument();
      });
    });

    it('shows chart axis labels', async () => {
      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Older')).toBeInTheDocument();
        expect(screen.getByText('Recent')).toBeInTheDocument();
      });
    });
  });

  describe('Summary Stats', () => {
    beforeEach(() => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockQualityData),
      });
    });

    it('displays total debates count', async () => {
      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('50 total debates')).toBeInTheDocument();
      });
    });

    it('displays consensus reached count', async () => {
      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('42 reached consensus')).toBeInTheDocument();
      });
    });
  });

  describe('Refresh Functionality', () => {
    it('refetches data when refresh button clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockQualityData),
      });

      render(<ConsensusQualityDashboard />);

      await waitFor(() => {
        expect(screen.getByText('82')).toBeInTheDocument();
      });

      expect(mockFetch).toHaveBeenCalledTimes(1);

      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      fireEvent.click(refreshButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledTimes(2);
      });
    });
  });
});
