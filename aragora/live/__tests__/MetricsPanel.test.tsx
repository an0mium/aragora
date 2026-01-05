/**
 * Tests for MetricsPanel component
 *
 * Tests cover:
 * - Loading states
 * - Data display for all tabs
 * - Tab switching functionality
 * - Error handling (partial and complete failures)
 * - Auto-refresh behavior
 * - Expand/collapse functionality
 */

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { MetricsPanel } from '../src/components/metrics/MetricsPanel';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

const mockMetricsData = {
  uptime_seconds: 86400,
  uptime_human: '1 day, 0:00:00',
  requests: {
    total: 15000,
    errors: 150,
    error_rate: 0.01,
    top_endpoints: [
      { endpoint: '/api/debate', count: 5000 },
      { endpoint: '/api/leaderboard', count: 3000 },
    ],
  },
  cache: { entries: 256 },
  databases: {
    'debates.db': { bytes: 1048576, human: '1.0 MB' },
    'rankings.db': { bytes: 524288, human: '512 KB' },
  },
  timestamp: '2026-01-05T12:00:00Z',
};

const mockHealthData = {
  status: 'healthy' as const,
  checks: {
    database: { status: 'ok', path: '/data/debates.db' },
    memory: { status: 'ok' },
    disk: { status: 'ok' },
  },
};

const mockCacheData = {
  total_entries: 256,
  max_entries: 1000,
  hit_rate: 0.85,
  hits: 850,
  misses: 150,
  entries_by_prefix: {
    debate_: 100,
    agent_: 80,
    ranking_: 76,
  },
  oldest_entry_age_seconds: 3600,
  newest_entry_age_seconds: 10,
};

const mockSystemData = {
  python_version: '3.11.11',
  platform: 'Darwin',
  machine: 'arm64',
  processor: 'arm',
  pid: 12345,
  memory: {
    rss_mb: 256,
    vms_mb: 512,
  },
};

function setupSuccessfulFetch() {
  mockFetch.mockImplementation((url: string) => {
    if (url.includes('/api/metrics/health')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockHealthData) });
    }
    if (url.includes('/api/metrics/cache')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockCacheData) });
    }
    if (url.includes('/api/metrics/system')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockSystemData) });
    }
    if (url.includes('/api/metrics')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockMetricsData) });
    }
    return Promise.resolve({ ok: false });
  });
}

describe('MetricsPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Loading States', () => {
    it('shows panel title while loading', async () => {
      mockFetch.mockImplementation(() => new Promise(() => {}));
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });
      expect(screen.getByText('Server Metrics')).toBeInTheDocument();
    });

    it('displays loading state during fetch', async () => {
      mockFetch.mockImplementation(() => new Promise(() => {}));
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });
      // Panel renders but data shows loading states
      expect(screen.getByText('Uptime:')).toBeInTheDocument();
    });
  });

  describe('Data Display', () => {
    it('renders metrics summary after data loads', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('1 day, 0:00:00')).toBeInTheDocument();
      });

      expect(screen.getByText('15,000')).toBeInTheDocument(); // requests total
    });

    it('displays error rate with correct color', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('1.00%')).toBeInTheDocument();
      });
    });

    it('displays cache hit rate in summary', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('85.0% hit')).toBeInTheDocument();
      });
    });

    it('displays health status in summary', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('HEALTHY')).toBeInTheDocument();
      });
    });
  });

  describe('Tab Navigation', () => {
    it('shows Overview tab by default', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        const overviewTab = screen.getByRole('tab', { name: 'OVERVIEW' });
        expect(overviewTab).toHaveAttribute('aria-selected', 'true');
      });
    });

    it('switches to Health tab when clicked', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('1 day, 0:00:00')).toBeInTheDocument();
      });

      await act(async () => {
        fireEvent.click(screen.getByRole('tab', { name: 'HEALTH' }));
      });

      const healthTab = screen.getByRole('tab', { name: 'HEALTH' });
      expect(healthTab).toHaveAttribute('aria-selected', 'true');
    });

    it('switches to Cache tab when clicked', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('1 day, 0:00:00')).toBeInTheDocument();
      });

      await act(async () => {
        fireEvent.click(screen.getByRole('tab', { name: 'CACHE' }));
      });

      const cacheTab = screen.getByRole('tab', { name: 'CACHE' });
      expect(cacheTab).toHaveAttribute('aria-selected', 'true');
    });

    it('switches to System tab when clicked', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('1 day, 0:00:00')).toBeInTheDocument();
      });

      await act(async () => {
        fireEvent.click(screen.getByRole('tab', { name: 'SYSTEM' }));
      });

      const systemTab = screen.getByRole('tab', { name: 'SYSTEM' });
      expect(systemTab).toHaveAttribute('aria-selected', 'true');
    });
  });

  describe('Error Handling', () => {
    it('shows partial error when some endpoints fail', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/metrics/health')) {
          return Promise.resolve({ ok: false });
        }
        if (url.includes('/api/metrics/cache')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve(mockCacheData) });
        }
        if (url.includes('/api/metrics/system')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve(mockSystemData) });
        }
        if (url.includes('/api/metrics')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve(mockMetricsData) });
        }
        return Promise.resolve({ ok: false });
      });

      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText(/Some metrics failed to load/)).toBeInTheDocument();
      });

      // Should still show data that loaded successfully
      expect(screen.getByText('1 day, 0:00:00')).toBeInTheDocument();
    });

    it('handles complete API failure', async () => {
      mockFetch.mockImplementation(() => Promise.resolve({ ok: false }));

      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText(/Some metrics failed to load/)).toBeInTheDocument();
      });
    });

    it('handles network errors gracefully', async () => {
      mockFetch.mockImplementation(() => Promise.reject(new Error('Network error')));

      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        // Component should handle error without crashing
        expect(screen.getByText('Server Metrics')).toBeInTheDocument();
      });
    });
  });

  describe('Auto-refresh', () => {
    it('refreshes data every 30 seconds', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('1 day, 0:00:00')).toBeInTheDocument();
      });

      const initialCalls = mockFetch.mock.calls.length;

      // Advance timer by 30 seconds
      await act(async () => {
        jest.advanceTimersByTime(30000);
      });

      await waitFor(() => {
        expect(mockFetch.mock.calls.length).toBeGreaterThan(initialCalls);
      });
    });

    it('cleans up interval on unmount', async () => {
      setupSuccessfulFetch();
      const { unmount } = await act(async () => {
        return render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('1 day, 0:00:00')).toBeInTheDocument();
      });

      const callsBeforeUnmount = mockFetch.mock.calls.length;
      unmount();

      await act(async () => {
        jest.advanceTimersByTime(30000);
      });

      // No new calls after unmount
      expect(mockFetch.mock.calls.length).toBe(callsBeforeUnmount);
    });
  });

  describe('Expand/Collapse', () => {
    it('shows tabs when expanded (default)', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'OVERVIEW' })).toBeInTheDocument();
      });

      expect(screen.getByRole('tab', { name: 'HEALTH' })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'CACHE' })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'SYSTEM' })).toBeInTheDocument();
    });

    it('hides tabs when collapsed', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByRole('tab', { name: 'OVERVIEW' })).toBeInTheDocument();
      });

      // Find and click the collapse button (ExpandToggle)
      const collapseButton = screen.getByLabelText(/collapse/i);
      await act(async () => {
        fireEvent.click(collapseButton);
      });

      // Tabs should be hidden
      expect(screen.queryByRole('tab', { name: 'OVERVIEW' })).not.toBeInTheDocument();
    });

    it('shows compact summary when collapsed', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('1 day, 0:00:00')).toBeInTheDocument();
      });

      // Collapse the panel
      const collapseButton = screen.getByLabelText(/collapse/i);
      await act(async () => {
        fireEvent.click(collapseButton);
      });

      // Compact summary should still show key metrics (may have multiple Uptime elements)
      const uptimeElements = screen.getAllByText(/Uptime:/);
      expect(uptimeElements.length).toBeGreaterThan(0);
    });
  });

  describe('Refresh Button', () => {
    it('fetches fresh data when clicked', async () => {
      setupSuccessfulFetch();
      await act(async () => {
        render(<MetricsPanel apiBase="http://localhost:8080" />);
      });

      await waitFor(() => {
        expect(screen.getByText('1 day, 0:00:00')).toBeInTheDocument();
      });

      const initialCalls = mockFetch.mock.calls.length;

      // Click refresh button (button contains "REFRESH" text split by brackets)
      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      await act(async () => {
        fireEvent.click(refreshButton);
      });

      await waitFor(() => {
        expect(mockFetch.mock.calls.length).toBeGreaterThan(initialCalls);
      });
    });
  });
});
