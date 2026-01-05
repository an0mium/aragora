/**
 * Tests for LeaderboardPanel component
 *
 * Tests cover:
 * - Rendering with mock data
 * - Tab switching functionality
 * - Error handling and display
 * - Domain filtering
 * - Consistency score display
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { LeaderboardPanel } from '../src/components/LeaderboardPanel';

// Mock next/link
jest.mock('next/link', () => {
  return ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  );
});

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

const mockLeaderboardData = {
  agents: [
    { name: 'claude-3-opus', elo: 1650, wins: 15, losses: 5, draws: 2, win_rate: 75, games: 22, consistency: 0.85, consistency_class: 'high' },
    { name: 'gemini-2.0-flash', elo: 1580, wins: 12, losses: 8, draws: 3, win_rate: 60, games: 23, consistency: 0.72, consistency_class: 'medium' },
    { name: 'grok-2', elo: 1520, wins: 10, losses: 10, draws: 2, win_rate: 50, games: 22, consistency: 0.55, consistency_class: 'low' },
  ],
  domains: ['technology', 'philosophy', 'science'],
};

const mockMatchesData = {
  matches: [
    {
      debate_id: 'debate-1',
      winner: 'claude-3-opus',
      participants: ['claude-3-opus', 'gemini-2.0-flash'],
      domain: 'technology',
      elo_changes: { 'claude-3-opus': 15, 'gemini-2.0-flash': -15 },
      created_at: '2024-01-15T10:30:00Z',
    },
  ],
};

const mockReputationData = {
  reputations: [
    { agent: 'claude-3-opus', score: 0.85, vote_weight: 1.2, proposal_acceptance_rate: 0.75, critique_value: 0.9, debates_participated: 22 },
  ],
};

const mockTeamsData = {
  combinations: [
    { agents: ['claude-3-opus', 'gemini-2.0-flash'], success_rate: 0.78, total_debates: 9, wins: 7 },
  ],
};

const mockStatsData = {
  mean_elo: 1550,
  median_elo: 1520,
  total_agents: 8,
  total_matches: 45,
  rating_distribution: { '1600': 2, '1500': 4, '1400': 2 },
  trending_up: ['claude-3-opus'],
  trending_down: ['grok-2'],
};

const mockIntrospectionData = {
  agents: [
    {
      agent: 'claude-3-opus',
      self_model: { strengths: ['reasoning'], weaknesses: ['verbosity'], biases: ['formality'] },
      confidence_calibration: 0.82,
      recent_performance_assessment: 'Performing well in technical debates',
      improvement_focus: ['conciseness'],
      last_updated: '2024-01-15T10:00:00Z',
    },
  ],
};

function setupSuccessfulFetch() {
  mockFetch.mockImplementation((url: string) => {
    if (url.includes('/api/leaderboard')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockLeaderboardData) });
    }
    if (url.includes('/api/matches/recent')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockMatchesData) });
    }
    if (url.includes('/api/reputation/all')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockReputationData) });
    }
    if (url.includes('/api/routing/best-teams')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockTeamsData) });
    }
    if (url.includes('/api/ranking/stats')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockStatsData) });
    }
    if (url.includes('/api/introspection/all')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(mockIntrospectionData) });
    }
    return Promise.resolve({ ok: false, text: () => Promise.resolve('Not found') });
  });
}

describe('LeaderboardPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('renders loading state initially', () => {
    mockFetch.mockImplementation(() => new Promise(() => {})); // Never resolves
    render(<LeaderboardPanel apiBase="http://localhost:3001" />);
    expect(screen.getByText('Loading rankings...')).toBeInTheDocument();
  });

  it('renders agent rankings after data loads', async () => {
    setupSuccessfulFetch();
    render(<LeaderboardPanel apiBase="http://localhost:3001" />);

    await waitFor(() => {
      expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
    });

    expect(screen.getByText('gemini-2.0-flash')).toBeInTheDocument();
    expect(screen.getByText('grok-2')).toBeInTheDocument();
    expect(screen.getByText('1650')).toBeInTheDocument(); // ELO score
  });

  it('displays consistency scores for agents', async () => {
    setupSuccessfulFetch();
    render(<LeaderboardPanel apiBase="http://localhost:3001" />);

    await waitFor(() => {
      expect(screen.getByText('85%')).toBeInTheDocument(); // claude consistency
    });

    expect(screen.getByText('72%')).toBeInTheDocument(); // gemini consistency
    expect(screen.getByText('55%')).toBeInTheDocument(); // grok consistency
  });

  it('switches between tabs correctly', async () => {
    setupSuccessfulFetch();
    render(<LeaderboardPanel apiBase="http://localhost:3001" />);

    await waitFor(() => {
      expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
    });

    // Switch to Matches tab
    fireEvent.click(screen.getByText('Matches'));
    await waitFor(() => {
      expect(screen.getByText('claude-3-opus wins')).toBeInTheDocument();
    });

    // Switch to Stats tab
    fireEvent.click(screen.getByText('Stats'));
    await waitFor(() => {
      expect(screen.getByText('Mean ELO')).toBeInTheDocument();
      expect(screen.getByText('1550')).toBeInTheDocument();
    });

    // Switch to Minds tab
    fireEvent.click(screen.getByText('Minds'));
    await waitFor(() => {
      expect(screen.getByText('82% calibrated')).toBeInTheDocument();
    });
  });

  it('shows domain filter when domains are available', async () => {
    setupSuccessfulFetch();
    render(<LeaderboardPanel apiBase="http://localhost:3001" />);

    await waitFor(() => {
      expect(screen.getByText('Domain:')).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    expect(select).toBeInTheDocument();
    expect(screen.getByText('All Domains')).toBeInTheDocument();
  });

  it('handles partial endpoint failures gracefully', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/api/leaderboard')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(mockLeaderboardData) });
      }
      if (url.includes('/api/matches/recent')) {
        return Promise.resolve({ ok: false, text: () => Promise.resolve('503 Service Unavailable') });
      }
      if (url.includes('/api/reputation/all')) {
        return Promise.resolve({ ok: false, text: () => Promise.resolve('503 Service Unavailable') });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    render(<LeaderboardPanel apiBase="http://localhost:3001" />);

    await waitFor(() => {
      // Should still show rankings even if other endpoints fail
      expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
    });

    // Should show partial error
    expect(screen.getByText(/endpoint\(s\) unavailable/)).toBeInTheDocument();
  });

  it('handles complete API failure', async () => {
    mockFetch.mockImplementation(() => {
      return Promise.resolve({ ok: false, text: () => Promise.resolve('500 Internal Server Error') });
    });

    render(<LeaderboardPanel apiBase="http://localhost:3001" />);

    await waitFor(() => {
      expect(screen.getByText('All endpoints failed. Check server connection.')).toBeInTheDocument();
    });
  });

  it('shows error details when expanded', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/api/leaderboard')) {
        return Promise.resolve({ ok: false, text: () => Promise.resolve('503: ELO system not available') });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    render(<LeaderboardPanel apiBase="http://localhost:3001" />);

    await waitFor(() => {
      expect(screen.getByText('Show details')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Show details'));
    expect(screen.getByText(/rankings:/)).toBeInTheDocument();
  });

  it('refreshes data when refresh button is clicked', async () => {
    setupSuccessfulFetch();
    render(<LeaderboardPanel apiBase="http://localhost:3001" />);

    await waitFor(() => {
      expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
    });

    expect(mockFetch).toHaveBeenCalledTimes(6); // 6 endpoints

    fireEvent.click(screen.getByText('Refresh'));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(12); // 6 more calls
    });
  });

  it('auto-refreshes every 30 seconds', async () => {
    setupSuccessfulFetch();
    render(<LeaderboardPanel apiBase="http://localhost:3001" />);

    await waitFor(() => {
      expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
    });

    expect(mockFetch).toHaveBeenCalledTimes(6);

    // Advance timer by 30 seconds
    jest.advanceTimersByTime(30000);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(12);
    });
  });

  it('includes loopId in API requests when provided', async () => {
    setupSuccessfulFetch();
    render(<LeaderboardPanel apiBase="http://localhost:3001" loopId="test-loop-123" />);

    await waitFor(() => {
      expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
    });

    const leaderboardCall = mockFetch.mock.calls.find((call: string[]) =>
      call[0].includes('/api/leaderboard')
    );
    expect(leaderboardCall[0]).toContain('loop_id=test-loop-123');
  });
});
