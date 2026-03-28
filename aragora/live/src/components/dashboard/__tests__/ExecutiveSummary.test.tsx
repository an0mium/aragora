import { render, screen } from '@testing-library/react';

import { ExecutiveSummary } from '../ExecutiveSummary';

const mockUseUsageDashboard = jest.fn();
const mockUseSWRFetch = jest.fn();

jest.mock('@/hooks/useUsageDashboard', () => ({
  useUsageDashboard: (...args: unknown[]) => mockUseUsageDashboard(...args),
}));

jest.mock('@/hooks/useSWRFetch', () => ({
  useSWRFetch: (...args: unknown[]) => mockUseSWRFetch(...args),
}));

describe('ExecutiveSummary', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockUseUsageDashboard.mockReturnValue({
      dashboardData: {
        debates: {
          today: 3,
          week: 10,
          month: 24,
          total: 42,
          completed: 38,
        },
        consensus: {
          rate: 0.71,
          avgConfidence: 0.84,
          avgTimeToDecision: 512,
        },
        costs: {
          todayTokens: 1200,
          weekTokens: 5600,
          estimatedCost: 12.34,
          totalCost: 456.78,
          monthlyCost: 456.78,
        },
        agents: {
          active: 7,
        },
        roi: null,
        budget: null,
        forecast: null,
        lastUpdated: '2026-03-28T12:00:00.000Z',
      },
      isLoading: false,
      error: null,
    });

    mockUseSWRFetch.mockReturnValue({
      data: {
        agents: [
          { name: 'claude-opus', elo: 1742, win_rate: 0.78 },
          { name: 'gpt-4.1', elo: 1698, win_rate: 0.73 },
          { name: 'gemini-pro', elo: 1655, win_rate: 0.69 },
        ],
      },
      isLoading: false,
      error: null,
    });
  });

  it('shows total debates, average confidence, top agents, and total spend', () => {
    render(<ExecutiveSummary refreshInterval={15000} />);

    expect(screen.getByText('Total Debates')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('Avg Confidence')).toBeInTheDocument();
    expect(screen.getByText('84%')).toBeInTheDocument();
    expect(screen.getByText('Total Spend')).toBeInTheDocument();
    expect(screen.getByText('$456.78')).toBeInTheDocument();
    expect(screen.getByText(/TOP AGENTS/i)).toBeInTheDocument();
    expect(screen.getByText('claude-opus')).toBeInTheDocument();
    expect(screen.getByText('1742 ELO | 78%')).toBeInTheDocument();
    expect(screen.getByText('gpt-4.1')).toBeInTheDocument();
    expect(screen.getByText('gemini-pro')).toBeInTheDocument();
  });

  it('falls back cleanly when leaderboard data is unavailable', () => {
    mockUseSWRFetch.mockReturnValue({
      data: null,
      isLoading: false,
      error: new Error('unavailable'),
    });

    render(<ExecutiveSummary refreshInterval={15000} />);

    expect(screen.getByText('Top Agent')).toBeInTheDocument();
    expect(screen.getAllByText('-').length).toBeGreaterThan(0);
    expect(screen.getByText('Leaderboard unavailable')).toBeInTheDocument();
  });
});
