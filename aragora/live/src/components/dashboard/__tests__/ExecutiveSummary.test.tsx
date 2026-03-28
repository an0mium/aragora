import { render, screen } from '@testing-library/react';

import { ExecutiveSummary } from '../ExecutiveSummary';
import { useUsageDashboard } from '@/hooks/useUsageDashboard';
import { useOutcomeAgents } from '@/hooks/useOutcomeAnalytics';

jest.mock('@/hooks/useUsageDashboard', () => ({
  useUsageDashboard: jest.fn(),
}));

jest.mock('@/hooks/useOutcomeAnalytics', () => ({
  useOutcomeAgents: jest.fn(),
}));

const mockUseUsageDashboard = useUsageDashboard as jest.Mock;
const mockUseOutcomeAgents = useOutcomeAgents as jest.Mock;

describe('ExecutiveSummary', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockUseUsageDashboard.mockReturnValue({
      dashboardData: {
        debates: {
          today: 6,
          week: 24,
          month: 96,
          total: 128,
          completed: 120,
        },
        consensus: {
          rate: 0.74,
          avgConfidence: 0.83,
          avgTimeToDecision: 372,
        },
        costs: {
          todayTokens: 15420,
          weekTokens: 84100,
          estimatedCost: 19.45,
          totalCost: 432.1,
          monthlyCost: 190.22,
        },
        agents: {
          active: 6,
          total: 15,
          topPerformer: 'Claude Opus',
          avgUptime: 99,
        },
        roi: null,
        budget: null,
        forecast: null,
        lastUpdated: '2026-03-28T12:00:00.000Z',
      },
      isLoading: false,
      error: null,
    });

    mockUseOutcomeAgents.mockReturnValue({
      leaderboard: {
        agents: [
          { agent_name: 'Claude Opus' },
          { agent_name: 'GPT-4.1' },
          { agent_name: 'Gemini Pro' },
        ],
      },
      isLoading: false,
      error: null,
    });
  });

  it('renders the dashboard metrics requested by the executive dashboard issue', () => {
    render(<ExecutiveSummary refreshInterval={15000} />);

    expect(screen.getByText('Debates Run')).toBeInTheDocument();
    expect(screen.getByText('128')).toBeInTheDocument();
    expect(screen.getByText('120 completed')).toBeInTheDocument();

    expect(screen.getByText('Avg Confidence')).toBeInTheDocument();
    expect(screen.getByText('83%')).toBeInTheDocument();
    expect(screen.getByText('74% consensus rate')).toBeInTheDocument();

    expect(screen.getByText('Total Spend')).toBeInTheDocument();
    expect(screen.getByText('$432.10')).toBeInTheDocument();

    expect(screen.getByText('BEST AGENTS')).toBeInTheDocument();
    expect(screen.getByText('Claude Opus, GPT-4.1, Gemini Pro')).toBeInTheDocument();
  });

  it('falls back to the usage summary top performer when leaderboard data is unavailable', () => {
    mockUseOutcomeAgents.mockReturnValue({
      leaderboard: null,
      isLoading: false,
      error: null,
    });

    render(<ExecutiveSummary />);

    expect(screen.getByText('Claude Opus')).toBeInTheDocument();
  });

  it('queries best agents with the same selected time range', () => {
    render(<ExecutiveSummary refreshInterval={15000} />);

    expect(mockUseOutcomeAgents).toHaveBeenCalledWith(
      '30d',
      expect.objectContaining({ refreshInterval: 120000 })
    );
  });
});
