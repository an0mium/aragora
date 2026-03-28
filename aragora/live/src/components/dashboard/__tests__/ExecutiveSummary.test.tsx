import { render, screen } from '@testing-library/react';

import { ExecutiveSummary } from '../ExecutiveSummary';

jest.mock('@/hooks/useUsageDashboard', () => ({
  useUsageDashboard: jest.fn(),
}));

const mockUseUsageDashboard = jest.requireMock('@/hooks/useUsageDashboard')
  .useUsageDashboard as jest.Mock;

describe('ExecutiveSummary', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders debates run, average confidence, best agent, and total spend', () => {
    mockUseUsageDashboard.mockReturnValue({
      dashboardData: {
        debates: {
          total: 42,
          completed: 40,
          periodLabel: 'month',
        },
        consensus: {
          rate: 0.875,
          avgConfidence: 0.92,
          avgTimeToDecision: 390,
        },
        costs: {
          totalCost: 12.5,
          avgPerDebate: 0.31,
          totalApiCalls: 210,
          totalTokens: 420000,
        },
        agents: {
          active: 2,
          total: 2,
          topPerformer: 'Claude Opus',
        },
        roi: null,
        budget: null,
        forecast: null,
      },
      isLoading: false,
      error: null,
    });

    render(<ExecutiveSummary refreshInterval={30000} />);

    expect(screen.getByText('Debates Run')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('Avg Confidence')).toBeInTheDocument();
    expect(screen.getByText('92%')).toBeInTheDocument();
    expect(screen.getByText('Best Agent')).toBeInTheDocument();
    expect(screen.getAllByText('Claude Opus')).not.toHaveLength(0);
    expect(screen.getByText('Total Spend')).toBeInTheDocument();
    expect(screen.getByText('$12.50')).toBeInTheDocument();
    expect(screen.getByText('Top Performer')).toBeInTheDocument();
  });
});
