import type { ReactNode } from 'react';
import { renderWithProviders } from '@/test-utils';
import SpendDashboardPage from '../page';

const mockUseSpendDashboardSummary = jest.fn();
const mockUseSpendDashboardTrends = jest.fn();
const mockUseSpendDashboardByAgent = jest.fn();
const mockUseSpendDashboardByDecision = jest.fn();
const mockUseSpendDashboardBudget = jest.fn();

jest.mock('@/hooks/useSpendAnalytics', () => ({
  useSpendDashboardSummary: (...args: unknown[]) =>
    mockUseSpendDashboardSummary(...args),
  useSpendDashboardTrends: (...args: unknown[]) =>
    mockUseSpendDashboardTrends(...args),
  useSpendDashboardByAgent: (...args: unknown[]) =>
    mockUseSpendDashboardByAgent(...args),
  useSpendDashboardByDecision: (...args: unknown[]) =>
    mockUseSpendDashboardByDecision(...args),
  useSpendDashboardBudget: (...args: unknown[]) =>
    mockUseSpendDashboardBudget(...args),
}));

jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => null,
  CRTVignette: () => null,
}));

jest.mock('@/components/PanelErrorBoundary', () => ({
  PanelErrorBoundary: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/analytics', () => ({
  MetricCard: () => null,
  TrendChart: () => null,
  CostBreakdown: () => null,
}));

describe('SpendDashboardPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockUseSpendDashboardSummary.mockReturnValue({
      summary: {
        total_spend_usd: '0.00',
        total_api_calls: 0,
        total_tokens: 0,
        budget_limit_usd: 0,
        budget_spent_usd: 0,
        utilization_pct: 0,
        trend_direction: 'stable',
        avg_cost_per_decision: 0,
      },
      isLoading: false,
      error: null,
    });
    mockUseSpendDashboardTrends.mockReturnValue({
      trends: { org_id: 'org-live', period: 'daily', days: 30, data_points: [] },
      isLoading: false,
    });
    mockUseSpendDashboardByAgent.mockReturnValue({
      agentBreakdown: { workspace_id: 'org-live', total_usd: '0.00', agents: [] },
      isLoading: false,
    });
    mockUseSpendDashboardByDecision.mockReturnValue({
      decisionBreakdown: { workspace_id: 'org-live', decisions: [], count: 0 },
      isLoading: false,
    });
    mockUseSpendDashboardBudget.mockReturnValue({
      budget: {
        org_id: 'org-live',
        budgets: [],
        total_budget_usd: 0,
        total_spent_usd: 0,
        total_remaining_usd: 0,
        utilization_pct: 0,
        forecast_exhaustion_days: null,
      },
      isLoading: false,
    });
  });

  it('passes the authenticated org scope into spend dashboard hooks', () => {
    renderWithProviders(<SpendDashboardPage />, {
      authOverrides: {
        isAuthenticated: true,
        user: {
          id: 'user-1',
          email: 'user@example.com',
          name: 'User',
          role: 'member',
          org_id: 'org-live',
          is_active: true,
          created_at: '2026-03-30T00:00:00Z',
        },
        organization: {
          id: 'org-live',
          name: 'Live Org',
          slug: 'live-org',
          tier: 'enterprise',
          owner_id: 'user-1',
        },
      },
    });

    expect(mockUseSpendDashboardSummary).toHaveBeenCalledWith(
      'org-live',
      'org-live',
    );
    expect(mockUseSpendDashboardTrends).toHaveBeenCalledWith(
      'org-live',
      'daily',
      30,
    );
    expect(mockUseSpendDashboardByAgent).toHaveBeenCalledWith(
      'org-live',
      'org-live',
    );
    expect(mockUseSpendDashboardByDecision).toHaveBeenCalledWith(
      'org-live',
      'org-live',
    );
    expect(mockUseSpendDashboardBudget).toHaveBeenCalledWith('org-live');
  });
});
