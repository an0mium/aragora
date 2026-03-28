import { renderHook } from '@testing-library/react';

import { useUsageDashboard } from '@/hooks/useUsageDashboard';
import { useSWRFetch } from '@/hooks/useSWRFetch';

jest.mock('@/hooks/useSWRFetch', () => ({
  useSWRFetch: jest.fn(),
}));

const mockUseSWRFetch = useSWRFetch as jest.Mock;

describe('useUsageDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockUseSWRFetch.mockImplementation((endpoint: string) => {
      if (endpoint === '/api/v1/usage/summary?range=30d') {
        return {
          data: {
            data: {
              debates: {
                total: 128,
                completed: 120,
                today: 6,
                this_week: 24,
                this_month: 96,
              },
              tokens: {
                total_in: 50000,
                total_out: 22000,
                today: 15420,
                this_week: 84100,
              },
              costs: {
                total_usd: 432.1,
                today_usd: 19.45,
                this_week_usd: 88.12,
                this_month_usd: 190.22,
              },
              consensus: {
                rate: 0.74,
                avg_confidence: 0.83,
                avg_time_seconds: 372,
              },
              active_agents: 6,
              period_start: '2026-03-01T00:00:00Z',
              period_end: '2026-03-28T00:00:00Z',
            },
          },
          error: null,
          isLoading: false,
        };
      }

      if (endpoint === '/api/v1/usage/roi?range=30d') {
        return {
          data: {
            data: {
              roi_percentage: 120,
              time_saved_hours: 48,
              cost_savings_usd: 1400,
              manual_equivalent_hours: 60,
              cost_per_decision: 3.38,
              value_generated_usd: 2200,
              benchmark: {
                industry: 'software',
                avg_roi: 70,
                percentile: 91,
              },
              trends: {
                roi_trend: 'increasing',
                efficiency_trend: 'improving',
              },
            },
          },
          error: null,
          isLoading: false,
        };
      }

      if (endpoint === '/api/v1/usage/budget-status') {
        return {
          data: {
            data: {
              monthly_limit_usd: 1000,
              spent_usd: 432.1,
              remaining_usd: 567.9,
              utilization_percent: 43.2,
              projected_end_of_month_usd: 515,
              will_exceed: false,
              alert_level: 'normal',
              daily_average_usd: 15.4,
              days_remaining: 3,
            },
          },
          error: null,
          isLoading: false,
        };
      }

      if (endpoint === '/api/v1/usage/forecast') {
        return {
          data: {
            data: {
              projected_monthly_tokens: 160000,
              projected_monthly_cost_usd: 515,
              projected_monthly_debates: 140,
              growth_rate_percent: 8.4,
              trend: 'increasing',
              confidence: 0.72,
              recommendations: [],
            },
          },
          error: null,
          isLoading: false,
        };
      }

      return {
        data: null,
        error: null,
        isLoading: false,
      };
    });
  });

  it('maps total debates, average confidence, and total spend into dashboard data', () => {
    const { result } = renderHook(() => useUsageDashboard());

    expect(result.current.dashboardData).toMatchObject({
      debates: {
        total: 128,
        completed: 120,
      },
      consensus: {
        rate: 0.74,
        avgConfidence: 0.83,
      },
      costs: {
        estimatedCost: 19.45,
        totalCost: 432.1,
      },
    });
  });

  it('returns null dashboard data until usage summary is available', () => {
    mockUseSWRFetch.mockReturnValue({
      data: null,
      error: null,
      isLoading: false,
    });

    const { result } = renderHook(() => useUsageDashboard());

    expect(result.current.dashboardData).toBeNull();
  });
});
