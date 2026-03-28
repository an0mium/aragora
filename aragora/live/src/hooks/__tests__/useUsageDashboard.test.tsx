import { renderHook } from '@testing-library/react';

import { useUsageDashboard } from '../useUsageDashboard';
import { useSWRFetch } from '../useSWRFetch';

jest.mock('../useSWRFetch', () => ({
  useSWRFetch: jest.fn(),
}));

const mockUseSWRFetch = useSWRFetch as jest.MockedFunction<typeof useSWRFetch>;

type MockSWRResult = ReturnType<typeof useSWRFetch>;

function makeSWRResult(data: unknown): MockSWRResult {
  return {
    data,
    error: null,
    isLoading: false,
    isValidating: false,
    mutate: jest.fn(),
  };
}

describe('useUsageDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('normalizes the executive dashboard metrics from the backend endpoints', () => {
    mockUseSWRFetch.mockImplementation((endpoint) => {
      switch (endpoint) {
        case '/api/v1/usage/summary?period=month':
          return makeSWRResult({
            data: {
              period: {
                type: 'month',
                start: '2026-03-01T00:00:00Z',
                end: '2026-03-31T00:00:00Z',
                days: 30,
              },
              debates: {
                total: 42,
                completed: 40,
                consensus_rate: 87.5,
              },
              costs: {
                total_usd: '12.50',
                avg_per_debate_usd: '0.31',
                by_provider: {},
              },
              tokens: {
                total: 420000,
                input: 300000,
                output: 120000,
              },
              activity: {
                active_days: 20,
                debates_per_day: 1.4,
                api_calls: 210,
              },
            },
          });
        case '/api/v1/usage/roi?period=month':
          return makeSWRResult({
            data: {
              time_savings: {
                estimated_hours_saved: 18.5,
                avg_debate_duration_minutes: 6.5,
              },
              cost: {
                total_manual_cost_usd: '96.00',
                cost_savings_usd: '83.50',
                cost_per_decision_usd: '0.31',
              },
              roi: {
                roi_percentage: 668.0,
              },
              quality: {
                consensus_rate: 87.5,
                avg_confidence_score: 0.92,
              },
              benchmark: {
                cost_usd: '200.00',
              },
            },
          });
        case '/api/v1/usage/budget-status':
          return makeSWRResult({
            data: {
              monthly: {
                limit_usd: '100.00',
                spent_usd: '12.50',
                remaining_usd: '87.50',
                percent_used: 12.5,
                days_remaining: 3,
                projected_end_spend_usd: '14.00',
              },
              alert_level: 'normal',
            },
          });
        case '/api/v1/usage/forecast':
          return makeSWRResult({
            data: {
              projections: {
                debates_per_month: 45,
                monthly: {
                  aragora_cost_usd: '14.00',
                },
              },
            },
          });
        case '/api/v1/outcome-dashboard/agents?period=30d':
          return makeSWRResult({
            data: {
              agents: [
                {
                  rank: 1,
                  agent_id: 'claude-opus',
                  agent_name: 'Claude Opus',
                  provider: 'anthropic',
                  model: 'claude-opus-4-1',
                  elo: 1847,
                  debates: 42,
                },
                {
                  rank: 2,
                  agent_id: 'gpt-4o',
                  agent_name: 'GPT-4o',
                  provider: 'openai',
                  model: 'gpt-4o',
                  elo: 1792,
                  debates: 38,
                },
              ],
              count: 2,
              period: '30d',
            },
          });
        default:
          return makeSWRResult(null);
      }
    });

    const { result } = renderHook(() => useUsageDashboard('30d'));

    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v1/usage/summary?period=month',
      undefined,
    );
    expect(mockUseSWRFetch).toHaveBeenCalledWith(
      '/api/v1/outcome-dashboard/agents?period=30d',
      undefined,
    );
    expect(result.current.dashboardData).toMatchObject({
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
        totalTokens: 420000,
        totalApiCalls: 210,
        totalCost: 12.5,
        avgPerDebate: 0.31,
      },
      agents: {
        active: 2,
        total: 2,
        topPerformer: 'Claude Opus',
      },
      budget: {
        limit: 100,
        spent: 12.5,
        remaining: 87.5,
        utilization: 12.5,
        projectedTotal: 14,
        willExceed: false,
        alertLevel: 'normal',
        daysRemaining: 3,
      },
      forecast: {
        monthlyCost: 14,
        monthlyDebates: 45,
      },
    });
  });

  it('falls back cleanly when ranked agents are unavailable', () => {
    mockUseSWRFetch.mockImplementation((endpoint) => {
      switch (endpoint) {
        case '/api/v1/usage/summary?period=day':
          return makeSWRResult({
            data: {
              period: {
                type: 'day',
                start: '2026-03-28T00:00:00Z',
                end: '2026-03-29T00:00:00Z',
                days: 1,
              },
              debates: {
                total: 3,
                completed: 2,
                consensus_rate: 50,
              },
              costs: {
                total_usd: '4.20',
                avg_per_debate_usd: '2.10',
                by_provider: {},
              },
              tokens: {
                total: 12000,
                input: 8000,
                output: 4000,
              },
              activity: {
                active_days: 1,
                debates_per_day: 3,
                api_calls: 15,
              },
            },
          });
        case '/api/v1/usage/roi?period=day':
          return makeSWRResult({
            data: {
              quality: {
                consensus_rate: 50,
                avg_confidence_score: 0.67,
              },
            },
          });
        case '/api/v1/usage/budget-status':
          return makeSWRResult({
            data: {
              monthly: {
                limit_usd: 'unlimited',
                spent_usd: '4.20',
                remaining_usd: 'unlimited',
                percent_used: 0,
                days_remaining: 3,
                projected_end_spend_usd: '4.20',
              },
              alert_level: null,
            },
          });
        case '/api/v1/usage/forecast':
          return makeSWRResult({ data: {} });
        case '/api/v1/outcome-dashboard/agents?period=24h':
          return makeSWRResult({
            data: {
              agents: [],
              count: 0,
              period: '24h',
            },
          });
        default:
          return makeSWRResult(null);
      }
    });

    const { result } = renderHook(() => useUsageDashboard('24h'));

    expect(result.current.dashboardData?.agents.topPerformer).toBe('-');
    expect(result.current.dashboardData?.agents.total).toBe(0);
    expect(result.current.dashboardData?.budget?.limit).toBe(0);
    expect(result.current.dashboardData?.budget?.alertLevel).toBe('normal');
  });
});
