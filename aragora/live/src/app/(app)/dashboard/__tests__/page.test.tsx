import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';

import DashboardPage from '../page';

const mockPush = jest.fn();
const mockUseAuth = jest.fn();
const mockUseDashboardEvents = jest.fn();
const mockUseSWRFetch = jest.fn();
const mockUseActiveDebates = jest.fn();

jest.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}));

jest.mock('next/link', () => {
  return function MockLink({
    children,
    href,
    className,
  }: {
    children: React.ReactNode;
    href: string;
    className?: string;
  }) {
    return (
      <a href={href} className={className}>
        {children}
      </a>
    );
  };
});

jest.mock('@/context/AuthContext', () => ({
  useAuth: () => mockUseAuth(),
}));

jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => <div data-testid="scanlines" />,
  CRTVignette: () => <div data-testid="crt-vignette" />,
}));

jest.mock('@/context/RightSidebarContext', () => ({
  useRightSidebar: () => ({
    setContext: jest.fn(),
    clearContext: jest.fn(),
  }),
}));

jest.mock('@/utils/supabase', () => ({
  fetchRecentDebates: jest.fn(async () => []),
}));

jest.mock('@/hooks/useDashboardEvents', () => ({
  useDashboardEvents: () => mockUseDashboardEvents(),
}));

jest.mock('@/hooks/useSWRFetch', () => ({
  useSWRFetch: (...args: unknown[]) => mockUseSWRFetch(...args),
  useActiveDebates: (...args: unknown[]) => mockUseActiveDebates(...args),
}));

jest.mock('@/components/dashboard/ExecutiveSummary', () => ({
  ExecutiveSummary: () => <div data-testid="executive-summary" />,
}));

jest.mock('@/components/dashboard/SettlementPanel', () => ({
  SettlementPanel: () => <div data-testid="settlement-panel" />,
}));

jest.mock('@/components/costs/CostSummaryWidget', () => ({
  CostSummaryWidget: () => <div data-testid="cost-summary-widget" />,
}));

jest.mock('@/components/billing/TrialStatusWidget', () => ({
  TrialStatusWidget: () => <div data-testid="trial-status-widget" />,
}));

jest.mock('@/components/templates/TemplateMarketplace', () => ({
  TemplateMarketplace: () => <div data-testid="template-marketplace" />,
}));

jest.mock('@/components/PanelErrorBoundary', () => ({
  PanelErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe('DashboardPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseDashboardEvents.mockReturnValue({ isConnected: false, updateCount: 0 });
    mockUseSWRFetch.mockReturnValue({ data: { debates: [] }, error: null, isLoading: false });
    mockUseActiveDebates.mockReturnValue({ data: { debates: [] }, isLoading: false });
  });

  it('gates unauthenticated users before dashboard hooks mount', async () => {
    mockUseAuth.mockReturnValue({
      isAuthenticated: false,
      isLoading: false,
      organization: null,
    });

    render(<DashboardPage />);

    expect(screen.getByText('AUTHENTICATION REQUIRED')).toBeInTheDocument();
    expect(screen.queryByTestId('executive-summary')).not.toBeInTheDocument();
    expect(mockUseDashboardEvents).not.toHaveBeenCalled();
    expect(mockUseSWRFetch).not.toHaveBeenCalled();
    expect(mockUseActiveDebates).not.toHaveBeenCalled();

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith('/auth/login?returnUrl=%2Fdashboard');
    });
  });

  it('renders dashboard content for authenticated users', () => {
    mockUseAuth.mockReturnValue({
      isAuthenticated: true,
      isLoading: false,
      organization: { tier: 'starter' },
    });

    render(<DashboardPage />);

    expect(screen.getByText('> EXECUTIVE DASHBOARD')).toBeInTheDocument();
    expect(screen.getByTestId('executive-summary')).toBeInTheDocument();
    expect(mockUseDashboardEvents).toHaveBeenCalled();
    expect(mockUseActiveDebates).toHaveBeenCalled();
    expect(mockUseSWRFetch).toHaveBeenCalled();
  });

  it('routes live and recent debate cards to the authenticated detail page', async () => {
    mockUseAuth.mockReturnValue({
      isAuthenticated: true,
      isLoading: false,
      organization: { tier: 'starter' },
    });

    mockUseActiveDebates.mockReturnValue({
      data: {
        debates: [
          {
            id: 'debate-live-123',
            topic: 'Keep internal debates on the private route',
            agents: ['claude', 'codex'],
            round: 1,
            total_rounds: 3,
            status: 'running',
            elapsed_seconds: 42,
          },
        ],
      },
      isLoading: false,
    });

    mockUseSWRFetch.mockImplementation((endpoint?: string) => {
      if (endpoint === '/api/v1/debates?limit=5&sort=created_at:desc') {
        return {
          data: {
            debates: [
              {
                id: 'debate-recent-456',
                task: 'Recent debate should stay inside the app shell',
                agents: ['gemini', 'gpt-4'],
                consensus_reached: true,
                confidence: 0.91,
                created_at: '2026-04-06T12:00:00Z',
              },
            ],
          },
          error: null,
          isLoading: false,
        };
      }

      return { data: {}, error: null, isLoading: false };
    });

    render(<DashboardPage />);

    await waitFor(() => {
      expect(
        screen.getByRole('link', { name: /Keep internal debates on the private route/i })
      ).toHaveAttribute('href', '/debates/debate-live-123');
    });

    expect(
      screen.getByRole('link', { name: /Recent debate should stay inside the app shell/i })
    ).toHaveAttribute('href', '/debates/debate-recent-456');
  });
});
