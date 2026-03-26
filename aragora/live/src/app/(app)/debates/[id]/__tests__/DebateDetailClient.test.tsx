import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import DebateDetailClient from '../DebateDetailClient';

const mockFetch = jest.fn();
const mockSetContext = jest.fn();
const mockClearContext = jest.fn();
const mockGetAuthHeaders = jest.fn(() => ({
  'Content-Type': 'application/json',
  Authorization: 'Bearer test-token',
}));

global.fetch = mockFetch as unknown as typeof fetch;

jest.mock('next/navigation', () => ({
  useParams: () => ({ id: 'debate-123' }),
}));

jest.mock('next/link', () => ({
  __esModule: true,
  default: ({ href, children }: { href: string; children: React.ReactNode }) => (
    <a href={href}>{children}</a>
  ),
}));

jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => null,
  CRTVignette: () => null,
}));

jest.mock('@/context/RightSidebarContext', () => ({
  useRightSidebar: () => ({
    setContext: mockSetContext,
    clearContext: mockClearContext,
  }),
}));

jest.mock('@/components/BackendSelector', () => ({
  useBackend: () => ({
    config: {
      api: 'http://backend.test',
      ws: 'ws://backend.test',
    },
  }),
}));

jest.mock('@/hooks/useAuthenticatedFetch', () => ({
  useAuthFetch: () => ({
    getAuthHeaders: mockGetAuthHeaders,
  }),
}));

jest.mock('@/components/debates/DecisionPackageView', () => ({
  DecisionPackageView: () => <div data-testid="decision-package" />,
}));

jest.mock('@/components/debates/CostBreakdown', () => ({
  CostBreakdown: () => <div data-testid="cost-breakdown" />,
}));

jest.mock('@/components/debates/ArgumentGraph', () => ({
  ArgumentGraph: () => <div data-testid="argument-graph" />,
}));

jest.mock('@/components/ExplanationPanel', () => ({
  ExplanationPanel: () => <div data-testid="explanation-panel" />,
}));

jest.mock('@/components/debates/RelatedKnowledge', () => ({
  RelatedKnowledge: () => <div data-testid="related-knowledge" />,
}));

jest.mock('@/components/debate-viewer/InterventionPanel', () => ({
  InterventionPanel: () => <div data-testid="intervention-panel" />,
}));

jest.mock('@/hooks/debate-websocket', () => ({
  useDebateWebSocket: () => ({
    messages: [],
    status: 'closed',
  }),
}));

jest.mock('@/components/debate/LiveDebateStream', () => ({
  LiveDebateStream: () => <div data-testid="live-debate-stream" />,
}));

jest.mock('../normalizeDecisionPackage', () => ({
  normalizeDecisionPackage: (data: unknown) => data,
}));

function jsonResponse(data: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: async () => data,
    text: async () => JSON.stringify(data),
  } as Response;
}

describe('DebateDetailClient bridge actions', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockFetch.mockImplementation((input: string | URL | Request) => {
      const url = String(input);

      if (url === 'http://backend.test/api/v1/debates/debate-123') {
        return Promise.resolve(jsonResponse({ status: 'completed' }));
      }

      if (url === 'http://backend.test/api/v1/debates/debate-123/package') {
        return Promise.resolve(
          jsonResponse({
            id: 'debate-123',
            question: 'Should we bridge this decision?',
            verdict: 'Yes',
            confidence: 0.93,
            consensus_reached: true,
            agents: ['claude', 'codex'],
            rounds: 3,
            duration_seconds: 42,
            final_answer: 'Bridge it.',
            receipt: { signers: ['sig-1'] },
            cost_breakdown: null,
            total_cost: 0,
          }),
        );
      }

      if (url === 'http://backend.test/api/v1/debates/debate-123/bridge') {
        return Promise.resolve(jsonResponse({ success: true }));
      }

      return Promise.reject(new Error(`Unexpected fetch: ${url}`));
    });
  });

  it('sends authenticated headers when triggering the bridge action', async () => {
    const user = userEvent.setup();

    render(<DebateDetailClient />);

    await user.click(await screen.findByRole('button', { name: /export/i }));
    await user.click(screen.getByRole('button', { name: /create jira issues/i }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'http://backend.test/api/v1/debates/debate-123/bridge',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            Authorization: 'Bearer test-token',
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify({ target: 'jira' }),
        }),
      );
    });

    expect(mockGetAuthHeaders).toHaveBeenCalled();
    expect(await screen.findByText(/jira triggered/i)).toBeInTheDocument();
  });
});
