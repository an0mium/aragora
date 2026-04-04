import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';

import DeliberationsPage from '../page';

const mockFetch = jest.fn();
const captureAgentInfluence = jest.fn();

global.fetch = mockFetch as typeof fetch;

class MockWebSocket {
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;

  constructor(public readonly url: string) {}

  close() {}
}

Object.defineProperty(globalThis, 'WebSocket', {
  value: MockWebSocket,
  writable: true,
});

jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => null,
  CRTVignette: () => null,
}));

jest.mock('@/components/BackendSelector', () => ({
  useBackend: () => ({
    config: { api: 'http://backend.test' },
  }),
}));

jest.mock('@/components/PanelErrorBoundary', () => ({
  PanelErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/deliberation-dashboard', () => ({
  DeliberationGrid: () => <div data-testid="deliberation-grid" />,
  ConsensusFormationChart: () => <div data-testid="consensus-chart" />,
  AgentInfluenceNetwork: ({ agents }: { agents: Array<{ agent_id: string; message_count: number }> }) => {
    captureAgentInfluence(agents);
    return <div data-testid="agent-influence">{agents.map(agent => `${agent.agent_id}:${agent.message_count}`).join(',')}</div>;
  },
  DeliberationStats: () => <div data-testid="deliberation-stats" />,
}));

describe('DeliberationsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockFetch.mockImplementation((url: string) => {
      if (url.endsWith('/api/v1/deliberations/active')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            deliberations: [
              {
                id: 'debate-1',
                task: 'Ship the fix?',
                status: 'active',
                agents: ['claude', 'gpt-4'],
                current_round: 2,
                total_rounds: 3,
                consensus_score: 0.75,
                started_at: '2026-04-04T20:00:00Z',
                updated_at: '2026-04-04T20:01:00Z',
                message_count: 3,
                votes: { claude: 2, 'gpt-4': 1 },
              },
            ],
          }),
        });
      }

      if (url.endsWith('/api/v1/deliberations/stats')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            active_count: 1,
            completed_today: 0,
            average_consensus_time: 120,
            average_rounds: 2.5,
            top_agents: [
              {
                agent_id: 'claude',
                influence_score: 0.704,
                message_count: 2,
                consensus_contributions: 0.667,
                average_confidence: 0.85,
              },
            ],
          }),
        });
      }

      return Promise.reject(new Error(`Unexpected URL: ${url}`));
    });
  });

  it('renders handler-provided agent influence instead of synthesizing placeholder rows', async () => {
    render(<DeliberationsPage />);

    await waitFor(() => {
      expect(screen.getByTestId('agent-influence')).toHaveTextContent('claude:2');
    });

    expect(screen.getByTestId('agent-influence')).not.toHaveTextContent('gpt-4');
    expect(captureAgentInfluence).toHaveBeenLastCalledWith([
      {
        agent_id: 'claude',
        influence_score: 0.704,
        message_count: 2,
        consensus_contributions: 0.667,
        average_confidence: 0.85,
      },
    ]);
  });
});
