import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DebateResultComparePanel } from '../DebateResultComparePanel';

const mockFetch = jest.fn();

global.fetch = mockFetch as unknown as typeof fetch;

jest.mock('next/link', () => ({
  __esModule: true,
  default: ({ href, children }: { href: string; children: React.ReactNode }) => (
    <a href={href}>{children}</a>
  ),
}));

jest.mock('@/components/BackendSelector', () => ({
  useBackend: () => ({
    config: {
      api: 'http://backend.test',
    },
  }),
}));

function jsonResponse(data: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: async () => data,
  } as Response;
}

describe('DebateResultComparePanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('loads two decision packages side by side and surfaces configuration drift', async () => {
    mockFetch.mockImplementation((input: string | URL | Request) => {
      const url = String(input);

      if (url === 'http://backend.test/api/v1/debates/debate-a/package') {
        return Promise.resolve(
          jsonResponse({
            debate_id: 'debate-a',
            question: 'Should we ship the release?',
            verdict: 'APPROVED',
            confidence: 0.74,
            consensus_reached: true,
            final_answer: 'Ship with monitoring.',
            agents: ['claude', 'gpt-4'],
            rounds: 3,
            duration_seconds: 48,
            total_cost: 0.004,
            next_steps: [{ action: 'Ship the release.', priority: 'high' }],
            agent_models: {
              claude: {
                provider: 'anthropic-api',
                provider_display: 'Anthropic',
                model: 'claude-opus-4-6',
                llm_label: 'claude-opus-4-6 via Anthropic',
              },
              'gpt-4': {
                provider: 'openai-api',
                provider_display: 'OpenAI',
                model: 'gpt-4.1',
                llm_label: 'gpt-4.1 via OpenAI',
              },
            },
          })
        );
      }

      if (url === 'http://backend.test/api/v1/debates/debate-b/package') {
        return Promise.resolve(
          jsonResponse({
            debate_id: 'debate-b',
            question: 'Should we ship the release?',
            verdict: 'APPROVED_WITH_CONDITIONS',
            confidence: 0.88,
            consensus_reached: true,
            final_answer: 'Ship after rollback automation is verified.',
            agents: ['claude', 'codex', 'gpt-4'],
            rounds: 4,
            duration_seconds: 61,
            total_cost: 0.0065,
            next_steps: [{ action: 'Verify rollback automation.', priority: 'high' }],
            agent_models: {
              claude: {
                provider: 'anthropic-api',
                provider_display: 'Anthropic',
                model: 'claude-sonnet-4-5',
                llm_label: 'claude-sonnet-4-5 via Anthropic',
              },
              codex: {
                provider: 'openai-api',
                provider_display: 'OpenAI',
                model: 'gpt-5-codex',
                llm_label: 'gpt-5-codex via OpenAI',
              },
              'gpt-4': {
                provider: 'openai-api',
                provider_display: 'OpenAI',
                model: 'gpt-4.1',
                llm_label: 'gpt-4.1 via OpenAI',
              },
            },
          })
        );
      }

      return Promise.reject(new Error(`Unexpected fetch: ${url}`));
    });

    render(
      <DebateResultComparePanel
        initialLeftDebateId="debate-a"
        initialRightDebateId="debate-b"
      />
    );

    expect(await screen.findByText('APPROVED -> APPROVED_WITH_CONDITIONS')).toBeInTheDocument();
    expect(screen.getByText('1 added / 0 removed')).toBeInTheDocument();
    expect(screen.getByText('1 shared agents reconfigured')).toBeInTheDocument();
    expect(screen.getByText('claude-sonnet-4-5 via Anthropic')).toBeInTheDocument();
    expect(screen.getByText('gpt-5-codex via OpenAI')).toBeInTheDocument();

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('http://backend.test/api/v1/debates/debate-a/package');
      expect(mockFetch).toHaveBeenCalledWith('http://backend.test/api/v1/debates/debate-b/package');
    });
  });

  it('validates distinct debate ids before comparing', async () => {
    const user = userEvent.setup();

    render(<DebateResultComparePanel />);

    await user.type(screen.getByLabelText('Left debate ID'), 'same-debate');
    await user.type(screen.getByLabelText('Right debate ID'), 'same-debate');
    await user.click(screen.getByRole('button', { name: 'COMPARE' }));

    expect(await screen.findByText('Choose two different debate IDs.')).toBeInTheDocument();
    expect(mockFetch).not.toHaveBeenCalled();
  });
});
