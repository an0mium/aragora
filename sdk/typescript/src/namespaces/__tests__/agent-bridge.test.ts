import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { AgentBridgeAPI } from '../agent-bridge';

interface MockClient {
  request: Mock;
}

describe('AgentBridgeAPI', () => {
  let api: AgentBridgeAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new AgentBridgeAPI(mockClient as any);
  });

  it('maps listRuns to the cursor-paginated agent-bridge route', async () => {
    mockClient.request.mockResolvedValue({
      schema_version: 1,
      runs: [],
      next_cursor: 'run:next',
    });

    await api.listRuns({ limit: 10, cursor: 'run:first' });

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/agent-bridge/runs', {
      params: { limit: 10, cursor: 'run:first' },
    });
  });

  it('maps startRun to the write-gated bridge route', async () => {
    mockClient.request.mockResolvedValue({ schema_version: 1, run_id: 'bridge-1' });

    await api.startRun({
      task: 'Coordinate review',
      actors: [{ role: 'implementer', harness: 'codex' }],
      run_id: 'bridge-1',
      next_actor: 'implementer',
    });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/agent-bridge/runs', {
      body: {
        task: 'Coordinate review',
        actors: [{ role: 'implementer', harness: 'codex' }],
        run_id: 'bridge-1',
        next_actor: 'implementer',
      },
    });
  });

  it('maps dispatchTurn and autoStep to run-scoped write routes', async () => {
    mockClient.request.mockResolvedValue({ schema_version: 1, event_id: 'evt-1' });

    await api.dispatchTurn('bridge-1', { role: 'reviewer', prompt: 'Review this' });
    await api.autoStep('bridge-1', { context_turns: 3 });

    expect(mockClient.request).toHaveBeenNthCalledWith(
      1,
      'POST',
      '/api/v1/agent-bridge/runs/bridge-1/dispatch',
      { body: { role: 'reviewer', prompt: 'Review this' } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      2,
      'POST',
      '/api/v1/agent-bridge/runs/bridge-1/auto-step',
      { body: { context_turns: 3 } }
    );
  });
});
