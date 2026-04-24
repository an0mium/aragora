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
});
