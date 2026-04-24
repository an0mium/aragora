import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { MCPAPI } from '../mcp';
import { SettlementAPI } from '../settlements';

interface MockClient {
  request: Mock;
}

describe('MCPAPI', () => {
  let api: MCPAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new MCPAPI(mockClient as any);
  });

  it('maps MCP tool discovery routes', async () => {
    mockClient.request.mockResolvedValue({ ok: true });

    await api.listTools('debate');
    await api.getTool('run/debate');

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'GET', '/api/v1/mcp/tools', {
      params: { category: 'debate' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(
      2,
      'GET',
      '/api/v1/mcp/tools/run%2Fdebate'
    );
  });
});

describe('SettlementAPI', () => {
  let api: SettlementAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new SettlementAPI(mockClient as any);
  });

  it('maps settlement routes', async () => {
    mockClient.request.mockResolvedValue({ ok: true });

    await api.list({ debate_id: 'deb_123', domain: 'ops', limit: 5 });
    await api.getHistory({ limit: 7 });
    await api.getSummary();
    await api.get('set/123');
    await api.settle('set/123', {
      outcome: 'correct',
      evidence: 'verified',
      settled_by: 'codex',
    });
    await api.settleBatch(
      [{ settlement_id: 'set_123', outcome: 'incorrect', evidence: 'counterexample' }],
      'reviewer'
    );
    await api.getAgentAccuracy('agent/demo');

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'GET', '/api/v1/settlements', {
      params: { debate_id: 'deb_123', domain: 'ops', limit: 5 },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(
      2,
      'GET',
      '/api/v1/settlements/history',
      { params: { limit: 7 } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(3, 'GET', '/api/v1/settlements/summary');
    expect(mockClient.request).toHaveBeenNthCalledWith(
      4,
      'GET',
      '/api/v1/settlements/set%2F123'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      5,
      'POST',
      '/api/v1/settlements/set%2F123/settle',
      { body: { outcome: 'correct', evidence: 'verified', settled_by: 'codex' } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      6,
      'POST',
      '/api/v1/settlements/batch',
      {
        body: {
          settlements: [
            {
              settlement_id: 'set_123',
              outcome: 'incorrect',
              evidence: 'counterexample',
            },
          ],
          settled_by: 'reviewer',
        },
      }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      7,
      'GET',
      '/api/v1/settlements/agent/agent%2Fdemo/accuracy'
    );
  });
});
