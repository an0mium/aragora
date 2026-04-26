import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { SettlementsAPI } from '../settlements';

interface MockClient {
  request: Mock;
}

describe('SettlementsAPI', () => {
  let api: SettlementsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new SettlementsAPI(mockClient as any);
  });

  it('lists pending settlements', async () => {
    mockClient.request.mockResolvedValue({ data: { count: 2 } });

    const result = await api.listPending({ limit: 10 });

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/settlements', {
      params: { limit: 10 },
    });
    expect(result.data.count).toBe(2);
  });

  it('gets agent accuracy', async () => {
    mockClient.request.mockResolvedValue({ data: { accuracy: 0.95 } });

    const result = await api.getAgentAccuracy('critic-1');

    expect(mockClient.request).toHaveBeenCalledWith(
      'GET',
      '/api/v1/settlements/agent/critic-1/accuracy'
    );
    expect(result.data.accuracy).toBe(0.95);
  });

  it('submits a settlement outcome', async () => {
    mockClient.request.mockResolvedValue({ data: { settlement_id: 's-1' } });

    const result = await api.settle('s-1', {
      outcome: 'correct',
      evidence: 'verified',
      settled_by: 'sdk-test',
    });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/settlements/s-1/settle', {
      body: {
        outcome: 'correct',
        evidence: 'verified',
        settled_by: 'sdk-test',
      },
    });
    expect(result.data.settlement_id).toBe('s-1');
  });
});
