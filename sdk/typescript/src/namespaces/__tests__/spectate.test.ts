import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { SpectateAPI } from '../spectate';

interface MockClient {
  request: Mock;
}

describe('SpectateAPI', () => {
  let api: SpectateAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn().mockResolvedValue({}),
    };
    api = new SpectateAPI(mockClient as any);
  });

  it('uses the canonical debate-filtered stream query route', async () => {
    await api.connectSSE('debate-1');

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/spectate/stream', {
      params: { debate_id: 'debate-1' },
    });
  });

  it('passes count and debate filters to the stream snapshot endpoint', async () => {
    await api.getStream({ count: 25, debateId: 'debate-2' });

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/spectate/stream', {
      params: { count: 25, debate_id: 'debate-2' },
    });
  });
});
