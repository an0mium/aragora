import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { OpenApiAPI } from '../openapi';

interface MockClient {
  request: Mock;
}

describe('OpenApiAPI docs helpers', () => {
  let api: OpenApiAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn().mockResolvedValue({}),
    };
    api = new OpenApiAPI(mockClient as any);
  });

  it('maps docs introspection routes to versioned endpoints', async () => {
    await api.requestGetApiV1DocsRoutes({ tag: 'chat' });
    await api.requestGetApiV1DocsStats();

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'GET', '/api/v1/docs/routes', {
      params: { tag: 'chat' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(2, 'GET', '/api/v1/docs/stats', {
      params: undefined,
    });
  });
});
