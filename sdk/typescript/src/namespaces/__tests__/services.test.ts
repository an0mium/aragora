import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';

import { ServicesNamespace } from '../services';

interface MockClient {
  request: Mock;
}

describe('ServicesNamespace', () => {
  let api: ServicesNamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new ServicesNamespace(mockClient as any);
  });

  it('lists services', async () => {
    mockClient.request.mockResolvedValue({
      services: [{ id: 'svc-1', name: 'API', status: 'healthy' }],
    });

    const result = await api.list({ status: 'healthy' });

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/services', {
      params: { status: 'healthy' },
    });
    expect(result).toHaveLength(1);
  });

  it('gets service details', async () => {
    mockClient.request.mockResolvedValue({ id: 'svc-1', name: 'API', status: 'healthy' });

    const result = await api.get('svc-1');

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/services/svc-1');
    expect(result.id).toBe('svc-1');
  });

  it('does not expose unsupported service helper methods', () => {
    expect('getHealth' in api).toBe(false);
    expect('getMetrics' in api).toBe(false);
    expect('register' in api).toBe(false);
    expect('deregister' in api).toBe(false);
    expect('getDependencies' in api).toBe(false);
  });
});
