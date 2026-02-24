/**
 * Status Namespace Tests
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { StatusNamespace } from '../status';

interface MockClient {
  request: Mock;
}

describe('StatusNamespace', () => {
  let api: StatusNamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new StatusNamespace(mockClient as any);
  });

  it('should get uptime summary', async () => {
    mockClient.request.mockResolvedValue({
      data: {
        current: { status: 'operational', uptime_seconds: 86400 },
        periods: { '24h': { uptime_pct: 100.0 } },
      },
    });

    const result = await api.getUptime();

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/status/uptime');
    expect((result as any).data.current.status).toBe('operational');
  });
});
