import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { TasksAPI } from '../tasks';

interface MockClient {
  request: Mock;
}

describe('TasksAPI', () => {
  let api: TasksAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new TasksAPI(mockClient as any);
  });

  it('maps task queue and lease routes', async () => {
    mockClient.request.mockResolvedValue({ data: {} });

    await api.listQueue({ status: 'pending', work_type: 'code', limit: 10 });
    await api.getQueueTask('task/demo');
    await api.getQueueStats();
    await api.syncQueue({ include_pending: false });
    await api.claimQueueTask('task/demo', { owner_agent: 'codex' });
    await api.listLeases();
    await api.heartbeatLease('lease/1', { ttl_hours: 2.5 });
    await api.releaseLease('lease/1');
    await api.completeLease('lease/1', { outcome: 'completed' });
    await api.listSalvage();

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'GET', '/api/v1/tasks/queue', {
      params: { status: 'pending', work_type: 'code', limit: 10 },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(2, 'GET', '/api/v1/tasks/queue/task%2Fdemo');
    expect(mockClient.request).toHaveBeenNthCalledWith(3, 'GET', '/api/v1/tasks/queue/stats');
    expect(mockClient.request).toHaveBeenNthCalledWith(4, 'POST', '/api/v1/tasks/queue/sync', {
      body: { include_pending: false },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(
      5,
      'POST',
      '/api/v1/tasks/queue/task%2Fdemo/claim',
      { body: { owner_agent: 'codex' } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(6, 'GET', '/api/v1/tasks/leases');
    expect(mockClient.request).toHaveBeenNthCalledWith(
      7,
      'POST',
      '/api/v1/tasks/leases/lease%2F1/heartbeat',
      { body: { ttl_hours: 2.5 } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      8,
      'POST',
      '/api/v1/tasks/leases/lease%2F1/release'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      9,
      'POST',
      '/api/v1/tasks/leases/lease%2F1/complete',
      { body: { outcome: 'completed' } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(10, 'GET', '/api/v1/tasks/salvage');
  });
});
