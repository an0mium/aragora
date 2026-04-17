import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { DashboardAPI } from '../dashboard';

interface MockClient {
  request: Mock;
}

describe('DashboardAPI', () => {
  let api: DashboardAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new DashboardAPI(mockClient as any);
  });

  it('maps Ralph dashboard routes', async () => {
    mockClient.request.mockResolvedValue({ data: {} });

    await api.listRalphCampaigns();
    await api.getRalphOverview();
    await api.getRalphBlockers();

    expect(mockClient.request).toHaveBeenNthCalledWith(
      1,
      'GET',
      '/api/v1/ralph/campaigns'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      2,
      'GET',
      '/api/v1/ralph/overview'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      3,
      'GET',
      '/api/v1/ralph/blockers'
    );
  });
});
