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
    await api.getRalphCampaign('camp/alpha');
    await api.getRalphCampaignTimeline('camp/alpha');
    await api.getRalphCampaignBlockers('camp/alpha');
    await api.getRalphCampaignRepairs('camp/alpha');
    await api.getRalphCampaignBudget('camp/alpha');
    await api.getRalphCampaignPrGate('camp/alpha');

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
    expect(mockClient.request).toHaveBeenNthCalledWith(
      4,
      'GET',
      '/api/v1/ralph/campaigns/camp%2Falpha'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      5,
      'GET',
      '/api/v1/ralph/campaigns/camp%2Falpha/timeline'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      6,
      'GET',
      '/api/v1/ralph/campaigns/camp%2Falpha/blockers'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      7,
      'GET',
      '/api/v1/ralph/campaigns/camp%2Falpha/repairs'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      8,
      'GET',
      '/api/v1/ralph/campaigns/camp%2Falpha/budget'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      9,
      'GET',
      '/api/v1/ralph/campaigns/camp%2Falpha/pr-gate'
    );
  });
});
