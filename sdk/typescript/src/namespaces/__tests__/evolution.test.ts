import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { EvolutionAPI } from '../evolution';

interface MockClient {
  request: Mock;
}

describe('EvolutionAPI', () => {
  let api: EvolutionAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new EvolutionAPI(mockClient as any);
  });

  it('maps agent evolution dashboard routes', async () => {
    mockClient.request.mockResolvedValue({ data: {} });

    await api.getAgentEvolutionTimeline({ limit: 10, offset: 5 });
    await api.getAgentEvolutionEloTrends({ period: '30d' });
    await api.getAgentEvolutionPending();
    await api.approveAgentEvolutionChange('change/1');
    await api.rejectAgentEvolutionChange('change/2');

    expect(mockClient.request).toHaveBeenNthCalledWith(
      1,
      'GET',
      '/api/v1/agent-evolution/timeline',
      { params: { limit: 10, offset: 5 } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      2,
      'GET',
      '/api/v1/agent-evolution/elo-trends',
      { params: { period: '30d' } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      3,
      'GET',
      '/api/v1/agent-evolution/pending'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      4,
      'POST',
      '/api/v1/agent-evolution/pending/change%2F1/approve'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      5,
      'POST',
      '/api/v1/agent-evolution/pending/change%2F2/reject'
    );
  });
});
