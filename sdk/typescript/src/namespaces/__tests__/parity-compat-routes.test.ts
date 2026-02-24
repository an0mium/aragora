import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { AgentsAPI } from '../agents';
import { DebatesAPI } from '../debates';
import { GitHubNamespace } from '../github';

interface MockClient {
  request: Mock;
}

describe('Parity Compatibility Routes', () => {
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
  });

  it('maps agent recommendation compatibility routes', async () => {
    const api = new AgentsAPI(mockClient as any);
    mockClient.request.mockResolvedValue({});

    await api.getAgentRecommendations({ domain: 'finance', limit: 5 });
    await api.getAgentLeaderboard({ domain: 'finance', limit: 10 });

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'GET', '/api/agents/recommend', {
      params: { domain: 'finance', limit: 5 },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(2, 'GET', '/api/agents/leaderboard', {
      params: { domain: 'finance', limit: 10 },
    });
  });

  it('maps debate stats compatibility routes', async () => {
    const api = new DebatesAPI(mockClient as any);
    mockClient.request.mockResolvedValue({});

    await api.getStats('week');
    await api.getStatsAgents(25);

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'GET', '/api/debates/stats', {
      params: { period: 'week' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(2, 'GET', '/api/debates/stats/agents', {
      params: { limit: 25 },
    });
  });

  it('maps debate share and package compatibility routes', async () => {
    const api = new DebatesAPI(mockClient as any);
    mockClient.request.mockResolvedValue({});

    await api.share('debate-1');
    await api.getPublicSpectate('debate-1');
    await api.getPackage('debate-1');
    await api.getPackageMarkdown('debate/1');

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'POST', '/api/debates/debate-1/share');
    expect(mockClient.request).toHaveBeenNthCalledWith(2, 'GET', '/api/debates/debate-1/spectate/public');
    expect(mockClient.request).toHaveBeenNthCalledWith(3, 'GET', '/api/debates/debate-1/package');
    expect(mockClient.request).toHaveBeenNthCalledWith(4, 'GET', '/api/debates/debate%2F1/package/markdown');
  });

  it('maps github PR review compatibility route', async () => {
    const api = new GitHubNamespace(mockClient as any);
    mockClient.request.mockResolvedValue({});

    await api.triggerPRReview({
      repository: 'org/repo',
      pr_number: 42,
      review_type: 'comprehensive',
    });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/github/pr/review', {
      body: {
        repository: 'org/repo',
        pr_number: 42,
        review_type: 'comprehensive',
      },
    });
  });
});
