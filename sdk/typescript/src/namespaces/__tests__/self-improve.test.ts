/**
 * Self-Improve Namespace Tests
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { SelfImproveAPI } from '../self-improve';

interface MockClient {
  request: Mock;
}

describe('SelfImproveAPI', () => {
  let api: SelfImproveAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = { request: vi.fn() };
    api = new SelfImproveAPI(mockClient as any);
  });

  it('should call feedback/goals/metrics/regression endpoints', async () => {
    mockClient.request.mockResolvedValue({ ok: true });

    await api.submitFeedback({ score: 5 });
    await api.getFeedbackSummary({ period: '30d' });
    await api.upsertGoals({ goals: ['reduce regressions'] });
    await api.getMetricsSummary({ period: '30d' });
    await api.getRegressionHistory({ period: '30d' });

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'POST', '/api/v1/self-improve/feedback', {
      json: { score: 5 },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(
      2,
      'POST',
      '/api/v1/self-improve/feedback-summary',
      { json: { period: '30d' } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(3, 'POST', '/api/v1/self-improve/goals', {
      json: { goals: ['reduce regressions'] },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(
      4,
      'POST',
      '/api/v1/self-improve/metrics/summary',
      { json: { period: '30d' } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      5,
      'POST',
      '/api/v1/self-improve/regression-history',
      { json: { period: '30d' } }
    );
  });
});
