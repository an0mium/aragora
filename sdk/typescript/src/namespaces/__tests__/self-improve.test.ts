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

  it('should call detail and improvement queue endpoints', async () => {
    mockClient.request.mockResolvedValue({ ok: true });

    await api.getMetaPlannerGoals();
    await api.getExecutionTimeline();
    await api.getLearningInsights();
    await api.getMetricsComparison();
    await api.getCycleTrends();
    await api.addImprovementQueueItem({
      goal: 'reduce flaky checks',
      priority: 80,
      source: 'operator',
    });
    await api.updateImprovementQueuePriority('item-1', { priority: 60 });
    await api.deleteImprovementQueueItem('item-1');

    expect(mockClient.request).toHaveBeenNthCalledWith(
      1,
      'GET',
      '/api/v1/self-improve/meta-planner/goals'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      2,
      'GET',
      '/api/v1/self-improve/execution/timeline'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      3,
      'GET',
      '/api/v1/self-improve/learning/insights'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      4,
      'GET',
      '/api/v1/self-improve/metrics/comparison'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      5,
      'GET',
      '/api/v1/self-improve/trends/cycles'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      6,
      'POST',
      '/api/v1/self-improve/improvement-queue',
      { json: { goal: 'reduce flaky checks', priority: 80, source: 'operator' } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      7,
      'PUT',
      '/api/v1/self-improve/improvement-queue/item-1/priority',
      { json: { priority: 60 } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      8,
      'DELETE',
      '/api/v1/self-improve/improvement-queue/item-1'
    );
  });
});
