import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { BudgetsAPI } from '../budgets';
import { CostsNamespace } from '../costs';
import { DocumentsAPI } from '../documents';
import { GatewayAPI } from '../gateway';
import { KnowledgeAPI } from '../knowledge';
import { PipelineNamespace } from '../pipeline';

interface MockClient {
  request: Mock;
  get: Mock;
  post: Mock;
}

describe('Cross SDK Python Compatibility Routes', () => {
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn().mockResolvedValue({}),
      get: vi.fn().mockResolvedValue({}),
      post: vi.fn().mockResolvedValue({}),
    };
  });

  it('maps pipeline compatibility routes', async () => {
    const api = new PipelineNamespace(mockClient as any);

    await api.demo({ ideas: ['a'] });
    await api.autoRun('raw idea');
    await api.extractPrinciples({ nodes: [], edges: [] });
    await api.fromSystemMetrics();
    await api.getIntelligence('p/1');
    await api.getBeliefs('p/1');
    await api.getExplanations('p/1');
    await api.getPrecedents('p/1');
    await api.selfImprove('p/1');
    await api.getGraphById('g/1');
    await api.getGraphIntegrity('g/1');
    await api.addGraphNode('g/1', { id: 'n1' });
    await api.updateGraphNode('g/1', 'n/1', { label: 'x' });
    await api.reassignGraphNode('g/1', 'n/1', { owner: 'agent-1' });
    await api.listGraphNodes('g/1');
    await api.promoteGraphNode('g/1', { node_id: 'n1' });
    await api.getGraphNodeProvenance('g/1', 'n/1');
    await api.getGraphReactFlow('g/1');
    await api.getGraphSuggestions('g/1');
    await api.ideasToGoals({ pipeline_id: 'p1' });
    await api.goalsToTasks({ pipeline_id: 'p1' });
    await api.tasksToWorkflow({ pipeline_id: 'p1' });
    await api.executeTransitions({ pipeline_id: 'p1' });
    await api.getTransitionProvenance('n/1');

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'POST', '/api/v1/canvas/pipeline/demo', {
      body: { ideas: ['a'] },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(10, 'GET', '/api/v1/pipeline/graph/g%2F1');
    expect(mockClient.request).toHaveBeenNthCalledWith(24, 'GET', '/api/v1/pipeline/transitions/n%2F1/provenance');
  });

  it('maps costs analytics compatibility routes', async () => {
    const api = new CostsNamespace(mockClient as any);

    await api.getAnalyticsTrend({ range: '7d' });
    await api.getAnalyticsByAgent({ workspace_id: 'w1' });
    await api.getAnalyticsByModel({ workspace_id: 'w1' });
    await api.getAnalyticsByDebate({ workspace_id: 'w1' });
    await api.getAnalyticsBudgetUtilization({ workspace_id: 'w1' });

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'GET', '/api/v1/costs/analytics/trend', {
      params: { range: '7d' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(5, 'GET', '/api/v1/costs/analytics/budget-utilization', {
      params: { workspace_id: 'w1' },
    });
  });

  it('maps gateway openclaw compatibility routes', async () => {
    const api = new GatewayAPI(mockClient as any);

    await api.listOpenClawSessions({ limit: 10 });
    await api.executeOpenClawAction({ session_id: 's1', action_type: 'click' });
    await api.listOpenClawCredentials();
    await api.getOpenClawHealth();
    await api.getOpenClawMetrics();
    await api.getOpenClawAudit({ session_id: 's1' });

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'GET', '/api/gateway/openclaw/sessions', {
      params: { limit: 10 },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(6, 'GET', '/api/gateway/openclaw/audit', {
      params: { session_id: 's1' },
    });
  });

  it('maps knowledge fact batch/validation compatibility routes', async () => {
    const api = new KnowledgeAPI(mockClient as any);

    await api.batchCreateFacts([{ statement: 'a' }]);
    await api.batchDeleteFacts(['f1', 'f2']);
    await api.mergeFacts({ target_fact_id: 'f1', source_fact_ids: ['f2'] });
    await api.listFactRelationships({ relation_type: 'supports' });
    await api.getFactStats();
    await api.validateFacts({ fact_ids: ['f1'] });

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'POST', '/api/v1/knowledge/facts/batch', {
      json: { facts: [{ statement: 'a' }] },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(6, 'POST', '/api/v1/knowledge/facts/validate', {
      json: { fact_ids: ['f1'] },
    });
  });

  it('maps budgets overrides/transactions/trends compatibility routes', async () => {
    const api = new BudgetsAPI(mockClient as any);

    await api.addOverride('b/1', { user_id: 'u/1', limit: 100 });
    await api.removeOverride('b/1', 'u/1');
    await api.getTransactions('b/1', { limit: 5 });
    await api.getTrends('b/1', { period: 'day', limit: 7 });

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'POST', '/api/v1/budgets/b%2F1/overrides', {
      body: { user_id: 'u/1', limit: 100 },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(4, 'GET', '/api/v1/budgets/b%2F1/trends', {
      params: { period: 'day', limit: 7 },
    });
  });

  it('maps document detail/chunk/reprocess compatibility routes', async () => {
    const api = new DocumentsAPI(mockClient as any);

    await api.get('doc/1');
    await api.download('doc/1');
    await api.getChunks('doc/1', { limit: 25, offset: 5 });
    await api.reprocess('doc/1');

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'GET', '/api/v1/documents/doc%2F1');
    expect(mockClient.request).toHaveBeenNthCalledWith(3, 'GET', '/api/v1/documents/doc%2F1/chunks', {
      params: { limit: 25, offset: 5 },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(4, 'POST', '/api/v1/documents/doc%2F1/reprocess');
  });
});
