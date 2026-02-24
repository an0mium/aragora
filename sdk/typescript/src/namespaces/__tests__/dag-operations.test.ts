import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { DagOperationsNamespace } from '../dag-operations';

interface MockClient {
  request: Mock;
}

describe('DagOperationsNamespace', () => {
  let api: DagOperationsNamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = { request: vi.fn() };
    api = new DagOperationsNamespace(mockClient as any);
  });

  it('gets DAG graph by id', async () => {
    mockClient.request.mockResolvedValue({ data: {} });

    await api.getGraph('graph-1');

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/pipeline/dag/graph-1');
  });

  it('posts debate node operation with defaults', async () => {
    mockClient.request.mockResolvedValue({ data: { success: true } });

    await api.debateNode('graph-1', 'node-1');

    expect(mockClient.request).toHaveBeenCalledWith(
      'POST',
      '/api/v1/pipeline/dag/graph-1/nodes/node-1/debate',
      { body: { rounds: 3 } },
    );
  });

  it('posts assign-agents with explicit node_ids', async () => {
    mockClient.request.mockResolvedValue({ data: { success: true } });

    await api.assignAgents('graph-1', 'node-1', { node_ids: ['node-1', 'node-2'] });

    expect(mockClient.request).toHaveBeenCalledWith(
      'POST',
      '/api/v1/pipeline/dag/graph-1/nodes/node-1/assign-agents',
      { body: { node_ids: ['node-1', 'node-2'] } },
    );
  });

  it('posts cluster-ideas with threshold', async () => {
    mockClient.request.mockResolvedValue({ data: { success: true } });

    await api.clusterIdeas('graph-1', ['a', 'b'], { threshold: 0.4 });

    expect(mockClient.request).toHaveBeenCalledWith(
      'POST',
      '/api/v1/pipeline/dag/graph-1/cluster-ideas',
      { body: { ideas: ['a', 'b'], threshold: 0.4 } },
    );
  });

  it('encodes graph and node ids', async () => {
    mockClient.request.mockResolvedValue({ data: { success: true } });

    await api.executeNode('graph/a', 'node:b');

    expect(mockClient.request).toHaveBeenCalledWith(
      'POST',
      '/api/v1/pipeline/dag/graph%2Fa/nodes/node%3Ab/execute',
    );
  });
});
