/**
 * DAG Operations Namespace API
 *
 * Provides operations for pipeline DAG node automation and graph-level flows.
 */

import type { AragoraClient } from '../client';

export interface DagOperationResult {
  success: boolean;
  message: string;
  created_nodes: string[];
  metadata: Record<string, unknown>;
}

export interface DagGraphResponse {
  data: Record<string, unknown>;
}

export interface DagOperationResponse {
  data: DagOperationResult;
}

export interface DebateNodeOptions {
  agents?: string[];
  rounds?: number;
}

export interface AssignAgentsOptions {
  node_ids?: string[];
}

export interface FindPrecedentsOptions {
  max_results?: number;
}

export interface ClusterIdeasOptions {
  threshold?: number;
}

export interface AutoFlowOptions {
  config?: Record<string, unknown>;
}

export class DagOperationsNamespace {
  constructor(private client: AragoraClient) {}

  async getGraph(graphId: string): Promise<DagGraphResponse> {
    return this.client.request<DagGraphResponse>(
      'GET',
      `/api/v1/pipeline/dag/${encodeURIComponent(graphId)}`
    );
  }

  async debateNode(
    graphId: string,
    nodeId: string,
    options?: DebateNodeOptions,
  ): Promise<DagOperationResponse> {
    return this.client.request<DagOperationResponse>(
      'POST',
      `/api/v1/pipeline/dag/${encodeURIComponent(graphId)}/nodes/${encodeURIComponent(nodeId)}/debate`,
      {
        body: {
          ...(options?.agents ? { agents: options.agents } : {}),
          rounds: options?.rounds ?? 3,
        },
      },
    );
  }

  async decomposeNode(graphId: string, nodeId: string): Promise<DagOperationResponse> {
    return this.client.request<DagOperationResponse>(
      'POST',
      `/api/v1/pipeline/dag/${encodeURIComponent(graphId)}/nodes/${encodeURIComponent(nodeId)}/decompose`,
    );
  }

  async prioritizeNode(graphId: string, nodeId: string): Promise<DagOperationResponse> {
    return this.client.request<DagOperationResponse>(
      'POST',
      `/api/v1/pipeline/dag/${encodeURIComponent(graphId)}/nodes/${encodeURIComponent(nodeId)}/prioritize`,
    );
  }

  async assignAgents(
    graphId: string,
    nodeId: string,
    options?: AssignAgentsOptions,
  ): Promise<DagOperationResponse> {
    return this.client.request<DagOperationResponse>(
      'POST',
      `/api/v1/pipeline/dag/${encodeURIComponent(graphId)}/nodes/${encodeURIComponent(nodeId)}/assign-agents`,
      {
        body: options?.node_ids ? { node_ids: options.node_ids } : {},
      },
    );
  }

  async executeNode(graphId: string, nodeId: string): Promise<DagOperationResponse> {
    return this.client.request<DagOperationResponse>(
      'POST',
      `/api/v1/pipeline/dag/${encodeURIComponent(graphId)}/nodes/${encodeURIComponent(nodeId)}/execute`,
    );
  }

  async findPrecedents(
    graphId: string,
    nodeId: string,
    options?: FindPrecedentsOptions,
  ): Promise<DagOperationResponse> {
    return this.client.request<DagOperationResponse>(
      'POST',
      `/api/v1/pipeline/dag/${encodeURIComponent(graphId)}/nodes/${encodeURIComponent(nodeId)}/find-precedents`,
      {
        body: { max_results: options?.max_results ?? 5 },
      },
    );
  }

  async clusterIdeas(
    graphId: string,
    ideas: string[],
    options?: ClusterIdeasOptions,
  ): Promise<DagOperationResponse> {
    return this.client.request<DagOperationResponse>(
      'POST',
      `/api/v1/pipeline/dag/${encodeURIComponent(graphId)}/cluster-ideas`,
      {
        body: {
          ideas,
          threshold: options?.threshold ?? 0.3,
        },
      },
    );
  }

  async autoFlow(
    graphId: string,
    ideas: string[],
    options?: AutoFlowOptions,
  ): Promise<DagOperationResponse> {
    return this.client.request<DagOperationResponse>(
      'POST',
      `/api/v1/pipeline/dag/${encodeURIComponent(graphId)}/auto-flow`,
      {
        body: {
          ideas,
          ...(options?.config ? { config: options.config } : {}),
        },
      },
    );
  }
}
