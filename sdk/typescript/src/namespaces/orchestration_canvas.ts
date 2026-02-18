/**
 * Orchestration Canvas Namespace API
 *
 * Provides endpoints for the Orchestration Canvas (Stage 4 of the
 * Idea-to-Execution Pipeline): canvas CRUD, node/edge management,
 * React Flow export, and pipeline execution.
 */

import type { AragoraClient } from '../client';

/** An orchestration node on the canvas */
export interface OrchestrationNode {
  id: string;
  label: string;
  orchestration_type: string;
  position?: { x: number; y: number };
  data?: Record<string, unknown>;
}

/** An edge connecting two orchestration nodes */
export interface OrchestrationEdge {
  id: string;
  source_id: string;
  target_id: string;
  edge_type?: string;
  label?: string;
  data?: Record<string, unknown>;
}

/** An orchestration canvas document */
export interface OrchestrationCanvas {
  id: string;
  name: string;
  description?: string;
  owner_id?: string;
  workspace_id?: string;
  source_canvas_id?: string;
  metadata?: Record<string, unknown>;
  nodes?: OrchestrationNode[];
  edges?: OrchestrationEdge[];
  created_at: string;
  updated_at: string;
}

/** Result of executing the pipeline */
export interface ExecutionResult {
  execution_id: string;
  canvas_id: string;
  stage: string;
  nodes_count: number;
  edges_count: number;
  metadata: Record<string, unknown>;
  status: string;
}

/**
 * Orchestration Canvas namespace (Stage 4 of the
 * Idea-to-Execution Pipeline).
 */
export class OrchestrationCanvasNamespace {
  constructor(private client: AragoraClient) {}

  // -------------------------------------------------------------------
  // Canvas CRUD
  // -------------------------------------------------------------------

  /** List orchestration canvases with optional filtering. */
  async listCanvases(options?: {
    workspace_id?: string;
    owner_id?: string;
    source_canvas_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ canvases: OrchestrationCanvas[] }> {
    return this.client.request<{ canvases: OrchestrationCanvas[] }>(
      'GET',
      '/api/v1/orchestration/canvas',
      { params: options },
    );
  }

  /** Create a new orchestration canvas. */
  async createCanvas(
    name: string,
    description: string = '',
    metadata?: Record<string, unknown>,
  ): Promise<OrchestrationCanvas> {
    const body: Record<string, unknown> = { name, description };
    if (metadata !== undefined) {
      body.metadata = metadata;
    }
    return this.client.request<OrchestrationCanvas>(
      'POST',
      '/api/v1/orchestration/canvas',
      { body },
    );
  }

  /** Get a canvas by ID, including its nodes and edges. */
  async getCanvas(canvasId: string): Promise<OrchestrationCanvas> {
    return this.client.request<OrchestrationCanvas>(
      'GET',
      `/api/v1/orchestration/canvas/${encodeURIComponent(canvasId)}`,
    );
  }

  /** Update canvas metadata. */
  async updateCanvas(
    canvasId: string,
    updates: {
      name?: string;
      description?: string;
      metadata?: Record<string, unknown>;
    },
  ): Promise<OrchestrationCanvas> {
    return this.client.request<OrchestrationCanvas>(
      'PUT',
      `/api/v1/orchestration/canvas/${encodeURIComponent(canvasId)}`,
      { body: updates },
    );
  }

  /** Delete a canvas and all its nodes and edges. */
  async deleteCanvas(canvasId: string): Promise<{ success: boolean }> {
    return this.client.request<{ success: boolean }>(
      'DELETE',
      `/api/v1/orchestration/canvas/${encodeURIComponent(canvasId)}`,
    );
  }

  // -------------------------------------------------------------------
  // Node CRUD
  // -------------------------------------------------------------------

  /** Add a node to a canvas. */
  async addNode(
    canvasId: string,
    label: string,
    orchestrationType: string = 'agent_task',
    position?: { x: number; y: number },
    data?: Record<string, unknown>,
  ): Promise<OrchestrationNode> {
    const body: Record<string, unknown> = {
      label,
      orchestration_type: orchestrationType,
    };
    if (position !== undefined) {
      body.position = position;
    }
    if (data !== undefined) {
      body.data = data;
    }
    return this.client.request<OrchestrationNode>(
      'POST',
      `/api/v1/orchestration/canvas/${encodeURIComponent(canvasId)}/nodes`,
      { body },
    );
  }

  /** Update a node on a canvas. */
  async updateNode(
    canvasId: string,
    nodeId: string,
    updates: {
      label?: string;
      position?: { x: number; y: number };
      data?: Record<string, unknown>;
    },
  ): Promise<OrchestrationNode> {
    return this.client.request<OrchestrationNode>(
      'PUT',
      `/api/v1/orchestration/canvas/${encodeURIComponent(canvasId)}/nodes/${encodeURIComponent(nodeId)}`,
      { body: updates },
    );
  }

  /** Delete a node from a canvas. */
  async deleteNode(
    canvasId: string,
    nodeId: string,
  ): Promise<{ success: boolean }> {
    return this.client.request<{ success: boolean }>(
      'DELETE',
      `/api/v1/orchestration/canvas/${encodeURIComponent(canvasId)}/nodes/${encodeURIComponent(nodeId)}`,
    );
  }

  // -------------------------------------------------------------------
  // Edge CRUD
  // -------------------------------------------------------------------

  /** Add an edge connecting two nodes on a canvas. */
  async addEdge(
    canvasId: string,
    sourceId: string,
    targetId: string,
    edgeType: string = 'default',
    label: string = '',
    data?: Record<string, unknown>,
  ): Promise<OrchestrationEdge> {
    const body: Record<string, unknown> = {
      source_id: sourceId,
      target_id: targetId,
      edge_type: edgeType,
      label,
    };
    if (data !== undefined) {
      body.data = data;
    }
    return this.client.request<OrchestrationEdge>(
      'POST',
      `/api/v1/orchestration/canvas/${encodeURIComponent(canvasId)}/edges`,
      { body },
    );
  }

  /** Delete an edge from a canvas. */
  async deleteEdge(
    canvasId: string,
    edgeId: string,
  ): Promise<{ success: boolean }> {
    return this.client.request<{ success: boolean }>(
      'DELETE',
      `/api/v1/orchestration/canvas/${encodeURIComponent(canvasId)}/edges/${encodeURIComponent(edgeId)}`,
    );
  }

  // -------------------------------------------------------------------
  // Export & Execute
  // -------------------------------------------------------------------

  /** Export a canvas as React Flow JSON (nodes + edges arrays). */
  async exportCanvas(canvasId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/orchestration/canvas/${encodeURIComponent(canvasId)}/export`,
    );
  }

  /** Execute the orchestration pipeline defined by this canvas. */
  async executePipeline(canvasId: string): Promise<ExecutionResult> {
    return this.client.request<ExecutionResult>(
      'POST',
      `/api/v1/orchestration/canvas/${encodeURIComponent(canvasId)}/execute`,
      { body: {} },
    );
  }
}
