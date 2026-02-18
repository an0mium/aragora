/**
 * Actions Namespace API
 *
 * Provides endpoints for the Action Canvas (Stage 3 of the
 * Idea-to-Execution Pipeline): canvas CRUD, node/edge management,
 * React Flow export, and advancement to orchestration.
 */

import type { AragoraClient } from '../client';

/** An action node on the canvas */
export interface ActionNode {
  id: string;
  label: string;
  action_type: string;
  position?: { x: number; y: number };
  data?: Record<string, unknown>;
}

/** An edge connecting two action nodes */
export interface ActionEdge {
  id: string;
  source_id: string;
  target_id: string;
  edge_type?: string;
  label?: string;
  data?: Record<string, unknown>;
}

/** An action canvas document */
export interface ActionCanvas {
  id: string;
  name: string;
  description?: string;
  owner_id?: string;
  workspace_id?: string;
  source_canvas_id?: string;
  metadata?: Record<string, unknown>;
  nodes?: ActionNode[];
  edges?: ActionEdge[];
  created_at: string;
  updated_at: string;
}

/** Result of advancing to orchestration stage */
export interface AdvanceResult {
  source_canvas_id: string;
  source_stage: string;
  target_stage: string;
  nodes: Record<string, unknown>[];
  edges: Record<string, unknown>[];
  metadata: Record<string, unknown>;
  status: string;
}

/**
 * Actions namespace for the Action Canvas (Stage 3 of the
 * Idea-to-Execution Pipeline).
 */
export class ActionsNamespace {
  constructor(private client: AragoraClient) {}

  // -------------------------------------------------------------------
  // Canvas CRUD
  // -------------------------------------------------------------------

  /** List action canvases with optional filtering. */
  async listCanvases(options?: {
    workspace_id?: string;
    owner_id?: string;
    source_canvas_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ canvases: ActionCanvas[] }> {
    return this.client.request<{ canvases: ActionCanvas[] }>(
      'GET',
      '/api/v1/actions',
      { params: options },
    );
  }

  /** Create a new action canvas. */
  async createCanvas(
    name: string,
    description: string = '',
    metadata?: Record<string, unknown>,
  ): Promise<ActionCanvas> {
    const body: Record<string, unknown> = { name, description };
    if (metadata !== undefined) {
      body.metadata = metadata;
    }
    return this.client.request<ActionCanvas>('POST', '/api/v1/actions', { body });
  }

  /** Get a canvas by ID, including its nodes and edges. */
  async getCanvas(canvasId: string): Promise<ActionCanvas> {
    return this.client.request<ActionCanvas>(
      'GET',
      `/api/v1/actions/${encodeURIComponent(canvasId)}`,
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
  ): Promise<ActionCanvas> {
    return this.client.request<ActionCanvas>(
      'PUT',
      `/api/v1/actions/${encodeURIComponent(canvasId)}`,
      { body: updates },
    );
  }

  /** Delete a canvas and all its nodes and edges. */
  async deleteCanvas(canvasId: string): Promise<{ success: boolean }> {
    return this.client.request<{ success: boolean }>(
      'DELETE',
      `/api/v1/actions/${encodeURIComponent(canvasId)}`,
    );
  }

  // -------------------------------------------------------------------
  // Node CRUD
  // -------------------------------------------------------------------

  /** Add a node to a canvas. */
  async addNode(
    canvasId: string,
    label: string,
    actionType: string = 'task',
    position?: { x: number; y: number },
    data?: Record<string, unknown>,
  ): Promise<ActionNode> {
    const body: Record<string, unknown> = { label, action_type: actionType };
    if (position !== undefined) {
      body.position = position;
    }
    if (data !== undefined) {
      body.data = data;
    }
    return this.client.request<ActionNode>(
      'POST',
      `/api/v1/actions/${encodeURIComponent(canvasId)}/nodes`,
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
  ): Promise<ActionNode> {
    return this.client.request<ActionNode>(
      'PUT',
      `/api/v1/actions/${encodeURIComponent(canvasId)}/nodes/${encodeURIComponent(nodeId)}`,
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
      `/api/v1/actions/${encodeURIComponent(canvasId)}/nodes/${encodeURIComponent(nodeId)}`,
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
  ): Promise<ActionEdge> {
    const body: Record<string, unknown> = {
      source_id: sourceId,
      target_id: targetId,
      edge_type: edgeType,
      label,
    };
    if (data !== undefined) {
      body.data = data;
    }
    return this.client.request<ActionEdge>(
      'POST',
      `/api/v1/actions/${encodeURIComponent(canvasId)}/edges`,
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
      `/api/v1/actions/${encodeURIComponent(canvasId)}/edges/${encodeURIComponent(edgeId)}`,
    );
  }

  // -------------------------------------------------------------------
  // Export & Advance
  // -------------------------------------------------------------------

  /** Export a canvas as React Flow JSON (nodes + edges arrays). */
  async exportCanvas(canvasId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/actions/${encodeURIComponent(canvasId)}/export`,
    );
  }

  /** Advance action canvas to the orchestration stage. */
  async advanceToOrchestration(
    canvasId: string,
    nodeIds?: string[],
  ): Promise<AdvanceResult> {
    const body: Record<string, unknown> = {};
    if (nodeIds !== undefined) {
      body.node_ids = nodeIds;
    }
    return this.client.request<AdvanceResult>(
      'POST',
      `/api/v1/actions/${encodeURIComponent(canvasId)}/advance`,
      { body },
    );
  }
}
