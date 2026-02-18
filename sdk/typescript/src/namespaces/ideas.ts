/**
 * Ideas Namespace API
 *
 * Provides endpoints for the Idea Canvas (Stage 1 of the
 * Idea-to-Execution Pipeline): canvas CRUD, node/edge management,
 * React Flow export, and promotion to goals.
 */

import type { AragoraClient } from '../client';

/** An idea node on the canvas */
export interface IdeaNode {
  id: string;
  label: string;
  idea_type: string;
  position?: { x: number; y: number };
  data?: Record<string, unknown>;
}

/** An edge connecting two idea nodes */
export interface IdeaEdge {
  id: string;
  source_id: string;
  target_id: string;
  edge_type?: string;
  label?: string;
  data?: Record<string, unknown>;
}

/** An idea canvas document */
export interface IdeaCanvas {
  id: string;
  name: string;
  description?: string;
  owner_id?: string;
  workspace_id?: string;
  metadata?: Record<string, unknown>;
  nodes?: IdeaNode[];
  edges?: IdeaEdge[];
  created_at: string;
  updated_at: string;
}

/** Result of promoting nodes to goals */
export interface PromotionResult {
  goals_canvas: Record<string, unknown>;
  provenance: Record<string, unknown>[];
  promoted_count: number;
}

/**
 * Ideas namespace for the Idea Canvas (Stage 1 of the
 * Idea-to-Execution Pipeline).
 *
 * @example
 * ```typescript
 * // Canvas CRUD
 * const canvases = await client.ideas.listCanvases();
 * const canvas = await client.ideas.createCanvas('Brainstorm');
 *
 * // Add nodes and edges
 * const node = await client.ideas.addNode(canvas.id, 'Rate Limiter');
 * const edge = await client.ideas.addEdge(canvas.id, nodeA.id, nodeB.id);
 *
 * // Export & promote
 * const exported = await client.ideas.exportCanvas(canvas.id);
 * const result = await client.ideas.promoteNodes(canvas.id, [node.id]);
 * ```
 */
export class IdeasNamespace {
  constructor(private client: AragoraClient) {}

  // -------------------------------------------------------------------
  // Canvas CRUD
  // -------------------------------------------------------------------

  /** List idea canvases with optional filtering. */
  async listCanvases(options?: {
    workspace_id?: string;
    owner_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ canvases: IdeaCanvas[] }> {
    return this.client.request<{ canvases: IdeaCanvas[] }>(
      'GET',
      '/api/v1/ideas',
      { params: options },
    );
  }

  /** Create a new idea canvas. */
  async createCanvas(
    name: string,
    description: string = '',
    metadata?: Record<string, unknown>,
  ): Promise<IdeaCanvas> {
    const body: Record<string, unknown> = { name, description };
    if (metadata !== undefined) {
      body.metadata = metadata;
    }
    return this.client.request<IdeaCanvas>('POST', '/api/v1/ideas', { body });
  }

  /** Get a canvas by ID, including its nodes and edges. */
  async getCanvas(canvasId: string): Promise<IdeaCanvas> {
    return this.client.request<IdeaCanvas>(
      'GET',
      `/api/v1/ideas/${encodeURIComponent(canvasId)}`,
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
  ): Promise<IdeaCanvas> {
    return this.client.request<IdeaCanvas>(
      'PUT',
      `/api/v1/ideas/${encodeURIComponent(canvasId)}`,
      { body: updates },
    );
  }

  /** Delete a canvas and all its nodes and edges. */
  async deleteCanvas(canvasId: string): Promise<{ success: boolean }> {
    return this.client.request<{ success: boolean }>(
      'DELETE',
      `/api/v1/ideas/${encodeURIComponent(canvasId)}`,
    );
  }

  // -------------------------------------------------------------------
  // Node CRUD
  // -------------------------------------------------------------------

  /** Add a node to a canvas. */
  async addNode(
    canvasId: string,
    label: string,
    ideaType: string = 'concept',
    position?: { x: number; y: number },
    data?: Record<string, unknown>,
  ): Promise<IdeaNode> {
    const body: Record<string, unknown> = { label, idea_type: ideaType };
    if (position !== undefined) {
      body.position = position;
    }
    if (data !== undefined) {
      body.data = data;
    }
    return this.client.request<IdeaNode>(
      'POST',
      `/api/v1/ideas/${encodeURIComponent(canvasId)}/nodes`,
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
  ): Promise<IdeaNode> {
    return this.client.request<IdeaNode>(
      'PUT',
      `/api/v1/ideas/${encodeURIComponent(canvasId)}/nodes/${encodeURIComponent(nodeId)}`,
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
      `/api/v1/ideas/${encodeURIComponent(canvasId)}/nodes/${encodeURIComponent(nodeId)}`,
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
  ): Promise<IdeaEdge> {
    const body: Record<string, unknown> = {
      source_id: sourceId,
      target_id: targetId,
      edge_type: edgeType,
      label,
    };
    if (data !== undefined) {
      body.data = data;
    }
    return this.client.request<IdeaEdge>(
      'POST',
      `/api/v1/ideas/${encodeURIComponent(canvasId)}/edges`,
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
      `/api/v1/ideas/${encodeURIComponent(canvasId)}/edges/${encodeURIComponent(edgeId)}`,
    );
  }

  // -------------------------------------------------------------------
  // Export & Promote
  // -------------------------------------------------------------------

  /** Export a canvas as React Flow JSON (nodes + edges arrays). */
  async exportCanvas(canvasId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/ideas/${encodeURIComponent(canvasId)}/export`,
    );
  }

  /** Promote selected nodes from the idea canvas to the goals stage. */
  async promoteNodes(
    canvasId: string,
    nodeIds: string[],
  ): Promise<PromotionResult> {
    return this.client.request<PromotionResult>(
      'POST',
      `/api/v1/ideas/${encodeURIComponent(canvasId)}/promote`,
      { body: { node_ids: nodeIds } },
    );
  }
}
