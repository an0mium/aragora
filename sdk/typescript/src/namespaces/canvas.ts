/**
 * Canvas Namespace API
 *
 * Provides endpoints for live canvas collaboration including
 * canvas CRUD, node/edge management, and action execution.
 */

import type { AragoraClient } from '../client';

/** Canvas node on the visual board */
export interface CanvasNode {
  id: string;
  type: string;
  label: string;
  position: { x: number; y: number };
  data?: Record<string, unknown>;
  created_at: string;
}

/** Edge connecting two canvas nodes */
export interface CanvasEdge {
  id: string;
  source_id: string;
  target_id: string;
  label?: string;
  created_at: string;
}

/** Canvas document */
export interface Canvas {
  id: string;
  name: string;
  description?: string;
  nodes: CanvasNode[];
  edges: CanvasEdge[];
  created_at: string;
  updated_at: string;
}

/** Options for creating a canvas */
export interface CreateCanvasRequest {
  name: string;
  description?: string;
}

/** Action to execute on a canvas */
export interface CanvasActionRequest {
  action: string;
  params?: Record<string, unknown>;
}

/**
 * Canvas namespace for visual collaboration boards.
 *
 * @example
 * ```typescript
 * const canvases = await client.canvas.list();
 * const canvas = await client.canvas.get(canvases[0].id);
 * ```
 */
export class CanvasNamespace {
  constructor(private client: AragoraClient) {}

  /** List all canvases. */
  async list(options?: { limit?: number; offset?: number }): Promise<Canvas[]> {
    const response = await this.client.request<{ canvases: Canvas[] }>(
      'GET',
      '/api/v1/canvas',
      { params: options }
    );
    return response.canvases;
  }

  /** Get a canvas by ID. */
  async get(canvasId: string): Promise<Canvas> {
    return this.client.request<Canvas>(
      'GET',
      `/api/v1/canvas/${encodeURIComponent(canvasId)}`
    );
  }

  /** Create a new canvas. */
  async create(request: CreateCanvasRequest): Promise<Canvas> {
    return this.client.request<Canvas>('POST', '/api/v1/canvas', { body: request });
  }

  /** Update a canvas. */
  async update(canvasId: string, updates: Partial<CreateCanvasRequest>): Promise<Canvas> {
    return this.client.request<Canvas>(
      'PUT',
      `/api/v1/canvas/${encodeURIComponent(canvasId)}`,
      { body: updates }
    );
  }

  /** Delete a canvas. */
  async delete(canvasId: string): Promise<{ success: boolean }> {
    return this.client.request<{ success: boolean }>(
      'DELETE',
      `/api/v1/canvas/${encodeURIComponent(canvasId)}`
    );
  }
}
