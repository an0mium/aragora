/**
 * Canvas Namespace API
 *
 * Provides endpoints for live canvas collaboration and the
 * idea-to-execution pipeline including canvas CRUD, node/edge
 * management, pipeline orchestration, and format conversion.
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

/** Pipeline stage type */
export type PipelineStage = 'ideas' | 'goals' | 'actions' | 'orchestration';

/** Pipeline result from idea-to-execution */
export interface PipelineResult {
  pipeline_id: string;
  ideas_canvas?: Record<string, unknown>;
  goal_graph?: Record<string, unknown>;
  actions_canvas?: Record<string, unknown>;
  orchestration_canvas?: Record<string, unknown>;
  stage_status: Record<string, string>;
  transitions?: Record<string, unknown>[];
  provenance?: Record<string, unknown>[];
}

/**
 * Canvas namespace for visual collaboration boards and
 * the idea-to-execution pipeline.
 *
 * @example
 * ```typescript
 * // Canvas CRUD
 * const canvases = await client.canvas.list();
 * const canvas = await client.canvas.get(canvases[0].id);
 *
 * // Pipeline from debate
 * const result = await client.canvas.runFromDebate(cartographerData);
 * const stage = await client.canvas.getStage(result.pipeline_id, 'goals');
 * ```
 */
export class CanvasNamespace {
  constructor(private client: AragoraClient) {}

  // -----------------------------------------------------------------------
  // Canvas CRUD
  // -----------------------------------------------------------------------

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

  // -----------------------------------------------------------------------
  // Idea-to-Execution Pipeline
  // -----------------------------------------------------------------------

  /**
   * Run full pipeline from an ArgumentCartographer debate export.
   *
   * Transforms debate argument graphs into actionable execution plans
   * through 4 stages: ideas → goals → actions → orchestration.
   */
  async runFromDebate(
    cartographerData: Record<string, unknown>,
    autoAdvance: boolean = true,
  ): Promise<PipelineResult> {
    return this.client.request<PipelineResult>(
      'POST',
      '/api/v1/canvas/pipeline/from-debate',
      {
        body: {
          cartographer_data: cartographerData,
          auto_advance: autoAdvance,
        },
      }
    );
  }

  /**
   * Run full pipeline from raw idea strings.
   *
   * Simpler entry point that skips debate graph parsing.
   */
  async runFromIdeas(
    ideas: string[],
    autoAdvance: boolean = true,
  ): Promise<PipelineResult> {
    return this.client.request<PipelineResult>(
      'POST',
      '/api/v1/canvas/pipeline/from-ideas',
      {
        body: {
          ideas,
          auto_advance: autoAdvance,
        },
      }
    );
  }

  /**
   * Advance pipeline to the next stage after human review.
   *
   * Used for human-in-the-loop workflows where each stage
   * requires approval before proceeding.
   */
  async advanceStage(
    pipelineId: string,
    targetStage: PipelineStage,
  ): Promise<PipelineResult> {
    return this.client.request<PipelineResult>(
      'POST',
      '/api/v1/canvas/pipeline/advance',
      {
        body: {
          pipeline_id: pipelineId,
          target_stage: targetStage,
        },
      }
    );
  }

  /** Get complete pipeline result by ID. */
  async getPipeline(pipelineId: string): Promise<PipelineResult> {
    return this.client.request<PipelineResult>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}`
    );
  }

  /** Get a specific stage canvas from a pipeline. */
  async getStage(
    pipelineId: string,
    stage: PipelineStage,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/stage/${encodeURIComponent(stage)}`
    );
  }

  /** Convert debate argument graph to ideas canvas (React Flow format). */
  async convertDebate(
    cartographerData: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/canvas/convert/debate',
      { body: { cartographer_data: cartographerData } }
    );
  }

  /** Convert workflow definition to actions canvas (React Flow format). */
  async convertWorkflow(
    workflowData: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/canvas/convert/workflow',
      { body: { workflow_data: workflowData } }
    );
  }

  /**
   * Run pipeline from raw unstructured text.
   *
   * Parses text into ideas, then runs through the full pipeline.
   */
  async runFromBraindump(
    text: string,
    options?: { context?: string; autoAdvance?: boolean },
  ): Promise<PipelineResult> {
    return this.client.request<PipelineResult>(
      'POST',
      '/api/v1/canvas/pipeline/from-braindump',
      {
        body: {
          text,
          auto_advance: options?.autoAdvance ?? true,
          ...(options?.context ? { context: options.context } : {}),
        },
      }
    );
  }

  /** Run pipeline from a named template. */
  async runFromTemplate(
    templateName: string,
    autoAdvance: boolean = true,
  ): Promise<PipelineResult> {
    return this.client.request<PipelineResult>(
      'POST',
      '/api/v1/canvas/pipeline/from-template',
      {
        body: {
          template_name: templateName,
          auto_advance: autoAdvance,
        },
      }
    );
  }

  /** Execute a pipeline's orchestration stage. */
  async executePipeline(
    pipelineId: string,
    options?: { dryRun?: boolean },
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/execute`,
      { body: { dry_run: options?.dryRun ?? false } }
    );
  }

  /** List available pipeline templates. */
  async listTemplates(options?: { category?: string }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      '/api/v1/canvas/pipeline/templates',
      { params: options }
    );
  }

  /** Get DecisionReceipt for a completed pipeline. */
  async getReceipt(pipelineId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/receipt`
    );
  }

  /** Get React Flow graph for pipeline stages. */
  async getGraph(
    pipelineId: string,
    options?: { stage?: string },
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/graph`,
      { params: options }
    );
  }

  /** Get pipeline per-stage status. */
  async getStatus(pipelineId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/status`
    );
  }

  /** Convert an existing debate into a pipeline. */
  async debateToPipeline(
    debateId: string,
    options?: { useUniversal?: boolean; autoAdvance?: boolean },
  ): Promise<PipelineResult> {
    return this.client.request<PipelineResult>(
      'POST',
      `/api/v1/debates/${encodeURIComponent(debateId)}/to-pipeline`,
      {
        body: {
          use_universal: options?.useUniversal ?? false,
          auto_advance: options?.autoAdvance ?? true,
        },
      }
    );
  }
}
