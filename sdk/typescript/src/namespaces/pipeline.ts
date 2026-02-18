/**
 * Pipeline Namespace API (Idea-to-Execution)
 *
 * Provides endpoints for running and monitoring the 4-stage
 * idea-to-execution pipeline: Ideas → Goals → Workflows → Orchestration.
 */

import type { AragoraClient } from '../client';

/** Pipeline run configuration */
export interface PipelineRunRequest {
  input_text: string;
  stages?: string[];
  debate_rounds?: number;
  workflow_mode?: 'quick' | 'debate';
  dry_run?: boolean;
  enable_receipts?: boolean;
  use_ai?: boolean;
}

/** Pipeline run response with initial status */
export interface PipelineRunResponse {
  pipeline_id: string;
  status: string;
  stages: string[];
}

/** Per-stage status within a pipeline */
export interface PipelineStageStatus {
  stage: string;
  status: string;
  duration?: number;
  error?: string;
}

/** Pipeline status response */
export interface PipelineStatusResponse {
  pipeline_id: string;
  overall_status: string;
  stages: PipelineStageStatus[];
}

/** React Flow graph data */
export interface PipelineGraphResponse {
  nodes: Record<string, unknown>[];
  edges: Record<string, unknown>[];
  pipeline_id: string;
  stage?: string;
}

/** Decision receipt for a completed pipeline */
export interface PipelineReceiptResponse {
  pipeline_id: string;
  receipt: Record<string, unknown>;
}

/** Stage canvas response */
export interface PipelineStageResponse {
  stage: string;
  data: Record<string, unknown>;
}

/** Goal extraction request */
export interface ExtractGoalsRequest {
  ideas_canvas_id: string;
  ideas_canvas_data?: Record<string, unknown>;
  config?: Record<string, unknown>;
}

/** Canvas conversion response (React Flow format) */
export interface CanvasConversionResponse {
  nodes: Record<string, unknown>[];
  edges: Record<string, unknown>[];
}

/**
 * Pipeline namespace for idea-to-execution orchestration.
 *
 * @example
 * ```typescript
 * // Start a pipeline run
 * const { pipeline_id } = await client.pipeline.run({ input_text: 'Build a rate limiter' });
 *
 * // Check status
 * const status = await client.pipeline.status(pipeline_id);
 *
 * // Get React Flow graph
 * const graph = await client.pipeline.graph(pipeline_id);
 *
 * // Get completion receipt
 * const receipt = await client.pipeline.receipt(pipeline_id);
 * ```
 */
export class PipelineNamespace {
  constructor(private client: AragoraClient) {}

  /**
   * Start an async pipeline execution.
   *
   * Runs the 4-stage pipeline: ideation → goals → workflow → orchestration.
   */
  async run(request: PipelineRunRequest): Promise<PipelineRunResponse> {
    return this.client.request<PipelineRunResponse>(
      'POST',
      '/api/v1/canvas/pipeline/run',
      { body: request }
    );
  }

  /**
   * Run full pipeline from an ArgumentCartographer debate export.
   */
  async fromDebate(
    cartographerData: Record<string, unknown>,
    options?: { autoAdvance?: boolean; useAi?: boolean },
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/canvas/pipeline/from-debate',
      {
        body: {
          cartographer_data: cartographerData,
          auto_advance: options?.autoAdvance ?? true,
          ...(options?.useAi ? { use_ai: true } : {}),
        },
      }
    );
  }

  /**
   * Run full pipeline from raw idea strings.
   */
  async fromIdeas(
    ideas: string[],
    options?: { autoAdvance?: boolean; useAi?: boolean },
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/canvas/pipeline/from-ideas',
      {
        body: {
          ideas,
          auto_advance: options?.autoAdvance ?? true,
          ...(options?.useAi ? { use_ai: true } : {}),
        },
      }
    );
  }

  /**
   * Get pipeline per-stage status.
   */
  async status(pipelineId: string): Promise<PipelineStatusResponse> {
    return this.client.request<PipelineStatusResponse>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/status`
    );
  }

  /**
   * Get full pipeline result.
   */
  async get(pipelineId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}`
    );
  }

  /**
   * Get React Flow JSON graph for pipeline stages.
   */
  async graph(
    pipelineId: string,
    options?: { stage?: string },
  ): Promise<PipelineGraphResponse> {
    return this.client.request<PipelineGraphResponse>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/graph`,
      { params: options }
    );
  }

  /**
   * Get DecisionReceipt for a completed pipeline.
   */
  async receipt(pipelineId: string): Promise<PipelineReceiptResponse> {
    return this.client.request<PipelineReceiptResponse>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/receipt`
    );
  }

  /**
   * Advance a pipeline to the next stage.
   */
  async advance(
    pipelineId: string,
    targetStage: string,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
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

  /**
   * Get a specific stage canvas from a pipeline.
   */
  async stage(
    pipelineId: string,
    stage: string,
  ): Promise<PipelineStageResponse> {
    return this.client.request<PipelineStageResponse>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/stage/${encodeURIComponent(stage)}`
    );
  }

  /**
   * Extract goals from an ideas canvas.
   */
  async extractGoals(
    request: ExtractGoalsRequest,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/canvas/pipeline/extract-goals',
      { body: request }
    );
  }

  /**
   * Convert ArgumentCartographer debate to React Flow ideas canvas.
   */
  async convertDebate(
    cartographerData: Record<string, unknown>,
  ): Promise<CanvasConversionResponse> {
    return this.client.request<CanvasConversionResponse>(
      'POST',
      '/api/v1/canvas/convert/debate',
      { body: { cartographer_data: cartographerData } }
    );
  }

  /**
   * Convert WorkflowDefinition to React Flow actions canvas.
   */
  async convertWorkflow(
    workflowData: Record<string, unknown>,
  ): Promise<CanvasConversionResponse> {
    return this.client.request<CanvasConversionResponse>(
      'POST',
      '/api/v1/canvas/convert/workflow',
      { body: { workflow_data: workflowData } }
    );
  }
}
