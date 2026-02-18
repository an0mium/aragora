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
    autoAdvance: boolean = true,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
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
   */
  async fromIdeas(
    ideas: string[],
    autoAdvance: boolean = true,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
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
}
