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
   * Approve or reject a pending stage transition.
   */
  async approveTransition(
    pipelineId: string,
    fromStage: string,
    toStage: string,
    options?: { approved?: boolean; comment?: string },
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/approve-transition`,
      {
        body: {
          from_stage: fromStage,
          to_stage: toStage,
          approved: options?.approved ?? true,
          ...(options?.comment ? { comment: options.comment } : {}),
        },
      }
    );
  }

  /**
   * Run pipeline from a raw text braindump.
   */
  async fromBraindump(
    text: string,
    options?: { context?: string; autoAdvance?: boolean },
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
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

  /**
   * Run pipeline from a named template.
   */
  async fromTemplate(
    templateName: string,
    options?: { autoAdvance?: boolean },
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/canvas/pipeline/from-template',
      {
        body: {
          template_name: templateName,
          auto_advance: options?.autoAdvance ?? true,
        },
      }
    );
  }

  /**
   * Execute a pipeline's orchestration stage.
   */
  async execute(
    pipelineId: string,
    options?: { dryRun?: boolean },
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/execute`,
      {
        body: { dry_run: options?.dryRun ?? false },
      }
    );
  }

  /**
   * List available pipeline templates.
   */
  async listTemplates(options?: { category?: string }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      '/api/v1/canvas/pipeline/templates',
      { params: options }
    );
  }

  /**
   * Convert an existing debate into a pipeline.
   */
  async debateToPipeline(
    debateId: string,
    options?: { useUniversal?: boolean; autoAdvance?: boolean },
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
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

  /**
   * Save/update a pipeline.
   */
  async save(
    pipelineId: string,
    data: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'PUT',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}`,
      { body: data }
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

  // ===========================================================================
  // Pipeline Graphs & Transitions
  // ===========================================================================

  /**
   * Get the current pipeline execution graph.
   */
  async getGraph(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      '/api/v1/pipeline/graph'
    );
  }

  /**
   * List saved pipeline graphs.
   */
  async listGraphs(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      '/api/v1/pipeline/graphs'
    );
  }

  /**
   * Create a new pipeline graph.
   *
   * @param graphData - Graph definition (nodes, edges, metadata)
   * @returns Created graph with ID
   */
  async createGraph(graphData: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/pipeline/graphs',
      { body: graphData }
    );
  }

  /**
   * Update an existing pipeline graph.
   *
   * @param graphData - Updated graph definition (must include graph_id)
   * @returns Updated graph
   */
  async updateGraph(graphData: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'PUT',
      '/api/v1/pipeline/graphs',
      { body: graphData }
    );
  }

  /**
   * Delete a pipeline graph.
   *
   * @param options - Delete parameters (graph_id, etc.)
   * @returns Deletion confirmation
   */
  async deleteGraph(options?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'DELETE',
      '/api/v1/pipeline/graphs',
      { body: options }
    );
  }

  /**
   * Create a pipeline stage transition.
   *
   * @param fromStage - Source stage
   * @param toStage - Target stage
   * @param options - Transition options
   * @returns Created transition details
   */
  async createTransition(
    fromStage: string,
    toStage: string,
    options?: { pipelineId?: string; conditions?: Record<string, unknown> },
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {
      from_stage: fromStage,
      to_stage: toStage,
    };
    if (options?.pipelineId) body.pipeline_id = options.pipelineId;
    if (options?.conditions) body.conditions = options.conditions;
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/pipeline/transitions',
      { body }
    );
  }

  // ===========================================================================
  // Pipeline Demo, Automation, and Intelligence
  // ===========================================================================

  /**
   * Create a pre-populated demo pipeline.
   */
  async demo(options?: { ideas?: string[] }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/canvas/pipeline/demo',
      { body: options ?? {} }
    );
  }

  /**
   * Auto-run the full pipeline from unstructured text.
   */
  async autoRun(
    text: string,
    options?: { automationLevel?: string },
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/canvas/pipeline/auto-run',
      {
        body: {
          text,
          automation_level: options?.automationLevel ?? 'full',
        },
      }
    );
  }

  /**
   * Extract principles/values from an ideas canvas.
   */
  async extractPrinciples(
    ideasCanvas: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/canvas/pipeline/extract-principles',
      { body: { ideas_canvas: ideasCanvas } }
    );
  }

  /**
   * Generate a pipeline from current system metrics.
   */
  async fromSystemMetrics(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/canvas/pipeline/from-system-metrics',
      { body: {} }
    );
  }

  /**
   * Get intelligence view for a pipeline.
   */
  async getIntelligence(pipelineId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/intelligence`
    );
  }

  /**
   * Get belief network state for pipeline nodes.
   */
  async getBeliefs(pipelineId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/beliefs`
    );
  }

  /**
   * Get explanations for pipeline nodes.
   */
  async getExplanations(pipelineId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/explanations`
    );
  }

  /**
   * Get precedent links for pipeline nodes.
   */
  async getPrecedents(pipelineId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/precedents`
    );
  }

  /**
   * Trigger self-improvement analysis for a pipeline.
   */
  async selfImprove(pipelineId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/self-improve`
    );
  }

  // ===========================================================================
  // Pipeline Graph Detail Operations
  // ===========================================================================

  /**
   * Get a specific pipeline graph.
   */
  async getGraphById(graphId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}`
    );
  }

  /**
   * Delete a specific pipeline graph.
   */
  async deleteGraphById(graphId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'DELETE',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}`
    );
  }

  /**
   * Add a node to a pipeline graph.
   */
  async addGraphNode(
    graphId: string,
    node: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}/node`,
      { body: node }
    );
  }

  /**
   * Update a node in a pipeline graph.
   */
  async updateGraphNode(
    graphId: string,
    nodeId: string,
    node: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'PUT',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}/node/${encodeURIComponent(nodeId)}`,
      { body: node }
    );
  }

  /**
   * Reassign node ownership in a pipeline graph.
   */
  async reassignGraphNode(
    graphId: string,
    nodeId: string,
    body: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}/node/${encodeURIComponent(nodeId)}/reassign`,
      { body }
    );
  }

  /**
   * List nodes for a pipeline graph.
   */
  async listGraphNodes(
    graphId: string,
    params?: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}/nodes`,
      { params }
    );
  }

  /**
   * Promote a node in a pipeline graph.
   */
  async promoteGraphNode(
    graphId: string,
    body: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}/promote`,
      { body }
    );
  }

  /**
   * Get provenance for a graph node.
   */
  async getGraphNodeProvenance(
    graphId: string,
    nodeId: string,
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}/provenance/${encodeURIComponent(nodeId)}`
    );
  }

  /**
   * Get React Flow view for a specific graph.
   */
  async getGraphReactFlow(graphId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}/react-flow`
    );
  }

  /**
   * Check graph integrity.
   */
  async getGraphIntegrity(graphId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}/integrity`
    );
  }

  /**
   * Get graph improvement suggestions.
   */
  async getGraphSuggestions(graphId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/pipeline/graph/${encodeURIComponent(graphId)}/suggestions`
    );
  }

  // ===========================================================================
  // Pipeline Transition Helpers
  // ===========================================================================

  /**
   * Run idea-to-goals transition helper.
   */
  async ideasToGoals(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/pipeline/transitions/ideas-to-goals',
      { body }
    );
  }

  /**
   * Run goals-to-tasks transition helper.
   */
  async goalsToTasks(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/pipeline/transitions/goals-to-tasks',
      { body }
    );
  }

  /**
   * Run tasks-to-workflow transition helper.
   */
  async tasksToWorkflow(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/pipeline/transitions/tasks-to-workflow',
      { body }
    );
  }

  /**
   * Execute transition plan.
   */
  async executeTransitions(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/pipeline/transitions/execute',
      { body }
    );
  }

  /**
   * Get transition provenance for a node.
   */
  async getTransitionProvenance(nodeId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/pipeline/transitions/${encodeURIComponent(nodeId)}/provenance`
    );
  }
}
