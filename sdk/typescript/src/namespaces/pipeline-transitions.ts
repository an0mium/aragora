/**
 * Pipeline Transitions Namespace API
 *
 * Stage-to-stage AI transitions for the Idea-to-Execution pipeline.
 */

interface PipelineTransitionsClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class PipelineTransitionsNamespace {
  constructor(private client: PipelineTransitionsClientInterface) {}

  /** Trigger a stage transition for a pipeline item. */
  async transition(
    pipelineId: string,
    itemId: string,
    targetStage: string,
  ): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v2/pipelines/${pipelineId}/items/${itemId}/transition`, {
      body: { target_stage: targetStage },
    });
  }

  /** Get transition history for a pipeline item. */
  async getHistory(
    pipelineId: string,
    itemId: string,
  ): Promise<Record<string, unknown>[]> {
    return this.client.request('GET', `/api/v2/pipelines/${pipelineId}/items/${itemId}/transitions`);
  }

  /** Validate whether a transition is allowed. */
  async validate(
    pipelineId: string,
    itemId: string,
    targetStage: string,
  ): Promise<{ allowed: boolean; reason?: string }> {
    return this.client.request('POST', `/api/v2/pipelines/${pipelineId}/items/${itemId}/transition/validate`, {
      body: { target_stage: targetStage },
    });
  }

  /** Get available transitions for current stage. */
  async available(
    pipelineId: string,
    itemId: string,
  ): Promise<{ stages: string[] }> {
    return this.client.request('GET', `/api/v2/pipelines/${pipelineId}/items/${itemId}/transitions/available`);
  }

  /** Rollback to a previous stage. */
  async rollback(
    pipelineId: string,
    itemId: string,
    targetStage: string,
  ): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v2/pipelines/${pipelineId}/items/${itemId}/transition/rollback`, {
      body: { target_stage: targetStage },
    });
  }
}
