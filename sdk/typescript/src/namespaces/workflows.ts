/**
 * Workflows Namespace API
 *
 * Provides a namespaced interface for workflow operations.
 */

import type { AragoraClient } from '../client';

/**
 * Workflows API namespace.
 *
 * Provides methods for managing automated workflows:
 * - Creating, listing, and managing workflows
 * - Executing workflows and tracking their status
 * - Working with workflow templates
 * - Managing workflow executions
 */
export class WorkflowsAPI {
  constructor(private client: AragoraClient) {}

  // ===========================================================================
  // Workflows CRUD
  // ===========================================================================

  /**
   * List all workflows.
   * @route GET /api/v1/workflows
   */
  async list(params?: { limit?: number; offset?: number }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/workflows', {
      params: params as Record<string, unknown>,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Create a new workflow.
   * @route POST /api/v1/workflows
   */
  async create(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/workflows', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get a workflow by ID.
   * @route GET /api/v1/workflows/{workflow_id}
   */
  async get(workflowId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Update a workflow.
   * @route PUT /api/v1/workflows/{workflow_id}
   */
  async update(workflowId: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'PUT',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Patch a workflow.
   * @route PATCH /api/v1/workflows/{workflow_id}
   */
  async patch(workflowId: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'PATCH',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Delete a workflow.
   * @route DELETE /api/v1/workflows/{workflow_id}
   */
  async delete(workflowId: string): Promise<void> {
    return this.client.request(
      'DELETE',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}`
    ) as Promise<void>;
  }

  /**
   * Execute a workflow.
   * @route POST /api/v1/workflows/{workflow_id}/execute
   */
  async execute(workflowId: string, inputs?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}/execute`,
      { body: inputs }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Simulate a workflow without executing.
   * @route POST /api/v1/workflows/{workflow_id}/simulate
   */
  async simulate(workflowId: string, inputs?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}/simulate`,
      { body: inputs }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get workflow status.
   * @route GET /api/v1/workflows/{workflow_id}/status
   */
  async getStatus(workflowId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}/status`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get workflow versions.
   * @route GET /api/v1/workflows/{workflow_id}/versions
   */
  async getVersions(workflowId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}/versions`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Restore a specific version of a workflow.
   * @route POST /api/v1/workflows/{workflow_id}/versions/{version}/restore
   */
  async restoreVersion(workflowId: string, version: number): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}/versions/${version}/restore`
    ) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Executions
  // ===========================================================================

  /**
   * List workflow executions.
   * @route POST /api/v1/workflows/executions
   */
  async listExecutions(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/workflows/executions', {
      body: params,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get a workflow execution by ID.
   * @route GET /api/v1/workflows/executions/{execution_id}
   */
  async getExecution(executionId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/workflows/executions/${encodeURIComponent(executionId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Delete a workflow execution.
   * @route DELETE /api/v1/workflows/executions/{execution_id}
   */
  async deleteExecution(executionId: string): Promise<void> {
    return this.client.request(
      'DELETE',
      `/api/v1/workflows/executions/${encodeURIComponent(executionId)}`
    ) as Promise<void>;
  }

  /**
   * Update executions.
   * @route PATCH /api/v1/workflows/executions
   */
  async updateExecutions(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('PATCH', '/api/v1/workflows/executions', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Replace executions.
   * @route PUT /api/v1/workflows/executions
   */
  async replaceExecutions(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('PUT', '/api/v1/workflows/executions', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Delete executions.
   * @route DELETE /api/v1/workflows/executions
   */
  async deleteExecutions(): Promise<void> {
    return this.client.request('DELETE', '/api/v1/workflows/executions') as Promise<void>;
  }

  // ===========================================================================
  // Templates
  // ===========================================================================

  /**
   * Create a workflow template.
   * @route POST /api/v1/workflows/templates
   */
  async createTemplate(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/workflows/templates', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get a workflow template by ID.
   * @route GET /api/v1/workflows/templates/{template_id}
   */
  async getTemplate(templateId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/workflows/templates/${encodeURIComponent(templateId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Update a workflow template.
   * @route PUT /api/v1/workflows/templates/{template_id}
   */
  async updateTemplate(templateId: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'PUT',
      `/api/v1/workflows/templates/${encodeURIComponent(templateId)}`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Delete a workflow template.
   * @route DELETE /api/v1/workflows/templates/{template_id}
   */
  async deleteTemplate(templateId: string): Promise<void> {
    return this.client.request(
      'DELETE',
      `/api/v1/workflows/templates/${encodeURIComponent(templateId)}`
    ) as Promise<void>;
  }

  /**
   * Patch templates.
   * @route PATCH /api/v1/workflows/templates
   */
  async patchTemplates(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('PATCH', '/api/v1/workflows/templates', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Replace templates.
   * @route PUT /api/v1/workflows/templates
   */
  async replaceTemplates(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('PUT', '/api/v1/workflows/templates', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Delete templates.
   * @route DELETE /api/v1/workflows/templates
   */
  async deleteTemplates(): Promise<void> {
    return this.client.request('DELETE', '/api/v1/workflows/templates') as Promise<void>;
  }

  // ===========================================================================
  // Workflow Templates (top-level /workflow-templates path)
  // ===========================================================================

  /**
   * List workflow templates.
   * @route GET /api/v1/workflow-templates
   */
  async listWorkflowTemplates(params?: { category?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/workflow-templates', {
      params: params as Record<string, unknown>,
    }) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Workflow Executions (top-level /workflow-executions path)
  // ===========================================================================

  /**
   * List workflow executions.
   * @route GET /api/v1/workflow-executions
   */
  async listWorkflowExecutions(params?: {
    workflow_id?: string;
    status?: string;
    limit?: number;
  }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/workflow-executions', {
      params: params as Record<string, unknown>,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get a workflow execution by ID.
   * @route GET /api/v1/workflow-executions/{execution_id}
   */
  async getWorkflowExecution(executionId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/workflow-executions/${encodeURIComponent(executionId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Terminate a running workflow execution.
   * @route DELETE /api/v1/workflow-executions/{execution_id}
   */
  async terminateExecution(executionId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'DELETE',
      `/api/v1/workflow-executions/${encodeURIComponent(executionId)}`
    ) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Approvals
  // ===========================================================================

  /**
   * List pending workflow approvals.
   * @route GET /api/v1/workflow-approvals
   */
  async listApprovals(params?: {
    workflow_id?: string;
    tenant_id?: string;
  }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/workflow-approvals', {
      params: params as Record<string, unknown>,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Resolve a pending workflow approval request.
   * @route POST /api/v1/workflow-approvals/{request_id}/resolve
   */
  async resolveApproval(
    requestId: string,
    body: { status?: string; notes?: string; checklist?: Record<string, unknown> }
  ): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/workflow-approvals/${encodeURIComponent(requestId)}/resolve`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Visual Builder
  // ===========================================================================

  /**
   * Auto-layout workflow steps for visual rendering.
   * @route POST /api/v1/workflows/auto-layout
   */
  async autoLayout(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/workflows/auto-layout', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Create a workflow from a known pattern.
   * @route POST /api/v1/workflows/from-pattern
   */
  async fromPattern(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/workflows/from-pattern', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Generate a workflow from natural language input.
   * @route POST /api/v1/workflows/generate
   */
  async generate(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/workflows/generate', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * List supported workflow builder step types.
   * @route GET /api/v1/workflows/step-types
   */
  async listStepTypes(params?: { category?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/workflows/step-types', {
      params: params as Record<string, unknown>,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Validate a workflow definition.
   * @route POST /api/v1/workflows/validate
   */
  async validate(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/workflows/validate', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Replay a workflow with provided inputs.
   * @route POST /api/v1/workflows/{workflow_id}/replay
   */
  async replay(workflowId: string, body?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/workflows/${encodeURIComponent(workflowId)}/replay`, {
      body,
    }) as Promise<Record<string, unknown>>;
  }
}
