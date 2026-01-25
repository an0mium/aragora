/**
 * Workflows Namespace API
 *
 * Provides a namespaced interface for workflow operations.
 */

import type {
  PaginationParams,
  Workflow,
  WorkflowApproval,
  WorkflowExecution,
  WorkflowSimulationResult,
  WorkflowTemplate,
  WorkflowTemplatePackage,
  WorkflowTemplateRunResult,
  WorkflowVersion,
} from '../types';

/**
 * Interface for the internal client methods used by WorkflowsAPI.
 */
interface WorkflowsClientInterface {
  listWorkflows(params?: PaginationParams): Promise<{ workflows: Workflow[] }>;
  getWorkflow(workflowId: string): Promise<Workflow>;
  createWorkflow(workflow: Partial<Workflow>): Promise<Workflow>;
  updateWorkflow(workflowId: string, updates: Partial<Workflow>): Promise<Workflow>;
  deleteWorkflow(workflowId: string): Promise<void>;
  executeWorkflow(workflowId: string, inputs?: Record<string, unknown>): Promise<{ execution_id: string }>;
  listWorkflowTemplates(params?: { category?: string; limit?: number; offset?: number }): Promise<{ templates: WorkflowTemplate[] }>;
  getWorkflowTemplate(templateId: string): Promise<WorkflowTemplate>;
  runWorkflowTemplate(templateId: string, body?: { inputs?: Record<string, unknown>; config?: Record<string, unknown> }): Promise<WorkflowTemplateRunResult>;
  listWorkflowCategories(): Promise<{ categories: string[] }>;
  listWorkflowPatterns(): Promise<{ patterns: string[] }>;
  getWorkflowTemplatePackage(templateId: string, options?: { include_examples?: boolean }): Promise<WorkflowTemplatePackage>;
  listWorkflowExecutions(params?: { workflow_id?: string; status?: string; limit?: number; offset?: number }): Promise<{ executions: WorkflowExecution[] }>;
  getWorkflowExecution(executionId: string): Promise<WorkflowExecution>;
  getWorkflowStatus(workflowId: string): Promise<WorkflowExecution>;
  getWorkflowVersions(workflowId: string): Promise<{ versions: WorkflowVersion[] }>;
  simulateWorkflow(workflowId: string, inputs?: Record<string, unknown>): Promise<WorkflowSimulationResult>;
  listWorkflowApprovals(params?: { workflow_id?: string; status?: string; limit?: number; offset?: number }): Promise<{ approvals: WorkflowApproval[] }>;
  resolveWorkflowApproval(approvalId: string, body: { approved: boolean; comment?: string }): Promise<WorkflowApproval>;
}

/**
 * Workflows API namespace.
 *
 * Provides methods for managing automated workflows:
 * - Creating, listing, and managing workflows
 * - Executing workflows and tracking their status
 * - Working with workflow templates
 * - Managing workflow approvals
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // List all workflows
 * const { workflows } = await client.workflows.list();
 *
 * // Execute a workflow
 * const { execution_id } = await client.workflows.execute('workflow-123', {
 *   input_param: 'value'
 * });
 *
 * // Check execution status
 * const execution = await client.workflows.getExecution(execution_id);
 *
 * // List available templates
 * const { templates } = await client.workflows.listTemplates();
 * ```
 */
export class WorkflowsAPI {
  constructor(private client: WorkflowsClientInterface) {}

  /**
   * List all workflows.
   */
  async list(params?: PaginationParams): Promise<{ workflows: Workflow[] }> {
    return this.client.listWorkflows(params);
  }

  /**
   * Get a workflow by ID.
   */
  async get(workflowId: string): Promise<Workflow> {
    return this.client.getWorkflow(workflowId);
  }

  /**
   * Create a new workflow.
   */
  async create(workflow: Partial<Workflow>): Promise<Workflow> {
    return this.client.createWorkflow(workflow);
  }

  /**
   * Update an existing workflow.
   */
  async update(workflowId: string, updates: Partial<Workflow>): Promise<Workflow> {
    return this.client.updateWorkflow(workflowId, updates);
  }

  /**
   * Delete a workflow.
   */
  async delete(workflowId: string): Promise<void> {
    return this.client.deleteWorkflow(workflowId);
  }

  /**
   * Execute a workflow with optional inputs.
   */
  async execute(workflowId: string, inputs?: Record<string, unknown>): Promise<{ execution_id: string }> {
    return this.client.executeWorkflow(workflowId, inputs);
  }

  /**
   * List workflow templates.
   */
  async listTemplates(params?: { category?: string; limit?: number; offset?: number }): Promise<{ templates: WorkflowTemplate[] }> {
    return this.client.listWorkflowTemplates(params);
  }

  /**
   * Get a workflow template by ID.
   */
  async getTemplate(templateId: string): Promise<WorkflowTemplate> {
    return this.client.getWorkflowTemplate(templateId);
  }

  /**
   * Run a workflow template with optional inputs.
   */
  async runTemplate(templateId: string, body?: { inputs?: Record<string, unknown>; config?: Record<string, unknown> }): Promise<WorkflowTemplateRunResult> {
    return this.client.runWorkflowTemplate(templateId, body);
  }

  /**
   * List all workflow categories.
   */
  async listCategories(): Promise<{ categories: string[] }> {
    return this.client.listWorkflowCategories();
  }

  /**
   * List all workflow patterns.
   */
  async listPatterns(): Promise<{ patterns: string[] }> {
    return this.client.listWorkflowPatterns();
  }

  /**
   * Get a workflow template package (for download/export).
   */
  async getTemplatePackage(templateId: string, options?: { include_examples?: boolean }): Promise<WorkflowTemplatePackage> {
    return this.client.getWorkflowTemplatePackage(templateId, options);
  }

  /**
   * List workflow executions.
   */
  async listExecutions(params?: { workflow_id?: string; status?: string; limit?: number; offset?: number }): Promise<{ executions: WorkflowExecution[] }> {
    return this.client.listWorkflowExecutions(params);
  }

  /**
   * Get a workflow execution by ID.
   */
  async getExecution(executionId: string): Promise<WorkflowExecution> {
    return this.client.getWorkflowExecution(executionId);
  }

  /**
   * Get the current status of a workflow.
   */
  async getStatus(workflowId: string): Promise<WorkflowExecution> {
    return this.client.getWorkflowStatus(workflowId);
  }

  /**
   * Get all versions of a workflow.
   */
  async getVersions(workflowId: string): Promise<{ versions: WorkflowVersion[] }> {
    return this.client.getWorkflowVersions(workflowId);
  }

  /**
   * Simulate a workflow without actually executing it.
   */
  async simulate(workflowId: string, inputs?: Record<string, unknown>): Promise<WorkflowSimulationResult> {
    return this.client.simulateWorkflow(workflowId, inputs);
  }

  /**
   * List pending workflow approvals.
   */
  async listApprovals(params?: { workflow_id?: string; status?: string; limit?: number; offset?: number }): Promise<{ approvals: WorkflowApproval[] }> {
    return this.client.listWorkflowApprovals(params);
  }

  /**
   * Approve or reject a workflow approval request.
   */
  async resolveApproval(approvalId: string, body: { approved: boolean; comment?: string }): Promise<WorkflowApproval> {
    return this.client.resolveWorkflowApproval(approvalId, body);
  }
}
