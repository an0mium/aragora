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
  restoreWorkflowVersion(workflowId: string, version: number): Promise<Workflow>;
  deleteWorkflowExecution(executionId: string): Promise<void>;
  listPatternTemplates(): Promise<{ patterns: Array<{ id: string; name: string; description: string }> }>;
  getPatternTemplate(patternId: string): Promise<Record<string, unknown>>;
  instantiatePattern(patternId: string, body: { name: string; description: string; category?: string; config?: Record<string, unknown>; agents?: string[] }): Promise<{ template_id: string; workflow: Workflow }>;
  listRecommendedTemplates(): Promise<{ templates: WorkflowTemplate[] }>;
  listSMEWorkflows(params?: { category?: string; industry?: string } & PaginationParams): Promise<{ workflows: WorkflowTemplate[] }>;
  getSMEWorkflow(workflowId: string): Promise<WorkflowTemplate>;
  executeSMEWorkflow(workflowId: string, body: { inputs?: Record<string, unknown>; context?: Record<string, unknown>; execute?: boolean; tenant_id?: string }): Promise<{ execution_id: string }>;
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
   * List all workflows with optional pagination.
   *
   * @param params - Pagination parameters
   * @param params.limit - Maximum number of workflows to return (default: 20)
   * @param params.offset - Number of workflows to skip (default: 0)
   * @returns List of workflows
   *
   * @example
   * ```typescript
   * const { workflows } = await client.workflows.list({ limit: 10 });
   * for (const workflow of workflows) {
   *   console.log(`${workflow.name}: ${workflow.status}`);
   * }
   * ```
   */
  async list(params?: PaginationParams): Promise<{ workflows: Workflow[] }> {
    return this.client.listWorkflows(params);
  }

  /**
   * Get a workflow by ID.
   *
   * @param workflowId - The unique workflow identifier
   * @returns The workflow details
   *
   * @example
   * ```typescript
   * const workflow = await client.workflows.get('workflow-123');
   * console.log(`Workflow: ${workflow.name}, Nodes: ${workflow.nodes.length}`);
   * ```
   */
  async get(workflowId: string): Promise<Workflow> {
    return this.client.getWorkflow(workflowId);
  }

  /**
   * Create a new workflow.
   *
   * @param workflow - Workflow definition including name, nodes, and edges
   * @returns The created workflow with generated ID
   *
   * @example
   * ```typescript
   * const workflow = await client.workflows.create({
   *   name: 'Document Review',
   *   description: 'Multi-agent document review workflow',
   *   nodes: [
   *     { id: 'review', type: 'debate', config: { agents: ['claude', 'gpt-4'] } },
   *     { id: 'approve', type: 'approval', config: { required_approvers: 1 } },
   *   ],
   *   edges: [{ from: 'review', to: 'approve' }],
   * });
   * ```
   */
  async create(workflow: Partial<Workflow>): Promise<Workflow> {
    return this.client.createWorkflow(workflow);
  }

  /**
   * Update an existing workflow.
   *
   * @param workflowId - The workflow ID to update
   * @param updates - Partial workflow object with fields to update
   * @returns The updated workflow
   */
  async update(workflowId: string, updates: Partial<Workflow>): Promise<Workflow> {
    return this.client.updateWorkflow(workflowId, updates);
  }

  /**
   * Delete a workflow.
   *
   * @param workflowId - The workflow ID to delete
   */
  async delete(workflowId: string): Promise<void> {
    return this.client.deleteWorkflow(workflowId);
  }

  /**
   * Execute a workflow with optional inputs.
   *
   * @param workflowId - The workflow ID to execute
   * @param inputs - Input parameters to pass to the workflow
   * @returns The execution ID for tracking progress
   *
   * @example
   * ```typescript
   * const { execution_id } = await client.workflows.execute('workflow-123', {
   *   document_url: 'https://example.com/doc.pdf',
   *   priority: 'high',
   * });
   * console.log(`Started execution: ${execution_id}`);
   *
   * // Poll for completion
   * let execution = await client.workflows.getExecution(execution_id);
   * while (execution.status === 'running') {
   *   await new Promise(r => setTimeout(r, 1000));
   *   execution = await client.workflows.getExecution(execution_id);
   * }
   * console.log(`Result: ${execution.result}`);
   * ```
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
   *
   * Useful for testing workflow logic, estimating costs, and validating
   * input parameters before running a real execution.
   *
   * @param workflowId - The workflow ID to simulate
   * @param inputs - Input parameters for the simulation
   * @returns Simulation result with predicted outputs, costs, and any validation errors
   *
   * @example
   * ```typescript
   * const simulation = await client.workflows.simulate('workflow-123', {
   *   document_url: 'https://example.com/contract.pdf',
   * });
   *
   * console.log(`Estimated duration: ${simulation.estimated_duration_ms}ms`);
   * console.log(`Estimated cost: $${simulation.estimated_cost}`);
   *
   * if (simulation.validation_errors.length > 0) {
   *   console.log('Validation errors:', simulation.validation_errors);
   * } else {
   *   // Proceed with actual execution
   *   await client.workflows.execute('workflow-123', inputs);
   * }
   * ```
   */
  async simulate(workflowId: string, inputs?: Record<string, unknown>): Promise<WorkflowSimulationResult> {
    return this.client.simulateWorkflow(workflowId, inputs);
  }

  /**
   * List pending workflow approvals.
   *
   * @param params - Filter and pagination options
   * @param params.workflow_id - Filter by specific workflow
   * @param params.status - Filter by approval status (pending, approved, rejected)
   * @returns List of approval requests awaiting resolution
   *
   * @example
   * ```typescript
   * // Get all pending approvals
   * const { approvals } = await client.workflows.listApprovals({ status: 'pending' });
   * for (const approval of approvals) {
   *   console.log(`${approval.workflow_name}: ${approval.reason}`);
   * }
   * ```
   */
  async listApprovals(params?: { workflow_id?: string; status?: string; limit?: number; offset?: number }): Promise<{ approvals: WorkflowApproval[] }> {
    return this.client.listWorkflowApprovals(params);
  }

  /**
   * Approve or reject a workflow approval request.
   *
   * @param approvalId - The approval request ID
   * @param body - Resolution details
   * @param body.approved - Whether to approve (true) or reject (false)
   * @param body.comment - Optional comment explaining the decision
   * @returns Updated approval record
   *
   * @example
   * ```typescript
   * // Approve a workflow
   * await client.workflows.resolveApproval('approval-123', {
   *   approved: true,
   *   comment: 'Reviewed and approved for production use',
   * });
   *
   * // Reject with feedback
   * await client.workflows.resolveApproval('approval-456', {
   *   approved: false,
   *   comment: 'Needs additional security review before proceeding',
   * });
   * ```
   */
  async resolveApproval(approvalId: string, body: { approved: boolean; comment?: string }): Promise<WorkflowApproval> {
    return this.client.resolveWorkflowApproval(approvalId, body);
  }

  /**
   * Restore a specific version of a workflow.
   */
  async restoreVersion(workflowId: string, version: number): Promise<Workflow> {
    return this.client.restoreWorkflowVersion(workflowId, version);
  }

  /**
   * Delete a workflow execution record.
   */
  async deleteExecution(executionId: string): Promise<void> {
    return this.client.deleteWorkflowExecution(executionId);
  }

  /**
   * List available pattern templates for workflow creation.
   */
  async listPatternTemplates(): Promise<{ patterns: Array<{ id: string; name: string; description: string }> }> {
    return this.client.listPatternTemplates();
  }

  /**
   * Get a specific pattern template.
   */
  async getPatternTemplate(patternId: string): Promise<Record<string, unknown>> {
    return this.client.getPatternTemplate(patternId);
  }

  /**
   * Create a workflow from a pattern template.
   */
  async instantiatePattern(patternId: string, body: { name: string; description: string; category?: string; config?: Record<string, unknown>; agents?: string[] }): Promise<{ template_id: string; workflow: Workflow }> {
    return this.client.instantiatePattern(patternId, body);
  }

  /**
   * List recommended workflow templates.
   */
  async listRecommended(): Promise<{ templates: WorkflowTemplate[] }> {
    return this.client.listRecommendedTemplates();
  }

  /**
   * List SME-specific workflow templates.
   */
  async listSME(params?: { category?: string; industry?: string } & PaginationParams): Promise<{ workflows: WorkflowTemplate[] }> {
    return this.client.listSMEWorkflows(params);
  }

  /**
   * Get an SME workflow template.
   */
  async getSME(workflowId: string): Promise<WorkflowTemplate> {
    return this.client.getSMEWorkflow(workflowId);
  }

  /**
   * Execute an SME workflow template.
   */
  async executeSME(workflowId: string, body: { inputs?: Record<string, unknown>; context?: Record<string, unknown>; execute?: boolean; tenant_id?: string }): Promise<{ execution_id: string }> {
    return this.client.executeSMEWorkflow(workflowId, body);
  }
}
