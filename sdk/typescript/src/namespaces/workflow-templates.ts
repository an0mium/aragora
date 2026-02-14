/**
 * Workflow Templates Namespace API
 *
 * Provides access to pre-built workflow templates for common automation patterns.
 */

export interface WorkflowTemplate {
  id: string;
  name: string;
  description?: string;
  category: string;
  pattern?: string;
  tags?: string[];
  version?: string;
  created_at?: string;
  updated_at?: string;
  metadata?: Record<string, unknown>;
}

export interface WorkflowTemplatePackage {
  template: WorkflowTemplate;
  definition: Record<string, unknown>;
  dependencies?: string[];
  examples?: Array<{ name: string; inputs: Record<string, unknown> }>;
}

export interface WorkflowTemplateRunResult {
  execution_id: string;
  status: 'started' | 'running' | 'completed' | 'failed';
  template_id: string;
  started_at: string;
  result?: Record<string, unknown>;
}

export interface ListWorkflowTemplatesParams {
  category?: string;
  pattern?: string;
  search?: string;
  tags?: string[];
  limit?: number;
  offset?: number;
}

export interface RunWorkflowTemplateParams {
  inputs?: Record<string, unknown>;
  config?: Record<string, unknown>;
  workspace_id?: string;
}

interface WorkflowTemplatesClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

export class WorkflowTemplatesAPI {
  constructor(private client: WorkflowTemplatesClientInterface) {}

  /**
   * List available workflow templates.
   */
  async list(
    params?: ListWorkflowTemplatesParams
  ): Promise<{ templates: WorkflowTemplate[]; total: number }> {
    return this.client.request('GET', '/api/v1/workflow/templates', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get a specific workflow template by ID.
   */
  async get(templateId: string): Promise<WorkflowTemplate> {
    return this.client.request('GET', `/api/v1/workflow/templates/${templateId}`);
  }

  /**
   * Get the full template package including definition and examples.
   */
  async getPackage(templateId: string): Promise<WorkflowTemplatePackage> {
    return this.client.request('GET', `/api/v1/workflow/templates/${templateId}/package`);
  }

  /**
   * Run a workflow template with the given inputs.
   */
  async run(
    templateId: string,
    params?: RunWorkflowTemplateParams
  ): Promise<WorkflowTemplateRunResult> {
    return this.client.request('POST', `/api/v1/workflow/templates/${templateId}/run`, {
      body: params,
    });
  }

  /**
   * List workflow executions.
   */
  async listExecutions(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/workflows/executions', { params }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get workflow execution.
   */
  async getExecution(executionId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/workflows/executions/${executionId}`) as Promise<Record<string, unknown>>;
  }

  /**
   * List workflow templates.
   */
  async listTemplates(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/workflows/templates', { params }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get workflow template.
   */
  async getTemplate(templateId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/workflows/templates/${templateId}`) as Promise<Record<string, unknown>>;
  }
}
