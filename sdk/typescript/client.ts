// Auto-generated TypeScript client from OpenAPI spec
// Do not edit manually

import type { ApiConfig } from './api-types';

export class AragoraClient {
  private config: ApiConfig;

  constructor(config: ApiConfig) {
    this.config = config;
  }

  private async request<T>(
    method: string,
    path: string,
    options: {
      body?: unknown;
      params?: Record<string, string | number | boolean>;
      headers?: Record<string, string>;
    } = {}
  ): Promise<T> {
    const url = new URL(path, this.config.baseUrl);

    if (options.params) {
      Object.entries(options.params).forEach(([key, value]) => {
        url.searchParams.append(key, String(value));
      });
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.config.headers,
      ...options.headers,
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    const response = await fetch(url.toString(), {
      method,
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Debates
  async listDebates(params?: { limit?: number; offset?: number }) {
    return this.request<{ debates: unknown[] }>('GET', '/api/debates', { params });
  }

  async getDebate(debateId: string) {
    return this.request<unknown>('GET', `/api/debates/${debateId}`);
  }

  async createDebate(body: { question: string; agents: string[]; rounds?: number }) {
    return this.request<{ debate_id: string; success: boolean }>('POST', '/api/debate', { body });
  }

  // Explainability
  async getExplanation(debateId: string, options?: {
    include_factors?: boolean;
    include_counterfactuals?: boolean;
    include_provenance?: boolean;
  }) {
    return this.request<unknown>('GET', `/api/debates/${debateId}/explainability`, { params: options });
  }

  async getFactors(debateId: string, options?: { min_contribution?: number }) {
    return this.request<unknown>('GET', `/api/debates/${debateId}/explainability/factors`, { params: options });
  }

  async getCounterfactuals(debateId: string, options?: { max_scenarios?: number }) {
    return this.request<unknown>('GET', `/api/debates/${debateId}/explainability/counterfactual`, { params: options });
  }

  async generateCounterfactual(debateId: string, body: { hypothesis: string; affected_agents?: string[] }) {
    return this.request<unknown>('POST', `/api/debates/${debateId}/explainability/counterfactual`, { body });
  }

  async getProvenance(debateId: string) {
    return this.request<unknown>('GET', `/api/debates/${debateId}/explainability/provenance`);
  }

  async getNarrative(debateId: string, options?: { format?: 'brief' | 'detailed' | 'executive_summary' }) {
    return this.request<unknown>('GET', `/api/debates/${debateId}/explainability/narrative`, { params: options });
  }

  // Workflow Templates
  async listWorkflowTemplates(params?: {
    category?: string;
    pattern?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }) {
    return this.request<{ templates: unknown[]; total: number }>('GET', '/api/workflow/templates', { params });
  }

  async getWorkflowTemplate(templateId: string) {
    return this.request<unknown>('GET', `/api/workflow/templates/${templateId}`);
  }

  async getWorkflowTemplatePackage(templateId: string, includeExamples?: boolean) {
    return this.request<unknown>('GET', `/api/workflow/templates/${templateId}/package`, {
      params: { include_examples: includeExamples },
    });
  }

  async runWorkflowTemplate(templateId: string, body: {
    inputs?: Record<string, unknown>;
    config?: { timeout?: number; priority?: string; async?: boolean };
    workspace_id?: string;
  }) {
    return this.request<unknown>('POST', `/api/workflow/templates/${templateId}/run`, { body });
  }

  async listWorkflowCategories() {
    return this.request<{ categories: unknown[] }>('GET', '/api/workflow/categories');
  }

  async listWorkflowPatterns() {
    return this.request<{ patterns: unknown[] }>('GET', '/api/workflow/patterns');
  }

  // Gauntlet
  async listGauntletReceipts(params?: { verdict?: string; limit?: number; offset?: number }) {
    return this.request<{ receipts: unknown[]; total: number }>('GET', '/api/gauntlet/receipts', { params });
  }

  async getGauntletReceipt(receiptId: string) {
    return this.request<unknown>('GET', `/api/gauntlet/receipts/${receiptId}`);
  }

  async verifyGauntletReceipt(receiptId: string) {
    return this.request<{ valid: boolean; hash: string }>('GET', `/api/gauntlet/receipts/${receiptId}/verify`);
  }

  async exportGauntletReceipt(receiptId: string, format: 'json' | 'html' | 'markdown' | 'sarif') {
    return this.request<unknown>('GET', `/api/gauntlet/receipts/${receiptId}/export`, { params: { format } });
  }

  // Agents
  async listAgents() {
    return this.request<{ agents: unknown[] }>('GET', '/api/agents');
  }

  async getAgent(agentName: string) {
    return this.request<unknown>('GET', `/api/agents/${agentName}`);
  }

  // Health
  async getHealth() {
    return this.request<{ status: string }>('GET', '/api/health');
  }
}

export function createClient(config: ApiConfig): AragoraClient {
  return new AragoraClient(config);
}
