/**
 * Synchronous wrapper for AragoraClient
 *
 * Provides a simplified interface for common operations.
 * Methods return Promises but are designed for use with async/await
 * or top-level await in Node.js environments.
 *
 * @example
 * ```typescript
 * import { AragoraClientSync } from '@aragora/sdk';
 *
 * // In an async context
 * async function main() {
 *   const client = new AragoraClientSync({
 *     baseUrl: 'https://api.aragora.ai',
 *     apiKey: 'your-api-key'
 *   });
 *
 *   const debates = await client.listDebates();
 *   console.log(debates);
 * }
 * ```
 */

import { AragoraClient } from './client';
import type {
  AragoraConfig,
  Debate,
  DebateCreateRequest,
  DebateCreateResponse,
  ExplainabilityResult,
  HealthCheck,
  MarketplaceTemplate,
  PaginationParams,
  WorkflowTemplate,
  DecisionReceipt,
  Agent,
} from './types';

/**
 * Synchronous-style wrapper for the Aragora API client.
 *
 * This class wraps the async AragoraClient and provides the same methods
 * for easier use with async/await patterns. Designed to match the Python
 * SDK's AragoraClientSync interface.
 */
export class AragoraClientSync {
  private client: AragoraClient;

  constructor(config: AragoraConfig) {
    this.client = new AragoraClient(config);
  }

  /**
   * Get the underlying async client for advanced use cases.
   */
  getAsyncClient(): AragoraClient {
    return this.client;
  }

  // ===========================================================================
  // Health
  // ===========================================================================

  async health(): Promise<HealthCheck> {
    return this.client.getHealth();
  }

  // ===========================================================================
  // Debates
  // ===========================================================================

  async listDebates(params?: PaginationParams & { status?: string }): Promise<Debate[]> {
    const result = await this.client.listDebates(params);
    return result.debates;
  }

  async getDebate(debateId: string): Promise<Debate> {
    return this.client.getDebate(debateId);
  }

  async createDebate(request: DebateCreateRequest): Promise<DebateCreateResponse> {
    return this.client.createDebate(request);
  }

  // ===========================================================================
  // Explainability
  // ===========================================================================

  async getExplanation(debateId: string, options?: {
    include_factors?: boolean;
    include_counterfactuals?: boolean;
    include_provenance?: boolean;
  }): Promise<ExplainabilityResult> {
    return this.client.getExplanation(debateId, options);
  }

  async createBatchExplanation(debateIds: string[], options?: {
    include_evidence?: boolean;
    include_counterfactuals?: boolean;
    format?: 'full' | 'summary' | 'minimal';
  }): Promise<{ batch_id: string; status_url: string }> {
    return this.client.createBatchExplanation({
      debate_ids: debateIds,
      options,
    });
  }

  async getBatchStatus(batchId: string): Promise<{
    batch_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress_pct: number;
  }> {
    return this.client.getBatchExplanationStatus(batchId);
  }

  async getBatchResults(batchId: string, params?: PaginationParams): Promise<
    Array<{ debate_id: string; status: string; explanation?: unknown }>
  > {
    const result = await this.client.getBatchExplanationResults(batchId, params);
    return result.results;
  }

  async compareExplanations(debateIds: string[]): Promise<unknown> {
    // Batch compare explanations
    const batch = await this.createBatchExplanation(debateIds, { format: 'full' });
    // Poll until complete
    let status = await this.getBatchStatus(batch.batch_id);
    while (status.status === 'pending' || status.status === 'processing') {
      await this.sleep(1000);
      status = await this.getBatchStatus(batch.batch_id);
    }
    return this.getBatchResults(batch.batch_id);
  }

  // ===========================================================================
  // Workflows
  // ===========================================================================

  async listWorkflowTemplates(params?: {
    category?: string;
    pattern?: string;
    search?: string;
    tags?: string;
  } & PaginationParams): Promise<WorkflowTemplate[]> {
    const result = await this.client.listWorkflowTemplates(params);
    return result.templates;
  }

  async getWorkflowTemplate(templateId: string): Promise<WorkflowTemplate> {
    return this.client.getWorkflowTemplate(templateId);
  }

  async getWorkflowTemplatePackage(templateId: string, options?: {
    include_examples?: boolean;
  }): Promise<{ template: WorkflowTemplate; examples?: unknown[] }> {
    return this.client.getWorkflowTemplatePackage(templateId, options);
  }

  async instantiatePattern(patternId: string, body: {
    name: string;
    description: string;
    category?: string;
    config?: Record<string, unknown>;
    agents?: string[];
  }): Promise<{ template_id: string; workflow: unknown }> {
    return this.client.instantiatePattern(patternId, body);
  }

  // ===========================================================================
  // Marketplace
  // ===========================================================================

  async browseMarketplace(params?: {
    category?: string;
    search?: string;
    sort_by?: 'downloads' | 'rating' | 'newest';
    min_rating?: number;
  } & PaginationParams): Promise<MarketplaceTemplate[]> {
    const result = await this.client.browseMarketplace(params);
    return result.templates;
  }

  async getMarketplaceTemplate(templateId: string): Promise<MarketplaceTemplate> {
    return this.client.getMarketplaceTemplate(templateId);
  }

  async publishTemplate(body: {
    template_id: string;
    name: string;
    description: string;
    category: string;
    tags?: string[];
    documentation?: string;
  }): Promise<{ marketplace_id: string }> {
    return this.client.publishTemplate(body);
  }

  async rateTemplate(templateId: string, rating: number): Promise<{ new_rating: number }> {
    return this.client.rateTemplate(templateId, rating);
  }

  async reviewTemplate(templateId: string, body: {
    rating: number;
    title: string;
    content: string;
  }): Promise<{ review_id: string }> {
    return this.client.reviewTemplate(templateId, body);
  }

  async importTemplate(templateId: string, workspaceId?: string): Promise<{ imported_id: string }> {
    return this.client.importTemplate(templateId, workspaceId);
  }

  async getFeaturedTemplates(): Promise<MarketplaceTemplate[]> {
    const result = await this.client.getFeaturedTemplates();
    return result.templates;
  }

  async getTrendingTemplates(): Promise<MarketplaceTemplate[]> {
    const result = await this.client.getTrendingTemplates();
    return result.templates;
  }

  async getMarketplaceCategories(): Promise<string[]> {
    const result = await this.client.getMarketplaceCategories();
    return result.categories;
  }

  // ===========================================================================
  // Gauntlet
  // ===========================================================================

  async listGauntletReceipts(params?: { verdict?: string } & PaginationParams): Promise<DecisionReceipt[]> {
    const result = await this.client.listGauntletReceipts(params);
    return result.receipts;
  }

  async getGauntletReceipt(receiptId: string): Promise<DecisionReceipt> {
    return this.client.getGauntletReceipt(receiptId);
  }

  async verifyGauntletReceipt(receiptId: string): Promise<{ valid: boolean; hash: string }> {
    return this.client.verifyGauntletReceipt(receiptId);
  }

  async exportGauntletReceipt(receiptId: string, format: 'json' | 'html' | 'markdown' | 'sarif'): Promise<unknown> {
    return this.client.exportGauntletReceipt(receiptId, format);
  }

  // ===========================================================================
  // Agents
  // ===========================================================================

  async listAgents(): Promise<Agent[]> {
    const result = await this.client.listAgents();
    return result.agents;
  }

  async getAgent(name: string): Promise<Agent> {
    return this.client.getAgent(name);
  }

  // ===========================================================================
  // Utilities
  // ===========================================================================

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

/**
 * Create a new sync-style Aragora client instance.
 */
export function createSyncClient(config: AragoraConfig): AragoraClientSync {
  return new AragoraClientSync(config);
}
