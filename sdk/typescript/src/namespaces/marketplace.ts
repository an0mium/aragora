/**
 * Marketplace Namespace API
 *
 * Provides a namespaced interface for marketplace operations including
 * template publishing, discovery, rating, and deployment.
 */

import type { MarketplaceTemplate, PaginationParams, TemplateReview } from '../types';

/**
 * Template deployment status.
 */
export type DeploymentStatus = 'pending' | 'active' | 'paused' | 'archived' | 'failed';

/**
 * Template deployment information.
 */
export interface TemplateDeployment {
  id: string;
  template_id: string;
  tenant_id: string;
  name: string;
  status: DeploymentStatus;
  config: Record<string, unknown>;
  deployed_at: string;
  last_run?: string;
  run_count: number;
}

/**
 * Template rating summary.
 */
export interface TemplateRatings {
  ratings: Array<{ user_id: string; rating: number; created_at: string }>;
  average: number;
  count: number;
}

/**
 * Interface for the internal client methods used by MarketplaceAPI.
 */
interface MarketplaceClientInterface {
  listMarketplaceTemplates(params?: {
    category?: string;
    search?: string;
    tags?: string[];
    verified_only?: boolean;
    min_rating?: number;
  } & PaginationParams): Promise<{ templates: MarketplaceTemplate[] }>;
  getMarketplaceTemplate(templateId: string): Promise<MarketplaceTemplate>;
  publishTemplate(body: {
    template_id: string;
    name: string;
    description: string;
    category: string;
    tags?: string[];
    workflow_definition?: Record<string, unknown>;
    documentation?: string;
  }): Promise<{ marketplace_id: string }>;
  importTemplate(templateId: string, workspaceId?: string): Promise<{ imported_id: string }>;
  rateTemplate(templateId: string, rating: number): Promise<{ new_rating: number }>;
  reviewTemplate(templateId: string, body: {
    rating: number;
    title: string;
    content: string;
  }): Promise<{ review_id: string }>;
  getFeaturedTemplates(): Promise<{ templates: MarketplaceTemplate[] }>;
  getTrendingTemplates(): Promise<{ templates: MarketplaceTemplate[] }>;
  getMarketplaceCategories(): Promise<{ categories: string[] }>;
  // Methods that may need to be added to client
  getTemplateReviews?(templateId: string, params?: PaginationParams): Promise<{ reviews: TemplateReview[]; total: number }>;
  getTemplateRatings?(templateId: string): Promise<TemplateRatings>;
  searchMarketplace?(params: { q?: string; category?: string; tags?: string[] }): Promise<{ results: MarketplaceTemplate[]; total: number }>;
  deployTemplate?(templateId: string, body?: { name?: string; config?: Record<string, unknown> }): Promise<{ deployment: TemplateDeployment }>;
  listDeployments?(): Promise<{ deployments: TemplateDeployment[] }>;
  getDeployment?(deploymentId: string): Promise<{ deployment: TemplateDeployment }>;
  deleteDeployment?(deploymentId: string): Promise<{ success: boolean }>;
}

/**
 * Marketplace API namespace.
 *
 * Provides methods for interacting with the Aragora template marketplace:
 * - Discovering and searching templates
 * - Publishing and managing templates
 * - Rating and reviewing templates
 * - Deploying templates to workspaces
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Browse featured templates
 * const { templates } = await client.marketplace.getFeatured();
 *
 * // Search for templates
 * const results = await client.marketplace.list({ category: 'analysis' });
 *
 * // Import a template to your workspace
 * const { imported_id } = await client.marketplace.import('template-123');
 *
 * // Rate a template
 * await client.marketplace.rate('template-123', 5);
 *
 * // Submit a review
 * await client.marketplace.review('template-123', {
 *   rating: 5,
 *   title: 'Great template!',
 *   content: 'Works perfectly for our use case.'
 * });
 *
 * // Publish your own template
 * const { marketplace_id } = await client.marketplace.publish({
 *   template_id: 'my-template',
 *   name: 'My Analysis Template',
 *   description: 'A template for analyzing data',
 *   category: 'analysis'
 * });
 * ```
 */
export class MarketplaceAPI {
  constructor(private client: MarketplaceClientInterface) {}

  // ===========================================================================
  // Discovery
  // ===========================================================================

  /**
   * List marketplace templates with optional filtering.
   *
   * @param params - Filter and pagination options
   * @returns List of templates matching the criteria
   */
  async list(params?: {
    category?: string;
    search?: string;
    tags?: string[];
    verified_only?: boolean;
    min_rating?: number;
    limit?: number;
    offset?: number;
  }): Promise<{ templates: MarketplaceTemplate[] }> {
    return this.client.listMarketplaceTemplates(params);
  }

  /**
   * Get a specific template by ID.
   *
   * @param templateId - The template ID
   * @returns The template details
   */
  async get(templateId: string): Promise<MarketplaceTemplate> {
    return this.client.getMarketplaceTemplate(templateId);
  }

  /**
   * Get featured templates curated by the Aragora team.
   *
   * @returns List of featured templates
   */
  async getFeatured(): Promise<{ templates: MarketplaceTemplate[] }> {
    return this.client.getFeaturedTemplates();
  }

  /**
   * Get trending templates based on recent activity.
   *
   * @returns List of trending templates
   */
  async getTrending(): Promise<{ templates: MarketplaceTemplate[] }> {
    return this.client.getTrendingTemplates();
  }

  /**
   * Get all marketplace categories.
   *
   * @returns List of category names
   */
  async getCategories(): Promise<{ categories: string[] }> {
    return this.client.getMarketplaceCategories();
  }

  /**
   * Search marketplace templates.
   *
   * @param params - Search parameters
   * @returns Search results
   */
  async search(params: {
    q?: string;
    category?: string;
    tags?: string[];
  }): Promise<{ results: MarketplaceTemplate[]; total: number }> {
    if (this.client.searchMarketplace) {
      return this.client.searchMarketplace(params);
    }
    // Fallback to list with search parameter
    const result = await this.list({ search: params.q, category: params.category, tags: params.tags });
    return { results: result.templates, total: result.templates.length };
  }

  // ===========================================================================
  // Publishing & Management
  // ===========================================================================

  /**
   * Publish a template to the marketplace.
   *
   * @param body - Template publication details
   * @returns The created marketplace ID
   */
  async publish(body: {
    template_id: string;
    name: string;
    description: string;
    category: string;
    tags?: string[];
    workflow_definition?: Record<string, unknown>;
    documentation?: string;
  }): Promise<{ marketplace_id: string }> {
    return this.client.publishTemplate(body);
  }

  /**
   * Import a template from the marketplace to your workspace.
   *
   * @param templateId - The template ID to import
   * @param workspaceId - Optional target workspace ID
   * @returns The imported template ID
   */
  async import(templateId: string, workspaceId?: string): Promise<{ imported_id: string }> {
    return this.client.importTemplate(templateId, workspaceId);
  }

  // ===========================================================================
  // Ratings & Reviews
  // ===========================================================================

  /**
   * Rate a template (1-5 stars).
   *
   * @param templateId - The template ID
   * @param rating - Rating value (1-5)
   * @returns The new average rating
   */
  async rate(templateId: string, rating: number): Promise<{ new_rating: number }> {
    if (rating < 1 || rating > 5) {
      throw new Error('Rating must be between 1 and 5');
    }
    return this.client.rateTemplate(templateId, rating);
  }

  /**
   * Submit a review for a template.
   *
   * @param templateId - The template ID
   * @param body - Review content
   * @returns The created review ID
   */
  async review(templateId: string, body: {
    rating: number;
    title: string;
    content: string;
  }): Promise<{ review_id: string }> {
    if (body.rating < 1 || body.rating > 5) {
      throw new Error('Rating must be between 1 and 5');
    }
    return this.client.reviewTemplate(templateId, body);
  }

  /**
   * Get reviews for a template.
   *
   * @param templateId - The template ID
   * @param params - Pagination options
   * @returns List of reviews
   */
  async getReviews(templateId: string, params?: PaginationParams): Promise<{ reviews: TemplateReview[]; total: number }> {
    if (this.client.getTemplateReviews) {
      return this.client.getTemplateReviews(templateId, params);
    }
    // Return empty if method not available
    return { reviews: [], total: 0 };
  }

  /**
   * Get rating summary for a template.
   *
   * @param templateId - The template ID
   * @returns Rating summary with average and count
   */
  async getRatings(templateId: string): Promise<TemplateRatings> {
    if (this.client.getTemplateRatings) {
      return this.client.getTemplateRatings(templateId);
    }
    // Return empty if method not available
    return { ratings: [], average: 0, count: 0 };
  }

  // ===========================================================================
  // Deployment
  // ===========================================================================

  /**
   * Deploy a template to your environment.
   *
   * @param templateId - The template ID to deploy
   * @param body - Deployment configuration
   * @returns The created deployment
   */
  async deploy(templateId: string, body?: {
    name?: string;
    config?: Record<string, unknown>;
  }): Promise<{ deployment: TemplateDeployment }> {
    if (this.client.deployTemplate) {
      return this.client.deployTemplate(templateId, body);
    }
    throw new Error('Template deployment not available');
  }

  /**
   * List all template deployments.
   *
   * @returns List of deployments
   */
  async listDeployments(): Promise<{ deployments: TemplateDeployment[] }> {
    if (this.client.listDeployments) {
      return this.client.listDeployments();
    }
    return { deployments: [] };
  }

  /**
   * Get a specific deployment by ID.
   *
   * @param deploymentId - The deployment ID
   * @returns The deployment details
   */
  async getDeployment(deploymentId: string): Promise<{ deployment: TemplateDeployment }> {
    if (this.client.getDeployment) {
      return this.client.getDeployment(deploymentId);
    }
    throw new Error('Deployment not found');
  }

  /**
   * Delete a deployment.
   *
   * @param deploymentId - The deployment ID to delete
   * @returns Success status
   */
  async deleteDeployment(deploymentId: string): Promise<{ success: boolean }> {
    if (this.client.deleteDeployment) {
      return this.client.deleteDeployment(deploymentId);
    }
    throw new Error('Deployment deletion not available');
  }
}
