/**
 * Marketplace Namespace API
 *
 * Provides a namespaced interface for marketplace operations including
 * template publishing, discovery, rating, and deployment.
 */

import type { MarketplaceTemplate, PaginationParams } from '../types';

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
 * Marketplace list parameters.
 */
export interface MarketplaceListParams {
  category?: string;
  search?: string;
  sort_by?: 'downloads' | 'rating' | 'newest';
  min_rating?: number;
  limit?: number;
  offset?: number;
}

/**
 * Interface for the internal client methods used by MarketplaceAPI.
 */
interface MarketplaceClientInterface {
  browseMarketplace(params?: MarketplaceListParams): Promise<{ templates: MarketplaceTemplate[] }>;
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
  async list(params?: MarketplaceListParams): Promise<{ templates: MarketplaceTemplate[] }> {
    return this.client.browseMarketplace(params);
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
  }): Promise<{ results: MarketplaceTemplate[]; total: number }> {
    const result = await this.list({ search: params.q, category: params.category });
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
}
