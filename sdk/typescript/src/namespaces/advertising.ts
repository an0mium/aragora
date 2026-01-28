/**
 * Advertising Namespace API
 *
 * Provides methods for managing advertising platform integrations:
 * - Connect and manage advertising platforms (Google Ads, Meta, etc.)
 * - View and manage campaigns
 * - Analyze performance and get budget recommendations
 */

/**
 * Advertising platform configuration
 */
export interface AdvertisingPlatform {
  name: string;
  display_name: string;
  description: string;
  connected: boolean;
  account_id?: string;
  connected_at?: string;
  capabilities: string[];
}

/**
 * Advertising campaign
 */
export interface Campaign {
  id: string;
  platform: string;
  name: string;
  status: 'active' | 'paused' | 'ended' | 'draft';
  objective: string;
  budget: number;
  budget_type: 'daily' | 'total';
  spent: number;
  impressions: number;
  clicks: number;
  conversions: number;
  start_date?: string;
  end_date?: string;
  targeting?: CampaignTargeting;
  created_at: string;
  updated_at: string;
}

/**
 * Campaign targeting configuration
 */
export interface CampaignTargeting {
  locations?: string[];
  age_range?: { min: number; max: number };
  genders?: string[];
  interests?: string[];
  keywords?: string[];
  audiences?: string[];
  placements?: string[];
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  impressions: number;
  clicks: number;
  conversions: number;
  spend: number;
  ctr: number;
  cpc: number;
  cpa: number;
  roas: number;
  period: { start: string; end: string };
  breakdown?: Record<string, PerformanceMetrics>;
}

/**
 * Budget recommendation
 */
export interface BudgetRecommendation {
  platform: string;
  campaign_id?: string;
  current_budget: number;
  recommended_budget: number;
  expected_improvement: number;
  confidence: number;
  rationale: string;
}

/**
 * Analysis result
 */
export interface AnalysisResult {
  analysis_type: string;
  insights: string[];
  recommendations: string[];
  metrics: Record<string, unknown>;
  created_at: string;
}

/**
 * Connect platform request
 */
export interface ConnectPlatformRequest {
  platform: string;
  credentials: Record<string, unknown>;
  account_id?: string;
}

/**
 * Create campaign request
 */
export interface CreateCampaignRequest {
  name: string;
  budget: number;
  objective: string;
  targeting?: CampaignTargeting;
  start_date?: string;
  end_date?: string;
  [key: string]: unknown;
}

/**
 * Update campaign request
 */
export interface UpdateCampaignRequest {
  name?: string;
  budget?: number;
  status?: 'active' | 'paused';
  targeting?: CampaignTargeting;
  [key: string]: unknown;
}

/**
 * Analyze request
 */
export interface AnalyzeRequest {
  campaign_ids?: string[];
  analysis_type?: string;
}

/**
 * Interface for the internal client used by AdvertisingAPI.
 */
interface AdvertisingClientInterface {
  get<T>(path: string, params?: Record<string, unknown>): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  put<T>(path: string, body?: unknown): Promise<T>;
  delete<T>(path: string): Promise<T>;
}

/**
 * Advertising API namespace.
 *
 * Provides methods for advertising platform management:
 * - Platform connections
 * - Campaign management
 * - Performance analytics
 * - Budget recommendations
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // List available platforms
 * const { platforms } = await client.advertising.listPlatforms();
 *
 * // Connect a platform
 * await client.advertising.connect({
 *   platform: 'google_ads',
 *   credentials: { client_id: '...', client_secret: '...' }
 * });
 *
 * // Get campaign performance
 * const performance = await client.advertising.getPerformance('google_ads');
 *
 * // Get budget recommendations
 * const recommendations = await client.advertising.getBudgetRecommendations();
 * ```
 */
export class AdvertisingAPI {
  constructor(private client: AdvertisingClientInterface) {}

  // ===========================================================================
  // Platform Management
  // ===========================================================================

  /**
   * List available advertising platforms.
   */
  async listPlatforms(): Promise<{ platforms: AdvertisingPlatform[] }> {
    return this.client.get('/api/v1/advertising/platforms');
  }

  /**
   * Connect an advertising platform.
   */
  async connect(body: ConnectPlatformRequest): Promise<AdvertisingPlatform> {
    return this.client.post('/api/v1/advertising/connect', body);
  }

  /**
   * Disconnect an advertising platform.
   */
  async disconnect(platform: string): Promise<{ disconnected: boolean }> {
    return this.client.delete(`/api/v1/advertising/${platform}`);
  }

  // ===========================================================================
  // Campaign Management
  // ===========================================================================

  /**
   * List all campaigns across platforms.
   */
  async listCampaigns(params?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ campaigns: Campaign[]; total: number }> {
    return this.client.get('/api/v1/advertising/campaigns', params);
  }

  /**
   * List campaigns for a specific platform.
   */
  async listPlatformCampaigns(
    platform: string,
    params?: { status?: string; limit?: number; offset?: number }
  ): Promise<{ campaigns: Campaign[]; total: number }> {
    return this.client.get(`/api/v1/advertising/${platform}/campaigns`, params);
  }

  /**
   * Get a specific campaign.
   */
  async getCampaign(platform: string, campaignId: string): Promise<Campaign> {
    return this.client.get(`/api/v1/advertising/${platform}/campaigns/${campaignId}`);
  }

  /**
   * Create a new campaign.
   */
  async createCampaign(platform: string, body: CreateCampaignRequest): Promise<Campaign> {
    return this.client.post(`/api/v1/advertising/${platform}/campaigns`, body);
  }

  /**
   * Update an existing campaign.
   */
  async updateCampaign(
    platform: string,
    campaignId: string,
    body: UpdateCampaignRequest
  ): Promise<Campaign> {
    return this.client.put(`/api/v1/advertising/${platform}/campaigns/${campaignId}`, body);
  }

  // ===========================================================================
  // Analytics & Recommendations
  // ===========================================================================

  /**
   * Get aggregated performance metrics across all platforms.
   */
  async getPerformance(params?: {
    start_date?: string;
    end_date?: string;
    metrics?: string;
  }): Promise<PerformanceMetrics> {
    return this.client.get('/api/v1/advertising/performance', params);
  }

  /**
   * Get performance metrics for a specific platform.
   */
  async getPlatformPerformance(
    platform: string,
    params?: { start_date?: string; end_date?: string; metrics?: string }
  ): Promise<PerformanceMetrics> {
    return this.client.get(`/api/v1/advertising/${platform}/performance`, params);
  }

  /**
   * Analyze advertising campaigns.
   */
  async analyze(body?: AnalyzeRequest): Promise<AnalysisResult> {
    return this.client.post('/api/v1/advertising/analyze', body);
  }

  /**
   * Get AI-powered budget recommendations.
   */
  async getBudgetRecommendations(params?: {
    goal?: string;
    timeframe?: string;
  }): Promise<{ recommendations: BudgetRecommendation[] }> {
    return this.client.get('/api/v1/advertising/budget-recommendations', params);
  }
}
