/**
 * Partner API Namespace
 *
 * Provides methods for partner management:
 * - Partner registration and profile
 * - API key management
 * - Usage statistics
 * - Webhook configuration
 * - Rate limit information
 *
 * Endpoints:
 *   POST   /api/v1/partners/register     - Register as partner
 *   GET    /api/v1/partners/me           - Get partner profile
 *   POST   /api/v1/partners/keys         - Create API key
 *   GET    /api/v1/partners/keys         - List API keys
 *   DELETE /api/v1/partners/keys/{id}    - Revoke API key
 *   GET    /api/v1/partners/usage        - Get usage statistics
 *   POST   /api/v1/partners/webhooks     - Configure webhook
 *   GET    /api/v1/partners/limits       - Get rate limits
 */

/**
 * Partner profile information.
 */
export interface PartnerProfile {
  partner_id: string;
  name: string;
  email: string;
  company?: string;
  status: 'pending' | 'active' | 'suspended';
  tier: 'free' | 'basic' | 'professional' | 'enterprise';
  referral_code: string;
  created_at: string;
  updated_at?: string;
  total_api_calls: number;
  active_keys: number;
}

/**
 * API key information.
 */
export interface PartnerApiKey {
  key_id: string;
  name: string;
  prefix: string;
  scopes: string[];
  created_at: string;
  expires_at?: string;
  last_used_at?: string;
  is_active: boolean;
}

/**
 * Usage statistics.
 */
export interface PartnerUsage {
  period_start: string;
  period_end: string;
  total_requests: number;
  requests_by_endpoint: Record<string, number>;
  requests_by_day: Array<{ date: string; count: number }>;
  average_latency_ms: number;
  error_rate: number;
}

/**
 * Rate limit information.
 */
export interface PartnerLimits {
  tier: string;
  limits: {
    requests_per_minute: number;
    requests_per_day: number;
    max_concurrent: number;
    max_batch_size: number;
  };
  current_usage: {
    requests_this_minute: number;
    requests_today: number;
    concurrent_requests: number;
  };
  allowed: boolean;
  reset_at?: string;
}

/**
 * Options for partner registration.
 */
export interface RegisterPartnerOptions {
  /** Partner name */
  name: string;
  /** Contact email */
  email: string;
  /** Optional company name */
  company?: string;
}

/**
 * Options for creating an API key.
 */
export interface CreateApiKeyOptions {
  /** Key name (default "API Key") */
  name?: string;
  /** Optional list of scopes */
  scopes?: string[];
  /** Optional expiration in days */
  expiresInDays?: number;
}

/**
 * Client interface for making HTTP requests.
 */
interface PartnerClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Partner API namespace.
 *
 * Provides methods for partner registration, API key management,
 * and usage tracking.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Register as a partner
 * const result = await client.partner.register({
 *   name: 'My Company',
 *   email: 'partner@example.com',
 *   company: 'My Company Inc',
 * });
 *
 * // Get partner profile
 * const profile = await client.partner.getProfile();
 * console.log(`Tier: ${profile.tier}`);
 *
 * // Create an API key
 * const key = await client.partner.createApiKey({
 *   name: 'Production Key',
 *   scopes: ['debates:read', 'debates:write'],
 * });
 * console.log(`API Key: ${key.key}`); // Only shown once!
 *
 * // Check rate limits
 * const limits = await client.partner.getLimits();
 * console.log(`Requests today: ${limits.current_usage.requests_today}`);
 * ```
 */
export class PartnerAPI {
  constructor(private client: PartnerClientInterface) {}

  // =========================================================================
  // Registration
  // =========================================================================

  /**
   * Register as a partner.
   *
   * @param options - Registration options
   * @returns Partner ID, status, and referral code
   */
  async register(options: RegisterPartnerOptions): Promise<{
    partner_id: string;
    status: string;
    referral_code: string;
  }> {
    const data: Record<string, unknown> = {
      name: options.name,
      email: options.email,
    };
    if (options.company) data.company = options.company;

    return this.client.request('POST', '/api/v1/partners/register', {
      json: data,
    });
  }

  /**
   * Get current partner profile.
   *
   * Requires X-Partner-ID header.
   *
   * @returns Partner profile with stats
   */
  async getProfile(): Promise<PartnerProfile> {
    return this.client.request('GET', '/api/v1/partners/me');
  }

  // =========================================================================
  // API Keys
  // =========================================================================

  /**
   * Create a new API key.
   *
   * @param options - API key creation options
   * @returns Key information including the key itself (only shown once!)
   */
  async createApiKey(options?: CreateApiKeyOptions): Promise<{
    key_id: string;
    key: string;
    scopes: string[];
  }> {
    const data: Record<string, unknown> = {
      name: options?.name ?? 'API Key',
    };
    if (options?.scopes) data.scopes = options.scopes;
    if (options?.expiresInDays) data.expires_in_days = options.expiresInDays;

    return this.client.request('GET', '/api/v1/partners/keys', {
      json: data,
    });
  }

  /**
   * List all API keys.
   *
   * @returns List of API keys with counts
   */
  async listApiKeys(): Promise<{
    keys: PartnerApiKey[];
    total: number;
    active: number;
  }> {
    return this.client.request('GET', '/api/v1/partners/keys');
  }

  /**
   * Revoke an API key.
   *
   * @param keyId - The key ID to revoke
   * @returns Success message
   */
  async revokeApiKey(keyId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    return this.client.request('DELETE', `/api/v1/partners/keys/${keyId}`);
  }

  // =========================================================================
  // Usage
  // =========================================================================

  /**
   * Get usage statistics.
   *
   * @param days - Number of days to look back (1-365, default 30)
   * @returns Usage statistics
   */
  async getUsage(days = 30): Promise<PartnerUsage> {
    return this.client.request('GET', '/api/v1/partners/usage', {
      params: { days },
    });
  }

  // =========================================================================
  // Webhooks
  // =========================================================================

  /**
   * Configure webhook endpoint.
   *
   * @param url - Webhook URL (must be HTTPS)
   * @returns Webhook URL and secret (only shown once!)
   */
  async configureWebhook(url: string): Promise<{
    webhook_url: string;
    webhook_secret: string;
  }> {
    return this.client.request('GET', '/api/v1/partners/webhooks', {
      json: { url },
    });
  }

  // =========================================================================
  // Limits
  // =========================================================================

  /**
   * Get rate limits for partner tier.
   *
   * @returns Rate limit information with current usage
   */
  async getLimits(): Promise<PartnerLimits> {
    return this.client.request('GET', '/api/v1/partners/limits');
  }
}
