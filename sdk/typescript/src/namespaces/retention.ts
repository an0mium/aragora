/**
 * Retention Namespace API
 *
 * Provides access to data retention policies and management.
 * Essential for compliance and data lifecycle management.
 */

/**
 * Retention policy definition
 */
export interface RetentionPolicy {
  id: string;
  name: string;
  description: string;
  retention_days: number;
  data_types: string[];
  action: 'archive' | 'delete' | 'anonymize';
  enabled: boolean;
  schedule?: string;
  last_executed?: string;
  next_execution?: string;
  created_at: string;
  updated_at: string;
}

/**
 * Expiring data item
 */
export interface ExpiringItem {
  id: string;
  type: string;
  expires_at: string;
  policy_id: string;
  metadata?: Record<string, unknown>;
}

/**
 * Policy execution result
 */
export interface ExecutionResult {
  policy_id: string;
  executed_at: string;
  items_processed: number;
  items_archived: number;
  items_deleted: number;
  items_anonymized: number;
  errors: string[];
}

/**
 * Internal client interface
 */
interface RetentionClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
}

/**
 * Retention API namespace.
 *
 * Provides methods for retention policy management:
 * - List retention policies
 * - Execute policies manually
 * - View expiring data
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // List retention policies
 * const policies = await client.retention.listPolicies();
 *
 * // Execute a policy
 * const result = await client.retention.executePolicy('policy-123');
 *
 * // View expiring data
 * const expiring = await client.retention.getExpiring();
 * ```
 */
export class RetentionAPI {
  constructor(private client: RetentionClientInterface) {}

  /**
   * List retention policies.
   */
  async listPolicies(): Promise<{ policies: RetentionPolicy[] }> {
    return this.client.get('/api/v1/retention/policies');
  }

  /**
   * Execute a retention policy manually.
   */
  async executePolicy(policyId: string, options?: { dry_run?: boolean }): Promise<ExecutionResult> {
    return this.client.post(`/api/v1/retention/policies/${policyId}/execute`, options);
  }

  /**
   * Get data items that are expiring soon.
   */
  async getExpiring(options?: { days?: number; limit?: number }): Promise<{ items: ExpiringItem[]; total: number }> {
    const params = new URLSearchParams();
    if (options?.days) params.set('days', options.days.toString());
    if (options?.limit) params.set('limit', options.limit.toString());
    const query = params.toString() ? `?${params.toString()}` : '';
    return this.client.get(`/api/v1/retention/expiring${query}`);
  }
}
