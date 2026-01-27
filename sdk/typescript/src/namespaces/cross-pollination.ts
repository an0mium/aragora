/**
 * Cross-Pollination Namespace API
 *
 * Provides a namespaced interface for cross-debate knowledge sharing
 * and Knowledge Mound integration. Enables insights and patterns from
 * one debate to inform and improve others.
 */

/**
 * Circuit breaker status
 */
export type CircuitBreakerStatus = 'closed' | 'open' | 'half_open';

/**
 * Cross-pollination statistics
 */
export interface CrossPollinationStats {
  total_events: number;
  successful_pollinations: number;
  failed_pollinations: number;
  circuit_breaker_status: CircuitBreakerStatus;
  last_event_at?: string;
  handlers_registered: number;
}

/**
 * Cross-pollination subscriber
 */
export interface CrossPollinationSubscriber {
  id: string;
  name: string;
  event_types: string[];
  active: boolean;
  created_at: string;
  last_triggered_at?: string;
  trigger_count: number;
}

/**
 * Cross-pollination bridge status
 */
export interface CrossPollinationBridge {
  source_debate_id: string;
  target_debate_id: string;
  bridge_type: string;
  status: 'active' | 'inactive' | 'failed';
  created_at: string;
  last_sync_at?: string;
  sync_count: number;
}

/**
 * Cross-pollination metrics
 */
export interface CrossPollinationMetrics {
  events_per_minute: number;
  avg_latency_ms: number;
  success_rate: number;
  queue_depth: number;
  active_bridges: number;
}

/**
 * Knowledge Mound integration status
 */
export interface KMIntegrationStatus {
  adapters: Record<string, {
    status: string;
    last_sync_at?: string;
    items_synced: number;
  }>;
  batch_queue: {
    pending: number;
    processing: number;
    failed: number;
  };
  handlers: Record<string, {
    events_processed: number;
    last_event_at?: string;
  }>;
}

/**
 * KM sync result
 */
export interface KMSyncResult {
  adapters: Record<string, {
    success: boolean;
    items_synced: number;
    duration_ms: number;
    error?: string;
  }>;
  total_duration_ms: number;
}

/**
 * KM staleness check result
 */
export interface KMStalenessResult {
  stale_nodes: number;
  checked_nodes: number;
  revalidated_nodes: number;
  errors?: string[];
}

/**
 * Culture pattern
 */
export interface CulturePattern {
  pattern_type: string;
  confidence: number;
  occurrences: number;
  last_seen_at: string;
  metadata?: Record<string, unknown>;
}

/**
 * Workspace culture
 */
export interface WorkspaceCulture {
  workspace_id: string;
  debate_style_preferences: Record<string, number>;
  organizational_patterns: CulturePattern[];
  extracted_at: string;
}

/**
 * Interface for the internal client used by CrossPollinationAPI.
 */
interface CrossPollinationClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown> }): Promise<T>;
}

/**
 * Cross-Pollination API namespace.
 *
 * Provides methods for cross-debate knowledge sharing:
 * - Monitor cross-pollination statistics and metrics
 * - Manage subscribers and bridges
 * - Knowledge Mound integration and sync
 * - Culture pattern extraction
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Get cross-pollination stats
 * const stats = await client.crossPollination.getStats();
 * console.log(`Success rate: ${stats.successful_pollinations / stats.total_events}`);
 *
 * // Check KM integration status
 * const kmStatus = await client.crossPollination.getKMStatus();
 *
 * // Trigger manual KM sync
 * const syncResult = await client.crossPollination.syncKM();
 *
 * // Get culture patterns for a workspace
 * const culture = await client.crossPollination.getCulture('workspace-123');
 * ```
 */
export class CrossPollinationAPI {
  constructor(private client: CrossPollinationClientInterface) {}

  // ===========================================================================
  // Statistics & Monitoring
  // ===========================================================================

  /**
   * Get cross-pollination statistics.
   */
  async getStats(): Promise<CrossPollinationStats> {
    return this.client.get('/api/v1/cross-pollination/stats');
  }

  /**
   * Get cross-pollination metrics.
   */
  async getMetrics(): Promise<CrossPollinationMetrics> {
    return this.client.get('/api/v1/cross-pollination/metrics');
  }

  /**
   * Reset all cross-pollination statistics.
   * Useful for testing and debugging.
   */
  async resetStats(): Promise<{ reset: boolean; handlers_count: number }> {
    return this.client.post('/api/v1/cross-pollination/reset');
  }

  // ===========================================================================
  // Subscribers & Bridges
  // ===========================================================================

  /**
   * List cross-pollination subscribers.
   */
  async getSubscribers(): Promise<{ subscribers: CrossPollinationSubscriber[]; total: number }> {
    return this.client.get('/api/v1/cross-pollination/subscribers');
  }

  /**
   * Get active cross-pollination bridges.
   */
  async getBridges(): Promise<{ bridges: CrossPollinationBridge[]; total: number }> {
    return this.client.get('/api/v1/cross-pollination/bridge');
  }

  // ===========================================================================
  // Knowledge Mound Integration
  // ===========================================================================

  /**
   * Get Knowledge Mound integration status.
   */
  async getKMStatus(): Promise<KMIntegrationStatus> {
    return this.client.get('/api/v1/cross-pollination/km');
  }

  /**
   * Trigger manual sync of KM adapters.
   * Useful for forcing persistence after important debates or before shutdown.
   */
  async syncKM(): Promise<KMSyncResult> {
    return this.client.post('/api/v1/cross-pollination/km/sync');
  }

  /**
   * Trigger manual staleness check for Knowledge Mound nodes.
   */
  async checkKMStaleness(): Promise<KMStalenessResult> {
    return this.client.post('/api/v1/cross-pollination/km/staleness-check');
  }

  /**
   * Get culture patterns for a workspace.
   * @param workspaceId - Workspace ID (default: 'default')
   */
  async getCulture(workspaceId: string = 'default'): Promise<WorkspaceCulture> {
    return this.client.request('GET', '/api/v1/cross-pollination/km/culture', {
      params: { workspace_id: workspaceId },
    });
  }
}
