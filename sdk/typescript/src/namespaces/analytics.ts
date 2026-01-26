/**
 * Analytics Namespace API
 *
 * Provides a namespaced interface for analytics and metrics operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  DisagreementAnalytics,
  RoleRotationAnalytics,
  EarlyStopAnalytics,
  ConsensusQualityAnalytics,
  RankingStats,
  MemoryStats,
} from '../types';

/**
 * Period options for analytics queries.
 */
export interface AnalyticsPeriodOptions {
  /** Time period (e.g., '7d', '30d', '90d') */
  period?: string;
}

/**
 * Interface for the internal client methods used by AnalyticsAPI.
 */
interface AnalyticsClientInterface {
  getDisagreementAnalytics(params?: AnalyticsPeriodOptions): Promise<DisagreementAnalytics>;
  getRoleRotationAnalytics(params?: AnalyticsPeriodOptions): Promise<RoleRotationAnalytics>;
  getEarlyStopAnalytics(params?: AnalyticsPeriodOptions): Promise<EarlyStopAnalytics>;
  getConsensusQualityAnalytics(params?: AnalyticsPeriodOptions): Promise<ConsensusQualityAnalytics>;
  getRankingStats(): Promise<RankingStats>;
  getMemoryStats(): Promise<MemoryStats>;
}

/**
 * Analytics API namespace.
 *
 * Provides methods for retrieving analytics and metrics:
 * - Disagreement patterns between agents
 * - Role rotation statistics
 * - Early stop analysis
 * - Consensus quality metrics
 * - Ranking and memory statistics
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get disagreement analytics for the last 7 days
 * const disagreements = await client.analytics.disagreements({ period: '7d' });
 *
 * // Get consensus quality metrics
 * const quality = await client.analytics.consensusQuality();
 *
 * // Get memory statistics
 * const memStats = await client.analytics.memoryStats();
 * ```
 */
export class AnalyticsAPI {
  constructor(private client: AnalyticsClientInterface) {}

  /**
   * Get disagreement analytics showing patterns of agent disagreements.
   */
  async disagreements(params?: AnalyticsPeriodOptions): Promise<DisagreementAnalytics> {
    return this.client.getDisagreementAnalytics(params);
  }

  /**
   * Get role rotation analytics showing how agents switch roles.
   */
  async roleRotation(params?: AnalyticsPeriodOptions): Promise<RoleRotationAnalytics> {
    return this.client.getRoleRotationAnalytics(params);
  }

  /**
   * Get early stop analytics showing debates that ended early.
   */
  async earlyStops(params?: AnalyticsPeriodOptions): Promise<EarlyStopAnalytics> {
    return this.client.getEarlyStopAnalytics(params);
  }

  /**
   * Get consensus quality analytics.
   */
  async consensusQuality(params?: AnalyticsPeriodOptions): Promise<ConsensusQualityAnalytics> {
    return this.client.getConsensusQualityAnalytics(params);
  }

  /**
   * Get ranking statistics for agents.
   */
  async rankingStats(): Promise<RankingStats> {
    return this.client.getRankingStats();
  }

  /**
   * Get memory system statistics.
   */
  async memoryStats(): Promise<MemoryStats> {
    return this.client.getMemoryStats();
  }
}
